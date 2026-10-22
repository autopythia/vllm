# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-scaled FP8 copy of a BF16 LM head for speculative drafting.

Experiment, VLLM_EXP_MTP_FP8_HEAD=1. A drafter that shares the target's BF16 LM
head reads all of it at every draft step. Qwen3.8-27B's head (248,320 x 5,120)
is 2.54 GB: 1.8 ms per draft step at one stream on an RTX PRO 6000, two of them
per decode step with two drafts. Drafts need the ranking of the logits to be
right most of the time, not exact logits (the target verifies them with its own
BF16 head), so the drafter reads an FP8 (E4M3) copy with one scale per
vocabulary row: half the bytes. The copy costs half the head's size in GPU
memory; the target keeps the BF16 head.

The GEMM handles 1 to MAX_ROWS rows in one pass over the weight: BF16
activations, the FP8 weight converted in registers (exactly: every E4M3 value is
a BF16 value), FP32 accumulation, the row scale applied at the end.
"""

import torch
import triton
import triton.language as tl

MAX_ROWS = 16
_FP8_MAX = 448.0


@triton.jit
def _fp8_head_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_om,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        acc = tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)), acc)
    s = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * s[None, :]
    ty = out_ptr.dtype.element_ty
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        acc.to(ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def quantize_rows(
    weight: torch.Tensor, chunk_rows: int = 16384
) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 values and FP32 scales of a [rows, k] weight, one scale per row (the
    row's largest magnitude maps to 448)."""
    n, k = weight.shape
    w8 = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty(n, dtype=torch.float32, device=weight.device)
    with torch.no_grad():
        for r0 in range(0, n, chunk_rows):
            w = weight[r0 : r0 + chunk_rows].float()
            s = (w.abs().amax(dim=1) / _FP8_MAX).clamp_min(1e-12)
            scale[r0 : r0 + chunk_rows] = s
            w8[r0 : r0 + chunk_rows] = (
                (w / s[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
            )
            del w
    return w8, scale


class Fp8DraftHead:
    """Row-scaled FP8 copy of an LM head weight [vocab, hidden] and its GEMM for 1
    to MAX_ROWS rows."""

    def __init__(
        self,
        weight: torch.Tensor,
        block_n: int = 32,
        block_k: int | None = None,
        num_warps: int = 4,
        num_stages: int = 4,
    ):
        assert weight.dim() == 2
        self.n, self.k = weight.shape
        if block_k is None:  # 256 was fastest for Qwen3.8-27B's head on sm_120
            block_k = next((b for b in (256, 128, 64) if self.k % b == 0), 0)
        assert block_k and self.k % block_k == 0, (self.k, block_k)
        self.w8, self.scale = quantize_rows(weight)
        self.block_n, self.block_k = block_n, block_k
        self.num_warps, self.num_stages = num_warps, num_stages

    @property
    def nbytes(self) -> int:
        return self.w8.numel() + self.scale.numel() * 4

    def __call__(
        self, x: torch.Tensor, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        m = x.shape[0]
        assert 1 <= m <= MAX_ROWS and x.shape[1] == self.k, (tuple(x.shape), self.k)
        if x.stride(1) != 1:
            x = x.contiguous()
        out = torch.empty(m, self.n, dtype=out_dtype or x.dtype, device=x.device)
        grid = (triton.cdiv(self.n, self.block_n),)
        _fp8_head_kernel[grid](
            x,
            self.w8,
            self.scale,
            out,
            m,
            self.n,
            x.stride(0),
            out.stride(0),
            K=self.k,
            BLOCK_M=16,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        return out
