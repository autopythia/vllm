# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 copy of the LM head for MTP drafting (experiment: VLLM_EXP_MTP_FP8_HEAD=1).

GLM-5.3's LM head stays BF16 in the NVFP4 checkpoint (154,880 x 4,096, 1.27 GB). The MTP drafter shares it, and
with one draft row per request its logits are a GEMV that reads all of it: 0.85 ms per draft on an RTX PRO 6000,
about a third of the drafter's GPU time. Drafts only need the ranking of the logits to be right most of the time
(verification uses the target's own BF16 head), so the drafter reads an FP8 (E4M3) copy with one scale per vocabulary
row instead: half the bytes. The copy costs 0.63 GB of GPU memory; the target keeps the BF16 head.
"""

import torch
import triton
import triton.language as tl

MAX_ROWS = 4
_FP8_MAX = 448.0


@triton.jit
def _fp8_gemv_kernel(x_ptr, w_ptr, s_ptr, out_ptr, N, stride_xm, stride_om,
                     K: tl.constexpr, M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
        x0 = tl.load(x_ptr + offs_k).to(tl.float32)
        acc0 += tl.sum(w * x0[None, :], axis=1)
        if M > 1:
            x1 = tl.load(x_ptr + stride_xm + offs_k).to(tl.float32)
            acc1 += tl.sum(w * x1[None, :], axis=1)
        if M > 2:
            x2 = tl.load(x_ptr + 2 * stride_xm + offs_k).to(tl.float32)
            acc2 += tl.sum(w * x2[None, :], axis=1)
        if M > 3:
            x3 = tl.load(x_ptr + 3 * stride_xm + offs_k).to(tl.float32)
            acc3 += tl.sum(w * x3[None, :], axis=1)
    s = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0)
    ty = out_ptr.dtype.element_ty
    tl.store(out_ptr + offs_n, (acc0 * s).to(ty), mask=mask_n)
    if M > 1:
        tl.store(out_ptr + stride_om + offs_n, (acc1 * s).to(ty), mask=mask_n)
    if M > 2:
        tl.store(out_ptr + 2 * stride_om + offs_n, (acc2 * s).to(ty), mask=mask_n)
    if M > 3:
        tl.store(out_ptr + 3 * stride_om + offs_n, (acc3 * s).to(ty), mask=mask_n)


class Fp8Head:
    """Row-scaled FP8 (E4M3) copy of an LM head weight [vocab, hidden] and its GEMV for up to MAX_ROWS rows."""

    def __init__(self, weight: torch.Tensor, chunk_rows: int = 16384, block_n: int = 64, block_k: int = 256,
                 num_warps: int = 4, num_stages: int = 3):
        n, k = weight.shape
        self.n, self.k = n, k
        self.w8 = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=weight.device)
        self.scale = torch.empty(n, dtype=torch.float32, device=weight.device)
        with torch.no_grad():
            for r0 in range(0, n, chunk_rows):
                w = weight[r0:r0 + chunk_rows].float()
                s = (w.abs().amax(dim=1) / _FP8_MAX).clamp_min(1e-12)
                self.scale[r0:r0 + chunk_rows] = s
                self.w8[r0:r0 + chunk_rows] = (w / s[:, None]).to(torch.float8_e4m3fn)
                del w
        self.block_n, self.block_k, self.num_warps, self.num_stages = block_n, block_k, num_warps, num_stages
        assert k % block_k == 0

    def __call__(self, x: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        m = x.shape[0]
        assert 1 <= m <= MAX_ROWS and x.shape[1] == self.k
        x = x.contiguous()
        out = torch.empty(m, self.n, dtype=out_dtype or x.dtype, device=x.device)
        grid = (triton.cdiv(self.n, self.block_n),)
        _fp8_gemv_kernel[grid](x, self.w8, self.scale, out, self.n, x.stride(0), out.stride(0),
                               K=self.k, M=m, BLOCK_N=self.block_n, BLOCK_K=self.block_k,
                               num_warps=self.num_warps, num_stages=self.num_stages)
        return out
