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


def load_draft_vocab_rows(model_dir: str, ids: torch.Tensor) -> torch.Tensor:
    """FP32 rows ``ids`` of the checkpoint's LM head (``lm_head.weight``), read on the CPU: BF16/FP16 as stored, or
    MXFP8 (E4M3 values with an E8M0 scale per 32 along the hidden size, ``lm_head.weight_scale``) dequantized."""
    import json
    import os

    from safetensors import safe_open

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    with safe_open(os.path.join(model_dir, wmap["lm_head.weight"]), framework="pt") as f:
        w = f.get_tensor("lm_head.weight")[ids]
    if w.dtype != torch.float8_e4m3fn:
        return w.float()
    with safe_open(os.path.join(model_dir, wmap["lm_head.weight_scale"]), framework="pt") as f:
        s = f.get_tensor("lm_head.weight_scale")[ids]
    n, k = w.shape
    scale = torch.exp2(s.float() - 127.0)  # E8M0
    return (w.float().view(n, s.shape[1], k // s.shape[1]) * scale[:, :, None]).view(n, k)


def load_draft_vocab_mxfp8(model_dir: str, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Rows ``ids`` of an MXFP8 LM head as stored (E4M3 values, E8M0 scales per 32), or None for another format."""
    import json
    import os

    from safetensors import safe_open

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    if "lm_head.weight_scale" not in wmap:
        return None
    with safe_open(os.path.join(model_dir, wmap["lm_head.weight"]), framework="pt") as f:
        w = f.get_tensor("lm_head.weight")
        if w.dtype != torch.float8_e4m3fn:
            return None
        w = w[ids].contiguous()
    with safe_open(os.path.join(model_dir, wmap["lm_head.weight_scale"]), framework="pt") as f:
        s = f.get_tensor("lm_head.weight_scale")[ids].contiguous()
    return w, s


@triton.jit
def _mxfp8_gemv_kernel(x_ptr, w_ptr, s_ptr, out_ptr, N, stride_xm, stride_om,
                       K: tl.constexpr, M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    G: tl.constexpr = BLOCK_K // 32
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_g = k0 // 32 + tl.arange(0, G)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
        sb = tl.load(s_ptr + offs_n[:, None] * (K // 32) + offs_g[None, :], mask=mask_n[:, None], other=127)
        sc = tl.exp2(sb.to(tl.float32) - 127.0)  # [BLOCK_N, G], E8M0
        x0 = tl.load(x_ptr + offs_k).to(tl.float32)
        acc0 += tl.sum(tl.sum(tl.reshape(w * x0[None, :], (BLOCK_N, G, 32)), axis=2) * sc, axis=1)
        if M > 1:
            x1 = tl.load(x_ptr + stride_xm + offs_k).to(tl.float32)
            acc1 += tl.sum(tl.sum(tl.reshape(w * x1[None, :], (BLOCK_N, G, 32)), axis=2) * sc, axis=1)
        if M > 2:
            x2 = tl.load(x_ptr + 2 * stride_xm + offs_k).to(tl.float32)
            acc2 += tl.sum(tl.sum(tl.reshape(w * x2[None, :], (BLOCK_N, G, 32)), axis=2) * sc, axis=1)
        if M > 3:
            x3 = tl.load(x_ptr + 3 * stride_xm + offs_k).to(tl.float32)
            acc3 += tl.sum(tl.sum(tl.reshape(w * x3[None, :], (BLOCK_N, G, 32)), axis=2) * sc, axis=1)
    ty = out_ptr.dtype.element_ty
    tl.store(out_ptr + offs_n, acc0.to(ty), mask=mask_n)
    if M > 1:
        tl.store(out_ptr + stride_om + offs_n, acc1.to(ty), mask=mask_n)
    if M > 2:
        tl.store(out_ptr + 2 * stride_om + offs_n, acc2.to(ty), mask=mask_n)
    if M > 3:
        tl.store(out_ptr + 3 * stride_om + offs_n, acc3.to(ty), mask=mask_n)


class Mxfp8Head:
    """MXFP8 rows of an LM head as stored ([n, k] E4M3 and [n, k/32] E8M0) and their GEMV for up to MAX_ROWS rows."""

    def __init__(self, w8: torch.Tensor, scales: torch.Tensor, block_n: int = 64, block_k: int = 256,
                 num_warps: int = 4, num_stages: int = 3):
        n, k = w8.shape
        assert w8.dtype == torch.float8_e4m3fn and scales.dtype == torch.uint8 and scales.shape == (n, k // 32)
        assert k % block_k == 0 and block_k % 32 == 0
        self.n, self.k, self.w8, self.scale = n, k, w8, scales
        self.block_n, self.block_k, self.num_warps, self.num_stages = block_n, block_k, num_warps, num_stages

    def __call__(self, x: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        m = x.shape[0]
        assert 1 <= m <= MAX_ROWS and x.shape[1] == self.k
        x = x.contiguous()
        out = torch.empty(m, self.n, dtype=out_dtype or x.dtype, device=x.device)
        grid = (triton.cdiv(self.n, self.block_n),)
        _mxfp8_gemv_kernel[grid](x, self.w8, self.scale, out, self.n, x.stride(0), out.stride(0),
                                 K=self.k, M=m, BLOCK_N=self.block_n, BLOCK_K=self.block_k,
                                 num_warps=self.num_warps, num_stages=self.num_stages)
        return out
