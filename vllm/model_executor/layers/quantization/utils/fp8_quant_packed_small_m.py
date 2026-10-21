# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment (KH15): the FP8 per-token-group quantization with UE8M0 scales
packed for DeepGEMM (``_C.per_token_group_fp8_quant_packed``) in Triton for the
few rows of a decode step (programs of up to 32 groups), with the CUDA
kernel's arithmetic and output bytes: the group's absmax (from ``eps``), an
IEEE division by the FP8 maximum, the power-of-two scale from the float's
exponent (rounded up), values times its exact reciprocal clamped to the FP8
range and rounded to nearest, and the scale bytes at the same packed positions,
zeros for padded rows and groups. ``VLLM_EXP_FP8_QUANT_TRITON=1`` opts in."""

import os

import torch

from vllm.triton_utils import tl, tldevice, triton

MAX_ROWS = 16


def enabled() -> bool:
    return os.environ.get("VLLM_EXP_FP8_QUANT_TRITON", "0") == "1"


@triton.jit
def _quant_fp8_packed_kernel(
    x_ptr,
    q_ptr,
    s_bytes_ptr,
    mn,
    tma_aligned_mn,
    num_scale_elems,
    eps,
    fp8_min,
    fp8_max,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    NGROUPS: tl.constexpr,
    PADDED_GROUPS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
    e_off = tl.arange(0, GROUP)
    is_row = row < mn
    gvalid = g < NGROUPS
    valid = is_row & gvalid
    offs = row * K + g[:, None] * GROUP + e_off[None, :]
    x = tl.load(x_ptr + offs, mask=valid[:, None], other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), eps)
    y = tldevice.div_rn(amax, fp8_max)
    y = tl.maximum(y, 1e-10)
    bits = y.to(tl.int32, bitcast=True)
    exp_bits = (bits >> 23) & 0xFF
    exp_byte = exp_bits + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    y_q = (exp_byte << 23).to(tl.float32, bitcast=True)
    inv = tldevice.div_rn(1.0, y_q)
    q = tl.minimum(tl.maximum(x * inv[:, None], fp8_min), fp8_max)
    tl.store(q_ptr + offs, q.to(tl.float8e4nv), mask=valid[:, None])
    out_idx = (g // 4) * tma_aligned_mn + row
    byte = tl.where(valid, exp_byte, 0).to(tl.uint8)
    in_grid = g < PADDED_GROUPS
    write = in_grid & (valid | (out_idx < num_scale_elems))
    tl.store(s_bytes_ptr + out_idx * 4 + (g % 4), byte, mask=write)


def usable(x: torch.Tensor, group_size: int, out_q: torch.Tensor | None) -> bool:
    k = x.shape[-1]
    return (
        enabled()
        and out_q is None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and group_size == 128
        and k % group_size == 0
        and x.numel() // k <= MAX_ROWS
    )


def quant_into(
    x_contiguous: torch.Tensor,
    x_q: torch.Tensor,
    x_s_packed: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
) -> None:
    """Fill ``x_q`` and ``x_s_packed`` as
    ``torch.ops._C.per_token_group_fp8_quant_packed`` does."""
    k = x_contiguous.shape[-1]
    mn = x_contiguous.numel() // k
    groups = k // group_size
    packed = (groups + 3) // 4
    tma_aligned_mn = ((mn + 3) // 4) * 4
    num_scale_elems = mn + (packed - 1) * tma_aligned_mn
    # The packed scales are strided ([mn, packed] with strides [1, tma_aligned_mn]):
    # write bytes into a flat view of their storage.
    s_bytes = x_s_packed.as_strided((num_scale_elems,), (1,)).view(torch.uint8)
    chunk = min(32, triton.next_power_of_2(packed * 4))  # groups per program
    grid = (tma_aligned_mn, triton.cdiv(packed * 4, chunk))
    _quant_fp8_packed_kernel[grid](
        x_contiguous,
        x_q,
        s_bytes,
        mn,
        tma_aligned_mn,
        num_scale_elems,
        float(eps),
        float(fp8_min),
        float(fp8_max),
        K=k,
        GROUP=group_size,
        NGROUPS=groups,
        PADDED_GROUPS=packed * 4,
        CHUNK=chunk,
        num_warps=4,
    )
