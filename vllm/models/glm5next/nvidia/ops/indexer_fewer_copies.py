# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fewer small copies in the GLM-5.3 indexer at decode (experiment).

Opt-in with ``VLLM_EXP_INDEXER_FEWER_COPIES=1``. At decode, the indexer ran
seven small elementwise kernels per call around its own (1-5 us each, ~18 us
per MLA layer on sm_120 at one stream): the BF16 cast of k (a strided copy),
the -1 fill of the top-k rows, two int32 casts of the positions, the int64
cast of the selected pool ids, the ``+ 1`` of the tail lengths and the copy of
the result into ``topk_indices_buffer``. Here:

- ``cast_bf16_rows`` casts a row-strided FP32 slice to BF16 in one small
  Triton kernel (round to nearest even, as torch's cast);
- ``expand_pools_into`` is ``expand_pools_and_append_tail`` reading the int32
  pool ids and the int64 positions as they are (tail length = position + 1),
  writing the rows, their -1 padding and the -1 rows after them straight into
  the buffer.

The kpool update takes the int64 positions directly (its kernel uses them only
in integer arithmetic). Every output is bit-identical to the default path.
"""

from __future__ import annotations

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@functools.cache
def enabled() -> bool:
    on = os.environ.get("VLLM_EXP_INDEXER_FEWER_COPIES", "0") == "1"
    if on:
        logger.info_once(
            "GLM-5.3 indexer: fewer small copies at decode "
            "(VLLM_EXP_INDEXER_FEWER_COPIES)"
        )
    return on


@triton.jit
def _cast_bf16_rows_kernel(x_ptr, out_ptr, x_s0, COLS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, COLS)
    x = tl.load(x_ptr + row * x_s0 + cols)
    tl.store(out_ptr + row * COLS + cols, x.to(tl.bfloat16))


def cast_bf16_rows(x: torch.Tensor) -> torch.Tensor:
    """``x.to(torch.bfloat16)`` for a [rows, cols] FP32 tensor whose rows may
    be strided (a column slice); the result is contiguous, as torch's."""
    rows, cols = x.shape
    assert x.dtype == torch.float32 and x.stride(1) == 1
    assert cols == triton.next_power_of_2(cols)
    out = torch.empty((rows, cols), dtype=torch.bfloat16, device=x.device)
    if rows:
        _cast_bf16_rows_kernel[(rows,)](x, out, x.stride(0), COLS=cols, num_warps=1)
    return out


@triton.jit
def _expand_pools_into_kernel(
    pool_ids_ptr,  # [rows, n_groups], any int dtype (-1 = none)
    positions_ptr,  # [rows], any int dtype: seq_len = position + 1
    out_ptr,  # [fill_rows, width] int32
    rows,
    topk,  # n_groups * POOL_SIZE
    out_cols,  # topk + POOL_SIZE - 1
    width,
    pid_s0,
    out_s0,
    POOL_SIZE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    # _expand_pools_and_append_tail_kernel's arithmetic, plus -1 in the
    # columns past out_cols and in the rows past `rows` (the caller's fill).
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = cols < width
    live = row < rows
    src = tl.minimum(row, rows - 1)

    # as _decode_topk_seq_lens: positions.to(torch.int32) + 1
    seq_len = tl.load(positions_ptr + src).to(tl.int32) + 1
    pool_len = seq_len // POOL_SIZE
    tail_start = pool_len * POOL_SIZE
    tail_count = seq_len - tail_start

    is_history = cols < topk
    g = cols // POOL_SIZE
    o = cols % POOL_SIZE
    pid = tl.load(
        pool_ids_ptr + src * pid_s0 + g, mask=mask & is_history & live, other=-1
    )
    hist_val = (pid * POOL_SIZE + o).to(tl.int32)
    hist_out = tl.where(pid >= 0, hist_val, -1)

    tail_off = cols - topk
    is_tail = (tail_off >= 0) & (tail_off < tail_count)
    tail_val = (tail_start + tail_off).to(tl.int32)
    tail_out = tl.where(is_tail, tail_val, -1)

    result = tl.where(is_history, hist_out, tail_out)
    result = tl.where(live & (cols < out_cols), result, -1)
    tl.store(out_ptr + row * out_s0 + cols, result, mask=mask)


def expand_pools_into(
    pool_ids: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    fill_rows: int,
    pool_size: int,
) -> None:
    """Write what ``out[:fill_rows] = -1``, then ``expand_pools_and_append_tail(
    pool_ids.to(torch.int64), positions[:rows].to(torch.int32) + 1, pool_size)``
    copied into ``out[:rows, :topk + pool_size - 1]`` would leave in
    ``out[:fill_rows]`` (rows = ``pool_ids.shape[0]``)."""
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    width = out.shape[1]
    assert out.dtype == torch.int32 and out.stride(1) == 1
    assert width >= out_cols and rows <= fill_rows <= out.shape[0]
    assert pool_ids.stride(1) == 1 and positions.ndim == 1
    assert positions.shape[0] >= rows and positions.stride(0) == 1
    if fill_rows == 0:
        return
    if rows == 0:
        out[:fill_rows] = -1
        return
    block_cols = 128
    _expand_pools_into_kernel[(fill_rows, triton.cdiv(width, block_cols))](
        pool_ids,
        positions,
        out,
        rows,
        topk,
        out_cols,
        width,
        pool_ids.stride(0),
        out.stride(0),
        POOL_SIZE=pool_size,
        BLOCK_COLS=block_cols,
    )
