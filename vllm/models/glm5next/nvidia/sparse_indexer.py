# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.models.glm5next.nvidia.deepselect_topk import deepselect_topk_
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.indexer_topk import (
    SparseIndexerTopk,
    get_indexer_topk,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    PrefillRowShard,
    _merge_dcp_topk_global,
    _reserve_row_shard_exchange_transient,
)
from vllm.models.glm5next.common.sparse_indexer import (
    RADIX_TOPK_WORKSPACE_SIZE,
    _build_decode_scatter_indices,
    _decode_topk_seq_lens,
    _fill_causal_indices,
    _fill_short_decode_causal_indices,
    _gather_workspace_shapes,
    _scatter_decode_tokens_by_request,
    kv_cache_as_quant_view,
)
from vllm.models.glm5next.nvidia.ops import indexer_fewer_copies
from vllm.models.glm5next.nvidia.ops import kpool_compress as kpool_ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import has_deep_gemm
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops

logger = init_logger(__name__)


# kpool write helper: form pools from the current token batch and compress them
# into the index K cache via the fused Triton kernel.


def _kpool_compress_insert(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Pool ``kpool`` consecutive tokens into one fp8 K and write at pool slots.

    ``slot_mapping`` is pool-granular (compress_ratio == kpool on the spec):
    only the *last* token of each complete pool carries a valid (>=0) slot;
    intra-pool tokens are -1. Every position is treated as a pool-completion
    candidate and non-completions are masked off inside the kernel. Compacting
    the valid rows first costs two device syncs on the eager prefill path and
    buys nothing numerically. Assumes pool-aligned chunk starts.
    """
    n = slot_mapping.shape[0]
    # No pool can complete in a batch smaller than one pool; also keeps the
    # clamped gather indices below in bounds.
    if n < kpool:
        return
    pos = torch.arange(n, device=k.device)
    valid = slot_mapping >= 0
    # Drop pools whose start falls before the batch (leading padding); their
    # gate/k data is undefined anyway.
    write_mask = valid & (pos >= kpool - 1)
    offs = torch.arange(kpool, device=k.device)
    idx = (pos - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
    kpool_ops.kpool_compress_and_write_cache(
        kv_cache,
        k[idx],  # [n, kpool, head_dim]
        gate_score[idx],
        ape,
        slot_mapping.to(torch.int64),
        pool_size=kpool,
        head_dim=head_dim,
        write_mask=write_mask,
        round_scale=round_scale,
        write_cache=True,
        return_compressed=False,
    )


def _maybe_sort_topk_(topk: torch.Tensor) -> None:
    """Give the top-k selection a fixed order (padding last).

    The radix top-k kernels emit the selected set in an order that depends on
    atomics, and sparse attention accumulates in index order, so without this
    identical requests can differ run-to-run once top-k actually selects.
    """
    if envs.VLLM_SPARSE_INDEXER_DETERMINISTIC_TOPK:
        topk.copy_(topk.sort(dim=-1, descending=True).values)


def _deterministic_topk_prefill_(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Drop-in for ``top_k_per_row_prefill`` (indices relative to each row's
    start, -1 padding) whose choice among exact ties at the top-k boundary is
    deterministic (torch.topk) instead of decided by atomics. Masks ``logits``
    outside each row's range in place.
    """
    if logits.shape[0] == 0:
        return
    starts = row_starts.unsqueeze(1).long()
    if logits.stride(1) == 1:
        # In place and without [M, N] temporaries (a bool mask would add 0.75x
        # the logits size at the chunker's logits cap).
        _fill_outside_rows_kernel[(logits.shape[0],)](
            logits,
            logits.stride(0),
            row_starts,
            row_ends,
            logits.shape[1],
            BLOCK=1024,
        )
    else:
        cols = torch.arange(logits.shape[1], device=logits.device)
        logits.masked_fill_(
            (cols < starts) | (cols >= row_ends.unsqueeze(1)), float("-inf")
        )
    k = min(out.shape[1], logits.shape[1])
    vals, idx = logits.topk(k, dim=1, sorted=False)
    out.fill_(-1)
    out[:, :k] = torch.where(vals == float("-inf"), -1, idx - starts).to(out.dtype)


@triton.jit(do_not_specialize=["stride", "starts_ptr", "ends_ptr", "n_cols"])
def _fill_outside_rows_kernel(
    logits_ptr, stride, starts_ptr, ends_ptr, n_cols, BLOCK: tl.constexpr
):
    """logits[row, c] = -inf for c < starts[row] or c >= ends[row]; one program
    per row, touching only the columns outside the row's range."""
    row = tl.program_id(0)
    start = tl.load(starts_ptr + row).to(tl.int64)
    end = tl.maximum(tl.load(ends_ptr + row).to(tl.int64), start)
    n = n_cols.to(tl.int64)
    base = logits_ptr + row.to(tl.int64) * stride
    offs = tl.arange(0, BLOCK)
    for c0 in range(0, start, BLOCK):
        cols = c0 + offs
        tl.store(base + cols, float("-inf"), mask=cols < start)
    for c0 in range(end, n, BLOCK):
        cols = c0 + offs
        tl.store(base + cols, float("-inf"), mask=cols < n)


def _fast_deterministic_topk() -> bool:
    """Experiment (A7): with ``VLLM_SPARSE_INDEXER_DETERMINISTIC_TOPK=1``,
    ``VLLM_EXP_INDEXER_FAST_DET_TOPK=1`` keeps the fast radix top-k kernels
    and makes their result deterministic afterwards
    (``_deterministic_topk_compact_``) instead of selecting with torch.topk:
    the deterministic top-k cost K3 4-8% of its decode speed (e118, KH6)."""
    return os.environ.get("VLLM_EXP_INDEXER_FAST_DET_TOPK", "0") == "1"


_DET_TOPK_CHUNK = 4096  # columns per program in the multi-chunk kernels
_DET_TOPK_BLOCK = 1024


@triton.jit
def _det_topk_key(x):
    """Order-preserving integer key of a float32 logit, ranking like
    torch.topk: every NaN above +inf (canonical NaN), -0.0 equal to +0.0."""
    x = tl.where(x != x, float("nan"), x + 0.0)  # +0.0 folds -0.0 into +0.0
    bits = x.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    return tl.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits + 0x80000000)


@triton.jit
def _det_topk_threshold(
    logits_row, sel_row, start, end, k, RELATIVE: tl.constexpr, KP2: tl.constexpr
):
    """The smallest key among a row's selection (2^32, above every key, if it
    is empty)."""
    kk = tl.arange(0, KP2)
    sel = tl.load(sel_row + kk, mask=kk < k, other=-1).to(tl.int64)
    col = sel + start if RELATIVE else sel
    valid = (kk < k) & (sel >= 0) & (col >= start) & (col < end)
    vals = tl.load(logits_row + col, mask=valid, other=0.0)
    keys = tl.where(valid, _det_topk_key(vals), 0x100000000)
    return tl.min(keys, axis=0)


@triton.jit
def _det_topk_masks(x, m, t, DROP_NEG_INF: tl.constexpr):
    kx = _det_topk_key(x)
    gt = m & (kx > t)
    eq = m & (kx == t)
    if DROP_NEG_INF:
        gt = gt & (x != float("-inf"))
        eq = eq & (x != float("-inf"))
    return gt, eq


@triton.jit
def _det_topk_emit(
    x, m, cols, start, t, out_row, k, n_valid, n_ties, sel_before, eq_before,
    RELATIVE: tl.constexpr, DROP_NEG_INF: tl.constexpr,
):
    """Write one block's selected columns at their final positions: the row's
    selection is every column above the threshold plus its first ``n_ties``
    tied columns, stored in descending column order (what ``_maybe_sort_topk_``
    makes), so position = n_valid - 1 - ascending rank. Returns the updated
    counts of selected and of tied columns before the next block."""
    gt, eq = _det_topk_masks(x, m, t, DROP_NEG_INF)
    tie_idx = eq_before + tl.cumsum(eq.to(tl.int32), axis=0) - 1
    take = gt | (eq & (tie_idx < n_ties))
    rank = sel_before + tl.cumsum(take.to(tl.int32), axis=0) - 1
    pos = n_valid - 1 - rank
    val = (cols - start) if RELATIVE else cols
    tl.store(out_row + pos, val.to(tl.int32), mask=take & (pos >= 0) & (pos < k))
    return (sel_before + tl.sum(take.to(tl.int32), axis=0),
            eq_before + tl.sum(eq.to(tl.int32), axis=0))


@triton.jit(do_not_specialize=["stride", "k"])
def _det_topk_fused_kernel(
    logits_ptr, stride, starts_ptr, ends_ptr, topk_ptr, k,
    HAS_STARTS: tl.constexpr, RELATIVE: tl.constexpr, DROP_NEG_INF: tl.constexpr,
    KP2: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr,
):
    """Rows of at most CHUNK columns, one program per row: threshold, counts,
    then the final order, in place (the selection is read before any write)."""
    row = tl.program_id(0).to(tl.int64)
    if HAS_STARTS:
        start = tl.load(starts_ptr + row).to(tl.int64)
    else:
        start = tl.zeros((), dtype=tl.int64)
    end = tl.maximum(tl.load(ends_ptr + row).to(tl.int64), start)
    base = logits_ptr + row * stride
    out = topk_ptr + row * k
    t = _det_topk_threshold(base, out, start, end, k, RELATIVE, KP2)
    offs = tl.arange(0, BLOCK)
    n_gt = tl.zeros((), dtype=tl.int32)
    n_eq = tl.zeros((), dtype=tl.int32)
    for b in tl.static_range(0, CHUNK, BLOCK):
        cols = start + b + offs
        m = cols < end
        x = tl.load(base + cols, mask=m, other=float("-inf"))
        gt, eq = _det_topk_masks(x, m, t, DROP_NEG_INF)
        n_gt += tl.sum(gt.to(tl.int32), axis=0)
        n_eq += tl.sum(eq.to(tl.int32), axis=0)
    n_ties = tl.maximum(tl.minimum(n_eq, k - n_gt), 0)
    n_valid = tl.minimum(n_gt + n_ties, k)
    sel_before = tl.zeros((), dtype=tl.int32)
    eq_before = tl.zeros((), dtype=tl.int32)
    for b in tl.static_range(0, CHUNK, BLOCK):
        cols = start + b + offs
        m = cols < end
        x = tl.load(base + cols, mask=m, other=float("-inf"))
        sel_before, eq_before = _det_topk_emit(
            x, m, cols, start, t, out, k, n_valid, n_ties, sel_before, eq_before,
            RELATIVE, DROP_NEG_INF,
        )
    for i in range(0, k, BLOCK):
        idx = i + offs
        tl.store(out + idx, -1, mask=(idx < k) & (idx >= n_valid))


@triton.jit(do_not_specialize=["stride", "k", "n_chunks"])
def _det_topk_count_kernel(
    logits_ptr, stride, starts_ptr, ends_ptr, topk_ptr, t_ptr, cnt_ptr, k, n_chunks,
    HAS_STARTS: tl.constexpr, RELATIVE: tl.constexpr, DROP_NEG_INF: tl.constexpr,
    KP2: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr,
):
    """Long rows, per (row, chunk): the threshold (every program computes it
    from the selection; chunk 0 stores it) and the chunk's counts of columns
    above and equal to it. Integer counts, no atomics."""
    row = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    if HAS_STARTS:
        start = tl.load(starts_ptr + row).to(tl.int64)
    else:
        start = tl.zeros((), dtype=tl.int64)
    end = tl.maximum(tl.load(ends_ptr + row).to(tl.int64), start)
    base = logits_ptr + row * stride
    t = _det_topk_threshold(base, topk_ptr + row * k, start, end, k, RELATIVE, KP2)
    if c == 0:
        tl.store(t_ptr + row, t)
    offs = tl.arange(0, BLOCK)
    n_gt = tl.zeros((), dtype=tl.int32)
    n_eq = tl.zeros((), dtype=tl.int32)
    c0 = start + c * CHUNK
    if c0 < end:
        for b in tl.static_range(0, CHUNK, BLOCK):
            cols = c0 + b + offs
            m = cols < end
            x = tl.load(base + cols, mask=m, other=float("-inf"))
            gt, eq = _det_topk_masks(x, m, t, DROP_NEG_INF)
            n_gt += tl.sum(gt.to(tl.int32), axis=0)
            n_eq += tl.sum(eq.to(tl.int32), axis=0)
    tl.store(cnt_ptr + (row * n_chunks + c) * 2, n_gt)
    tl.store(cnt_ptr + (row * n_chunks + c) * 2 + 1, n_eq)


@triton.jit(do_not_specialize=["stride", "k", "n_chunks"])
def _det_topk_write_kernel(
    logits_ptr, stride, starts_ptr, ends_ptr, topk_ptr, t_ptr, cnt_ptr, k, n_chunks,
    HAS_STARTS: tl.constexpr, RELATIVE: tl.constexpr, DROP_NEG_INF: tl.constexpr,
    NCP2: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr,
):
    """Long rows, per (row, chunk): the chunk's selected columns at their final
    positions (offsets from the counts of the chunks before it), in place: the
    count kernel, launched before, has read the selection. Chunk 0 pads."""
    row = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    if HAS_STARTS:
        start = tl.load(starts_ptr + row).to(tl.int64)
    else:
        start = tl.zeros((), dtype=tl.int64)
    end = tl.maximum(tl.load(ends_ptr + row).to(tl.int64), start)
    t = tl.load(t_ptr + row)
    jj = tl.arange(0, NCP2)
    cm = jj < n_chunks
    cnt = cnt_ptr + row * n_chunks * 2
    gts = tl.load(cnt + jj * 2, mask=cm, other=0)
    eqs = tl.load(cnt + jj * 2 + 1, mask=cm, other=0)
    tot_gt = tl.sum(gts, axis=0)
    n_ties = tl.maximum(tl.minimum(tl.sum(eqs, axis=0), k - tot_gt), 0)
    n_valid = tl.minimum(tot_gt + n_ties, k)
    gt_before = tl.sum(tl.where(jj < c, gts, 0), axis=0)
    eq_before = tl.sum(tl.where(jj < c, eqs, 0), axis=0)
    sel_before = gt_before + tl.minimum(eq_before, n_ties)
    out = topk_ptr + row * k
    base = logits_ptr + row * stride
    offs = tl.arange(0, BLOCK)
    c0 = start + c.to(tl.int64) * CHUNK
    if c0 < end:
        for b in tl.static_range(0, CHUNK, BLOCK):
            cols = c0 + b + offs
            m = cols < end
            x = tl.load(base + cols, mask=m, other=float("-inf"))
            sel_before, eq_before = _det_topk_emit(
                x, m, cols, start, t, out, k, n_valid, n_ties, sel_before, eq_before,
                RELATIVE, DROP_NEG_INF,
            )
    if c == 0:
        for i in range(0, k, BLOCK):
            idx = i + offs
            tl.store(out + idx, -1, mask=(idx < k) & (idx >= n_valid))


def _deterministic_topk_compact_(
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    topk: torch.Tensor,
    *,
    relative: bool,
    drop_neg_inf: bool,
    max_len: int | None = None,
) -> None:
    """Make a fast top-k kernel's result ``topk`` ([rows, k] int32, any order,
    -1 padding) deterministic, in place: every column above the k-th value plus
    the lowest tied columns, in descending column order with -1 last, i.e.
    already as ``_maybe_sort_topk_`` leaves it. ``relative``: the indices count
    from each row's start (prefill); ``drop_neg_inf``: -inf logits are padding
    (prefill's reference, ``_deterministic_topk_prefill_``). ``max_len`` bounds
    the rows' lengths in columns (host side; default the logits' width): rows of
    up to 4,096 columns take one launch, longer ones two."""
    rows, k = topk.shape
    if rows == 0 or k == 0:
        return
    assert logits.stride(1) == 1
    width = logits.shape[1]
    span = width if max_len is None else max(1, min(width, int(max_len)))
    n_chunks = triton.cdiv(span, _DET_TOPK_CHUNK)
    out = topk if topk.is_contiguous() else topk.contiguous()
    starts = row_starts if row_starts is not None else row_ends
    flags = dict(
        HAS_STARTS=row_starts is not None,
        RELATIVE=relative,
        DROP_NEG_INF=drop_neg_inf,
        CHUNK=_DET_TOPK_CHUNK,
        BLOCK=_DET_TOPK_BLOCK,
    )
    if n_chunks == 1:
        _det_topk_fused_kernel[(rows,)](
            logits, logits.stride(0), starts, row_ends, out, k,
            KP2=triton.next_power_of_2(k), **flags,
        )
    else:
        t = torch.empty(rows, dtype=torch.int64, device=logits.device)
        cnt = torch.empty(rows, n_chunks, 2, dtype=torch.int32, device=logits.device)
        _det_topk_count_kernel[(rows, n_chunks)](
            logits, logits.stride(0), starts, row_ends, out, t, cnt, k, n_chunks,
            KP2=triton.next_power_of_2(k), **flags,
        )
        _det_topk_write_kernel[(rows, n_chunks)](
            logits, logits.stride(0), starts, row_ends, out, t, cnt, k, n_chunks,
            NCP2=triton.next_power_of_2(n_chunks), **flags,
        )
    if out is not topk:
        topk.copy_(out)


@eager_break_during_capture
def sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_pool_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    # kpool params (Plan-A: gate is consumed at write time and read back at
    # topk time to softmax-weight the pool).
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    # Paged tail cache (in-progress pool's raw K + gate score), replacing the
    # transient _DECODE_TAIL ring. tail_prefix resolves attn_metadata[tail_prefix]
    # for the tail group's token-granular slot_mapping. None on the dummy/profiling
    # path and when the tail cache is disabled.
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
    topk_backend: str = "auto",
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Reserve profiler-visible memory for the worst-case decode logits,
        # whose shape is [B * next_n, max_pool_len]. This profiling branch
        # returns before invoking the logits kernel itself.
        cfg = get_current_vllm_config_or_none()
        worst_decode_tokens = 0
        if cfg is not None:
            sched = cfg.scheduler_config
            num_spec = (
                cfg.speculative_config.num_speculative_tokens
                if cfg.speculative_config is not None
                else 0
            )
            worst_decode_tokens = min(
                sched.max_num_seqs * (num_spec + 1),
                sched.max_num_batched_tokens,
            )
        # float32 logits -> 4 bytes/element; uint8 sentinel so elems == bytes.
        decode_logits_elems = worst_decode_tokens * max_pool_len * 4
        prefill_cap_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        max_logits_elems = max(decode_logits_elems, prefill_cap_elems)
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        # The dummy run returns before exchange_topk executes, so reserve its
        # worst-case transient for the memory profile.
        _reserve_row_shard_exchange_transient(topk_indices_buffer, hidden_states.device)

        return topk_indices_buffer
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
    # Pools are the DCP sharding unit of the index K cache.
    pool_cp_interleave = (
        cp_kv_cache_interleave_size // max(index_kpool, 1)
        if dcp_world_size > 1
        else cp_kv_cache_interleave_size
    )

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        if index_kpool > 1 and gate_score is not None and compress_ape is not None:
            # kpool prefill write: pool kpool consecutive prefill tokens via
            # softmax(gate+ape)-weighted sum -> Hadamard -> fp8 -> pool slots.
            # Decode tokens (the first num_decode_tokens in the batch) cannot be
            # pooled here — their pool's earlier tokens are not in this batch —
            # so they are deferred to the tail-buffer kernel in has_decode.
            # compress_ratio == index_kpool makes slot_mapping pool-granular.
            n_prefill = num_tokens - num_decode_tokens
            if n_prefill > 0:
                # decode tokens are batched first; prefill tokens follow.
                prefill_slice = slice(num_decode_tokens, num_tokens)
                _kpool_compress_insert(
                    k[prefill_slice],
                    gate_score[prefill_slice],
                    compress_ape,
                    kv_cache,
                    slot_mapping[prefill_slice],
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
                # Persist each request's incomplete prefill pool so decode can
                # finish it, including after PD transfer. Tail slots use
                # ``pos % kpool`` within the request's tail block. Processing
                # only the batch's trailing tokens would miss all but the last
                # request in a multi-request prefill.
                if tail_kv_cache is not None and tail_prefix is not None:
                    tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                    if tail_meta is not None:
                        assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                        kpool_ops.kpool_seed_tail_cache(
                            tail_kv_cache,
                            k[prefill_slice],
                            gate_score[prefill_slice],
                            tail_meta.slot_mapping[prefill_slice],
                            index_kpool,
                            head_dim,
                        )
        else:
            # standard: per-token fp8 quant + scatter (all tokens).
            assert scale_fmt is not None
            ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )

    # VLLM_EXP_INDEXER_FEWER_COPIES=1: a pure decode batch on the pool path
    # writes its top-k rows, their padding and the rows after them in one
    # kernel (indexer_fewer_copies.expand_pools_into), so no fill here
    fewer_copies = (
        has_decode
        and not has_prefill
        and index_kpool > 1
        and positions is not None
        and dcp_world_size <= 1
        and indexer_fewer_copies.enabled()
        and not attn_metadata_narrowed.decode.requires_padding
        and attn_metadata_narrowed.max_seq_len > topk_tokens
        and topk_indices_buffer.shape[1] >= topk_tokens + index_kpool - 1
    )
    if not fewer_copies:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Short sequences select every pool, so skip sparse scoring and fill
        # the top-k buffer with all causal token indices. The index-K cache was
        # already written above.
        n_prefill_sf = num_tokens - num_decode_tokens
        # Host-side short-prefill predicate: max_prefill_seq_len is computed
        # in the metadata builder (exact for prefill rows) and equals
        # positions[prefill_slice].max() + 1, so this replaces a
        # positions.max().item() device sync per layer. -1 (unknown metadata)
        # falls back to the device-side check.
        if prefill_metadata.max_prefill_seq_len >= 0:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and prefill_metadata.max_prefill_seq_len <= topk_tokens
            )
        else:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
                <= topk_tokens
            )
        if short_prefill:
            # short_prefill is only True when positions is not None (above),
            # but narrow explicitly for the indexer below.
            assert positions is not None
            _pos = positions[num_decode_tokens:num_tokens].to(torch.int32)
            _buf = topk_indices_buffer[num_decode_tokens:num_tokens]
            _fill_causal_indices(_buf, _pos)
        # Short prefills score nothing locally; suppress the shard there.
        row_shard = None
        if not short_prefill:
            row_shard = PrefillRowShard.from_metadata(
                prefill_metadata, num_decode_tokens
            )

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks if not short_prefill else ():
            # Under DCP this rank only gathers the pools it owns.
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]

            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            row_start, row_end = chunk.token_start, chunk.token_end
            cu_seqlen_ks, cu_seqlen_ke = chunk.cu_seqlen_ks, chunk.cu_seqlen_ke
            if row_shard is not None:
                narrowed = row_shard.narrow(chunk)
                if narrowed is None:
                    continue
                row_start, row_end, cu_seqlen_ks, cu_seqlen_ke = narrowed

            q_slice = q_quant[row_start:row_end]
            q_scale_slice = q_scale[row_start:row_end] if q_scale is not None else None
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

            num_rows = q_slice.shape[0]
            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((num_rows, 0), dtype=torch.float32)
            else:
                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights[row_start:row_end],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    clean_logits=False,
                )

            # kpool: logits are pool-granular (compress_ratio == index_kpool),
            # so topk selects pools. We pick topk_tokens // kpool pools then
            # expand each pool back to its kpool constituent tokens.
            select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
            if index_kpool > 1:
                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=logits.device
                )
                topk_dst = pool_topk
            else:
                topk_dst = topk_indices_buffer[row_start:row_end, :topk_tokens]

            compacted = False
            if logits.shape[1] == 0:
                topk_dst.fill_(-1)
            elif (
                envs.VLLM_SPARSE_INDEXER_DETERMINISTIC_TOPK
                and not _fast_deterministic_topk()
            ):
                _deterministic_topk_prefill_(
                    logits, cu_seqlen_ks, cu_seqlen_ke, topk_dst
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
                if envs.VLLM_SPARSE_INDEXER_DETERMINISTIC_TOPK:
                    _deterministic_topk_compact_(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_dst,
                        relative=True,
                        drop_neg_inf=True,
                    )
                    compacted = True
            # Local pool ids -> request-relative global pool ids.
            _merge_dcp_topk_global(
                logits,
                topk_dst,
                select_k,
                dcp_rank,
                dcp_world_size,
                pool_cp_interleave,
                row_starts=cu_seqlen_ks,
            )
            if not (compacted and dcp_world_size <= 1):  # else already in this order
                _maybe_sort_topk_(topk_dst)

            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:
                    # Fused expand-pools + append-tail into one Triton kernel
                    # (replaces ~25 elementwise ops). seq_len is token-granular
                    # (pos+1); the kernel derives pool_len internally.
                    q_seq = positions[row_start:row_end].to(torch.int32) + 1
                    expanded = kpool_ops.expand_pools_and_append_tail(
                        pool_ids, q_seq, index_kpool
                    )
                else:
                    valid = pool_ids >= 0
                    expanded = kpool_ops.expand_pools_to_tokens(
                        pool_ids, valid, topk_tokens, index_kpool
                    )
                topk_indices_buffer[row_start:row_end, : expanded.shape[-1]] = expanded

        if row_shard is not None:
            # The k-pool expansion appends the request's incomplete tail
            # (index_kpool - 1 entries) after the logical top-k history; those
            # columns are part of the attention index and are exchanged too.
            row_shard.exchange_topk(topk_indices_buffer, topk_tokens + index_kpool - 1)

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim+4] for writes
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)

        # Update the tail before reading logits; completed pools are compressed
        # into the slot supplied by slot_mapping.
        # Spec verification groups tokens by request and preserves position
        # order so each token is stashed before the next completes its pool.
        # Positions must remain token-granular because the kernel derives the
        # pool phase and tail index from ``pos % kpool``.
        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
        ):
            num_requests = attn_metadata_narrowed.num_decodes
            # Kpool writes must recover the original request grouping after the
            # indexer's flattened decode path. Host metadata avoids a CUDA graph
            # sync when choosing the uniform or padded layout.
            per_req_lens = decode_metadata.per_req_decode_lens
            if per_req_lens is not None:
                use_uniform = (
                    decode_metadata.decode_is_uniform
                    and num_decode_tokens
                    == num_requests * decode_metadata.write_max_decode_len
                )
                group_lens = per_req_lens
                lmax = decode_metadata.write_max_decode_len
            else:
                # Legacy metadata without per-request lens: fall back to the
                # host-side requires_padding flag. Unreached now (per-request
                # lens is always populated for decode), kept defensive.
                use_uniform = not decode_metadata.requires_padding
                group_lens = decode_metadata.decode_lens
                lmax = int(decode_metadata.decode_lens.max().item())
            if not use_uniform:
                # Non-uniform decode_lens (mixed plain-decode + spec-verify, or
                # a variable MTP-verify batch): scatter actual tokens into a
                # padded [B, lmax] layout. int32 tensors can't go through
                # pack_seq_triton (float/uint8 only). The scatter indices are
                # shared by all five scatters below (and the tail slot one).
                scatter_idx = _build_decode_scatter_indices(
                    group_lens, num_requests, num_decode_tokens
                )
                dec_k = _scatter_decode_tokens_by_request(
                    k[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
                )
                dec_gate = _scatter_decode_tokens_by_request(
                    gate_score[:num_decode_tokens],
                    0,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_slot = _scatter_decode_tokens_by_request(
                    slot_mapping[:num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_pos = _scatter_decode_tokens_by_request(
                    positions[:num_decode_tokens].to(torch.int32),
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                next_n = num_decode_tokens // num_requests
                shape2 = (num_requests, next_n)
                dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
                dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
                dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
                # with fewer_copies the int64 positions as they are (the
                # kernel uses them only in integer arithmetic)
                dec_pos = (
                    positions[:num_decode_tokens]
                    if fewer_copies
                    else positions[:num_decode_tokens].to(torch.int32)
                ).view(shape2)
            tail_meta = (
                attn_metadata.get(_resolve_layer_name(tail_prefix))
                if tail_prefix is not None
                else None
            )
            # Paged tail cache replaces the transient _DECODE_TAIL ring. Group
            # the tail group's token-granular slot_mapping per-request, mirroring
            # dec_slot / dec_pos, so the kernel gets each request's current-token
            # tail slot (block * kpool + pos % kpool).
            if tail_meta is not None:
                assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            if tail_meta is None or tail_kv_cache is None:
                dec_tail_slot = None
            elif not use_uniform:
                dec_tail_slot = _scatter_decode_tokens_by_request(
                    tail_meta.slot_mapping[:num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                dec_tail_slot = tail_meta.slot_mapping[:num_decode_tokens].view(shape2)
            # The compress kernel writes the raw fp8 cache (not the quant view);
            # pass the underlying kv_cache, not kv_cache_quant_view.
            if dec_tail_slot is not None:
                # Single batched launch over [num_requests, next_n] replaces the
                # per-token sequential loop. The kernel iterates each request's
                # tokens in position order internally, preserving the
                # pool-completion read-after-stash dependency that the loop
                # provided. Inputs are already grouped per request (uniform:
                # view; non-uniform: _scatter_decode_tokens_by_request padded to
                # [B, lmax]) — no per-token .contiguous() copies needed.
                kpool_ops.kpool_decode_update_and_maybe_write_cache_batched(
                    kv_cache_raw,
                    tail_kv_cache,
                    dec_tail_slot,
                    dec_k,
                    dec_gate,
                    compress_ape,
                    dec_slot,
                    dec_pos,
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
        if current_platform.is_cuda_alike() and _fill_short_decode_causal_indices(
            topk_indices_buffer,
            positions,
            num_decode_tokens,
            attn_metadata_narrowed.max_seq_len,
            topk_tokens,
        ):
            return topk_indices_buffer
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # Padding also covers short chunked prefills classified as decode.
            # MXFP4 uses zero-byte padding so padded slots dequantize to zero.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(-1, *weights.shape[1:])
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
            padded_weights = weights[:num_decode_tokens]
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

        logits = fp8_fp4_paged_mqa_logits(
            (padded_q_quant_cast, padded_q_scale),
            kv_cache,
            padded_weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_pool_len,
            clean_logits=False,
            indices=decode_metadata.indices,
        )
        num_rows = logits.shape[0]
        # kpool: logits are pool-granular -> select topk_tokens//kpool pools,
        # then expand each pool back to its kpool tokens.
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # Shared dispatcher with the DSA sparse indexer, so the kpool select
        # honors kernel_config.sparse_indexer_topk_backend and picks between
        # cooperative/persistent by batch size instead of always taking the
        # same kernel.
        fast_det = (
            envs.VLLM_SPARSE_INDEXER_DETERMINISTIC_TOPK and _fast_deterministic_topk()
        )
        # VLLM_EXP_INDEXER_DEEPSELECT=1: DeepSelect's deterministic top-k
        # (deepselect_topk.py), already in _maybe_sort_topk_'s order; rows
        # are pools when index_kpool > 1 (max_seq_len counts tokens)
        dsel = (
            envs.VLLM_SPARSE_INDEXER_DETERMINISTIC_TOPK
            and dcp_world_size <= 1
            and deepselect_topk_(
                logits,
                SparseIndexerTopk._row_ends(seq_lens, next_n, logits.shape[0]),
                topk_dst,
                select_k,
                -(-attn_metadata_narrowed.max_seq_len // max(index_kpool, 1)) + 1,
            )
        )
        if not dsel:
            get_indexer_topk(
                "torch"
                if envs.VLLM_SPARSE_INDEXER_DETERMINISTIC_TOPK and not fast_det
                else topk_backend
            )(
                logits,
                seq_lens,
                next_n,
                topk_dst,
                select_k,
                attn_metadata_narrowed.max_seq_len,
            )
        if fast_det and not dsel:
            _deterministic_topk_compact_(
                logits,
                None,
                SparseIndexerTopk._row_ends(seq_lens, next_n, logits.shape[0]),
                topk_dst,
                relative=False,
                drop_neg_inf=False,
                max_len=attn_metadata_narrowed.max_seq_len,
            )
        _merge_dcp_topk_global(
            logits,
            topk_dst,
            select_k,
            dcp_rank,
            dcp_world_size,
            pool_cp_interleave,
        )
        if not dsel and not (fast_det and dcp_world_size <= 1):  # else in order
            _maybe_sort_topk_(topk_dst)

        # Resolve to token-level indices in the output buffer.
        if fewer_copies:
            indexer_fewer_copies.expand_pools_into(
                pool_topk,
                positions,
                topk_indices_buffer,
                min(hidden_states.shape[0], topk_indices_buffer.shape[0]),
                index_kpool,
            )
            return topk_indices_buffer
        if index_kpool > 1:
            pool_ids = pool_topk.to(torch.int64)
            n = pool_topk.shape[0]
            # Decode seq_lens are pool-granular; recover token lengths from
            # positions using the padded [B, next_n] row layout when needed.
            if positions is not None:
                dec_seq = _decode_topk_seq_lens(
                    positions,
                    decode_lens,
                    num_decode_tokens,
                    batch_size,
                    next_n,
                    decode_metadata.requires_padding,
                )
            else:
                dec_seq = decode_metadata.seq_lens[:n]
                if dec_seq.ndim == 2:
                    dec_seq = dec_seq[:, -1]
                dec_seq = dec_seq.to(torch.int32)
            out = kpool_ops.expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            # Drop padded query rows introduced by the next_n padding above.
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_pool_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_pool_len = max_pool_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )
        _cfg = get_current_vllm_config_or_none()
        self.topk_backend = (
            _cfg.kernel_config.sparse_indexer_topk_backend
            if _cfg is not None
            else "auto"
        )
        _parallel = _cfg.parallel_config if _cfg is not None else None
        self._parallel_config = _parallel
        self._cp_kv_cache_interleave_size: int | None = None
        self.dcp_world_size = 1
        self.dcp_rank = 0
        if _parallel is not None and _parallel.decode_context_parallel_size > 1:
            if _parallel.prefill_context_parallel_size > 1:
                raise NotImplementedError(
                    "SparseAttnIndexerKpool does not support PCP+DCP."
                )
            self.dcp_world_size = _parallel.decode_context_parallel_size
            self.dcp_rank = get_dcp_group().rank_in_group
            if current_platform.is_cuda() and has_cutedsl():
                from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (  # noqa: E501
                    _PACK_DCP_TOPK_CANDIDATES_KERNEL,
                    _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL,
                )

                _PACK_DCP_TOPK_CANDIDATES_KERNEL.register_warmup()
                _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL.register_warmup()

    @property
    def cp_kv_cache_interleave_size(self) -> int:
        """Resolved lazily like ``SparseAttnIndexer``: NIXL P/D can adjust it
        after the model is built, and the indexer metadata builder sees the
        adjusted value.
        """
        if self.dcp_world_size == 1:
            return 1
        if self._cp_kv_cache_interleave_size is None:
            assert self._parallel_config is not None
            value = self._parallel_config.cp_kv_cache_interleave_size
            if isinstance(get_forward_context().attn_metadata, dict):
                self._cp_kv_cache_interleave_size = value
            return value
        return self._cp_kv_cache_interleave_size

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        return self.forward_cuda(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return sparse_attn_indexer_kpool(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_pool_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
            self.topk_backend,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
        )
