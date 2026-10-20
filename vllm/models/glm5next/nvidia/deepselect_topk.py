# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSelect top-k for GLM-5.3's indexer decode rows on sm_120 (experiment).

With ``VLLM_SPARSE_INDEXER_DETERMINISTIC_TOPK=1`` the decode rows' top-k runs
``torch.topk`` plus a sort: deterministic, but 100-170 us per call.
``VLLM_EXP_INDEXER_DEEPSELECT=1`` runs DeepSeek's DeepSelect kernel instead
(MIT, github.com/deepseek-ai/deepselect), built for sm_120 into the shared
library named by ``VLLM_EXP_DEEPSELECT_LIB`` (a C ABI, loaded with ctypes). A
row's selection depends only on that row, so the result is deterministic and
independent of the batch; the indices come out in the order
``_maybe_sort_topk_`` gives (descending, -1 last). Rows up to
``VLLM_EXP_DEEPSELECT_SINGLE_MAX`` columns (default 163840, where the two
ways cost the same per call on an RTX PRO 6000) run in one pass; longer ones
are cut into slices that are selected in parallel, then merged.
"""

import ctypes
import functools
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_SINGLE_MAX = int(os.environ.get("VLLM_EXP_DEEPSELECT_SINGLE_MAX", "163840"))
_MIN_WORKSPACE = 4 << 20
_P = ctypes.c_void_p
_I = ctypes.c_int
_LL = ctypes.c_longlong


@functools.cache
def _lib():
    if os.environ.get("VLLM_EXP_INDEXER_DEEPSELECT", "0") != "1":
        return None
    path = os.environ.get("VLLM_EXP_DEEPSELECT_LIB", "")
    if not path:
        logger.warning(
            "VLLM_EXP_INDEXER_DEEPSELECT=1 without VLLM_EXP_DEEPSELECT_LIB: unused"
        )
        return None
    lib = ctypes.CDLL(path)
    lib.dsel_topk_fp32.restype = _I
    lib.dsel_topk_fp32.argtypes = [_P, _I, _I, _LL, _P, _P, _I, _I, _I, _I, _I]
    lib.dsel_topk_fp32.argtypes += [_P, _P, _LL]
    lib.dsel_workspace_bytes.restype = _LL
    lib.dsel_workspace_bytes.argtypes = [_I, _I, _LL, _I, _I]
    lib.dsel_last_error.restype = ctypes.c_char_p
    logger.info(
        "Indexer decode top-k: DeepSelect from %s (one pass up to %d columns)",
        path,
        _SINGLE_MAX,
    )
    return lib


_workspace: dict[torch.device, torch.Tensor] = {}


def deepselect_topk_(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
    k: int,
    max_len: int,
) -> bool:
    """Write each row's top-k columns into ``out`` (descending, -1 last).

    ``logits`` (rows, width) FP32 with 1024-byte aligned rows; ``row_ends``
    int32 (rows,); ``out`` (rows, >= k) int32. Returns False when it did not
    run (switched off, unsupported input, or during a graph capture before
    its workspace exists); the caller then takes its own path.
    """
    lib = _lib()
    if lib is None:
        return False
    rows = logits.shape[0] if logits.dim() == 2 else -1
    if (
        rows <= 0
        or logits.dtype != torch.float32
        or logits.stride(1) != 1
        or out.dtype != torch.int32
        or out.dim() != 2
        or out.stride(1) != 1
        or out.shape[0] != rows
        or row_ends.dtype != torch.int32
        or row_ends.numel() != rows
        or not row_ends.is_contiguous()
    ):
        return False
    width = logits.shape[1]
    max_len = max(0, min(int(max_len), width))
    need = lib.dsel_workspace_bytes(rows, width, logits.stride(0), max_len, _SINGLE_MAX)
    ws = _workspace.get(logits.device)
    if need > 0 and (ws is None or ws.numel() < need):
        if torch.cuda.is_current_stream_capturing():
            return False
        size = max(need, _MIN_WORKSPACE)
        ws = torch.empty(size, dtype=torch.uint8, device=logits.device)
        _workspace[logits.device] = ws
    rc = lib.dsel_topk_fp32(
        logits.data_ptr(),
        rows,
        width,
        logits.stride(0),
        row_ends.data_ptr(),
        out.data_ptr(),
        out.stride(0),
        k,
        max_len,
        _SINGLE_MAX,
        1,
        torch.cuda.current_stream(logits.device).cuda_stream,
        ws.data_ptr() if ws is not None else None,
        ws.numel() if ws is not None else 0,
    )
    if rc != 0:
        logger.warning_once(
            "DeepSelect top-k not used: %s", lib.dsel_last_error().decode()
        )
        return False
    return True
