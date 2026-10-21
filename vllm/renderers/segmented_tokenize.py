# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Segmented tokenization of long prompts (experiment).

Opt-in with ``VLLM_EXP_SEGMENTED_TOKENIZE=1``. A fast tokenizer encodes one long
string on one thread, ~0.5-0.8 M tokens/s, so a 630K-token chat prompt spends ~1 s
in tokenization before it reaches the engine, even when its whole prefix is cached.
The tokenizer splits a text at its added tokens first and encodes the pieces between
them independently, so encoding those pieces as one batch (which the Rust library
runs on several threads) and joining the ids gives the same ids, as long as no added
token strips the whitespace around it or is matched after normalization and there is
no normalizer. Prompts shorter than ``VLLM_EXP_SEGMENTED_TOKENIZE_MIN_CHARS``
characters (default 65536), requests with offsets, and tokenizers whose
post-processor adds special tokens keep the usual path.

With ``VLLM_EXP_SEGMENTED_TOKENIZE_CACHE=N`` (N > 0) the ids of recently encoded
segments are also kept, up to N ids in all (least recently used first out): an
agent session resends its whole history every turn, so most segments of its next
prompt were encoded for the previous one, and only the new ones are encoded.
"""

import contextlib
import os
import re
import threading
import weakref
from array import array
from collections import OrderedDict
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_splitters: "weakref.WeakKeyDictionary[Any, tuple[Any, re.Pattern, bool] | None]" = (
    weakref.WeakKeyDictionary()
)


def _enabled() -> bool:
    return os.environ.get("VLLM_EXP_SEGMENTED_TOKENIZE", "0") == "1"


def _min_chars() -> int:
    return int(os.environ.get("VLLM_EXP_SEGMENTED_TOKENIZE_MIN_CHARS", "65536"))


def _cache_ids() -> int:
    return int(os.environ.get("VLLM_EXP_SEGMENTED_TOKENIZE_CACHE", "0") or 0)


class _SegmentCache:
    """Ids of recently encoded segments, least recently used first out, at most
    ``max_ids`` ids in all. Safe to share between threads."""

    def __init__(self, max_ids: int) -> None:
        self.max_ids = max_ids
        self.num_ids = 0
        self.entries: OrderedDict[str, array] = OrderedDict()
        self.lock = threading.Lock()

    def encode(self, backend: Any, parts: list[str]) -> list[int]:
        with self.lock:
            found = [self.entries.get(p) for p in parts]
            for p, ids in zip(parts, found):
                if ids is not None:
                    self.entries.move_to_end(p)
        new = list(dict.fromkeys(p for p, ids in zip(parts, found) if ids is None))
        encoded = {
            p: array("i", enc.ids)
            for p, enc in zip(
                new, backend.encode_batch(new, add_special_tokens=False) if new else []
            )
        }
        out: list[int] = []
        for p, ids in zip(parts, found):
            out.extend(encoded[p] if ids is None else ids)
        with self.lock:
            for p, ids in encoded.items():
                if p not in self.entries and len(ids) <= self.max_ids:
                    self.entries[p] = ids
                    self.num_ids += len(ids)
            while self.num_ids > self.max_ids:
                self.num_ids -= len(self.entries.popitem(last=False)[1])
        return out


_caches: "weakref.WeakKeyDictionary[Any, _SegmentCache]" = weakref.WeakKeyDictionary()


def _cache(tokenizer: Any) -> "_SegmentCache | None":
    max_ids = _cache_ids()
    if max_ids <= 0:
        return None
    try:
        return _caches[tokenizer]
    except KeyError:
        pass
    except TypeError:  # not weakly referenceable: no cache
        return None
    cache = _caches.setdefault(tokenizer, _SegmentCache(max_ids))
    logger.info_once(
        "The segments of long prompts are cached, up to %d token ids "
        "(VLLM_EXP_SEGMENTED_TOKENIZE_CACHE)",
        max_ids,
    )
    return cache


def _splitter(tokenizer: Any) -> "tuple[Any, re.Pattern, bool] | None":
    """The backend tokenizer, a pattern for its added tokens and whether added
    special tokens change an encoding, or None when splitting would change the
    ids."""
    try:
        return _splitters[tokenizer]
    except (KeyError, TypeError):
        pass
    result = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if (
        backend is not None
        and hasattr(backend, "encode_batch")
        and backend.normalizer is None
    ):
        added = list(backend.get_added_tokens_decoder().values())
        if added and not any(
            t.lstrip or t.rstrip or t.single_word or t.normalized for t in added
        ):
            # longest first: at a position the longest added token wins, as in
            # the tokenizer
            contents = sorted(
                {t.content for t in added if t.content}, key=len, reverse=True
            )
            # do the added special tokens of a post-processor change anything?
            probe = "a" + contents[-1] + "b"
            adds = (
                tokenizer(probe, add_special_tokens=True)["input_ids"]
                != tokenizer(probe, add_special_tokens=False)["input_ids"]
            )
            result = (
                backend,
                re.compile("(" + "|".join(re.escape(c) for c in contents) + ")"),
                adds,
            )
    with contextlib.suppress(TypeError):  # not weakly referenceable: no cache
        _splitters[tokenizer] = result
    if result is not None:
        logger.info_once(
            "Long prompts are tokenized in segments split at the added tokens "
            "(VLLM_EXP_SEGMENTED_TOKENIZE, from %d characters)",
            _min_chars(),
        )
    return result


def segmented_encode(
    tokenizer: Any, text: str, kwargs: dict[str, Any]
) -> list[int] | None:
    """``tokenizer(text, **kwargs)["input_ids"]`` computed in segments, or None
    for the usual path."""
    if not _enabled() or len(text) < _min_chars():
        return None
    if set(kwargs) - {"truncation", "max_length", "add_special_tokens"}:
        return None
    truncate = bool(kwargs.get("truncation")) and kwargs.get("max_length") is not None
    side = getattr(tokenizer, "truncation_side", "right")
    if truncate and side not in ("left", "right"):
        return None
    split = _splitter(tokenizer)
    if split is None:
        return None
    backend, pattern, adds_special = split
    if kwargs.get("add_special_tokens", True) and adds_special:
        return None
    parts = [p for p in pattern.split(text) if p]
    cache = _cache(tokenizer)
    if cache is not None:
        ids = cache.encode(backend, parts)
    else:
        ids = [
            i
            for enc in backend.encode_batch(parts, add_special_tokens=False)
            for i in enc.ids
        ]
    excess = len(ids) - kwargs["max_length"] if truncate else 0
    if excess > 0:  # as the tokenizer: keep the first or the last max_length
        if side == "left":
            del ids[:excess]
        else:
            del ids[kwargs["max_length"] :]
    return ids
