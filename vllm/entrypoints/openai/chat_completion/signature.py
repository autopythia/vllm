# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-exact replay of assistant turns ("signatures").

With ``return_signature: true``, every assistant message in a chat completion
response carries ``signature``: a stateless, compact encoding of the exact token
transcript up to the end of that message -- the prompt token ids the model saw
plus the ids it sampled, minus a final stop token (sampled, never fed back).

When a later request contains that assistant message with its ``signature``,
the prompt is the transcript verbatim, followed by the chat template's rendering
of only the messages after it. The model then sees exactly what it produced
(its reasoning, whitespace and tool-call formatting) instead of the template's
re-rendering of the parsed message, and the prompt strictly extends the earlier
one, so prefix caching can reuse it.

Format: ``vtr1.`` + base64url(zlib(magic, model fingerprint, inputs digest,
token count, uint32 little-endian token ids)). The inputs digest binds the
transcript to the request's non-assistant messages and tools: if a client edits
those, replay is refused instead of silently using the old transcript. The
fingerprint covers the model, its vocabulary and the chat template the server
renders with (the checkpoint's own when none is configured).
"""

import base64
import hashlib
import json
import struct
import sys
import zlib
from array import array
from collections.abc import Sequence
from typing import Any

PREFIX = "vtr1."
_MAGIC = b"VTR1"
_HEADER = struct.Struct("<4s8s16sI")
# zlib level 1: at 870K tokens 30 ms and 1.53 MB, against 164 ms and 1.29 MB at
# level 6. Decoding does not depend on the level.
_ZLIB_LEVEL = 1


class SignatureError(ValueError):
    pass


def fingerprint(model: str, vocab_size: int, chat_template: str | None) -> bytes:
    """8 bytes identifying the model, vocabulary and chat template."""
    h = hashlib.sha256()
    for part in (model, str(vocab_size), chat_template or ""):
        h.update(part.encode())
        h.update(b"\0")
    return h.digest()[:8]


def _text(content: Any) -> Any:
    """Content as the template sees it: text parts joined; other parts as-is."""
    if isinstance(content, list):
        if all(isinstance(p, dict) and p.get("type") == "text" for p in content):
            return "".join(p.get("text", "") for p in content)
        return content
    return content


def inputs_digest(messages: Sequence[Any], tools: Any) -> bytes:
    """16 bytes over the non-assistant messages and the tool definitions."""
    items = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant":
            continue
        items.append(
            [m.get("role"), _text(m.get("content")), m.get("tool_call_id"), m.get("name")]
        )
    blob = json.dumps([items, tools], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).digest()[:16]


def transcript_of(
    prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    finish_reason: str | None,
    stop_reason: int | str | None,
) -> list[int]:
    """What the next turn should start from: prompt + output, minus a final
    stop token. A stop token was sampled but never fed back; the chat template
    renders the next turn's header after it."""
    out = list(output_token_ids)
    if out and finish_reason == "stop" and not isinstance(stop_reason, str):
        out = out[:-1]
    return [*prompt_token_ids, *out]


def encode(token_ids: Sequence[int], fp: bytes, digest: bytes) -> str:
    ids = array("I", token_ids)
    if sys.byteorder != "little":
        ids.byteswap()
    raw = _HEADER.pack(_MAGIC, fp, digest, len(ids)) + ids.tobytes()
    compressed = zlib.compress(raw, _ZLIB_LEVEL)
    return PREFIX + base64.urlsafe_b64encode(compressed).decode().rstrip("=")


def decode(signature: str) -> tuple[bytes, bytes, list[int]]:
    """(fingerprint, inputs digest, token ids); raises SignatureError."""
    if not isinstance(signature, str) or not signature.startswith(PREFIX):
        raise SignatureError("not a vtr1 signature")
    body = signature[len(PREFIX):]
    try:
        raw = zlib.decompress(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except Exception as e:
        raise SignatureError(f"undecodable signature: {e}") from None
    if len(raw) < _HEADER.size:
        raise SignatureError("truncated signature")
    magic, fp, digest, n = _HEADER.unpack_from(raw)
    if magic != _MAGIC or len(raw) != _HEADER.size + 4 * n:
        raise SignatureError("corrupt signature")
    ids = array("I")
    ids.frombytes(raw[_HEADER.size:])
    if sys.byteorder != "little":
        ids.byteswap()
    return fp, digest, ids.tolist()


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def stand_in_history(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """The messages before a signed one with their contents replaced by stand-ins.

    Replay only needs to know where the template's rendering of the signed
    message ends; rendering stand-ins finds that without tokenizing the history.
    Roles, order, names and ids are kept (templates use them across messages,
    e.g. to find the last user message or to pair tool results with calls), as
    are the tool calls' names; contents, reasoning and tool-call arguments
    become short placeholders.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        s: dict[str, Any] = {
            k: v for k in ("role", "name", "tool_call_id", "id")
            if (v := _get(m, k)) is not None
        }
        s["content"] = "x"
        for k in ("reasoning", "reasoning_content"):
            if _get(m, k) is not None:
                s[k] = "x"
        if tool_calls := _get(m, "tool_calls"):
            s["tool_calls"] = [
                {
                    "id": _get(tc, "id"),
                    "type": _get(tc, "type") or "function",
                    "function": {
                        "name": _get(_get(tc, "function") or {}, "name") or "",
                        "arguments": "{}",
                    },
                }
                for tc in tool_calls
            ]
        out.append(s)
    return out


def last_signed_assistant(messages: Sequence[Any]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("signature"):
            return i
    return None


def tools_of(request: Any) -> Any:
    tools = getattr(request, "tools", None)
    return [t.model_dump() for t in tools] if tools else None
