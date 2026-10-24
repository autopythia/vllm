# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The drafts `DraftTokensHandler` hands the scheduler for grammar validation.

Under pipeline parallelism with async scheduling the steps of several decode
groups interleave, and the engine takes a deferred batch's drafts after the
other groups' steps have run. With keep_steps > 1 the handler returns the
drafts of the last keep_steps steps, the newest for each request; with only
the last step's, the deferred batch's structured-output requests would keep
their -1 placeholders and every draft would be rejected.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires CUDA or ROCm"
)


def _step(handler, req_ids, rows, structured=True):
    batch = SimpleNamespace(
        req_ids=list(req_ids), has_structured_output_reqs=structured
    )
    drafts = torch.tensor(rows, dtype=torch.int32, device="cuda")
    handler.set_draft_tokens(batch, drafts)


def _drafts(handler):
    out = handler.get_draft_tokens()
    return dict(zip(out.req_ids, out.draft_token_ids))


def test_only_the_last_step_by_default():
    handler = DraftTokensHandler(torch.device("cuda"))
    _step(handler, ["a1", "a2"], [[1, 2], [3, 4]])
    _step(handler, ["b1"], [[5, 6]])
    assert _drafts(handler) == {"b1": [5, 6]}


def test_the_other_decode_groups_drafts_are_kept():
    # PP=2: group A ran at step s and group B at s + 1; the engine validates
    # A's deferred batch after B's step has replaced the last drafts.
    handler = DraftTokensHandler(torch.device("cuda"), keep_steps=3)
    _step(handler, ["a1", "a2"], [[1, 2], [3, 4]])
    _step(handler, ["b1"], [[5, 6]])
    assert _drafts(handler) == {"a1": [1, 2], "a2": [3, 4], "b1": [5, 6]}


def test_newer_drafts_replace_older_and_old_steps_drop_out():
    handler = DraftTokensHandler(torch.device("cuda"), keep_steps=3)
    _step(handler, ["a1", "a2"], [[1, 2], [3, 4]])
    _step(handler, ["b1", "b2"], [[5, 6], [7, 8]])
    _step(handler, ["a1"], [[11, 12]])
    _step(handler, ["b1"], [[15, 16]])
    # a2's only step is now four steps old.
    assert _drafts(handler) == {"b1": [15, 16], "b2": [7, 8], "a1": [11, 12]}


def test_batches_without_structured_outputs():
    handler = DraftTokensHandler(torch.device("cuda"), keep_steps=3)
    _step(handler, ["a1"], [[1, 2]])
    _step(handler, ["c1"], [[9, 9]], structured=False)
    assert _drafts(handler) == {"a1": [1, 2], "c1": [-1, -1]}


def test_returned_lists_are_fresh():
    # The scheduler trims the lists in place (update_draft_token_ids_in_output).
    handler = DraftTokensHandler(torch.device("cuda"), keep_steps=3)
    _step(handler, ["a1"], [[1, 2, 3]])
    _drafts(handler)["a1"].clear()
    assert _drafts(handler) == {"a1": [1, 2, 3]}


def test_copies_complete_after_many_steps():
    handler = DraftTokensHandler(torch.device("cuda"), keep_steps=3)
    for k in range(50):
        _step(handler, [f"r{k % 2}"], [[k, k + 1, k + 2]])
    assert _drafts(handler) == {"r0": [48, 49, 50], "r1": [49, 50, 51]}
