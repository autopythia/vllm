# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size
        # Only XPU can disable PP decode microbatching via VLLM_XPU_PP_MICROBATCH.
        self.decode_stagger = self.pp_size
        if current_platform.is_xpu() and not envs.VLLM_XPU_PP_MICROBATCH:
            self.decode_stagger = 1
        # Experiment: balance the decode groups of pipeline parallelism (on unless
        # VLLM_EXP_PP_DECODE_BALANCE=0). See _balance_decode_groups.
        self.balance_decode_groups = (
            self.use_v2_model_runner
            and self.decode_stagger > 1
            and os.environ.get("VLLM_EXP_PP_DECODE_BALANCE", "1") != "0"
        )
        if self.balance_decode_groups:
            logger.info(
                "Balancing decode requests between the %d pipeline decode groups "
                "(VLLM_EXP_PP_DECODE_BALANCE=0 turns it off).",
                self.decode_stagger,
            )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = (
                    self.current_step + self.decode_stagger
                )
        if self.balance_decode_groups:
            self._balance_decode_groups(scheduler_output)

    def _balance_decode_groups(self, scheduler_output: SchedulerOutput) -> None:
        """Even out the pipeline's decode groups (experiment).

        With the cadence above, a request decodes only in steps of one residue
        modulo decode_stagger, fixed by the step in which its prefill ended.
        Requests whose prefills end in the same step, e.g. requests sent
        together, share a residue and leave the other steps empty, so the
        pipeline stages take turns instead of working on several groups at once.
        After each step, the requests just scheduled (due again in
        decode_stagger steps) move to the least loaded residue while their
        residue has at least two more requests due; a moved request skips the
        steps up to its new residue, once. Newest decoders move first.
        """
        stagger = self.decode_stagger
        step = self.current_step
        # due[d]: running requests next due at step + 1 + d.
        due = [0] * stagger
        movable: list[Request] = []
        for request in self.running:
            d = request.next_decode_eligible_step - step - 1
            if 0 <= d < stagger:
                due[d] += 1
                if (
                    d == stagger - 1
                    and request.request_id in scheduler_output.num_scheduled_tokens
                ):
                    movable.append(request)
        movable.sort(key=lambda r: (r.num_output_tokens, r.request_id))
        last = stagger - 1
        for request in movable:
            target = min(range(last), key=lambda i: (due[i], i))
            if due[last] - due[target] < 2:
                break
            request.next_decode_eligible_step = step + 1 + target + stagger
            due[last] -= 1
            due[target] += 1

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
