# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)


class MTPSpeculator(TargetDependentARSpeculator):
    share_mtp_topk_indices: bool = False

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        if os.environ.get("VLLM_EXP_MTP_FP8_HEAD", "0") == "1":
            # Experiment: draft logits from an FP8 copy of the shared LM head
            # (GLM-5.3 builds it in its inner model, Qwen3.5 in the outer one).
            if hasattr(draft_model.model, "build_fp8_head"):
                draft_model.model.build_fp8_head()
            elif hasattr(draft_model, "build_fp8_head"):
                draft_model.build_fp8_head()
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        # Detect index_share_for_mtp_iteration. When True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        self.share_mtp_topk_indices = (
            self.vllm_config.parallel_config.prefill_context_parallel_size == 1
            and getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        return draft_model

    def _cpu_moe_layers(self) -> list[nn.Module]:
        # MTP layers whose routed experts run on the CPU (VLLM_EXP_MTP_CPU_MOE=1) need
        # to know which rows of a draft forward are sampled.
        layers = getattr(self, "_cpu_moe_cache", None)
        if layers is None:
            layers = [m for m in self.model.modules() if hasattr(m, "set_sample_rows")]
            self._cpu_moe_cache = layers
        return layers

    def on_prefill_begin(self, num_reqs: int) -> None:
        # CPU MoE, draft step 0: only the requests' last token indices are sampled;
        # padded request slots have seq_len 0. Set in this hook because it fires at
        # graph capture (num_reqs = max_num_reqs) and before every replay alike.
        for layer in self._cpu_moe_layers():
            layer.set_sample_rows(
                self.last_token_indices, num_reqs, self.input_buffers.seq_lens
            )
        # Step 0 computes its own top-k. Unconditional, so a step that died
        # midway cannot leave reuse mode on.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        # Step 0 (prefill) wrote topk indices for every query token in the
        # multi-token batch. Compact them down to each request's last token so
        # steps 1+ can reuse them from the shared buffer.
        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        # CPU MoE, draft steps 1+ (fused or not): one row per request.
        for layer in self._cpu_moe_layers():
            layer.set_sample_rows(None, num_reqs, self.input_buffers.seq_lens)
        # Switch to reuse mode so draft steps 1+ skip the indexer op and read
        # the indices that step 0 wrote into the shared buffer.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)
