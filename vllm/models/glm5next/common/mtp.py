# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MixtureOfExperts
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.offloader import get_offloader
from vllm.models.glm5next.nvidia.ops.fused_eh_norm import fused_eh_norm
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from . import mtp_cpu_moe
from .model import (
    Glm5NextDecoderLayer,
    Glm5NextMLAAttention,
    Glm5NextMoE,
    _fused_shared_expert_name,
    _num_fused_shared_experts,
    _try_load_fp8_attn_proj,
    _try_load_fp8_indexer_wk,
    get_spec_layer_idx_from_weight_name,
    mtp_keeps_quantized_linears,
)

logger = init_logger(__name__)

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# KH8: the MTP-layer projections served in MXFP8 when mtp_keeps_quantized_linears(),
# grouped by the model's fused modules (fused_qkv_a_proj, the shared experts'
# gate_up_proj), whose shards must share one format. kv_b_proj (MLA absorbs it
# into BF16 matrices) and the indexer (its fused wk_weights_proj and the skinny
# GEMM of patch 0028 expect BF16) stay dequantized.
_MTP_MXFP8_LINEAR_GROUPS = (
    ("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"),
    ("self_attn.q_b_proj",),
    ("self_attn.o_proj",),
    ("mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"),
    ("mlp.shared_experts.down_proj",),
)
# Checkpoint projection -> the model's fused module that holds it.
_FUSED_PROJ = {
    "q_a_proj": "fused_qkv_a_proj",
    "kv_a_proj_with_mqa": "fused_qkv_a_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
}


def _alias_mtp_expert_quantization(quant_config, config) -> None:
    """Experiment (KH2): with the MTP layer's routed experts on the GPU
    (``VLLM_EXP_MTP_CPU_MOE=0``), serve a mixed-precision ModelOpt
    checkpoint's quantized MTP experts (Local Inference Lab's ``qad2``
    branches: W4A16 NVFP4, 3.8 GiB) in that format instead of BF16 (13.5 GiB).
    Such checkpoints list them under the multimodal name
    (``model.language_model.layers.N.mlp.experts``); the MTP model builds
    ``model.layers.N.mlp.experts``. Only the routed experts get an alias: the
    layer's other quantized weights are dequantized at load, as before, unless
    mtp_keeps_quantized_linears() (KH8): then its MXFP8 attention projections
    and shared experts get one too (_MTP_MXFP8_LINEAR_GROUPS)."""
    if mtp_cpu_moe.enabled() or not isinstance(
        quant_config, ModelOptMixedPrecisionConfig
    ):
        return
    layers = quant_config.quantized_layers
    if not layers:
        return
    first = config.num_hidden_layers
    for n in range(first, first + (config.num_nextn_predict_layers or 0)):
        target = f"model.layers.{n}.mlp.experts"
        for source in (
            f"model.language_model.layers.{n}.mlp.experts",
            f"language_model.model.layers.{n}.mlp.experts",
        ):
            if source in layers and target not in layers:
                layers[target] = layers[source]
                logger.info(
                    "MTP layer %d: routed experts on the GPU in %s (from %s)",
                    n,
                    layers[source].get("quant_algo"),
                    source,
                )
        if not mtp_keeps_quantized_linears():
            continue
        kept = []
        for group in _MTP_MXFP8_LINEAR_GROUPS:
            found = {}
            for suffix in group:
                for base in (
                    f"model.language_model.layers.{n}",
                    f"language_model.model.layers.{n}",
                ):
                    if f"{base}.{suffix}" in layers:
                        found[suffix] = layers[f"{base}.{suffix}"]
                        break
            algos = {str(info.get("quant_algo", "")).upper() for info in found.values()}
            if len(found) != len(group) or algos != {"MXFP8"}:
                continue  # all shards of a fused module in MXFP8, or none
            for suffix, info in found.items():
                layers.setdefault(f"model.layers.{n}.{suffix}", info)
                # The fused module's own name too, for a quant config without
                # the model's packed-module mapping.
                module, _, proj = suffix.rpartition(".")
                if proj in _FUSED_PROJ:
                    fused = f"model.layers.{n}.{module}.{_FUSED_PROJ[proj]}"
                    layers.setdefault(fused, info)
            kept += group
        logger.info(
            "MTP layer %d: %d projections kept in MXFP8 (KH8): %s",
            n,
            len(kept),
            ", ".join(kept) or "none",
        )


def _dequant_nvfp4(
    weight: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor
) -> torch.Tensor:
    """[N, K/2] uint8 (two E2M1 values, the low nibble first), [N, K/16] E4M3
    scales and an FP32 multiplier -> [N, K] BF16 (ModelOpt's NVFP4)."""
    table = torch.tensor(
        _E2M1 + tuple(-x for x in _E2M1), dtype=torch.float32, device=weight.device
    )
    nibbles = torch.stack((weight & 0xF, weight >> 4), dim=-1).view(
        weight.shape[0], -1
    )
    values = table[nibbles.long()]
    n, k = values.shape
    scales = scale.float() * scale_2.float()
    return (values.view(n, k // 16, 16) * scales[..., None]).view(n, k).bfloat16()


class _MtpWeightDequantizer:
    """BF16 weights from the quantized MTP-layer tensors of mixed-precision
    ModelOpt checkpoints.

    The MTP layer is built in BF16: its module names are not in a checkpoint's
    quantized-layer map, and the drafter's routed experts run on the CPU from
    BF16 weights (mtp_cpu_moe.py). Local Inference Lab's ``qad2`` branches store
    its attention, indexer and shared experts in MXFP8 (``weight`` E4M3 and
    ``weight_scale`` E8M0 per 32, uint8) and its routed experts in W4A16 NVFP4
    (``weight``: two E2M1 per uint8, ``weight_scale``: E4M3 per 16,
    ``weight_scale_2``: FP32 multiplier). The parts arrive in any order; each
    weight is passed on as BF16 once complete. An ``input_scale`` is not needed.
    BF16 checkpoints have no such tensors and pass through unchanged. The
    conversion runs on the GPU when there is one (on the CPU it took minutes
    for the 864 expert matrices of an MTP layer).
    """

    _SUFFIXES = (".weight_scale_2", ".weight_scale", ".input_scale", ".weight")

    def __init__(self) -> None:
        self._parts: dict[str, dict[str, torch.Tensor]] = {}
        self._done: set[str] = set()

    def add(
        self, name: str, tensor: torch.Tensor
    ) -> tuple[str, torch.Tensor] | bool:
        """False: not a quantized part (load as is). True: kept, nothing to
        load yet. ``(name, weight)``: a complete weight, dequantized to BF16."""
        suffix = next((s for s in self._SUFFIXES if name.endswith(s)), None)
        if suffix is None:
            return False
        if suffix == ".weight" and tensor.dtype not in (
            torch.float8_e4m3fn,
            torch.uint8,
        ):
            return False
        stem = name[: -len(suffix)]
        if stem in self._done:
            return True  # e.g. an input scale after its weight
        parts = self._parts.setdefault(stem, {})
        parts[suffix[1:]] = tensor
        weight, scale = parts.get("weight"), parts.get("weight_scale")
        if weight is None or scale is None:
            return True
        device = weight.device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        if weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.uint8:
            out = dequant_mxfp8_to_bf16(weight.to(device), scale.to(device))
        elif weight.dtype == torch.uint8 and scale.dtype == torch.float8_e4m3fn:
            if "weight_scale_2" not in parts:
                return True
            out = _dequant_nvfp4(
                weight.to(device),
                scale.to(device),
                parts["weight_scale_2"].to(device),
            )
        else:
            raise ValueError(
                f"{stem}: unsupported quantized MTP weight "
                f"(weight {weight.dtype}, weight_scale {scale.dtype})"
            )
        del self._parts[stem]
        self._done.add(stem)
        return f"{stem}.weight", out

    def finish(self) -> None:
        pending = {
            stem: sorted(parts)
            for stem, parts in self._parts.items()
            if set(parts) != {"input_scale"}
        }
        if pending:
            raise ValueError(f"Incomplete quantized MTP weights: {pending}")


class Glm5NextMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_text_config
        self.config = config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        # Reserve room for the incomplete pool tail and align the sparse MLA
        # buffer width to BLOCK_N=128.
        topk_tokens = config.index_topk
        assert topk_tokens is not None
        kpool = config.index_kpool
        assert kpool is not None
        buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
        sparse_topk_block_n = 128
        buffer_width = (
            (buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n
        ) * sparse_topk_block_n
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            buffer_width,
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.shared_head = SharedHead(
            config=config,
            prefix=prefix,
            defer_lm_head=True,
        )
        # MTP layers sit past the base model's hidden layers; parse the index
        # from the prefix (e.g. "...layers.32") so the decoder builds an MLA
        # (DSA) layer rather than KDA for the MTP path.
        layer_idx = int(prefix.rsplit(".", 1)[-1])
        self.mtp_block = Glm5NextDecoderLayer(
            vllm_config=vllm_config,
            config=config,
            layer_idx=layer_idx,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            is_mtp_layer=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        cpu_moe = self.mtp_block.mlp
        if getattr(cpu_moe, "is_cpu_moe", False):
            # Wake the CPU workers now, so they spin by the time the MoE starts.
            cpu_moe.prewake()
        # Fused: zero pos-0 embeds + enorm(embeds) + hnorm(prev) + cat -> [N, 2H].
        eh_input = fused_eh_norm(
            positions,
            inputs_embeds,
            previous_hidden_states,
            self.enorm.weight,
            self.hnorm.weight,
            self.enorm.variance_epsilon,
        )
        hidden_states = self.eh_proj(eh_input)
        # Fuse the residual add and final RMSNorm. Glm5NextMoE already performs
        # its all-reduce, so no collective is needed here. The post-norm result
        # feeds both draft logits and the next recycled hidden state.
        hidden_states, residual, _, _ = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states, _ = self.shared_head.norm(hidden_states, residual=residual)
        return hidden_states, hidden_states


class Glm5NextMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # The MTP layers are built outside make_layers, so they are routed through
        # the offloader here (as the target's embed_tokens): e.g.
        # --cpu-offload-params mtp_block.mlp.experts keeps the routed experts
        # (bf16, ~13 GiB) in pinned host memory (UVA). wrap_modules only moves a
        # module after it was built on the GPU, which needs room for the whole
        # layer, so when the offload targets the MTP block the layers are built
        # on the CPU and then placed (offloaded parameters stay in host memory).
        offloader = get_offloader()
        build_on_cpu = getattr(offloader, "uva_offloading", False) and any(
            p.startswith("mtp_block") for p in getattr(offloader, "cpu_offload_params", ())
        )
        with torch.device("cpu") if build_on_cpu else contextlib.nullcontext():
            self.layers = torch.nn.ModuleDict(
                {
                    str(idx): Glm5NextMultiTokenPredictorLayer(
                        vllm_config, f"{prefix}.layers.{idx}"
                    )
                    for idx in range(
                        self.mtp_start_layer_idx,
                        self.mtp_start_layer_idx + self.num_mtp_layers,
                    )
                }
            )
        if build_on_cpu:
            offloader.place_cpu_built_modules(
                self.layers.values(), prefix="mtp", device=current_platform.device_type
            )
        elif offloader.supports_tower_offload:
            offloader.wrap_modules(iter(self.layers.values()), prefix="mtp")
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # The drafter's own embedding table (used on PP ranks that do not hold the
        # target's): --cpu-offload-params embed_tokens.
        if offloader.supports_tower_offload:
            offloader.wrap_modules(iter([self.embed_tokens]), prefix="embed_tokens")
        # Plain list for the per-propose lookup: ModuleDict[str(...)] builds a
        # string and hashes it on every draft step.
        self._mtp_layers = list(self.layers.values())
        self._mtp_mla_attns = []
        for layer in self._mtp_layers:
            self_attn = layer.mtp_block.self_attn
            assert isinstance(self_attn, Glm5NextMLAAttention)
            self._mtp_mla_attns.append(self_attn.mla_attn)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self._fp8_heads: list | None = None

    def build_fp8_head(self) -> None:
        """Experiment (VLLM_EXP_MTP_FP8_HEAD=1): draft logits from a row-scaled FP8 copy of the shared LM head
        (mtp_fp8_head.py), half the bytes of the BF16 head per draft. Call after the head is shared."""
        from .mtp_fp8_head import Fp8Head

        copies: dict[int, Fp8Head] = {}
        self._fp8_heads = []
        for layer in self._mtp_layers:
            head = layer.shared_head.head
            if head.weight.dtype != torch.bfloat16:
                # A quantized LM head (e.g. MXFP8 in mixed-precision
                # checkpoints) already reads 8-bit weights: draft from it.
                logger.info(
                    "MTP FP8 draft head not built: the LM head is %s already",
                    head.weight.dtype,
                )
                self._fp8_heads = None
                return
            if id(head) not in copies:
                copies[id(head)] = Fp8Head(head.weight)
            self._fp8_heads.append(copies[id(head)])

    def _fp8_logits(self, step_idx: int, hidden_states: torch.Tensor) -> torch.Tensor | None:
        from .mtp_fp8_head import MAX_ROWS

        if self._fp8_heads is None or not 1 <= hidden_states.shape[0] <= MAX_ROWS:
            return None
        if self._mtp_layers[step_idx].shared_head.head.tp_size > 1:
            return None
        lp = self.logits_processor
        logits = self._fp8_heads[step_idx](hidden_states)[..., : lp.org_vocab_size]
        if lp.soft_cap is not None:
            logits = torch.tanh(logits / lp.soft_cap) * lp.soft_cap
        if lp.scale != 1.0:
            logits = logits * lp.scale
        return logits

    def set_skip_topk(self, skip: bool):
        # index_share_for_mtp_iteration: step 0 computes top-k, steps 1+ reuse.
        for mla_attn in self._mtp_mla_attns:
            mla_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor):
        """Gather the top-k index rows at ``slot_ids`` to the front of the buffer."""
        num_slots = slot_ids.numel()
        for mla_attn in self._mtp_mla_attns:
            topk_indices_buffer = mla_attn.topk_indices_buffer
            assert topk_indices_buffer is not None
            topk_indices_buffer[:num_slots] = topk_indices_buffer[slot_ids]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self._mtp_layers[current_step_idx](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self._mtp_layers[current_step_idx]
        # hidden_states is already post-final-norm (produced in the layer
        # forward and recycled as-is); apply the LM head only, without a
        # second RMSNorm.
        logits = self._fp8_logits(current_step_idx, hidden_states)
        if logits is not None:
            return logits
        return self.logits_processor(mtp_layer.shared_head.head, hidden_states)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self._mtp_layers[current_step_idx]
        # Vocab-parallel argmax for the greedy draft: per-rank head projection
        # + local argmax + a [batch, 2*tp] (value, index) reduce, instead of
        # materializing and all-gathering full [N, vocab] logits per draft
        # step. Tie-breaking matches the full argmax (shards are contiguous
        # and rank-ordered, so the lowest-rank winner is the lowest global
        # index), so greedy draft tokens are unchanged.
        logits = self._fp8_logits(current_step_idx, hidden_states)
        if logits is not None:
            return logits.max(dim=-1).indices
        return self.logits_processor.get_top_tokens(
            mtp_layer.shared_head.head, hidden_states
        )


class Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        _alias_mtp_expert_quantization(self.quant_config, self.config)
        self.model = Glm5NextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.model.layers.values(), Glm5NextMoE, "mtp_block.mlp"
        )

    def set_moe_parameters(self):
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = self.config.n_group
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            mlp = layer.mtp_block.mlp
            if isinstance(mlp, Glm5NextMoE):
                example_moe = mlp
                self.moe_mlp_layers.append(mlp)
                self.moe_layers.append(mlp.experts)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Greedy-draft path used when use_local_argmax_reduction is enabled:
        # vocab-parallel argmax, no full-vocab logits.
        return self.model.get_top_tokens(hidden_states, spec_step_idx)

    def _load_cpu_moe_expert(
        self, spec_layer: int, name: str, loaded_weight: torch.Tensor
    ) -> bool:
        """Routed-expert weights of an MTP layer whose experts run on the CPU go to its
        engine (mtp_cpu_moe.py); returns whether ``name`` was one."""
        layer = self.model.layers[str(spec_layer)]
        moe = layer.mtp_block.mlp
        if not getattr(moe, "is_cpu_moe", False) or ".mlp.experts." not in name:
            return False
        stem = name.rsplit(".mlp.experts.", 1)[1]  # "<e>.<gate|up|down>_proj.weight"
        expert, proj = stem.split(".")[:2]
        which = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}[proj]
        moe.load_expert_weight(int(expert), which, loaded_weight)
        return True

    def _keeps_quantized_experts(self, spec_layer: int, name: str) -> bool:
        """A routed-expert tensor of an MTP layer whose experts run on the GPU
        in a quantized format (``_alias_mtp_expert_quantization``): loaded as
        stored rather than dequantized to BF16."""
        if ".mlp.experts." not in name:
            return False
        cache = self.__dict__.setdefault("_quantized_experts", {})
        if spec_layer not in cache:
            # FusedMoEFactory wraps the module that holds the weights, so look
            # them up by name (a CPU-MoE layer has none).
            moe = self.model.layers[str(spec_layer)].mtp_block.mlp
            w13 = next(
                (p for n, p in moe.named_parameters() if n.endswith("w13_weight")),
                None,
            )
            cache[spec_layer] = w13 is not None and not w13.dtype.is_floating_point
        return cache[spec_layer]

    def _keeps_quantized_linear(self, name: str, params_dict: dict) -> bool:
        """A tensor of an MTP-layer projection that the model built quantized
        (KH8: ModelOpt's MXFP8 linear method, an FP8 ``weight`` with a
        ``weight_scale``): loaded as stored rather than dequantized to BF16."""
        if ".mlp.experts." in name or not mtp_keeps_quantized_linears():
            return False
        stem, _, leaf = name.rpartition(".")
        if leaf not in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            return False
        module, _, proj = stem.rpartition(".")
        held = f"{module}.{_FUSED_PROJ.get(proj, proj)}"
        weight = params_dict.get(f"{held}.weight")
        return (
            weight is not None
            and weight.dtype == torch.float8_e4m3fn
            and f"{held}.weight_scale" in params_dict
        )

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        num_fused_shared = _num_fused_shared_experts(
            self.config.n_shared_experts, self.is_fused_shared_expert_enabled
        )
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + num_fused_shared,
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        _pending_wk_fp8: dict = {}
        dequantizer = _MtpWeightDequantizer()
        kept_quantized = 0  # KH8: projection weights loaded in MXFP8
        # GLM-5.3-Flash NoPE checkpoints omit the RoPE rows from
        # ``kv_a_proj_with_mqa``; the FP8-to-BF16 path pads them for the model.
        kv_a_pad_size = 0
        if self.config.mla_use_nope and self.config.qk_rope_head_dim > 0:
            kv_a_pad_size = self.config.qk_rope_head_dim
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # Multimodal (Glm5NextForConditionalGeneration) checkpoints prefix
            # the text-tower weights with "model.language_model."; the MTP head
            # is built as a text-only model (model.layers.*), so strip the
            # prefix to match.
            if name.startswith("model.language_model."):
                name = name.replace("model.language_model.", "model.", 1)
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                # The drafter's input embedding is normally the target's, shared
                # after loading (maybe_share_target_embed). With pipeline
                # parallelism the drafter runs on the last stage, which has no
                # target embedding, and the sharing step returns early: load the
                # drafter's own copy from the checkpoint instead of leaving it
                # uninitialized.
                if (
                    name == "model.embed_tokens.weight"
                    and get_pp_group().world_size > 1
                    and name in params_dict
                ):
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)
            keeps_linear = self._keeps_quantized_linear(name, params_dict)
            if keeps_linear and name.endswith(".weight"):
                kept_quantized += 1
            if not (keeps_linear or self._keeps_quantized_experts(spec_layer, name)):
                dequantized = dequantizer.add(name, loaded_weight)
                if dequantized is True:
                    continue
                if dequantized is not False:
                    name, loaded_weight = dequantized
            if self._load_cpu_moe_expert(spec_layer, name, loaded_weight):
                loaded_params.add(name)
                continue
            if self.is_fused_shared_expert_enabled:
                name = _fused_shared_expert_name(name, self.config.n_routed_experts)

            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
            ):
                continue

            # FP8 checkpoint: dequantize the BF16-kept MLA projections
            # (q_a_proj / kv_a_proj_with_mqa / o_proj) to BF16, mirroring the
            # target model. The model holds fused_qkv_a_proj / o_proj in BF16,
            # so the checkpoint's block-FP8 weight + weight_scale_inv for these
            # has no param home and would KeyError without this dequant.
            if _try_load_fp8_attn_proj(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                kv_a_pad_size,
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping  # type: ignore[assignment]
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)  # type: ignore[assignment]
                    if name is None:
                        continue
                    if (
                        spec_layer != self.model.mtp_start_layer_idx
                        and ".layers" not in name
                    ):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        dequantizer.finish()
        if mtp_keeps_quantized_linears():
            logger.info(
                "MTP: %d projection weights loaded quantized (KH8), the others "
                "dequantized to BF16",
                kept_quantized,
            )
        for layer in self.model.layers.values():
            if getattr(layer.mtp_block.mlp, "is_cpu_moe", False):
                layer.mtp_block.mlp.finish_loading()

        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"MTP speculative decoding layer {layer_idx} weights "
                    f"missing from checkpoint."
                )
        return loaded_params
