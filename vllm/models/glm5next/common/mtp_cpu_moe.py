# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts of the GLM-5.3 MTP layer on the CPU (experiment: VLLM_EXP_MTP_CPU_MOE=1).

The MTP layer's 288 BF16 experts (14.5 GB) do not fit on the GPU next to the model, and reading them over PCIe for
every draft (UVA offload) halves decode speed. But the layer's MoE output at a position only feeds that position's
draft logits (one decoder layer: its KV comes from the attention input, before the MoE), so a draft forward needs the
routed experts for the sampled rows only: one per request. This module keeps the router, the shared expert and
everything else on the GPU, and computes the routed experts of the sampled rows on the CPU (mtp_cpu_moe.c, AVX512-BF16,
one pinned worker thread per core, each owning a NUMA-local slice of every expert).

Hand-off, inside the drafter's CUDA graphs: by default a doorbell (mtp_cpu_moe_kernels.cu): one kernel copies the
staged rows into mapped host memory and bumps a flag, a CPU watcher thread sees it and runs the workers, and a second
kernel spins until the CPU's flag matches, then copies the result back. The shared expert runs on the GPU in between.
The fallback (VLLM_EXP_MTP_CPU_MOE_DOORBELL=0) uses CUDA host functions, which cost ~0.45 ms more per hand-off. A
third flag, rung at the start of the MTP layer, wakes the sleeping workers ~0.5 ms before their call.

The speculator says which rows are sampled (``set_sample_rows``, from its on_prefill_begin and
on_multi_step_decode_begin hooks): in draft step 0 the requests' last token indices, in later draft steps every row.
Rows of padded request slots (``seq_len`` 0) are skipped. Other rows get no routed-expert output; their hidden states
are discarded by the speculator.

Two host settings matter (measured on a 2-socket Xeon 6767P): keep the server's own threads off the workers' cores
and their SMT siblings (mtp_cpu_plan.py prints a taskset list; vLLM binds GPU workers to their GPU's NUMA node, which
otherwise overlaps the workers), and pin the expert weights (default) so that automatic NUMA balancing does not keep
unmapping them: its hinting faults stall a few workers at a time, and every call waits for the slowest.

Environment: VLLM_EXP_MTP_CPU_MOE_CPUS (CPU list, e.g. "0-11,32-43"; default: the first
VLLM_EXP_MTP_CPU_MOE_PER_NODE (12) physical cores of every NUMA node, capped below the container's CPU quota),
VLLM_EXP_MTP_CPU_MOE_PREFETCH (1024 bytes), VLLM_EXP_MTP_CPU_MOE_SPIN_US (3000: idle workers spin this long before
they sleep), VLLM_EXP_MTP_CPU_MOE_PREWAKE (1: wake the workers at the start of the MTP layer),
VLLM_EXP_MTP_CPU_MOE_PIN (1: pin the expert weights, which exempts them from automatic NUMA balancing),
VLLM_EXP_MTP_CPU_MOE_DOORBELL (1: the doorbell hand-off, built with nvcc; 0: CUDA host functions),
VLLM_EXP_MTP_CPU_MOE_WATCHER_CPU (-1: the watcher thread unpinned),
VLLM_EXP_MTP_CPU_MOE_REPORT_S (0: a timing report to stderr every so many seconds), VLLM_EXP_MTP_CPU_MOE_MODE_FILE
(a file holding 0 or 1, read at most every 0.2 s; 1 skips the CPU compute and returns zeros, for measurements).
Options measured on the Xeon 6767P (exp/cpu_moe_bench.c): VLLM_EXP_MTP_CPU_MOE_PF_HINT (0: prefetch into L1, 1: L2,
2: L3, 3: non-temporal; L2 at 4096 or L3 at 2048 bytes was ~14% faster than L1 at 1024 in the benchmark),
VLLM_EXP_MTP_CPU_MOE_WBITS (16: BF16 weights; 8: INT8 with one FP32 scale per row, quantized at load, half the bytes
and about half the time; changes the draft's numerics, not the target's).
"""

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

_SRC = Path(__file__).with_name("mtp_cpu_moe.c")
_CU_SRC = Path(__file__).with_name("mtp_cpu_moe_kernels.cu")
_LIBS: dict[tuple[int, int], ctypes.CDLL] = {}


def enabled() -> bool:
    return os.environ.get("VLLM_EXP_MTP_CPU_MOE", "0") == "1"


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _parse_list(s: str) -> list[int]:
    out: list[int] = []
    for tok in s.strip().split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def _cpu_quota() -> float | None:
    """CPUs the container may use on average (cgroup CFS quota), or None if unlimited."""
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if q == "max" else int(q) / int(p)
    except (OSError, ValueError):
        pass
    try:
        base = Path("/sys/fs/cgroup/cpu,cpuacct")
        q = int((base / "cpu.cfs_quota_us").read_text())
        p = int((base / "cpu.cfs_period_us").read_text())
        return None if q <= 0 else q / p
    except (OSError, ValueError):
        return None


def default_cpus(per_node: int) -> list[int]:
    """The first ``per_node`` physical cores (one thread each) of every NUMA node, capped below the CPU quota."""
    nodes = sorted(int(p.name[4:]) for p in Path("/sys/devices/system/node").glob("node[0-9]*"))
    per_node_cores: list[list[int]] = []
    for n in nodes:
        cpus = _parse_list(Path(f"/sys/devices/system/node/node{n}/cpulist").read_text())
        cores: list[int] = []
        seen: set[int] = set()
        for c in cpus:
            if c in seen:
                continue
            sib = Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
            seen.update(_parse_list(sib.read_text()) if sib.exists() else [c])
            cores.append(c)
        if cores:
            per_node_cores.append(cores)
    quota = _cpu_quota()
    if quota is not None and per_node_cores:
        # Threads beyond the quota are throttled until the next period; leave room for the server's own threads.
        per_node = max(1, min(per_node, int(0.8 * quota) // len(per_node_cores)))
    return [c for cores in per_node_cores for c in cores[:per_node]]


def _cache_dir() -> Path:
    root = os.environ.get("VLLM_CACHE_ROOT") or str(Path.home() / ".cache" / "vllm")
    return Path(root) / "mtp_cpu_moe"


def build_cubin(arch: str) -> Path:
    """The doorbell kernels (mtp_cpu_moe_kernels.cu) as a cubin for ``arch`` (e.g. "sm_120"), built with nvcc."""
    nvcc = next((p for p in (os.path.join(os.environ.get("CUDA_HOME", "/nonexistent"), "bin", "nvcc"),
                             shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc") if p and os.path.exists(p)), None)
    if nvcc is None:
        raise RuntimeError("the MTP CPU MoE doorbell needs nvcc (set CUDA_HOME), or VLLM_EXP_MTP_CPU_MOE_DOORBELL=0")
    flags = ["-O3", "-cubin", f"-arch={arch}"]
    tag = hashlib.sha256(_CU_SRC.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    out = _cache_dir() / f"mtp_cpu_moe_kernels_{arch}_{tag}.cubin"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".cubin")
        os.close(fd)
        r = subprocess.run([nvcc, *flags, "-o", tmp, str(_CU_SRC)], capture_output=True, text=True)
        if r.returncode:
            os.unlink(tmp)
            raise RuntimeError(f"building {_CU_SRC.name} failed:\n{r.stderr[-4000:]}")
        os.replace(tmp, out)
    return out


def _load_library(hidden: int, inter: int) -> ctypes.CDLL:
    key = (hidden, inter)
    if key in _LIBS:
        return _LIBS[key]
    if "avx512_bf16" not in Path("/proc/cpuinfo").read_text():
        raise RuntimeError("VLLM_EXP_MTP_CPU_MOE needs a CPU with AVX512-BF16 (e.g. Sapphire Rapids or newer)")
    flags = ["-O3", "-march=native", "-mavx512bf16", "-fPIC", "-shared", "-pthread",
             f"-DCMOE_H={hidden}", f"-DCMOE_I={inter}"]
    tag = hashlib.sha256(_SRC.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    cache = _cache_dir()
    so = cache / f"mtp_cpu_moe_{hidden}x{inter}_{tag}.so"
    if not so.exists():
        cache.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache, suffix=".so")
        os.close(fd)
        cc = os.environ.get("CC", "gcc")
        r = subprocess.run([cc, *flags, "-o", tmp, str(_SRC), "-lm", "-ldl"], capture_output=True, text=True)
        if r.returncode:
            os.unlink(tmp)
            raise RuntimeError(f"building {_SRC.name} failed:\n{r.stderr[-4000:]}")
        os.replace(tmp, so)
    lib = ctypes.CDLL(str(so))
    vp, sz, ull = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong
    lib.cmoe_last_error.restype = ctypes.c_char_p
    lib.cmoe_create.restype = vp
    lib.cmoe_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
    lib.cmoe_set_prefetch.argtypes = [vp, ctypes.c_int, ctypes.c_int]
    lib.cmoe_destroy.argtypes = [vp]
    lib.cmoe_register_host_buffers.argtypes = [vp]
    lib.cmoe_set_weight.argtypes = [vp, ctypes.c_int, ctypes.c_int, vp]
    lib.cmoe_num_missing.argtypes = [vp]
    lib.cmoe_run.argtypes = [vp]
    lib.cmoe_enqueue.argtypes = [vp, ull, ull, sz, ull, sz]
    lib.cmoe_enqueue_prewake.argtypes = [vp, ull]
    for fn in ("cmoe_in_buffer", "cmoe_out_buffer"):
        getattr(lib, fn).restype = vp
        getattr(lib, fn).argtypes = [vp]
    for fn in ("cmoe_in_bytes", "cmoe_out_bytes"):
        getattr(lib, fn).restype = sz
        getattr(lib, fn).argtypes = [vp]
    lib.cmoe_offset.restype = sz
    lib.cmoe_offset.argtypes = [vp, ctypes.c_int]
    lib.cmoe_stats.argtypes = [vp, ctypes.POINTER(ctypes.c_double)]
    lib.cmoe_configure.argtypes = [vp, ctypes.c_double, ctypes.c_char_p]
    lib.cmoe_note_eager.argtypes = [vp]
    lib.cmoe_set_pin_weights.argtypes = [vp, ctypes.c_int]
    lib.cmoe_enable_doorbell.argtypes = [vp, ctypes.c_char_p, ctypes.c_int, ctypes.c_double]
    lib.cmoe_ring.argtypes = [vp, ull, ull, sz, ull]
    lib.cmoe_wait.argtypes = [vp, ull, ull, sz, ull]
    lib.cmoe_prewake_db.argtypes = [vp, ull, ull]
    lib.cmoe_doorbell_errors.argtypes = [vp]
    lib.cmoe_reset_stats.argtypes = [vp]
    _LIBS[key] = lib
    return lib


class CpuMoEEngine:
    """ctypes wrapper of one mtp_cpu_moe.c engine: the expert weights in NUMA-local slices and the worker threads."""

    def __init__(self, num_experts: int, hidden: int, inter: int, top_k: int, max_rows: int,
                 swiglu_limit: float | None, cpus: list[int], prefetch: int = 1024, spin_us: int = 3000,
                 wbits: int = 16, prefetch_hint: int = 0):
        self.lib = _load_library(hidden, inter)
        self.num_experts, self.hidden, self.inter = num_experts, hidden, inter
        self.top_k, self.max_rows, self.cpus = top_k, max_rows, list(cpus)
        arr = (ctypes.c_int * len(cpus))(*cpus)
        t0 = time.monotonic()
        if wbits not in (16, 8):
            raise ValueError(f"mtp_cpu_moe: weight bits {wbits} (16 or 8)")
        self.wbits = wbits
        self.h = self.lib.cmoe_create(num_experts, top_k, max_rows, arr, len(cpus), prefetch, spin_us,
                                      float(swiglu_limit or 0.0), wbits)
        if not self.h:
            raise RuntimeError(f"mtp_cpu_moe: {self.lib.cmoe_last_error().decode()}")
        self.lib.cmoe_set_prefetch(self.h, prefetch, prefetch_hint)
        self.create_s = time.monotonic() - t0
        self.in_bytes = self.lib.cmoe_in_bytes(self.h)
        self.out_bytes = self.lib.cmoe_out_bytes(self.h)
        self.offsets = {k: self.lib.cmoe_offset(self.h, i) for i, k in enumerate(("valid", "rows", "ids", "w", "x"))}
        self._registered = False

    def register_host_buffers(self, pin_weights: bool = True) -> float:
        """Page-lock the staging buffers and, with pin_weights, the expert weights (see the C side); returns seconds."""
        t0 = time.monotonic()
        if not self._registered:
            self.lib.cmoe_set_pin_weights(self.h, int(pin_weights))
            if self.lib.cmoe_register_host_buffers(self.h):
                raise RuntimeError(f"mtp_cpu_moe: {self.lib.cmoe_last_error().decode()}")
            self._registered = True
        return time.monotonic() - t0

    def set_weight(self, expert: int, which: int, w: torch.Tensor) -> None:
        """which: 0 gate [I, H], 1 up [I, H], 2 down [H, I]; BF16."""
        shape = (self.inter, self.hidden) if which < 2 else (self.hidden, self.inter)
        if tuple(w.shape) != shape or w.dtype != torch.bfloat16:
            raise ValueError(f"expert {expert} weight {which}: {tuple(w.shape)} {w.dtype}, expected {shape} bf16")
        w = w.detach()
        if w.device.type != "cpu":
            w = w.cpu()
        w = w.contiguous()
        if self.lib.cmoe_set_weight(self.h, expert, which, w.data_ptr()):
            raise RuntimeError(f"mtp_cpu_moe: {self.lib.cmoe_last_error().decode()}")

    def num_missing(self) -> int:
        return self.lib.cmoe_num_missing(self.h)

    def host_input(self) -> torch.Tensor:
        """The input buffer as a uint8 CPU tensor (tests)."""
        buf = (ctypes.c_uint8 * self.in_bytes).from_address(self.lib.cmoe_in_buffer(self.h))
        return torch.frombuffer(buf, dtype=torch.uint8)

    def host_output(self) -> torch.Tensor:
        buf = (ctypes.c_uint8 * self.out_bytes).from_address(self.lib.cmoe_out_buffer(self.h))
        return torch.frombuffer(buf, dtype=torch.uint8).view(torch.float32).view(self.max_rows, self.hidden)

    def run(self) -> None:
        """Compute from the host input buffer, synchronously (tests)."""
        self.lib.cmoe_run(self.h)

    def enqueue(self, stream: int, dev_in: int, in_bytes: int, dev_out: int, out_bytes: int) -> None:
        if self.lib.cmoe_enqueue(self.h, stream, dev_in, in_bytes, dev_out, out_bytes):
            raise RuntimeError(f"mtp_cpu_moe: {self.lib.cmoe_last_error().decode()}")

    def enqueue_prewake(self, stream: int) -> None:
        if self.lib.cmoe_enqueue_prewake(self.h, stream):
            raise RuntimeError(f"mtp_cpu_moe: {self.lib.cmoe_last_error().decode()}")

    # -- doorbell hand-off (flags in mapped host memory, a CPU watcher thread; see mtp_cpu_moe_kernels.cu) --
    def enable_doorbell(self, cubin: Path, watcher_cpu: int = -1, timeout_s: float = 5.0) -> None:
        if self.lib.cmoe_enable_doorbell(self.h, str(cubin).encode(), watcher_cpu, timeout_s):
            raise RuntimeError(f"mtp_cpu_moe: {self.lib.cmoe_last_error().decode()}")

    def ring(self, stream: int, dev_in: int, in_bytes: int, counters: int) -> None:
        if self.lib.cmoe_ring(self.h, stream, dev_in, in_bytes, counters):
            raise RuntimeError(f"mtp_cpu_moe: {self.lib.cmoe_last_error().decode()}")

    def wait(self, stream: int, dev_out: int, out_bytes: int, counters: int) -> None:
        if self.lib.cmoe_wait(self.h, stream, dev_out, out_bytes, counters):
            raise RuntimeError(f"mtp_cpu_moe: {self.lib.cmoe_last_error().decode()}")

    def prewake_db(self, stream: int, counters: int) -> None:
        if self.lib.cmoe_prewake_db(self.h, stream, counters):
            raise RuntimeError(f"mtp_cpu_moe: {self.lib.cmoe_last_error().decode()}")

    def doorbell_errors(self) -> int:
        return self.lib.cmoe_doorbell_errors(self.h)

    def stats(self) -> dict[str, float]:
        out = (ctypes.c_double * 7)()
        self.lib.cmoe_stats(self.h, out)
        n = max(out[0], 1.0)
        return dict(calls=out[0], rows_per_call=out[1] / n, experts_per_call=out[2] / n, mean_ms=out[3] / n,
                    max_ms=out[4], last_ms=out[5], threads=out[6])

    def reset_stats(self) -> None:
        self.lib.cmoe_reset_stats(self.h)

    def configure(self, report_s: float = 0.0, mode_file: str = "") -> None:
        """Experiment instrumentation: a timing report to stderr every report_s seconds; a file whose content (0/1)
        switches the CPU compute on or off at run time (1: zero output)."""
        self.lib.cmoe_configure(self.h, float(report_s), mode_file.encode())

    def note_eager(self) -> None:
        self.lib.cmoe_note_eager(self.h)

    def close(self) -> None:
        if getattr(self, "h", None):
            self.lib.cmoe_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class StagedRows:
    """Device staging buffer with the input layout of the engine, and typed views into it."""

    def __init__(self, engine: CpuMoEEngine, device: torch.device | str):
        e = engine
        self.buf = torch.zeros(e.in_bytes, dtype=torch.uint8, device=device)
        mr, k, hd = e.max_rows, e.top_k, e.hidden

        def view(off: int, n: int, dtype: torch.dtype) -> torch.Tensor:
            nbytes = n * torch.empty((), dtype=dtype).element_size()
            return self.buf[off:off + nbytes].view(dtype)

        self.num_rows = view(0, 1, torch.int32)
        self.valid = view(e.offsets["valid"], mr, torch.int32)
        self.rows = view(e.offsets["rows"], mr, torch.int32)
        self.ids = view(e.offsets["ids"], mr * k, torch.int32).view(mr, k)
        self.w = view(e.offsets["w"], mr * k, torch.float32).view(mr, k)
        self.x = view(e.offsets["x"], mr * hd, torch.bfloat16).view(mr, hd)
        self.out = torch.zeros(mr, hd, dtype=torch.float32, device=device)


class MtpCpuMoE(nn.Module):
    """Drop-in for Glm5NextMoE in the MTP layer: router and shared expert on the GPU, routed experts on the CPU."""

    is_cpu_moe = True

    def __init__(self, config, vllm_config, quant_config, prefix: str = ""):
        super().__init__()
        from vllm.logger import init_logger
        from vllm.model_executor.layers.fused_moe import GateLinear
        from vllm.model_executor.layers.fused_moe.router.router_factory import create_fused_moe_router
        from vllm.model_executor.models.deepseek_v2 import _get_moe_router_dtype

        from .model import Glm5NextMLP

        self.logger = init_logger(__name__)
        self.prefix = prefix
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        if config.hidden_act != "silu" or config.topk_method != "noaux_tc":
            raise ValueError("MtpCpuMoE supports SiLU experts with noaux_tc routing only")
        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(config.hidden_size, config.n_routed_experts, out_dtype=self.router_dtype,
                               prefix=f"{prefix}.gate")
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))
        self.shared_experts = None
        if config.n_shared_experts:
            self.shared_experts = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=config.swiglu_limit,
            )
        # The same routing as FusedMoE builds for Glm5NextMoE (routed scale applied to the top-k weights).
        self.router = create_fused_moe_router(
            top_k=self.top_k,
            global_num_experts=config.n_routed_experts,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func=config.scoring_func,
            routed_scaling_factor=config.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
        )
        self.max_rows = max(1, vllm_config.scheduler_config.max_num_seqs)
        spec = os.environ.get("VLLM_EXP_MTP_CPU_MOE_CPUS", "").strip()
        cpus = _parse_list(spec) if spec else default_cpus(_env_int("VLLM_EXP_MTP_CPU_MOE_PER_NODE", 12))
        self.engine = CpuMoEEngine(
            config.n_routed_experts, config.hidden_size, config.moe_intermediate_size, self.top_k, self.max_rows,
            config.swiglu_limit, cpus, prefetch=_env_int("VLLM_EXP_MTP_CPU_MOE_PREFETCH", 1024),
            spin_us=_env_int("VLLM_EXP_MTP_CPU_MOE_SPIN_US", 3000), wbits=_env_int("VLLM_EXP_MTP_CPU_MOE_WBITS", 16),
            prefetch_hint=_env_int("VLLM_EXP_MTP_CPU_MOE_PF_HINT", 0))
        self.prewake_enabled = _env_int("VLLM_EXP_MTP_CPU_MOE_PREWAKE", 1) == 1
        self.engine.configure(float(os.environ.get("VLLM_EXP_MTP_CPU_MOE_REPORT_S", "0") or 0),
                              os.environ.get("VLLM_EXP_MTP_CPU_MOE_MODE_FILE", ""))
        self.logger.info(
            "MTP routed experts on the CPU: %d threads on CPUs %s (%.1f GB of %s weights, slices first-touched in "
            "%.1f s; prefetch %d B, hint %d), up to %d rows per draft forward",
            len(cpus), spec or ",".join(map(str, cpus)),
            config.n_routed_experts * 3 * config.hidden_size * config.moe_intermediate_size * self.engine.wbits / 8e9,
            "INT8 (row scales)" if self.engine.wbits == 8 else "BF16", self.engine.create_s,
            _env_int("VLLM_EXP_MTP_CPU_MOE_PREFETCH", 1024), _env_int("VLLM_EXP_MTP_CPU_MOE_PF_HINT", 0), self.max_rows)
        self.doorbell = _env_int("VLLM_EXP_MTP_CPU_MOE_DOORBELL", 1) == 1
        self.watcher_cpu = _env_int("VLLM_EXP_MTP_CPU_MOE_WATCHER_CPU", -1)
        self._counters: torch.Tensor | None = None
        self._staged: StagedRows | None = None
        self._rows_src: torch.Tensor | None = None
        self._valid_src: torch.Tensor | None = None
        self._num_rows: int | None = None
        self._arange: torch.Tensor | None = None
        self._load_t0: float | None = None

    # -- loading --
    def load_expert_weight(self, expert: int, which: int, w: torch.Tensor) -> None:
        if self._load_t0 is None:
            self._load_t0 = time.monotonic()
        self.engine.set_weight(expert, which, w)

    def finish_loading(self) -> None:
        missing = self.engine.num_missing()
        if missing:
            raise ValueError(f"{self.prefix}: {missing} of {3 * self.n_routed_experts} expert matrices missing")
        self.logger.info("MTP routed experts loaded into CPU memory in %.1f s",
                         time.monotonic() - (self._load_t0 or time.monotonic()))
        # Page-lock the host buffers and allocate the staging buffers now, outside any CUDA graph capture.
        self._ensure_buffers(self.gate.weight.device)

    # -- the speculator's view --
    def set_sample_rows(self, rows: torch.Tensor | None, num_rows: int, valid_src: torch.Tensor | None) -> None:
        """Rows that need the MoE output in the next forward: ``rows[:num_rows]`` (None: rows 0..num_rows-1);
        rows whose ``valid_src`` entry is 0 (padded request slots) are skipped."""
        self._rows_src, self._num_rows, self._valid_src = rows, num_rows, valid_src

    def _ensure_buffers(self, device: torch.device) -> StagedRows:
        if self._staged is None:
            pin = _env_int("VLLM_EXP_MTP_CPU_MOE_PIN", 1) == 1
            dt = self.engine.register_host_buffers(pin_weights=pin)
            self.logger.info("MTP CPU MoE host buffers registered%s in %.1f s",
                             " with the expert weights pinned" if pin else "", dt)
            if self.doorbell:
                major, minor = torch.cuda.get_device_capability(device)
                cubin = build_cubin(f"sm_{major}{minor}")
                self._counters = torch.zeros(4, dtype=torch.int32, device=device)
                self.engine.enable_doorbell(cubin, self.watcher_cpu)
                self.logger.info("MTP CPU MoE hand-off: doorbell (watcher thread %s)",
                                 f"on CPU {self.watcher_cpu}" if self.watcher_cpu >= 0 else "unpinned")
            self._staged = StagedRows(self.engine, device)
            self._arange = torch.arange(self.max_rows, dtype=torch.int64, device=device)
        return self._staged

    def prewake(self) -> None:
        if not self.prewake_enabled:
            return
        stream = torch.cuda.current_stream().cuda_stream
        if self.doorbell and self._counters is not None:
            self.engine.prewake_db(stream, self._counters.data_ptr())
        else:
            self.engine.enqueue_prewake(stream)

    def forward(self, hidden_states: torch.Tensor, already_sequence_parallel: bool = False) -> torch.Tensor:
        n, hd = hidden_states.shape
        st = self._ensure_buffers(hidden_states.device)
        self.engine.note_eager()  # not reached in CUDA graph replays
        r = min(self._num_rows if self._num_rows is not None else n, n, self.max_rows)
        if self._rows_src is None:
            idx = self._arange[:r]
        else:
            idx = self._rows_src[:r].clamp(0, n - 1)
        x = hidden_states.index_select(0, idx)
        logits = self.gate(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        topk_w, topk_ids = self.router.select_experts(x, logits)
        st.num_rows.fill_(r)
        if self._valid_src is not None:
            st.valid[:r].copy_((self._valid_src[:r] > 0).to(torch.int32))
        else:
            st.valid[:r].fill_(1)
        st.rows[:r].copy_(idx)
        st.ids[:r].copy_(topk_ids)
        st.w[:r].copy_(topk_w)
        st.x[:r].copy_(x)
        stream = torch.cuda.current_stream().cuda_stream
        if self.doorbell:
            # Ring, compute the shared expert on the GPU while the CPU computes the routed ones, then wait.
            cnt = self._counters.data_ptr()
            self.engine.ring(stream, st.buf.data_ptr(), self.engine.in_bytes, cnt)
            shared = self.shared_experts(x) if self.shared_experts is not None else None
            self.engine.wait(stream, st.out.data_ptr(), r * hd * 4, cnt)
            vals = st.out[:r]
            if shared is not None:
                vals = vals + shared.float()
        else:
            self.engine.enqueue(stream, st.buf.data_ptr(), self.engine.in_bytes, st.out.data_ptr(), r * hd * 4)
            vals = st.out[:r]
            if self.shared_experts is not None:
                vals = vals + self.shared_experts(x).float()
        # Valid rows go to their position; invalid ones to a spare row past the end (discarded).
        tgt = torch.where(st.valid[:r] != 0, idx, n)
        out = torch.zeros(n + 1, hd, dtype=hidden_states.dtype, device=hidden_states.device)
        out.index_copy_(0, tgt, vals.to(hidden_states.dtype))
        return out[:n]

    def extra_repr(self) -> str:
        return (f"experts={self.n_routed_experts} (CPU, {len(self.engine.cpus)} threads), top_k={self.top_k}, "
                f"max_rows={self.max_rows}")
