#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU plan for the MTP drafter's CPU MoE (VLLM_EXP_MTP_CPU_MOE=1): which cores run the MoE workers, which one runs the
doorbell watcher, and which CPUs the rest of the server may use. Torch-free, so a launcher can run it before
starting vLLM (by path: it does not import vllm). Prints shell assignments:

  MTP_MOE_CPUS=...      VLLM_EXP_MTP_CPU_MOE_CPUS: --workers physical cores (default 48; one hardware thread each),
                        spread evenly over the NUMA nodes, the first cores of each; at most 80% of the container's CPU
                        quota, and leaving --reserve cores (default 8) and the watcher's to the server
  MTP_WATCHER_CPU=...   VLLM_EXP_MTP_CPU_MOE_WATCHER_CPU: a free physical core on the NUMA node of the drafter's GPU
                        (the last pipeline stage's, by default the last visible GPU)
  MTP_SERVER_CPUS=...   for taskset: every allowed CPU except those cores and their SMT siblings, so no server thread
                        shares a core with a worker (a worker slowed down by a busy sibling stalls every call at the
                        barrier)

The CPU MoE streams the selected experts' weights from memory; its time per call falls with the number of memory
channels and of workers reading them. A fixed count per node (--per-node, the earlier default of 12) gave a 1- or
2-node machine 12 or 24 workers: per call 2.5-3x or 1.3-1.8x slower than 48 on the same socket (exp/cpu_moe_bench.c
on CPU subsets of a 2-socket Xeon 6767P).
usage: mtp_cpu_plan.py [--workers 48] [--per-node N] [--reserve 8] [--gpu-bus-id 0000:41:00.0]
"""
import argparse, glob, os, re, subprocess


def parse(s):
    out = []
    for tok in s.strip().split(","):
        if not tok.strip():
            continue
        a, _, b = tok.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def fmt(cpus):
    cpus, out, i = sorted(set(cpus)), [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def siblings(c):
    p = f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
    return parse(open(p).read()) if os.path.exists(p) else [c]


def quota():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else int(q) / int(p)
    except (OSError, ValueError):
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us").read())
        return None if q <= 0 else q / p
    except (OSError, ValueError):
        return None


def gpu_node(bus_id):
    if not bus_id:
        try:
            ids = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout.split()
            vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            if vis:
                ids = [ids[int(i)] for i in vis.split(",") if i.strip().isdigit()]
            bus_id = ids[-1] if ids else ""
        except (OSError, subprocess.SubprocessError, IndexError, ValueError):
            return None
    m = re.match(r"(?:0000)?([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])", bus_id.strip()[-12:])
    path = f"/sys/bus/pci/devices/{m.group(1).lower()}/numa_node" if m else ""
    try:
        n = int(open(path).read())
        return n if n >= 0 else None
    except (OSError, ValueError):
        return None


def topology():
    """(allowed CPUs, {node: allowed CPUs}) from sysfs and this process's affinity."""
    allowed = sorted(os.sched_getaffinity(0))
    nodes = {}
    for d in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
        nodes[int(d.rsplit("node", 1)[1])] = [c for c in parse(open(f"{d}/cpulist").read()) if c in allowed]
    return allowed, (nodes or {0: allowed})


def plan(allowed, nodes, sib, q, gn, workers=48, per_node=None, reserve=8):
    """The CPU split. allowed: CPUs; nodes: {node: CPUs}; sib(cpu) -> its SMT siblings (itself included); q: the CPU
    quota or None; gn: the drafter's GPU's node or None. Returns (moe CPUs, watcher CPU or -1, server CPUs, note)."""
    cores = {}
    for n, cpus in nodes.items():
        seen, cs = set(), []
        for c in cpus:
            if c not in seen:
                seen.update(sib(c)); cs.append(c)
        if cs:
            cores[n] = cs
    total = sum(len(cs) for cs in cores.values())
    if per_node is not None:  # the earlier rule: a fixed count per node
        cap = per_node if q is None else max(1, min(per_node, int(0.8 * q) // max(1, len(cores))))
        take = {n: min(cap, len(cores[n])) for n in cores}
    else:
        w = min(workers, total - reserve - 1)  # leave the server `reserve` cores and the watcher one
        if q is not None:
            w = min(w, int(0.8 * q))  # threads beyond the quota are throttled until the next period
        take = {n: 0 for n in cores}
        for _ in range(max(0, w)):  # round robin: the same count on every node; a node out of cores is skipped
            n = min((n for n in cores if take[n] < len(cores[n])), key=lambda n: (take[n], n), default=None)
            if n is None:
                break
            take[n] += 1
    moe = [c for n in sorted(cores) for c in cores[n][:take[n]]]
    reserved = {s for c in moe for s in sib(c)}
    order = ([gn] if gn in cores else []) + [n for n in sorted(cores) if n != gn]
    watcher = next((c for n in order for c in cores[n] if c not in reserved), -1)
    if watcher >= 0:
        reserved |= set(sib(watcher))
    server = [c for c in allowed if c not in reserved]
    counts = sorted(set(take.values()))
    note = (f"{len(moe)} MoE workers ({'/'.join(map(str, counts))} per node on {len(cores)} nodes, quota {q}), "
            f"watcher on node {gn}, {len(server)} CPUs for the server")
    return moe, watcher, server, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("VLLM_EXP_MTP_CPU_MOE_WORKERS", "48")))
    ap.add_argument("--per-node", type=int, default=None,
                    help="a fixed count per NUMA node instead of --workers (the earlier rule)")
    ap.add_argument("--reserve", type=int, default=8, help="physical cores left to the server")
    ap.add_argument("--gpu-bus-id", default="")
    a = ap.parse_args()
    allowed, nodes = topology()
    moe, watcher, server, note = plan(allowed, nodes, siblings, quota(), gpu_node(a.gpu_bus_id), a.workers,
                                      a.per_node, a.reserve)
    if not moe or len(server) < 8:
        raise SystemExit(f"mtp_cpu_plan: {len(moe)} MoE workers and {len(server)} CPUs for the server "
                         f"({len(allowed)} allowed); the CPU MoE needs a larger machine (or MTP=0)")
    print(f"MTP_MOE_CPUS={fmt(moe)}")
    print(f"MTP_WATCHER_CPU={watcher}")
    print(f"MTP_SERVER_CPUS={fmt(server)}")
    print(f"# {note}")


if __name__ == "__main__":
    main()
