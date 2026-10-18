#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU plan for the MTP drafter's CPU MoE (VLLM_EXP_MTP_CPU_MOE=1): which cores run the MoE workers, which one runs the
doorbell watcher, and which CPUs the rest of the server may use. Torch-free, so a launcher can run it before
starting vLLM (by path: it does not import vllm). Prints shell assignments:

  MTP_MOE_CPUS=...      VLLM_EXP_MTP_CPU_MOE_CPUS: the first --per-node physical cores of every NUMA node (one hardware
                        thread each), capped so the workers stay below 80% of the container's CPU quota
  MTP_WATCHER_CPU=...   VLLM_EXP_MTP_CPU_MOE_WATCHER_CPU: a free physical core on the NUMA node of the drafter's GPU
                        (the last pipeline stage's, by default the last visible GPU)
  MTP_SERVER_CPUS=...   for taskset: every allowed CPU except those cores and their SMT siblings, so no server thread
                        shares a core with a worker (a worker slowed down by a busy sibling stalls every call at the
                        barrier)
usage: mtp_cpu_plan.py [--per-node 12] [--gpu-bus-id 0000:41:00.0]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-node", type=int, default=12)
    ap.add_argument("--gpu-bus-id", default="")
    a = ap.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    nodes = {}
    for d in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
        nodes[int(d.rsplit("node", 1)[1])] = [c for c in parse(open(f"{d}/cpulist").read()) if c in allowed]
    if not nodes:
        nodes = {0: allowed}
    cores = {}
    for n, cpus in nodes.items():
        seen, cs = set(), []
        for c in cpus:
            if c not in seen:
                seen.update(siblings(c)); cs.append(c)
        cores[n] = cs
    per_node = a.per_node
    q = quota()
    if q is not None:
        per_node = max(1, min(per_node, int(0.8 * q) // max(1, len([n for n in cores if cores[n]]))))
    moe = [c for n in sorted(cores) for c in cores[n][:per_node]]
    reserved = {s for c in moe for s in siblings(c)}
    gn = gpu_node(a.gpu_bus_id)
    order = ([gn] if gn in cores else []) + [n for n in sorted(cores) if n != gn]
    watcher = next((c for n in order for c in cores[n] if c not in reserved), -1)
    if watcher >= 0:
        reserved |= set(siblings(watcher))
    server = [c for c in allowed if c not in reserved]
    if len(server) < 8:
        raise SystemExit(f"mtp_cpu_plan: only {len(server)} CPUs would be left for the server ({len(allowed)} allowed); "
                         "the CPU MoE needs a larger machine, or a smaller --per-node")
    print(f"MTP_MOE_CPUS={fmt(moe)}")
    print(f"MTP_WATCHER_CPU={watcher}")
    print(f"MTP_SERVER_CPUS={fmt(server)}")
    print(f"# {len(moe)} MoE workers ({per_node} per node, quota {q}), watcher on node {gn}, {len(server)} CPUs for the "
          f"server")


if __name__ == "__main__":
    main()
