// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Doorbell kernels of the MTP drafter's CPU MoE (mtp_cpu_moe.{c,py}): the GPU hands rows to the CPU and takes the
// result back through flags in mapped, page-locked host memory instead of CUDA host functions, whose round trip
// costs ~0.45 ms each on the stream. A CPU watcher thread polls the flags. All three kernels are single-block and
// capturable in CUDA graphs. counters: device-side sequence numbers ([0] calls, [1] prewakes), so a replayed graph
// rings a new number every time.

#include <stdint.h>

// Copy the staged input (device) to the host input buffer, then publish the call number in *ready.
extern "C" __global__ void cmoe_stage_ring(const uint4* __restrict__ src, uint4* dst_host, unsigned n16,
                                           unsigned* counters, volatile unsigned* ready) {
  for (unsigned i = threadIdx.x; i < n16; i += blockDim.x) dst_host[i] = src[i];
  __threadfence_system();  // each thread's writes to host memory before the flag
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned c = counters[0] + 1;
    counters[0] = c;
    __threadfence_system();
    *ready = c;
  }
}

// Wait until the CPU published the same call number in *done, then copy the host output buffer to dst (device).
// After timeout_ns the kernel gives up, writes zeros and counts the failure in err[0] (a draft from a zero MoE output
// is only a worse draft: verification uses the target model). err is in host memory (no atomics there).
extern "C" __global__ void cmoe_wait_load(const unsigned* counters, volatile const unsigned* done,
                                          const uint4* src_host, uint4* dst, unsigned n16,
                                          unsigned long long timeout_ns, unsigned* err) {
  __shared__ int ok;
  if (threadIdx.x == 0) {
    const unsigned c = counters[0];
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    ok = 1;
    while (*done != c) {
      __nanosleep(128);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
      if (t - t0 > timeout_ns) { ok = 0; *(volatile unsigned*)err = *(volatile unsigned*)err + 1u; break; }
    }
    __threadfence_system();
  }
  __syncthreads();
  for (unsigned i = threadIdx.x; i < n16; i += blockDim.x) {
    uint4 v = make_uint4(0, 0, 0, 0);
    if (ok)
      asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src_host + i));
    dst[i] = v;
  }
}

// Bump the prewake number: the watcher wakes the sleeping workers when it changes.
extern "C" __global__ void cmoe_prewake_ring(unsigned* counters, volatile unsigned* flag) {
  if (threadIdx.x == 0) {
    const unsigned c = counters[1] + 1;
    counters[1] = c;
    *flag = c;
  }
}
