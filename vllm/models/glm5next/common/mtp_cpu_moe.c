/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Routed experts of the GLM-5.3 MTP layer on the CPU (see mtp_cpu_moe.py).
 *
 * The MTP layer's MoE output at a position only feeds that position's draft logits, so a draft forward needs the
 * routed experts for one row per request (the sampled position), not for every token. Those rows are copied to host
 * memory, this library computes them, and the result is copied back, all inside the drafter's CUDA graph. Two
 * hand-offs: a doorbell (cmoe_enable_doorbell, cmoe_ring, cmoe_wait; kernels in mtp_cpu_moe_kernels.cu): flags in a
 * mapped host page, polled by a watcher thread on the CPU side and by a spinning kernel on the GPU side; or a host
 * function (cmoe_enqueue: copy down, cmoe_host_fn through cuLaunchHostFunc, copy up), whose round trip costs ~0.45 ms
 * more. A host function must not call CUDA; this one only touches the registered host buffers.
 *
 * Kernel (from exp/cpu_moe_bench.c, Track B gate 1): T pinned worker threads; thread t owns a fixed slice of EVERY
 * expert, intermediate rows [j0, j0+nj) of gate and up and hidden rows [h0, h0+nh) of down, first-touched by itself
 * so that the pages sit on its NUMA node (the vast.ai container forbids mbind). Phase 1: act = silu(clamp(x.gate)) *
 * clamp(x.up) for its j rows, AVX512-BF16 dot products, fp32 accumulation, act rounded to BF16 as the GPU kernels do.
 * Barrier. Phase 2: out[r][h] += w * (down_h . act) for its h rows. Barrier. The calling thread (the CUDA driver's
 * host-function thread) only publishes the call and waits. Fixed thread layout: fixed summation order, so results
 * are bit-reproducible.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <immintrin.h>
#include <limits.h>
#include <linux/futex.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifndef CMOE_H
#define CMOE_H 4096
#endif
#ifndef CMOE_I
#define CMOE_I 2048
#endif
#define H CMOE_H
#define I CMOE_I
#if (H % 64) || (I % 64)
#error "hidden and intermediate sizes must be multiples of 64"
#endif

typedef uint16_t bf16;
static inline bf16 f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (bf16)(u >> 16); }
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

/* ---- errors ---- */
static __thread char g_err[512];
static void set_err(const char *fmt, ...) {
  va_list ap; va_start(ap, fmt); vsnprintf(g_err, sizeof g_err, fmt, ap); va_end(ap);
}
const char *cmoe_last_error(void) { return g_err; }

/* ---- CUDA driver API, resolved at run time ---- */
typedef int CUresult;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;
typedef void (*CUhostFn)(void *);
typedef void *CUmodule;
typedef void *CUfunction;
static CUresult (*p_cuMemHostRegister)(void *, size_t, unsigned int);
static CUresult (*p_cuMemHostUnregister)(void *);
static CUresult (*p_cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
static CUresult (*p_cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
static CUresult (*p_cuLaunchHostFunc)(CUstream, CUhostFn, void *);
static CUresult (*p_cuGetErrorString)(CUresult, const char **);
static CUresult (*p_cuModuleLoad)(CUmodule *, const char *);
static CUresult (*p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                    CUstream, void **, void **);
static CUresult (*p_cuMemHostGetDevicePointer)(CUdeviceptr *, void *, unsigned int);
static pthread_mutex_t g_cuda_lock = PTHREAD_MUTEX_INITIALIZER;

static int load_cuda(void) {
  pthread_mutex_lock(&g_cuda_lock);
  int ok = p_cuLaunchHostFunc != NULL;
  if (!ok) {
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);  /* the driver torch already loaded */
    if (!h) h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) { set_err("dlopen libcuda.so.1: %s", dlerror()); pthread_mutex_unlock(&g_cuda_lock); return -1; }
    p_cuMemHostRegister = (CUresult (*)(void *, size_t, unsigned int))dlsym(h, "cuMemHostRegister_v2");
    p_cuMemHostUnregister = (CUresult (*)(void *))dlsym(h, "cuMemHostUnregister");
    p_cuMemcpyDtoHAsync = (CUresult (*)(void *, CUdeviceptr, size_t, CUstream))dlsym(h, "cuMemcpyDtoHAsync_v2");
    p_cuMemcpyHtoDAsync = (CUresult (*)(CUdeviceptr, const void *, size_t, CUstream))dlsym(h, "cuMemcpyHtoDAsync_v2");
    p_cuGetErrorString = (CUresult (*)(CUresult, const char **))dlsym(h, "cuGetErrorString");
    p_cuLaunchHostFunc = (CUresult (*)(CUstream, CUhostFn, void *))dlsym(h, "cuLaunchHostFunc");
    p_cuModuleLoad = (CUresult (*)(CUmodule *, const char *))dlsym(h, "cuModuleLoad");
    p_cuModuleGetFunction = (CUresult (*)(CUfunction *, CUmodule, const char *))dlsym(h, "cuModuleGetFunction");
    p_cuLaunchKernel = (CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                     CUstream, void **, void **))dlsym(h, "cuLaunchKernel");
    p_cuMemHostGetDevicePointer = (CUresult (*)(CUdeviceptr *, void *, unsigned int))dlsym(h, "cuMemHostGetDevicePointer_v2");
    ok = p_cuMemHostRegister && p_cuMemHostUnregister && p_cuMemcpyDtoHAsync && p_cuMemcpyHtoDAsync &&
         p_cuLaunchHostFunc && p_cuModuleLoad && p_cuModuleGetFunction && p_cuLaunchKernel && p_cuMemHostGetDevicePointer;
    if (!ok) { set_err("libcuda.so.1 lacks a needed driver API entry point"); p_cuLaunchHostFunc = NULL; }
  }
  pthread_mutex_unlock(&g_cuda_lock);
  return ok ? 0 : -1;
}
static int cu_check(CUresult r, const char *what) {
  if (r == 0) return 0;
  const char *s = "?";
  if (p_cuGetErrorString) p_cuGetErrorString(r, &s);
  set_err("%s failed: CUDA error %d (%s)", what, r, s);
  return -1;
}

/* ---- engine ---- */
enum { JOB_MOE = 0, JOB_LOAD = 1, JOB_EXIT = 2 };
typedef struct cmoe cmoe_t;
typedef struct {
  cmoe_t *c;
  int tid, cpu, j0, nj, h0, nh, pin_ok;
  bf16 *w13, *w2;            /* [E][2][nj][H] (gate rows, then up rows), [E][nh][I] */
  size_t w13_bytes, w2_bytes;
  float *acc;                /* [max_rows][nh] */
  pthread_t th;
} worker_t;

struct cmoe {
  int E, topk, max_rows, max_sel, T, pf, spin_us, nstarted;
  float limit;
  worker_t *W;
  uint8_t *map13, *map2; size_t len13, len2;
  uint8_t *w13_base, *w2_base; size_t w13_len, w2_len, act_bytes; int reg_w13, reg_w2, reg_act, pin_weights;
  /* the current call */
  int R, nrows, *rows_list, *comp, *src_of;
  int nsel, *sel_e, *sel_nt, *sel_tok, *slot_of_e; float *sel_w;
  const bf16 *xb;            /* [max_rows][H], in the input buffer */
  bf16 *act;                 /* [max_sel][max_rows][I] */
  float *out;                /* [max_rows][H], the output buffer */
  /* a weight load */
  const bf16 *load_src; int load_e, load_which;
  unsigned char *loaded;     /* [E][3] */
  /* synchronization */
  atomic_int go_gen, job, bar_count, bar_sense, done_gen, nready;
  pthread_mutex_t call_lock;
  /* host buffers, page-locked */
  uint8_t *hin, *hout; size_t in_bytes, out_bytes; int reg_in, reg_out;
  size_t off_valid, off_rows, off_ids, off_w, off_x;
  /* statistics */
  long ncalls, rows_total, sel_total; double sum_ms, max_ms, last_ms;
  /* experiment instrumentation (cmoe_configure): a report every report_s seconds to stderr, and a mode read from a
   * file (0: compute, 1: skip the compute and return zeros) so a running server can be measured without its CPU part */
  double report_s, t_report, t_prewake, t_prev_call, t_mode_check;
  int mode; char mode_path[512];
  long n_eager, n_eager_reported, n_skipped;
#define CMOE_RB 16384
  float rb_ms[CMOE_RB], rb_gap[CMOE_RB], rb_ivl[CMOE_RB]; int rb_n, rb_ngap, rb_nivl;
  /* doorbell (cmoe_enable_doorbell): flags in a mapped host page instead of CUDA host functions */
  volatile uint32_t *db; int reg_db, watcher_cpu, watcher_started; atomic_int watcher_stop; pthread_t watcher;
  CUdeviceptr hin_dev, hout_dev, db_dev; CUfunction k_ring, k_wait, k_prewake; unsigned long long timeout_ns;
  double watch_active_s;
};
enum { DB_PREWAKE = 0, DB_READY = 16, DB_DONE = 32, DB_ERR = 48 };   /* uint32 offsets: separate cache lines */

static long futex(atomic_int *addr, int op, int val) {
  return syscall(SYS_futex, (int *)addr, op, val, NULL, NULL, 0);
}
static void barrier(cmoe_t *c, int *sense) {
  int s = !*sense; *sense = s;
  if (atomic_fetch_add_explicit(&c->bar_count, 1, memory_order_acq_rel) == c->T - 1) {
    atomic_store_explicit(&c->bar_count, 0, memory_order_relaxed);
    atomic_store_explicit(&c->bar_sense, s, memory_order_release);
  } else {
    while (atomic_load_explicit(&c->bar_sense, memory_order_acquire) != s) _mm_pause();
  }
}

#define LD(p) ((__m512bh)_mm512_loadu_si512((const void *)(p)))

/* gate row g and up row u (length H) against the x of n tokens */
static inline __attribute__((always_inline)) void dot_gu(const bf16 *g, const bf16 *u, const bf16 *xb, const int *tok,
                                                         const int n, const int pf, float *gs, float *us) {
  __m512 ag[4][2], au[4][2];
  const bf16 *xs[4];
  for (int i = 0; i < 4; i++) {
    xs[i] = xb + (size_t)tok[i < n ? i : 0] * H;
    for (int r = 0; r < 2; r++) { ag[i][r] = _mm512_setzero_ps(); au[i][r] = _mm512_setzero_ps(); }
  }
  for (int h = 0; h < H; h += 64) {
    _mm_prefetch((const char *)(g + h) + pf, _MM_HINT_T0); _mm_prefetch((const char *)(g + h) + pf + 64, _MM_HINT_T0);
    _mm_prefetch((const char *)(u + h) + pf, _MM_HINT_T0); _mm_prefetch((const char *)(u + h) + pf + 64, _MM_HINT_T0);
    __m512bh g0 = LD(g + h), g1 = LD(g + h + 32), u0 = LD(u + h), u1 = LD(u + h + 32);
    for (int i = 0; i < n; i++) {
      __m512bh a = LD(xs[i] + h), b = LD(xs[i] + h + 32);
      ag[i][0] = _mm512_dpbf16_ps(ag[i][0], g0, a); ag[i][1] = _mm512_dpbf16_ps(ag[i][1], g1, b);
      au[i][0] = _mm512_dpbf16_ps(au[i][0], u0, a); au[i][1] = _mm512_dpbf16_ps(au[i][1], u1, b);
    }
  }
  for (int i = 0; i < n; i++) {
    gs[i] = _mm512_reduce_add_ps(_mm512_add_ps(ag[i][0], ag[i][1]));
    us[i] = _mm512_reduce_add_ps(_mm512_add_ps(au[i][0], au[i][1]));
  }
}

/* down row (length I) against n consecutive act rows */
static inline __attribute__((always_inline)) void dot_w2(const bf16 *row, const bf16 *act, const int n, const int pf,
                                                         float *d) {
  __m512 a[4][2];
  for (int i = 0; i < 4; i++) a[i][0] = a[i][1] = _mm512_setzero_ps();
  for (int j = 0; j < I; j += 64) {
    _mm_prefetch((const char *)(row + j) + pf, _MM_HINT_T0); _mm_prefetch((const char *)(row + j) + pf + 64, _MM_HINT_T0);
    __m512bh w0 = LD(row + j), w1 = LD(row + j + 32);
    for (int i = 0; i < n; i++) {
      a[i][0] = _mm512_dpbf16_ps(a[i][0], w0, LD(act + (size_t)i * I + j));
      a[i][1] = _mm512_dpbf16_ps(a[i][1], w1, LD(act + (size_t)i * I + j + 32));
    }
  }
  for (int i = 0; i < n; i++) d[i] = _mm512_reduce_add_ps(_mm512_add_ps(a[i][0], a[i][1]));
}

static void run_moe(worker_t *w, int *sense) {
  cmoe_t *c = w->c;
  const int MR = c->max_rows, pf = c->pf;
  const float lim = c->limit;
  for (int k = 0; k < c->nsel; k++) {                /* phase 1: gate/up rows j0.. of every selected expert */
    const bf16 *G = w->w13 + (size_t)c->sel_e[k] * 2 * w->nj * H, *U = G + (size_t)w->nj * H;
    const int nt = c->sel_nt[k];
    const int *tok = c->sel_tok + (size_t)k * MR;
    bf16 *act = c->act + (size_t)k * MR * I;
    for (int jj = 0; jj < w->nj; jj++) {
      const bf16 *g = G + (size_t)jj * H, *u = U + (size_t)jj * H;
      for (int q0 = 0; q0 < nt; q0 += 4) {
        const int n = nt - q0 < 4 ? nt - q0 : 4;
        float gs[4], us[4];
        switch (n) {
          case 1: dot_gu(g, u, c->xb, tok + q0, 1, pf, gs, us); break;
          case 2: dot_gu(g, u, c->xb, tok + q0, 2, pf, gs, us); break;
          case 3: dot_gu(g, u, c->xb, tok + q0, 3, pf, gs, us); break;
          default: dot_gu(g, u, c->xb, tok + q0, 4, pf, gs, us); break;
        }
        for (int i = 0; i < n; i++) {
          float gv = gs[i], uv = us[i];
          if (lim > 0.f) { gv = fminf(gv, lim); uv = fminf(fmaxf(uv, -lim), lim); }
          act[(size_t)(q0 + i) * I + w->j0 + jj] = f2bf(gv / (1.f + expf(-gv)) * uv);
        }
      }
    }
  }
  barrier(c, sense);
  memset(w->acc, 0, sizeof(float) * (size_t)MR * w->nh);
  for (int k = 0; k < c->nsel; k++) {                /* phase 2: down rows h0.. of every selected expert */
    const bf16 *W2e = w->w2 + (size_t)c->sel_e[k] * w->nh * I;
    const int nt = c->sel_nt[k];
    const int *tok = c->sel_tok + (size_t)k * MR;
    const float *wt = c->sel_w + (size_t)k * MR;
    const bf16 *act = c->act + (size_t)k * MR * I;
    for (int hh = 0; hh < w->nh; hh++) {
      const bf16 *row = W2e + (size_t)hh * I;
      for (int q0 = 0; q0 < nt; q0 += 4) {
        const int n = nt - q0 < 4 ? nt - q0 : 4;
        float d[4];
        switch (n) {
          case 1: dot_w2(row, act + (size_t)q0 * I, 1, pf, d); break;
          case 2: dot_w2(row, act + (size_t)q0 * I, 2, pf, d); break;
          case 3: dot_w2(row, act + (size_t)q0 * I, 3, pf, d); break;
          default: dot_w2(row, act + (size_t)q0 * I, 4, pf, d); break;
        }
        for (int i = 0; i < n; i++) w->acc[(size_t)tok[q0 + i] * w->nh + hh] += wt[q0 + i] * d[i];
      }
    }
  }
  for (int m = 0; m < c->nrows; m++) {
    const int r = c->rows_list[m];
    memcpy(c->out + (size_t)r * H + w->h0, w->acc + (size_t)r * w->nh, sizeof(float) * w->nh);
  }
  barrier(c, sense);
}

static void run_load(worker_t *w, int *sense) {
  cmoe_t *c = w->c;
  const int e = c->load_e, which = c->load_which;
  const bf16 *src = c->load_src;
  if (which < 2) {                                   /* gate (0) or up (1): rows [I][H] */
    bf16 *dst = w->w13 + ((size_t)e * 2 + which) * w->nj * H;
    memcpy(dst, src + (size_t)w->j0 * H, sizeof(bf16) * (size_t)w->nj * H);
  } else {                                           /* down: rows [H][I] */
    bf16 *dst = w->w2 + (size_t)e * w->nh * I;
    memcpy(dst, src + (size_t)w->h0 * I, sizeof(bf16) * (size_t)w->nh * I);
  }
  barrier(c, sense);
}

static void *worker_main(void *arg) {
  worker_t *w = arg;
  cmoe_t *c = w->c;
  int sense = 0, seen = 0;
  cpu_set_t s; CPU_ZERO(&s); CPU_SET(w->cpu, &s);
  w->pin_ok = pthread_setaffinity_np(pthread_self(), sizeof s, &s) == 0;
  w->acc = aligned_alloc(64, sizeof(float) * (size_t)c->max_rows * (w->nh + 16));
  if (w->acc) memset(w->acc, 0, sizeof(float) * (size_t)c->max_rows * (w->nh + 16));
  memset(w->w13, 0, w->w13_bytes);                   /* first touch: the pages go to this thread's node */
  memset(w->w2, 0, w->w2_bytes);
  atomic_fetch_add_explicit(&c->nready, 1, memory_order_acq_rel);
  for (;;) {
    int g;
    double t_spin = now();
    while ((g = atomic_load_explicit(&c->go_gen, memory_order_acquire)) == seen) {
      if (now() - t_spin > c->spin_us * 1e-6) {
        futex(&c->go_gen, FUTEX_WAIT_PRIVATE, seen); /* woken by a call, or by a prewake (then spin again) */
        t_spin = now();
      } else {
        _mm_pause();
      }
    }
    seen = g;
    const int j = atomic_load_explicit(&c->job, memory_order_acquire);
    if (j == JOB_EXIT) break;
    if (j == JOB_MOE) run_moe(w, &sense); else run_load(w, &sense);
    if (w->tid == 0) {
      atomic_store_explicit(&c->done_gen, g, memory_order_release);
      futex(&c->done_gen, FUTEX_WAKE_PRIVATE, INT_MAX);
    }
  }
  return NULL;
}

/* publish a job to the workers and wait until they are done; caller holds call_lock */
static void dispatch(cmoe_t *c, int job) {
  atomic_store_explicit(&c->job, job, memory_order_release);
  const int g = atomic_load_explicit(&c->go_gen, memory_order_relaxed) + 1;
  atomic_store_explicit(&c->go_gen, g, memory_order_release);
  futex(&c->go_gen, FUTEX_WAKE_PRIVATE, INT_MAX);
  double t_spin = now();
  int d;
  while ((d = atomic_load_explicit(&c->done_gen, memory_order_acquire)) != g) {
    if (now() - t_spin > 0.002) futex(&c->done_gen, FUTEX_WAIT_PRIVATE, d);
    else _mm_pause();
  }
}

static size_t al64(size_t x) { return (x + 63) & ~(size_t)63; }
static void *alloc_2m(size_t bytes, uint8_t **map, size_t *len) {
  const size_t al = (size_t)2 << 20, sz = (bytes + al - 1) & ~(al - 1);
  uint8_t *p = mmap(NULL, sz + al, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return NULL;
  uint8_t *a = (uint8_t *)(((uintptr_t)p + al - 1) & ~(uintptr_t)(al - 1));
  madvise(a, sz, MADV_NOHUGEPAGE);   /* on a busy host, huge-page faults compact memory synchronously */
  *map = p; *len = sz + al;
  return a;
}

void cmoe_destroy(cmoe_t *c);

/* n_experts, top_k, max_rows: the call shape; cpus[n_threads]: one pinned worker per CPU; prefetch_bytes: software
 * prefetch distance in each weight stream (0: off); spin_us: how long idle workers spin before they sleep on a futex;
 * swiglu_limit: clamp of gate (max) and up (+-) before silu(gate) * up, 0 for none. Returns NULL on error. */
cmoe_t *cmoe_create(int n_experts, int top_k, int max_rows, const int *cpus, int n_threads, int prefetch_bytes,
                    int spin_us, float swiglu_limit) {
  if (n_experts < 1 || top_k < 1 || max_rows < 1 || n_threads < 1 || n_threads > 4096) {
    set_err("bad arguments"); return NULL;
  }
  if (!__builtin_cpu_supports("avx512bf16")) { set_err("this CPU lacks AVX512-BF16"); return NULL; }
  cmoe_t *c = calloc(1, sizeof *c);
  if (!c) { set_err("out of memory"); return NULL; }
  c->E = n_experts; c->topk = top_k; c->max_rows = max_rows; c->max_sel = max_rows * top_k; c->T = n_threads;
  c->pf = prefetch_bytes; c->spin_us = spin_us; c->limit = swiglu_limit;
  pthread_mutex_init(&c->call_lock, NULL);
  c->W = calloc(n_threads, sizeof(worker_t));
  c->rows_list = calloc(max_rows, sizeof(int)); c->comp = calloc(max_rows, sizeof(int));
  c->src_of = calloc(max_rows, sizeof(int));
  c->sel_e = calloc(c->max_sel, sizeof(int)); c->sel_nt = calloc(c->max_sel, sizeof(int));
  c->sel_tok = calloc((size_t)c->max_sel * max_rows, sizeof(int));
  c->sel_w = calloc((size_t)c->max_sel * max_rows, sizeof(float));
  c->slot_of_e = malloc(sizeof(int) * n_experts);
  c->act_bytes = (sizeof(bf16) * (size_t)c->max_sel * max_rows * I + 4095) & ~(size_t)4095;
  c->act = aligned_alloc(4096, c->act_bytes);
  c->pin_weights = 1;
  c->loaded = calloc((size_t)n_experts * 3, 1);
  if (!c->W || !c->rows_list || !c->comp || !c->src_of || !c->sel_e || !c->sel_nt || !c->sel_tok || !c->sel_w ||
      !c->slot_of_e || !c->act || !c->loaded) { set_err("out of memory"); cmoe_destroy(c); return NULL; }
  for (int e = 0; e < n_experts; e++) c->slot_of_e[e] = -1;
  /* host buffers: [hdr 64 B: int32 R][int32 valid[MR]][int32 rows[MR]][int32 ids[MR][K]][f32 w[MR][K]][bf16 x[MR][H]] */
  c->off_valid = 64;
  c->off_rows = c->off_valid + al64(4 * (size_t)max_rows);
  c->off_ids = c->off_rows + al64(4 * (size_t)max_rows);
  c->off_w = c->off_ids + al64(4 * (size_t)max_rows * top_k);
  c->off_x = c->off_w + al64(4 * (size_t)max_rows * top_k);
  c->in_bytes = c->off_x + al64(2 * (size_t)max_rows * H);
  c->out_bytes = 4 * (size_t)max_rows * H;
  if (posix_memalign((void **)&c->hin, 4096, c->in_bytes) || posix_memalign((void **)&c->hout, 4096, c->out_bytes)) {
    set_err("out of memory"); cmoe_destroy(c); return NULL;
  }
  memset(c->hin, 0, c->in_bytes); memset(c->hout, 0, c->out_bytes);
  c->xb = (const bf16 *)(c->hin + c->off_x);
  c->out = (float *)c->hout;
  /* thread slices, one 2 MB-aligned region per matrix */
  const size_t al = (size_t)2 << 20;
  size_t o13 = 0, o2 = 0, off13[4096], off2[4096];
  for (int t = 0; t < n_threads; t++) {
    worker_t *w = &c->W[t];
    w->c = c; w->tid = t; w->cpu = cpus[t];
    w->j0 = (int)((long)t * I / n_threads); w->nj = (int)((long)(t + 1) * I / n_threads) - w->j0;
    w->h0 = (int)((long)t * H / n_threads); w->nh = (int)((long)(t + 1) * H / n_threads) - w->h0;
    w->w13_bytes = (size_t)n_experts * 2 * w->nj * H * sizeof(bf16);
    w->w2_bytes = (size_t)n_experts * w->nh * I * sizeof(bf16);
    off13[t] = o13; o13 += (w->w13_bytes + al - 1) & ~(al - 1);
    off2[t] = o2; o2 += (w->w2_bytes + al - 1) & ~(al - 1);
  }
  uint8_t *b13 = alloc_2m(o13, &c->map13, &c->len13), *b2 = alloc_2m(o2, &c->map2, &c->len2);
  if (!b13 || !b2) { set_err("mmap of %.1f GB failed", (o13 + o2) / 1e9); cmoe_destroy(c); return NULL; }
  for (int t = 0; t < n_threads; t++) { c->W[t].w13 = (bf16 *)(b13 + off13[t]); c->W[t].w2 = (bf16 *)(b2 + off2[t]); }
  c->w13_base = b13; c->w13_len = o13; c->w2_base = b2; c->w2_len = o2;
  for (int t = 0; t < n_threads; t++) {
    if (pthread_create(&c->W[t].th, NULL, worker_main, &c->W[t])) {
      set_err("pthread_create failed"); cmoe_destroy(c); return NULL;
    }
    c->nstarted++;
  }
  while (atomic_load_explicit(&c->nready, memory_order_acquire) < n_threads) usleep(1000);
  for (int t = 0; t < n_threads; t++)
    if (!c->W[t].pin_ok || !c->W[t].acc) {
      set_err(!c->W[t].acc ? "out of memory" : "cannot pin a worker to CPU %d", c->W[t].cpu);
      cmoe_destroy(c); return NULL;
    }
  return c;
}

/* page-lock the input and output buffers for asynchronous copies (needs a current CUDA context). Unless disabled
 * (cmoe_set_pin_weights), also pin the expert weights and the activation buffer: automatic NUMA balancing
 * (kernel.numa_balancing=1) keeps write-protecting ranges of unpinned anonymous memory to sample its accesses, and the
 * hinting faults that follow arrive in bursts that stall whichever workers own the range (all others wait at the
 * barrier). The kernel skips pinned pages of private mappings (an extra page reference) when it scans. */
int cmoe_register_host_buffers(cmoe_t *c) {
  if (load_cuda()) return -1;
  if (!c->reg_in) {
    if (cu_check(p_cuMemHostRegister(c->hin, c->in_bytes, 3 /* PORTABLE | DEVICEMAP */), "cuMemHostRegister")) return -1;
    c->reg_in = 1;
  }
  if (!c->reg_out) {
    if (cu_check(p_cuMemHostRegister(c->hout, c->out_bytes, 3), "cuMemHostRegister")) return -1;
    c->reg_out = 1;
  }
  if (c->pin_weights) {
    if (!c->reg_w13) {
      if (cu_check(p_cuMemHostRegister(c->w13_base, c->w13_len, 1), "cuMemHostRegister(gate/up weights)")) return -1;
      c->reg_w13 = 1;
    }
    if (!c->reg_w2) {
      if (cu_check(p_cuMemHostRegister(c->w2_base, c->w2_len, 1), "cuMemHostRegister(down weights)")) return -1;
      c->reg_w2 = 1;
    }
    if (!c->reg_act) {
      if (cu_check(p_cuMemHostRegister(c->act, c->act_bytes, 1), "cuMemHostRegister(activations)")) return -1;
      c->reg_act = 1;
    }
  }
  return 0;
}
void cmoe_set_pin_weights(cmoe_t *c, int on) { c->pin_weights = on; }

void cmoe_destroy(cmoe_t *c) {
  if (!c) return;
  if (c->watcher_started) { atomic_store(&c->watcher_stop, 1); pthread_join(c->watcher, NULL); }
  if (c->reg_db && p_cuMemHostUnregister) p_cuMemHostUnregister((void *)c->db);
  free((void *)c->db);
  if (c->nstarted) {
    atomic_store(&c->job, JOB_EXIT);
    atomic_fetch_add(&c->go_gen, 1);
    futex(&c->go_gen, FUTEX_WAKE_PRIVATE, INT_MAX);
    for (int t = 0; t < c->nstarted; t++) pthread_join(c->W[t].th, NULL);
  }
  if (c->reg_in && p_cuMemHostUnregister) p_cuMemHostUnregister(c->hin);
  if (c->reg_out && p_cuMemHostUnregister) p_cuMemHostUnregister(c->hout);
  if (c->reg_w13 && p_cuMemHostUnregister) p_cuMemHostUnregister(c->w13_base);
  if (c->reg_w2 && p_cuMemHostUnregister) p_cuMemHostUnregister(c->w2_base);
  if (c->reg_act && p_cuMemHostUnregister) p_cuMemHostUnregister(c->act);
  if (c->map13) munmap(c->map13, c->len13);
  if (c->map2) munmap(c->map2, c->len2);
  if (c->W) for (int t = 0; t < c->T; t++) free(c->W[t].acc);
  free(c->W); free(c->rows_list); free(c->comp); free(c->src_of); free(c->sel_e); free(c->sel_nt);
  free(c->sel_tok); free(c->sel_w); free(c->slot_of_e); free(c->act); free(c->loaded); free(c->hin); free(c->hout);
  free(c);
}

/* copy one expert matrix into the workers' slices: which 0 gate [I][H], 1 up [I][H], 2 down [H][I] (BF16) */
int cmoe_set_weight(cmoe_t *c, int e, int which, const void *src) {
  if (e < 0 || e >= c->E || which < 0 || which > 2 || !src) { set_err("bad expert weight (%d, %d)", e, which); return -1; }
  pthread_mutex_lock(&c->call_lock);
  c->load_e = e; c->load_which = which; c->load_src = src;
  dispatch(c, JOB_LOAD);
  c->loaded[(size_t)e * 3 + which] = 1;
  pthread_mutex_unlock(&c->call_lock);
  return 0;
}
int cmoe_num_missing(cmoe_t *c) {
  int n = 0;
  for (size_t i = 0; i < (size_t)c->E * 3; i++) n += !c->loaded[i];
  return n;
}

/* one call, from the input buffer to the output buffer */
static void compute(cmoe_t *c) {
  const double t0 = now();
  int R = *(const int32_t *)c->hin;
  if (R < 0) R = 0;
  if (R > c->max_rows) R = c->max_rows;
  const int32_t *valid = (const int32_t *)(c->hin + c->off_valid), *rows = (const int32_t *)(c->hin + c->off_rows);
  const int32_t *ids = (const int32_t *)(c->hin + c->off_ids);
  const float *wts = (const float *)(c->hin + c->off_w);
  const int MR = c->max_rows, K = c->topk;
  /* rows to compute: valid, first of their row index; other valid rows copy, invalid rows are zero */
  c->R = R; c->nrows = 0;
  for (int r = 0; r < R; r++) {
    c->comp[r] = 0; c->src_of[r] = -1;
    if (!valid[r]) continue;
    for (int q = 0; q < r; q++) if (c->comp[q] && rows[q] == rows[r]) { c->src_of[r] = q; break; }
    if (c->src_of[r] < 0) { c->comp[r] = 1; c->rows_list[c->nrows++] = r; }
  }
  c->nsel = 0;
  for (int m = 0; m < c->nrows; m++) {
    const int r = c->rows_list[m];
    for (int k = 0; k < K; k++) {
      const int e = ids[(size_t)r * K + k];
      if (e < 0 || e >= c->E) continue;
      int s = c->slot_of_e[e];
      if (s < 0) { s = c->nsel++; c->slot_of_e[e] = s; c->sel_e[s] = e; c->sel_nt[s] = 0; }
      const int q = c->sel_nt[s]++;
      c->sel_tok[(size_t)s * MR + q] = r; c->sel_w[(size_t)s * MR + q] = wts[(size_t)r * K + k];
    }
  }
  for (int s = 0; s < c->nsel; s++) c->slot_of_e[c->sel_e[s]] = -1;
  if (c->nsel > 0) dispatch(c, JOB_MOE);
  for (int r = 0; r < R; r++) {
    if (c->comp[r] && c->nsel > 0) continue;
    if (c->src_of[r] >= 0 && c->nsel > 0) memcpy(c->out + (size_t)r * H, c->out + (size_t)c->src_of[r] * H, 4 * H);
    else memset(c->out + (size_t)r * H, 0, 4 * H);
  }
  const double ms = (now() - t0) * 1e3;
  c->ncalls++; c->rows_total += c->nrows; c->sel_total += c->nsel; c->sum_ms += ms; c->last_ms = ms;
  if (ms > c->max_ms) c->max_ms = ms;
}

static int cmpf(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
static float pct(float *v, int n, double q) { return n ? v[(int)(q * (n - 1) + 0.5)] : 0.f; }
static void read_mode(cmoe_t *c, double t) {
  if (!c->mode_path[0] || t - c->t_mode_check < 0.2) return;
  c->t_mode_check = t;
  FILE *f = fopen(c->mode_path, "r");
  int m = 0;
  if (f) { if (fscanf(f, "%d", &m) != 1) m = 0; fclose(f); }
  if (m != c->mode) { fprintf(stderr, "[mtp_cpu_moe] mode %d -> %d\n", c->mode, m); c->mode = m; }
}
static void report(cmoe_t *c, double t) {
  const int n = c->rb_n < CMOE_RB ? c->rb_n : CMOE_RB, ng = c->rb_ngap < CMOE_RB ? c->rb_ngap : CMOE_RB;
  const int ni = c->rb_nivl < CMOE_RB ? c->rb_nivl : CMOE_RB;
  double sum = 0; for (int i = 0; i < n; i++) sum += c->rb_ms[i];
  qsort(c->rb_ms, n, sizeof(float), cmpf); qsort(c->rb_gap, ng, sizeof(float), cmpf); qsort(c->rb_ivl, ni, sizeof(float), cmpf);
  int slow = 0; for (int i = 0; i < n; i++) slow += c->rb_ms[i] > 3.f;
  fprintf(stderr, "[mtp_cpu_moe] %.1f s: %d calls (%ld eager, %ld skipped), host function ms mean %.3f p50 %.3f p90 %.3f "
          "p99 %.3f max %.3f, >3 ms %d; prewake->call ms p50 %.3f p90 %.3f (%d); call interval ms p50 %.2f p10 %.2f "
          "p90 %.2f; rows/call %.2f, experts/call %.1f, threads %d, mode %d, %s, timeouts %u\n",
          t - c->t_report, n, c->n_eager - c->n_eager_reported, c->n_skipped, n ? sum / n : 0.0, pct(c->rb_ms, n, .5),
          pct(c->rb_ms, n, .9), pct(c->rb_ms, n, .99), n ? c->rb_ms[n - 1] : 0.f, slow, pct(c->rb_gap, ng, .5),
          pct(c->rb_gap, ng, .9), ng, pct(c->rb_ivl, ni, .5), pct(c->rb_ivl, ni, .1), pct(c->rb_ivl, ni, .9),
          c->ncalls ? (double)c->rows_total / c->ncalls : 0.0, c->ncalls ? (double)c->sel_total / c->ncalls : 0.0, c->T,
          c->mode, c->watcher_started ? "doorbell" : "host functions", c->db ? c->db[DB_ERR] : 0u);
  c->rb_n = c->rb_ngap = c->rb_nivl = 0; c->n_eager_reported = c->n_eager; c->n_skipped = 0; c->t_report = t;
}

/* one call from the input buffer to the output buffer, with the experiment mode and the statistics */
static void run_call(cmoe_t *c) {
  pthread_mutex_lock(&c->call_lock);
  const double t0 = now();
  read_mode(c, t0);
  if (c->mode == 1) {                                /* experiment: no CPU work, zero output */
    int R = *(const int32_t *)c->hin;
    if (R < 0) R = 0;
    if (R > c->max_rows) R = c->max_rows;
    memset(c->out, 0, sizeof(float) * (size_t)R * H);
    c->n_skipped++;
  } else {
    compute(c);
  }
  if (c->report_s > 0) {
    const double t1 = now();
    if (c->rb_n < CMOE_RB) c->rb_ms[c->rb_n++] = (float)((t1 - t0) * 1e3);
    if (c->t_prewake > 0 && t0 - c->t_prewake < 0.05 && c->rb_ngap < CMOE_RB)
      c->rb_gap[c->rb_ngap++] = (float)((t0 - c->t_prewake) * 1e3);
    if (c->t_prev_call > 0 && c->rb_nivl < CMOE_RB) c->rb_ivl[c->rb_nivl++] = (float)((t0 - c->t_prev_call) * 1e3);
    c->t_prev_call = t0;
    if (c->t_report == 0) c->t_report = t0;
    else if (t1 - c->t_report >= c->report_s) report(c, t1);
  }
  pthread_mutex_unlock(&c->call_lock);
}
/* CUhostFn: one call (enqueued by cmoe_enqueue) */
void cmoe_host_fn(void *arg) { run_call((cmoe_t *)arg); }
/* CUhostFn: wake the sleeping workers ahead of a call; they spin for spin_us, then sleep again */
void cmoe_prewake_fn(void *arg) {
  cmoe_t *c = arg;
  futex(&c->go_gen, FUTEX_WAKE_PRIVATE, INT_MAX);
  c->t_prewake = now();
}
/* ---- doorbell ---- */
static void *watcher_main(void *arg) {
  cmoe_t *c = arg;
  if (c->watcher_cpu >= 0) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(c->watcher_cpu, &s);
    if (pthread_setaffinity_np(pthread_self(), sizeof s, &s))
      fprintf(stderr, "[mtp_cpu_moe] cannot pin the watcher to CPU %d; it runs unpinned\n", c->watcher_cpu);
  }
  volatile uint32_t *db = c->db;
  uint32_t seen_pw = db[DB_PREWAKE], seen_rd = db[DB_READY];
  double t_active = now();
  while (!atomic_load_explicit(&c->watcher_stop, memory_order_relaxed)) {
    const uint32_t pw = db[DB_PREWAKE];
    if (pw != seen_pw) {
      seen_pw = pw;
      futex(&c->go_gen, FUTEX_WAKE_PRIVATE, INT_MAX);  /* the workers spin for spin_us, then sleep again */
      c->t_prewake = t_active = now();
    }
    const uint32_t rd = db[DB_READY];
    if (rd != seen_rd) {
      seen_rd = rd;
      atomic_thread_fence(memory_order_acquire);
      run_call(c);
      atomic_thread_fence(memory_order_seq_cst);     /* the output before the flag */
      db[DB_DONE] = rd;
      t_active = now();
      continue;
    }
    if (now() - t_active < c->watch_active_s) _mm_pause();
    else { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); }   /* idle: poll every 50 us */
  }
  return NULL;
}

/* Switch to the doorbell hand-off: load the kernels (a cubin built from mtp_cpu_moe_kernels.cu), map a flag page,
 * start the watcher thread (pinned to watcher_cpu if >= 0). Needs a current CUDA context and registered buffers. */
int cmoe_enable_doorbell(cmoe_t *c, const char *cubin_path, int watcher_cpu, double timeout_s) {
  if (load_cuda()) return -1;
  if (c->watcher_started) return 0;
  if (!c->reg_in || !c->reg_out) { set_err("host buffers not registered"); return -1; }
  CUmodule m;
  if (cu_check(p_cuModuleLoad(&m, cubin_path), "cuModuleLoad")) return -1;
  if (cu_check(p_cuModuleGetFunction(&c->k_ring, m, "cmoe_stage_ring"), "cuModuleGetFunction(cmoe_stage_ring)") ||
      cu_check(p_cuModuleGetFunction(&c->k_wait, m, "cmoe_wait_load"), "cuModuleGetFunction(cmoe_wait_load)") ||
      cu_check(p_cuModuleGetFunction(&c->k_prewake, m, "cmoe_prewake_ring"), "cuModuleGetFunction(cmoe_prewake_ring)"))
    return -1;
  void *page = NULL;
  if (posix_memalign(&page, 4096, 4096)) { set_err("out of memory"); return -1; }
  memset(page, 0, 4096);
  c->db = page;
  if (cu_check(p_cuMemHostRegister(page, 4096, 3), "cuMemHostRegister(doorbell)")) return -1;
  c->reg_db = 1;
  if (cu_check(p_cuMemHostGetDevicePointer(&c->db_dev, page, 0), "cuMemHostGetDevicePointer(doorbell)") ||
      cu_check(p_cuMemHostGetDevicePointer(&c->hin_dev, c->hin, 0), "cuMemHostGetDevicePointer(input)") ||
      cu_check(p_cuMemHostGetDevicePointer(&c->hout_dev, c->hout, 0), "cuMemHostGetDevicePointer(output)"))
    return -1;
  c->watcher_cpu = watcher_cpu;
  c->timeout_ns = (unsigned long long)(timeout_s * 1e9);
  c->watch_active_s = 0.2;
  atomic_store(&c->watcher_stop, 0);
  if (pthread_create(&c->watcher, NULL, watcher_main, c)) { set_err("pthread_create (watcher) failed"); return -1; }
  c->watcher_started = 1;
  return 0;
}
/* on stream: copy in_bytes of the staged input (device) to the host input buffer and ring (counters: device int32[4]) */
int cmoe_ring(cmoe_t *c, unsigned long long stream, unsigned long long dev_in, size_t in_bytes,
              unsigned long long counters) {
  if (!c->watcher_started) { set_err("doorbell not enabled"); return -1; }
  if (in_bytes > c->in_bytes || in_bytes % 16) { set_err("bad input size"); return -1; }
  CUdeviceptr src = (CUdeviceptr)dev_in, dst = c->hin_dev, cnt = (CUdeviceptr)counters, flag = c->db_dev + 4 * DB_READY;
  unsigned n16 = (unsigned)(in_bytes / 16);
  void *args[] = {&src, &dst, &n16, &cnt, &flag};
  return cu_check(p_cuLaunchKernel(c->k_ring, 1, 1, 1, 256, 1, 1, 0, (CUstream)(uintptr_t)stream, args, NULL),
                  "cuLaunchKernel(cmoe_stage_ring)");
}
/* on stream: wait for the CPU's result of the last ring, then copy out_bytes of it to dev_out */
int cmoe_wait(cmoe_t *c, unsigned long long stream, unsigned long long dev_out, size_t out_bytes,
              unsigned long long counters) {
  if (!c->watcher_started) { set_err("doorbell not enabled"); return -1; }
  if (out_bytes > c->out_bytes || out_bytes % 16) { set_err("bad output size"); return -1; }
  CUdeviceptr cnt = (CUdeviceptr)counters, done = c->db_dev + 4 * DB_DONE, src = c->hout_dev, dst = (CUdeviceptr)dev_out;
  CUdeviceptr err = c->db_dev + 4 * DB_ERR;
  unsigned n16 = (unsigned)(out_bytes / 16);
  unsigned long long to = c->timeout_ns;
  void *args[] = {&cnt, &done, &src, &dst, &n16, &to, &err};
  return cu_check(p_cuLaunchKernel(c->k_wait, 1, 1, 1, 256, 1, 1, 0, (CUstream)(uintptr_t)stream, args, NULL),
                  "cuLaunchKernel(cmoe_wait_load)");
}
/* on stream: ring the prewake flag (the watcher wakes the workers) */
int cmoe_prewake_db(cmoe_t *c, unsigned long long stream, unsigned long long counters) {
  if (!c->watcher_started) { set_err("doorbell not enabled"); return -1; }
  CUdeviceptr cnt = (CUdeviceptr)counters, flag = c->db_dev + 4 * DB_PREWAKE;
  void *args[] = {&cnt, &flag};
  return cu_check(p_cuLaunchKernel(c->k_prewake, 1, 1, 1, 32, 1, 1, 0, (CUstream)(uintptr_t)stream, args, NULL),
                  "cuLaunchKernel(cmoe_prewake_ring)");
}
/* wait-kernel timeouts so far (doorbell) */
int cmoe_doorbell_errors(cmoe_t *c) { return c->db ? (int)c->db[DB_ERR] : 0; }

/* experiment instrumentation, see struct cmoe; mode_path may be NULL or empty */
void cmoe_configure(cmoe_t *c, double report_s, const char *mode_path) {
  pthread_mutex_lock(&c->call_lock);
  c->report_s = report_s;
  snprintf(c->mode_path, sizeof c->mode_path, "%s", mode_path ? mode_path : "");
  pthread_mutex_unlock(&c->call_lock);
}
/* counts forwards that ran eagerly, i.e. outside a CUDA graph replay (called from Python) */
void cmoe_note_eager(cmoe_t *c) { c->n_eager++; }
/* the same call, synchronously (tests) */
void cmoe_run(cmoe_t *c) { cmoe_host_fn(c); }

/* on stream: copy in_bytes from dev_in to the input buffer, run cmoe_host_fn, copy out_bytes of the output buffer
 * to dev_out. Capturable into a CUDA graph (memcpy, host and memcpy nodes). */
int cmoe_enqueue(cmoe_t *c, unsigned long long stream, unsigned long long dev_in, size_t in_bytes,
                 unsigned long long dev_out, size_t out_bytes) {
  if (load_cuda()) return -1;
  if (!c->reg_in || !c->reg_out) { set_err("host buffers not registered"); return -1; }
  if (in_bytes > c->in_bytes || out_bytes > c->out_bytes) { set_err("copy larger than the host buffers"); return -1; }
  CUstream s = (CUstream)(uintptr_t)stream;
  if (cu_check(p_cuMemcpyDtoHAsync(c->hin, (CUdeviceptr)dev_in, in_bytes, s), "cuMemcpyDtoHAsync")) return -1;
  if (cu_check(p_cuLaunchHostFunc(s, cmoe_host_fn, c), "cuLaunchHostFunc")) return -1;
  if (out_bytes && cu_check(p_cuMemcpyHtoDAsync((CUdeviceptr)dev_out, c->hout, out_bytes, s), "cuMemcpyHtoDAsync"))
    return -1;
  return 0;
}
int cmoe_enqueue_prewake(cmoe_t *c, unsigned long long stream) {
  if (load_cuda()) return -1;
  return cu_check(p_cuLaunchHostFunc((CUstream)(uintptr_t)stream, cmoe_prewake_fn, c), "cuLaunchHostFunc");
}

/* layout and statistics for the Python side */
void *cmoe_in_buffer(cmoe_t *c) { return c->hin; }
void *cmoe_out_buffer(cmoe_t *c) { return c->hout; }
size_t cmoe_in_bytes(cmoe_t *c) { return c->in_bytes; }
size_t cmoe_out_bytes(cmoe_t *c) { return c->out_bytes; }
/* which: 0 valid, 1 rows, 2 ids, 3 weights, 4 x */
size_t cmoe_offset(cmoe_t *c, int which) {
  switch (which) {
    case 0: return c->off_valid; case 1: return c->off_rows; case 2: return c->off_ids;
    case 3: return c->off_w; case 4: return c->off_x; default: return 0;
  }
}
/* out[0..6]: calls, rows computed, experts read, total ms, max ms, last ms, threads */
void cmoe_stats(cmoe_t *c, double *out) {
  out[0] = (double)c->ncalls; out[1] = (double)c->rows_total; out[2] = (double)c->sel_total;
  out[3] = c->sum_ms; out[4] = c->max_ms; out[5] = c->last_ms; out[6] = (double)c->T;
}
void cmoe_reset_stats(cmoe_t *c) { c->ncalls = c->rows_total = c->sel_total = 0; c->sum_ms = c->max_ms = c->last_ms = 0; }
