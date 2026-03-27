/*
 * rac_hal.h — RAC Hardware Abstraction Layer
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Self-optimizing dispatch layer that probes the silicon at startup and
 * routes every RAC operation to the fastest available path.
 *
 * The HAL:
 *   1. Enumerates hardware at init (CPU, SIMD, cache, NUMA, GPU)
 *   2. Builds a capability profile (what's fast, not just what's present)
 *   3. Dispatches to the optimal kernel (AVX2/AVX-512/NEON/scalar)
 *   4. Selects tile sizes based on cache topology + matrix dimensions
 *   5. Distributes work across cores with optimal affinity
 *   6. Exposes the profile for caller inspection and override
 *
 * Usage:
 *   rac_hal_init();                           // probe hardware once
 *   const rac_hw_profile *hw = rac_hal_profile();  // inspect what was found
 *   rac_hal_sgemm(A, B, C, M, N, K, 1, 0);  // auto-dispatched
 *   rac_hal_shutdown();                       // cleanup
 *
 * Thread safety: rac_hal_init() is not thread-safe (call once at startup).
 *               All dispatch functions are thread-safe after init.
 */

#ifndef RAC_HAL_H
#define RAC_HAL_H

#include "rac_cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── SIMD capability flags ──────────────────────────────────────────────── */

typedef enum {
    RAC_SIMD_NONE    = 0,
    RAC_SIMD_SSE2    = (1 << 0),
    RAC_SIMD_SSE4    = (1 << 1),
    RAC_SIMD_AVX     = (1 << 2),
    RAC_SIMD_AVX2    = (1 << 3),
    RAC_SIMD_FMA     = (1 << 4),
    RAC_SIMD_AVX512F = (1 << 5),
    RAC_SIMD_AVX512VL= (1 << 6),
    RAC_SIMD_NEON    = (1 << 7),
    RAC_SIMD_SVE     = (1 << 8),
    RAC_SIMD_SVE2    = (1 << 9),
} rac_simd_flags;

/* ── CPU microarchitecture IDs ──────────────────────────────────────────── */

typedef enum {
    RAC_UARCH_UNKNOWN = 0,
    /* Intel */
    RAC_UARCH_SKYLAKE,
    RAC_UARCH_ICELAKE,
    RAC_UARCH_ALDERLAKE,
    RAC_UARCH_SAPPHIRERAPIDS,
    RAC_UARCH_GRANITERAPIDS,
    /* AMD */
    RAC_UARCH_ZEN2,
    RAC_UARCH_ZEN3,
    RAC_UARCH_ZEN4,
    RAC_UARCH_ZEN5,
    /* ARM */
    RAC_UARCH_CORTEX_A76,
    RAC_UARCH_CORTEX_X2,
    RAC_UARCH_NEOVERSE_V1,
    RAC_UARCH_NEOVERSE_V2,
    RAC_UARCH_APPLE_M1,
    RAC_UARCH_APPLE_M2,
    RAC_UARCH_APPLE_M3,
    RAC_UARCH_APPLE_M4,
} rac_uarch;

/* ── Cache topology ─────────────────────────────────────────────────────── */

typedef struct {
    int l1d_size_kb;     /* L1 data cache per core (KB) */
    int l1d_line_size;   /* L1 cache line size (bytes) */
    int l2_size_kb;      /* L2 cache per core (KB) */
    int l3_size_kb;      /* L3 total (KB), 0 if none */
    int l3_shared_cores; /* how many cores share L3 */
} rac_cache_info;

/* ── NUMA topology ──────────────────────────────────────────────────────── */

typedef struct {
    int num_nodes;       /* NUMA node count */
    int num_sockets;     /* physical socket count */
    int cores_per_node;  /* cores per NUMA node */
} rac_numa_info;

/* ── GPU info (if detected) ─────────────────────────────────────────────── */

typedef enum {
    RAC_GPU_NONE = 0,
    RAC_GPU_NVIDIA,
    RAC_GPU_AMD,
    RAC_GPU_INTEL,
    RAC_GPU_APPLE,
} rac_gpu_vendor;

typedef struct {
    rac_gpu_vendor vendor;
    char name[128];
    int compute_units;    /* SMs or CUs */
    int clock_mhz;
    int vram_mb;
    int pcie_gen;
    int pcie_width;
} rac_gpu_info;

/* ── Hardware profile ───────────────────────────────────────────────────── */

typedef struct {
    /* CPU */
    char cpu_name[128];
    rac_uarch uarch;
    int num_physical_cores;
    int num_logical_cores;
    int base_clock_mhz;
    uint32_t simd_flags;      /* bitmask of rac_simd_flags */

    /* SIMD performance characteristics */
    int avx512_throttle;      /* 1 if AVX-512 causes frequency throttle (e.g. Zen3) */
    int preferred_simd_width; /* best SIMD width in bytes (16=SSE, 32=AVX2, 64=AVX-512) */

    /* Cache */
    rac_cache_info cache;

    /* NUMA */
    rac_numa_info numa;

    /* Optimal tile sizes (computed from cache + uarch) */
    int optimal_tile_sgemm;     /* tile for SGEMM (L1-fitted) */
    int optimal_tile_sgemm_l2;  /* larger tile for L2 blocking */

    /* GPU */
    int num_gpus;
    rac_gpu_info gpu[8];  /* up to 8 GPUs */

    /* Dispatch decisions */
    int use_avx2;         /* 1 = AVX2 path is optimal */
    int use_avx512;       /* 1 = AVX-512 path is optimal (no throttle) */
    int use_neon;         /* 1 = NEON path is optimal */
    int use_gpu;          /* 1 = GPU offload is available */
    int omp_num_threads;  /* recommended thread count */
} rac_hw_profile;

/* ── HAL lifecycle ──────────────────────────────────────────────────────── */

/*
 * rac_hal_init: Probe hardware and build capability profile.
 * Call once at program startup. Not thread-safe.
 * Returns RAC_OK on success.
 */
rac_status rac_hal_init(void);

/*
 * rac_hal_shutdown: Release HAL resources.
 */
void rac_hal_shutdown(void);

/*
 * rac_hal_profile: Get the detected hardware profile.
 * Returns NULL if rac_hal_init() was not called.
 */
const rac_hw_profile* rac_hal_profile(void);

/*
 * rac_hal_print_profile: Print human-readable hardware summary.
 */
void rac_hal_print_profile(void);

/* ── Override interface ─────────────────────────────────────────────────── */

/*
 * rac_hal_override: Let advanced users force specific dispatch decisions.
 * Pass NULL fields to keep auto-detected values.
 */
typedef struct {
    int force_avx2;       /* -1=auto, 0=disable, 1=force */
    int force_avx512;
    int force_neon;
    int force_tile_size;  /* 0=auto, >0=force this tile */
    int force_threads;    /* 0=auto, >0=force this count */
} rac_hal_override;

void rac_hal_set_override(const rac_hal_override *ovr);

/* ── Auto-dispatched operations ─────────────────────────────────────────── */
/*
 * These functions automatically route to the fastest available backend
 * based on the HAL profile. They select:
 *   - SIMD path (scalar / AVX2 / AVX-512 / NEON)
 *   - Tile size (from cache topology)
 *   - Thread count (from core count + NUMA)
 *   - GPU offload (if beneficial for the problem size)
 */

rac_status rac_hal_sgemm(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta);

rac_status rac_hal_matmul(
    const float *A, const float *B, float *C,
    int M, int N, int K);

rac_status rac_hal_fused_linear(
    const float *input, const float *weight, const float *bias,
    float *output, int M, int N, int K,
    rac_activation act);

void rac_hal_relu(const float *x, float *out, int n);
void rac_hal_gelu(const float *x, float *out, int n);
void rac_hal_silu(const float *x, float *out, int n);

#ifdef __cplusplus
}
#endif

#endif /* RAC_HAL_H */
