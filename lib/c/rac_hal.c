/*
 * rac_hal.c — RAC Hardware Abstraction Layer Implementation
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Build: cc -O3 -march=native -fopenmp rac_hal.c rac_cpu.c rac_avx2.c -lm
 */

#include "rac_hal.h"
#include "rac_avx2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define RAC_X86 1
#include <cpuid.h>
#else
#define RAC_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define RAC_ARM64 1
#else
#define RAC_ARM64 0
#endif

/* ── Global state ───────────────────────────────────────────────────────── */

static rac_hw_profile g_profile;
static int g_initialized = 0;
static rac_hal_override g_override = {-1, -1, -1, 0, 0};

/* ── x86 CPUID probing ──────────────────────────────────────────────────── */

#if RAC_X86
static void _probe_x86(rac_hw_profile *p) {
    unsigned int eax, ebx, ecx, edx;

    /* Brand string */
    for (int i = 0; i < 3; i++) {
        __cpuid(0x80000002 + i, eax, ebx, ecx, edx);
        memcpy(p->cpu_name + i * 16 + 0,  &eax, 4);
        memcpy(p->cpu_name + i * 16 + 4,  &ebx, 4);
        memcpy(p->cpu_name + i * 16 + 8,  &ecx, 4);
        memcpy(p->cpu_name + i * 16 + 12, &edx, 4);
    }
    p->cpu_name[48] = '\0';

    /* Feature flags */
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    if (edx & (1 << 26)) p->simd_flags |= RAC_SIMD_SSE2;
    if (ecx & (1 << 19)) p->simd_flags |= RAC_SIMD_SSE4;
    if (ecx & (1 << 28)) p->simd_flags |= RAC_SIMD_AVX;
    if (ecx & (1 << 12)) p->simd_flags |= RAC_SIMD_FMA;

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    if (ebx & (1 << 5))  p->simd_flags |= RAC_SIMD_AVX2;
    if (ebx & (1 << 16)) p->simd_flags |= RAC_SIMD_AVX512F;
    if (ebx & (1 << 31)) p->simd_flags |= RAC_SIMD_AVX512VL;

    /* Detect microarchitecture from family/model */
    __cpuid(1, eax, ebx, ecx, edx);
    int family = ((eax >> 8) & 0xF) + ((eax >> 20) & 0xFF);
    int model  = ((eax >> 4) & 0xF) | ((eax >> 12) & 0xF0);

    /* AMD */
    if (family == 0x19) {
        if (model <= 0x0F) p->uarch = RAC_UARCH_ZEN3;
        else if (model >= 0x10 && model <= 0x1F) p->uarch = RAC_UARCH_ZEN3;
        else if (model >= 0x60) p->uarch = RAC_UARCH_ZEN4;
        else p->uarch = RAC_UARCH_ZEN3;
    } else if (family == 0x1A) {
        p->uarch = RAC_UARCH_ZEN5;
    }
    /* Intel (simplified) */
    else if (family == 0x06) {
        if (model == 0x8E || model == 0x9E || model == 0x55) p->uarch = RAC_UARCH_SKYLAKE;
        else if (model == 0x6A || model == 0x6C) p->uarch = RAC_UARCH_ICELAKE;
        else if (model == 0x97 || model == 0x9A) p->uarch = RAC_UARCH_ALDERLAKE;
        else if (model == 0x8F) p->uarch = RAC_UARCH_SAPPHIRERAPIDS;
    }

    /* AVX-512 throttling detection */
    p->avx512_throttle = 0;
    if (p->uarch == RAC_UARCH_ZEN3 && (p->simd_flags & RAC_SIMD_AVX512F)) {
        p->avx512_throttle = 1; /* Zen3 AVX-512 is 256-bit doubled, slower */
    }
    if (p->uarch == RAC_UARCH_SKYLAKE && (p->simd_flags & RAC_SIMD_AVX512F)) {
        p->avx512_throttle = 1; /* Skylake throttles frequency on AVX-512 */
    }

    /* Cache detection via CPUID leaf 4 */
    for (int idx = 0; idx < 16; idx++) {
        __cpuid_count(4, idx, eax, ebx, ecx, edx);
        int type = eax & 0x1F;
        if (type == 0) break;

        int level = (eax >> 5) & 0x7;
        int line_size = (ebx & 0xFFF) + 1;
        int partitions = ((ebx >> 12) & 0x3FF) + 1;
        int ways = ((ebx >> 22) & 0x3FF) + 1;
        int sets = ecx + 1;
        int size_kb = (ways * partitions * line_size * sets) / 1024;

        if (level == 1 && (type == 1)) { /* L1 data */
            p->cache.l1d_size_kb = size_kb;
            p->cache.l1d_line_size = line_size;
        } else if (level == 2) {
            p->cache.l2_size_kb = size_kb;
        } else if (level == 3) {
            p->cache.l3_size_kb = size_kb;
            p->cache.l3_shared_cores = ((eax >> 14) & 0xFFF) + 1;
        }
    }
}
#endif /* RAC_X86 */

#if RAC_ARM64
static void _probe_arm64(rac_hw_profile *p) {
    strncpy(p->cpu_name, "ARM64", sizeof(p->cpu_name));
    p->simd_flags |= RAC_SIMD_NEON;

    /* SVE detection via /proc/cpuinfo on Linux */
    #ifdef __linux__
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "sve")) p->simd_flags |= RAC_SIMD_SVE;
            if (strstr(line, "sve2")) p->simd_flags |= RAC_SIMD_SVE2;
        }
        fclose(f);
    }
    #endif

    /* Default cache sizes for modern ARM */
    if (p->cache.l1d_size_kb == 0) {
        p->cache.l1d_size_kb = 64;
        p->cache.l1d_line_size = 64;
        p->cache.l2_size_kb = 512;
        p->cache.l3_size_kb = 4096;
    }
}
#endif /* RAC_ARM64 */

/* ── Core/thread counting ───────────────────────────────────────────────── */

static void _probe_cores(rac_hw_profile *p) {
#ifdef _OPENMP
    p->num_logical_cores = omp_get_max_threads();
#elif defined(_SC_NPROCESSORS_ONLN)
    p->num_logical_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#else
    p->num_logical_cores = 1;
#endif
    /* Estimate physical cores (assume SMT-2) */
    p->num_physical_cores = p->num_logical_cores;  /* conservative default */

#ifdef __linux__
    /* Try to detect from /sys */
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    if (f) {
        char buf[64];
        if (fgets(buf, sizeof(buf), f)) {
            if (strchr(buf, ',') || strchr(buf, '-')) {
                p->num_physical_cores = p->num_logical_cores / 2;
            }
        }
        fclose(f);
    }
#endif

    /* NUMA */
    p->numa.num_nodes = 1;
    p->numa.num_sockets = 1;
    p->numa.cores_per_node = p->num_physical_cores;

#ifdef __linux__
    FILE *fn = fopen("/sys/devices/system/node/possible", "r");
    if (fn) {
        char buf[64];
        if (fgets(buf, sizeof(buf), fn)) {
            /* Parse "0-N" format */
            char *dash = strchr(buf, '-');
            if (dash) p->numa.num_nodes = atoi(dash + 1) + 1;
        }
        fclose(fn);
    }
    if (p->numa.num_nodes > 1)
        p->numa.cores_per_node = p->num_physical_cores / p->numa.num_nodes;
#endif
}

/* ── Optimal tile size calculation ──────────────────────────────────────── */

static void _compute_optimal_tiles(rac_hw_profile *p) {
    /*
     * Tile sizing strategy:
     *   - L1 tile: 3 tiles (A, B, C sub-blocks) must fit in L1d
     *     tile^2 * sizeof(float) * 3 <= L1d_size
     *     tile = sqrt(L1d_size / (3 * 4))
     *   - L2 tile: for outer blocking
     *     tile^2 * sizeof(float) * 3 <= L2_size
     */
    int l1_bytes = p->cache.l1d_size_kb * 1024;
    int l2_bytes = p->cache.l2_size_kb * 1024;

    if (l1_bytes <= 0) l1_bytes = 32 * 1024;  /* default 32KB */
    if (l2_bytes <= 0) l2_bytes = 256 * 1024;

    /* L1 tile: leave 25% for other data */
    int l1_usable = (l1_bytes * 3) / 4;
    int tile_l1 = (int)sqrtf((float)l1_usable / (3.0f * sizeof(float)));
    /* Round down to multiple of 8 for SIMD alignment */
    tile_l1 = (tile_l1 / 8) * 8;
    if (tile_l1 < 16) tile_l1 = 16;
    if (tile_l1 > 128) tile_l1 = 128;
    p->optimal_tile_sgemm = tile_l1;

    /* L2 tile */
    int l2_usable = (l2_bytes * 3) / 4;
    int tile_l2 = (int)sqrtf((float)l2_usable / (3.0f * sizeof(float)));
    tile_l2 = (tile_l2 / 8) * 8;
    if (tile_l2 < tile_l1) tile_l2 = tile_l1;
    if (tile_l2 > 512) tile_l2 = 512;
    p->optimal_tile_sgemm_l2 = tile_l2;
}

/* ── Dispatch decisions ─────────────────────────────────────────────────── */

static void _compute_dispatch(rac_hw_profile *p) {
    /* Default: no SIMD acceleration */
    p->use_avx2 = 0;
    p->use_avx512 = 0;
    p->use_neon = 0;
    p->preferred_simd_width = 16; /* SSE/NEON baseline */

    /* AVX2 + FMA: best on most x86 */
    if ((p->simd_flags & RAC_SIMD_AVX2) && (p->simd_flags & RAC_SIMD_FMA)) {
        p->use_avx2 = 1;
        p->preferred_simd_width = 32;
    }

    /* AVX-512: only if no throttle */
    if ((p->simd_flags & RAC_SIMD_AVX512F) && !p->avx512_throttle) {
        p->use_avx512 = 1;
        p->preferred_simd_width = 64;
    }

    /* ARM NEON */
    if (p->simd_flags & RAC_SIMD_NEON) {
        p->use_neon = 1;
        p->preferred_simd_width = 16;
    }
    if (p->simd_flags & RAC_SIMD_SVE) {
        p->preferred_simd_width = 32; /* SVE is variable, assume 256-bit */
    }

    /* Thread count: use physical cores, not logical (SMT hurts matmul) */
    p->omp_num_threads = p->num_physical_cores;
    if (p->omp_num_threads <= 0) p->omp_num_threads = 1;

    /* Apply overrides */
    if (g_override.force_avx2 >= 0) p->use_avx2 = g_override.force_avx2;
    if (g_override.force_avx512 >= 0) p->use_avx512 = g_override.force_avx512;
    if (g_override.force_neon >= 0) p->use_neon = g_override.force_neon;
    if (g_override.force_threads > 0) p->omp_num_threads = g_override.force_threads;
    /* Tile override applied after dispatch so it persists */
    if (g_override.force_tile_size > 0) {
        p->optimal_tile_sgemm = g_override.force_tile_size;
        p->optimal_tile_sgemm_l2 = g_override.force_tile_size;
    }
}

/* ── Public API ─────────────────────────────────────────────────────────── */

rac_status rac_hal_init(void) {
    /* Preserve overrides across re-init */
    rac_hal_override saved_ovr = g_override;
    memset(&g_profile, 0, sizeof(g_profile));

    #if RAC_X86
    _probe_x86(&g_profile);
    #endif

    #if RAC_ARM64
    _probe_arm64(&g_profile);
    #endif

    _probe_cores(&g_profile);
    _compute_optimal_tiles(&g_profile);
    g_override = saved_ovr;  /* restore overrides */
    _compute_dispatch(&g_profile);  /* applies overrides last */

    #ifdef _OPENMP
    omp_set_num_threads(g_profile.omp_num_threads);
    #endif

    g_initialized = 1;
    return RAC_OK;
}

void rac_hal_shutdown(void) {
    g_initialized = 0;
}

const rac_hw_profile* rac_hal_profile(void) {
    return g_initialized ? &g_profile : NULL;
}

void rac_hal_set_override(const rac_hal_override *ovr) {
    if (ovr) {
        g_override = *ovr;
        if (g_initialized) {
            _compute_optimal_tiles(&g_profile);
            _compute_dispatch(&g_profile);  /* overrides applied last */
        }
    }
}

void rac_hal_print_profile(void) {
    const rac_hw_profile *p = &g_profile;
    if (!g_initialized) { printf("HAL not initialized\n"); return; }

    printf("RAC HAL — Hardware Profile\n");
    printf("══════════════════════════════════════════════════\n");
    printf("CPU:           %s\n", p->cpu_name);
    printf("Cores:         %d physical, %d logical\n",
           p->num_physical_cores, p->num_logical_cores);
    printf("Microarch:     %d\n", p->uarch);
    printf("SIMD:          ");
    if (p->simd_flags & RAC_SIMD_AVX512F) printf("AVX-512%s ", p->avx512_throttle ? "(throttled)" : "");
    if (p->simd_flags & RAC_SIMD_AVX2) printf("AVX2 ");
    if (p->simd_flags & RAC_SIMD_FMA) printf("FMA ");
    if (p->simd_flags & RAC_SIMD_AVX) printf("AVX ");
    if (p->simd_flags & RAC_SIMD_SSE4) printf("SSE4 ");
    if (p->simd_flags & RAC_SIMD_NEON) printf("NEON ");
    if (p->simd_flags & RAC_SIMD_SVE) printf("SVE ");
    printf("\n");
    printf("SIMD width:    %d bytes (%d floats)\n",
           p->preferred_simd_width, p->preferred_simd_width / 4);
    printf("Cache:         L1d=%dKB (line=%dB)  L2=%dKB  L3=%dKB\n",
           p->cache.l1d_size_kb, p->cache.l1d_line_size,
           p->cache.l2_size_kb, p->cache.l3_size_kb);
    printf("NUMA:          %d nodes, %d sockets\n",
           p->numa.num_nodes, p->numa.num_sockets);
    printf("Tile (L1):     %d\n", p->optimal_tile_sgemm);
    printf("Tile (L2):     %d\n", p->optimal_tile_sgemm_l2);
    printf("Threads:       %d\n", p->omp_num_threads);
    printf("Dispatch:      %s%s%s%s\n",
           p->use_avx512 ? "AVX-512" : (p->use_avx2 ? "AVX2+FMA" : (p->use_neon ? "NEON" : "scalar")),
           p->use_gpu ? " + GPU" : "",
           p->avx512_throttle ? " (AVX-512 throttled, using AVX2)" : "",
           "");
    printf("══════════════════════════════════════════════════\n");
}

/* ── Auto-dispatched operations ─────────────────────────────────────────── */

static rac_config _hal_config(void) {
    rac_config cfg;
    cfg.num_threads = g_profile.omp_num_threads;
    cfg.tile_size = g_profile.optimal_tile_sgemm;
    cfg.cordic_iters = RAC_ITERS;
    return cfg;
}

rac_status rac_hal_sgemm(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    if (!g_initialized) rac_hal_init();

    rac_config cfg = _hal_config();

    /* Dispatch to fastest path */
    if (g_profile.use_avx2 || g_profile.use_avx512) {
        return rac_sgemm_avx2(A, B, C, M, N, K, alpha, beta, &cfg);
    }
    return rac_sgemm(A, B, C, M, N, K, alpha, beta, &cfg);
}

rac_status rac_hal_matmul(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    return rac_hal_sgemm(A, B, C, M, N, K, 1.0f, 0.0f);
}

rac_status rac_hal_fused_linear(
    const float *input, const float *weight, const float *bias,
    float *output, int M, int N, int K,
    rac_activation act)
{
    if (!g_initialized) rac_hal_init();

    rac_config cfg = _hal_config();

    if (g_profile.use_avx2 || g_profile.use_avx512) {
        return rac_fused_linear_avx2(input, weight, bias, output,
                                      M, N, K, act, &cfg);
    }
    return rac_fused_linear(input, weight, bias, output, M, N, K, act, &cfg);
}

void rac_hal_relu(const float *x, float *out, int n) {
    if (!g_initialized) rac_hal_init();
    if (g_profile.use_avx2) rac_relu_avx2(x, out, n);
    else rac_relu(x, out, n);
}

void rac_hal_gelu(const float *x, float *out, int n) {
    if (!g_initialized) rac_hal_init();
    if (g_profile.use_avx2) rac_gelu_avx2(x, out, n);
    else rac_gelu(x, out, n);
}

void rac_hal_silu(const float *x, float *out, int n) {
    if (!g_initialized) rac_hal_init();
    if (g_profile.use_avx2) rac_silu_avx2(x, out, n);
    else rac_silu(x, out, n);
}
