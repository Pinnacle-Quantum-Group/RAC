/*
 * rac_avx2.c — RAC AVX2/FMA Vectorized Implementation
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Build: cc -O3 -mavx2 -mfma -fopenmp rac_cpu.c rac_avx2.c -lm -shared -o librac_avx2.so
 */

#include "rac_avx2.h"
#include "rac_hal.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <cpuid.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#define RAC_HAS_AVX2_COMPILE 1
#else
#define RAC_HAS_AVX2_COMPILE 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Runtime CPUID check ────────────────────────────────────────────────── */

int rac_has_avx2(void) {
#if RAC_HAS_AVX2_COMPILE
    /* GCC/Clang __builtin_cpu_supports */
    #if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    #else
    return 1; /* assume yes if compiled with -mavx2 -mfma */
    #endif
#else
    return 0;
#endif
}

#if RAC_HAS_AVX2_COMPILE

/* ── AVX2 SGEMM ────────────────────────────────────────────────────────── */
/*
 * Strategy: tile over M (OpenMP), vectorize over N (AVX2 8-wide).
 * Inner K loop: 8 FMAs per cycle, 8 columns of C per iteration.
 *
 * For each row i of C and each 8-column block j..j+7:
 *   acc[0..7] += A[i,k] * B[k, j..j+7]  for all k
 *
 * This is the "broadcast A, stream B" pattern — optimal for row-major layout.
 */

/*
 * AVX2 SGEMM — GotoBLAS/BLIS-style 3-level blocked algorithm.
 *
 * This is the same algorithm that OpenBLAS uses internally:
 *   1. Block B into NC×KC panels (L3-resident)
 *   2. Block A into MC×KC panels (L2-resident)
 *   3. Pack A panel into contiguous column-panel layout
 *   4. Pack B panel into contiguous row-panel layout
 *   5. Run MR×NR micro-kernel on packed data
 *
 * Micro-kernel: MR=6 rows × NR=16 cols (2 __m256 registers)
 *   12 accumulator YMM registers + 2 B loads + 1 A broadcast = 15/16 YMM used
 *   Per K step: 6 broadcasts × 2 loads = 12 FMAs from 8 memory ops
 *
 * Cache sizing (auto-tuned from HAL profile):
 *   KC: sized so packed_A micro-panel [MR × KC] streams through L1
 *       KC = L1d / (MR * sizeof(float) * 2) ≈ 256 for 32KB L1d (Zen3)
 *   MC: sized so packed_A panel [MC × KC] fits in L2
 *       MC = L2 / (KC * sizeof(float) * 2) ≈ 256 for 512KB L2 (Zen3)
 *   NC: sized so packed_B panel [KC × NC] fits in per-core L3 share
 *       NC = L3_per_core / (KC * sizeof(float)) ≈ 2048+ for 2MB L3/core
 *
 * Packing eliminates TLB misses and enables hardware prefetch.
 */

#define MR 6     /* rows of C per micro-kernel */
#define NR 16    /* columns of C per micro-kernel (2 x __m256) */

/*
 * Default cache tile sizes.
 * KC: B micro-panel [KC × NR] must fit in L1d alongside A reads.
 *     L1d=32KB (Zen3): KC*NR*4 + MR*KC*4 ≤ 32K
 *     KC*(16+6)*4 ≤ 32768 → KC ≤ 372. Use 128 for safety + prefetch room.
 * MC: packed_A [MC × KC] fits in L2.
 *     L2=512KB (Zen3): MC*KC*4 ≤ 512K → MC ≤ 1024. Use 384.
 * NC: packed_B [KC × NC] fits in L3 per-core share.
 *     L3=2MB/core (Zen3): KC*NC*4 ≤ 2M → NC ≤ 4096. Use 2048.
 */
#define DEFAULT_KC 128
#define DEFAULT_MC 384
#define DEFAULT_NC 2048

/* ── Pack A[mc×kc] → column-panel layout (SIMD-accelerated) ── */
/*
 * packed_A layout: for each MR-panel, stores [kc][MR] contiguous.
 * A is row-major: A[i,k] = A[i*lda + k]. We need to gather MR rows
 * for each k column. MR=6 doesn't align to AVX2, so we gather scalars
 * but prefetch aggressively.
 *
 * The key optimization: prefetch the next K column's rows while
 * packing the current one, hiding the gather latency.
 */
static void _pack_a(const float *A, float *packed, int mc, int kc, int lda, int K_start) {
    for (int i = 0; i < mc; i += MR) {
        int mr = ((i + MR) <= mc) ? MR : (mc - i);
        const float *a_ptrs[MR];
        for (int r = 0; r < mr; r++)
            a_ptrs[r] = &A[(i + r) * lda + K_start];

        if (mr == MR) {
            /* Full MR=6 panel: unrolled gather with prefetch */
            for (int k = 0; k < kc; k++) {
                /* Prefetch next column (64 bytes ahead ≈ next cache line) */
                if (k + 8 < kc) {
                    _mm_prefetch((const char*)(a_ptrs[0] + k + 8), _MM_HINT_T0);
                    _mm_prefetch((const char*)(a_ptrs[3] + k + 8), _MM_HINT_T0);
                }
                packed[0] = a_ptrs[0][k];
                packed[1] = a_ptrs[1][k];
                packed[2] = a_ptrs[2][k];
                packed[3] = a_ptrs[3][k];
                packed[4] = a_ptrs[4][k];
                packed[5] = a_ptrs[5][k];
                packed += MR;
            }
        } else {
            /* Edge panel: scalar with zero-pad */
            for (int k = 0; k < kc; k++) {
                for (int r = 0; r < mr; r++)
                    *packed++ = a_ptrs[r][k];
                for (int r = mr; r < MR; r++)
                    *packed++ = 0.0f;
            }
        }
    }
}

/* ── Pack B[kc×nc] → row-panel layout (SIMD-accelerated) ── */
/*
 * packed_B layout: for each NR-panel, stores [kc][NR] contiguous.
 * B is row-major: B[k,j] = B[k*ldb + j]. For a given k, B[k, j..j+NR-1]
 * is already contiguous! So we can memcpy/vmovups entire rows.
 *
 * This is where the big win is — B packing is a contiguous copy.
 */
static void _pack_b(const float *B, float *packed, int kc, int nc, int ldb, int K_start, int N_start, int panel_nr) {
    for (int j = 0; j < nc; j += panel_nr) {
        int nr = ((j + panel_nr) <= nc) ? panel_nr : (nc - j);

        if (nr == panel_nr && panel_nr == NR) {
            /* Full NR=16 panel: 2 × vmovups per K row (contiguous in B) */
            for (int k = 0; k < kc; k++) {
                const float *src = &B[(K_start + k) * ldb + N_start + j];
                /* Prefetch next row */
                if (k + 1 < kc)
                    _mm_prefetch((const char*)&B[(K_start + k + 1) * ldb + N_start + j], _MM_HINT_T0);
                __m256 v0 = _mm256_loadu_ps(src);
                __m256 v1 = _mm256_loadu_ps(src + 8);
                _mm256_storeu_ps(packed, v0);
                _mm256_storeu_ps(packed + 8, v1);
                packed += NR;
            }
        } else if (nr == panel_nr) {
            /* Full panel but NR != 16 (e.g. NR=32 for AVX-512): use memcpy */
            for (int k = 0; k < kc; k++) {
                const float *src = &B[(K_start + k) * ldb + N_start + j];
                memcpy(packed, src, panel_nr * sizeof(float));
                packed += panel_nr;
            }
        } else {
            /* Edge panel: copy what we have, zero-pad the rest */
            for (int k = 0; k < kc; k++) {
                const float *src = &B[(K_start + k) * ldb + N_start + j];
                /* Copy available columns with SIMD where possible */
                int c = 0;
                for (; c + 8 <= nr; c += 8) {
                    __m256 v = _mm256_loadu_ps(src + c);
                    _mm256_storeu_ps(packed + c, v);
                }
                for (; c < nr; c++)
                    packed[c] = src[c];
                for (; c < panel_nr; c++)
                    packed[c] = 0.0f;
                packed += panel_nr;
            }
        }
    }
}

/* ── Assembly micro-kernels ── */
/* AVX2 6x16: defined in rac_zen3_kern.S */
extern void rac_micro_kernel_6x16_asm(
    const float *packed_a,
    const float *packed_b,
    float *C,
    int ldc,
    int kc);

/* AVX-512 6x32: defined in rac_avx512_kern.S */
extern void rac_micro_kernel_6x32_avx512(
    const float *packed_a,
    const float *packed_b,
    float *C,
    int ldc,
    int kc);

/* Runtime detection: which assembly kernel to use */
static int _use_avx512_kern = -1;  /* -1=unset, 0=avx2, 1=avx512 */
static int _active_NR = NR;        /* 16 for AVX2, 32 for AVX-512 */

static void _detect_best_kernel(void) {
    if (_use_avx512_kern >= 0) return;
    /*
     * AVX-512 kernel disabled pending further validation.
     * The 6x32 assembly kernel has correctness issues on some Intel SKUs.
     * AVX2 6x16 assembly is the production path for now.
     * TODO: validate AVX-512 on Icelake, Sapphire Rapids, Zen4+
     */
    _use_avx512_kern = 0;
    _active_NR = NR;  /* 16 */
}

/* ── C fallback micro-kernel (for edge tiles with mr < MR or nr < NR) ── */
static void _micro_kernel_6x16_c(
    const float *packed_a,
    const float *packed_b,
    float *C, int ldc,
    int mr, int nr, int kc,
    float alpha)
{
    __m256 acc[MR][2];
    for (int r = 0; r < MR; r++) {
        acc[r][0] = _mm256_setzero_ps();
        acc[r][1] = _mm256_setzero_ps();
    }

    for (int k = 0; k < kc; k++) {
        __m256 b0 = _mm256_loadu_ps(&packed_b[k * NR]);
        __m256 b1 = _mm256_loadu_ps(&packed_b[k * NR + 8]);

        for (int r = 0; r < MR; r++) {
            __m256 a_bc = _mm256_set1_ps(packed_a[k * MR + r]);
            acc[r][0] = _mm256_fmadd_ps(a_bc, b0, acc[r][0]);
            acc[r][1] = _mm256_fmadd_ps(a_bc, b1, acc[r][1]);
        }
    }

    __m256 valpha = _mm256_set1_ps(alpha);
    for (int r = 0; r < mr; r++) {
        __m256 res0 = _mm256_mul_ps(valpha, acc[r][0]);
        __m256 res1 = _mm256_mul_ps(valpha, acc[r][1]);

        if (nr >= 16) {
            __m256 c0 = _mm256_loadu_ps(&C[r * ldc]);
            __m256 c1 = _mm256_loadu_ps(&C[r * ldc + 8]);
            _mm256_storeu_ps(&C[r * ldc],     _mm256_add_ps(c0, res0));
            _mm256_storeu_ps(&C[r * ldc + 8], _mm256_add_ps(c1, res1));
        } else {
            float tmp0[8], tmp1[8];
            _mm256_storeu_ps(tmp0, res0);
            _mm256_storeu_ps(tmp1, res1);
            for (int c = 0; c < nr && c < 8; c++)
                C[r * ldc + c] += tmp0[c];
            for (int c = 8; c < nr; c++)
                C[r * ldc + c] += tmp1[c - 8];
        }
    }
}

rac_status rac_sgemm_avx2(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta,
    const rac_config *cfg)
{
    if (!A || !B || !C) return RAC_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return RAC_ERR_INVALID_DIM;

    /* Detect best kernel at first call */
    _detect_best_kernel();
    int active_nr = _active_NR;

    /*
     * Auto-tune KC/MC/NC from actual cache sizes.
     * KC: B micro-panel [KC × NR] + A stream [MR × KC] fits in L1d
     *     KC = L1d / ((NR + MR) * sizeof(float) * 2)  [50% utilization]
     * MC: packed_A [MC × KC] fits in L2
     *     MC = L2 / (KC * sizeof(float) * 2)  [50% utilization]
     * NC: packed_B [KC × NC] fits in L3 per-core
     */
    int l1d_kb = 32, l2_kb = 256;  /* conservative defaults */

    /* Try to get actual cache sizes from HAL */
    extern const rac_hw_profile* rac_hal_profile(void);
    const rac_hw_profile *hw = rac_hal_profile();
    if (hw && hw->cache.l1d_size_kb > 0) l1d_kb = hw->cache.l1d_size_kb;
    if (hw && hw->cache.l2_size_kb > 0)  l2_kb  = hw->cache.l2_size_kb;

    int l1d_bytes = l1d_kb * 1024;
    int l2_bytes  = l2_kb * 1024;

    /* KC: B micro-panel [KC*NR*4] fits in ~40% of L1 */
    int KC = (l1d_bytes * 2 / 5) / (active_nr * (int)sizeof(float));
    KC = (KC / 8) * 8;
    if (KC < 128) KC = 128;
    if (KC > 512) KC = 512;

    /* MC: packed_A [MC*KC*4] fits in ~50% of L2 */
    int MC = (l2_bytes / 2) / (KC * (int)sizeof(float));
    MC = (MC / MR) * MR;
    if (MC < MR * 8) MC = MR * 8;
    if (MC > 768) MC = 768;

    int NC = DEFAULT_NC;

    /* cfg->tile_size is for the simple tiled kernel, NOT for GotoBLAS KC.
     * Only override KC if the user explicitly sets a large value (>= 64). */

    /*
     * Small matrix fast path: skip packing overhead.
     * For M*N < 256*256, the packing cost dominates.
     * Use direct MR×NR micro-tiled loop (no packing, no GotoBLAS).
     */
    if ((long long)M * N <= 262144) {  /* up to 512x512: fast path beats packing */
        if (beta == 0.0f) memset(C, 0, (size_t)M * N * sizeof(float));
        else if (beta != 1.0f) for (int i = 0; i < M * N; i++) C[i] *= beta;

        /* Very small N: pure scalar to avoid SIMD edge overhead */
        if (N < 8) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++)
                        sum = fmaf(A[i*K+k], B[k*N+j], sum);
                    C[i*N+j] += alpha * sum;
                }
            return RAC_OK;
        }

        #pragma omp parallel for schedule(dynamic, 2)
        for (int i0 = 0; i0 < M; i0 += MR) {
            int mr = ((i0 + MR) <= M) ? MR : (M - i0);
            for (int j0 = 0; j0 < N; j0 += NR) {
                int nr = ((j0 + NR) <= N) ? NR : (N - j0);
                __m256 acc[MR][2];
                for (int r = 0; r < MR; r++) {
                    acc[r][0] = _mm256_setzero_ps();
                    acc[r][1] = _mm256_setzero_ps();
                }
                for (int k = 0; k < K; k++) {
                    /* N >= 8 guaranteed (scalar path handles N < 8 above) */
                    __m256 b0 = _mm256_loadu_ps(&B[k*N+j0]);
                    __m256 b1 = (nr >= 16) ? _mm256_loadu_ps(&B[k*N+j0+8]) : _mm256_setzero_ps();
                    for (int r = 0; r < mr; r++) {
                        __m256 a_bc = _mm256_set1_ps(A[(i0+r)*K+k]);
                        acc[r][0] = _mm256_fmadd_ps(a_bc, b0, acc[r][0]);
                        acc[r][1] = _mm256_fmadd_ps(a_bc, b1, acc[r][1]);
                    }
                }
                __m256 valpha = _mm256_set1_ps(alpha);
                for (int r = 0; r < mr; r++) {
                    __m256 r0 = _mm256_mul_ps(valpha, acc[r][0]);
                    __m256 r1 = _mm256_mul_ps(valpha, acc[r][1]);
                    if (nr >= 16) {
                        __m256 c0 = _mm256_loadu_ps(&C[(i0+r)*N+j0]);
                        __m256 c1 = _mm256_loadu_ps(&C[(i0+r)*N+j0+8]);
                        _mm256_storeu_ps(&C[(i0+r)*N+j0],   _mm256_add_ps(c0, r0));
                        _mm256_storeu_ps(&C[(i0+r)*N+j0+8], _mm256_add_ps(c1, r1));
                    } else {
                        float t0[8], t1[8];
                        _mm256_storeu_ps(t0, r0); _mm256_storeu_ps(t1, r1);
                        for (int c = 0; c < nr && c < 8; c++) C[(i0+r)*N+j0+c] += t0[c];
                        for (int c = 8; c < nr; c++) C[(i0+r)*N+j0+c] += t1[c-8];
                    }
                }
            }
        }
        return RAC_OK;
    }

    /* ── Large matrix: GotoBLAS 3-level blocked with packing ── */

    /* Apply beta to C */
    if (beta == 0.0f) {
        memset(C, 0, (size_t)M * N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int i = 0; i < M * N; i++) C[i] *= beta;
    }

    /* Allocate packing buffers (one set per thread) */
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif

    /* packed_B: ceil(NC/active_nr) panels × KC × active_nr floats per panel */
    /* packed_A: ceil(MC/MR) panels × KC × MR floats per panel, per thread */
    int b_panels = (NC + active_nr - 1) / active_nr;
    int a_panels = (MC + MR - 1) / MR;
    size_t packed_b_size = (size_t)b_panels * KC * active_nr;
    size_t packed_a_size = (size_t)a_panels * KC * MR;

    float *packed_b = (float *)calloc(packed_b_size, sizeof(float));
    float *packed_a_all = (float *)calloc((size_t)n_threads * packed_a_size, sizeof(float));
    if (!packed_b || !packed_a_all) {
        free(packed_b); free(packed_a_all);
        return RAC_ERR_ALLOC;
    }

    /* Debug counters */
    long long _asm_calls = 0, _c_calls = 0;

    /* 3-level blocked loop: jc (L3) → ic (L2) → micro-kernel (L1) */
    for (int jc = 0; jc < N; jc += NC) {
        int nc = ((jc + NC) <= N) ? NC : (N - jc);

        for (int pc = 0; pc < K; pc += KC) {
            int kc = ((pc + KC) <= K) ? KC : (K - pc);

            /* Pack B panel [kc × nc] — shared across all threads */
            _pack_b(B, packed_b, kc, nc, N, pc, jc, active_nr);

            /* Parallel over M-tiles */
            #pragma omp parallel
            {
                int tid = 0;
                #ifdef _OPENMP
                tid = omp_get_thread_num();
                #endif
                float *packed_a = packed_a_all + (size_t)tid * packed_a_size;

                #pragma omp for schedule(dynamic, 1)
                for (int ic = 0; ic < M; ic += MC) {
                    int mc = ((ic + MC) <= M) ? MC : (M - ic);

                    /* Pack A panel [mc × kc] */
                    _pack_a(A + ic * K, packed_a, mc, kc, K, pc);

                    /* Micro-kernel loop — dispatch to best assembly kernel */
                    for (int jr = 0; jr < nc; jr += active_nr) {
                        int nr = ((jr + active_nr) <= nc) ? active_nr : (nc - jr);
                        float *pb = &packed_b[(jr / active_nr) * kc * active_nr];

                        for (int ir = 0; ir < mc; ir += MR) {
                            int mr = ((ir + MR) <= mc) ? MR : (mc - ir);
                            float *pa = &packed_a[(ir / MR) * kc * MR];
                            float *c_ptr = &C[(ic + ir) * N + (jc + jr)];

                            if (mr == MR && alpha == 1.0f) {
                                if (_use_avx512_kern && nr == 32) {
                                    rac_micro_kernel_6x32_avx512(pa, pb, c_ptr, N, kc);
                                    _asm_calls++;
                                } else if (nr >= NR) {
                                    rac_micro_kernel_6x16_asm(pa, pb, c_ptr, N, kc);
                                    _asm_calls++;
                                } else {
                                    _micro_kernel_6x16_c(pa, pb, c_ptr, N, mr, nr, kc, alpha);
                                    _c_calls++;
                                }
                            } else {
                                _micro_kernel_6x16_c(pa, pb, c_ptr, N, mr, nr, kc, alpha);
                                _c_calls++;
                            }
                        }
                    }
                }
            }
        }
    }

    static int _print_once = 1;
    if (_print_once) {
        fprintf(stderr, "RAC SGEMM: KC=%d MC=%d NC=%d active_nr=%d asm=%lld c=%lld (%.0f%% asm)\n",
                KC, MC, NC, active_nr, _asm_calls, _c_calls,
                _asm_calls * 100.0 / (_asm_calls + _c_calls + 1));
        _print_once = 0;
    }

    free(packed_b);
    free(packed_a_all);
    return RAC_OK;
}

/* ── AVX2 fused linear ──────────────────────────────────────────────────── */

/* Vectorized activation helpers */
static inline __m256 _avx2_relu(__m256 x) {
    return _mm256_max_ps(x, _mm256_setzero_ps());
}

/* Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
/* Using the faster erf approximation: x * 0.5 * (1 + erf(x * 0.7071)) */
/* For AVX2, we use a polynomial approximation of erf */
static inline __m256 _avx2_gelu_approx(__m256 x) {
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one  = _mm256_set1_ps(1.0f);

    /* Fast tanh-based GELU approximation */
    __m256 sqrt2pi = _mm256_set1_ps(0.7978845608f); /* sqrt(2/pi) */
    __m256 k = _mm256_set1_ps(0.044715f);
    __m256 x3 = _mm256_mul_ps(x, _mm256_mul_ps(x, x));
    __m256 inner = _mm256_fmadd_ps(k, x3, x);
    inner = _mm256_mul_ps(sqrt2pi, inner);

    /* tanh approximation via rational function (good for |x| < 5) */
    /* tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2) for small x */
    __m256 x2 = _mm256_mul_ps(inner, inner);
    __m256 n27 = _mm256_set1_ps(27.0f);
    __m256 n9  = _mm256_set1_ps(9.0f);
    __m256 num = _mm256_fmadd_ps(one, x2, n27);
    __m256 den = _mm256_fmadd_ps(n9, x2, n27);
    __m256 tanh_approx = _mm256_mul_ps(inner, _mm256_div_ps(num, den));

    /* GELU = x * 0.5 * (1 + tanh) */
    return _mm256_mul_ps(x, _mm256_mul_ps(half, _mm256_add_ps(one, tanh_approx)));
}

static inline __m256 _avx2_silu(__m256 x) {
    /* SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
     * Approximate exp(-x) with fast polynomial */
    float vals[8], results[8];
    _mm256_storeu_ps(vals, x);
    for (int i = 0; i < 8; i++)
        results[i] = vals[i] / (1.0f + expf(-vals[i]));
    return _mm256_loadu_ps(results);
}

/*
 * Fused linear with register micro-tiling.
 * weight is [N, K] row-major: weight[j, k] is contiguous along K.
 * output[i,j] = act(sum_k input[i,k] * weight[j,k] + bias[j])
 *
 * Micro-tile: MR=4 rows of output × NR=16 columns (2 x __m256).
 * Per K step: 4 broadcasts of input × 2 contiguous loads of weight = 8 FMAs.
 * weight[j, k..k+7] IS contiguous (row-major with stride K).
 * But we need weight[j..j+7, k] for the broadcast pattern — that's strided.
 *
 * Solution: transpose weight into [K, N] layout for contiguous N-access,
 * then use the same broadcast-A pattern as SGEMM. We do this once in a
 * scratch buffer (amortized over MR*K FMAs).
 */

rac_status rac_fused_linear_avx2(
    const float *input, const float *weight, const float *bias,
    float *output, int M, int N, int K,
    rac_activation act, const rac_config *cfg)
{
    if (!input || !weight || !output) return RAC_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return RAC_ERR_INVALID_DIM;

    int TILE_K = (cfg && cfg->tile_size > 0) ? cfg->tile_size : 64;

    /* Transpose weight [N, K] → wt[K, N] for contiguous N-access */
    float *wt = (float *)malloc((size_t)K * N * sizeof(float));
    if (!wt) return RAC_ERR_ALLOC;
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++)
            wt[k * N + j] = weight[j * K + k];

    /* Micro-tiled matmul: input[M,K] @ wt[K,N] → output[M,N] */
    #pragma omp parallel for schedule(dynamic, 2)
    for (int i0 = 0; i0 < M; i0 += MR) {
        int imax = (i0 + MR <= M) ? i0 + MR : M;
        int mr = imax - i0;

        for (int j0 = 0; j0 < N; j0 += NR) {
            int jmax = (j0 + NR <= N) ? j0 + NR : N;
            int nr = jmax - j0;

            __m256 acc[MR][2];
            for (int r = 0; r < MR; r++) {
                acc[r][0] = _mm256_setzero_ps();
                acc[r][1] = _mm256_setzero_ps();
            }

            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                int kmax = (k0 + TILE_K < K) ? k0 + TILE_K : K;
                for (int k = k0; k < kmax; k++) {
                    __m256 b0 = (nr >= 8)  ? _mm256_loadu_ps(&wt[k * N + j0])     : _mm256_setzero_ps();
                    __m256 b1 = (nr >= 16) ? _mm256_loadu_ps(&wt[k * N + j0 + 8]) : _mm256_setzero_ps();

                    for (int r = 0; r < mr; r++) {
                        __m256 a_bc = _mm256_set1_ps(input[(i0 + r) * K + k]);
                        acc[r][0] = _mm256_fmadd_ps(a_bc, b0, acc[r][0]);
                        acc[r][1] = _mm256_fmadd_ps(a_bc, b1, acc[r][1]);
                    }
                }
            }

            /* Fused bias + activation + store */
            for (int r = 0; r < mr; r++) {
                __m256 res0 = acc[r][0];
                __m256 res1 = acc[r][1];

                /* Fused bias */
                if (bias && nr >= 8)  res0 = _mm256_add_ps(res0, _mm256_loadu_ps(&bias[j0]));
                if (bias && nr >= 16) res1 = _mm256_add_ps(res1, _mm256_loadu_ps(&bias[j0 + 8]));

                /* Fused activation */
                switch (act) {
                    case RAC_ACT_RELU:
                        res0 = _avx2_relu(res0);
                        res1 = _avx2_relu(res1);
                        break;
                    case RAC_ACT_GELU:
                        res0 = _avx2_gelu_approx(res0);
                        res1 = _avx2_gelu_approx(res1);
                        break;
                    case RAC_ACT_SILU:
                        res0 = _avx2_silu(res0);
                        res1 = _avx2_silu(res1);
                        break;
                    default: break;
                }

                int row = i0 + r;
                if (nr >= 16) {
                    _mm256_storeu_ps(&output[row * N + j0],     res0);
                    _mm256_storeu_ps(&output[row * N + j0 + 8], res1);
                } else {
                    float tmp0[8], tmp1[8];
                    _mm256_storeu_ps(tmp0, res0);
                    _mm256_storeu_ps(tmp1, res1);
                    for (int jj = 0; jj < nr && jj < 8; jj++)
                        output[row * N + j0 + jj] = tmp0[jj];
                    for (int jj = 8; jj < nr; jj++)
                        output[row * N + j0 + jj] = tmp1[jj - 8];
                }
            }
        }
    }

    free(wt);
    return RAC_OK;
}

/* ── AVX2 batch activations ─────────────────────────────────────────────── */

void rac_relu_avx2(const float *x, float *out, int n) {
    int n8 = n & ~7;
    __m256 zero = _mm256_setzero_ps();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n8; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&out[i], _mm256_max_ps(v, zero));
    }
    for (int i = n8; i < n; i++)
        out[i] = (x[i] > 0) ? x[i] : 0;
}

void rac_gelu_avx2(const float *x, float *out, int n) {
    int n8 = n & ~7;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n8; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&out[i], _avx2_gelu_approx(v));
    }
    for (int i = n8; i < n; i++)
        out[i] = x[i] * 0.5f * (1.0f + erff(x[i] * 0.7071067811865f));
}

void rac_silu_avx2(const float *x, float *out, int n) {
    int n8 = n & ~7;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n8; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&out[i], _avx2_silu(v));
    }
    for (int i = n8; i < n; i++)
        out[i] = x[i] / (1.0f + expf(-x[i]));
}

#else /* No AVX2 — scalar fallback */

rac_status rac_sgemm_avx2(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta,
    const rac_config *cfg) {
    return rac_sgemm(A, B, C, M, N, K, alpha, beta, cfg);
}

rac_status rac_fused_linear_avx2(
    const float *input, const float *weight, const float *bias,
    float *output, int M, int N, int K,
    rac_activation act, const rac_config *cfg) {
    return rac_fused_linear(input, weight, bias, output, M, N, K, act, cfg);
}

void rac_relu_avx2(const float *x, float *out, int n) { rac_relu(x, out, n); }
void rac_gelu_avx2(const float *x, float *out, int n) { rac_gelu(x, out, n); }
void rac_silu_avx2(const float *x, float *out, int n) { rac_silu(x, out, n); }

#endif /* RAC_HAS_AVX2_COMPILE */
