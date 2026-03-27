/*
 * rac_avx2.c — RAC AVX2/FMA Vectorized Implementation
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Build: cc -O3 -mavx2 -mfma -fopenmp rac_cpu.c rac_avx2.c -lm -shared -o librac_avx2.so
 */

#include "rac_avx2.h"
#include <math.h>
#include <string.h>

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

/* Default cache tile sizes — overridden by HAL if available */
#define DEFAULT_KC 256
#define DEFAULT_MC 256
#define DEFAULT_NC 2048

/* ── Pack A[mc×kc] → column-panel layout ── */
/* packed_A stores MR-wide column panels: [MR, kc, mc/MR] contiguous */
static void _pack_a(const float *A, float *packed, int mc, int kc, int lda, int K_start) {
    for (int i = 0; i < mc; i += MR) {
        int mr = ((i + MR) <= mc) ? MR : (mc - i);
        for (int k = 0; k < kc; k++) {
            for (int r = 0; r < mr; r++)
                *packed++ = A[(i + r) * lda + K_start + k];
            for (int r = mr; r < MR; r++)
                *packed++ = 0.0f;  /* zero-pad edge */
        }
    }
}

/* ── Pack B[kc×nc] → row-panel layout ── */
/* packed_B stores NR-wide row panels: [kc, NR, nc/NR] contiguous */
static void _pack_b(const float *B, float *packed, int kc, int nc, int ldb, int K_start, int N_start) {
    for (int j = 0; j < nc; j += NR) {
        int nr = ((j + NR) <= nc) ? NR : (nc - j);
        for (int k = 0; k < kc; k++) {
            for (int c = 0; c < nr; c++)
                *packed++ = B[(K_start + k) * ldb + N_start + j + c];
            for (int c = nr; c < NR; c++)
                *packed++ = 0.0f;  /* zero-pad edge */
        }
    }
}

/* ── MR×NR micro-kernel: packed_A[MR×kc] × packed_B[kc×NR] → C[MR×NR] ── */
static void _micro_kernel_6x16(
    const float *packed_a,  /* [MR × kc] packed column-panel */
    const float *packed_b,  /* [kc × NR] packed row-panel */
    float *C, int ldc,
    int mr, int nr, int kc,
    float alpha)
{
    __m256 acc[MR][2];
    for (int r = 0; r < MR; r++) {
        acc[r][0] = _mm256_setzero_ps();
        acc[r][1] = _mm256_setzero_ps();
    }

    /* Main K-loop over packed data — fully contiguous, prefetch-friendly */
    for (int k = 0; k < kc; k++) {
        /* Load NR=16 floats from packed B (contiguous) */
        __m256 b0 = _mm256_loadu_ps(&packed_b[k * NR]);
        __m256 b1 = _mm256_loadu_ps(&packed_b[k * NR + 8]);

        /* Broadcast MR=6 values from packed A (contiguous) */
        for (int r = 0; r < MR; r++) {
            __m256 a_bc = _mm256_set1_ps(packed_a[k * MR + r]);
            acc[r][0] = _mm256_fmadd_ps(a_bc, b0, acc[r][0]);
            acc[r][1] = _mm256_fmadd_ps(a_bc, b1, acc[r][1]);
        }
    }

    /* Write back: C += alpha * acc */
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

    /* Cache tile sizes — use HAL-tuned values or defaults */
    int KC = (cfg && cfg->tile_size > 0) ? cfg->tile_size : DEFAULT_KC;
    int MC = DEFAULT_MC;
    int NC = DEFAULT_NC;

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

    /* packed_B: ceil(NC/NR) panels × KC × NR floats per panel */
    /* packed_A: ceil(MC/MR) panels × KC × MR floats per panel, per thread */
    int b_panels = (NC + NR - 1) / NR;
    int a_panels = (MC + MR - 1) / MR;
    size_t packed_b_size = (size_t)b_panels * KC * NR;
    size_t packed_a_size = (size_t)a_panels * KC * MR;

    float *packed_b = (float *)calloc(packed_b_size, sizeof(float));
    float *packed_a_all = (float *)calloc((size_t)n_threads * packed_a_size, sizeof(float));
    if (!packed_b || !packed_a_all) {
        free(packed_b); free(packed_a_all);
        return RAC_ERR_ALLOC;
    }

    /* 3-level blocked loop: jc (L3) → ic (L2) → micro-kernel (L1) */
    for (int jc = 0; jc < N; jc += NC) {
        int nc = ((jc + NC) <= N) ? NC : (N - jc);

        for (int pc = 0; pc < K; pc += KC) {
            int kc = ((pc + KC) <= K) ? KC : (K - pc);

            /* Pack B panel [kc × nc] — shared across all threads */
            _pack_b(B, packed_b, kc, nc, N, pc, jc);

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

                    /* Micro-kernel loop */
                    for (int jr = 0; jr < nc; jr += NR) {
                        int nr = ((jr + NR) <= nc) ? NR : (nc - jr);
                        /* packed_b layout: panel (jr/NR) has kc*NR floats */
                        float *pb = &packed_b[(jr / NR) * kc * NR];

                        for (int ir = 0; ir < mc; ir += MR) {
                            int mr = ((ir + MR) <= mc) ? MR : (mc - ir);
                            /* packed_a layout: panel (ir/MR) has kc*MR floats */
                            float *pa = &packed_a[(ir / MR) * kc * MR];

                            _micro_kernel_6x16(
                                pa, pb,
                                &C[(ic + ir) * N + (jc + jr)],
                                N, mr, nr, kc, alpha);
                        }
                    }
                }
            }
        }
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
