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
 * AVX2 SGEMM with register micro-tiling.
 *
 * Strategy: each thread computes a MR x NR micro-tile of C.
 *   MR = 6 rows (6 broadcasts of A per K step)
 *   NR = 16 columns (2 x __m256 = 16 floats)
 *   Per K step: 6 broadcasts × 2 loads = 12 FMAs from 8 memory ops
 *   Arithmetic intensity: 12/8 = 1.5 FMAs/load (vs 1/1 = 1.0 before)
 *
 * Outer loop tiles over M in blocks of MR, N in blocks of NR.
 * K-loop tiles for L1 cache (TILE_K elements at a time).
 * OpenMP parallelizes over M-tiles.
 *
 * This is the same register-blocking strategy that OpenBLAS and
 * the GPU micro-8x8 kernel use — adapted for AVX2's 256-bit width.
 */

#define MR 6    /* rows of C per micro-tile */
#define NR 16   /* columns of C per micro-tile (2 x AVX2 = 16 floats) */

rac_status rac_sgemm_avx2(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta,
    const rac_config *cfg)
{
    if (!A || !B || !C) return RAC_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return RAC_ERR_INVALID_DIM;

    int TILE_K = (cfg && cfg->tile_size > 0) ? cfg->tile_size : 64;

    /* Apply beta to C */
    if (beta == 0.0f) {
        memset(C, 0, (size_t)M * N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int i = 0; i < M * N; i++) C[i] *= beta;
    }

    /* Main tiled loop */
    #pragma omp parallel for schedule(dynamic, 2)
    for (int i0 = 0; i0 < M; i0 += MR) {
        int imax = (i0 + MR <= M) ? i0 + MR : M;
        int mr = imax - i0;  /* actual rows in this micro-tile (may be < MR at edge) */

        for (int j0 = 0; j0 < N; j0 += NR) {
            int jmax = (j0 + NR <= N) ? j0 + NR : N;
            int nr = jmax - j0;

            /* Accumulator registers: MR rows × 2 AVX2 vectors (16 cols) */
            __m256 acc[MR][2];
            for (int r = 0; r < MR; r++) {
                acc[r][0] = _mm256_setzero_ps();
                acc[r][1] = _mm256_setzero_ps();
            }

            /* K-loop tiled for L1 */
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                int kmax = (k0 + TILE_K < K) ? k0 + TILE_K : K;

                for (int k = k0; k < kmax; k++) {
                    /* Load 2 vectors (16 floats) from B[k, j0..j0+15] */
                    __m256 b0, b1;
                    if (nr >= 16) {
                        b0 = _mm256_loadu_ps(&B[k * N + j0]);
                        b1 = _mm256_loadu_ps(&B[k * N + j0 + 8]);
                    } else if (nr >= 8) {
                        b0 = _mm256_loadu_ps(&B[k * N + j0]);
                        b1 = _mm256_setzero_ps();
                        /* Load remaining scalars into b1 */
                        float tmp[8] = {0};
                        for (int jj = 8; jj < nr; jj++)
                            tmp[jj - 8] = B[k * N + j0 + jj];
                        b1 = _mm256_loadu_ps(tmp);
                    } else {
                        float tmp0[8] = {0};
                        for (int jj = 0; jj < nr; jj++)
                            tmp0[jj] = B[k * N + j0 + jj];
                        b0 = _mm256_loadu_ps(tmp0);
                        b1 = _mm256_setzero_ps();
                    }

                    /* Broadcast A[i0+r, k] and FMA for each row */
                    for (int r = 0; r < mr; r++) {
                        __m256 a_bc = _mm256_set1_ps(A[(i0 + r) * K + k]);
                        acc[r][0] = _mm256_fmadd_ps(a_bc, b0, acc[r][0]);
                        acc[r][1] = _mm256_fmadd_ps(a_bc, b1, acc[r][1]);
                    }
                }
            }

            /* Write back with alpha scaling */
            __m256 valpha = _mm256_set1_ps(alpha);
            for (int r = 0; r < mr; r++) {
                int row = i0 + r;
                __m256 res0 = _mm256_mul_ps(valpha, acc[r][0]);
                __m256 res1 = _mm256_mul_ps(valpha, acc[r][1]);

                if (nr >= 16) {
                    /* Full micro-tile: load existing C, add, store */
                    __m256 c0 = _mm256_loadu_ps(&C[row * N + j0]);
                    __m256 c1 = _mm256_loadu_ps(&C[row * N + j0 + 8]);
                    _mm256_storeu_ps(&C[row * N + j0],     _mm256_add_ps(c0, res0));
                    _mm256_storeu_ps(&C[row * N + j0 + 8], _mm256_add_ps(c1, res1));
                } else {
                    /* Edge: scalar fallback for remaining columns */
                    float tmp0[8], tmp1[8];
                    _mm256_storeu_ps(tmp0, res0);
                    _mm256_storeu_ps(tmp1, res1);
                    for (int jj = 0; jj < nr && jj < 8; jj++)
                        C[row * N + j0 + jj] += tmp0[jj];
                    for (int jj = 8; jj < nr; jj++)
                        C[row * N + j0 + jj] += tmp1[jj - 8];
                }
            }
        }
    }

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
