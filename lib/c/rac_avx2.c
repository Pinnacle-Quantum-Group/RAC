/*
 * rac_avx2.c — RAC AVX2/FMA Vectorized Implementation
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Build: cc -O3 -mavx2 -mfma -fopenmp rac_cpu.c rac_avx2.c -lm -shared -o librac_avx2.so
 */

#include "rac_avx2.h"
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

rac_status rac_sgemm_avx2(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta,
    const rac_config *cfg)
{
    if (!A || !B || !C) return RAC_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return RAC_ERR_INVALID_DIM;

    int TILE_K = (cfg && cfg->tile_size > 0) ? cfg->tile_size : 64;
    int N8 = N & ~7;  /* N rounded down to multiple of 8 */

    __m256 valpha = _mm256_set1_ps(alpha);
    __m256 vbeta  = _mm256_set1_ps(beta);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < M; i++) {
        /* Process 8 columns at a time */
        for (int j = 0; j < N8; j += 8) {
            __m256 acc = _mm256_setzero_ps();

            if (beta != 0.0f) {
                acc = _mm256_mul_ps(vbeta, _mm256_loadu_ps(&C[i * N + j]));
            }

            /* K-loop tiled for L1 cache */
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                int kmax = (k0 + TILE_K < K) ? k0 + TILE_K : K;
                for (int k = k0; k < kmax; k++) {
                    /* Broadcast A[i,k] to all 8 lanes */
                    __m256 a_val = _mm256_set1_ps(A[i * K + k]);
                    /* Load 8 values of B[k, j..j+7] */
                    __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                    /* FMA: acc += a_val * b_vec */
                    acc = _mm256_fmadd_ps(a_val, b_vec, acc);
                }
            }

            /* Scale by alpha and store */
            acc = _mm256_mul_ps(valpha, acc);
            if (beta != 0.0f) {
                /* alpha was already applied, beta was applied at init */
                _mm256_storeu_ps(&C[i * N + j], acc);
            } else {
                _mm256_storeu_ps(&C[i * N + j], acc);
            }
        }

        /* Handle remainder columns (N % 8) */
        for (int j = N8; j < N; j++) {
            float acc = (beta != 0.0f) ? beta * C[i * N + j] : 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * acc;
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
    __m256 coeff = _mm256_set1_ps(0.7071067811865f);

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
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);

    /* Fast exp approximation: exp(x) ≈ (1 + x/256)^256 via repeated squaring */
    /* Simpler: use the IEEE float trick */
    /* For production, use a proper exp approximation. Here we use scalar fallback */
    float vals[8], results[8];
    _mm256_storeu_ps(vals, x);
    for (int i = 0; i < 8; i++)
        results[i] = vals[i] / (1.0f + expf(-vals[i]));
    return _mm256_loadu_ps(results);
}

rac_status rac_fused_linear_avx2(
    const float *input, const float *weight, const float *bias,
    float *output, int M, int N, int K,
    rac_activation act, const rac_config *cfg)
{
    if (!input || !weight || !output) return RAC_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return RAC_ERR_INVALID_DIM;

    int TILE_K = (cfg && cfg->tile_size > 0) ? cfg->tile_size : 64;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < M; i++) {
        int N8 = N & ~7;

        for (int j = 0; j < N8; j += 8) {
            __m256 acc = _mm256_setzero_ps();

            /* Accumulate: input[i,:] @ weight[j:j+8, :]^T */
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                int kmax = (k0 + TILE_K < K) ? k0 + TILE_K : K;
                for (int k = k0; k < kmax; k++) {
                    __m256 a_val = _mm256_set1_ps(input[i * K + k]);
                    /* weight[j+0..7, k] — stride is K between rows */
                    float b_vals[8];
                    for (int jj = 0; jj < 8; jj++)
                        b_vals[jj] = weight[(j + jj) * K + k];
                    __m256 b_vec = _mm256_loadu_ps(b_vals);
                    acc = _mm256_fmadd_ps(a_val, b_vec, acc);
                }
            }

            /* Fused bias */
            if (bias) {
                __m256 b_vec = _mm256_loadu_ps(&bias[j]);
                acc = _mm256_add_ps(acc, b_vec);
            }

            /* Fused activation */
            switch (act) {
                case RAC_ACT_RELU: acc = _avx2_relu(acc); break;
                case RAC_ACT_GELU: acc = _avx2_gelu_approx(acc); break;
                case RAC_ACT_SILU: acc = _avx2_silu(acc); break;
                default: break;
            }

            _mm256_storeu_ps(&output[i * N + j], acc);
        }

        /* Scalar remainder */
        for (int j = N8; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += input[i * K + k] * weight[j * K + k];
            if (bias) sum += bias[j];
            switch (act) {
                case RAC_ACT_RELU: sum = (sum > 0) ? sum : 0; break;
                case RAC_ACT_GELU: sum = sum * 0.5f * (1.0f + erff(sum * 0.7071067811865f)); break;
                case RAC_ACT_SILU: sum = sum / (1.0f + expf(-sum)); break;
                default: break;
            }
            output[i * N + j] = sum;
        }
    }

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
