/*
 * rac_cpu.c — RAC Portable CPU Implementation (C99 + OpenMP)
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Build: cc -O3 -march=native -fopenmp rac_cpu.c -lm -shared -o librac.so
 */

#include "rac_cpu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── CORDIC tables ──────────────────────────────────────────────────────── */

static const float rac_atan_table[RAC_ITERS] = {
    0.78539816f, 0.46364761f, 0.24497866f, 0.12435499f,
    0.06241881f, 0.03123983f, 0.01562373f, 0.00781234f,
    0.00390623f, 0.00195312f, 0.00097656f, 0.00048828f,
    0.00024414f, 0.00012207f, 0.00006104f, 0.00003052f
};

static const float rac_atanh_table[RAC_ITERS] = {
    0.54930614f, 0.25541281f, 0.12565721f, 0.06258157f,
    0.03126017f, 0.01562627f, 0.00781265f, 0.00390626f,
    0.00195313f, 0.00097656f, 0.00048828f, 0.00024414f,
    0.00012207f, 0.00006104f, 0.00003052f, 0.00001526f
};

/* ── Configuration ──────────────────────────────────────────────────────── */

rac_config rac_default_config(void) {
    rac_config cfg;
    cfg.num_threads = 0;
    cfg.tile_size = 64;
    cfg.cordic_iters = RAC_ITERS;
    return cfg;
}

static int _get_threads(const rac_config *cfg) {
    if (cfg && cfg->num_threads > 0) return cfg->num_threads;
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static int _get_tile(const rac_config *cfg) {
    if (cfg && cfg->tile_size > 0) return cfg->tile_size;
    return 64;
}

/* ── CORDIC core ────────────────────────────────────────────────────────── */

static inline rac_vec2 _cordic_rotate(rac_vec2 v, float theta, int iters) {
    float x = v.x, y = v.y, angle = theta, scale = 1.0f;
    for (int i = 0; i < iters; i++) {
        float d = (angle >= 0.0f) ? 1.0f : -1.0f;
        float xn = x - d * y * scale;
        float yn = y + d * x * scale;
        angle -= d * rac_atan_table[i];
        x = xn; y = yn;
        scale *= 0.5f;
    }
    return (rac_vec2){x, y};
}

static inline rac_vec2 _cordic_hyperbolic(float x_in, float y_in, float z_in) {
    float x = x_in, y = y_in, z = z_in, scale = 0.5f;
    static const int iter_map[RAC_ITERS] = {1,2,3,4,4,5,6,7,8,9,10,11,12,13,13,14};
    for (int i = 0; i < RAC_ITERS; i++) {
        float d = (z >= 0.0f) ? 1.0f : -1.0f;
        float xn = x + d * y * scale;
        float yn = y + d * x * scale;
        z -= d * rac_atanh_table[iter_map[i] - 1];
        x = xn; y = yn;
        if (i != 3 && i != 12) scale *= 0.5f;
    }
    return (rac_vec2){x, y};
}

/* ── 1. Core rotation ───────────────────────────────────────────────────── */

rac_vec2 rac_rotate(rac_vec2 v, float theta) {
    rac_vec2 comp = {v.x * RAC_K_INV, v.y * RAC_K_INV};
    return _cordic_rotate(comp, theta, RAC_ITERS);
}

rac_vec2 rac_rotate_raw(rac_vec2 v, float theta) {
    return _cordic_rotate(v, theta, RAC_ITERS);
}

rac_vec2 rac_compensate(rac_vec2 v, int chain_length) {
    float c = powf(RAC_K_INV, (float)chain_length);
    return (rac_vec2){v.x * c, v.y * c};
}

float rac_project(rac_vec2 v, float theta) {
    float c = cosf(theta);
    float s = sinf(theta);
    return fmaf(v.x, c, v.y * s);
}

/* ── 2. Polar / vectoring ───────────────────────────────────────────────── */

void rac_polar(rac_vec2 v, float *mag, float *angle) {
    float x = v.x, y = v.y, z = 0.0f, scale = 1.0f;
    for (int i = 0; i < RAC_ITERS; i++) {
        float d = (y < 0.0f) ? 1.0f : -1.0f;
        float xn = x - d * y * scale;
        float yn = y + d * x * scale;
        z += d * rac_atan_table[i];
        x = xn; y = yn;
        scale *= 0.5f;
    }
    *mag = x * RAC_K_INV;
    *angle = z;
}

float rac_norm(rac_vec2 v) {
    float mag, angle;
    rac_polar(v, &mag, &angle);
    return mag;
}

rac_vec2 rac_normalize(rac_vec2 v) {
    float mag, angle;
    rac_polar(v, &mag, &angle);
    return (rac_vec2){cosf(angle), sinf(angle)};
}

/* ── 3. Dot product / similarity ────────────────────────────────────────── */

float rac_dot(rac_vec2 a, rac_vec2 b) {
    float ma, aa, mb, ab;
    rac_polar(a, &ma, &aa);
    rac_polar(b, &mb, &ab);
    return ma * mb * cosf(aa - ab);
}

float rac_coherence(rac_vec2 a, rac_vec2 b) {
    float ma, aa, mb, ab;
    rac_polar(a, &ma, &aa);
    rac_polar(b, &mb, &ab);
    return cosf(aa - ab);
}

/* ── 4. Complex / DSP ───────────────────────────────────────────────────── */

rac_vec2 rac_complex_mul(rac_vec2 a, rac_vec2 b) {
    float mb, ab;
    rac_polar(b, &mb, &ab);
    rac_vec2 rot = rac_rotate(a, ab);
    return (rac_vec2){rot.x * mb, rot.y * mb};
}

void rac_dct(const float *x, float *X, int n) {
    for (int k = 0; k < n; k++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float theta = RAC_PI * (float)(2*i + 1) * (float)k / (float)(2 * n);
            sum += rac_project((rac_vec2){x[i], 0.0f}, theta);
        }
        X[k] = sum;
    }
}

/* ── 5. Transcendental / activations ────────────────────────────────────── */
/*
 * CPU path uses libm for exp, tanh, softmax.
 * CORDIC hyperbolic is reserved for the GPU SFU path (rac_cuda.cu,
 * rac_hip.cpp) where hardware transcendental units provide a real
 * performance advantage over FPU expf/tanhf.
 *
 * The CORDIC rotation primitives (rac_rotate, rac_project, rac_polar,
 * etc.) DO use CORDIC on CPU — those are the core RAC operations
 * where the shift-add decomposition IS the point.
 *
 * Don't "fix" this to use CORDIC exp on CPU — there's no benefit,
 * and the 16-iteration hyperbolic CORDIC has ~30% error on exp(0)
 * due to gain factor precision limits at this iteration count.
 */

/*
 * On CPU there is no SFU advantage to CORDIC — use standard libm.
 * The CORDIC hyperbolic path is retained in rac_cuda.cu for GPU SFUs.
 */
float rac_exp(float x) {
    return expf(x);
}

float rac_tanh(float x) {
    return tanhf(x);
}

void rac_softmax(const float *x, float *out, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++)
        out[i] *= inv;
}

/* ── 6. Batch operations (OpenMP) ───────────────────────────────────────── */

void rac_rotate_batch(const rac_vec2 *v, const float *theta,
                       rac_vec2 *out, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        out[i] = rac_rotate(v[i], theta[i]);
}

float rac_inner(const rac_vec2 *a, const rac_vec2 *b, int n) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        float mb, ab;
        rac_polar(b[i], &mb, &ab);
        sum += rac_project(a[i], ab) * mb;
    }
    return sum;
}

void rac_outer(const rac_vec2 *a, const rac_vec2 *b, float *C, int m, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float mb, ab;
            rac_polar(b[j], &mb, &ab);
            C[i * n + j] = rac_project(a[i], ab) * mb;
        }
    }
}

/* ── 7. SGEMM (OpenMP + cache tiling) ──────────────────────────────────── */
/*
 * Cache-tiled SGEMM with OpenMP parallelism.
 * Uses the RAC degenerate encoding (a*b = rac_project((a,0), 0|pi) * |b|)
 * but optimized to direct FMA since cos(0)=1, cos(pi)=-1.
 *
 * Tiling strategy:
 *   - Outer loop tiles over M and N (L2 cache)
 *   - Inner loop tiles over K (L1 cache)
 *   - Each tile: TILE x TILE block of C accumulated from TILE-wide K strips
 *   - OpenMP parallelizes over M tiles
 */

rac_status rac_sgemm(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta,
    const rac_config *cfg)
{
    if (!A || !B || !C) return RAC_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return RAC_ERR_INVALID_DIM;

    int TILE = _get_tile(cfg);
    (void)_get_threads(cfg);

    /* Apply beta to C */
    if (beta == 0.0f) {
        memset(C, 0, (size_t)M * N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int i = 0; i < M * N; i++)
            C[i] *= beta;
    }

    /* Tiled matmul */
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int imax = (i0 + TILE < M) ? i0 + TILE : M;
                int jmax = (j0 + TILE < N) ? j0 + TILE : N;
                int kmax = (k0 + TILE < K) ? k0 + TILE : K;

                for (int i = i0; i < imax; i++) {
                    for (int j = j0; j < jmax; j++) {
                        float sum = 0.0f;
                        for (int k = k0; k < kmax; k++) {
                            sum = fmaf(A[i * K + k], B[k * N + j], sum);
                        }
                        C[i * N + j] = fmaf(alpha, sum, C[i * N + j]);
                    }
                }
            }
        }
    }

    return RAC_OK;
}

rac_status rac_matmul(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    const rac_config *cfg)
{
    return rac_sgemm(A, B, C, M, N, K, 1.0f, 0.0f, cfg);
}

/* ── 8. Activation functions (batch, OpenMP) ────────────────────────────── */

void rac_relu(const float *x, float *out, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        out[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}

void rac_gelu(const float *x, float *out, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        out[i] = x[i] * 0.5f * (1.0f + erff(x[i] * 0.7071067811865f));
}

void rac_silu(const float *x, float *out, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        out[i] = x[i] / (1.0f + expf(-x[i]));
}

void rac_softmax_batch(const float *x, float *out, int batch, int n) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch; b++)
        rac_softmax(x + b * n, out + b * n, n);
}

/* ── 9. Fused linear ────────────────────────────────────────────────────── */

static inline float _apply_activation(float x, rac_activation act) {
    switch (act) {
        case RAC_ACT_RELU: return (x > 0.0f) ? x : 0.0f;
        case RAC_ACT_GELU: return x * 0.5f * (1.0f + erff(x * 0.7071067811865f));
        case RAC_ACT_SILU: return x / (1.0f + expf(-x));
        default: return x;
    }
}

rac_status rac_fused_linear(
    const float *input, const float *weight, const float *bias,
    float *output, int M, int N, int K,
    rac_activation act, const rac_config *cfg)
{
    if (!input || !weight || !output) return RAC_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || K <= 0) return RAC_ERR_INVALID_DIM;

    int TILE = _get_tile(cfg);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; i++) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            int jmax = (j0 + TILE < N) ? j0 + TILE : N;
            for (int j = j0; j < jmax; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    /* weight is [N, K] row-major: weight[j * K + k] */
                    sum = fmaf(input[i * K + k], weight[j * K + k], sum);
                }
                if (bias) sum += bias[j];
                output[i * N + j] = _apply_activation(sum, act);
            }
        }
    }

    return RAC_OK;
}
