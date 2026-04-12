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

/* Extended atan table for up to 24-iteration precision. Entries beyond
 * RAC_ITERS provide the long-tail precision used in training-time CORDIC.
 * Computed as atan(2^-i) for i in [16, 23]. */
static const float rac_atan_table_ext[8] = {
    0.00001526f, 0.00000763f, 0.00000381f, 0.00000191f,
    0.00000095f, 0.00000048f, 0.00000024f, 0.00000012f,
};

/* Precomputed CORDIC gain for iters 4..24.
 * K(n) = product_{i=0..n-1} sqrt(1 + 2^-2i). Gain tends to ~1.64676. */
static const float rac_k_table[25] = {
    1.00000000f, 1.41421356f, 1.58113883f, 1.62979295f,  /* 0..3 */
    1.64248460f, 1.64568447f, 1.64648404f, 1.64668388f,  /* 4..7 */
    1.64673383f, 1.64674632f, 1.64674944f, 1.64675022f,  /* 8..11 */
    1.64675042f, 1.64675047f, 1.64675048f, 1.64675049f,  /* 12..15 */
    1.64675049f, 1.64675049f, 1.64675049f, 1.64675049f,  /* 16..19 */
    1.64675049f, 1.64675049f, 1.64675049f, 1.64675049f,  /* 20..23 */
    1.64675049f,                                          /* 24      */
};

static inline int _clamp_iters(int iters) {
    if (iters < 4) return 4;
    if (iters > 24) return 24;
    return iters;
}

static inline float _atan_entry(int i) {
    return (i < RAC_ITERS) ? rac_atan_table[i] : rac_atan_table_ext[i - RAC_ITERS];
}

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
    /* Wrap theta into (-π, π]. */
    const float TAU = 2.0f * RAC_PI;
    while (theta >  RAC_PI) theta -= TAU;
    while (theta <= -RAC_PI) theta += TAU;

    /* CORDIC circular rotation converges for theta in [-π/2, π/2].
     * Outside that range, pre-rotate by ±π (i.e. (x,y) -> (-x,-y)) and
     * rotate by the residual. */
    float x = v.x, y = v.y;
    if (theta > RAC_PI * 0.5f) {
        x = -x; y = -y;
        theta -= RAC_PI;
    } else if (theta < -RAC_PI * 0.5f) {
        x = -x; y = -y;
        theta += RAC_PI;
    }

    float angle = theta, scale = 1.0f;
    for (int i = 0; i < iters; i++) {
        float d = (angle >= 0.0f) ? 1.0f : -1.0f;
        float xn = x - d * y * scale;
        float yn = y + d * x * scale;
        angle -= d * _atan_entry(i);
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
    /* CORDIC vectoring only converges for x >= 0. Pre-rotate by ±π if
     * necessary, then add that offset back to the accumulated angle.
     * `z` ends up at -theta (we rotate toward zero y), so angle = -z. */
    float pre_angle = 0.0f;
    float vx = v.x, vy = v.y;
    if (vx < 0.0f) {
        pre_angle = (vy >= 0.0f) ? RAC_PI : -RAC_PI;
        vx = -vx;
        vy = -vy;
    }
    float x = vx, y = vy, z = 0.0f, scale = 1.0f;
    for (int i = 0; i < RAC_ITERS; i++) {
        float d = (y < 0.0f) ? 1.0f : -1.0f;
        float xn = x - d * y * scale;
        float yn = y + d * x * scale;
        z += d * rac_atan_table[i];
        x = xn; y = yn;
        scale *= 0.5f;
    }
    *mag = x * RAC_K_INV;
    *angle = pre_angle - z;
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

/* ── 5b. Tunable-precision CORDIC ───────────────────────────────────────── */

rac_vec2 rac_rotate_n(rac_vec2 v, float theta, int iters) {
    iters = _clamp_iters(iters);
    float k_inv = 1.0f / rac_k_table[iters];
    rac_vec2 comp = {v.x * k_inv, v.y * k_inv};
    return _cordic_rotate(comp, theta, iters);
}

float rac_project_n(rac_vec2 v, float theta, int iters) {
    iters = _clamp_iters(iters);
    /* Use CORDIC: rotate (1, 0) by theta scaled by K_INV^n to get (cos, sin). */
    rac_vec2 unit = {1.0f / rac_k_table[iters], 0.0f};
    rac_vec2 r = _cordic_rotate(unit, theta, iters);
    return fmaf(v.x, r.x, v.y * r.y);
}

void rac_polar_n(rac_vec2 v, float *mag, float *angle, int iters) {
    iters = _clamp_iters(iters);
    float pre_angle = 0.0f;
    float vx = v.x, vy = v.y;
    if (vx < 0.0f) {
        pre_angle = (vy >= 0.0f) ? RAC_PI : -RAC_PI;
        vx = -vx;
        vy = -vy;
    }
    float x = vx, y = vy, z = 0.0f, scale = 1.0f;
    for (int i = 0; i < iters; i++) {
        float d = (y < 0.0f) ? 1.0f : -1.0f;
        float xn = x - d * y * scale;
        float yn = y + d * x * scale;
        z += d * _atan_entry(i);
        x = xn; y = yn;
        scale *= 0.5f;
    }
    *mag = x / rac_k_table[iters];
    *angle = pre_angle - z;
}

float rac_exp_n(float x, int iters) {
    (void)iters;  /* CPU path uses libm — see rationale above */
    return expf(x);
}

float rac_tanh_n(float x, int iters) {
    (void)iters;
    return tanhf(x);
}

void rac_sincos(float theta, float *s, float *c) {
    /* One CORDIC pass of (K_INV, 0) by theta yields (cos, sin) directly. */
    rac_vec2 unit = {RAC_K_INV, 0.0f};
    rac_vec2 r = _cordic_rotate(unit, theta, RAC_ITERS);
    *c = r.x;
    *s = r.y;
}

float rac_rsqrt(float x) {
    /* CPU uses libm for correctness. GPU backends (rac_cuda.cu/rac_hip.cpp)
     * implement this as hyperbolic CORDIC vectoring for SFU routing. */
    if (x <= 0.0f) return 0.0f;
    return 1.0f / sqrtf(x);
}

float rac_sigmoid(float x) {
    /* sigmoid(x) = 0.5 * (1 + tanh(x/2)) — one transcendental call. */
    return 0.5f * (1.0f + tanhf(0.5f * x));
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

/* ── 10. Transformer primitives ─────────────────────────────────────────── */

rac_status rac_layernorm(
    const float *x, float *y,
    const float *gamma, const float *beta,
    float eps, int rows, int d,
    const rac_config *cfg)
{
    (void)cfg;
    if (!x || !y) return RAC_ERR_NULL_PTR;
    if (rows <= 0 || d <= 0) return RAC_ERR_INVALID_DIM;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * d;
        float       *yr = y + r * d;

        /* Linear accumulate: mean */
        float mean = 0.0f;
        for (int i = 0; i < d; i++) mean += xr[i];
        mean /= (float)d;

        /* Linear accumulate: variance */
        float var = 0.0f;
        for (int i = 0; i < d; i++) {
            float z = xr[i] - mean;
            var = fmaf(z, z, var);
        }
        var /= (float)d;

        /* Hyperbolic vectoring: 1/sqrt(var + eps) */
        float inv = rac_rsqrt(var + eps);

        for (int i = 0; i < d; i++) {
            float z = (xr[i] - mean) * inv;
            float g = gamma ? gamma[i] : 1.0f;
            float b = beta  ? beta[i]  : 0.0f;
            yr[i] = fmaf(g, z, b);
        }
    }
    return RAC_OK;
}

rac_status rac_rmsnorm(
    const float *x, float *y,
    const float *gamma,
    float eps, int rows, int d,
    const rac_config *cfg)
{
    (void)cfg;
    if (!x || !y) return RAC_ERR_NULL_PTR;
    if (rows <= 0 || d <= 0) return RAC_ERR_INVALID_DIM;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * d;
        float       *yr = y + r * d;

        float ms = 0.0f;
        for (int i = 0; i < d; i++) ms = fmaf(xr[i], xr[i], ms);
        ms /= (float)d;

        float inv = rac_rsqrt(ms + eps);
        for (int i = 0; i < d; i++) {
            float g = gamma ? gamma[i] : 1.0f;
            yr[i] = xr[i] * inv * g;
        }
    }
    return RAC_OK;
}

rac_status rac_rope_cache(
    float *cos_out, float *sin_out,
    int max_seq, int head_dim, float base)
{
    if (!cos_out || !sin_out) return RAC_ERR_NULL_PTR;
    if (max_seq <= 0 || head_dim <= 0 || (head_dim & 1)) return RAC_ERR_INVALID_DIM;

    int half = head_dim / 2;
    /* Frequencies: inv_freq[i] = 1 / base^(2i / head_dim) */
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < max_seq; p++) {
        for (int i = 0; i < half; i++) {
            float exponent = (2.0f * (float)i) / (float)head_dim;
            float inv_freq = 1.0f / powf(base, exponent);
            float angle = (float)p * inv_freq;
            float s, c;
            rac_sincos(angle, &s, &c);
            cos_out[p * half + i] = c;
            sin_out[p * half + i] = s;
        }
    }
    return RAC_OK;
}

rac_status rac_rope_apply(
    float *x,
    const float *cos_tab, const float *sin_tab,
    int batch, int n_heads, int seq, int head_dim,
    const rac_config *cfg)
{
    (void)cfg;
    if (!x || !cos_tab || !sin_tab) return RAC_ERR_NULL_PTR;
    if (batch <= 0 || n_heads <= 0 || seq <= 0 || head_dim <= 0 || (head_dim & 1))
        return RAC_ERR_INVALID_DIM;

    int half = head_dim / 2;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int t = 0; t < seq; t++) {
                float *row = x + ((b * n_heads + h) * seq + t) * head_dim;
                const float *cr = cos_tab + t * half;
                const float *sr = sin_tab + t * half;
                /* Each adjacent pair (row[2i], row[2i+1]) is a Givens rotation.
                 * This IS rac_rotate — native circular CORDIC. No multipliers
                 * dedicated to this op on a RAC ASIC. */
                for (int i = 0; i < half; i++) {
                    float a = row[2 * i];
                    float b2 = row[2 * i + 1];
                    float c = cr[i];
                    float s = sr[i];
                    row[2 * i]     = fmaf(a, c, -b2 * s);
                    row[2 * i + 1] = fmaf(a, s,  b2 * c);
                }
            }
        }
    }
    return RAC_OK;
}

rac_status rac_scaled_dot_attention(
    const float *q, const float *k, const float *v,
    const float *mask, int is_causal,
    float *out,
    int batch, int n_heads, int seq, int head_dim,
    const rac_config *cfg)
{
    (void)cfg;
    if (!q || !k || !v || !out) return RAC_ERR_NULL_PTR;
    if (batch <= 0 || n_heads <= 0 || seq <= 0 || head_dim <= 0)
        return RAC_ERR_INVALID_DIM;

    float scale = 1.0f / sqrtf((float)head_dim);

    /* Scratch scores per (batch, head) — allocated inside parallel region. */
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_heads; h++) {
            float *scores = (float *)malloc((size_t)seq * seq * sizeof(float));
            if (!scores) continue;

            const float *qbh = q + ((b * n_heads + h) * seq) * head_dim;
            const float *kbh = k + ((b * n_heads + h) * seq) * head_dim;
            const float *vbh = v + ((b * n_heads + h) * seq) * head_dim;
            float       *obh = out + ((b * n_heads + h) * seq) * head_dim;

            /* scores[t, s] = (Q[t] . K[s]) * scale */
            for (int t = 0; t < seq; t++) {
                for (int s = 0; s < seq; s++) {
                    float acc = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        acc = fmaf(qbh[t * head_dim + d], kbh[s * head_dim + d], acc);
                    }
                    float sc = acc * scale;
                    if (is_causal && s > t) sc = -INFINITY;
                    if (mask) sc += mask[t * seq + s];
                    scores[t * seq + s] = sc;
                }
            }

            /* softmax per row, then O[t] = sum_s p[t,s] * V[s] */
            for (int t = 0; t < seq; t++) {
                float *row = scores + t * seq;
                float m = row[0];
                for (int s = 1; s < seq; s++) if (row[s] > m) m = row[s];
                float sum = 0.0f;
                for (int s = 0; s < seq; s++) {
                    row[s] = expf(row[s] - m);
                    sum += row[s];
                }
                float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                for (int s = 0; s < seq; s++) row[s] *= inv;

                for (int d = 0; d < head_dim; d++) {
                    float acc = 0.0f;
                    for (int s = 0; s < seq; s++) {
                        acc = fmaf(row[s], vbh[s * head_dim + d], acc);
                    }
                    obh[t * head_dim + d] = acc;
                }
            }

            free(scores);
        }
    }
    return RAC_OK;
}
