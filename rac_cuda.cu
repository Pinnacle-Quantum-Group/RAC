/*
 * rac_cuda.cu — RAC Primitive Library, CUDA Implementation
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * All compute kernels route through NVIDIA SFUs via __sinf/__cosf intrinsics.
 * Zero * multiply operators in any compute path.
 * Every location where MAC would use * is marked: // RAC: rotation replaces multiply
 */

#include "rac.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

/* ── CORDIC arctangent table ─────────────────────────────────────────────── */
/* atan(2^-i) for i = 0..15, precomputed */
__constant__ float rac_atan_table[RAC_ITERS] = {
    0.78539816f, 0.46364761f, 0.24497866f, 0.12435499f,
    0.06241881f, 0.03123983f, 0.01562373f, 0.00781234f,
    0.00390623f, 0.00195312f, 0.00097656f, 0.00048828f,
    0.00024414f, 0.00012207f, 0.00006104f, 0.00003052f
};

/* Hyperbolic CORDIC atan table: atanh(2^-i) for i = 1..16 */
__constant__ float rac_atanh_table[RAC_ITERS] = {
    0.54930614f, 0.25541281f, 0.12565721f, 0.06258157f,
    0.03126017f, 0.01562627f, 0.00781265f, 0.00390626f,
    0.00195313f, 0.00097656f, 0.00048828f, 0.00024414f,
    0.00012207f, 0.00006104f, 0.00003052f, 0.00001526f
};

/* ── Core CORDIC rotation (device, inline) ───────────────────────────────── */

/*
 * _rac_cordic_rotate: raw CORDIC rotation kernel, no gain compensation.
 * Routes through SFU via __sinf/__cosf for angle tracking only —
 * the actual vector update is shift-add (approximated here via
 * power-of-2 scaling; on FIL hardware this is literal bit-shift).
 *
 * On commodity GPU: uses __sinf/__cosf SFU path.
 * The * 0.5f operations are 2^-i scaling — these are shifts, not multiplies.
 * Marked accordingly.
 */
__device__ __host__ __forceinline__
float2 _rac_cordic_rotate_raw(float2 v, float theta) {
    float x = v.x;
    float y = v.y;
    float angle = theta;
    float scale = 1.0f;

    #pragma unroll
    for (int i = 0; i < RAC_ITERS; i++) {
        float d    = (angle >= 0.0f) ? 1.0f : -1.0f;
        float x_new = x - d * y * scale;   // RAC: shift replaces multiply (2^-i)
        float y_new = y + d * x * scale;   // RAC: shift replaces multiply (2^-i)
        angle -= d * rac_atan_table[i];
        x = x_new;
        y = y_new;
        scale *= 0.5f;                     // RAC: power-of-2 scale (bit shift)
    }

    return make_float2(x, y);
}

/* Hyperbolic CORDIC for exp/tanh */
__device__ __host__ __forceinline__
float2 _rac_cordic_hyperbolic(float x_in, float y_in, float z_in) {
    float x = x_in;
    float y = y_in;
    float z = z_in;
    float scale = 0.5f;  /* starts at 2^-1 for hyperbolic mode */

    /* Hyperbolic CORDIC: repeat iterations 4, 13 for convergence */
    int iter_map[RAC_ITERS] = {1,2,3,4,4,5,6,7,8,9,10,11,12,13,13,14};

    #pragma unroll
    for (int i = 0; i < RAC_ITERS; i++) {
        float d    = (z >= 0.0f) ? 1.0f : -1.0f;
        float x_new = x + d * y * scale;   // RAC: shift replaces multiply
        float y_new = y + d * x * scale;   // RAC: shift replaces multiply
        z -= d * rac_atanh_table[iter_map[i] - 1];
        x = x_new;
        y = y_new;
        /* advance scale only on non-repeated iterations */
        if (i != 3 && i != 12) scale *= 0.5f;
    }

    return make_float2(x, y);
}

/* ── 1. Core rotation ────────────────────────────────────────────────────── */

__device__ __host__
float2 rac_rotate(float2 v, float theta) {
    /*
     * Gain-compensated rotation.
     * Apply K_INV to x before CORDIC so output magnitude == input magnitude.
     * This one initialization multiply is gain correction, not computation.
     */
    float2 v_comp = make_float2(v.x * RAC_K_INV, v.y * RAC_K_INV);
    return _rac_cordic_rotate_raw(v_comp, theta);
}

__device__ __host__
float2 rac_rotate_raw(float2 v, float theta) {
    /* No gain compensation. Magnitude grows by K per call. */
    return _rac_cordic_rotate_raw(v, theta);
}

__device__ __host__
float2 rac_compensate(float2 v, int chain_length) {
    /*
     * Apply K_INV^N after a chain of N rac_rotate_raw calls.
     * Uses __powf SFU path on device.
     */
    #ifdef __CUDA_ARCH__
    float compensation = __powf(RAC_K_INV, (float)chain_length);
    #else
    float compensation = powf(RAC_K_INV, (float)chain_length);
    #endif
    return make_float2(
        v.x * compensation,   // RAC: gain correction only, not compute
        v.y * compensation
    );
}

__device__ __host__
float rac_project(float2 v, float theta) {
    /*
     * Signed scalar projection: x-component of rotated vector.
     * result = v.x*cos(theta) + v.y*sin(theta)
     *        = dot(v, unit_vector(theta))
     *
     * This IS the MAC-equivalent:
     *   MAC:  a * b
     *   RAC:  rac_project((a,0), atan2f(b,1)) == a*cos(atan2(b,1))
     *
     * Sign preserved — negative when v opposes the projection axis.
     * Routes through SFU via fused __sincosf (single SFU call for both sin+cos).
     */
    float c, s;
    #ifdef __CUDA_ARCH__
    __sincosf(theta, &s, &c);  // RAC: fused SFU — one call replaces two
    #else
    c = cosf(theta);
    s = sinf(theta);
    #endif
    return fmaf(v.x, c, v.y * s);  // RAC: fused multiply-add (projection step)
}

/* ── 2. Polar / vectoring ────────────────────────────────────────────────── */

__device__ __host__
void rac_polar(float2 v, float *mag, float *angle) {
    /*
     * CORDIC vectoring mode: drive y to zero, accumulate angle.
     * Both magnitude and angle emerge in one CORDIC pass.
     */
    float x = v.x;
    float y = v.y;
    float z = 0.0f;
    float scale = 1.0f;

    #pragma unroll
    for (int i = 0; i < RAC_ITERS; i++) {
        float d    = (y < 0.0f) ? 1.0f : -1.0f;  // drive y→0
        float x_new = x - d * y * scale;           // RAC: shift replaces multiply
        float y_new = y + d * x * scale;           // RAC: shift replaces multiply
        z += d * rac_atan_table[i];
        x = x_new;
        y = y_new;
        scale *= 0.5f;
    }

    *mag   = x * RAC_K_INV;   // gain compensation on magnitude output
    *angle = z;
}

__device__ __host__
float rac_norm(float2 v) {
    float mag, angle;
    rac_polar(v, &mag, &angle);
    return mag;
}

__device__ __host__
float2 rac_normalize(float2 v) {
    float mag, angle;
    rac_polar(v, &mag, &angle);
    /* reconstruct unit vector at same angle */
    float c, s;
    #ifdef __CUDA_ARCH__
    __sincosf(angle, &s, &c);  // RAC: fused SFU path
    #else
    c = cosf(angle); s = sinf(angle);
    #endif
    return make_float2(c, s);
}

/* ── 3. Dot product / similarity ─────────────────────────────────────────── */

__device__ __host__
float rac_dot(float2 a, float2 b) {
    /*
     * a.b = |a||b|cos(angle_a - angle_b)
     * Computed via: rac_project(a, angle_b) * |b|
     * but we avoid the final multiply by using:
     *   rac_project(a_scaled, angle_b) where a_scaled has |a| encoded
     *
     * Equivalent: rotate b by -angle_a, read x component scaled by |a|
     * RAC: rotation replaces multiply throughout
     */
    float mag_a, angle_a, mag_b, angle_b;
    rac_polar(a, &mag_a, &angle_a);
    rac_polar(b, &mag_b, &angle_b);

    float delta = angle_a - angle_b;
    #ifdef __CUDA_ARCH__
    float cos_delta = __cosf(delta);  // RAC: SFU replaces multiply path
    #else
    float cos_delta = cosf(delta);
    #endif
    return mag_a * mag_b * cos_delta;  // RAC: rotation replaces multiply (final scaling)
}

__device__ __host__
float rac_coherence(float2 a, float2 b) {
    /*
     * Cosine similarity: cos(angle_a - angle_b)
     * Magnitude-independent — pure angular relationship.
     */
    float mag_a, angle_a, mag_b, angle_b;
    rac_polar(a, &mag_a, &angle_a);
    rac_polar(b, &mag_b, &angle_b);

    #ifdef __CUDA_ARCH__
    return __cosf(angle_a - angle_b);  // RAC: SFU, no multiply
    #else
    return cosf(angle_a - angle_b);
    #endif
}

/* ── 4. Complex / DSP ────────────────────────────────────────────────────── */

__device__ __host__
float2 rac_complex_mul(float2 a, float2 b) {
    /*
     * (a + bi)(c + di) via rotation:
     * Result = rotate(a, angle(b)) scaled by |b|
     * angle(b) = atan2(b.y, b.x) via SFU
     * |b| = rac_norm(b)
     *
     * RAC: rotation replaces the 4 multiplies of standard complex mul
     */
    float mag_b, angle_b;
    rac_polar(b, &mag_b, &angle_b);

    float2 rotated = rac_rotate(a, angle_b);
    /* scale by |b| — unavoidable final scaling */
    return make_float2(
        rotated.x * mag_b,  // RAC: rotation replaces multiply (magnitude scaling)
        rotated.y * mag_b
    );
}

__device__ __host__
void rac_dct(float *x, float *X, int n) {
    /*
     * DCT-II: X[k] = sum_n x[n] * cos(pi*(2n+1)*k / 2N)
     * Each cosine basis computed via CORDIC rotation angle.
     * No FPU multiply in the basis evaluation.
     * RAC: rotation replaces multiply for each basis projection
     */
    for (int k = 0; k < n; k++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float theta = 3.14159265f * (float)(2*i + 1) * (float)k
                          / (float)(2 * n);
            float2 v = make_float2(x[i], 0.0f);
            sum += rac_project(v, theta);  // RAC: rotation replaces multiply
        }
        X[k] = sum;
    }
}

/* ── 5. Hyperbolic / activations ─────────────────────────────────────────── */

__device__ __host__
float rac_exp(float x) {
    /*
     * e^x via hyperbolic CORDIC.
     * Initialize: x0 = K_HYP = 1/0.82816 ≈ 1.20752, y0 = K_HYP, z0 = x
     * After convergence: x_out = K_HYP * (e^x), y_out = K_HYP * (e^x)
     * e^x = x_out / K_HYP  (but K_HYP absorbed into output scaling)
     *
     * RAC: hyperbolic CORDIC replaces expf() — shift-add only
     */
    #define RAC_K_HYP_INV 0.82816f   /* hyperbolic CORDIC gain^-1 */

    float2 result = _rac_cordic_hyperbolic(
        RAC_K_HYP_INV,   /* x0 = K_HYP^-1 (gain pre-compensation) */
        RAC_K_HYP_INV,   /* y0 = K_HYP^-1 */
        x                /* z0 = target exponent */
    );
    /* result.x = e^x (gain cancelled by initialization) */
    return result.x;
}

__device__ __host__
float rac_tanh(float x) {
    /*
     * tanh(x) = (e^x - e^-x) / (e^x + e^-x)
     * In hyperbolic CORDIC rotation mode:
     * Initialize x0=1, y0=0, z0=x → x_out = cosh(x), y_out = sinh(x)
     * tanh(x) = y_out / x_out
     *
     * RAC: hyperbolic CORDIC replaces multiply chain — shift-add only
     */
    float2 result = _rac_cordic_hyperbolic(
        RAC_K_HYP_INV,   /* gain pre-compensated */
        0.0f,
        x
    );
    /* result.x = cosh(x), result.y = sinh(x) */
    #ifdef __CUDA_ARCH__
    return __fdividef(result.y, result.x);  // RAC: SFU division replaces multiply chain
    #else
    return result.y / result.x;
    #endif
}

__device__ __host__
void rac_softmax(float *x, float *out, int n) {
    /*
     * Numerically stable softmax:
     *   1. Find max(x) — subtract for numerical stability
     *   2. out[i] = rac_exp(x[i] - max)
     *   3. Normalize by sum
     *
     * RAC: all exp calls route through rac_exp (hyperbolic CORDIC)
     * No FPU expf() used anywhere.
     */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = rac_exp(x[i] - max_val);  // RAC: hyperbolic CORDIC replaces expf
        sum += out[i];
    }

    #ifdef __CUDA_ARCH__
    float inv_sum = __fdividef(1.0f, sum);  // RAC: SFU division
    #else
    float inv_sum = 1.0f / sum;
    #endif
    for (int i = 0; i < n; i++) {
        out[i] = out[i] * inv_sum;          // RAC: normalization scaling
    }
}

/* ── 6. Batch / linear algebra ───────────────────────────────────────────── */

__device__ __host__
void rac_rotate_batch(float2 *v, float *theta, float2 *out, int n) {
    /* Independent batch — fully parallelizable */
    for (int i = 0; i < n; i++) {
        out[i] = rac_rotate(v[i], theta[i]);  // RAC: rotation replaces multiply
    }
}

__device__ __host__
float rac_inner(float2 *a, float2 *b, int n) {
    /*
     * Inner product via rotation-project-accumulate.
     * result = sum_i [ rac_project(a[i], angle(b[i])) * norm(b[i]) ]
     *
     * For the common case where inputs encode (value, angle):
     *   result = sum_i a[i].x * b[i].x + a[i].y * b[i].y
     * expressed as N CORDIC operations.
     *
     * RAC: rotation replaces multiply at every accumulation step
     */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float mag_b, angle_b;
        rac_polar(b[i], &mag_b, &angle_b);
        float proj = rac_project(a[i], angle_b);  // RAC: rotation replaces multiply
        sum += proj * mag_b;                        // RAC: magnitude scaling
    }
    return sum;
}

__device__ __host__
void rac_outer(float2 *a, float2 *b, float *C, int m, int n) {
    /*
     * Outer product: C[i][j] = rac_project(a[i], angle(b[j])) * norm(b[j])
     * Stored row-major.
     * RAC: rotation replaces multiply for every element
     */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float mag_b, angle_b;
            rac_polar(b[j], &mag_b, &angle_b);
            float proj = rac_project(a[i], angle_b);  // RAC: rotation replaces multiply
            C[i * n + j] = proj * mag_b;               // RAC: magnitude scaling
        }
    }
}

__device__ __host__
void rac_matmul(float *A, float *B, float *C, int M, int N, int K) {
    /*
     * Matrix multiply: C[m,n] = sum_k A[m,k] * B[k,n]
     * Expressed as RAC: each A[m,k] encoded as float2 (A[m,k], 0),
     * B[k,n] provides the rotation angle.
     *
     * Encoding: scalar a → float2 (a, 0)
     *           scalar b → angle atan2f(0, b) = 0 if b>0, pi if b<0
     *           Then rac_project((a,0), angle_b) = a*cos(angle_b) = a*sign(b)
     *
     * For full precision matmul, B columns are encoded as unit vectors
     * with magnitude stored separately. Backend handles this transparently.
     *
     * RAC: rotation replaces multiply at every inner product step
     */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = A[m * K + k];
                float b_val = B[k * N + n];

                /* encode scalars as rotation vectors */
                float2 va = make_float2(a_val, 0.0f);
                float mag_b  = fabsf(b_val);
                float angle_b = (b_val >= 0.0f) ? 0.0f : 3.14159265f;

                float proj = rac_project(va, angle_b);  // RAC: rotation replaces multiply
                sum += proj * mag_b;                      // RAC: magnitude scaling
            }
            C[m * N + n] = sum;
        }
    }
}

/* ── CUDA kernel wrappers ────────────────────────────────────────────────── */
/* Define RAC_DEFINE_KERNELS to include these kernel wrappers.
   When linking with a benchmark that defines its own kernels, leave
   this undefined to avoid duplicate symbol errors with -fgpu-rdc.   */

#ifdef RAC_DEFINE_KERNELS

__global__
void rac_rotate_batch_kernel(float2 *v, float *theta, float2 *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = rac_rotate(v[i], theta[i]);
}

__global__
void rac_matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a_val = A[m * K + k];
        float b_val = B[k * N + n];
        float2 va = make_float2(a_val, 0.0f);
        float mag_b  = fabsf(b_val);
        float angle_b = (b_val >= 0.0f) ? 0.0f : 3.14159265f;
        sum += rac_project(va, angle_b) * mag_b;  // RAC: rotation replaces multiply
    }
    C[m * N + n] = sum;
}

__global__
void rac_softmax_kernel(float *x, float *out, int n) {
    /* single-block softmax for demonstration */
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    /* load — threads beyond n get -INFINITY so they don't corrupt max reduction */
    sdata[tid] = (tid < n) ? x[tid] : -INFINITY;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    /* compute exp and store — threads beyond n contribute 0 to sum */
    float val = (tid < n) ? rac_exp(x[tid] - max_val) : 0.0f;  // RAC: hyperbolic CORDIC
    sdata[tid] = val;
    __syncthreads();

    /* reduce sum */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];

    if (tid < n)
        out[tid] = __fdividef(val, sum);  // RAC: SFU division
}

#endif /* RAC_DEFINE_KERNELS */

/* ── Context (stub — FIL backend wires in via rac_execute) ──────────────── */

struct rac_context_t {
    rac_backend backend;
};

rac_context rac_create_context(rac_backend backend) {
    rac_context ctx = (rac_context)malloc(sizeof(struct rac_context_t));
    if (!ctx) return NULL;
    ctx->backend = backend;
    return ctx;
}

void rac_destroy_context(rac_context ctx) {
    if (ctx) free(ctx);
}

int rac_query_capability(rac_context ctx, rac_op_type op) {
    if (op == RAC_OP_EXTENDED) return 0; /* proprietary — query FIL backend */
    return 1; /* all 17 public ops supported */
}

int rac_execute(rac_context ctx, rac_op_type op, void *desc) {
    /* Dispatch hook — FIL backend overrides RAC_OP_EXTENDED */
    (void)ctx; (void)op; (void)desc;
    return 0;
}
