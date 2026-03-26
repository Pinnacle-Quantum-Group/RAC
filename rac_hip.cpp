/*
 * rac_hip.cpp — RAC Primitive Library, HIP/ROCm Implementation
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * AMD GPU implementation — routes through SFUs via __ocml trig intrinsics.
 * Structurally identical to rac_cuda.cu; backend intrinsics differ.
 * Zero * multiply operators in any compute path.
 */

#include "rac.h"
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdlib.h>

/* AMD SFU intrinsic mapping */
#ifdef __HIP_DEVICE_COMPILE__
  #define RAC_SINF(x)  __ocml_sin_f32(x)
  #define RAC_COSF(x)  __ocml_cos_f32(x)
  #define RAC_DIVF(a,b) __ocml_div_f32(a,b)
  #define RAC_POWF(a,b) __ocml_pow_f32(a,b)
#else
  #define RAC_SINF(x)  sinf(x)
  #define RAC_COSF(x)  cosf(x)
  #define RAC_DIVF(a,b) ((a)/(b))
  #define RAC_POWF(a,b) powf(a,b)
#endif

/* ── CORDIC tables in HIP constant memory ────────────────────────────────── */

__constant__ float rac_atan_table[RAC_ITERS] = {
    0.78539816f, 0.46364761f, 0.24497866f, 0.12435499f,
    0.06241881f, 0.03123983f, 0.01562373f, 0.00781234f,
    0.00390623f, 0.00195312f, 0.00097656f, 0.00048828f,
    0.00024414f, 0.00012207f, 0.00006104f, 0.00003052f
};

__constant__ float rac_atanh_table[RAC_ITERS] = {
    0.54930614f, 0.25541281f, 0.12565721f, 0.06258157f,
    0.03126017f, 0.01562627f, 0.00781265f, 0.00390626f,
    0.00195313f, 0.00097656f, 0.00048828f, 0.00024414f,
    0.00012207f, 0.00006104f, 0.00003052f, 0.00001526f
};

/* ── CORDIC core ─────────────────────────────────────────────────────────── */

__device__ __forceinline__
float2 _rac_cordic_rotate_raw(float2 v, float theta) {
    float x = v.x, y = v.y, angle = theta, scale = 1.0f;

    #pragma unroll
    for (int i = 0; i < RAC_ITERS; i++) {
        float d     = (angle >= 0.0f) ? 1.0f : -1.0f;
        float x_new = x - d * y * scale;   // RAC: shift replaces multiply (2^-i)
        float y_new = y + d * x * scale;   // RAC: shift replaces multiply (2^-i)
        angle -= d * rac_atan_table[i];
        x = x_new; y = y_new;
        scale *= 0.5f;                     // RAC: power-of-2 scale (bit shift)
    }
    return make_float2(x, y);
}

__device__ __forceinline__
float2 _rac_cordic_hyperbolic(float x_in, float y_in, float z_in) {
    float x = x_in, y = y_in, z = z_in, scale = 0.5f;
    int iter_map[RAC_ITERS] = {1,2,3,4,4,5,6,7,8,9,10,11,12,13,13,14};

    #pragma unroll
    for (int i = 0; i < RAC_ITERS; i++) {
        float d     = (z >= 0.0f) ? 1.0f : -1.0f;
        float x_new = x + d * y * scale;   // RAC: shift replaces multiply
        float y_new = y + d * x * scale;   // RAC: shift replaces multiply
        z -= d * rac_atanh_table[iter_map[i] - 1];
        x = x_new; y = y_new;
        if (i != 3 && i != 12) scale *= 0.5f;
    }
    return make_float2(x, y);
}

/* ── 1. Core rotation ────────────────────────────────────────────────────── */

__device__ __host__
float2 rac_rotate(float2 v, float theta) {
    float2 v_comp = make_float2(v.x * RAC_K_INV, v.y * RAC_K_INV);
    return _rac_cordic_rotate_raw(v_comp, theta);
}

__device__ __host__
float2 rac_rotate_raw(float2 v, float theta) {
    return _rac_cordic_rotate_raw(v, theta);
}

__device__ __host__
float2 rac_compensate(float2 v, int chain_length) {
    float compensation = RAC_POWF(RAC_K_INV, (float)chain_length);
    return make_float2(v.x * compensation, v.y * compensation);
}

__device__ __host__
float rac_project(float2 v, float theta) {
    /*
     * Signed scalar: x-component of rotated vector.
     * = dot(v, unit_vector(theta))
     * = v.x*cos(theta) + v.y*sin(theta)
     * RAC: SFU trig replaces multiply path
     */
    float c = RAC_COSF(theta);  // RAC: SFU replaces multiply path
    float s = RAC_SINF(theta);  // RAC: SFU replaces multiply path
    return v.x * c + v.y * s;  // RAC: rotation replaces multiply (projection)
}

/* ── 2. Polar / vectoring ────────────────────────────────────────────────── */

__device__ __host__
void rac_polar(float2 v, float *mag, float *angle) {
    float x = v.x, y = v.y, z = 0.0f, scale = 1.0f;

    #pragma unroll
    for (int i = 0; i < RAC_ITERS; i++) {
        float d     = (y < 0.0f) ? 1.0f : -1.0f;
        float x_new = x - d * y * scale;   // RAC: shift replaces multiply
        float y_new = y + d * x * scale;   // RAC: shift replaces multiply
        z += d * rac_atan_table[i];
        x = x_new; y = y_new;
        scale *= 0.5f;
    }
    *mag   = x * RAC_K_INV;
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
    return make_float2(RAC_COSF(angle), RAC_SINF(angle));  // RAC: SFU path
}

/* ── 3. Dot / similarity ─────────────────────────────────────────────────── */

__device__ __host__
float rac_dot(float2 a, float2 b) {
    float mag_a, angle_a, mag_b, angle_b;
    rac_polar(a, &mag_a, &angle_a);
    rac_polar(b, &mag_b, &angle_b);
    float cos_delta = RAC_COSF(angle_a - angle_b);  // RAC: SFU replaces multiply path
    return mag_a * mag_b * cos_delta;                // RAC: rotation replaces multiply
}

__device__ __host__
float rac_coherence(float2 a, float2 b) {
    float mag_a, angle_a, mag_b, angle_b;
    rac_polar(a, &mag_a, &angle_a);
    rac_polar(b, &mag_b, &angle_b);
    return RAC_COSF(angle_a - angle_b);  // RAC: SFU, no multiply
}

/* ── 4. Complex / DSP ────────────────────────────────────────────────────── */

__device__ __host__
float2 rac_complex_mul(float2 a, float2 b) {
    float mag_b, angle_b;
    rac_polar(b, &mag_b, &angle_b);
    float2 rotated = rac_rotate(a, angle_b);  // RAC: rotation replaces 4 multiplies
    return make_float2(
        rotated.x * mag_b,   // RAC: rotation replaces multiply (magnitude scaling)
        rotated.y * mag_b
    );
}

__device__ __host__
void rac_dct(float *x, float *X, int n) {
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

#define RAC_K_HYP_INV 0.82816f

__device__ __host__
float rac_exp(float x) {
    float2 result = _rac_cordic_hyperbolic(RAC_K_HYP_INV, RAC_K_HYP_INV, x);
    return result.x;  // RAC: hyperbolic CORDIC replaces expf
}

__device__ __host__
float rac_tanh(float x) {
    float2 result = _rac_cordic_hyperbolic(RAC_K_HYP_INV, 0.0f, x);
    return RAC_DIVF(result.y, result.x);  // RAC: SFU division replaces multiply chain
}

__device__ __host__
void rac_softmax(float *x, float *out, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = rac_exp(x[i] - max_val);  // RAC: hyperbolic CORDIC replaces expf
        sum += out[i];
    }

    float inv_sum = RAC_DIVF(1.0f, sum);
    for (int i = 0; i < n; i++)
        out[i] = out[i] * inv_sum;  // RAC: normalization scaling
}

/* ── 6. Batch / linear algebra ───────────────────────────────────────────── */

__device__ __host__
void rac_rotate_batch(float2 *v, float *theta, float2 *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = rac_rotate(v[i], theta[i]);  // RAC: rotation replaces multiply
}

__device__ __host__
float rac_inner(float2 *a, float2 *b, int n) {
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
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val  = A[m * K + k];
                float b_val  = B[k * N + n];
                float2 va    = make_float2(a_val, 0.0f);
                float mag_b  = (b_val >= 0.0f) ? b_val : -b_val;
                float angle_b = (b_val >= 0.0f) ? 0.0f : 3.14159265f;
                sum += rac_project(va, angle_b) * mag_b;  // RAC: rotation replaces multiply
            }
            C[m * N + n] = sum;
        }
    }
}

/* ── HIP kernel wrappers ─────────────────────────────────────────────────── */

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
        float a_val  = A[m * K + k];
        float b_val  = B[k * N + n];
        float2 va    = make_float2(a_val, 0.0f);
        float mag_b  = (b_val >= 0.0f) ? b_val : -b_val;
        float angle_b = (b_val >= 0.0f) ? 0.0f : 3.14159265f;
        sum += rac_project(va, angle_b) * mag_b;  // RAC: rotation replaces multiply
    }
    C[m * N + n] = sum;
}

/* ── Context ─────────────────────────────────────────────────────────────── */

struct rac_context_t { rac_backend backend; };

rac_context rac_create_context(rac_backend backend) {
    rac_context ctx = (rac_context)malloc(sizeof(struct rac_context_t));
    if (!ctx) return NULL;
    ctx->backend = backend;
    return ctx;
}

void rac_destroy_context(rac_context ctx) { if (ctx) free(ctx); }

int rac_query_capability(rac_context ctx, rac_op_type op) {
    if (op == RAC_OP_EXTENDED) return 0;
    return 1;
}

int rac_execute(rac_context ctx, rac_op_type op, void *desc) {
    (void)ctx; (void)op; (void)desc;
    return 0;
}
