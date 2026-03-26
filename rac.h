#pragma once
#ifndef RAC_H
#define RAC_H

/*
 * rac.h — Rotation-Accumulate (RAC) Primitive Interface
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * RAC replaces Multiply-Accumulate (MAC) with CORDIC-based geometric rotation.
 * All 17 primitives operate in float32. No fixed-point types are exposed.
 * All matrix/vector inputs are standard float arrays.
 *
 * CORDIC constants:
 *   K     = 1.64676   (CORDIC gain, 16-iteration)
 *   K_INV = 0.60725   (gain compensation factor)
 *
 * MAC equivalence:
 *   Traditional:  a * b                = rac_project((float2){a,0}, atan2f(b,1))
 *   Traditional:  sum(a[i] * b[i])     = rac_inner(a, b, n)
 *   Traditional:  A * B (matmul)       = rac_matmul(A, B, C, M, N, K)
 */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__CUDACC__)
  #include <cuda_runtime.h>
  #define RAC_FLOAT2 float2
#elif defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #define RAC_FLOAT2 float2
#else
  typedef struct { float x; float y; } float2;
  #define RAC_FLOAT2 float2
#endif

#define RAC_K_INV  0.60725f   /* CORDIC gain compensation: K^-1 */
#define RAC_K      1.64676f   /* CORDIC gain factor */
#define RAC_ITERS  16         /* CORDIC iteration count */

/* ── Opaque context handle ───────────────────────────────────────────────── */

typedef struct rac_context_t* rac_context;

typedef enum {
    RAC_BACKEND_CUDA    = 0,   /* NVIDIA GPU via CUDA SFUs          */
    RAC_BACKEND_HIP     = 1,   /* AMD GPU via ROCm SFUs             */
    RAC_BACKEND_CPU     = 2,   /* CPU reference implementation      */
    RAC_BACKEND_FIL     = 3    /* FIL hardware (proprietary)        */
} rac_backend;

typedef enum {
    RAC_OP_ROTATE       = 0,
    RAC_OP_ROTATE_RAW   = 1,
    RAC_OP_PROJECT      = 2,
    RAC_OP_POLAR        = 3,
    RAC_OP_NORM         = 4,
    RAC_OP_NORMALIZE    = 5,
    RAC_OP_DOT          = 6,
    RAC_OP_COHERENCE    = 7,
    RAC_OP_COMPLEX_MUL  = 8,
    RAC_OP_DCT          = 9,
    RAC_OP_EXP          = 10,
    RAC_OP_TANH         = 11,
    RAC_OP_SOFTMAX      = 12,
    RAC_OP_ROTATE_BATCH = 13,
    RAC_OP_INNER        = 14,
    RAC_OP_OUTER        = 15,
    RAC_OP_MATMUL       = 16,
    RAC_OP_EXTENDED     = 0xFF  /* reserved — FIL proprietary ops   */
} rac_op_type;

rac_context rac_create_context(rac_backend backend);
void        rac_destroy_context(rac_context ctx);
int         rac_query_capability(rac_context ctx, rac_op_type op);
int         rac_execute(rac_context ctx, rac_op_type op, void *desc);

/* ── 1. Core rotation ────────────────────────────────────────────────────── */

/*
 * rac_rotate: rotate vector v by angle theta (radians).
 * Output is CORDIC-gain-compensated (magnitude preserved).
 * Use for single rotations. For chained rotations, use rac_rotate_raw
 * and apply K_INV once at the end of the chain.
 *
 * Returns: rotated float2, magnitude == |v|
 * Accurate to ±2^-16 for theta in [-π, π]
 */
__device__ __host__ float2 rac_rotate(float2 v, float theta);

/*
 * rac_rotate_raw: rotate vector v by angle theta, NO gain compensation.
 * Output magnitude is scaled by K = 1.64676 per call.
 * Use for chained rotations: apply K_INV^N manually after N calls,
 * or call rac_compensate() once at chain end.
 *
 * Returns: rotated float2, magnitude == |v| * K
 */
__device__ __host__ float2 rac_rotate_raw(float2 v, float theta);

/*
 * rac_compensate: apply gain compensation to a raw-rotated vector.
 * Call once after a chain of N rac_rotate_raw calls.
 * compensates for K^N by applying K_INV^N.
 */
__device__ __host__ float2 rac_compensate(float2 v, int chain_length);

/*
 * rac_project: rotate v by theta, return x-component (signed scalar).
 * This is the MAC-equivalent degenerate case:
 *   rac_project((a,0), atan2f(b,a)) == a*cos(atan2(b,a)) == MAC(a,b)/norm
 *
 * More generally: returns dot product of v with unit vector at angle theta.
 *   result = v.x*cos(theta) + v.y*sin(theta)
 * Sign is preserved — negative when v opposes the projection axis.
 *
 * Returns: signed scalar float
 */
__device__ __host__ float  rac_project(float2 v, float theta);

/* ── 2. Polar / vectoring ────────────────────────────────────────────────── */

/*
 * rac_polar: cartesian to polar via CORDIC vectoring mode.
 * Computes magnitude and angle simultaneously in one CORDIC pass.
 * *mag   = sqrt(v.x^2 + v.y^2)
 * *angle = atan2(v.y, v.x)
 */
__device__ __host__ void   rac_polar(float2 v, float *mag, float *angle);

/*
 * rac_norm: euclidean magnitude via CORDIC vectoring.
 * Returns: sqrt(v.x^2 + v.y^2)
 */
__device__ __host__ float  rac_norm(float2 v);

/*
 * rac_normalize: unit vector via fused CORDIC vectoring + rotation.
 * Returns: v / |v|
 */
__device__ __host__ float2 rac_normalize(float2 v);

/* ── 3. Dot product / similarity ─────────────────────────────────────────── */

/*
 * rac_dot: dot product via CORDIC angle subtraction.
 * a.b = |a||b|cos(angle_a - angle_b)
 * Computed as: rac_norm(a) * rac_project(b_normalized, angle_a)
 * Returns: signed scalar
 */
__device__ __host__ float  rac_dot(float2 a, float2 b);

/*
 * rac_coherence: cosine similarity in [-1, 1].
 * cos(angle_a - angle_b) — independent of magnitude.
 * Returns: 1.0 when aligned, -1.0 when opposed, 0.0 when orthogonal.
 */
__device__ __host__ float  rac_coherence(float2 a, float2 b);

/* ── 4. Complex / DSP ────────────────────────────────────────────────────── */

/*
 * rac_complex_mul: complex multiplication via rotation.
 * (a + bi)(c + di) = rotate(a, atan2(d,c)) scaled by |(c,d)|
 * Implemented as: rac_rotate(a, atan2(b.y, b.x)) scaled by rac_norm(b)
 * Returns: float2 result
 */
__device__ __host__ float2 rac_complex_mul(float2 a, float2 b);

/*
 * rac_dct: DCT-II via CORDIC basis rotation.
 * X[k] = sum_n x[n] * cos(pi*(2n+1)*k / 2N)
 * Each cosine basis evaluated via CORDIC rotation, no FPU multiply.
 * x:   input array, length n
 * X:   output array, length n
 */
__device__ __host__ void   rac_dct(float *x, float *X, int n);

/* ── 5. Hyperbolic / activations ─────────────────────────────────────────── */

/*
 * rac_exp: e^x via hyperbolic CORDIC (shift-add, no FPU expf).
 * Uses Walther unified CORDIC in hyperbolic mode.
 * Valid range: x in [-88, 88] (float32 exp range)
 */
__device__ __host__ float  rac_exp(float x);

/*
 * rac_tanh: tanh(x) via hyperbolic CORDIC.
 * tanh(x) = (e^x - e^-x) / (e^x + e^-x)
 * Computed in single hyperbolic CORDIC pass.
 */
__device__ __host__ float  rac_tanh(float x);

/*
 * rac_softmax: softmax via batched hyperbolic CORDIC.
 * out[i] = rac_exp(x[i]) / sum_j rac_exp(x[j])
 * Numerically stable: subtracts max(x) before exponentiation.
 * x:   input array, length n
 * out: output array, length n (sum == 1.0)
 */
__device__ __host__ void   rac_softmax(float *x, float *out, int n);

/* ── 6. Batch / linear algebra ───────────────────────────────────────────── */

/*
 * rac_rotate_batch: independent batch rotation.
 * out[i] = rac_rotate(v[i], theta[i]) for i in [0, n)
 * All rotations are independent — fully parallelizable.
 */
__device__ __host__ void   rac_rotate_batch(float2 *v, float *theta,
                                            float2 *out, int n);

/*
 * rac_inner: inner product via paired rotation-project-accumulate.
 * result = sum_i rac_project(a[i], angle(b[i])) * norm(b[i])
 * Equivalent to sum_i (a[i].x*b[i].x + a[i].y*b[i].y)
 * Zero multiply operators in compute path.
 * Returns: signed scalar
 */
__device__ __host__ float  rac_inner(float2 *a, float2 *b, int n);

/*
 * rac_outer: outer product via rotation-project.
 * C[i][j] = rac_project(a[i], angle(b[j])) * norm(b[j])
 * C is stored row-major, dimensions m x n.
 */
__device__ __host__ void   rac_outer(float2 *a, float2 *b, float *C,
                                     int m, int n);

/*
 * rac_matmul: matrix multiply via rotation-project-accumulate.
 * C[m,n] = A[m,k] * B[k,n] expressed as RAC operations.
 * A, B, C are float arrays (row-major). Backend encodes internally.
 * Zero multiply operators in compute kernel.
 */
__device__ __host__ void   rac_matmul(float *A, float *B, float *C,
                                      int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif /* RAC_H */
