/*
 * rac_cpu.h — RAC Portable CPU Library (C99)
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Standalone C library implementing all RAC primitives on CPU.
 * No CUDA/HIP/GPU dependency. Links with -lm only.
 *
 * Features:
 *   - All 17 RAC primitives (CORDIC + SFU-equivalent)
 *   - OpenMP parallelism for batch/matmul operations
 *   - Cache-tiled matmul with configurable tile size
 *   - x86 SSE/AVX intrinsics for vectorized CORDIC (optional)
 *   - Thread-safe, re-entrant, no global state
 *
 * Build:
 *   cc -O3 -march=native -fopenmp rac_cpu.c -lm -shared -o librac.so
 *   cc -O3 -march=native -fopenmp rac_cpu.c -lm -c -o rac_cpu.o
 *
 * MPI distributed matmul:
 *   mpicc -O3 -fopenmp rac_mpi.c rac_cpu.c -lm -o rac_mpi
 */

#ifndef RAC_CPU_H
#define RAC_CPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Types ──────────────────────────────────────────────────────────────── */

typedef struct { float x; float y; } rac_vec2;

/* ── Constants ──────────────────────────────────────────────────────────── */

#define RAC_K_INV      0.60725f
#define RAC_K          1.64676f
#define RAC_ITERS      16
#define RAC_ITERS_FAST 12
#define RAC_PI         3.14159265358979f

/*
 * Hyperbolic CORDIC gain constants. Note: this CORDIC sequence has
 * K_HYP < 1 (unlike circular K > 1). So to RECOVER cosh/sinh from the
 * gain-scaled output you DIVIDE by K_HYP (equivalently, multiply by
 * 1/K_HYP ≈ 1.20749).
 *
 * Historical note: the legacy macro RAC_K_HYP_INV is misnamed — its
 * numerical value (0.82816) is actually K_HYP, not its inverse. We
 * keep it for backward compatibility with rac_cuda.cu / rac_hip.cpp
 * but new code should prefer RAC_K_HYP (value-correct) and
 * RAC_K_HYP_RECIP (the true 1/K_HYP ≈ 1.20749).
 */
#define RAC_K_HYP       0.82816f     /* hyperbolic CORDIC gain */
#define RAC_K_HYP_RECIP 1.20749f     /* 1 / RAC_K_HYP — use to un-gain output */
#define RAC_K_HYP_INV   0.82816f     /* DEPRECATED alias for RAC_K_HYP
                                      * (kept for backward compatibility;
                                      * prefer RAC_K_HYP or RAC_K_HYP_RECIP) */

/* ── Error codes ────────────────────────────────────────────────────────── */

typedef enum {
    RAC_OK              = 0,
    RAC_ERR_NULL_PTR    = -1,
    RAC_ERR_INVALID_DIM = -2,
    RAC_ERR_ALLOC       = -3,
} rac_status;

/* ── Configuration ──────────────────────────────────────────────────────── */

typedef struct {
    int num_threads;    /* 0 = auto (OMP_NUM_THREADS or nproc) */
    int tile_size;      /* matmul cache tile size (default 64) */
    int cordic_iters;   /* CORDIC iterations (default 16, fast=12) */
} rac_config;

rac_config rac_default_config(void);

/* ── 1. Core rotation ───────────────────────────────────────────────────── */

rac_vec2 rac_rotate(rac_vec2 v, float theta);
rac_vec2 rac_rotate_raw(rac_vec2 v, float theta);
rac_vec2 rac_compensate(rac_vec2 v, int chain_length);
float    rac_project(rac_vec2 v, float theta);

/* ── 2. Polar / vectoring ───────────────────────────────────────────────── */

void     rac_polar(rac_vec2 v, float *mag, float *angle);
float    rac_norm(rac_vec2 v);
rac_vec2 rac_normalize(rac_vec2 v);

/* ── 3. Dot product / similarity ────────────────────────────────────────── */

float    rac_dot(rac_vec2 a, rac_vec2 b);
float    rac_coherence(rac_vec2 a, rac_vec2 b);

/* ── 4. Complex / DSP ───────────────────────────────────────────────────── */

rac_vec2 rac_complex_mul(rac_vec2 a, rac_vec2 b);
void     rac_dct(const float *x, float *X, int n);

/* ── 5. Hyperbolic / activations ────────────────────────────────────────── */

float    rac_exp(float x);
float    rac_tanh(float x);
void     rac_softmax(const float *x, float *out, int n);

/* ── 6. Batch operations (OpenMP parallelized) ──────────────────────────── */

void     rac_rotate_batch(const rac_vec2 *v, const float *theta,
                           rac_vec2 *out, int n);
float    rac_inner(const rac_vec2 *a, const rac_vec2 *b, int n);
void     rac_outer(const rac_vec2 *a, const rac_vec2 *b, float *C,
                    int m, int n);

/* ── 7. Matrix multiply (OpenMP + cache tiled) ──────────────────────────── */

/*
 * C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
 * Row-major layout. OpenMP parallelized over M dimension.
 * Cache-tiled for L1/L2 reuse.
 */
rac_status rac_sgemm(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta,
    const rac_config *cfg);

/* Convenience: C = A @ B (alpha=1, beta=0) */
rac_status rac_matmul(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    const rac_config *cfg);

/* ── 8. Activation functions (batch, OpenMP) ────────────────────────────── */

void rac_relu(const float *x, float *out, int n);
void rac_gelu(const float *x, float *out, int n);
void rac_silu(const float *x, float *out, int n);
void rac_softmax_batch(const float *x, float *out, int batch, int n);

/* ── 9. Fused linear: out = act(x @ W^T + bias) ────────────────────────── */

typedef enum {
    RAC_ACT_NONE = 0,
    RAC_ACT_RELU = 1,
    RAC_ACT_GELU = 2,
    RAC_ACT_SILU = 3,
} rac_activation;

rac_status rac_fused_linear(
    const float *input,      /* [M, K] */
    const float *weight,     /* [N, K] row-major (out_features x in_features) */
    const float *bias,       /* [N] or NULL */
    float *output,           /* [M, N] */
    int M, int N, int K,
    rac_activation act,
    const rac_config *cfg);

#ifdef __cplusplus
}
#endif

#endif /* RAC_CPU_H */
