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
/* Hyperbolic CORDIC forward gain: K_hyp = prod sqrt(1 - 2^-2i). */
#define RAC_K_HYP      0.82816f
/* Inverse of the hyperbolic gain. Use as initial (x, y) so that
 * cordic_hyperbolic(K_HYP_INV, K_HYP_INV, z) produces e^z unscaled. */
#define RAC_K_HYP_INV  1.2074970f
/* Alias kept so both spellings resolve to the same (correct) reciprocal. */
#define RAC_K_HYP_RECIP RAC_K_HYP_INV

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

/* ── 5b. Tunable-precision CORDIC ───────────────────────────────────────── */
/*
 * Iteration count is a runtime knob. Fewer iterations -> lower precision,
 * lower latency, lower power. This mirrors what quantization-aware
 * training wants: training at 32 iters, inference at 16, edge at 8.
 *
 * Valid range: 4 <= iters <= 24. Values beyond 24 saturate (no extra
 * precision from the atan table) and are clamped.
 */
rac_vec2 rac_rotate_n(rac_vec2 v, float theta, int iters);
float    rac_project_n(rac_vec2 v, float theta, int iters);
void     rac_polar_n(rac_vec2 v, float *mag, float *angle, int iters);
float    rac_exp_n(float x, int iters);
float    rac_tanh_n(float x, int iters);

/*
 * rac_sincos: single CORDIC rotation of (K_INV, 0) by theta.
 * Writes sin(theta) to *s and cos(theta) to *c in one pass.
 * This is the core primitive behind RoPE, DCT, and general rotation.
 */
void     rac_sincos(float theta, float *s, float *c);

/*
 * rac_rsqrt: 1 / sqrt(x) via hyperbolic CORDIC vectoring.
 * This is the "layer-norm killer" — no multipliers in the critical path.
 *   x > 0 required. Returns 1/sqrt(x).
 *
 * Implementation: y = sqrt(x) from hyperbolic vectoring of (x+1, x-1)
 *   using the identity sqrt(x) = sqrt((x+1)^2 - (x-1)^2)/2, then 1/y
 *   via linear vectoring (SFU divide on GPU backends).
 */
float    rac_rsqrt(float x);

/*
 * rac_sigmoid: sigmoid(x) = 1 / (1 + e^-x).
 * Computed via the identity sigmoid(x) = 0.5 * (1 + tanh(x/2)),
 * which costs one hyperbolic CORDIC pass — no exp, no divide.
 */
float    rac_sigmoid(float x);

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

/* ── 10. Transformer primitives (native CORDIC) ─────────────────────────── */
/*
 * Every transformer primitive expressed in its most natural CORDIC mode:
 *
 *   QK^T / A@V         linear MAC         (rac_matmul)
 *   Softmax exp(x)     hyperbolic rotate  (rac_exp)
 *   Softmax normalize  linear vectoring   (divide)
 *   Layer norm mean    linear accumulate
 *   Layer norm rsqrt   hyperbolic vector  (rac_rsqrt)
 *   RMS norm rsqrt     hyperbolic vector  (rac_rsqrt)
 *   RoPE               circular rotate    (rac_rotate — native!)
 *   GELU / SiLU        hyperbolic+circle  (rac_sigmoid, rac_tanh)
 *
 * RoPE is the standout — rotary position embeddings are *literally*
 * Givens rotations. Every other accelerator spends multipliers
 * emulating them. RAC executes them natively.
 */

/*
 * rac_layernorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
 * Per-row (last-dim) normalization on a [rows, d] tensor.
 *   gamma, beta: length-d scale / shift, or NULL for (1, 0).
 *   eps:         numerical stability epsilon (e.g. 1e-5).
 * Uses rac_rsqrt (hyperbolic vectoring) for the 1/sqrt(var+eps) step.
 */
rac_status rac_layernorm(
    const float *x, float *y,
    const float *gamma, const float *beta,
    float eps, int rows, int d,
    const rac_config *cfg);

/*
 * rac_rmsnorm: y = gamma * x / sqrt(mean(x^2) + eps)
 * Root-Mean-Square normalization (Llama / T5 style). No mean subtraction.
 *   gamma: length-d scale, or NULL for 1.0.
 * Uses rac_rsqrt for the inverse-sqrt step.
 */
rac_status rac_rmsnorm(
    const float *x, float *y,
    const float *gamma,
    float eps, int rows, int d,
    const rac_config *cfg);

/*
 * rac_rope_cache: precompute sin/cos tables for a given head_dim and
 * maximum sequence length. Every pair of dims uses frequency
 *   inv_freq[i] = 1 / (base ^ (2i / head_dim))
 * and the rotation at position p for pair i is angle = p * inv_freq[i].
 *
 *   cos_out, sin_out: [max_seq, head_dim/2] row-major.
 *   base:             standard value is 10000.0f.
 */
rac_status rac_rope_cache(
    float *cos_out, float *sin_out,
    int max_seq, int head_dim, float base);

/*
 * rac_rope_apply: apply RoPE to a query or key tensor in place.
 *   x:          [batch, n_heads, seq, head_dim] row-major
 *   cos, sin:   [seq, head_dim/2] (from rac_rope_cache or user-supplied)
 *   Each 2D subvector (x[2i], x[2i+1]) is rotated by
 *     (cos[p,i], sin[p,i]) — one rac_rotate per pair — native CORDIC.
 */
rac_status rac_rope_apply(
    float *x,
    const float *cos_tab, const float *sin_tab,
    int batch, int n_heads, int seq, int head_dim,
    const rac_config *cfg);

/*
 * rac_scaled_dot_attention: fused scaled dot-product attention.
 *   out[b,h,t,d] = sum_s softmax_s( q @ k^T / sqrt(d_head) + mask ) * v
 *
 *   q, k, v: [batch, n_heads, seq, head_dim]
 *   mask:    optional [seq, seq] additive mask (or NULL). Causal mask
 *            can be baked into this tensor or enabled via `is_causal`.
 *   is_causal: non-zero -> apply upper-triangular -inf mask.
 *   out:     [batch, n_heads, seq, head_dim]
 *
 * Reference implementation — single-threaded CPU correctness anchor.
 */
rac_status rac_scaled_dot_attention(
    const float *q, const float *k, const float *v,
    const float *mask, int is_causal,
    float *out,
    int batch, int n_heads, int seq, int head_dim,
    const rac_config *cfg);

#ifdef __cplusplus
}
#endif

#endif /* RAC_CPU_H */
