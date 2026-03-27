/*
 * rac_avx2.h — RAC AVX2 SIMD-Accelerated Primitives
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * 8-wide vectorized operations using AVX2 + FMA intrinsics.
 * Process 8 floats per cycle for matmul, activations, and CORDIC.
 *
 * Build: cc -O3 -mavx2 -mfma rac_cpu.c rac_avx2.c -lm -fopenmp -shared -o librac_avx2.so
 *
 * Falls back to scalar on non-x86 or missing AVX2.
 */

#ifndef RAC_AVX2_H
#define RAC_AVX2_H

#include "rac_cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── AVX2 SGEMM: 8-wide vectorized inner loop ──────────────────────────── */

/*
 * rac_sgemm_avx2: C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
 *
 * Inner loop processes 8 columns of N simultaneously using __m256.
 * Combined with OpenMP over M for full core + SIMD utilization.
 * Falls back to rac_sgemm for the remainder (N % 8 != 0).
 */
rac_status rac_sgemm_avx2(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta,
    const rac_config *cfg);

/* ── AVX2 fused linear ──────────────────────────────────────────────────── */

rac_status rac_fused_linear_avx2(
    const float *input, const float *weight, const float *bias,
    float *output, int M, int N, int K,
    rac_activation act, const rac_config *cfg);

/* ── AVX2 batch activations ─────────────────────────────────────────────── */

void rac_relu_avx2(const float *x, float *out, int n);
void rac_gelu_avx2(const float *x, float *out, int n);
void rac_silu_avx2(const float *x, float *out, int n);

/* ── Runtime dispatch: auto-select AVX2 or scalar ───────────────────────── */

int rac_has_avx2(void);

#ifdef __cplusplus
}
#endif

#endif /* RAC_AVX2_H */
