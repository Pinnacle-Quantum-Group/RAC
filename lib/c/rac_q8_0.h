/*
 * rac_q8_0.h — Q8_0 quantization format + matmul/GEMV
 * Pinnacle Quantum Group — April 2026
 *
 * Format matches llama.cpp's Q8_0 exactly:
 *   - weights split into blocks of 32 int8 values
 *   - each block has one fp16 scale factor
 *   - memory per block: 2 + 32 = 34 bytes (vs 128 bytes for f32)
 *   -> 4x memory bandwidth reduction, near-lossless quality
 *
 * Layout:
 *   struct rac_q8_0_block { uint16_t d; int8_t qs[32]; }
 *
 * Dequantized value: f[i] = f16_to_f32(d) * qs[i]
 * Quantization range: per-block amax / 127
 *
 * This is the Q8_0 that llama.cpp ships. Using the same format lets a
 * future RAC loader consume llama.cpp GGUFs directly, and lets us make
 * apples-to-apples decode-throughput comparisons on the same weights.
 */

#ifndef RAC_Q8_0_H
#define RAC_Q8_0_H

#include <stddef.h>
#include <stdint.h>
#include "rac_cpu.h"  /* for rac_activation + rac_status */

#ifdef __cplusplus
extern "C" {
#endif

#define RAC_Q8_0_BLOCK_SIZE 32

typedef struct {
    uint16_t d;                            /* fp16 scale */
    int8_t   qs[RAC_Q8_0_BLOCK_SIZE];      /* 32 int8 quantized weights */
} rac_q8_0_block;   /* 34 bytes, packed */

/* Number of Q8_0 blocks needed to hold N f32 elements. Requires N%32==0. */
static inline size_t rac_q8_0_blocks(size_t n) { return n / RAC_Q8_0_BLOCK_SIZE; }
static inline size_t rac_q8_0_bytes(size_t n)  { return rac_q8_0_blocks(n) * sizeof(rac_q8_0_block); }

/* ── fp16 <-> fp32 helpers ──────────────────────────────────────────────── */
/* Small inline converters — no external libs. IEEE 754 half <-> single.  */

float    rac_fp16_to_fp32(uint16_t h);
uint16_t rac_fp32_to_fp16(float f);

/* ── Quantization ───────────────────────────────────────────────────────── */

/*
 * rac_q8_0_quantize_row: convert one row of N f32 values into N/32 Q8_0
 * blocks. N must be a multiple of 32. Per-block amax scaling.
 */
void rac_q8_0_quantize_row(const float *src, rac_q8_0_block *dst, int N);

/*
 * rac_q8_0_quantize_matrix: quantize a full [rows, K] matrix row-by-row.
 * K must be a multiple of 32. Dst is [rows, K/32] blocks row-major.
 * OpenMP-parallel over rows.
 */
void rac_q8_0_quantize_matrix(const float *src, rac_q8_0_block *dst,
                               int rows, int K);

/* ── Dequantization (debug / verification) ──────────────────────────────── */

void rac_q8_0_dequantize_row(const rac_q8_0_block *src, float *dst, int N);

/* ── Q8_0 × F32 GEMV (decode hot path) ──────────────────────────────────── */

/*
 * rac_q8_0_gemv: output[N] = weight[N,K] @ input[K] (+ bias) (+ act)
 *
 *   input:   f32 activations [K]
 *   weight:  Q8_0 blocks     [N, K/32]
 *   bias:    f32 [N] or NULL
 *   output:  f32 [N]
 *   K:       must be multiple of 32
 *
 * Parallelizes over N. Inner loop dequantizes each block on-the-fly
 * (int8 -> int32 -> f32 -> vfmadd) so weight memory traffic is 4x less
 * than a f32 GEMV. This is the bandwidth-bound decode path that
 * closes the Q8_0 gap against llama.cpp.
 *
 * Picks AVX2 intrinsics when rac_avx2 is linked; otherwise scalar+OMP.
 */
rac_status rac_q8_0_gemv(
    const float *input,
    const rac_q8_0_block *weight,
    const float *bias,
    float *output,
    int N, int K,
    rac_activation act);

#ifdef __cplusplus
}
#endif

#endif /* RAC_Q8_0_H */
