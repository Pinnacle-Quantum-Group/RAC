/*
 * rac_q8_0.c — Q8_0 quantization + GEMV kernel
 * Pinnacle Quantum Group — April 2026
 */

#include "rac_q8_0.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ── fp16 <-> fp32 (IEEE 754) ───────────────────────────────────────────── */

float rac_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant =  h        & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            /* subnormal half -> normalized single */
            int e = -14;
            while ((mant & 0x400) == 0) { mant <<= 1; e--; }
            mant &= 0x3ff;
            bits = (sign << 31) | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | (0xffu << 23) | (mant << 13);   /* inf / nan */
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } u;
    u.u = bits;
    return u.f;
}

uint16_t rac_fp32_to_fp16(float f) {
    union { float f; uint32_t u; } u;
    u.f = f;
    uint32_t b = u.u;
    uint32_t sign = (b >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((b >> 23) & 0xff) - 127 + 15;
    uint32_t mant = b & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;  /* underflow -> 0 */
        /* subnormal */
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));  /* inf / nan */
    }
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

/* ── Quantization ───────────────────────────────────────────────────────── */

void rac_q8_0_quantize_row(const float *src, rac_q8_0_block *dst, int N) {
    int nb = N / RAC_Q8_0_BLOCK_SIZE;
    for (int b = 0; b < nb; b++) {
        const float *blk = src + b * RAC_Q8_0_BLOCK_SIZE;
        float amax = 0.0f;
        for (int i = 0; i < RAC_Q8_0_BLOCK_SIZE; i++) {
            float a = fabsf(blk[i]);
            if (a > amax) amax = a;
        }
        float d  = amax / 127.0f;
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        dst[b].d = rac_fp32_to_fp16(d);
        for (int i = 0; i < RAC_Q8_0_BLOCK_SIZE; i++) {
            int q = (int)lrintf(blk[i] * id);
            if (q < -127) q = -127;   /* matches llama.cpp: symmetric [-127, 127] */
            if (q >  127) q =  127;
            dst[b].qs[i] = (int8_t)q;
        }
    }
}

void rac_q8_0_quantize_matrix(const float *src, rac_q8_0_block *dst,
                               int rows, int K) {
    int bpr = K / RAC_Q8_0_BLOCK_SIZE;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        rac_q8_0_quantize_row(src + (size_t)r * K,
                               dst + (size_t)r * bpr, K);
    }
}

void rac_q8_0_dequantize_row(const rac_q8_0_block *src, float *dst, int N) {
    int nb = N / RAC_Q8_0_BLOCK_SIZE;
    for (int b = 0; b < nb; b++) {
        float d = rac_fp16_to_fp32(src[b].d);
        float *o = dst + b * RAC_Q8_0_BLOCK_SIZE;
        for (int i = 0; i < RAC_Q8_0_BLOCK_SIZE; i++) {
            o[i] = d * (float)src[b].qs[i];
        }
    }
}

/* ── Q8_0 × F32 GEMV ────────────────────────────────────────────────────── */

#if defined(__AVX2__)
/*
 * AVX2 hot path. For each output row:
 *   - Four __m256 accumulators across one 32-element block (8 floats * 4)
 *   - Load 32 int8 weights, sign-extend 8 at a time to int32, convert to f32
 *   - vfmadd with the aligned input lane
 *   - At block end, scale once by the fp16 -> f32 block scale, reduce-sum
 *
 * Peak throughput on Zen3 is ~4 vfmadd231ps per cycle per core; this
 * loop issues 4 FMAs + 4 vpmovsxbd + 4 vcvtdq2ps + 4 loads per block.
 * Memory traffic per block: 2 + 32 = 34 bytes weight, 128 bytes input
 * (input lanes reused across N so usually cached).
 */
static inline float q8_0_block_dot_avx2(
    const rac_q8_0_block *blk, const float *in)
{
    __m128i qw_lo = _mm_loadu_si128((const __m128i *)(blk->qs));      /* 16 i8 */
    __m128i qw_hi = _mm_loadu_si128((const __m128i *)(blk->qs + 16)); /* 16 i8 */

    __m256i w0 = _mm256_cvtepi8_epi32(qw_lo);                        /*  8 i32 */
    __m256i w1 = _mm256_cvtepi8_epi32(_mm_srli_si128(qw_lo, 8));
    __m256i w2 = _mm256_cvtepi8_epi32(qw_hi);
    __m256i w3 = _mm256_cvtepi8_epi32(_mm_srli_si128(qw_hi, 8));

    __m256 f0 = _mm256_cvtepi32_ps(w0);
    __m256 f1 = _mm256_cvtepi32_ps(w1);
    __m256 f2 = _mm256_cvtepi32_ps(w2);
    __m256 f3 = _mm256_cvtepi32_ps(w3);

    __m256 i0 = _mm256_loadu_ps(in +  0);
    __m256 i1 = _mm256_loadu_ps(in +  8);
    __m256 i2 = _mm256_loadu_ps(in + 16);
    __m256 i3 = _mm256_loadu_ps(in + 24);

    __m256 acc = _mm256_mul_ps(f0, i0);
    acc = _mm256_fmadd_ps(f1, i1, acc);
    acc = _mm256_fmadd_ps(f2, i2, acc);
    acc = _mm256_fmadd_ps(f3, i3, acc);

    /* horizontal add of the 8 lanes */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif /* __AVX2__ */

static inline float q8_0_block_dot_scalar(
    const rac_q8_0_block *blk, const float *in)
{
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < RAC_Q8_0_BLOCK_SIZE; i++) {
        s += (float)blk->qs[i] * in[i];
    }
    return s;
}

static inline float _q8_0_apply_act(float s, rac_activation act) {
    switch (act) {
        case RAC_ACT_RELU: return s > 0.0f ? s : 0.0f;
        case RAC_ACT_GELU: return 0.5f * s * (1.0f + erff(s * 0.7071067811865f));
        case RAC_ACT_SILU: return s / (1.0f + expf(-s));
        default:           return s;
    }
}

rac_status rac_q8_0_gemv(
    const float *input,
    const rac_q8_0_block *weight,
    const float *bias,
    float *output,
    int N, int K,
    rac_activation act)
{
    if (!input || !weight || !output) return RAC_ERR_NULL_PTR;
    if (N <= 0 || K <= 0 || (K % RAC_Q8_0_BLOCK_SIZE) != 0)
        return RAC_ERR_INVALID_DIM;

    const int nb = K / RAC_Q8_0_BLOCK_SIZE;

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j++) {
        const rac_q8_0_block *row = weight + (size_t)j * nb;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const float *in = input + b * RAC_Q8_0_BLOCK_SIZE;
#if defined(__AVX2__)
            float dot = q8_0_block_dot_avx2(&row[b], in);
#else
            float dot = q8_0_block_dot_scalar(&row[b], in);
#endif
            acc += rac_fp16_to_fp32(row[b].d) * dot;
        }
        if (bias) acc += bias[j];
        output[j] = _q8_0_apply_act(acc, act);
    }
    return RAC_OK;
}
