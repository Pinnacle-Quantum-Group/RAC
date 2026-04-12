/*
 * rac_kernels.hip — Pure HIP/CUDA kernels, NO PyTorch headers
 * This file compiles with hipcc/amdclang++ cleanly.
 * Linked with rac_torch_bind.cpp which handles pybind11.
 *
 * Three matmul compute paths, selected by `mode`:
 *   0 = FAST     — sign-XOR + single sign-XOR FMA. Hardware multiplier
 *                  engaged, best throughput. The production default.
 *   1 = CORDIC   — port of _rac_cordic_rotate_raw from rac_cuda.cu. N
 *                  iterative shift-add-structured micro-rotations followed
 *                  by a projection FMA. On GPU, the * scale operations
 *                  compile to FP multiplies; on a RAC ASIC each is a
 *                  literal bit-shift. Iteration count = `iters`.
 *   2 = SHIFTADD — pure integer shift-add multiply (rac_shiftadd_mul).
 *                  No FP multiplier or FMA is engaged. Integer ALU only.
 *                  Iteration count = `iters`; precision ≈ iters bits.
 */

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

#define RAC_MODE_FAST     0
#define RAC_MODE_CORDIC   1
#define RAC_MODE_SHIFTADD 2

#define RAC_ITERS_DEFAULT 16

/* CORDIC arctangent lookup: atan(2^-i) for i=0..23.
 * Same values used in rac_cuda.cu / lib/c/rac_cpu.c. Extended past
 * RAC_ITERS=16 so runtime iters can reach 24. */
__constant__ float rac_atan_table[24] = {
    0.78539816f, 0.46364761f, 0.24497866f, 0.12435499f,
    0.06241881f, 0.03123983f, 0.01562373f, 0.00781234f,
    0.00390623f, 0.00195312f, 0.00097656f, 0.00048828f,
    0.00024414f, 0.00012207f, 0.00006104f, 0.00003052f,
    0.00001526f, 0.00000763f, 0.00000381f, 0.00000191f,
    0.00000095f, 0.00000048f, 0.00000024f, 0.00000012f,
};

/* CORDIC circular gain K(n) = prod_{i=0..n-1} sqrt(1 + 2^-2i), indexed 0..24. */
__constant__ float rac_k_table[25] = {
    1.00000000f, 1.41421356f, 1.58113883f, 1.62979295f,
    1.64248460f, 1.64568447f, 1.64648404f, 1.64668388f,
    1.64673383f, 1.64674632f, 1.64674944f, 1.64675022f,
    1.64675042f, 1.64675047f, 1.64675048f, 1.64675049f,
    1.64675049f, 1.64675049f, 1.64675049f, 1.64675049f,
    1.64675049f, 1.64675049f, 1.64675049f, 1.64675049f,
    1.64675049f,
};

/* RAC primitives use sign decomposition — no lookup tables needed */

__device__ __forceinline__
float rac_fma(float a, float b, float acc) {
    unsigned int ai = __float_as_uint(a);
    unsigned int bi = __float_as_uint(b);
    unsigned int sign_b = bi & 0x80000000u;
    float proj = __uint_as_float(ai ^ sign_b);       /* RAC: rotation via sign XOR */
    float mag_b = __uint_as_float(bi & 0x7FFFFFFFu); /* |b| */
    return fmaf(proj, mag_b, acc);                    /* RAC: project + accumulate */
}

/*
 * rac_shiftadd_mul: a * b via pure integer shift-add.
 *
 * NO hardware multiplier and NO FMA is engaged in the computation — only
 * integer shifts, adds, masks, a count-leading-zeros, and one final
 * sign/exponent bit-pack to produce the fp32 result.
 *
 * This is the honest no-multiplier compute path the RAC pitch describes.
 * On a dedicated ASIC each loop iteration is one shift-add cycle; on a GPU
 * we're deliberately leaving the FP multiplier idle, which is slower in
 * raw TOPS but demonstrates the algorithmic claim.
 *
 * `iters` selects how many top bits of b's mantissa are consumed.
 *   24 → full fp32 precision (but prefer rac_fma_iters' fast path)
 *   16 → ~16 bits of precision, bf16-ish
 *    8 → ~8 bits, int8-ish
 *    4 → ~4 bits, int4-ish
 *
 * Absolute error is bounded by roughly 2^-iters * |a*b|.
 */
__device__ __forceinline__
float rac_shiftadd_mul(float a, float b, int iters) {
    if (a == 0.0f || b == 0.0f) return 0.0f;

    unsigned int ai = __float_as_uint(a);
    unsigned int bi = __float_as_uint(b);
    unsigned int sign = (ai ^ bi) & 0x80000000u;
    int exp_a = (int)((ai >> 23) & 0xFFu) - 127;
    int exp_b = (int)((bi >> 23) & 0xFFu) - 127;
    /* 24-bit mantissas with the implicit leading 1 reinstated. */
    unsigned int ma = 0x00800000u | (ai & 0x007FFFFFu);
    unsigned int mb = 0x00800000u | (bi & 0x007FFFFFu);

    /* Conditional shift-add: for each of the top `iters` bits of mb,
     * add ma (shifted to align with the bit's weight) into the 48-bit
     * product accumulator. One shift + mask + compare + conditional add
     * per iteration. No multiplier, anywhere. */
    unsigned long long prod = 0ULL;
    int lim = iters < 1 ? 1 : (iters > 24 ? 24 : iters);
    for (int i = 0; i < lim; i++) {
        unsigned int bit = (mb >> (23 - i)) & 1u;
        if (bit) prod += ((unsigned long long)ma) << (23 - i);
    }
    if (prod == 0ULL) return 0.0f;

    /* Normalize: find the leading 1 and realign into the fp32 mantissa. */
    int top = 63 - __clzll(prod);            /* top set bit position */
    int shift_r = top - 23;
    unsigned int mantissa;
    if (shift_r > 0)      mantissa = (unsigned int)(prod >> shift_r) & 0x007FFFFFu;
    else if (shift_r < 0) mantissa = (unsigned int)(prod << -shift_r) & 0x007FFFFFu;
    else                  mantissa = (unsigned int)prod & 0x007FFFFFu;

    int rexp = exp_a + exp_b + (top - 46) + 127;
    if (rexp <= 0 || rexp >= 0xFF) return 0.0f;  /* flush sub-/super-normals */

    unsigned int bits = sign | ((unsigned int)(rexp & 0xFFu) << 23) | mantissa;
    return __uint_as_float(bits);
}

/*
 * rac_cordic_rotate_n: iterative CORDIC circular rotation — port of
 * _rac_cordic_rotate_raw from rac_cuda.cu (commodity-GPU structure).
 *
 * Each iteration is a shift-add-structured micro-rotation:
 *   x_new = x - d * y * scale     // scale = 2^-i  (ASIC: bit-shift)
 *   y_new = y + d * x * scale
 *   angle -= d * atan(2^-i)
 * Over `iters` iterations, accumulates a rotation of v by theta, with the
 * standard CORDIC gain K(iters) applied to the output magnitude.
 *
 * On GPU the `* scale` lines emit FP multiplies. On a RAC ASIC the
 * compiler target would emit literal shifts. Same algebra, different
 * silicon mapping.
 */
__device__ __forceinline__
float2 rac_cordic_rotate_n(float2 v, float theta, int iters) {
    /* Wrap to (-pi, pi] */
    const float PI  = 3.14159265358979f;
    const float TAU = 6.28318530717958f;
    while (theta >  PI) theta -= TAU;
    while (theta <= -PI) theta += TAU;

    /* Pre-rotate into [-pi/2, pi/2] so CORDIC converges. */
    float x = v.x, y = v.y;
    if (theta >  PI * 0.5f) { x = -x; y = -y; theta -= PI; }
    else if (theta < -PI * 0.5f) { x = -x; y = -y; theta += PI; }

    float angle = theta;
    float scale = 1.0f;
    int lim = iters < 1 ? 1 : (iters > 24 ? 24 : iters);
    for (int i = 0; i < lim; i++) {
        float d = (angle >= 0.0f) ? 1.0f : -1.0f;
        float xn = x - d * y * scale;      /* RAC: shift replaces multiply (2^-i) */
        float yn = y + d * x * scale;      /* RAC: shift replaces multiply (2^-i) */
        angle -= d * rac_atan_table[i];
        x = xn;
        y = yn;
        scale *= 0.5f;                     /* RAC: power-of-2 scale (bit shift) */
    }
    return make_float2(x, y);
}

/*
 * rac_fma_cordic: a * b + acc via CORDIC projection.
 * Matches rac_matmul_kernel from rac_cuda.cu:
 *   va = (a, 0); angle_b = (b>=0) ? 0 : pi; mag_b = |b|
 *   return fma(rotate(va, angle_b).x * K^-1, mag_b, acc)
 * With sign-decomposed b, the N CORDIC iterations still run but the
 * target angle is 0 after pre-rotation, so precision is full fp32 modulo
 * gain-compensation error. Cycle cost scales linearly with iters.
 */
__device__ __forceinline__
float rac_fma_cordic(float a, float b, float acc, int iters) {
    unsigned int bi = __float_as_uint(b);
    float mag_b = __uint_as_float(bi & 0x7FFFFFFFu);
    float angle_b = (bi & 0x80000000u) ? 3.14159265358979f : 0.0f;
    float2 rotated = rac_cordic_rotate_n(make_float2(a, 0.0f), angle_b, iters);
    int lim = iters < 1 ? 1 : (iters > 24 ? 24 : iters);
    float k_inv = 1.0f / rac_k_table[lim];     /* gain compensation */
    return fmaf(rotated.x * k_inv, mag_b, acc);
}

/*
 * Tunable-precision multiply-accumulate — three compute paths.
 *
 *   mode == FAST (0):     sign-XOR FMA. Hardware multiplier engaged.
 *                         iters ignored. Production perf.
 *   mode == CORDIC (1):   iterative CORDIC projection (N = iters).
 *                         Structure matches rac_cuda.cu. On GPU uses
 *                         FP multiplies for scale ops; on ASIC these
 *                         are shifts. Iteration count is the knob.
 *   mode == SHIFTADD (2): pure integer shift-add multiply + one add.
 *                         FP multiplier and FMA sit idle. Integer ALU
 *                         only. Precision ≈ iters bits.
 *
 * mode and iters are warp-uniform per kernel launch, so both dispatch
 * branches are free.
 */
__device__ __forceinline__
float rac_fma_iters(float a, float b, float acc, int iters, int mode) {
    if (mode == RAC_MODE_FAST) {
        unsigned int ai = __float_as_uint(a);
        unsigned int bi = __float_as_uint(b);
        unsigned int sign_b = bi & 0x80000000u;
        float proj = __uint_as_float(ai ^ sign_b);
        float mag_b = __uint_as_float(bi & 0x7FFFFFFFu);
        return fmaf(proj, mag_b, acc);
    }
    if (mode == RAC_MODE_CORDIC) {
        return rac_fma_cordic(a, b, acc, iters);
    }
    /* RAC_MODE_SHIFTADD — multiplier-free integer path. */
    float prod = rac_shiftadd_mul(a, b, iters);
    return acc + prod;
}

/* ── Tile parameters ──────────────────────────────────────────────────── */

#define TILE_S 16

#define BM   64
#define BN   64
#define BK   16
#define TM   4
#define TN   4

#define BM8 128
#define BN8 128
#define BK8 16
#define TM8 8
#define TN8 8

/* ── Tier 1: Small tiled kernel ──────────────────────────────────────── */

__global__
void rac_matmul_small_k(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta, int iters, int mode)
{
    __shared__ float sA[TILE_S][TILE_S];
    __shared__ float sB[TILE_S][TILE_S];

    int row = blockIdx.y * TILE_S + threadIdx.y;
    int col = blockIdx.x * TILE_S + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE_S) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_S; i++)
            acc = rac_fma_iters(sA[threadIdx.y][i], sB[i][threadIdx.x], acc, iters, mode);  /* RAC: rotation replaces multiply */
        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f)
            C[row * N + col] = fmaf(alpha, acc, beta * C[row * N + col]);
        else
            C[row * N + col] = alpha * acc;
    }
}

/* ── Tier 2: Micro-4×4 NN kernel ────────────────────────────────────── */

__global__
void rac_matmul_micro_nn_k(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta, int iters, int mode)
{
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM + ty * TM;
    const int col0 = bx * BN + tx * TN;
    const int tid = ty * (BN / TN) + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK) {
        #pragma unroll
        for (int i = 0; i < (BM * BK) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM, sm = idx % BM;
            int gm = by * BM + sm, gk = t + sk;
            sA[sk][sm] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN, sn = idx % BN;
            int gk = t + sk, gn = bx * BN + sn;
            sB[sk][sn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a_reg[i] = sA[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_reg[j] = sB[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters, mode);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── Tier 3: Micro-8×8 NN kernel ────────────────────────────────────── */

__global__
void rac_matmul_micro8_nn_k(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta, int iters, int mode)
{
    __shared__ float sA8[BK8][BM8];
    __shared__ float sB8[BK8][BN8];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM8 + ty * TM8;
    const int col0 = bx * BN8 + tx * TN8;
    const int tid = ty * (BN8 / TN8) + tx;

    float acc[TM8][TN8];
    #pragma unroll
    for (int i = 0; i < TM8; i++)
        #pragma unroll
        for (int j = 0; j < TN8; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK8) {
        #pragma unroll
        for (int i = 0; i < (BM8 * BK8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM8, sm = idx % BM8;
            int gm = by * BM8 + sm, gk = t + sk;
            sA8[sk][sm] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK8 * BN8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN8, sn = idx % BN8;
            int gk = t + sk, gn = bx * BN8 + sn;
            sB8[sk][sn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK8; kk++) {
            float a_reg[TM8], b_reg[TN8];
            #pragma unroll
            for (int i = 0; i < TM8; i++) a_reg[i] = sA8[kk][ty * TM8 + i];
            #pragma unroll
            for (int j = 0; j < TN8; j++) b_reg[j] = sB8[kk][tx * TN8 + j];
            #pragma unroll
            for (int i = 0; i < TM8; i++)
                #pragma unroll
                for (int j = 0; j < TN8; j++)
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters, mode);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM8; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN8; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── Tier 1: Small tiled NT kernel (B transposed) ──────────────────── */

__global__
void rac_matmul_small_nt_k(
    const float* __restrict__ A,
    const float* __restrict__ B,   /* [N, K] row-major, used transposed */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta, int iters, int mode)
{
    __shared__ float sA[TILE_S][TILE_S];
    __shared__ float sB[TILE_S][TILE_S];

    int row = blockIdx.y * TILE_S + threadIdx.y;
    int col = blockIdx.x * TILE_S + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE_S) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[col * K + bRow] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_S; i++)
            acc = rac_fma_iters(sA[threadIdx.y][i], sB[i][threadIdx.x], acc, iters, mode);  /* RAC: rotation replaces multiply */
        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f)
            C[row * N + col] = fmaf(alpha, acc, beta * C[row * N + col]);
        else
            C[row * N + col] = alpha * acc;
    }
}

/* ── Tier 2: Micro-4×4 NT kernel (B transposed) ───────────────────── */

__global__
void rac_matmul_micro_nt_k(
    const float* __restrict__ A,   /* [M, K] */
    const float* __restrict__ B,   /* [N, K] row-major, used transposed */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta, int iters, int mode)
{
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM + ty * TM;
    const int col0 = bx * BN + tx * TN;
    const int tid = ty * (BN / TN) + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK) {
        #pragma unroll
        for (int i = 0; i < (BM * BK) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM, sm = idx % BM;
            int gm = by * BM + sm, gk = t + sk;
            sA[sk][sm] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN, sn = idx % BN;
            int gk = t + sk, gn = bx * BN + sn;
            sB[sk][sn] = (gk < K && gn < N) ? B[gn * K + gk] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a_reg[i] = sA[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_reg[j] = sB[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters, mode);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── Tier 3: Micro-8×8 NT kernel (B transposed) ───────────────────── */

__global__
void rac_matmul_micro8_nt_k(
    const float* __restrict__ A,   /* [M, K] */
    const float* __restrict__ B,   /* [N, K] row-major, used transposed */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta, int iters, int mode)
{
    __shared__ float sA8[BK8][BM8];
    __shared__ float sB8[BK8][BN8];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM8 + ty * TM8;
    const int col0 = bx * BN8 + tx * TN8;
    const int tid = ty * (BN8 / TN8) + tx;

    float acc[TM8][TN8];
    #pragma unroll
    for (int i = 0; i < TM8; i++)
        #pragma unroll
        for (int j = 0; j < TN8; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK8) {
        #pragma unroll
        for (int i = 0; i < (BM8 * BK8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM8, sm = idx % BM8;
            int gm = by * BM8 + sm, gk = t + sk;
            sA8[sk][sm] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK8 * BN8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN8, sn = idx % BN8;
            int gk = t + sk, gn = bx * BN8 + sn;
            sB8[sk][sn] = (gk < K && gn < N) ? B[gn * K + gk] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK8; kk++) {
            float a_reg[TM8], b_reg[TN8];
            #pragma unroll
            for (int i = 0; i < TM8; i++) a_reg[i] = sA8[kk][ty * TM8 + i];
            #pragma unroll
            for (int j = 0; j < TN8; j++) b_reg[j] = sB8[kk][tx * TN8 + j];
            #pragma unroll
            for (int i = 0; i < TM8; i++)
                #pragma unroll
                for (int j = 0; j < TN8; j++)
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters, mode);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM8; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN8; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── Tier 1: Small tiled TN kernel (A transposed) ─────────────────── */

__global__
void rac_matmul_small_tn_k(
    const float* __restrict__ A,   /* [K, M] row-major, used transposed */
    const float* __restrict__ B,   /* [K, N] */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta, int iters, int mode)
{
    __shared__ float sA[TILE_S][TILE_S];
    __shared__ float sB[TILE_S][TILE_S];

    int row = blockIdx.y * TILE_S + threadIdx.y;
    int col = blockIdx.x * TILE_S + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE_S) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[aCol * M + row] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_S; i++)
            acc = rac_fma_iters(sA[threadIdx.y][i], sB[i][threadIdx.x], acc, iters, mode);  /* RAC: rotation replaces multiply */
        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f)
            C[row * N + col] = fmaf(alpha, acc, beta * C[row * N + col]);
        else
            C[row * N + col] = alpha * acc;
    }
}

/* ── Tier 2: Micro-4×4 TN kernel (A transposed) ───────────────────── */

__global__
void rac_matmul_micro_tn_k(
    const float* __restrict__ A,   /* [K, M] row-major, used transposed */
    const float* __restrict__ B,   /* [K, N] */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta, int iters, int mode)
{
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM + ty * TM;
    const int col0 = bx * BN + tx * TN;
    const int tid = ty * (BN / TN) + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK) {
        #pragma unroll
        for (int i = 0; i < (BM * BK) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM, sm = idx % BM;
            int gm = by * BM + sm, gk = t + sk;
            sA[sk][sm] = (gm < M && gk < K) ? A[gk * M + gm] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN, sn = idx % BN;
            int gk = t + sk, gn = bx * BN + sn;
            sB[sk][sn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a_reg[i] = sA[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_reg[j] = sB[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters, mode);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── Tier 3: Micro-8×8 TN kernel (A transposed) ───────────────────── */

__global__
void rac_matmul_micro8_tn_k(
    const float* __restrict__ A,   /* [K, M] row-major, used transposed */
    const float* __restrict__ B,   /* [K, N] */
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta, int iters, int mode)
{
    __shared__ float sA8[BK8][BM8];
    __shared__ float sB8[BK8][BN8];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int row0 = by * BM8 + ty * TM8;
    const int col0 = bx * BN8 + tx * TN8;
    const int tid = ty * (BN8 / TN8) + tx;

    float acc[TM8][TN8];
    #pragma unroll
    for (int i = 0; i < TM8; i++)
        #pragma unroll
        for (int j = 0; j < TN8; j++)
            acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK8) {
        #pragma unroll
        for (int i = 0; i < (BM8 * BK8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM8, sm = idx % BM8;
            int gm = by * BM8 + sm, gk = t + sk;
            sA8[sk][sm] = (gm < M && gk < K) ? A[gk * M + gm] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (BK8 * BN8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN8, sn = idx % BN8;
            int gk = t + sk, gn = bx * BN8 + sn;
            sB8[sk][sn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK8; kk++) {
            float a_reg[TM8], b_reg[TN8];
            #pragma unroll
            for (int i = 0; i < TM8; i++) a_reg[i] = sA8[kk][ty * TM8 + i];
            #pragma unroll
            for (int j = 0; j < TN8; j++) b_reg[j] = sB8[kk][tx * TN8 + j];
            #pragma unroll
            for (int i = 0; i < TM8; i++)
                #pragma unroll
                for (int j = 0; j < TN8; j++)
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters, mode);  /* RAC: rotation replaces multiply */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM8; i++) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN8; j++) {
            int gn = col0 + j;
            if (gn < N) {
                if (beta != 0.0f)
                    C[gm * N + gn] = fmaf(alpha, acc[i][j], beta * C[gm * N + gn]);
                else
                    C[gm * N + gn] = alpha * acc[i][j];
            }
        }
    }
}

/* ── C-callable dispatch functions ───────────────────────────────────── */

extern "C" {

/* Internal launchers take explicit `iters`. The legacy rac_launch_* wrappers
 * below default iters=24 (the fast sign-XOR path) for backward compat with
 * any caller that hasn't been updated. Callers that want the tunable knob
 * use rac_launch_*_iters directly. */

static inline int _clamp_iters(int iters) {
    if (iters < 1) return 1;
    if (iters > 24) return 24;
    return iters;
}

void rac_launch_nn_iters(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    int iters, int mode, hipStream_t stream)
{
    iters = _clamp_iters(iters);
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters, mode);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_nn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters, mode);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_nn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters, mode);
    }
}

void rac_launch_nt_iters(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    int iters, int mode, hipStream_t stream)
{
    iters = _clamp_iters(iters);
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters, mode);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters, mode);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters, mode);
    }
}

void rac_launch_tn_iters(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    int iters, int mode, hipStream_t stream)
{
    iters = _clamp_iters(iters);
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters, mode);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters, mode);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters, mode);
    }
}

/* ── Legacy ABI: default to iters=24, mode=FAST (sign-XOR FMA) ─────────── */

void rac_launch_nn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    rac_launch_nn_iters(A, B, C, M, N, K, alpha, beta, 24, RAC_MODE_FAST, stream);
}

void rac_launch_nt(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    rac_launch_nt_iters(A, B, C, M, N, K, alpha, beta, 24, RAC_MODE_FAST, stream);
}

void rac_launch_tn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    rac_launch_tn_iters(A, B, C, M, N, K, alpha, beta, 24, RAC_MODE_FAST, stream);
}

} /* extern "C" */
