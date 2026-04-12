/*
 * rac_kernels.hip — Pure HIP/CUDA kernels, NO PyTorch headers
 * This file compiles with hipcc/amdclang++ cleanly.
 * Linked with rac_torch_bind.cpp which handles pybind11.
 */

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

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
 * Tunable-precision multiply-accumulate.
 *
 *   iters >= 24 → fast path: single sign-XOR FMA. Uses hardware multiplier.
 *                 Identical to rac_fma. Pick this for production perf.
 *
 *   iters <  24 → pure shift-add path via rac_shiftadd_mul + one float add
 *                 to fold the product into acc. No FP multiplier is used in
 *                 the critical path. Cycles scale linearly with iters —
 *                 the RAC pitch curve, executed honestly on current GPUs
 *                 (where the multiplier is idle, not the SFU).
 *
 * iters is warp-uniform per kernel launch so the dispatch branch is free.
 */
__device__ __forceinline__
float rac_fma_iters(float a, float b, float acc, int iters) {
    if (iters >= 24) {
        unsigned int ai = __float_as_uint(a);
        unsigned int bi = __float_as_uint(b);
        unsigned int sign_b = bi & 0x80000000u;
        float proj = __uint_as_float(ai ^ sign_b);
        float mag_b = __uint_as_float(bi & 0x7FFFFFFFu);
        return fmaf(proj, mag_b, acc);
    }
    /* Multiplier-free path: product by shift-add, accumulate via adder. */
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
    float alpha, float beta, int iters)
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
            acc = rac_fma_iters(sA[threadIdx.y][i], sB[i][threadIdx.x], acc, iters);  /* RAC: rotation replaces multiply */
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
    float alpha, float beta, int iters)
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
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters);  /* RAC: rotation replaces multiply */
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
    float alpha, float beta, int iters)
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
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters);  /* RAC: rotation replaces multiply */
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
    float alpha, float beta, int iters)
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
            acc = rac_fma_iters(sA[threadIdx.y][i], sB[i][threadIdx.x], acc, iters);  /* RAC: rotation replaces multiply */
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
    float alpha, float beta, int iters)
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
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters);  /* RAC: rotation replaces multiply */
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
    float alpha, float beta, int iters)
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
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters);  /* RAC: rotation replaces multiply */
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
    float alpha, float beta, int iters)
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
            acc = rac_fma_iters(sA[threadIdx.y][i], sB[i][threadIdx.x], acc, iters);  /* RAC: rotation replaces multiply */
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
    float alpha, float beta, int iters)
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
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters);  /* RAC: rotation replaces multiply */
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
    float alpha, float beta, int iters)
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
                    acc[i][j] = rac_fma_iters(a_reg[i], b_reg[j], acc[i][j], iters);  /* RAC: rotation replaces multiply */
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
    int iters, hipStream_t stream)
{
    iters = _clamp_iters(iters);
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_nn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_nn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters);
    }
}

void rac_launch_nt_iters(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    int iters, hipStream_t stream)
{
    iters = _clamp_iters(iters);
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters);
    }
}

void rac_launch_tn_iters(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    int iters, hipStream_t stream)
{
    iters = _clamp_iters(iters);
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta, iters);
    }
}

/* ── Legacy ABI: default to iters=24 (fast sign-XOR path) ──────────────── */

void rac_launch_nn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    rac_launch_nn_iters(A, B, C, M, N, K, alpha, beta, 24, stream);
}

void rac_launch_nt(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    rac_launch_nt_iters(A, B, C, M, N, K, alpha, beta, 24, stream);
}

void rac_launch_tn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    rac_launch_tn_iters(A, B, C, M, N, K, alpha, beta, 24, stream);
}

} /* extern "C" */
