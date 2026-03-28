/*
 * rac_kernels.hip — Pure HIP/CUDA kernels, NO PyTorch headers
 * This file compiles with hipcc/amdclang++ cleanly.
 * Linked with rac_torch_bind.cpp which handles pybind11.
 */

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

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
    float alpha, float beta)
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
            acc = fmaf(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);
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
    float alpha, float beta)
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
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
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
    float alpha, float beta)
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
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
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
    float alpha, float beta)
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
            acc = fmaf(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);
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
    float alpha, float beta)
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
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
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
    float alpha, float beta)
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
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
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
    float alpha, float beta)
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
            acc = fmaf(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);
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
    float alpha, float beta)
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
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
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
    float alpha, float beta)
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
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
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

void rac_launch_nn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_nn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_nn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
}

void rac_launch_nt(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_nt_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
}

void rac_launch_tn(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta,
    hipStream_t stream)
{
    if ((long long)M * N < 4096) {
        dim3 block(TILE_S, TILE_S);
        dim3 grid((N + TILE_S-1)/TILE_S, (M + TILE_S-1)/TILE_S);
        rac_matmul_small_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else if (M >= 128 && N >= 128 && K >= 128) {
        dim3 block(BN8/TN8, BM8/TM8);
        dim3 grid((N + BN8-1)/BN8, (M + BM8-1)/BM8);
        rac_matmul_micro8_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        dim3 block(BN/TN, BM/TM);
        dim3 grid((N + BN-1)/BN, (M + BM-1)/BM);
        rac_matmul_micro_tn_k<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
}

} /* extern "C" */
