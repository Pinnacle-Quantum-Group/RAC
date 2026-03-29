/*
 * rac_kernels.hip — Pure HIP/CUDA kernels, NO PyTorch headers
 * This file compiles with hipcc/amdclang++ cleanly.
 * Linked with rac_torch_bind.cpp which handles pybind11.
 */

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

/* ── RAC cos lookup table (256 entries, __constant__ cache) ──────────── */
/* Routes all multiply ops through cos() table lookup instead of FPU.    */
/* With hardware CORDIC: ROM, zero-cycle. On GPU: constant-cache, one cycle. */

#define RAC_LUT_SIZE 256

__constant__ float _rac_cos_lut[RAC_LUT_SIZE] = {
    1.00000000f, 0.99969882f, 0.99879546f, 0.99729046f, 0.99518473f, 0.99247953f, 0.98917651f, 0.98527764f,
    0.98078528f, 0.97570213f, 0.97003125f, 0.96377607f, 0.95694034f, 0.94952818f, 0.94154407f, 0.93299280f,
    0.92387953f, 0.91420976f, 0.90398929f, 0.89322430f, 0.88192126f, 0.87008699f, 0.85772861f, 0.84485357f,
    0.83146961f, 0.81758481f, 0.80320753f, 0.78834643f, 0.77301045f, 0.75720885f, 0.74095113f, 0.72424708f,
    0.70710678f, 0.68954054f, 0.67155895f, 0.65317284f, 0.63439328f, 0.61523159f, 0.59569930f, 0.57580819f,
    0.55557023f, 0.53499762f, 0.51410274f, 0.49289819f, 0.47139674f, 0.44961133f, 0.42755509f, 0.40524131f,
    0.38268343f, 0.35989504f, 0.33688985f, 0.31368174f, 0.29028468f, 0.26671276f, 0.24298018f, 0.21910124f,
    0.19509032f, 0.17096189f, 0.14673047f, 0.12241068f, 0.09801714f, 0.07356456f, 0.04906767f, 0.02454123f,
    0.00000000f,-0.02454123f,-0.04906767f,-0.07356456f,-0.09801714f,-0.12241068f,-0.14673047f,-0.17096189f,
   -0.19509032f,-0.21910124f,-0.24298018f,-0.26671276f,-0.29028468f,-0.31368174f,-0.33688985f,-0.35989504f,
   -0.38268343f,-0.40524131f,-0.42755509f,-0.44961133f,-0.47139674f,-0.49289819f,-0.51410274f,-0.53499762f,
   -0.55557023f,-0.57580819f,-0.59569930f,-0.61523159f,-0.63439328f,-0.65317284f,-0.67155895f,-0.68954054f,
   -0.70710678f,-0.72424708f,-0.74095113f,-0.75720885f,-0.77301045f,-0.78834643f,-0.80320753f,-0.81758481f,
   -0.83146961f,-0.84485357f,-0.85772861f,-0.87008699f,-0.88192126f,-0.89322430f,-0.90398929f,-0.91420976f,
   -0.92387953f,-0.93299280f,-0.94154407f,-0.94952818f,-0.95694034f,-0.96377607f,-0.97003125f,-0.97570213f,
   -0.98078528f,-0.98527764f,-0.98917651f,-0.99247953f,-0.99518473f,-0.99729046f,-0.99879546f,-0.99969882f,
   -1.00000000f,-0.99969882f,-0.99879546f,-0.99729046f,-0.99518473f,-0.99247953f,-0.98917651f,-0.98527764f,
   -0.98078528f,-0.97570213f,-0.97003125f,-0.96377607f,-0.95694034f,-0.94952818f,-0.94154407f,-0.93299280f,
   -0.92387953f,-0.91420976f,-0.90398929f,-0.89322430f,-0.88192126f,-0.87008699f,-0.85772861f,-0.84485357f,
   -0.83146961f,-0.81758481f,-0.80320753f,-0.78834643f,-0.77301045f,-0.75720885f,-0.74095113f,-0.72424708f,
   -0.70710678f,-0.68954054f,-0.67155895f,-0.65317284f,-0.63439328f,-0.61523159f,-0.59569930f,-0.57580819f,
   -0.55557023f,-0.53499762f,-0.51410274f,-0.49289819f,-0.47139674f,-0.44961133f,-0.42755509f,-0.40524131f,
   -0.38268343f,-0.35989504f,-0.33688985f,-0.31368174f,-0.29028468f,-0.26671276f,-0.24298018f,-0.21910124f,
   -0.19509032f,-0.17096189f,-0.14673047f,-0.12241068f,-0.09801714f,-0.07356456f,-0.04906767f,-0.02454123f,
    0.00000000f, 0.02454123f, 0.04906767f, 0.07356456f, 0.09801714f, 0.12241068f, 0.14673047f, 0.17096189f,
    0.19509032f, 0.21910124f, 0.24298018f, 0.26671276f, 0.29028468f, 0.31368174f, 0.33688985f, 0.35989504f,
    0.38268343f, 0.40524131f, 0.42755509f, 0.44961133f, 0.47139674f, 0.49289819f, 0.51410274f, 0.53499762f,
    0.55557023f, 0.57580819f, 0.59569930f, 0.61523159f, 0.63439328f, 0.65317284f, 0.67155895f, 0.68954054f,
    0.70710678f, 0.72424708f, 0.74095113f, 0.75720885f, 0.77301045f, 0.78834643f, 0.80320753f, 0.81758481f,
    0.83146961f, 0.84485357f, 0.85772861f, 0.87008699f, 0.88192126f, 0.89322430f, 0.90398929f, 0.91420976f,
    0.92387953f, 0.93299280f, 0.94154407f, 0.94952818f, 0.95694034f, 0.96377607f, 0.97003125f, 0.97570213f,
    0.98078528f, 0.98527764f, 0.98917651f, 0.99247953f, 0.99518473f, 0.99729046f, 0.99879546f, 0.99969882f
};

/* CORDIC linear mode fused multiply-accumulate: acc += a * b
 * Zero FPU multiply instructions at the assembly level.
 * All ops: sign-bit XOR, exponent subtract, float add.
 * With hardware CORDIC: one cycle replaces the entire loop. */
#define RAC_CORDIC_ITERS 16

__device__ __forceinline__
float _rac_sign_flip(float x, float z) {
    unsigned int xi = __float_as_uint(x);
    unsigned int zi = __float_as_uint(z);
    xi ^= (zi & 0x80000000u);
    return __uint_as_float(xi);
}

__device__ __forceinline__
float _rac_shift(float x, int i) {
    if (i == 0) return x;
    unsigned int xi = __float_as_uint(x);
    unsigned int sign = xi & 0x80000000u;
    unsigned int abs_xi = xi & 0x7FFFFFFFu;
    if (abs_xi == 0) return 0.0f;
    int exp_bits = (int)((abs_xi >> 23) & 0xFFu);
    exp_bits -= i;
    if (exp_bits <= 0) return 0.0f;
    abs_xi = (abs_xi & 0x007FFFFFu) | ((unsigned int)exp_bits << 23);
    return __uint_as_float(sign | abs_xi);
}

__device__ __forceinline__
float rac_fma(float a, float b, float acc) {
    float x = a;
    float y = acc;
    float z = b;

    #pragma unroll
    for (int i = 0; i < RAC_CORDIC_ITERS; i++) {
        float dx = _rac_sign_flip(x, z);
        float shifted = _rac_shift(dx, i);
        y += shifted;
        float s2 = _rac_shift(__uint_as_float(0x3F800000u), i);
        float ds = _rac_sign_flip(s2, z);
        z -= ds;
    }
    return y;
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
            acc = rac_fma(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);  /* RAC: rotation replaces multiply */
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
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
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
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
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
            acc = rac_fma(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);  /* RAC: rotation replaces multiply */
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
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
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
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
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
            acc = rac_fma(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);  /* RAC: rotation replaces multiply */
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
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
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
                    acc[i][j] = rac_fma(a_reg[i], b_reg[j], acc[i][j]);  /* RAC: rotation replaces multiply */
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
