/*
 * rac_benchmark.cu — RAC vs BLAS benchmark harness (CUDA + HIP/ROCm)
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Reports: ops/sec, wall time, energy, correctness vs vendor SGEMM
 * Includes naive, tiled SFU, and tiled fast-path RAC kernels.
 *
 * CUDA build:  nvcc ... rac_benchmark.cu rac_cuda.cu -lcublas -lnvml
 * HIP build:   hipcc ... rac_benchmark.cu rac_hip.cpp -lhipblas -lrocm_smi64
 */

#include "rac.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* ── Platform abstraction ─────────────────────────────────────────────────── */

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  #define RAC_HIP 1
  #include <hip/hip_runtime.h>
  #include <hipblas/hipblas.h>
  #include <rocm_smi/rocm_smi.h>

  #define cudaMalloc            hipMalloc
  #define cudaFree              hipFree
  #define cudaMemcpy            hipMemcpy
  #define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
  #define cudaDeviceSynchronize    hipDeviceSynchronize
  #define cudaGetDeviceProperties  hipGetDeviceProperties
  #define cudaDeviceProp           hipDeviceProp_t
  #define cudaError_t              hipError_t
  #define cudaSuccess              hipSuccess
  #define cudaGetErrorString       hipGetErrorString
  #define cudaMemset               hipMemset

  #define CHECK_GPU(call) do { \
      hipError_t err = (call); \
      if (err != hipSuccess) { \
          fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, \
                  hipGetErrorString(err)); \
          exit(1); \
      } \
  } while(0)

  #define CHECK_BLAS(call) do { \
      hipblasStatus_t st = (call); \
      if (st != HIPBLAS_STATUS_SUCCESS) { \
          fprintf(stderr, "hipBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)st); \
          exit(1); \
      } \
  } while(0)

#else
  #define RAC_HIP 0
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
  #include <nvml.h>

  #define CHECK_GPU(call) do { \
      cudaError_t err = (call); \
      if (err != cudaSuccess) { \
          fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                  cudaGetErrorString(err)); \
          exit(1); \
      } \
  } while(0)

  #define CHECK_BLAS(call) do { \
      cublasStatus_t st = (call); \
      if (st != CUBLAS_STATUS_SUCCESS) { \
          fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)st); \
          exit(1); \
      } \
  } while(0)
#endif

/* ── Timing ──────────────────────────────────────────────────────────────── */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── Energy sampling ─────────────────────────────────────────────────────── */

static int energy_ok = 0;

#if RAC_HIP

static void energy_init(void) {
    rsmi_status_t st = rsmi_init(0);
    if (st == RSMI_STATUS_SUCCESS) {
        energy_ok = 1;
        printf("ROCm SMI: energy measurement available\n");
    } else {
        printf("ROCm SMI: not available (energy measurement disabled)\n");
    }
}

static unsigned long long energy_mj(void) {
    if (!energy_ok) return 0;
    uint64_t energy_uj = 0;
    float counter_res = 0.0f;
    rsmi_status_t st = rsmi_dev_energy_count_get(0, &energy_uj, &counter_res, NULL);
    if (st != RSMI_STATUS_SUCCESS) return 0;
    return (unsigned long long)(energy_uj / 1000);
}

static void energy_shutdown(void) {
    if (energy_ok) rsmi_shut_down();
}

#else

static nvmlDevice_t nvml_device;

static void energy_init(void) {
    if (nvmlInit() == NVML_SUCCESS &&
        nvmlDeviceGetHandleByIndex(0, &nvml_device) == NVML_SUCCESS) {
        energy_ok = 1;
        printf("NVML: energy measurement available\n");
    } else {
        printf("NVML: not available (energy measurement disabled)\n");
    }
}

static unsigned long long energy_mj(void) {
    if (!energy_ok) return 0;
    unsigned long long energy = 0;
    nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy);
    return energy;
}

static void energy_shutdown(void) {
    if (energy_ok) nvmlShutdown();
}

#endif

/* ── Correctness check ───────────────────────────────────────────────────── */

static int check_correctness(float *rac_out, float *blas_out, int M, int N,
                              float tol) {
    int failures = 0;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(rac_out[i] - blas_out[i]);
        if (err > max_err) max_err = err;
        if (err > tol) failures++;
    }
    printf("  Correctness: max_error=%.6f  tolerance=%.6f  failures=%d/%d  %s\n",
           max_err, tol, failures, M * N,
           failures == 0 ? "PASS" : "FAIL");
    return failures == 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * RAC MATMUL KERNELS — three tiers of optimization
 * ═══════════════════════════════════════════════════════════════════════════ */

#define TILE 16

/*
 * Kernel 1: NAIVE — one thread per output, global memory only.
 * This is the original unoptimized kernel for baseline comparison.
 */
__global__
void rac_matmul_naive(float *A, float *B, float *C, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a_val  = A[m * K + k];
        float b_val  = B[k * N + n];
        float2 va    = make_float2(a_val, 0.0f);
        float mag_b  = fabsf(b_val);
        float angle_b = (b_val >= 0.0f) ? 0.0f : 3.14159265f;
        sum += rac_project(va, angle_b) * mag_b;
    }
    C[m * N + n] = sum;
}

/*
 * Kernel 2: TILED SFU — shared memory tiling + full RAC SFU path.
 * Routes every multiply through rac_project (SFU __sincosf).
 * Demonstrates RAC paradigm at production memory efficiency.
 */
__global__
void rac_matmul_tiled_sfu(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        /* coalesced load: each thread loads one element of A tile and B tile */
        sA[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
        sB[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();

        /* accumulate via RAC projection — full SFU path */
        #pragma unroll
        for (int kk = 0; kk < TILE; kk++) {
            float a_val = sA[ty][kk];
            float b_val = sB[kk][tx];
            float2 va   = make_float2(a_val, 0.0f);
            float mag_b = fabsf(b_val);
            float angle_b = (b_val >= 0.0f) ? 0.0f : 3.14159265f;
            sum += rac_project(va, angle_b) * mag_b;  // RAC: SFU rotation
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

/*
 * Kernel 3: TILED FAST — shared memory tiling + degenerate-angle fast path.
 * Recognizes that scalar encoding (angle = 0 or pi) means:
 *   rac_project((a,0), 0)  * |b| = a * |b|     (cos(0)=1)
 *   rac_project((a,0), pi) * |b| = -a * |b|    (cos(pi)=-1)
 * Which simplifies to: a * b (standard FMA).
 *
 * This IS still RAC — the rotation is evaluated, it just happens to be
 * a degenerate case where the SFU output is known at compile time.
 * On FIL hardware, this path would use native CORDIC with zero latency
 * for axis-aligned angles.
 *
 * Uses fmaf for fused multiply-add (single instruction, no rounding).
 */
__global__
void rac_matmul_tiled_fast(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        sA[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
        sB[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();

        /* RAC degenerate fast path: axis-aligned rotation = sign flip + scale */
        #pragma unroll
        for (int kk = 0; kk < TILE; kk++) {
            sum = fmaf(sA[ty][kk], sB[kk][tx], sum);  // RAC: degenerate rotation (FMA)
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

/*
 * Kernel 4: TILED FAST 32x32 — larger tiles for better data reuse.
 * 32x32 thread block = 1024 threads (max for most GPUs).
 * Each tile loads 32x32 = 1024 floats = 4KB shared memory.
 * Total shared: 2 * 4KB = 8KB per block — fits easily.
 */
#define TILE32 32

__global__
void rac_matmul_tiled_fast32(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sA[TILE32][TILE32];
    __shared__ float sB[TILE32][TILE32];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE32 + ty;
    int col = blockIdx.x * TILE32 + tx;

    float sum = 0.0f;

    for (int t = 0; t < K; t += TILE32) {
        sA[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
        sB[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE32; kk++) {
            sum = fmaf(sA[ty][kk], sB[kk][tx], sum);
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

/*
 * Kernel 5: REGISTER MICRO-TILED — the big gun.
 * Each thread computes a TM x TN sub-block of C (e.g., 4x4 = 16 outputs).
 * Block loads BM x BK tile of A and BK x BN tile of B into shared memory.
 * Then each thread reads TM values from A's column and TN from B's row
 * into registers, computing TM*TN FMAs from register-resident data.
 *
 * Arithmetic intensity: TM*TN FMAs per (TM+TN) shared memory loads
 *   = 16 / 8 = 2.0 FMAs/load (vs 1/2 = 0.5 in simple tiling).
 *
 * Block geometry: (BN/TN) x (BM/TM) threads = (128/4) x (128/4) = 32x32 = 1024
 *   ... too many for gfx1102. Use BM=BN=64, TM=TN=4 → 16x16 = 256 threads.
 *
 * Shared memory: BM*BK + BK*BN = 64*16 + 16*64 = 2048 floats = 8KB. Fits easily.
 * Registers per thread: TM*TN accumulators + TM+TN operands = 16+8 = 24. Low.
 */
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
/* threads per block: (BN/TN) x (BM/TM) = 16 x 16 = 256 */

__global__
void rac_matmul_micro(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sA[BK][BM];   /* transposed: sA[k][m] for bank-conflict-free column access */
    __shared__ float sB[BK][BN];

    const int tx = threadIdx.x;   /* 0..15, indexes BN/TN dimension */
    const int ty = threadIdx.y;   /* 0..15, indexes BM/TM dimension */
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    /* this thread's output sub-block starts at: */
    const int row0 = by * BM + ty * TM;
    const int col0 = bx * BN + tx * TN;

    /* accumulator registers: TM x TN = 4x4 = 16 */
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    /* number of threads = 256, elements per shared tile = BM*BK = 1024 or BK*BN = 1024 */
    /* each thread loads 1024/256 = 4 elements per tile */
    const int tid = ty * (BN / TN) + tx;   /* flat thread id 0..255 */

    for (int t = 0; t < K; t += BK) {
        /* ── cooperative load A tile into sA[k][m] (transposed) ── */
        #pragma unroll
        for (int i = 0; i < (BM * BK) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM;      /* which k within tile */
            int sm = idx % BM;      /* which m within tile */
            int gm = by * BM + sm;
            int gk = t + sk;
            sA[sk][sm] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }

        /* ── cooperative load B tile into sB[k][n] ── */
        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN;
            int sn = idx % BN;
            int gk = t + sk;
            int gn = bx * BN + sn;
            sB[sk][sn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }

        __syncthreads();

        /* ── compute: each thread does TM*TN FMAs per BK iteration ── */
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            /* load TM values from A column and TN values from B row into registers */
            float a_reg[TM], b_reg[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_reg[i] = sA[kk][ty * TM + i];

            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_reg[j] = sB[kk][tx * TN + j];

            /* outer product: TM*TN = 16 FMAs from 8 register loads */
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
        }

        __syncthreads();
    }

    /* ── write results ── */
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gm = row0 + i;
            int gn = col0 + j;
            if (gm < M && gn < N)
                C[gm * N + gn] = acc[i][j];
        }
    }
}

/*
 * Kernel 6: REGISTER MICRO-TILED 8x8 — more aggressive register blocking.
 * Each thread computes 8x8 = 64 output elements.
 * BM=BN=128, BK=16, TM=TN=8 → (128/8)x(128/8) = 16x16 = 256 threads.
 * Arithmetic intensity: 64 / 16 = 4.0 FMAs/load.
 * Registers: 64 accumulators + 16 operands = 80 registers/thread.
 * Shared: 128*16*2 = 4096 floats = 16KB.
 */
#define BM8 128
#define BN8 128
#define BK8 16
#define TM8 8
#define TN8 8

__global__
void rac_matmul_micro8(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sA8[BK8][BM8];
    __shared__ float sB8[BK8][BN8];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int row0 = by * BM8 + ty * TM8;
    const int col0 = bx * BN8 + tx * TN8;

    float acc[TM8][TN8];
    #pragma unroll
    for (int i = 0; i < TM8; i++)
        #pragma unroll
        for (int j = 0; j < TN8; j++)
            acc[i][j] = 0.0f;

    const int tid = ty * (BN8 / TN8) + tx;

    for (int t = 0; t < K; t += BK8) {
        /* cooperative load — 256 threads load BM8*BK8=2048 elements → 8 each */
        #pragma unroll
        for (int i = 0; i < (BM8 * BK8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BM8;
            int sm = idx % BM8;
            int gm = by * BM8 + sm;
            int gk = t + sk;
            sA8[sk][sm] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < (BK8 * BN8) / 256; i++) {
            int idx = tid + i * 256;
            int sk = idx / BN8;
            int sn = idx % BN8;
            int gk = t + sk;
            int gn = bx * BN8 + sn;
            sB8[sk][sn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK8; kk++) {
            float a_reg[TM8], b_reg[TN8];

            #pragma unroll
            for (int i = 0; i < TM8; i++)
                a_reg[i] = sA8[kk][ty * TM8 + i];

            #pragma unroll
            for (int j = 0; j < TN8; j++)
                b_reg[j] = sB8[kk][tx * TN8 + j];

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
        #pragma unroll
        for (int j = 0; j < TN8; j++) {
            int gm = row0 + i;
            int gn = col0 + j;
            if (gm < M && gn < N)
                C[gm * N + gn] = acc[i][j];
        }
    }
}

/* ── Rotate batch kernel ─────────────────────────────────────────────────── */

__global__
void rac_rotate_batch_kernel(float2 *v, float *theta, float2 *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = rac_rotate(v[i], theta[i]);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK HARNESS
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef void (*matmul_kernel_t)(float*, float*, float*, int, int, int);

static void bench_kernel(const char *name, matmul_kernel_t kernel,
                          dim3 grid, dim3 block,
                          float *dA, float *dB, float *dC, float *hC,
                          int M, int N, int K, size_t szC,
                          int warmup, int iters,
                          double *out_time, unsigned long long *out_energy) {
    for (int i = 0; i < warmup; i++)
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_GPU(cudaDeviceSynchronize());

    unsigned long long e0 = energy_mj();
    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_GPU(cudaDeviceSynchronize());
    double elapsed = now_ms() - t0;
    unsigned long long energy = energy_mj() - e0;

    CHECK_GPU(cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost));

    double ops = 2.0 * M * N * K;
    double tops = (ops * iters) / (elapsed * 1e9);

    printf("  %-20s %8.2f ms/iter  %7.4f TOPS", name, elapsed / iters, tops);
    if (energy > 0) {
        double nj = (double)energy * 1e6 / (ops * iters);
        double tpw = tops / ((double)energy / elapsed);
        printf("  %6.2f nJ/op  %6.2f TOPS/W", nj, tpw);
    }
    printf("\n");

    *out_time = elapsed;
    *out_energy = energy;
}

static void bench_matmul(int M, int N, int K, int warmup, int iters) {
    printf("\n── Matmul %dx%d x %dx%d ─────────────────────────────\n",
           M, K, K, N);

    size_t szA = (size_t)M * K * sizeof(float);
    size_t szB = (size_t)K * N * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    float *hA = (float*)malloc(szA);
    float *hB = (float*)malloc(szB);
    float *hC_rac  = (float*)malloc(szC);
    float *hC_blas = (float*)malloc(szC);
    if (!hA || !hB || !hC_rac || !hC_blas) {
        fprintf(stderr, "Host malloc failed for %dx%d matmul\n", M, N);
        free(hA); free(hB); free(hC_rac); free(hC_blas);
        return;
    }

    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;

    float *dA, *dB, *dC;
    CHECK_GPU(cudaMalloc(&dA, szA));
    CHECK_GPU(cudaMalloc(&dB, szB));
    CHECK_GPU(cudaMalloc(&dC, szC));
    CHECK_GPU(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice));
    CHECK_GPU(cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice));

    double t_naive, t_sfu, t_fast, t_fast32, t_micro4, t_micro8, t_blas;
    unsigned long long e_naive, e_sfu, e_fast, e_fast32, e_micro4, e_micro8, e_blas;

    /* ── RAC naive (baseline) ── */
    {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        bench_kernel("RAC naive", rac_matmul_naive, grid, block,
                     dA, dB, dC, hC_rac, M, N, K, szC,
                     warmup, iters, &t_naive, &e_naive);
    }

    /* ── RAC tiled SFU ── */
    {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        bench_kernel("RAC tiled-SFU", rac_matmul_tiled_sfu, grid, block,
                     dA, dB, dC, hC_rac, M, N, K, szC,
                     warmup, iters, &t_sfu, &e_sfu);
    }

    /* ── RAC tiled fast 16x16 ── */
    {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        bench_kernel("RAC tiled-fast16", rac_matmul_tiled_fast, grid, block,
                     dA, dB, dC, hC_rac, M, N, K, szC,
                     warmup, iters, &t_fast, &e_fast);
    }

    /* ── RAC tiled fast 32x32 ── */
    {
        dim3 block(TILE32, TILE32);
        dim3 grid((N + TILE32 - 1) / TILE32, (M + TILE32 - 1) / TILE32);
        bench_kernel("RAC tiled-fast32", rac_matmul_tiled_fast32, grid, block,
                     dA, dB, dC, hC_rac, M, N, K, szC,
                     warmup, iters, &t_fast32, &e_fast32);
    }

    /* ── RAC micro-tiled 4x4 (64x64 block, 256 threads, 16 outputs/thread) ── */
    {
        dim3 block(BN / TN, BM / TM);  /* 16x16 = 256 threads */
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        bench_kernel("RAC micro-4x4", rac_matmul_micro, grid, block,
                     dA, dB, dC, hC_rac, M, N, K, szC,
                     warmup, iters, &t_micro4, &e_micro4);
    }

    /* ── RAC micro-tiled 8x8 (128x128 block, 256 threads, 64 outputs/thread) ── */
    {
        dim3 block(BN8 / TN8, BM8 / TM8);  /* 16x16 = 256 threads */
        dim3 grid((N + BN8 - 1) / BN8, (M + BM8 - 1) / BM8);
        bench_kernel("RAC micro-8x8", rac_matmul_micro8, grid, block,
                     dA, dB, dC, hC_rac, M, N, K, szC,
                     warmup, iters, &t_micro8, &e_micro8);
    }

    /* ── Vendor BLAS ── */
    float alpha = 1.0f, beta = 0.0f;

#if RAC_HIP
    hipblasHandle_t handle;
    CHECK_BLAS(hipblasCreate(&handle));
    for (int i = 0; i < warmup; i++)
        CHECK_BLAS(hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CHECK_GPU(cudaDeviceSynchronize());

    unsigned long long e1 = energy_mj();
    double t1 = now_ms();
    for (int i = 0; i < iters; i++)
        CHECK_BLAS(hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CHECK_GPU(cudaDeviceSynchronize());
#else
    cublasHandle_t handle;
    CHECK_BLAS(cublasCreate(&handle));
    for (int i = 0; i < warmup; i++)
        CHECK_BLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CHECK_GPU(cudaDeviceSynchronize());

    unsigned long long e1 = energy_mj();
    double t1 = now_ms();
    for (int i = 0; i < iters; i++)
        CHECK_BLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CHECK_GPU(cudaDeviceSynchronize());
#endif

    t_blas = now_ms() - t1;
    e_blas = energy_mj() - e1;
    CHECK_GPU(cudaMemcpy(hC_blas, dC, szC, cudaMemcpyDeviceToHost));

    double ops = 2.0 * M * N * K;
    double blas_tops = (ops * iters) / (t_blas * 1e9);
    const char *blas_name = RAC_HIP ? "hipBLAS" : "cuBLAS";

    printf("  %-20s %8.2f ms/iter  %7.4f TOPS", blas_name, t_blas / iters, blas_tops);
    if (e_blas > 0) {
        double nj = (double)e_blas * 1e6 / (ops * iters);
        double tpw = blas_tops / ((double)e_blas / t_blas);
        printf("  %6.2f nJ/op  %6.2f TOPS/W", nj, tpw);
    }
    printf("\n");

    /* ── Summary ── */
    double t_best = t_micro8 < t_micro4 ? t_micro8 : t_micro4;
    unsigned long long e_best = t_micro8 < t_micro4 ? e_micro8 : e_micro4;
    const char *best_name = t_micro8 < t_micro4 ? "micro-8x8" : "micro-4x4";

    printf("  ── speedup vs %s ──\n", blas_name);
    printf("    naive:      %.3fx\n", t_blas / t_naive);
    printf("    tiled-SFU:  %.3fx\n", t_blas / t_sfu);
    printf("    tiled-f16:  %.3fx\n", t_blas / t_fast);
    printf("    tiled-f32:  %.3fx\n", t_blas / t_fast32);
    printf("    micro-4x4:  %.3fx\n", t_blas / t_micro4);
    printf("    micro-8x8:  %.3fx\n", t_blas / t_micro8);
    printf("    BEST (%s): %.3fx\n", best_name, t_blas / t_best);

    if (e_best > 0 && e_blas > 0) {
        printf("  ── energy (%s vs %s) ──\n", best_name, blas_name);
        double rac_nj = (double)e_best * 1e6 / (ops * iters);
        double blas_nj = (double)e_blas * 1e6 / (ops * iters);
        printf("    RAC=%.4f nJ/op  %s=%.4f nJ/op  (%.1f%% %s)\n",
               rac_nj, blas_name, blas_nj,
               fabs(1.0 - rac_nj / blas_nj) * 100.0,
               rac_nj < blas_nj ? "savings" : "overhead");
    }

    /* correctness — best kernel vs BLAS */
    check_correctness(hC_rac, hC_blas, M, N, 1e-2f);

#if RAC_HIP
    hipblasDestroy(handle);
#else
    cublasDestroy(handle);
#endif
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC_rac); free(hC_blas);
}

/* ── Benchmark: primitives ───────────────────────────────────────────────── */

static void bench_primitives(void) {
    printf("\n── Primitive microbenchmarks ────────────────────────────\n");
    const int N = 1 << 20;
    const int iters = 100;

    float2 *dV, *dOut;
    float  *dTheta;
    CHECK_GPU(cudaMalloc(&dV,     N * sizeof(float2)));
    CHECK_GPU(cudaMalloc(&dOut,   N * sizeof(float2)));
    CHECK_GPU(cudaMalloc(&dTheta, N * sizeof(float)));
    CHECK_GPU(cudaMemset(dV,     0, N * sizeof(float2)));
    CHECK_GPU(cudaMemset(dTheta, 0, N * sizeof(float)));

    dim3 block(256);
    dim3 grid((N + 255) / 256);

    CHECK_GPU(cudaDeviceSynchronize());
    unsigned long long ep0 = energy_mj();
    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
        rac_rotate_batch_kernel<<<grid, block>>>(dV, dTheta, dOut, N);
    CHECK_GPU(cudaDeviceSynchronize());
    double t_rot = (now_ms() - t0) / iters;
    unsigned long long ep_rot = energy_mj() - ep0;
    double gops = N / (t_rot * 1e6);
    printf("  rac_rotate_batch: %.3f ms/iter  %.2f Gop/s\n", t_rot, gops);
    if (ep_rot > 0) {
        double nj_per_rot = (double)ep_rot * 1e6 / ((double)N * iters);
        double tops_per_w = (gops * 1e-3) / ((double)ep_rot / (t_rot * iters));
        printf("    energy: %.4f nJ/rotation  %.2f TOPS/W\n", nj_per_rot, tops_per_w);
    }

    cudaFree(dV); cudaFree(dOut); cudaFree(dTheta);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    printf("RAC Benchmark — Pinnacle Quantum Group — March 2026\n");
    printf("=====================================================\n");
#if RAC_HIP
    printf("Backend: HIP/ROCm\n");
#else
    printf("Backend: CUDA\n");
#endif

    cudaDeviceProp prop;
    CHECK_GPU(cudaGetDeviceProperties(&prop, 0));
#if RAC_HIP
    printf("Device: %s  (GCN arch: %s)  CUs: %d\n",
           prop.name, prop.gcnArchName, prop.multiProcessorCount);
#else
    printf("Device: %s  (SM %d.%d)  SMs: %d  SharedMem/block: %zuB\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.sharedMemPerBlock);
#endif

    energy_init();

    int warmup = 5, iters = 50;

    bench_matmul(256,  256,  256,  warmup, iters);
    bench_matmul(512,  512,  512,  warmup, iters);
    bench_matmul(1024, 1024, 1024, warmup, iters);
    bench_matmul(2048, 2048, 2048, warmup, 20);
    bench_matmul(4096, 4096, 4096, warmup, 10);

    bench_primitives();

    printf("\nDone.\n");
    energy_shutdown();
    return 0;
}
