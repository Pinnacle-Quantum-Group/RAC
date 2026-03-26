/*
 * rac_benchmark.cu — RAC vs BLAS benchmark harness (CUDA + HIP/ROCm)
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Reports: ops/sec, wall time, energy, correctness vs vendor SGEMM
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

  /* map CUDA names to HIP equivalents for shared code paths */
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

/* ROCm SMI energy measurement */
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
    uint64_t energy_uj = 0;    /* ROCm SMI reports in microjoules */
    float counter_res = 0.0f;
    rsmi_status_t st = rsmi_dev_energy_count_get(0, &energy_uj, &counter_res, NULL);
    if (st != RSMI_STATUS_SUCCESS) return 0;
    return (unsigned long long)(energy_uj / 1000);  /* convert uJ → mJ */
}

static void energy_shutdown(void) {
    if (energy_ok) rsmi_shut_down();
}

#else

/* NVML energy measurement */
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

/* ── RAC matmul kernel (GPU) ─────────────────────────────────────────────── */

extern __global__ void rac_matmul_kernel(float *A, float *B, float *C,
                                          int M, int N, int K);

/* ── Benchmark: single size ──────────────────────────────────────────────── */

static void bench_matmul(int M, int N, int K, int warmup, int iters) {
    printf("\n── Matmul %dx%d x %dx%d ─────────────────────────────\n",
           M, K, K, N);

    size_t szA = (size_t)M * K * sizeof(float);
    size_t szB = (size_t)K * N * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    /* host data */
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
    for (int i = 0; i < M * K; i++) hA[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    /* device data */
    float *dA, *dB, *dC;
    CHECK_GPU(cudaMalloc(&dA, szA));
    CHECK_GPU(cudaMalloc(&dB, szB));
    CHECK_GPU(cudaMalloc(&dC, szC));
    CHECK_GPU(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice));
    CHECK_GPU(cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    /* ── RAC benchmark ── */
    for (int i = 0; i < warmup; i++)
        rac_matmul_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_GPU(cudaDeviceSynchronize());

    unsigned long long e0 = energy_mj();
    double t0 = now_ms();

    for (int i = 0; i < iters; i++)
        rac_matmul_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_GPU(cudaDeviceSynchronize());

    double t_rac = now_ms() - t0;
    unsigned long long e_rac = energy_mj() - e0;

    CHECK_GPU(cudaMemcpy(hC_rac, dC, szC, cudaMemcpyDeviceToHost));

    double ops = 2.0 * M * N * K;  /* flops per matmul */
    double rac_tops = (ops * iters) / (t_rac * 1e9);

    printf("  RAC:    %.2f ms/iter  %.4f TOPS  energy=%llu mJ\n",
           t_rac / iters, rac_tops, e_rac / (unsigned long long)iters);
    if (e_rac > 0) {
        double rac_nj_per_op = (double)e_rac * 1e6 / (ops * iters);  /* mJ → nJ */
        double rac_tops_per_w = rac_tops / ((double)e_rac / (t_rac));  /* TOPS/(mJ/ms)=TOPS/W */
        printf("  RAC energy/op: %.4f nJ/op  %.2f TOPS/W\n", rac_nj_per_op, rac_tops_per_w);
    }

    /* ── Vendor BLAS benchmark ── */
    float alpha = 1.0f, beta = 0.0f;

#if RAC_HIP
    hipblasHandle_t handle;
    CHECK_BLAS(hipblasCreate(&handle));

    /* hipBLAS is column-major; transpose trick for row-major inputs */
    for (int i = 0; i < warmup; i++)
        CHECK_BLAS(hipblasSgemm(handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, M, K, &alpha,
            dB, N, dA, K, &beta, dC, N));
    CHECK_GPU(cudaDeviceSynchronize());

    unsigned long long e1 = energy_mj();
    double t1 = now_ms();

    for (int i = 0; i < iters; i++)
        CHECK_BLAS(hipblasSgemm(handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, M, K, &alpha,
            dB, N, dA, K, &beta, dC, N));
    CHECK_GPU(cudaDeviceSynchronize());
#else
    cublasHandle_t handle;
    CHECK_BLAS(cublasCreate(&handle));

    /* cuBLAS is column-major; transpose trick for row-major inputs */
    for (int i = 0; i < warmup; i++)
        CHECK_BLAS(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            dB, N, dA, K, &beta, dC, N));
    CHECK_GPU(cudaDeviceSynchronize());

    unsigned long long e1 = energy_mj();
    double t1 = now_ms();

    for (int i = 0; i < iters; i++)
        CHECK_BLAS(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            dB, N, dA, K, &beta, dC, N));
    CHECK_GPU(cudaDeviceSynchronize());
#endif

    double t_blas = now_ms() - t1;
    unsigned long long e_blas = energy_mj() - e1;
    double blas_tops = (ops * iters) / (t_blas * 1e9);

    CHECK_GPU(cudaMemcpy(hC_blas, dC, szC, cudaMemcpyDeviceToHost));

    const char *blas_name = RAC_HIP ? "hipBLAS" : "cuBLAS";
    printf("  %s: %.2f ms/iter  %.4f TOPS  energy=%llu mJ\n",
           blas_name, t_blas / iters, blas_tops, e_blas / (unsigned long long)iters);
    if (e_blas > 0) {
        double blas_nj_per_op = (double)e_blas * 1e6 / (ops * iters);
        double blas_tops_per_w = blas_tops / ((double)e_blas / (t_blas));
        printf("  %s energy/op: %.4f nJ/op  %.2f TOPS/W\n", blas_name, blas_nj_per_op, blas_tops_per_w);
    }
    printf("  Speedup: RAC/%s = %.3fx\n", blas_name, t_blas / t_rac);

    if (e_rac > 0 && e_blas > 0) {
        printf("  Energy ratio: RAC/%s = %.3fx\n",
               blas_name, (double)e_rac / (double)e_blas);
        double rac_nj = (double)e_rac * 1e6 / (ops * iters);
        double blas_nj = (double)e_blas * 1e6 / (ops * iters);
        printf("  Energy/op:    RAC=%.4f nJ  %s=%.4f nJ  (%.1f%% %s)\n",
               rac_nj, blas_name, blas_nj,
               fabs(1.0 - rac_nj / blas_nj) * 100.0,
               rac_nj < blas_nj ? "savings" : "overhead");
    }

    /* correctness — RAC vs BLAS (row-major trick: B^T * A^T in col-major = A*B row-major) */
    check_correctness(hC_rac, hC_blas, M, N, 1e-3f);

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
    const int N = 1 << 20;  /* 1M elements */
    const int iters = 100;

    float2 *dV, *dOut;
    float  *dTheta;
    CHECK_GPU(cudaMalloc(&dV,     N * sizeof(float2)));
    CHECK_GPU(cudaMalloc(&dOut,   N * sizeof(float2)));
    CHECK_GPU(cudaMalloc(&dTheta, N * sizeof(float)));

    /* zero-initialize device memory for deterministic results */
    {
        void *zV = calloc(N, sizeof(float2));
        void *zT = calloc(N, sizeof(float));
        if (zV) { CHECK_GPU(cudaMemcpy(dV, zV, N * sizeof(float2), cudaMemcpyHostToDevice)); free(zV); }
        if (zT) { CHECK_GPU(cudaMemcpy(dTheta, zT, N * sizeof(float), cudaMemcpyHostToDevice)); free(zT); }
    }

    /* rotate_batch */
    dim3 block(256);
    dim3 grid((N + 255) / 256);

    extern __global__ void rac_rotate_batch_kernel(float2*, float*, float2*, int);

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

    /* device info */
    cudaDeviceProp prop;
    CHECK_GPU(cudaGetDeviceProperties(&prop, 0));
#if RAC_HIP
    printf("Device: %s  (GCN arch: %s)  CUs: %d\n",
           prop.name, prop.gcnArchName, prop.multiProcessorCount);
#else
    printf("Device: %s  (SM %d.%d)  SMs: %d\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
#endif

    energy_init();

    int warmup = 5, iters = 50;

    /* sweep matmul sizes */
    bench_matmul(256,  256,  256,  warmup, iters);
    bench_matmul(512,  512,  512,  warmup, iters);
    bench_matmul(1024, 1024, 1024, warmup, iters);
    bench_matmul(4096, 4096, 4096, warmup, 10);

    bench_primitives();

    printf("\nDone.\n");
    energy_shutdown();
    return 0;
}
