/*
 * rac_benchmark.cu — RAC vs cuBLAS benchmark harness
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Reports: ops/sec, wall time, energy via NVML, correctness vs cuBLAS SGEMM
 */

#include "rac.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, st); \
        exit(1); \
    } \
} while(0)

/* ── Timing ──────────────────────────────────────────────────────────────── */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── NVML energy sampling ────────────────────────────────────────────────── */

static nvmlDevice_t nvml_device;
static int nvml_ok = 0;

static void nvml_init(void) {
    if (nvmlInit() == NVML_SUCCESS &&
        nvmlDeviceGetHandleByIndex(0, &nvml_device) == NVML_SUCCESS) {
        nvml_ok = 1;
        printf("NVML: energy measurement available\n");
    } else {
        printf("NVML: not available (energy measurement disabled)\n");
    }
}

static unsigned long long nvml_energy_mj(void) {
    if (!nvml_ok) return 0;
    unsigned long long energy = 0;
    nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy);
    return energy;
}

/* ── Correctness check ───────────────────────────────────────────────────── */

static int check_correctness(float *rac_out, float *cublas_out, int M, int N,
                              float tol) {
    int failures = 0;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(rac_out[i] - cublas_out[i]);
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

    size_t szA = M * K * sizeof(float);
    size_t szB = K * N * sizeof(float);
    size_t szC = M * N * sizeof(float);

    /* host data */
    float *hA = (float*)malloc(szA);
    float *hB = (float*)malloc(szB);
    float *hC_rac    = (float*)malloc(szC);
    float *hC_cublas = (float*)malloc(szC);
    if (!hA || !hB || !hC_rac || !hC_cublas) {
        fprintf(stderr, "Host malloc failed for %dx%d matmul\n", M, N);
        free(hA); free(hB); free(hC_rac); free(hC_cublas);
        return;
    }

    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    /* device data */
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, szA));
    CHECK_CUDA(cudaMalloc(&dB, szB));
    CHECK_CUDA(cudaMalloc(&dC, szC));
    CHECK_CUDA(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    /* ── RAC benchmark ── */
    for (int i = 0; i < warmup; i++)
        rac_matmul_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    unsigned long long e0 = nvml_energy_mj();
    double t0 = now_ms();

    for (int i = 0; i < iters; i++)
        rac_matmul_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    double t_rac = now_ms() - t0;
    unsigned long long e_rac = nvml_energy_mj() - e0;

    CHECK_CUDA(cudaMemcpy(hC_rac, dC, szC, cudaMemcpyDeviceToHost));

    double ops = 2.0 * M * N * K;  /* flops per matmul */
    double rac_tops = (ops * iters) / (t_rac * 1e9);

    printf("  RAC:    %.2f ms/iter  %.4f TOPS  energy=%llu mJ\n",
           t_rac / iters, rac_tops, e_rac / iters);

    /* ── cuBLAS benchmark ── */
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;

    /* cuBLAS is column-major; transpose trick for row-major inputs */
    for (int i = 0; i < warmup; i++)
        CHECK_CUBLAS(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            dB, N, dA, K, &beta, dC, N));
    CHECK_CUDA(cudaDeviceSynchronize());

    unsigned long long e1 = nvml_energy_mj();
    double t1 = now_ms();

    for (int i = 0; i < iters; i++)
        CHECK_CUBLAS(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            dB, N, dA, K, &beta, dC, N));
    CHECK_CUDA(cudaDeviceSynchronize());

    double t_cublas = now_ms() - t1;
    unsigned long long e_cublas = nvml_energy_mj() - e1;
    double cublas_tops = (ops * iters) / (t_cublas * 1e9);

    CHECK_CUDA(cudaMemcpy(hC_cublas, dC, szC, cudaMemcpyDeviceToHost));

    printf("  cuBLAS: %.2f ms/iter  %.4f TOPS  energy=%llu mJ\n",
           t_cublas / iters, cublas_tops, e_cublas / iters);
    printf("  Speedup: RAC/cuBLAS = %.3fx\n", t_cublas / t_rac);

    if (e_rac > 0 && e_cublas > 0)
        printf("  Energy:  RAC/cuBLAS = %.3fx\n",
               (double)e_rac / (double)e_cublas);

    /* correctness — RAC vs cuBLAS (row-major trick: B^T * A^T in col-major = A*B row-major) */
    check_correctness(hC_rac, hC_cublas, M, N, 1e-3f);

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC_rac); free(hC_cublas);
}

/* ── Benchmark: primitives ───────────────────────────────────────────────── */

static void bench_primitives(void) {
    printf("\n── Primitive microbenchmarks ────────────────────────────\n");
    const int N = 1 << 20;  /* 1M elements */
    const int iters = 100;

    float2 *dV, *dOut;
    float  *dTheta;
    CHECK_CUDA(cudaMalloc(&dV,     N * sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&dOut,   N * sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&dTheta, N * sizeof(float)));

    /* fill with test data on device via kernel */
    /* (omitted for brevity — would use cuRAND or a fill kernel) */

    /* rotate_batch */
    dim3 block(256);
    dim3 grid((N + 255) / 256);

    extern __global__ void rac_rotate_batch_kernel(float2*, float*, float2*, int);

    CHECK_CUDA(cudaDeviceSynchronize());
    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
        rac_rotate_batch_kernel<<<grid, block>>>(dV, dTheta, dOut, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    double t_rot = (now_ms() - t0) / iters;
    printf("  rac_rotate_batch: %.3f ms/iter  %.2f Gop/s\n",
           t_rot, N / (t_rot * 1e6));

    cudaFree(dV); cudaFree(dOut); cudaFree(dTheta);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    printf("RAC Benchmark — Pinnacle Quantum Group — March 2026\n");
    printf("=====================================================\n");

    /* device info */
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (SM %d.%d)  SMs: %d\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    nvml_init();

    int warmup = 5, iters = 50;

    /* sweep matmul sizes */
    bench_matmul(256,  256,  256,  warmup, iters);
    bench_matmul(512,  512,  512,  warmup, iters);
    bench_matmul(1024, 1024, 1024, warmup, iters);
    bench_matmul(4096, 4096, 4096, warmup, 10);

    bench_primitives();

    printf("\nDone.\n");
    if (nvml_ok) nvmlShutdown();
    return 0;
}
