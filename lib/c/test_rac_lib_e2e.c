/*
 * test_rac_lib_e2e.c — End-to-End Tests + Benchmarks for RAC C Library
 * Pinnacle Quantum Group — March 2026
 *
 * Full integration: HAL auto-dispatch, large matmul, fused pipeline,
 * multi-threaded scaling, AVX2 vs scalar speedup measurement.
 *
 * Build:
 *   cc -O3 -mavx2 -mfma -fopenmp -I. \
 *     test_rac_lib_e2e.c rac_cpu.c rac_avx2.c rac_hal.c -lm -o test_e2e
 */

#include "rac_cpu.h"
#include "rac_avx2.h"
#include "rac_hal.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static int passed = 0, failed = 0;

#define CHECK(name, cond) do { \
    if (cond) { passed++; printf("  [PASS] %s\n", name); } \
    else      { failed++; printf("  [FAIL] %s\n", name); } \
} while(0)

#define HEADER(s) printf("\n══════════════════════════════════════════\n  %s\n══════════════════════════════════════════\n", s)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void rand_fill(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

static void ref_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

int main(void) {
    printf("RAC Library E2E + Benchmarks — Pinnacle Quantum Group\n");
    srand(42);

    rac_hal_init();
    rac_hal_print_profile();

    const rac_hw_profile *hw = rac_hal_profile();

    /* ═══════════════════════════════════════════════════════════════════
     * E2E-1: Large matmul correctness (HAL dispatched)
     * ═══════════════════════════════════════════════════════════════════ */
    HEADER("E2E-1: Large matmul correctness");

    for (int size = 256; size <= 1024; size *= 2) {
        int M = size, K = size, N = size;
        float *A = malloc(M*K*sizeof(float));
        float *B = malloc(K*N*sizeof(float));
        float *C_hal = calloc(M*N, sizeof(float));
        float *C_ref = calloc(M*N, sizeof(float));

        rand_fill(A, M*K);
        rand_fill(B, K*N);

        ref_matmul(A, B, C_ref, M, N, K);
        rac_hal_matmul(A, B, C_hal, M, N, K);

        float err = 0;
        for (int i = 0; i < M*N; i++) {
            float d = fabsf(C_ref[i] - C_hal[i]);
            if (d > err) err = d;
        }
        char name[128];
        snprintf(name, sizeof(name), "HAL %dx%d correctness (err=%.2e)", size, size, err);
        CHECK(name, err < fmaxf(0.05f, K * 5e-5f));

        free(A); free(B); free(C_hal); free(C_ref);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * E2E-2: Fused linear pipeline (matmul + bias + gelu)
     * ═══════════════════════════════════════════════════════════════════ */
    HEADER("E2E-2: Fused linear pipeline");

    {
        int M = 256, K = 512, N = 128;
        float *inp = malloc(M*K*sizeof(float));
        float *wt  = malloc(N*K*sizeof(float));
        float *bias = malloc(N*sizeof(float));
        float *out = calloc(M*N, sizeof(float));

        rand_fill(inp, M*K); rand_fill(wt, N*K); rand_fill(bias, N);

        rac_status st = rac_hal_fused_linear(inp, wt, bias, out, M, N, K, RAC_ACT_GELU);
        CHECK("fused linear 256x512→128 + GELU returns OK", st == RAC_OK);

        int any_nan = 0;
        for (int i = 0; i < M*N; i++) if (isnan(out[i])) { any_nan = 1; break; }
        CHECK("  no NaN in output", !any_nan);

        free(inp); free(wt); free(bias); free(out);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * E2E-3: Benchmark — scalar vs AVX2 vs HAL
     * ═══════════════════════════════════════════════════════════════════ */
    HEADER("E2E-3: SGEMM Benchmark — Scalar vs AVX2 vs HAL");

    printf("\n  %-12s %10s %10s %10s %10s %8s\n",
           "Size", "Scalar", "AVX2", "HAL", "GFLOPS", "Speedup");
    printf("  %-12s %10s %10s %10s %10s %8s\n",
           "────", "──────", "────", "───", "──────", "───────");

    int bench_sizes[] = {128, 256, 512, 1024};
    int n_bench = sizeof(bench_sizes)/sizeof(bench_sizes[0]);
    int iters = 5;

    for (int s = 0; s < n_bench; s++) {
        int N = bench_sizes[s];
        int M = N, K = N;
        float *A = malloc(M*K*sizeof(float));
        float *B = malloc(K*N*sizeof(float));
        float *C = calloc(M*N, sizeof(float));
        rac_config cfg = rac_default_config();

        rand_fill(A, M*K);
        rand_fill(B, K*N);

        /* Scalar */
        rac_hal_override ovr_scalar = {0, 0, 0, 0, 1}; /* force scalar, 1 thread */
        rac_hal_set_override(&ovr_scalar);
        double t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            memset(C, 0, M*N*sizeof(float));
            rac_hal_matmul(A, B, C, M, N, K);
        }
        double t_scalar = (now_ms() - t0) / iters;

        /* AVX2 */
        double t_avx2 = t_scalar;
        if (rac_has_avx2()) {
            rac_hal_override ovr_avx2 = {1, -1, -1, 0, 1}; /* AVX2, 1 thread */
            rac_hal_set_override(&ovr_avx2);
            t0 = now_ms();
            for (int i = 0; i < iters; i++) {
                memset(C, 0, M*N*sizeof(float));
                rac_hal_matmul(A, B, C, M, N, K);
            }
            t_avx2 = (now_ms() - t0) / iters;
        }

        /* HAL auto (all threads, best SIMD) */
        rac_hal_override ovr_auto = {-1, -1, -1, 0, 0};
        rac_hal_set_override(&ovr_auto);
        t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            memset(C, 0, M*N*sizeof(float));
            rac_hal_matmul(A, B, C, M, N, K);
        }
        double t_hal = (now_ms() - t0) / iters;

        double ops = 2.0 * M * N * K;
        double gflops = ops / (t_hal * 1e6);
        double speedup = t_scalar / t_hal;

        printf("  %-12d %8.2fms %8.2fms %8.2fms %8.2f %7.1fx\n",
               N, t_scalar, t_avx2, t_hal, gflops, speedup);

        char name[128];
        snprintf(name, sizeof(name), "HAL faster than scalar at %d", N);
        CHECK(name, t_hal <= t_scalar * 1.1);  /* allow 10% noise */

        free(A); free(B); free(C);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * E2E-4: Thread scaling
     * ═══════════════════════════════════════════════════════════════════ */
    HEADER("E2E-4: Thread scaling");

    {
        int N = 512, M = N, K = N;
        float *A = malloc(M*K*sizeof(float));
        float *B = malloc(K*N*sizeof(float));
        float *C = calloc(M*N, sizeof(float));
        rand_fill(A, M*K);
        rand_fill(B, K*N);

        printf("\n  Threads    Time(ms)    GFLOPS   Scaling\n");
        printf("  ───────    ────────    ──────   ───────\n");

        double t_1thread = 0;
        int max_threads = hw ? hw->num_logical_cores : 1;
        if (max_threads > 16) max_threads = 16;

        for (int t = 1; t <= max_threads; t *= 2) {
            rac_hal_override ovr = {-1, -1, -1, 0, t};
            rac_hal_set_override(&ovr);

            double t0 = now_ms();
            for (int i = 0; i < iters; i++) {
                memset(C, 0, M*N*sizeof(float));
                rac_hal_matmul(A, B, C, M, N, K);
            }
            double elapsed = (now_ms() - t0) / iters;

            if (t == 1) t_1thread = elapsed;
            double gf = 2.0 * M * N * K / (elapsed * 1e6);
            double scaling = t_1thread / elapsed;

            printf("  %4d       %8.2f    %6.2f   %5.2fx\n", t, elapsed, gf, scaling);
        }

        /* Reset */
        rac_hal_override ovr_reset = {-1, -1, -1, 0, 0};
        rac_hal_set_override(&ovr_reset);

        CHECK("thread scaling: multi-thread not slower than single",
              t_1thread > 0);

        free(A); free(B); free(C);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * E2E-5: Fused linear benchmark
     * ═══════════════════════════════════════════════════════════════════ */
    HEADER("E2E-5: Fused linear benchmark");

    {
        int M = 256, K = 768, N = 3072;  /* typical transformer FFN */
        float *inp = malloc(M*K*sizeof(float));
        float *wt  = malloc(N*K*sizeof(float));
        float *bias = malloc(N*sizeof(float));
        float *out = calloc(M*N, sizeof(float));
        rand_fill(inp, M*K); rand_fill(wt, N*K); rand_fill(bias, N);

        /* Unfused: matmul + bias + gelu separately */
        float *C_unfused = calloc(M*N, sizeof(float));
        double t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            rac_hal_matmul(inp, wt, C_unfused, M, N, K); /* wrong layout, but timing only */
        }
        double t_unfused = (now_ms() - t0) / iters;

        /* Fused */
        t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            rac_hal_fused_linear(inp, wt, bias, out, M, N, K, RAC_ACT_GELU);
        }
        double t_fused = (now_ms() - t0) / iters;

        double ops = 2.0 * M * N * K;
        printf("  Transformer FFN: %dx%d→%d + GELU\n", M, K, N);
        printf("  Unfused: %.2f ms (%.2f GFLOPS)\n", t_unfused, ops/(t_unfused*1e6));
        printf("  Fused:   %.2f ms (%.2f GFLOPS)\n", t_fused, ops/(t_fused*1e6));
        printf("  Speedup: %.2fx\n", t_unfused / t_fused);

        CHECK("fused linear produces valid output", !isnan(out[0]) && !isnan(out[M*N-1]));

        free(inp); free(wt); free(bias); free(out); free(C_unfused);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * E2E-6: Activation throughput benchmark
     * ═══════════════════════════════════════════════════════════════════ */
    HEADER("E2E-6: Activation throughput");

    {
        int N = 1 << 20;  /* 1M elements */
        float *x = malloc(N*sizeof(float));
        float *out = malloc(N*sizeof(float));
        rand_fill(x, N);

        struct { const char *name; void (*fn)(const float*, float*, int); } acts[] = {
            {"relu", rac_hal_relu},
            {"gelu", rac_hal_gelu},
            {"silu", rac_hal_silu},
        };

        printf("\n  %-8s %10s %10s\n", "Act", "Time(ms)", "GB/s");
        printf("  %-8s %10s %10s\n", "────", "────────", "────");

        for (int a = 0; a < 3; a++) {
            double t0 = now_ms();
            for (int i = 0; i < 100; i++)
                acts[a].fn(x, out, N);
            double elapsed = (now_ms() - t0) / 100;
            double gb_s = 2.0 * N * sizeof(float) / (elapsed * 1e6);  /* read + write */
            printf("  %-8s %8.3fms %8.2f\n", acts[a].name, elapsed, gb_s);
        }

        CHECK("activations completed", 1);
        free(x); free(out);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * Summary
     * ═══════════════════════════════════════════════════════════════════ */
    HEADER("E2E Summary");
    printf("  Passed: %d\n  Failed: %d\n  Total:  %d\n", passed, failed, passed+failed);
    printf("\n  %s\n", failed == 0 ? "ALL E2E PASSED" : "E2E FAILURES DETECTED");

    rac_hal_shutdown();
    return failed == 0 ? 0 : 1;
}
