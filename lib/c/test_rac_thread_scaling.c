/*
 * test_rac_thread_scaling.c — RAC Thread Scaling Benchmark
 * Pinnacle Quantum Group — March 2026
 *
 * Sweeps 1 → max_logical_cores, measuring SGEMM + fused FFN throughput
 * at each thread count. Shows where physical cores saturate and whether
 * SMT (hyperthreading) helps or hurts.
 *
 * Build:
 *   cc -O3 -mavx2 -mfma -fopenmp -I. \
 *     test_rac_thread_scaling.c rac_cpu.c rac_avx2.c rac_hal.c -lm -o test_threads
 */

#include "rac_cpu.h"
#include "rac_avx2.h"
#include "rac_hal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void rand_fill(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

typedef struct {
    int threads;
    double sgemm_256_ms;
    double sgemm_512_ms;
    double sgemm_1024_ms;
    double ffn_ms;
    double sgemm_256_gf;
    double sgemm_512_gf;
    double sgemm_1024_gf;
    double ffn_gf;
} thread_result;

int main(int argc, char **argv) {
    printf("RAC Thread Scaling Benchmark — Pinnacle Quantum Group\n");
    printf("══════════════════════════════════════════════════════\n\n");

    rac_hal_init();
    const rac_hw_profile *hw = rac_hal_profile();

    int max_threads = hw->num_logical_cores;
    if (argc > 1) max_threads = atoi(argv[1]);
    if (max_threads < 1) max_threads = 1;

    int phys = hw->num_physical_cores;

    printf("CPU:      %s\n", hw->cpu_name);
    printf("Physical: %d cores\n", phys);
    printf("Logical:  %d threads (%.1fx SMT)\n", hw->num_logical_cores,
           (float)hw->num_logical_cores / phys);
    printf("SIMD:     %s, %d-byte width\n",
           hw->use_avx2 ? "AVX2+FMA" : (hw->use_avx512 ? "AVX-512" : "scalar"),
           hw->preferred_simd_width);
    printf("Tile:     %d (L1), %d (L2)\n", hw->optimal_tile_sgemm, hw->optimal_tile_sgemm_l2);

    srand(42);
    int iters = 10;

    /* Pre-allocate for largest size */
    int N = 1024;
    float *A1024 = malloc(N*N*sizeof(float));
    float *B1024 = malloc(N*N*sizeof(float));
    float *C1024 = calloc(N*N, sizeof(float));
    rand_fill(A1024, N*N);
    rand_fill(B1024, N*N);

    N = 512;
    float *A512 = malloc(N*N*sizeof(float));
    float *B512 = malloc(N*N*sizeof(float));
    float *C512 = calloc(N*N, sizeof(float));
    rand_fill(A512, N*N);
    rand_fill(B512, N*N);

    N = 256;
    float *A256 = malloc(N*N*sizeof(float));
    float *B256 = malloc(N*N*sizeof(float));
    float *C256 = calloc(N*N, sizeof(float));
    rand_fill(A256, N*N);
    rand_fill(B256, N*N);

    /* FFN: 128x768 → 3072 */
    int fb = 128, fk = 768, fn = 3072;
    float *finp = malloc(fb*fk*sizeof(float));
    float *fwt  = malloc(fn*fk*sizeof(float));
    float *fbias= malloc(fn*sizeof(float));
    float *fout = malloc(fb*fn*sizeof(float));
    rand_fill(finp, fb*fk);
    rand_fill(fwt, fn*fk);
    rand_fill(fbias, fn);

    /* Build thread count list: 1, 2, 4, ..., phys, phys+2, ..., max */
    int thread_counts[64];
    int n_counts = 0;

    /* Powers of 2 up to physical */
    for (int t = 1; t <= phys && n_counts < 64; t *= 2)
        thread_counts[n_counts++] = t;
    /* Physical core count if not already included */
    if (n_counts == 0 || thread_counts[n_counts-1] != phys)
        thread_counts[n_counts++] = phys;
    /* SMT steps: phys+25%, phys+50%, phys+75%, 2x phys (logical) */
    if (hw->num_logical_cores > phys) {
        int smt_steps[] = {
            phys + phys/4,
            phys + phys/2,
            phys + phys*3/4,
            hw->num_logical_cores
        };
        for (int s = 0; s < 4 && n_counts < 64; s++) {
            if (smt_steps[s] > thread_counts[n_counts-1] &&
                smt_steps[s] <= hw->num_logical_cores)
                thread_counts[n_counts++] = smt_steps[s];
        }
    }

    /* Results */
    thread_result results[64];

    /* ── Header ── */
    printf("\n  Threads │  SGEMM 256   SGEMM 512   SGEMM 1024  │  Fused FFN   │ Note\n");
    printf("  ────────┼─────────────────────────────────────────┼──────────────┼──────\n");

    double baseline_1024 = 0;

    for (int ti = 0; ti < n_counts; ti++) {
        int t = thread_counts[ti];
        thread_result *r = &results[ti];
        r->threads = t;

        /* Set thread count via HAL override */
        rac_hal_override ovr = {-1, -1, -1, 0, t};
        rac_hal_set_override(&ovr);

        /* Warm up */
        memset(C256, 0, 256*256*sizeof(float));
        rac_hal_matmul(A256, B256, C256, 256, 256, 256);

        /* SGEMM 256 */
        double t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            memset(C256, 0, 256*256*sizeof(float));
            rac_hal_matmul(A256, B256, C256, 256, 256, 256);
        }
        r->sgemm_256_ms = (now_ms() - t0) / iters;
        r->sgemm_256_gf = (2.0*256*256*256) / (r->sgemm_256_ms * 1e6);

        /* SGEMM 512 */
        t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            memset(C512, 0, 512*512*sizeof(float));
            rac_hal_matmul(A512, B512, C512, 512, 512, 512);
        }
        r->sgemm_512_ms = (now_ms() - t0) / iters;
        r->sgemm_512_gf = (2.0*512*512*512) / (r->sgemm_512_ms * 1e6);

        /* SGEMM 1024 */
        t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            memset(C1024, 0, 1024*1024*sizeof(float));
            rac_hal_matmul(A1024, B1024, C1024, 1024, 1024, 1024);
        }
        r->sgemm_1024_ms = (now_ms() - t0) / iters;
        r->sgemm_1024_gf = (2.0*1024*1024*1024) / (r->sgemm_1024_ms * 1e6);

        if (ti == 0) baseline_1024 = r->sgemm_1024_ms;

        /* Fused FFN: 128x768→3072+GELU */
        t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            rac_hal_fused_linear(finp, fwt, fbias, fout, fb, fn, fk, RAC_ACT_GELU);
        }
        r->ffn_ms = (now_ms() - t0) / iters;
        r->ffn_gf = (2.0*fb*fn*fk) / (r->ffn_ms * 1e6);

        /* Determine note */
        const char *note = "";
        if (t == 1) note = "baseline";
        else if (t == phys) note = "<── physical";
        else if (t == hw->num_logical_cores) note = "<── logical (SMT)";
        else if (t > phys) note = "SMT";

        printf("  %4d    │ %6.1f GF   %6.1f GF   %6.1f GF    │ %6.1f GF    │ %s\n",
               t, r->sgemm_256_gf, r->sgemm_512_gf, r->sgemm_1024_gf,
               r->ffn_gf, note);
    }

    /* ── Scaling analysis ── */
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  SCALING ANALYSIS (SGEMM 1024x1024)\n");
    printf("══════════════════════════════════════════════════════\n\n");

    printf("  Threads │  Time(ms)  │  GFLOPS  │  Scaling  │  Efficiency\n");
    printf("  ────────┼────────────┼──────────┼───────────┼────────────\n");

    double t1_gf = results[0].sgemm_1024_gf;
    for (int ti = 0; ti < n_counts; ti++) {
        thread_result *r = &results[ti];
        double scaling = r->sgemm_1024_gf / t1_gf;
        double efficiency = scaling / r->threads * 100;
        printf("  %4d    │ %8.2f   │ %6.1f   │  %5.2fx   │  %5.1f%%\n",
               r->threads, r->sgemm_1024_ms, r->sgemm_1024_gf,
               scaling, efficiency);
    }

    /* ── SMT verdict ── */
    printf("\n");
    if (hw->num_logical_cores > phys && n_counts >= 2) {
        /* Find physical and logical results */
        double gf_phys = 0, gf_logical = 0;
        for (int ti = 0; ti < n_counts; ti++) {
            if (results[ti].threads == phys) gf_phys = results[ti].sgemm_1024_gf;
            if (results[ti].threads == hw->num_logical_cores)
                gf_logical = results[ti].sgemm_1024_gf;
        }
        if (gf_phys > 0 && gf_logical > 0) {
            double smt_ratio = gf_logical / gf_phys;
            printf("  SMT verdict (1024x1024): %d threads → %.1f GFLOPS vs %d threads → %.1f GFLOPS\n",
                   phys, gf_phys, hw->num_logical_cores, gf_logical);
            if (smt_ratio > 1.05)
                printf("  ✓ SMT HELPS: +%.1f%% throughput. Consider using all %d threads.\n",
                       (smt_ratio - 1) * 100, hw->num_logical_cores);
            else if (smt_ratio < 0.95)
                printf("  ✗ SMT HURTS: -%.1f%% throughput. Stick to %d physical cores.\n",
                       (1 - smt_ratio) * 100, phys);
            else
                printf("  ≈ SMT NEUTRAL: <5%% difference. Physical cores (%d) are optimal.\n",
                       phys);
        }
    }

    printf("\n══════════════════════════════════════════════════════\n");

    /* Cleanup */
    free(A256); free(B256); free(C256);
    free(A512); free(B512); free(C512);
    free(A1024); free(B1024); free(C1024);
    free(finp); free(fwt); free(fbias); free(fout);

    /* Reset HAL */
    rac_hal_override reset = {-1, -1, -1, 0, 0};
    rac_hal_set_override(&reset);
    rac_hal_shutdown();

    return 0;
}
