/*
 * bench_rac_vs_blas.c — RAC vs OpenBLAS head-to-head benchmark
 * Pinnacle Quantum Group — March 2026
 *
 * Build:
 *   cc -O3 -mavx2 -mfma -fopenmp -I../c \
 *     bench_rac_vs_blas.c ../c/rac_cpu.c ../c/rac_avx2.c ../c/rac_hal.c \
 *     -lopenblas -lm -o bench_vs_blas
 */

#include "rac_cpu.h"
#include "rac_avx2.h"
#include "rac_hal.h"
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void rand_fill(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

int main(void) {
    printf("RAC vs OpenBLAS — Head-to-Head Benchmark\n");
    printf("Pinnacle Quantum Group — March 2026\n");
    printf("══════════════════════════════════════════════════════\n\n");

    rac_hal_init();
    rac_hal_print_profile();

    printf("\nOpenBLAS config: %s\n", openblas_get_config());
    printf("OpenBLAS threads: %d\n\n", openblas_get_num_threads());

    srand(42);
    int iters = 10;

    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("  %-6s │ %10s %8s │ %10s %8s │ %8s │ Winner\n",
           "Size", "OpenBLAS", "GFLOPS", "RAC HAL", "GFLOPS", "Ratio");
    printf("  %-6s─┼─%10s─%8s─┼─%10s─%8s─┼─%8s─┼─%s\n",
           "──────", "──────────", "────────", "──────────", "────────", "────────", "──────");

    for (int si = 0; si < n_sizes; si++) {
        int N = sizes[si];
        float *A = malloc(N*N*sizeof(float));
        float *B = malloc(N*N*sizeof(float));
        float *C_blas = calloc(N*N, sizeof(float));
        float *C_rac  = calloc(N*N, sizeof(float));

        rand_fill(A, N*N);
        rand_fill(B, N*N);

        /* Warmup both */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0f, A, N, B, N, 0.0f, C_blas, N);
        rac_hal_matmul(A, B, C_rac, N, N, N);

        /* Benchmark OpenBLAS */
        double t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            memset(C_blas, 0, N*N*sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N, N, N, 1.0f, A, N, B, N, 0.0f, C_blas, N);
        }
        double t_blas = (now_ms() - t0) / iters;

        /* Benchmark RAC HAL */
        t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            memset(C_rac, 0, N*N*sizeof(float));
            rac_hal_matmul(A, B, C_rac, N, N, N);
        }
        double t_rac = (now_ms() - t0) / iters;

        double ops = 2.0 * N * N * N;
        double gf_blas = ops / (t_blas * 1e6);
        double gf_rac  = ops / (t_rac * 1e6);
        double ratio = gf_rac / gf_blas;
        const char *winner = (ratio >= 1.0) ? "RAC" : "BLAS";

        /* Correctness */
        float max_err = 0;
        for (int i = 0; i < N*N; i++) {
            float d = fabsf(C_rac[i] - C_blas[i]);
            if (d > max_err) max_err = d;
        }

        printf("  %4dx%-1d │ %8.2fms %6.1f   │ %8.2fms %6.1f   │ %6.2fx  │ %s",
               N, N, t_blas, gf_blas, t_rac, gf_rac, ratio, winner);
        if (max_err > 0.1f) printf("  ⚠ err=%.2e", max_err);
        printf("\n");

        free(A); free(B); free(C_blas); free(C_rac);
    }

    /* Fused linear comparison: RAC fused vs OpenBLAS SGEMM + manual bias + GELU */
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  Fused Linear: RAC (matmul+bias+GELU) vs OpenBLAS (3 steps)\n");
    printf("══════════════════════════════════════════════════════\n\n");

    int fb = 128, fk = 768, fn = 3072;
    float *finp  = malloc(fb*fk*sizeof(float));
    float *fwt   = malloc(fn*fk*sizeof(float));
    float *fbias = malloc(fn*sizeof(float));
    float *fout  = malloc(fb*fn*sizeof(float));
    float *fout2 = malloc(fb*fn*sizeof(float));
    rand_fill(finp, fb*fk);
    rand_fill(fwt, fn*fk);
    rand_fill(fbias, fn);

    /* OpenBLAS: SGEMM + bias + GELU (3 separate steps) */
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        /* Step 1: matmul (input @ weight^T) — weight is [fn, fk], need transpose */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    fb, fn, fk, 1.0f, finp, fk, fwt, fk, 0.0f, fout2, fn);
        /* Step 2: add bias */
        for (int r = 0; r < fb; r++)
            for (int c = 0; c < fn; c++)
                fout2[r*fn+c] += fbias[c];
        /* Step 3: GELU */
        for (int j = 0; j < fb*fn; j++)
            fout2[j] = fout2[j] * 0.5f * (1.0f + erff(fout2[j] * 0.7071067811865f));
    }
    double t_blas_fused = (now_ms() - t0) / iters;

    /* RAC: single fused call */
    t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        rac_hal_fused_linear(finp, fwt, fbias, fout, fb, fn, fk, RAC_ACT_GELU);
    }
    double t_rac_fused = (now_ms() - t0) / iters;

    double fused_ops = 2.0 * fb * fn * fk;
    printf("  Transformer FFN: %dx%d→%d + GELU\n\n", fb, fk, fn);
    printf("  OpenBLAS (3-step):  %8.2fms  %6.1f GFLOPS\n",
           t_blas_fused, fused_ops/(t_blas_fused*1e6));
    printf("  RAC fused:          %8.2fms  %6.1f GFLOPS\n",
           t_rac_fused, fused_ops/(t_rac_fused*1e6));
    printf("  Ratio:              %6.2fx  %s\n",
           t_blas_fused / t_rac_fused,
           (t_rac_fused < t_blas_fused) ? "← RAC wins" : "← BLAS wins");

    free(finp); free(fwt); free(fbias); free(fout); free(fout2);

    /* ── 8×4 kernel test: direct call to prove the trick works ── */
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  8×4 vmovsldup/vmovshdup kernel test\n");
    printf("══════════════════════════════════════════════════════\n\n");

    {
        extern void rac_micro_kernel_8x4_asm(const float*, const float*, float*, int, int);

        /* Pack A: [kc=4][MR=8] = 32 floats */
        float pa[32], pb[16], c_out[32];
        /* A = identity-like: a[k*8+r] = (r==k) ? 1 : 0 for simple test */
        memset(pa, 0, sizeof(pa));
        for (int k = 0; k < 4; k++) pa[k * 8 + k] = 1.0f;
        /* B = [1,2,3,4] repeated for 4 K steps */
        for (int k = 0; k < 4; k++) { pb[k*4+0]=1; pb[k*4+1]=2; pb[k*4+2]=3; pb[k*4+3]=4; }
        memset(c_out, 0, sizeof(c_out));

        rac_micro_kernel_8x4_asm(pa, pb, c_out, 4, 4);

        printf("  8x4 kernel output (first 4 rows, 4 cols):\n");
        for (int r = 0; r < 4; r++) {
            printf("    row %d: %.1f %.1f %.1f %.1f\n",
                   r, c_out[r*4+0], c_out[r*4+1], c_out[r*4+2], c_out[r*4+3]);
        }
        /* Row 0 should be [1,2,3,4], row 1 = [1,2,3,4], etc since A=I */
        int ok = (fabsf(c_out[0] - 1.0f) < 0.01f && fabsf(c_out[1] - 2.0f) < 0.01f);
        printf("  8x4 correctness: %s\n", ok ? "PASS" : "FAIL");
    }

    printf("\n══════════════════════════════════════════════════════\n");
    rac_hal_shutdown();
    return 0;
}
