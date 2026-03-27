/*
 * test_rac_longrun.c — RAC Long-Run Stress Test
 * Pinnacle Quantum Group — March 2026
 *
 * Exercises the full stack in concert: HAL + AVX2 + OpenMP + SGEMM +
 * fused linear + activations + CORDIC primitives, all running together
 * under sustained load with correctness checks throughout.
 *
 * Simulates a realistic inference/training pipeline:
 *   1. HAL auto-configures for the hardware
 *   2. Fused FFN blocks (matmul+bias+gelu, matmul+bias) run in a loop
 *   3. Attention-like QK^T + softmax + @V pattern
 *   4. Gradient-like backward matmul (TN and NT patterns)
 *   5. CORDIC primitives running alongside (rotate, polar, project)
 *   6. Activation throughput under sustained load
 *   7. Multi-size sweep (small → large) interleaved
 *   8. Correctness validated every N iterations
 *
 * Default: ~5 minutes. Set RAC_LONGRUN_ITERS=N to control duration.
 *
 * Build:
 *   cc -O3 -mavx2 -mfma -fopenmp -I. \
 *     test_rac_longrun.c rac_cpu.c rac_avx2.c rac_hal.c -lm -o test_longrun
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

/* ── Helpers ────────────────────────────────────────────────────────────── */

static int g_passed = 0, g_failed = 0;
static double g_total_gflops = 0;
static long long g_total_ops = 0;

#define CHECK(name, cond) do { \
    if (cond) { g_passed++; } \
    else { g_failed++; printf("  [FAIL] %s\n", name); } \
} while(0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void rand_fill(float *x, int n, unsigned int *seed) {
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1103515245 + 12345;
        x[i] = ((float)((*seed >> 16) & 0x7FFF) / 16384.0f) - 1.0f;
    }
}

static float max_err(const float *a, const float *b, int n) {
    float e = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > e) e = d;
    }
    return e;
}

static void ref_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test phases
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Phase 1: Sustained SGEMM at multiple sizes
 * Simulates batch inference with varying sequence lengths
 */
static void phase_sgemm_sweep(int iters, unsigned int *seed) {
    printf("\n── Phase 1: Sustained SGEMM sweep (%d iterations) ──\n", iters);

    int sizes[][3] = {
        {64, 64, 64},
        {128, 256, 128},
        {256, 256, 256},
        {384, 512, 384},
        {512, 512, 512},
        {768, 768, 768},
        {1024, 1024, 1024},
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    double phase_start = now_ms();
    int checks = 0;

    for (int iter = 0; iter < iters; iter++) {
        int si = iter % n_sizes;
        int M = sizes[si][0], K = sizes[si][1], N = sizes[si][2];

        float *A = malloc(M*K*sizeof(float));
        float *B = malloc(K*N*sizeof(float));
        float *C = calloc(M*N, sizeof(float));

        rand_fill(A, M*K, seed);
        rand_fill(B, K*N, seed);

        rac_hal_matmul(A, B, C, M, N, K);

        double ops = 2.0 * M * N * K;
        g_total_ops += (long long)ops;

        /* Correctness check every 10 iterations */
        if (iter % 10 == 0) {
            float *C_ref = calloc(M*N, sizeof(float));
            ref_matmul(A, B, C_ref, M, N, K);
            float err = max_err(C, C_ref, M*N);
            float tol = fmaxf(0.05f, K * 5e-5f);
            CHECK("sgemm correctness", err < tol);
            checks++;
            free(C_ref);
        }

        /* NaN/Inf check every iteration */
        int bad = 0;
        for (int i = 0; i < M*N; i++) {
            if (isnan(C[i]) || isinf(C[i])) { bad = 1; break; }
        }
        CHECK("no NaN/Inf", !bad);

        free(A); free(B); free(C);
    }

    double elapsed = now_ms() - phase_start;
    printf("  %d iterations, %d correctness checks, %.1fs\n",
           iters, checks, elapsed / 1000);
}

/*
 * Phase 2: Fused linear pipeline (simulates transformer FFN)
 * input → linear+gelu → linear → output (repeated)
 */
static void phase_fused_ffn(int iters, unsigned int *seed) {
    printf("\n── Phase 2: Fused FFN pipeline (%d iterations) ──\n", iters);

    int batch = 128, d_model = 768, ff_dim = 3072;

    float *input  = malloc(batch * d_model * sizeof(float));
    float *W1     = malloc(ff_dim * d_model * sizeof(float));
    float *b1     = malloc(ff_dim * sizeof(float));
    float *hidden = malloc(batch * ff_dim * sizeof(float));
    float *W2     = malloc(d_model * ff_dim * sizeof(float));
    float *b2     = malloc(d_model * sizeof(float));
    float *output = malloc(batch * d_model * sizeof(float));

    rand_fill(W1, ff_dim * d_model, seed);
    rand_fill(b1, ff_dim, seed);
    rand_fill(W2, d_model * ff_dim, seed);
    rand_fill(b2, d_model, seed);

    double phase_start = now_ms();

    for (int iter = 0; iter < iters; iter++) {
        rand_fill(input, batch * d_model, seed);

        /* Layer 1: linear + GELU */
        rac_hal_fused_linear(input, W1, b1, hidden,
                              batch, ff_dim, d_model, RAC_ACT_GELU);

        /* Layer 2: linear (no activation) */
        rac_hal_fused_linear(hidden, W2, b2, output,
                              batch, d_model, ff_dim, RAC_ACT_NONE);

        double ops = 2.0 * batch * ff_dim * d_model + 2.0 * batch * d_model * ff_dim;
        g_total_ops += (long long)ops;

        /* Check output is finite */
        int bad = 0;
        for (int i = 0; i < batch * d_model; i++) {
            if (isnan(output[i]) || isinf(output[i])) { bad = 1; break; }
        }
        CHECK("FFN output finite", !bad);
    }

    double elapsed = now_ms() - phase_start;
    double total_ops = 2.0 * (2.0 * batch * ff_dim * d_model) * iters;
    double gflops = total_ops / (elapsed * 1e6);
    printf("  %d FFN blocks, %.1fs, %.1f GFLOPS sustained\n",
           iters, elapsed / 1000, gflops);
    g_total_gflops += gflops;

    free(input); free(W1); free(b1); free(hidden);
    free(W2); free(b2); free(output);
}

/*
 * Phase 3: Attention-like pattern (QK^T + softmax + @V)
 * Tests interleaved matmul + softmax under sustained load
 */
static void phase_attention(int iters, unsigned int *seed) {
    printf("\n── Phase 3: Attention pattern (%d iterations) ──\n", iters);

    int batch = 8, heads = 12, seq = 128, d_head = 64;

    /* Per-head buffers */
    float *Q     = malloc(seq * d_head * sizeof(float));
    float *K     = malloc(seq * d_head * sizeof(float));
    float *V     = malloc(seq * d_head * sizeof(float));
    float *scores= malloc(seq * seq * sizeof(float));
    float *attn  = malloc(seq * seq * sizeof(float));
    float *out   = malloc(seq * d_head * sizeof(float));

    double phase_start = now_ms();
    int total_heads = 0;

    for (int iter = 0; iter < iters; iter++) {
        for (int h = 0; h < heads; h++) {
            rand_fill(Q, seq * d_head, seed);
            rand_fill(K, seq * d_head, seed);
            rand_fill(V, seq * d_head, seed);

            /* scores = Q @ K^T  [seq, seq] */
            /* K^T: we transpose by treating K as [d_head, seq] */
            for (int i = 0; i < seq; i++) {
                for (int j = 0; j < seq; j++) {
                    float s = 0;
                    for (int k = 0; k < d_head; k++)
                        s += Q[i*d_head+k] * K[j*d_head+k];
                    scores[i*seq+j] = s / sqrtf((float)d_head);
                }
            }

            /* softmax per row */
            rac_softmax_batch(scores, attn, seq, seq);

            /* out = attn @ V  [seq, d_head] */
            rac_hal_matmul(attn, V, out, seq, d_head, seq);

            total_heads++;
            g_total_ops += (long long)(2.0 * seq * seq * d_head * 2); /* QK^T + attn@V */
        }

        /* Spot check: attn rows sum to 1 */
        if (iter % 5 == 0) {
            float row_sum = 0;
            for (int j = 0; j < seq; j++) row_sum += attn[j]; /* first row */
            CHECK("attn row sums to 1", fabsf(row_sum - 1.0f) < 0.02f);
        }

        /* Finite check */
        int bad = 0;
        for (int i = 0; i < seq * d_head; i++) {
            if (isnan(out[i]) || isinf(out[i])) { bad = 1; break; }
        }
        CHECK("attention output finite", !bad);
    }

    double elapsed = now_ms() - phase_start;
    printf("  %d iterations x %d heads = %d attention ops, %.1fs\n",
           iters, heads, total_heads, elapsed / 1000);

    free(Q); free(K); free(V); free(scores); free(attn); free(out);
}

/*
 * Phase 4: Backward-like matmul patterns (TN and NT)
 * Simulates gradient computation
 */
static void phase_backward(int iters, unsigned int *seed) {
    printf("\n── Phase 4: Backward matmul patterns (%d iterations) ──\n", iters);

    int M = 256, K = 768, N = 256;

    float *grad_out = malloc(M * N * sizeof(float));
    float *input    = malloc(M * K * sizeof(float));
    float *weight   = malloc(N * K * sizeof(float));
    float *grad_inp = calloc(M * K, sizeof(float));
    float *grad_wt  = calloc(N * K, sizeof(float));

    rand_fill(weight, N * K, seed);

    double phase_start = now_ms();

    for (int iter = 0; iter < iters; iter++) {
        rand_fill(grad_out, M * N, seed);
        rand_fill(input, M * K, seed);

        /* grad_input = grad_out @ weight  [M,K] = [M,N] @ [N,K] */
        memset(grad_inp, 0, M * K * sizeof(float));
        rac_hal_matmul(grad_out, weight, grad_inp, M, K, N);

        /* grad_weight = grad_out^T @ input  [N,K] = [N,M] @ [M,K]
         * Simulate by transposing grad_out and using regular matmul */
        float *grad_out_t = malloc(N * M * sizeof(float));
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                grad_out_t[j * M + i] = grad_out[i * N + j];
        memset(grad_wt, 0, N * K * sizeof(float));
        rac_hal_matmul(grad_out_t, input, grad_wt, N, K, M);
        free(grad_out_t);

        g_total_ops += (long long)(2.0 * M * K * N * 2); /* two matmuls */

        int bad = 0;
        for (int i = 0; i < M*K; i++) if (isnan(grad_inp[i])) { bad = 1; break; }
        for (int i = 0; i < N*K; i++) if (isnan(grad_wt[i])) { bad = 1; break; }
        CHECK("backward grads finite", !bad);
    }

    double elapsed = now_ms() - phase_start;
    printf("  %d backward passes, %.1fs\n", iters, elapsed / 1000);

    free(grad_out); free(input); free(weight);
    free(grad_inp); free(grad_wt);
}

/*
 * Phase 5: CORDIC primitives under sustained load
 * Exercises the rotation/vectoring path alongside the matmul path
 */
static void phase_cordic(int iters, unsigned int *seed) {
    printf("\n── Phase 5: CORDIC primitives (%d iterations) ──\n", iters);

    int N = 4096;
    rac_vec2 *vecs = malloc(N * sizeof(rac_vec2));
    float *thetas  = malloc(N * sizeof(float));
    rac_vec2 *out  = malloc(N * sizeof(rac_vec2));

    double phase_start = now_ms();

    for (int iter = 0; iter < iters; iter++) {
        rand_fill((float*)vecs, N * 2, seed);
        rand_fill(thetas, N, seed);

        /* Batch rotate */
        rac_rotate_batch(vecs, thetas, out, N);

        /* Check magnitude preservation */
        if (iter % 10 == 0) {
            int mag_ok = 1;
            for (int i = 0; i < 100; i++) {
                float mag_in  = sqrtf(vecs[i].x * vecs[i].x + vecs[i].y * vecs[i].y);
                float mag_out = sqrtf(out[i].x * out[i].x + out[i].y * out[i].y);
                if (mag_in > 0.01f && fabsf(mag_out / mag_in - 1.0f) > 0.05f) {
                    mag_ok = 0; break;
                }
            }
            CHECK("CORDIC magnitude preserved", mag_ok);
        }

        /* Also exercise polar, project, dot, coherence */
        float mag, angle;
        rac_polar(vecs[0], &mag, &angle);
        float proj = rac_project(vecs[0], thetas[0]);
        float dot  = rac_dot(vecs[0], vecs[1]);
        float coh  = rac_coherence(vecs[0], vecs[1]);
        (void)proj; (void)dot; (void)coh;

        CHECK("CORDIC scalars finite",
              isfinite(mag) && isfinite(angle) && isfinite(proj) &&
              isfinite(dot) && isfinite(coh));

        g_total_ops += (long long)(N * 32); /* ~32 ops per CORDIC rotation */
    }

    double elapsed = now_ms() - phase_start;
    printf("  %d iterations x %d rotations, %.1fs\n", iters, N, elapsed / 1000);

    free(vecs); free(thetas); free(out);
}

/*
 * Phase 6: Activation throughput under sustained load
 */
static void phase_activations(int iters, unsigned int *seed) {
    printf("\n── Phase 6: Activation throughput (%d iterations) ──\n", iters);

    int N = 1 << 20; /* 1M */
    float *x   = malloc(N * sizeof(float));
    float *out = malloc(N * sizeof(float));
    rand_fill(x, N, seed);

    double phase_start = now_ms();

    for (int iter = 0; iter < iters; iter++) {
        switch (iter % 3) {
            case 0: rac_hal_relu(x, out, N); break;
            case 1: rac_hal_gelu(x, out, N); break;
            case 2: rac_hal_silu(x, out, N); break;
        }

        /* Use output as next input (chain) */
        float *tmp = x; x = out; out = tmp;

        g_total_ops += (long long)N;
    }

    double elapsed = now_ms() - phase_start;
    double gb_s = (double)N * sizeof(float) * 2 * iters / (elapsed * 1e6);
    printf("  %d activations, %.1fs, %.1f GB/s average\n",
           iters, elapsed / 1000, gb_s);

    free(x); free(out);
}

/*
 * Phase 7: Mixed-size interleaved (simulates real inference with padding)
 */
static void phase_interleaved(int iters, unsigned int *seed) {
    printf("\n── Phase 7: Mixed-size interleaved (%d iterations) ──\n", iters);

    double phase_start = now_ms();

    for (int iter = 0; iter < iters; iter++) {
        /* Randomly pick a size */
        *seed = *seed * 1103515245 + 12345;
        int si = (*seed >> 16) % 5;
        int sizes[] = {32, 128, 256, 512, 1024};
        int S = sizes[si];

        float *A = malloc(S * S * sizeof(float));
        float *B = malloc(S * S * sizeof(float));
        float *C = calloc(S * S, sizeof(float));
        rand_fill(A, S*S, seed);
        rand_fill(B, S*S, seed);

        rac_hal_matmul(A, B, C, S, S, S);
        g_total_ops += (long long)(2.0 * S * S * S);

        int bad = 0;
        for (int i = 0; i < S*S; i++) {
            if (isnan(C[i]) || isinf(C[i])) { bad = 1; break; }
        }
        CHECK("interleaved output finite", !bad);

        free(A); free(B); free(C);
    }

    double elapsed = now_ms() - phase_start;
    printf("  %d mixed-size matmuls, %.1fs\n", iters, elapsed / 1000);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    printf("RAC Long-Run Stress Test — Pinnacle Quantum Group\n");
    printf("══════════════════════════════════════════════════\n");

    /* Duration control */
    int base_iters = 50;
    const char *env = getenv("RAC_LONGRUN_ITERS");
    if (env) base_iters = atoi(env);
    if (argc > 1) base_iters = atoi(argv[1]);
    if (base_iters < 1) base_iters = 1;

    printf("Base iterations: %d (set RAC_LONGRUN_ITERS or pass as arg)\n", base_iters);

    /* Init HAL */
    rac_hal_init();
    rac_hal_print_profile();

    unsigned int seed = 42;
    double total_start = now_ms();

    /* ── Run all phases ── */
    phase_sgemm_sweep(base_iters * 2, &seed);     /* 2x: core workload */
    phase_fused_ffn(base_iters, &seed);            /* 1x: FFN blocks */
    phase_attention(base_iters / 2 + 1, &seed);    /* 0.5x: attention is expensive */
    phase_backward(base_iters, &seed);             /* 1x: backward passes */
    phase_cordic(base_iters * 4, &seed);           /* 4x: CORDIC is fast */
    phase_activations(base_iters * 10, &seed);     /* 10x: activations are cheap */
    phase_interleaved(base_iters * 2, &seed);      /* 2x: mixed sizes */

    double total_elapsed = now_ms() - total_start;

    /* ═══════════════════════════════════════════════════════════════════════
     * Summary
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n══════════════════════════════════════════════════\n");
    printf("  LONG-RUN SUMMARY\n");
    printf("══════════════════════════════════════════════════\n");
    printf("  Total time:     %.1f seconds\n", total_elapsed / 1000);
    printf("  Total ops:      %.2e\n", (double)g_total_ops);
    printf("  Avg throughput:  %.1f GFLOPS\n",
           (double)g_total_ops / (total_elapsed * 1e6));
    printf("  Checks passed:  %d\n", g_passed);
    printf("  Checks failed:  %d\n", g_failed);
    printf("  Total checks:   %d\n", g_passed + g_failed);
    printf("\n  %s\n",
           g_failed == 0 ? "LONG-RUN STRESS TEST PASSED" : "FAILURES DETECTED");
    printf("══════════════════════════════════════════════════\n");

    rac_hal_shutdown();
    return g_failed == 0 ? 0 : 1;
}
