/*
 * bench_rac_alu.c — Microbenchmarks for the RAC Adder + ALU
 * Pinnacle Quantum Group — April 2026
 *
 * Measures:
 *   - Raw CORDIC micro-step throughput (the dual-adder cell)
 *   - Full 16-iter rotate / project / polar / normalize (ALU vs rac_cpu)
 *   - Inner-product scaling (projection accumulator under load)
 *   - Hyperbolic exp / tanh (ALU vs libm)
 *
 * All timings use CLOCK_MONOTONIC. Runs are sized so each benchmark takes
 * ~200ms on a modern desktop. Results are reported as:
 *   - ns/op         average time per operation
 *   - Mops/s        millions of operations per second
 *   - speedup       rac_cpu / rac_alu ratio (>1 = ALU faster)
 *
 * Build:
 *   cc -O3 -march=native -I. bench_rac_alu.c rac_alu.c rac_cpu.c -lm -o bench
 */

#define _POSIX_C_SOURCE 200112L
#include "rac_alu.h"
#include "rac_cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Timing helpers ──────────────────────────────────────────────────────── */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Prevent the compiler from optimising the benchmark body away. */
static volatile float _sink_f;
static volatile int   _sink_i;

#define SINK(x) do { _sink_f += (float)(x); } while (0)

/* ── Report helpers ──────────────────────────────────────────────────────── */

static void banner(const char *title) {
    printf("\n── %s ─────────────────────────────────────\n", title);
}

static void report(const char *name, double secs, long iters) {
    double ns  = secs * 1e9 / (double)iters;
    double mps = (double)iters / secs / 1e6;
    printf("  %-38s  %9.2f ns/op   %9.2f Mops/s\n", name, ns, mps);
}

static void report_pair(const char *name,
                        double s_cpu, long n_cpu,
                        double s_alu, long n_alu) {
    double ns_cpu = s_cpu * 1e9 / (double)n_cpu;
    double ns_alu = s_alu * 1e9 / (double)n_alu;
    double mps_cpu = (double)n_cpu / s_cpu / 1e6;
    double mps_alu = (double)n_alu / s_alu / 1e6;
    double ratio   = ns_cpu / ns_alu;
    printf("  %-28s  cpu %8.2f ns  %7.2f Mops/s | alu %8.2f ns  %7.2f Mops/s | ratio %4.2fx\n",
           name, ns_cpu, mps_cpu, ns_alu, mps_alu, ratio);
}

/* ── Benchmark ============================================================ */

/* Size targets per benchmark (tuned for ~200ms on a fast core). */
#define N_MICRO   20000000
#define N_ROT       800000
#define N_POLAR     800000
#define N_PROJECT  1500000
#define N_EXP      1500000
#define N_INNER_VEC  4096
#define N_INNER_REPS  1000

int main(int argc, char **argv) {
    int verbose = (argc > 1 && strcmp(argv[1], "-v") == 0);
    printf("RAC ALU microbenchmark — Pinnacle Quantum Group\n");
    printf("  compiled: %s %s\n", __DATE__, __TIME__);
    (void)verbose;

    /* ── 1. Raw micro-step cell throughput ─────────────────────────── */
    banner("1. CORDIC micro-step (the dual shift-add cell)");
    {
        rac_alu_state s;
        rac_alu_reset(&s);
        rac_alu_load(&s, 1.0f, 0.0f, 0.5f);
        rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);

        double t0 = now_sec();
        for (long i = 0; i < N_MICRO; i++) {
            if (s.iter >= RAC_ALU_ITERS) {
                /* re-prime without full reset so we stay hot */
                s.iter = 0;
                s.z = 0.5f;
            }
            rac_alu_micro_step(&s);
        }
        double dt = now_sec() - t0;
        SINK(s.x + s.y);
        report("circular micro-step", dt, N_MICRO);
    }
    {
        rac_alu_state s;
        rac_alu_reset(&s);
        rac_alu_load(&s, 1.0f, 0.0f, 0.5f);
        rac_alu_set_mode(&s, RAC_ALU_MODE_HYPERBOLIC, RAC_ALU_DIR_ROTATION);

        double t0 = now_sec();
        for (long i = 0; i < N_MICRO; i++) {
            if (s.iter >= RAC_ALU_ITERS) { s.iter = 0; s.z = 0.5f; }
            rac_alu_micro_step(&s);
        }
        double dt = now_sec() - t0;
        SINK(s.x + s.y);
        report("hyperbolic micro-step", dt, N_MICRO);
    }

    /* ── 2. Full 16-iter rotate (ALU vs rac_cpu) ───────────────────── */
    banner("2. Rotate — full 16-iteration sequence");
    {
        double t0 = now_sec();
        float acc = 0.0f;
        for (long i = 0; i < N_ROT; i++) {
            rac_vec2 r = rac_rotate((rac_vec2){1.0f, 0.0f},
                                    (float)(i & 127) * 0.01f);
            acc += r.x + r.y;
        }
        double s_cpu = now_sec() - t0; SINK(acc);

        t0 = now_sec();
        acc = 0.0f;
        for (long i = 0; i < N_ROT; i++) {
            rac_vec2 r = rac_alu_rotate((rac_vec2){1.0f, 0.0f},
                                        (float)(i & 127) * 0.01f);
            acc += r.x + r.y;
        }
        double s_alu = now_sec() - t0; SINK(acc);
        report_pair("rac_rotate", s_cpu, N_ROT, s_alu, N_ROT);
    }

    /* ── 3. Project (MAC equivalent) ───────────────────────────────── */
    banner("3. Project — MAC equivalent (single dot against unit vector)");
    {
        double t0 = now_sec();
        float acc = 0.0f;
        for (long i = 0; i < N_PROJECT; i++) {
            acc += rac_project((rac_vec2){1.0f, 0.5f},
                               (float)(i & 255) * 0.01f);
        }
        double s_cpu = now_sec() - t0; SINK(acc);

        t0 = now_sec();
        acc = 0.0f;
        for (long i = 0; i < N_PROJECT; i++) {
            acc += rac_alu_project((rac_vec2){1.0f, 0.5f},
                                   (float)(i & 255) * 0.01f);
        }
        double s_alu = now_sec() - t0; SINK(acc);
        report_pair("rac_project", s_cpu, N_PROJECT, s_alu, N_PROJECT);
    }

    /* ── 4. Polar — vectoring mode ─────────────────────────────────── */
    banner("4. Polar — Cartesian→polar via vectoring mode");
    {
        double t0 = now_sec();
        float acc = 0.0f;
        for (long i = 0; i < N_POLAR; i++) {
            float m, a;
            rac_polar((rac_vec2){3.0f + 0.001f * i, 4.0f}, &m, &a);
            acc += m + a;
        }
        double s_cpu = now_sec() - t0; SINK(acc);

        t0 = now_sec();
        acc = 0.0f;
        for (long i = 0; i < N_POLAR; i++) {
            float m, a;
            rac_alu_polar((rac_vec2){3.0f + 0.001f * i, 4.0f}, &m, &a);
            acc += m + a;
        }
        double s_alu = now_sec() - t0; SINK(acc);
        report_pair("rac_polar", s_cpu, N_POLAR, s_alu, N_POLAR);
    }

    /* ── 5. Inner product — projection accumulator ─────────────────── */
    banner("5. Inner product — projection accumulator under load");
    {
        rac_vec2 *a = aligned_alloc(64, N_INNER_VEC * sizeof(rac_vec2));
        rac_vec2 *b = aligned_alloc(64, N_INNER_VEC * sizeof(rac_vec2));
        if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }
        for (int i = 0; i < N_INNER_VEC; i++) {
            a[i].x = (float)((i % 31) - 15) * 0.1f;
            a[i].y = (float)((i % 17) - 8)  * 0.1f;
            b[i].x = (float)((i % 23) - 11) * 0.1f;
            b[i].y = (float)((i % 13) - 6)  * 0.1f;
        }

        /* rac_cpu.c path is OpenMP-parallel — for a fair 1-thread
         * comparison we measure both paths on a single thread. */
        double t0 = now_sec();
        float acc = 0.0f;
        for (int r = 0; r < N_INNER_REPS; r++) {
            acc += rac_inner(a, b, N_INNER_VEC);
        }
        double s_cpu = now_sec() - t0; SINK(acc);

        t0 = now_sec();
        acc = 0.0f;
        for (int r = 0; r < N_INNER_REPS; r++) {
            acc += rac_alu_inner(a, b, N_INNER_VEC);
        }
        double s_alu = now_sec() - t0; SINK(acc);

        long N = (long)N_INNER_VEC * (long)N_INNER_REPS;
        report_pair("rac_inner (per element)", s_cpu, N, s_alu, N);

        free(a); free(b);
    }

    /* ── 6. Hyperbolic exp / tanh ──────────────────────────────────── */
    banner("6. Hyperbolic — exp / tanh (ALU CORDIC vs libm)");
    {
        double t0 = now_sec();
        float acc = 0.0f;
        for (long i = 0; i < N_EXP; i++) {
            acc += expf((float)(i & 255) * 0.01f - 1.0f);
        }
        double s_lib = now_sec() - t0; SINK(acc);

        t0 = now_sec();
        acc = 0.0f;
        for (long i = 0; i < N_EXP; i++) {
            acc += rac_alu_exp((float)(i & 255) * 0.01f - 1.0f);
        }
        double s_alu = now_sec() - t0; SINK(acc);
        report_pair("exp (libm vs ALU)", s_lib, N_EXP, s_alu, N_EXP);

        t0 = now_sec();
        acc = 0.0f;
        for (long i = 0; i < N_EXP; i++) {
            acc += tanhf((float)(i & 255) * 0.01f - 1.0f);
        }
        s_lib = now_sec() - t0; SINK(acc);

        t0 = now_sec();
        acc = 0.0f;
        for (long i = 0; i < N_EXP; i++) {
            acc += rac_alu_tanh((float)(i & 255) * 0.01f - 1.0f);
        }
        s_alu = now_sec() - t0; SINK(acc);
        report_pair("tanh (libm vs ALU)", s_lib, N_EXP, s_alu, N_EXP);
    }

    /* ── 7. AVX2 batch rotate — 8-wide parallel CORDIC ─────────────── */
    banner("7. Batch rotate — AVX2 8-wide vs scalar ALU");
    {
        const int N = 1 << 16;
        const int REPS = 200;
        rac_vec2 *v     = aligned_alloc(64, N * sizeof(rac_vec2));
        float    *theta = aligned_alloc(64, N * sizeof(float));
        rac_vec2 *out   = aligned_alloc(64, N * sizeof(rac_vec2));
        if (!v || !theta || !out) { fprintf(stderr, "alloc failed\n"); return 1; }
        for (int i = 0; i < N; i++) {
            v[i].x = 1.0f;  v[i].y = 0.0f;
            theta[i] = (float)(i & 127) * 0.01f;
        }

        printf("  AVX2 compiled in + runtime supported: %d\n", rac_alu_has_avx2());

        /* Scalar-per-element baseline */
        double t0 = now_sec();
        float acc = 0.0f;
        for (int r = 0; r < REPS; r++) {
            for (int i = 0; i < N; i++) {
                rac_vec2 o = rac_alu_rotate(v[i], theta[i]);
                acc += o.x + o.y;
            }
        }
        double s_scalar = now_sec() - t0; SINK(acc);

        /* Batch AVX2 path */
        t0 = now_sec();
        acc = 0.0f;
        for (int r = 0; r < REPS; r++) {
            rac_alu_rotate_batch(v, theta, out, N);
            acc += out[0].x + out[N-1].y;
        }
        double s_batch = now_sec() - t0; SINK(acc);

        long total = (long)N * (long)REPS;
        report_pair("rotate_batch (per element)",
                    s_scalar, total, s_batch, total);

        free(v); free(theta); free(out);
    }

    /* ── 8. AVX2 batch inner product ───────────────────────────────── */
    banner("8. Batch inner product — AVX2 vs scalar ALU");
    {
        const int N = 1 << 12;
        const int REPS = 2000;
        rac_vec2 *a = aligned_alloc(64, N * sizeof(rac_vec2));
        rac_vec2 *b = aligned_alloc(64, N * sizeof(rac_vec2));
        if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }
        for (int i = 0; i < N; i++) {
            a[i].x = (float)((i % 31) - 15) * 0.1f;
            a[i].y = (float)((i % 17) - 8)  * 0.1f;
            b[i].x = (float)((i % 23) - 11) * 0.1f;
            b[i].y = (float)((i % 13) - 6)  * 0.1f;
        }

        double t0 = now_sec();
        float acc = 0.0f;
        for (int r = 0; r < REPS; r++) acc += rac_alu_inner(a, b, N);
        double s_scalar = now_sec() - t0; SINK(acc);

        t0 = now_sec();
        acc = 0.0f;
        for (int r = 0; r < REPS; r++) acc += rac_alu_inner_batch(a, b, N);
        double s_batch = now_sec() - t0; SINK(acc);

        long total = (long)N * (long)REPS;
        report_pair("inner_batch (per element)",
                    s_scalar, total, s_batch, total);

        free(a); free(b);
    }

    /* ── 9. exp argument reduction accuracy ────────────────────────── */
    banner("9. exp argument reduction — range & accuracy");
    {
        float xs[] = {-20.0f, -5.0f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 5.0f, 20.0f};
        int  nx   = (int)(sizeof(xs)/sizeof(xs[0]));
        printf("     x        libm expf       alu exp        rel err\n");
        for (int i = 0; i < nx; i++) {
            float lib = expf(xs[i]);
            float alu = rac_alu_exp(xs[i]);
            float err = (lib != 0.0f) ? fabsf(alu - lib) / fabsf(lib) : fabsf(alu - lib);
            printf("  %+8.2f  %14.6g  %14.6g   %.2e\n", xs[i], lib, alu, err);
        }
    }

    /* Keep the sinks from being optimised out. */
    _sink_i = (int)_sink_f;
    printf("\n  (sink=%d — prevents DCE)\n", _sink_i);
    return 0;
}
