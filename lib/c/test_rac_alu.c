/*
 * test_rac_alu.c — Build Verification Tests for the RAC Adder + ALU
 * Pinnacle Quantum Group — April 2026
 *
 * Verifies that:
 *   1. The ALU state machine behaves correctly for each opcode.
 *   2. Circular + hyperbolic CORDIC micro-steps produce correct results.
 *   3. Every ALU-routed primitive matches the rac_cpu.c reference within
 *      CORDIC precision.
 *   4. The projection accumulator reproduces rac_inner element-for-element.
 *
 * Build:
 *   cc -O3 -march=native -I. test_rac_alu.c rac_alu.c rac_cpu.c -lm -o test_alu
 */

#include "rac_alu.h"
#include "rac_cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int passed = 0, failed = 0;

#define CHECK(name, cond) do { \
    if (cond) { passed++; printf("  [PASS] %s\n", name); } \
    else      { failed++; printf("  [FAIL] %s\n", name); } \
} while(0)

#define HEADER(s) printf("\n== %s ==\n", s)

/* Tolerance: CORDIC-16 is accurate to ~2^-14 for the circular
 * primitives. Hyperbolic is coarser. */
#define TOL_CIRC 0.003f
#define TOL_HYP  0.05f

int main(void) {
    printf("RAC ALU BVT — Pinnacle Quantum Group\n");

    /* ── 1. Opcode name / mode name sanity ───────────────────────────── */
    HEADER("1. Introspection");
    CHECK("op name LOAD",      strcmp(rac_alu_op_name(RAC_ALU_OP_LOAD),      "LOAD") == 0);
    CHECK("op name MICRO",     strcmp(rac_alu_op_name(RAC_ALU_OP_MICRO),     "MICRO") == 0);
    CHECK("op name ACCUM",     strcmp(rac_alu_op_name(RAC_ALU_OP_ACCUM),     "ACCUM") == 0);
    CHECK("op name COMPENSATE",strcmp(rac_alu_op_name(RAC_ALU_OP_COMPENSATE),"COMPENSATE") == 0);
    CHECK("mode CIRCULAR",     strcmp(rac_alu_mode_name(RAC_ALU_MODE_CIRCULAR),   "CIRCULAR") == 0);
    CHECK("mode HYPERBOLIC",   strcmp(rac_alu_mode_name(RAC_ALU_MODE_HYPERBOLIC), "HYPERBOLIC") == 0);

    /* ── 2. State machine + dispatch ─────────────────────────────────── */
    HEADER("2. ALU state machine");
    rac_alu_state s;
    rac_alu_reset(&s);
    CHECK("reset x=0",       s.x == 0.0f);
    CHECK("reset y=0",       s.y == 0.0f);
    CHECK("reset z=0",       s.z == 0.0f);
    CHECK("reset acc=0",     s.acc == 0.0f);
    CHECK("reset iter=0",    s.iter == 0);
    CHECK("reset chain=0",   s.chain == 0);

    rac_alu_dispatch(&s, RAC_ALU_OP_LOAD, 1.0f, 2.0f, 3.0f);
    CHECK("dispatch LOAD x", s.x == 1.0f);
    CHECK("dispatch LOAD y", s.y == 2.0f);
    CHECK("dispatch LOAD z", s.z == 3.0f);

    rac_alu_dispatch(&s, RAC_ALU_OP_CLEAR_ACC, 0, 0, 0);
    CHECK("dispatch CLEAR_ACC", s.acc == 0.0f);

    rac_alu_dispatch(&s, RAC_ALU_OP_SET_MODE,
                     (float)RAC_ALU_MODE_HYPERBOLIC,
                     (float)RAC_ALU_DIR_VECTORING, 0);
    CHECK("dispatch SET_MODE mode", s.mode == RAC_ALU_MODE_HYPERBOLIC);
    CHECK("dispatch SET_MODE dir",  s.dir  == RAC_ALU_DIR_VECTORING);

    /* Sign decide */
    rac_alu_reset(&s);
    s.z = -0.5f;
    rac_alu_sign_decide(&s);
    CHECK("sign (rotation, z<0) = -1", s.d == -1.0f);
    s.z =  0.5f;
    rac_alu_sign_decide(&s);
    CHECK("sign (rotation, z>0) = +1", s.d ==  1.0f);
    s.dir = RAC_ALU_DIR_VECTORING;
    s.y   = -0.5f;
    rac_alu_sign_decide(&s);
    CHECK("sign (vectoring, y<0) = +1", s.d ==  1.0f);

    /* ── 3. Micro-step advances iter & chain ─────────────────────────── */
    HEADER("3. Micro-step counters");
    rac_alu_reset(&s);
    rac_alu_load(&s, 1.0f, 0.0f, 0.5f);
    rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);
    for (int i = 0; i < 4; i++) rac_alu_micro_step(&s);
    CHECK("iter advanced to 4",  s.iter  == 4);
    CHECK("chain advanced to 4", s.chain == 4);

    rac_alu_reset(&s);
    rac_alu_load(&s, 1.0f, 0.0f, 0.0f);
    rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);
    rac_alu_run(&s, RAC_ALU_ITERS);
    CHECK("run completes iter=ITERS", s.iter == RAC_ALU_ITERS);
    int micro_fail = rac_alu_micro_step(&s);
    CHECK("micro after completion returns -1", micro_fail == -1);

    /* ── 4. Rotation correctness ─────────────────────────────────────── */
    HEADER("4. Rotation");
    rac_vec2 v = {1.0f, 0.0f};
    rac_vec2 r0  = rac_rotate(v, 0.0f);
    rac_vec2 r0a = rac_alu_rotate(v, 0.0f);
    CHECK("rotate θ=0 x matches cpu", fabsf(r0.x - r0a.x) < TOL_CIRC);
    CHECK("rotate θ=0 y matches cpu", fabsf(r0.y - r0a.y) < TOL_CIRC);

    rac_vec2 rH  = rac_rotate(v, (float)M_PI / 2.0f);
    rac_vec2 rHa = rac_alu_rotate(v, (float)M_PI / 2.0f);
    CHECK("rotate π/2 x (~0)",            fabsf(rHa.x) < TOL_CIRC);
    CHECK("rotate π/2 y (~1)",            fabsf(rHa.y - 1.0f) < TOL_CIRC);
    CHECK("rotate π/2 x matches cpu",     fabsf(rH.x - rHa.x) < TOL_CIRC);
    CHECK("rotate π/2 y matches cpu",     fabsf(rH.y - rHa.y) < TOL_CIRC);

    /* ── 5. Projection = MAC equivalence ─────────────────────────────── */
    HEADER("5. Projection (MAC equivalent)");
    float p = rac_alu_project(v, 0.0f);
    CHECK("project (1,0) @ 0 ~ 1", fabsf(p - 1.0f) < TOL_CIRC);
    p = rac_alu_project(v, (float)M_PI);
    CHECK("project (1,0) @ π ~ -1", fabsf(p - (-1.0f)) < TOL_CIRC);
    p = rac_alu_project((rac_vec2){3.0f, 4.0f}, 0.0f);
    CHECK("project (3,4) @ 0 ~ 3", fabsf(p - 3.0f) < TOL_CIRC * 5);

    /* ── 6. Polar (vectoring mode) ───────────────────────────────────── */
    HEADER("6. Polar / vectoring");
    float mag, ang;
    rac_alu_polar((rac_vec2){3.0f, 4.0f}, &mag, &ang);
    CHECK("polar (3,4) mag ~5",      fabsf(mag - 5.0f) < 0.02f);
    CHECK("polar (3,4) angle ~0.927",fabsf(ang - atan2f(4.0f, 3.0f)) < TOL_CIRC);

    float nm = rac_alu_norm((rac_vec2){5.0f, 12.0f});
    CHECK("norm (5,12) ~13",         fabsf(nm - 13.0f) < 0.03f);

    rac_vec2 u = rac_alu_normalize((rac_vec2){5.0f, 0.0f});
    CHECK("normalize (5,0) x~1",     fabsf(u.x - 1.0f) < TOL_CIRC);
    CHECK("normalize (5,0) y~0",     fabsf(u.y) < TOL_CIRC);

    /* ── 7. Inner product via projection accumulator ─────────────────── */
    HEADER("7. Inner product / projection accumulator");
    rac_vec2 a[4] = {{1,0}, {0,1}, {1,1}, {-1,2}};
    rac_vec2 b[4] = {{1,0}, {1,0}, {1,1}, {2,1}};
    float ref = 0.0f;
    for (int i = 0; i < 4; i++) ref += a[i].x*b[i].x + a[i].y*b[i].y;
    float got = rac_alu_inner(a, b, 4);
    printf("  ref=%.4f alu=%.4f\n", ref, got);
    CHECK("inner product matches reference", fabsf(ref - got) < 0.05f);

    /* ── 8. Outer product ────────────────────────────────────────────── */
    HEADER("8. Outer product");
    rac_vec2 ao[2] = {{1,0}, {0,1}};
    rac_vec2 bo[2] = {{2,0}, {0,3}};
    float    C[4]  = {0};
    rac_alu_outer(ao, bo, C, 2, 2);
    /* C[i][j] = a[i] dot b[j] */
    CHECK("outer[0][0] ~ 2", fabsf(C[0] - 2.0f) < 0.05f);
    CHECK("outer[0][1] ~ 0", fabsf(C[1])        < 0.05f);
    CHECK("outer[1][0] ~ 0", fabsf(C[2])        < 0.05f);
    CHECK("outer[1][1] ~ 3", fabsf(C[3] - 3.0f) < 0.05f);

    /* ── 9. Gain compensation opcode ─────────────────────────────────── */
    HEADER("9. Gain compensation");
    rac_alu_reset(&s);
    rac_alu_load(&s, 1.0f, 0.0f, 0.5f);
    rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);
    rac_alu_run(&s, RAC_ALU_ITERS);
    /* raw magnitude should be ~K */
    float raw_mag = sqrtf(s.x*s.x + s.y*s.y);
    CHECK("raw mag ~ K", fabsf(raw_mag - RAC_ALU_K) < 0.01f);
    /* compensate 16 chained rotations */
    s.chain = 1;  /* one logical rotate */
    rac_alu_compensate(&s);
    float comp_mag = sqrtf(s.x*s.x + s.y*s.y);
    CHECK("compensated mag ~ 1", fabsf(comp_mag - 1.0f) < 0.01f);
    CHECK("compensate clears chain", s.chain == 0);

    /* ── 10. Hyperbolic CORDIC via ALU ───────────────────────────────── */
    HEADER("10. Hyperbolic");
    float ex_alu = rac_alu_exp(0.5f);
    float ex_ref = expf(0.5f);
    printf("  exp(0.5) alu=%.4f libm=%.4f\n", ex_alu, ex_ref);
    CHECK("exp(0.5) within hyp tolerance", fabsf(ex_alu - ex_ref) < TOL_HYP);

    float th_alu = rac_alu_tanh(0.5f);
    float th_ref = tanhf(0.5f);
    printf("  tanh(0.5) alu=%.4f libm=%.4f\n", th_alu, th_ref);
    CHECK("tanh(0.5) within hyp tolerance", fabsf(th_alu - th_ref) < TOL_HYP);

    float th0 = rac_alu_tanh(0.0f);
    CHECK("tanh(0) ~ 0", fabsf(th0) < TOL_HYP);

    /* ── 11. Dispatch table is complete ──────────────────────────────── */
    HEADER("11. Opcode dispatch completeness");
    rac_alu_reset(&s);
    int rc;
    rc = rac_alu_dispatch(&s, RAC_ALU_OP_LOAD, 1.0f, 0.0f, 1.0f); CHECK("dispatch LOAD ok",       rc == 0);
    rc = rac_alu_dispatch(&s, RAC_ALU_OP_SET_MODE, 0, 0, 0);       CHECK("dispatch SET_MODE ok",   rc == 0);
    rc = rac_alu_dispatch(&s, RAC_ALU_OP_MICRO, 0, 0, 0);          CHECK("dispatch MICRO ok",      rc == 0);
    rc = rac_alu_dispatch(&s, RAC_ALU_OP_SIGN,  0, 0, 0);          CHECK("dispatch SIGN ok",       rc == 0);
    rc = rac_alu_dispatch(&s, RAC_ALU_OP_RUN,   15.0f, 0, 0);      CHECK("dispatch RUN ok",        rc == 0);
    rc = rac_alu_dispatch(&s, RAC_ALU_OP_ACCUM, 1.0f, 0, 0);       CHECK("dispatch ACCUM ok",      rc == 0);
    rc = rac_alu_dispatch(&s, RAC_ALU_OP_COMPENSATE, 0, 0, 0);     CHECK("dispatch COMPENSATE ok", rc == 0);
    rc = rac_alu_dispatch(&s, RAC_ALU_OP_CLEAR_ACC, 0, 0, 0);      CHECK("dispatch CLEAR_ACC ok",  rc == 0);
    rc = rac_alu_dispatch(&s, (rac_alu_opcode)99, 0, 0, 0);        CHECK("dispatch unknown = -1",  rc == -1);

    /* ── 12. Cross-check ALU vs rac_cpu inside convergence range ────── */
    HEADER("12. ALU vs rac_cpu rotation sweep (|θ| ≤ π/2)");
    int mismatches = 0;
    for (float t = -1.5f; t <= 1.5f; t += 0.1f) {
        rac_vec2 r_cpu = rac_rotate((rac_vec2){1.0f, 0.0f}, t);
        rac_vec2 r_alu = rac_alu_rotate((rac_vec2){1.0f, 0.0f}, t);
        if (fabsf(r_cpu.x - r_alu.x) > TOL_CIRC) mismatches++;
        if (fabsf(r_cpu.y - r_alu.y) > TOL_CIRC) mismatches++;
    }
    printf("  mismatches across 31 angles: %d\n", mismatches);
    CHECK("no rotation mismatches in convergence range", mismatches == 0);

    /* ── 12b. Extended-range exp via argument reduction ──────────────── */
    HEADER("12b. Argument-reduced exp");
    {
        /* Runtime invariant: RAC_K_HYP_RECIP must be the reciprocal of K_HYP.
         * If this ever trips, someone edited one constant without the other. */
        CHECK("K_HYP * K_HYP_RECIP ≈ 1",
              fabsf(RAC_K_HYP * RAC_K_HYP_RECIP - 1.0f) < 0.01f);

        float xs[] = {-20.0f, -5.0f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 5.0f, 20.0f};
        int fails = 0;
        for (size_t i = 0; i < sizeof(xs)/sizeof(xs[0]); i++) {
            float lib = expf(xs[i]);
            float alu = rac_alu_exp(xs[i]);
            float rel = (lib != 0.0f) ? fabsf(alu - lib) / fabsf(lib) : 0.0f;
            if (rel > 0.05f) {
                fails++;
                printf("  x=%+6.2f alu=%-14g libm=%-14g rel=%.3e\n",
                       xs[i], (double)alu, (double)lib, rel);
            }
        }
        if (fails > 0) {
            printf("  HINT: rel~0.31 across all inputs => hyperbolic gain applied\n"
                   "        the wrong direction. Check rac_alu_exp uses\n"
                   "        RAC_ALU_K_HYP_RECIP (≈1.207), NOT RAC_ALU_K_HYP_INV.\n"
                   "        Also: if you got this after upgrading source, REBUILD\n"
                   "        the test binary — a stale .o from before the fix shows\n"
                   "        exactly this symptom.\n");
        }
        CHECK("exp range [-20, 20] within 5%% relative error", fails == 0);
    }

    /* ── 12c. AVX2 batch rotate (skipped if not compiled in) ─────────── */
    HEADER("12c. Batch rotate");
    {
        const int N = 64;
        rac_vec2 v[64], out[64];
        float theta[64];
        for (int i = 0; i < N; i++) {
            v[i] = (rac_vec2){1.0f, 0.0f};
            theta[i] = (float)(i % 16) * 0.1f;
        }
        rac_alu_rotate_batch(v, theta, out, N);
        int fails = 0;
        for (int i = 0; i < N; i++) {
            rac_vec2 ref = rac_alu_rotate(v[i], theta[i]);
            if (fabsf(out[i].x - ref.x) > TOL_CIRC ||
                fabsf(out[i].y - ref.y) > TOL_CIRC) fails++;
        }
        CHECK("batch rotate matches scalar per-element", fails == 0);
        printf("  has_avx2 = %d\n", rac_alu_has_avx2());
    }

    /* ── 12d. AVX2 batch inner — full-quadrant input ─────────────────── */
    HEADER("12d. Batch inner product (all quadrants)");
    {
        const int N = 1024;
        rac_vec2 *a = (rac_vec2 *)malloc(N * sizeof(rac_vec2));
        rac_vec2 *b = (rac_vec2 *)malloc(N * sizeof(rac_vec2));
        float ref = 0.0f;
        for (int i = 0; i < N; i++) {
            a[i].x = (float)((i % 31) - 15) * 0.1f;
            a[i].y = (float)((i % 17) - 8)  * 0.1f;
            b[i].x = (float)((i % 23) - 11) * 0.1f;  /* spans negative x */
            b[i].y = (float)((i % 13) - 6)  * 0.1f;
            ref += a[i].x * b[i].x + a[i].y * b[i].y;
        }
        float scalar = rac_alu_inner(a, b, N);
        float batch  = rac_alu_inner_batch(a, b, N);
        printf("  N=%d naive=%.4f scalar_alu=%.4f batch_alu=%.4f\n",
               N, ref, scalar, batch);
        CHECK("scalar ALU inner matches naive dot within 1%%",
              fabsf(scalar - ref) / fabsf(ref) < 0.01f);
        CHECK("batch ALU inner matches naive dot within 1%%",
              fabsf(batch - ref) / fabsf(ref) < 0.01f);
        free(a); free(b);
    }

    /* ── 13. Extended range via quadrant folding ─────────────────────── */
    HEADER("13. Quadrant folding (extended range)");
    /* The ALU folds angles outside [-π/2, π/2] by flipping the input
     * vector — a feature bare CORDIC lacks. Verify for a few angles. */
    rac_vec2 rot_pi = rac_alu_rotate((rac_vec2){1.0f, 0.0f}, (float)M_PI);
    CHECK("rotate π gives (-1, 0)",
          fabsf(rot_pi.x + 1.0f) < TOL_CIRC && fabsf(rot_pi.y) < TOL_CIRC);

    rac_vec2 rot_3pi2 = rac_alu_rotate((rac_vec2){1.0f, 0.0f},
                                       1.5f * (float)M_PI);
    CHECK("rotate 3π/2 gives (0, -1)",
          fabsf(rot_3pi2.x) < TOL_CIRC && fabsf(rot_3pi2.y + 1.0f) < TOL_CIRC);

    rac_vec2 rot_2pi = rac_alu_rotate((rac_vec2){1.0f, 0.0f},
                                      2.0f * (float)M_PI);
    CHECK("rotate 2π gives (1, 0)",
          fabsf(rot_2pi.x - 1.0f) < TOL_CIRC && fabsf(rot_2pi.y) < TOL_CIRC);

    HEADER("Summary");
    printf("  passed=%d failed=%d total=%d\n", passed, failed, passed + failed);
    printf("  %s\n", failed == 0 ? "ALL RAC ALU BVT PASSED" : "RAC ALU BVT FAILURES");
    return failed == 0 ? 0 : 1;
}
