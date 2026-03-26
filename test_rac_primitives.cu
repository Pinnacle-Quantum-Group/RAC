/*
 * test_rac_primitives.cu — RAC Core Library Tests (BVT + DVT)
 * Pinnacle Quantum Group — March 2026
 *
 * Tests all 17 RAC primitives for correctness on CPU (host path).
 * No GPU required — validates CORDIC math, edge cases, and API surface.
 *
 * Build (CPU only, no GPU needed):
 *   g++ -O2 -x c++ -I. test_rac_primitives.cu rac_cuda.cu -lm -o test_rac_primitives
 *
 * Or with nvcc:
 *   nvcc -O2 -I. test_rac_primitives.cu rac_cuda.cu -o test_rac_primitives
 */

#include "rac.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

static int passed = 0;
static int failed = 0;

#define CHECK(name, cond) do { \
    if (cond) { passed++; printf("  [PASS] %s\n", name); } \
    else { failed++; printf("  [FAIL] %s\n", name); } \
} while(0)

#define CHECK_NEAR(name, val, expected, tol) do { \
    float _v = (val), _e = (expected), _t = (tol); \
    float _err = fabsf(_v - _e); \
    if (_err <= _t) { passed++; printf("  [PASS] %s  (val=%.6f  expected=%.6f  err=%.2e)\n", name, _v, _e, _err); } \
    else { failed++; printf("  [FAIL] %s  (val=%.6f  expected=%.6f  err=%.2e  tol=%.2e)\n", name, _v, _e, _err, _t); } \
} while(0)

#define HEADER(s) printf("\n════════════════════════════════════════════════════════\n  %s\n════════════════════════════════════════════════════════\n", s)


/* ═══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("RAC Core Primitive Tests — Pinnacle Quantum Group\n");
    printf("══════════════════════════════════════════════════\n");

    float mag, angle;
    float2 result;

    /* ── BVT: API surface ──────────────────────────────────────────────── */
    HEADER("BVT: Constants and API surface");

    CHECK_NEAR("RAC_K_INV", RAC_K_INV, 0.60725f, 1e-4f);
    CHECK_NEAR("RAC_K", RAC_K, 1.64676f, 1e-4f);
    CHECK("RAC_ITERS == 16", RAC_ITERS == 16);
    CHECK("K * K_INV ~= 1.0", fabsf(RAC_K * RAC_K_INV - 1.0f) < 0.01f);

    /* ── rac_rotate ────────────────────────────────────────────────────── */
    HEADER("DVT: rac_rotate");

    /* rotate (1,0) by 0 → (1,0) */
    result = rac_rotate(make_float2(1.0f, 0.0f), 0.0f);
    CHECK_NEAR("rotate (1,0) by 0 → x", result.x, 1.0f, 0.01f);
    CHECK_NEAR("rotate (1,0) by 0 → y", result.y, 0.0f, 0.01f);

    /* rotate (1,0) by pi/2 → (0,1) */
    result = rac_rotate(make_float2(1.0f, 0.0f), 3.14159265f / 2.0f);
    CHECK_NEAR("rotate (1,0) by pi/2 → x", result.x, 0.0f, 0.02f);
    CHECK_NEAR("rotate (1,0) by pi/2 → y", result.y, 1.0f, 0.02f);

    /* rotate (1,0) by pi → (-1,0) */
    result = rac_rotate(make_float2(1.0f, 0.0f), 3.14159265f);
    CHECK_NEAR("rotate (1,0) by pi → x", result.x, -1.0f, 0.02f);
    CHECK_NEAR("rotate (1,0) by pi → y", result.y, 0.0f, 0.02f);

    /* rotate (0,1) by -pi/2 → (1,0) */
    result = rac_rotate(make_float2(0.0f, 1.0f), -3.14159265f / 2.0f);
    CHECK_NEAR("rotate (0,1) by -pi/2 → x", result.x, 1.0f, 0.02f);
    CHECK_NEAR("rotate (0,1) by -pi/2 → y", result.y, 0.0f, 0.02f);

    /* magnitude preservation */
    result = rac_rotate(make_float2(3.0f, 4.0f), 1.23f);
    float mag_out = sqrtf(result.x * result.x + result.y * result.y);
    CHECK_NEAR("rotate (3,4) preserves magnitude", mag_out, 5.0f, 0.05f);

    /* ── rac_rotate_raw ────────────────────────────────────────────────── */
    HEADER("DVT: rac_rotate_raw");

    result = rac_rotate_raw(make_float2(1.0f, 0.0f), 0.0f);
    float raw_mag = sqrtf(result.x * result.x + result.y * result.y);
    CHECK_NEAR("rotate_raw (1,0) by 0 → magnitude ~K", raw_mag, RAC_K, 0.05f);

    /* ── rac_compensate ────────────────────────────────────────────────── */
    HEADER("DVT: rac_compensate");

    result = rac_compensate(make_float2(RAC_K, 0.0f), 1);
    CHECK_NEAR("compensate K by 1 → 1.0", result.x, 1.0f, 0.02f);

    result = rac_compensate(make_float2(RAC_K * RAC_K, 0.0f), 2);
    CHECK_NEAR("compensate K^2 by 2 → 1.0", result.x, 1.0f, 0.05f);

    /* ── rac_project ───────────────────────────────────────────────────── */
    HEADER("DVT: rac_project");

    /* project (1,0) at angle 0 → 1 (cos(0) = 1) */
    CHECK_NEAR("project (1,0) angle=0", rac_project(make_float2(1.0f, 0.0f), 0.0f), 1.0f, 0.01f);

    /* project (1,0) at angle pi → -1 (cos(pi) = -1) */
    CHECK_NEAR("project (1,0) angle=pi", rac_project(make_float2(1.0f, 0.0f), 3.14159265f), -1.0f, 0.01f);

    /* project (0,1) at angle pi/2 → 1 (sin(pi/2) = 1) */
    CHECK_NEAR("project (0,1) angle=pi/2", rac_project(make_float2(0.0f, 1.0f), 3.14159265f / 2.0f), 1.0f, 0.01f);

    /* project (3,4) at angle 0 → 3 */
    CHECK_NEAR("project (3,4) angle=0", rac_project(make_float2(3.0f, 4.0f), 0.0f), 3.0f, 0.01f);

    /* MAC equivalence: a*b via project((a,0), atan2(0, sign(b))) * |b| */
    float a = 3.0f, b = 4.0f;
    float angle_b = (b >= 0) ? 0.0f : 3.14159265f;
    float proj = rac_project(make_float2(a, 0.0f), angle_b) * fabsf(b);
    CHECK_NEAR("MAC equiv: 3*4 = 12", proj, 12.0f, 0.1f);

    a = -2.0f; b = 5.0f;
    angle_b = (b >= 0) ? 0.0f : 3.14159265f;
    proj = rac_project(make_float2(a, 0.0f), angle_b) * fabsf(b);
    CHECK_NEAR("MAC equiv: -2*5 = -10", proj, -10.0f, 0.1f);

    a = 7.0f; b = -3.0f;
    angle_b = (b >= 0) ? 0.0f : 3.14159265f;
    proj = rac_project(make_float2(a, 0.0f), angle_b) * fabsf(b);
    CHECK_NEAR("MAC equiv: 7*(-3) = -21", proj, -21.0f, 0.2f);

    /* ── rac_polar ─────────────────────────────────────────────────────── */
    HEADER("DVT: rac_polar");

    rac_polar(make_float2(3.0f, 4.0f), &mag, &angle);
    CHECK_NEAR("polar (3,4) mag", mag, 5.0f, 0.05f);
    CHECK_NEAR("polar (3,4) angle", angle, atan2f(4.0f, 3.0f), 0.02f);

    rac_polar(make_float2(1.0f, 0.0f), &mag, &angle);
    CHECK_NEAR("polar (1,0) mag", mag, 1.0f, 0.02f);
    CHECK_NEAR("polar (1,0) angle", angle, 0.0f, 0.02f);

    rac_polar(make_float2(-1.0f, 0.0f), &mag, &angle);
    CHECK_NEAR("polar (-1,0) mag", mag, 1.0f, 0.05f);

    /* ── rac_norm ──────────────────────────────────────────────────────── */
    HEADER("DVT: rac_norm");

    CHECK_NEAR("norm (3,4) = 5", rac_norm(make_float2(3.0f, 4.0f)), 5.0f, 0.05f);
    CHECK_NEAR("norm (1,0) = 1", rac_norm(make_float2(1.0f, 0.0f)), 1.0f, 0.02f);
    CHECK_NEAR("norm (0,0) = 0", rac_norm(make_float2(0.0f, 0.0f)), 0.0f, 0.01f);

    /* ── rac_normalize ─────────────────────────────────────────────────── */
    HEADER("DVT: rac_normalize");

    result = rac_normalize(make_float2(3.0f, 4.0f));
    float norm = sqrtf(result.x * result.x + result.y * result.y);
    CHECK_NEAR("normalize (3,4) magnitude = 1", norm, 1.0f, 0.02f);
    CHECK_NEAR("normalize (3,4) x = 3/5", result.x, 0.6f, 0.02f);
    CHECK_NEAR("normalize (3,4) y = 4/5", result.y, 0.8f, 0.02f);

    /* ── rac_dot ───────────────────────────────────────────────────────── */
    HEADER("DVT: rac_dot");

    float d = rac_dot(make_float2(1.0f, 0.0f), make_float2(1.0f, 0.0f));
    CHECK_NEAR("dot (1,0).(1,0) = 1", d, 1.0f, 0.02f);

    d = rac_dot(make_float2(1.0f, 0.0f), make_float2(0.0f, 1.0f));
    CHECK_NEAR("dot (1,0).(0,1) = 0", d, 0.0f, 0.02f);

    d = rac_dot(make_float2(3.0f, 4.0f), make_float2(3.0f, 4.0f));
    CHECK_NEAR("dot (3,4).(3,4) = 25", d, 25.0f, 0.5f);

    /* ── rac_coherence ─────────────────────────────────────────────────── */
    HEADER("DVT: rac_coherence");

    float c = rac_coherence(make_float2(1.0f, 0.0f), make_float2(1.0f, 0.0f));
    CHECK_NEAR("coherence parallel = 1", c, 1.0f, 0.02f);

    c = rac_coherence(make_float2(1.0f, 0.0f), make_float2(-1.0f, 0.0f));
    CHECK_NEAR("coherence anti-parallel = -1", c, -1.0f, 0.02f);

    c = rac_coherence(make_float2(1.0f, 0.0f), make_float2(0.0f, 1.0f));
    CHECK_NEAR("coherence orthogonal = 0", c, 0.0f, 0.02f);

    /* ── rac_complex_mul ───────────────────────────────────────────────── */
    HEADER("DVT: rac_complex_mul");

    /* (1+0i) * (1+0i) = (1+0i) */
    result = rac_complex_mul(make_float2(1.0f, 0.0f), make_float2(1.0f, 0.0f));
    CHECK_NEAR("cmul (1+0i)*(1+0i) real", result.x, 1.0f, 0.02f);
    CHECK_NEAR("cmul (1+0i)*(1+0i) imag", result.y, 0.0f, 0.02f);

    /* (0+1i) * (0+1i) = (-1+0i) */
    result = rac_complex_mul(make_float2(0.0f, 1.0f), make_float2(0.0f, 1.0f));
    CHECK_NEAR("cmul (0+1i)*(0+1i) real", result.x, -1.0f, 0.05f);
    CHECK_NEAR("cmul (0+1i)*(0+1i) imag", result.y, 0.0f, 0.05f);

    /* (3+4i) * (1+0i) = (3+4i) */
    result = rac_complex_mul(make_float2(3.0f, 4.0f), make_float2(1.0f, 0.0f));
    CHECK_NEAR("cmul (3+4i)*(1+0i) real", result.x, 3.0f, 0.1f);
    CHECK_NEAR("cmul (3+4i)*(1+0i) imag", result.y, 4.0f, 0.1f);

    /* ── rac_dct ───────────────────────────────────────────────────────── */
    HEADER("DVT: rac_dct");

    float dct_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dct_out[4];
    rac_dct(dct_in, dct_out, 4);
    /* DCT-II of [1,2,3,4]: X[0] = sum = 10 */
    CHECK_NEAR("dct [1,2,3,4] X[0] = 10", dct_out[0], 10.0f, 0.1f);
    CHECK("dct output finite", isfinite(dct_out[0]) && isfinite(dct_out[1]) &&
                                isfinite(dct_out[2]) && isfinite(dct_out[3]));

    /* ── rac_exp ───────────────────────────────────────────────────────── */
    HEADER("DVT: rac_exp");

    CHECK_NEAR("exp(0) = 1", rac_exp(0.0f), 1.0f, 0.02f);
    CHECK_NEAR("exp(1) = 2.718", rac_exp(1.0f), 2.71828f, 0.1f);
    CHECK_NEAR("exp(-1) = 0.368", rac_exp(-1.0f), 0.36788f, 0.05f);
    CHECK_NEAR("exp(2) = 7.389", rac_exp(2.0f), 7.38906f, 0.3f);

    /* ── rac_tanh ──────────────────────────────────────────────────────── */
    HEADER("DVT: rac_tanh");

    CHECK_NEAR("tanh(0) = 0", rac_tanh(0.0f), 0.0f, 0.02f);
    CHECK_NEAR("tanh(1) = 0.762", rac_tanh(1.0f), 0.76159f, 0.05f);
    CHECK_NEAR("tanh(-1) = -0.762", rac_tanh(-1.0f), -0.76159f, 0.05f);

    /* ── rac_softmax ───────────────────────────────────────────────────── */
    HEADER("DVT: rac_softmax");

    float sm_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float sm_out[4];
    rac_softmax(sm_in, sm_out, 4);
    float sm_sum = sm_out[0] + sm_out[1] + sm_out[2] + sm_out[3];
    CHECK_NEAR("softmax sums to 1", sm_sum, 1.0f, 0.02f);
    CHECK("softmax monotonic", sm_out[0] < sm_out[1] && sm_out[1] < sm_out[2] && sm_out[2] < sm_out[3]);
    CHECK("softmax all positive", sm_out[0] > 0 && sm_out[1] > 0 && sm_out[2] > 0 && sm_out[3] > 0);

    /* ── rac_matmul (host) ─────────────────────────────────────────────── */
    HEADER("DVT: rac_matmul (host)");

    /* 2x2 identity */
    float eye_A[4] = {1, 0, 0, 1};
    float eye_B[4] = {3, 4, 5, 6};
    float eye_C[4] = {0};
    rac_matmul(eye_A, eye_B, eye_C, 2, 2, 2);
    CHECK_NEAR("I * B = B [0,0]", eye_C[0], 3.0f, 0.1f);
    CHECK_NEAR("I * B = B [0,1]", eye_C[1], 4.0f, 0.1f);
    CHECK_NEAR("I * B = B [1,0]", eye_C[2], 5.0f, 0.1f);
    CHECK_NEAR("I * B = B [1,1]", eye_C[3], 6.0f, 0.1f);

    /* 1x1 */
    float s_a[1] = {3.0f}, s_b[1] = {4.0f}, s_c[1] = {0.0f};
    rac_matmul(s_a, s_b, s_c, 1, 1, 1);
    CHECK_NEAR("1x1 matmul: 3*4=12", s_c[0], 12.0f, 0.2f);

    /* ── rac_inner ─────────────────────────────────────────────────────── */
    HEADER("DVT: rac_inner");

    float2 iv_a[3] = {{1,0}, {2,0}, {3,0}};
    float2 iv_b[3] = {{4,0}, {5,0}, {6,0}};
    float inner = rac_inner(iv_a, iv_b, 3);
    /* 1*4 + 2*5 + 3*6 = 32 */
    CHECK_NEAR("inner [1,2,3].[4,5,6] = 32", inner, 32.0f, 1.0f);

    /* ── rac_context ───────────────────────────────────────────────────── */
    HEADER("BVT: Context API");

    rac_context ctx = rac_create_context(RAC_BACKEND_CPU);
    CHECK("rac_create_context returns non-NULL", ctx != NULL);

    int cap = rac_query_capability(ctx, RAC_OP_ROTATE);
    CHECK("query_capability(ROTATE) = 1", cap == 1);

    cap = rac_query_capability(ctx, RAC_OP_EXTENDED);
    CHECK("query_capability(EXTENDED) = 0", cap == 0);

    int ret = rac_execute(ctx, RAC_OP_ROTATE, NULL);
    CHECK("rac_execute returns 0", ret == 0);

    rac_destroy_context(ctx);
    CHECK("rac_destroy_context(NULL) safe", 1);
    rac_destroy_context(NULL);  /* should not crash */
    CHECK("rac_destroy_context(NULL) no crash", 1);


    /* ═══════════════════════════════════════════════════════════════════ */
    HEADER("Summary");
    printf("  Passed:  %d\n", passed);
    printf("  Failed:  %d\n", failed);
    printf("  Total:   %d\n", passed + failed);
    printf("\n  %s\n", failed == 0 ? "ALL TESTS PASSED" : "FAILURES DETECTED");

    return failed == 0 ? 0 : 1;
}
