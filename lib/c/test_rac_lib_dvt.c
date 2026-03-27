/*
 * test_rac_lib_dvt.c — Design Verification Tests for RAC C Library + HAL
 * Pinnacle Quantum Group — March 2026
 *
 * Thorough correctness: shape sweeps, numerical stability, AVX2 vs scalar
 * parity, HAL cache-tuned tiling, activation accuracy. ~30s.
 *
 * Build:
 *   cc -O3 -mavx2 -mfma -fopenmp -I. \
 *     test_rac_lib_dvt.c rac_cpu.c rac_avx2.c rac_hal.c -lm -o test_dvt
 */

#include "rac_cpu.h"
#include "rac_avx2.h"
#include "rac_hal.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static int passed = 0, failed = 0;

#define CHECK(name, cond) do { \
    if (cond) { passed++; printf("  [PASS] %s\n", name); } \
    else      { failed++; printf("  [FAIL] %s\n", name); } \
} while(0)

#define HEADER(s) printf("\n══════════════════════════════════════════\n  %s\n══════════════════════════════════════════\n", s)

/* Reference scalar matmul for comparison */
static void ref_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
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

static void rand_fill(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

int main(void) {
    printf("RAC Library DVT — Pinnacle Quantum Group\n");
    srand(42);

    rac_hal_init();
    rac_config cfg = rac_default_config();

    /* ── DVT-1: SGEMM shape sweep (scalar) ────────────────────────── */
    HEADER("DVT-1: SGEMM shape sweep");

    int shapes[][3] = {
        {1,1,1}, {1,64,1}, {4,4,4}, {16,16,16},
        {32,64,128}, {128,64,32}, {63,65,67},
        {127,255,131}, {256,256,256}, {512,512,512},
    };
    int n_shapes = sizeof(shapes)/sizeof(shapes[0]);

    for (int s = 0; s < n_shapes; s++) {
        int M = shapes[s][0], K = shapes[s][1], N = shapes[s][2];
        float *A = malloc(M*K*sizeof(float));
        float *B = malloc(K*N*sizeof(float));
        float *C_rac = malloc(M*N*sizeof(float));
        float *C_ref = malloc(M*N*sizeof(float));

        rand_fill(A, M*K);
        rand_fill(B, K*N);

        ref_matmul(A, B, C_ref, M, N, K);
        rac_matmul(A, B, C_rac, M, N, K, &cfg);

        float err = max_err(C_ref, C_rac, M*N);
        float tol = fmaxf(0.01f, K * 1e-5f);
        char name[128];
        snprintf(name, sizeof(name), "scalar %dx%d@%dx%d err=%.2e", M, K, K, N, err);
        CHECK(name, err < tol);

        free(A); free(B); free(C_rac); free(C_ref);
    }

    /* ── DVT-2: AVX2 vs scalar parity ─────────────────────────────── */
    HEADER("DVT-2: AVX2 vs scalar parity");

    if (rac_has_avx2()) {
        for (int s = 0; s < n_shapes; s++) {
            int M = shapes[s][0], K = shapes[s][1], N = shapes[s][2];
            float *A = malloc(M*K*sizeof(float));
            float *B = malloc(K*N*sizeof(float));
            float *C_scalar = calloc(M*N, sizeof(float));
            float *C_avx2   = calloc(M*N, sizeof(float));

            rand_fill(A, M*K);
            rand_fill(B, K*N);

            rac_sgemm(A, B, C_scalar, M, N, K, 1.0f, 0.0f, &cfg);
            rac_sgemm_avx2(A, B, C_avx2, M, N, K, 1.0f, 0.0f, &cfg);

            float err = max_err(C_scalar, C_avx2, M*N);
            char name[128];
            snprintf(name, sizeof(name), "AVX2==scalar %dx%dx%d err=%.2e", M, K, N, err);
            CHECK(name, err < 1e-4f);

            free(A); free(B); free(C_scalar); free(C_avx2);
        }
    } else {
        printf("  [SKIP] AVX2 not available\n");
    }

    /* ── DVT-3: HAL dispatch vs reference ─────────────────────────── */
    HEADER("DVT-3: HAL dispatch correctness");

    for (int size = 64; size <= 512; size *= 2) {
        int M=size, K=size, N=size;
        float *A = malloc(M*K*sizeof(float));
        float *B = malloc(K*N*sizeof(float));
        float *C_hal = calloc(M*N, sizeof(float));
        float *C_ref = calloc(M*N, sizeof(float));

        rand_fill(A, M*K);
        rand_fill(B, K*N);

        ref_matmul(A, B, C_ref, M, N, K);
        rac_hal_matmul(A, B, C_hal, M, N, K);

        float err = max_err(C_ref, C_hal, M*N);
        char name[128];
        snprintf(name, sizeof(name), "HAL %dx%d err=%.2e", size, size, err);
        CHECK(name, err < fmaxf(0.01f, K * 1e-5f));

        free(A); free(B); free(C_hal); free(C_ref);
    }

    /* ── DVT-4: Alpha/Beta SGEMM ──────────────────────────────────── */
    HEADER("DVT-4: Alpha/Beta SGEMM");

    {
        float A[4] = {1,2,3,4}, B[4] = {1,0,0,1};
        float C[4] = {10,20,30,40};
        rac_sgemm(A, B, C, 2, 2, 2, 2.0f, 0.5f, &cfg);
        /* C[0] = 2*(1*1+2*0) + 0.5*10 = 2+5 = 7 */
        CHECK("alpha=2 beta=0.5 C[0]=7", fabsf(C[0] - 7.0f) < 0.1f);
        /* C[3] = 2*(3*0+4*1) + 0.5*40 = 8+20 = 28 */
        CHECK("alpha=2 beta=0.5 C[3]=28", fabsf(C[3] - 28.0f) < 0.1f);
    }

    /* ── DVT-5: Numerical stability ───────────────────────────────── */
    HEADER("DVT-5: Numerical stability");

    {
        int N = 64;
        float *A = malloc(N*N*sizeof(float));
        float *B = malloc(N*N*sizeof(float));
        float *C = calloc(N*N, sizeof(float));

        /* Large values */
        for (int i = 0; i < N*N; i++) { A[i] = 1000.0f; B[i] = 1000.0f; }
        rac_hal_matmul(A, B, C, N, N, N);
        int any_nan = 0, any_inf = 0;
        for (int i = 0; i < N*N; i++) {
            if (isnan(C[i])) any_nan = 1;
            if (isinf(C[i])) any_inf = 1;
        }
        CHECK("large values: no NaN", !any_nan);
        CHECK("large values: no Inf", !any_inf);

        /* Small values */
        for (int i = 0; i < N*N; i++) { A[i] = 1e-6f; B[i] = 1e-6f; }
        rac_hal_matmul(A, B, C, N, N, N);
        CHECK("small values: C[0] >= 0", C[0] >= 0.0f);

        /* Zero */
        memset(A, 0, N*N*sizeof(float));
        rand_fill(B, N*N);
        rac_hal_matmul(A, B, C, N, N, N);
        float mx = 0;
        for (int i = 0; i < N*N; i++) if (fabsf(C[i]) > mx) mx = fabsf(C[i]);
        CHECK("zero A → zero C", mx < 1e-6f);

        free(A); free(B); free(C);
    }

    /* ── DVT-6: Fused linear activations ──────────────────────────── */
    HEADER("DVT-6: Fused linear activation correctness");

    {
        int M = 32, K = 64, N = 16;
        float *inp = malloc(M*K*sizeof(float));
        float *wt  = malloc(N*K*sizeof(float));
        float *bias = malloc(N*sizeof(float));
        float *out_none = calloc(M*N, sizeof(float));
        float *out_relu = calloc(M*N, sizeof(float));
        float *out_gelu = calloc(M*N, sizeof(float));
        float *out_silu = calloc(M*N, sizeof(float));

        rand_fill(inp, M*K);
        rand_fill(wt, N*K);
        rand_fill(bias, N);

        rac_fused_linear(inp, wt, bias, out_none, M, N, K, RAC_ACT_NONE, &cfg);
        rac_fused_linear(inp, wt, bias, out_relu, M, N, K, RAC_ACT_RELU, &cfg);
        rac_fused_linear(inp, wt, bias, out_gelu, M, N, K, RAC_ACT_GELU, &cfg);
        rac_fused_linear(inp, wt, bias, out_silu, M, N, K, RAC_ACT_SILU, &cfg);

        /* ReLU: should match max(0, out_none) */
        int relu_ok = 1;
        for (int i = 0; i < M*N; i++) {
            float expected = (out_none[i] > 0) ? out_none[i] : 0;
            if (fabsf(out_relu[i] - expected) > 1e-5f) { relu_ok = 0; break; }
        }
        CHECK("fused relu == max(0, linear)", relu_ok);

        /* GELU: out_gelu >= 0 when out_none > 1 (approximately) */
        CHECK("fused gelu runs without NaN",
              !isnan(out_gelu[0]) && !isnan(out_gelu[M*N-1]));

        /* SiLU: silu(x) < x for x > 0 (since sigmoid < 1) */
        CHECK("fused silu runs without NaN",
              !isnan(out_silu[0]) && !isnan(out_silu[M*N-1]));

        free(inp); free(wt); free(bias);
        free(out_none); free(out_relu); free(out_gelu); free(out_silu);
    }

    /* ── DVT-7: HAL cache-tuned tile size ─────────────────────────── */
    HEADER("DVT-7: HAL tile selection");

    const rac_hw_profile *hw = rac_hal_profile();
    if (hw) {
        CHECK("L1 tile is multiple of 8", hw->optimal_tile_sgemm % 8 == 0);
        CHECK("L1 tile >= 16", hw->optimal_tile_sgemm >= 16);
        CHECK("L1 tile <= 128", hw->optimal_tile_sgemm <= 128);
        CHECK("L2 tile >= L1 tile", hw->optimal_tile_sgemm_l2 >= hw->optimal_tile_sgemm);

        /* Verify: 3 tiles fit in L1 */
        int tile = hw->optimal_tile_sgemm;
        int tile_bytes = tile * tile * sizeof(float) * 3;
        int l1_bytes = hw->cache.l1d_size_kb * 1024;
        CHECK("3 L1 tiles fit in L1d", tile_bytes <= l1_bytes);

        printf("  L1d=%dKB → tile=%d (%d bytes for 3 tiles, %d avail)\n",
               hw->cache.l1d_size_kb, tile, tile_bytes, l1_bytes);
    }

    /* ── DVT-8: AVX2 batch activations ────────────────────────────── */
    HEADER("DVT-8: AVX2 activation parity");

    if (rac_has_avx2()) {
        int N = 1024;
        float *x = malloc(N*sizeof(float));
        float *out_scalar = malloc(N*sizeof(float));
        float *out_avx2   = malloc(N*sizeof(float));

        rand_fill(x, N);

        rac_relu(x, out_scalar, N);
        rac_relu_avx2(x, out_avx2, N);
        CHECK("AVX2 relu matches scalar", max_err(out_scalar, out_avx2, N) < 1e-6f);

        rac_gelu(x, out_scalar, N);
        rac_gelu_avx2(x, out_avx2, N);
        float gerr = max_err(out_scalar, out_avx2, N);
        char name[128];
        snprintf(name, sizeof(name), "AVX2 gelu matches scalar (err=%.2e)", gerr);
        CHECK(name, gerr < 0.05f);  /* GELU approx has some error */

        free(x); free(out_scalar); free(out_avx2);
    } else {
        printf("  [SKIP] AVX2 not available\n");
    }

    /* ── DVT-9: CORDIC primitives sweep ───────────────────────────── */
    HEADER("DVT-9: CORDIC primitive accuracy");

    /* Rotate at many angles */
    int rot_ok = 1;
    for (int deg = 0; deg < 360; deg += 15) {
        float theta = deg * RAC_PI / 180.0f;
        rac_vec2 v = {1.0f, 0.0f};
        rac_vec2 r = rac_rotate(v, theta);
        float mag = sqrtf(r.x*r.x + r.y*r.y);
        if (fabsf(mag - 1.0f) > 0.05f) { rot_ok = 0; break; }
    }
    CHECK("rotate preserves magnitude at all angles", rot_ok);

    /* Dot product accuracy */
    CHECK("dot orthogonal = 0",
          fabsf(rac_dot((rac_vec2){1,0}, (rac_vec2){0,1})) < 0.02f);
    CHECK("dot parallel = |a||b|",
          fabsf(rac_dot((rac_vec2){3,0}, (rac_vec2){4,0}) - 12.0f) < 0.2f);

    /* Softmax */
    float sm_in[8] = {1,2,3,4,5,6,7,8}, sm_out[8];
    rac_softmax(sm_in, sm_out, 8);
    float sm_sum = 0; for (int i = 0; i < 8; i++) sm_sum += sm_out[i];
    CHECK("softmax(8) sums to 1", fabsf(sm_sum - 1.0f) < 0.02f);
    CHECK("softmax monotonic", sm_out[7] > sm_out[0]);

    /* ── Summary ──────────────────────────────────────────────────── */
    HEADER("DVT Summary");
    printf("  Passed: %d\n  Failed: %d\n  Total:  %d\n", passed, failed, passed+failed);
    printf("\n  %s\n", failed == 0 ? "ALL DVT PASSED" : "DVT FAILURES DETECTED");

    rac_hal_shutdown();
    return failed == 0 ? 0 : 1;
}
