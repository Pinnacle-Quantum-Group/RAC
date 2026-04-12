/*
 * test_rac_lib_bvt.c — Build Verification Tests for RAC C Library + HAL
 * Pinnacle Quantum Group — March 2026
 *
 * Quick smoke tests: API surface, HAL init, basic dispatch. <5s.
 *
 * Build:
 *   cc -O3 -mavx2 -mfma -fopenmp -I. \
 *     test_rac_lib_bvt.c rac_cpu.c rac_avx2.c rac_hal.c -lm -o test_bvt
 */

#include "rac_cpu.h"
#include "rac_avx2.h"
#include "rac_hal.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int passed = 0, failed = 0;

#define CHECK(name, cond) do { \
    if (cond) { passed++; printf("  [PASS] %s\n", name); } \
    else      { failed++; printf("  [FAIL] %s\n", name); } \
} while(0)

#define HEADER(s) printf("\n══════════════════════════════════════════\n  %s\n══════════════════════════════════════════\n", s)

int main(void) {
    printf("RAC Library BVT — Pinnacle Quantum Group\n");

    /* ── BVT-1: Constants ─────────────────────────────────────────── */
    HEADER("BVT-1: Constants");
    CHECK("RAC_K_INV defined", fabsf(RAC_K_INV - 0.60725f) < 1e-4f);
    CHECK("RAC_K defined", fabsf(RAC_K - 1.64676f) < 1e-4f);
    CHECK("RAC_ITERS == 16", RAC_ITERS == 16);
    CHECK("RAC_PI defined", fabsf(RAC_PI - 3.14159265f) < 1e-6f);
    CHECK("K * K_INV ~= 1", fabsf(RAC_K * RAC_K_INV - 1.0f) < 0.01f);

    /* ── BVT-2: Config ────────────────────────────────────────────── */
    HEADER("BVT-2: Config");
    rac_config cfg = rac_default_config();
    CHECK("default threads == 0 (auto)", cfg.num_threads == 0);
    CHECK("default tile == 64", cfg.tile_size == 64);
    CHECK("default iters == 16", cfg.cordic_iters == 16);

    /* ── BVT-3: Scalar primitives ─────────────────────────────────── */
    HEADER("BVT-3: Scalar primitives");
    rac_vec2 v = {1.0f, 0.0f};
    rac_vec2 r = rac_rotate(v, 0.0f);
    CHECK("rotate (1,0) by 0 runs", fabsf(r.x - 1.0f) < 0.02f);

    float p = rac_project(v, 0.0f);
    CHECK("project (1,0) at 0 = 1", fabsf(p - 1.0f) < 0.01f);

    float mag, angle;
    rac_polar((rac_vec2){3.0f, 4.0f}, &mag, &angle);
    CHECK("polar (3,4) mag ~5", fabsf(mag - 5.0f) < 0.1f);

    CHECK("exp(0) = 1", fabsf(rac_exp(0.0f) - 1.0f) < 1e-6f);  /* CPU uses libm expf */
    CHECK("tanh(0) = 0", fabsf(rac_tanh(0.0f)) < 0.02f);

    /* ── BVT-4: SGEMM runs ───────────────────────────────────────── */
    HEADER("BVT-4: SGEMM smoke test");
    float A[4] = {1,0,0,1}, B[4] = {3,4,5,6}, C[4] = {0};
    rac_status st = rac_matmul(A, B, C, 2, 2, 2, &cfg);
    CHECK("rac_matmul returns OK", st == RAC_OK);
    CHECK("I @ B = B [0,0]", fabsf(C[0] - 3.0f) < 0.1f);
    CHECK("I @ B = B [1,1]", fabsf(C[3] - 6.0f) < 0.1f);

    /* ── BVT-5: Error handling ────────────────────────────────────── */
    HEADER("BVT-5: Error handling");
    CHECK("NULL A → ERR", rac_matmul(NULL, B, C, 2, 2, 2, &cfg) == RAC_ERR_NULL_PTR);
    CHECK("M=0 → ERR", rac_matmul(A, B, C, 0, 2, 2, &cfg) == RAC_ERR_INVALID_DIM);

    /* ── BVT-6: Activations ───────────────────────────────────────── */
    HEADER("BVT-6: Batch activations");
    float x[4] = {-1, 0, 1, 2}, out[4];
    rac_relu(x, out, 4);
    CHECK("relu(-1)=0", fabsf(out[0]) < 1e-6f);
    CHECK("relu(2)=2", fabsf(out[3] - 2.0f) < 1e-6f);

    rac_gelu(x, out, 4);
    CHECK("gelu(0)=0", fabsf(out[1]) < 0.01f);
    CHECK("gelu runs on negative", 1);

    float sm_in[4] = {1,2,3,4}, sm_out[4];
    rac_softmax(sm_in, sm_out, 4);
    float sm_sum = sm_out[0]+sm_out[1]+sm_out[2]+sm_out[3];
    CHECK("softmax sums to 1", fabsf(sm_sum - 1.0f) < 0.02f);

    /* ── BVT-7: Fused linear ─────────────────────────────────────── */
    HEADER("BVT-7: Fused linear");
    float inp[4] = {1,2,3,4};     /* 1x4 */
    float wt[8] = {1,0,0,0, 0,1,0,0}; /* 2x4 */
    float bias[2] = {10, 20};
    float fout[2] = {0};
    st = rac_fused_linear(inp, wt, bias, fout, 1, 2, 4, RAC_ACT_RELU, &cfg);
    CHECK("fused_linear returns OK", st == RAC_OK);
    CHECK("out[0] = relu(1+10) = 11", fabsf(fout[0] - 11.0f) < 0.1f);
    CHECK("out[1] = relu(2+20) = 22", fabsf(fout[1] - 22.0f) < 0.1f);

    /* ── BVT-8: AVX2 detection ────────────────────────────────────── */
    HEADER("BVT-8: AVX2 detection");
    int avx2 = rac_has_avx2();
    printf("  rac_has_avx2() = %d\n", avx2);
    CHECK("rac_has_avx2 returns 0 or 1", avx2 == 0 || avx2 == 1);

    if (avx2) {
        memset(C, 0, sizeof(C));
        st = rac_sgemm_avx2(A, B, C, 2, 2, 2, 1.0f, 0.0f, &cfg);
        CHECK("AVX2 SGEMM returns OK", st == RAC_OK);
        CHECK("AVX2 I @ B = B [0,0]", fabsf(C[0] - 3.0f) < 0.1f);
    }

    /* ── BVT-9: HAL init ──────────────────────────────────────────── */
    HEADER("BVT-9: HAL init");
    st = rac_hal_init();
    CHECK("rac_hal_init returns OK", st == RAC_OK);

    const rac_hw_profile *hw = rac_hal_profile();
    CHECK("profile is non-NULL", hw != NULL);
    CHECK("cpu_name is non-empty", hw && strlen(hw->cpu_name) > 0);
    CHECK("num_physical_cores > 0", hw && hw->num_physical_cores > 0);
    CHECK("num_logical_cores > 0", hw && hw->num_logical_cores > 0);
    CHECK("L1d > 0", hw && hw->cache.l1d_size_kb > 0);
    CHECK("optimal_tile > 0", hw && hw->optimal_tile_sgemm > 0);
    CHECK("omp_threads > 0", hw && hw->omp_num_threads > 0);

    rac_hal_print_profile();

    /* ── BVT-10: HAL dispatched ops ───────────────────────────────── */
    HEADER("BVT-10: HAL dispatch");
    memset(C, 0, sizeof(C));
    st = rac_hal_matmul(A, B, C, 2, 2, 2);
    CHECK("rac_hal_matmul returns OK", st == RAC_OK);
    CHECK("HAL I @ B = B [0,0]", fabsf(C[0] - 3.0f) < 0.1f);

    memset(fout, 0, sizeof(fout));
    st = rac_hal_fused_linear(inp, wt, bias, fout, 1, 2, 4, RAC_ACT_GELU);
    CHECK("rac_hal_fused_linear returns OK", st == RAC_OK);
    CHECK("HAL fused output finite", isfinite(fout[0]) && isfinite(fout[1]));

    float rx[4] = {-1, 0, 1, 2}, rout[4];
    rac_hal_relu(rx, rout, 4);
    CHECK("HAL relu(-1)=0", fabsf(rout[0]) < 1e-6f);
    CHECK("HAL relu(2)=2", fabsf(rout[3] - 2.0f) < 1e-6f);

    /* ── BVT-11: HAL override ─────────────────────────────────────── */
    HEADER("BVT-11: HAL override");
    rac_hal_override ovr = {0, -1, -1, 32, 2}; /* force no AVX2, tile=32, threads=2 */
    rac_hal_set_override(&ovr);
    hw = rac_hal_profile();
    CHECK("override disables AVX2", hw && hw->use_avx2 == 0);
    CHECK("override tile=32", hw && hw->optimal_tile_sgemm == 32);
    CHECK("override threads=2", hw && hw->omp_num_threads == 2);

    /* Reset */
    rac_hal_override reset = {-1, -1, -1, 0, 0};
    rac_hal_set_override(&reset);

    rac_hal_shutdown();
    CHECK("rac_hal_shutdown OK", 1);
    CHECK("profile after shutdown is NULL", rac_hal_profile() == NULL);

    /* ── BVT-12: Tunable-precision CORDIC ─────────────────────────── */
    HEADER("BVT-12: Tunable-precision CORDIC");
    {
        rac_vec2 v_unit = {1.0f, 0.0f};
        rac_vec2 r8  = rac_rotate_n(v_unit, RAC_PI * 0.25f, 8);
        rac_vec2 r16 = rac_rotate_n(v_unit, RAC_PI * 0.25f, 16);
        float expected = cosf(RAC_PI * 0.25f);   /* ~0.7071 */
        CHECK("rotate_n(8) converges to ±1%", fabsf(r8.x - expected) < 0.01f);
        CHECK("rotate_n(16) within ±0.1%",    fabsf(r16.x - expected) < 0.001f);

        float sx, cx;
        rac_sincos(RAC_PI * 0.5f, &sx, &cx);
        CHECK("sincos(π/2) → sin≈1", fabsf(sx - 1.0f) < 0.01f);
        CHECK("sincos(π/2) → cos≈0", fabsf(cx) < 0.01f);

        float rs = rac_rsqrt(4.0f);
        CHECK("rsqrt(4) = 0.5",  fabsf(rs - 0.5f) < 1e-6f);
        CHECK("rsqrt(<=0) = 0",  rac_rsqrt(0.0f) == 0.0f && rac_rsqrt(-1.0f) == 0.0f);

        CHECK("sigmoid(0)  = 0.5",   fabsf(rac_sigmoid(0.0f) - 0.5f) < 1e-5f);
        CHECK("sigmoid(10) ≈ 1",     rac_sigmoid(10.0f) > 0.999f);
        CHECK("sigmoid(-10) ≈ 0",    rac_sigmoid(-10.0f) < 0.001f);
    }

    /* ── BVT-13: LayerNorm + RMSNorm ──────────────────────────────── */
    HEADER("BVT-13: LayerNorm / RMSNorm");
    {
        float xn[8]   = {1,2,3,4,5,6,7,8};     /* 2 rows of d=4 */
        float yn[8]   = {0};
        rac_status st = rac_layernorm(xn, yn, NULL, NULL, 1e-5f, 2, 4, &cfg);
        CHECK("layernorm returns OK", st == RAC_OK);

        /* Row 0: mean=(1+2+3+4)/4=2.5 → zero-mean after norm */
        float row0_mean = (yn[0]+yn[1]+yn[2]+yn[3]) / 4.0f;
        CHECK("layernorm row0 mean ≈ 0", fabsf(row0_mean) < 0.01f);

        /* Variance ≈ 1 */
        float row0_var = 0.0f;
        for (int i = 0; i < 4; i++) row0_var += (yn[i] - row0_mean)*(yn[i] - row0_mean);
        row0_var /= 4.0f;
        CHECK("layernorm row0 var ≈ 1", fabsf(row0_var - 1.0f) < 0.05f);

        float rm[4] = {3, 4, 0, 0}, rmo[4] = {0};
        st = rac_rmsnorm(rm, rmo, NULL, 1e-6f, 1, 4, &cfg);
        CHECK("rmsnorm returns OK", st == RAC_OK);
        /* ms=(9+16)/4=6.25 → rsqrt=0.4 → y=[1.2,1.6,0,0] */
        CHECK("rmsnorm y[0] ≈ 1.2", fabsf(rmo[0] - 1.2f) < 0.01f);
        CHECK("rmsnorm y[1] ≈ 1.6", fabsf(rmo[1] - 1.6f) < 0.01f);
    }

    /* ── BVT-14: RoPE ─────────────────────────────────────────────── */
    HEADER("BVT-14: Rotary Position Embeddings (RoPE)");
    {
        const int seq = 4, head_dim = 4;
        float cos_tab[4 * 2] = {0}, sin_tab[4 * 2] = {0};
        rac_status st = rac_rope_cache(cos_tab, sin_tab, seq, head_dim, 10000.0f);
        CHECK("rope_cache returns OK", st == RAC_OK);

        /* Position 0 has cos=1, sin=0 for every frequency pair. */
        CHECK("rope p=0 cos=1", fabsf(cos_tab[0] - 1.0f) < 0.001f && fabsf(cos_tab[1] - 1.0f) < 0.001f);
        CHECK("rope p=0 sin=0", fabsf(sin_tab[0]) < 0.001f && fabsf(sin_tab[1]) < 0.001f);

        /* Apply to a single batch/head/seq vector. Magnitude of each
         * pair must be preserved — RoPE is a rotation. */
        float x[16] = {
            1,2,3,4,               /* t=0 */
            5,6,7,8,               /* t=1 */
            -1,0,0,1,              /* t=2 */
            0.5f,-0.5f,1,-1,       /* t=3 */
        };
        float orig[16];
        memcpy(orig, x, sizeof(x));
        st = rac_rope_apply(x, cos_tab, sin_tab, 1, 1, seq, head_dim, &cfg);
        CHECK("rope_apply returns OK", st == RAC_OK);

        int pair_ok = 1;
        for (int t = 0; t < seq; t++) {
            for (int i = 0; i < head_dim / 2; i++) {
                float a0 = orig[t*head_dim + 2*i];
                float b0 = orig[t*head_dim + 2*i + 1];
                float a1 = x[t*head_dim + 2*i];
                float b1 = x[t*head_dim + 2*i + 1];
                float n0 = sqrtf(a0*a0 + b0*b0);
                float n1 = sqrtf(a1*a1 + b1*b1);
                if (fabsf(n0 - n1) > 0.01f) pair_ok = 0;
            }
        }
        CHECK("rope preserves each pair's magnitude", pair_ok);
    }

    /* ── BVT-15: Scaled dot-product attention ─────────────────────── */
    HEADER("BVT-15: Scaled dot-product attention");
    {
        const int B=1, H=1, T=3, D=4;
        float qv[B*H*T*D], kv[B*H*T*D], vv[B*H*T*D], ov[B*H*T*D];
        for (int i = 0; i < B*H*T*D; i++) {
            qv[i] = kv[i] = 1.0f;
            vv[i] = 1.0f;
        }
        rac_status st = rac_scaled_dot_attention(qv, kv, vv, NULL, 0,
                                                  ov, B, H, T, D, &cfg);
        CHECK("attention returns OK", st == RAC_OK);
        /* Uniform softmax, all-ones V → output is all-ones. */
        int ones_ok = 1;
        for (int i = 0; i < B*H*T*D; i++) if (fabsf(ov[i] - 1.0f) > 0.01f) ones_ok = 0;
        CHECK("attention uniform case outputs ones", ones_ok);
    }

    /* ── Summary ──────────────────────────────────────────────────── */
    HEADER("BVT Summary");
    printf("  Passed: %d\n  Failed: %d\n  Total:  %d\n", passed, failed, passed+failed);
    printf("\n  %s\n", failed == 0 ? "ALL BVT PASSED" : "BVT FAILURES DETECTED");
    return failed == 0 ? 0 : 1;
}
