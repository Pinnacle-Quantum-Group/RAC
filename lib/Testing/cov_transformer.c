/*
 * cov_transformer.c — coverage driver for RAC transformer primitives.
 * Pinnacle Quantum Group — April 2026
 *
 * Exercises every new primitive so that gcov/lcov reports full line
 * coverage of rac_cpu.c's transformer surface (layernorm, rmsnorm,
 * rope_cache, rope_apply, scaled_dot_attention, rsqrt, sigmoid,
 * sincos, rotate_n, project_n, polar_n, exp_n, tanh_n).
 *
 * Build (from lib/coverage.sh):
 *   gcc -O1 -g --coverage -I lib/c lib/Testing/cov_transformer.c \
 *       lib/c/rac_cpu.c -fopenmp -lm -o cov_transformer
 */

#include "../c/rac_cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int hits = 0;
#define HIT(n) do { hits++; printf("  [COV] %s\n", n); } while (0)

int main(void) {
    printf("RAC coverage driver — transformer surface\n");

    /* ── Core + tunable rotate/project/polar ─────────────────────── */
    rac_config cfg = rac_default_config();
    rac_vec2 v = {1.0f, 0.5f};
    for (int it = 4; it <= 24; it += 2) {
        (void)rac_rotate_n(v, 0.7f, it);
        (void)rac_project_n(v, 0.7f, it);
        float m, a;
        rac_polar_n(v, &m, &a, it);
        (void)rac_exp_n(0.5f, it);
        (void)rac_tanh_n(0.5f, it);
    }
    HIT("tunable-precision sweep");

    /* Clamp paths (too low + too high) */
    (void)rac_rotate_n(v, 0.1f, 0);
    (void)rac_rotate_n(v, 0.1f, 100);
    HIT("iter-clamp paths");

    /* ── sincos / rsqrt / sigmoid ────────────────────────────────── */
    float s, c;
    rac_sincos(0.5f, &s, &c);
    (void)rac_rsqrt(4.0f);
    (void)rac_rsqrt(0.0f);     /* guarded branch */
    (void)rac_rsqrt(-1.0f);    /* guarded branch */
    (void)rac_sigmoid(0.0f);
    (void)rac_sigmoid(5.0f);
    (void)rac_sigmoid(-5.0f);
    HIT("sincos / rsqrt / sigmoid corner cases");

    /* ── LayerNorm + RMSNorm (with and without gamma/beta) ──────── */
    const int R = 8, D = 32;
    float *xr = malloc(R*D*sizeof(float));
    float *yr = malloc(R*D*sizeof(float));
    float *gm = malloc(D*sizeof(float));
    float *bt = malloc(D*sizeof(float));
    for (int i = 0; i < R*D; i++) xr[i] = (float)((i*7) % 17) * 0.1f - 0.7f;
    for (int i = 0; i < D;   i++) { gm[i] = 1.1f; bt[i] = 0.01f; }

    rac_layernorm(xr, yr, NULL, NULL, 1e-5f, R, D, &cfg);
    rac_layernorm(xr, yr, gm,   bt,   1e-5f, R, D, &cfg);
    rac_rmsnorm  (xr, yr, NULL,        1e-6f, R, D, &cfg);
    rac_rmsnorm  (xr, yr, gm,          1e-6f, R, D, &cfg);

    /* Error paths */
    if (rac_layernorm(NULL, yr, NULL, NULL, 1e-5f, R, D, &cfg) != RAC_ERR_NULL_PTR) return 1;
    if (rac_layernorm(xr,   yr, NULL, NULL, 1e-5f, 0, D, &cfg) != RAC_ERR_INVALID_DIM) return 1;
    if (rac_rmsnorm  (NULL, yr, NULL,       1e-6f, R, D, &cfg) != RAC_ERR_NULL_PTR) return 1;
    if (rac_rmsnorm  (xr,   yr, NULL,       1e-6f, R, 0, &cfg) != RAC_ERR_INVALID_DIM) return 1;
    HIT("layernorm / rmsnorm main + error paths");

    /* ── RoPE cache + apply ─────────────────────────────────────── */
    const int SEQ = 8, HD = 8;
    int half = HD / 2;
    float *cs = malloc(SEQ*half*sizeof(float));
    float *sn = malloc(SEQ*half*sizeof(float));
    rac_rope_cache(cs, sn, SEQ, HD, 10000.0f);
    if (rac_rope_cache(NULL, sn, SEQ, HD, 10000.0f) != RAC_ERR_NULL_PTR) return 2;
    if (rac_rope_cache(cs, sn, SEQ, 3, 10000.0f)   != RAC_ERR_INVALID_DIM) return 2;
    HIT("rope_cache main + error paths");

    int total = 2*2*SEQ*HD;
    float *qx = malloc(total*sizeof(float));
    for (int i = 0; i < total; i++) qx[i] = 0.1f * ((i % 7) - 3);
    rac_rope_apply(qx, cs, sn, 2, 2, SEQ, HD, &cfg);
    if (rac_rope_apply(NULL, cs, sn, 1, 1, SEQ, HD, &cfg) != RAC_ERR_NULL_PTR) return 3;
    if (rac_rope_apply(qx, cs, sn, 1, 1, SEQ, 3, &cfg) != RAC_ERR_INVALID_DIM) return 3;
    HIT("rope_apply main + error paths");

    /* ── Scaled dot-product attention ───────────────────────────── */
    const int B=2, H=2, T=4, HD2=8;
    int tot = B*H*T*HD2;
    float *q = malloc(tot*sizeof(float));
    float *k = malloc(tot*sizeof(float));
    float *v2 = malloc(tot*sizeof(float));
    float *o = malloc(tot*sizeof(float));
    for (int i = 0; i < tot; i++) { q[i] = 0.2f; k[i] = 0.1f; v2[i] = 0.3f; }
    float mask[T*T];
    for (int i = 0; i < T*T; i++) mask[i] = 0.0f;

    rac_scaled_dot_attention(q, k, v2, NULL, 0, o, B, H, T, HD2, &cfg);
    rac_scaled_dot_attention(q, k, v2, mask, 1, o, B, H, T, HD2, &cfg);
    if (rac_scaled_dot_attention(NULL, k, v2, NULL, 0, o, B, H, T, HD2, &cfg) != RAC_ERR_NULL_PTR) return 4;
    if (rac_scaled_dot_attention(q, k, v2, NULL, 0, o, 0, H, T, HD2, &cfg) != RAC_ERR_INVALID_DIM) return 4;
    HIT("attention main + error paths");

    free(xr); free(yr); free(gm); free(bt);
    free(cs); free(sn); free(qx);
    free(q); free(k); free(v2); free(o);

    /* ── Original RAC primitive surface ─────────────────────────── */
    /* Exercise every non-transformer primitive so rac_cpu.c sees
     * full-line coverage on a fresh build. */
    rac_vec2 raw = rac_rotate_raw((rac_vec2){1.0f, 0.0f}, 0.3f);
    (void)rac_compensate(raw, 2);
    (void)rac_norm((rac_vec2){3.0f, 4.0f});
    (void)rac_normalize((rac_vec2){3.0f, 4.0f});
    (void)rac_coherence((rac_vec2){1.0f, 0.0f}, (rac_vec2){0.5f, 0.5f});
    (void)rac_dot((rac_vec2){1.0f, 0.5f}, (rac_vec2){-0.5f, 0.25f});
    (void)rac_complex_mul((rac_vec2){1.0f, 2.0f}, (rac_vec2){0.5f, 0.25f});
    /* Hit the x<0 pre-rotation branches (y>=0 and y<0) */
    (void)rac_norm((rac_vec2){-1.0f,  1.0f});
    (void)rac_norm((rac_vec2){-1.0f, -1.0f});
    (void)rac_polar_n((rac_vec2){-1.0f, -0.5f}, &(float){0}, &(float){0}, 16);

    /* DCT on a small signal */
    float dct_in[8]  = {1,2,3,4,5,6,7,8};
    float dct_out[8] = {0};
    rac_dct(dct_in, dct_out, 8);
    HIT("rotate_raw / compensate / norm / normalize / dot / coherence / complex_mul / dct");

    /* Batch rotation */
    rac_vec2 batch_in[8], batch_out[8];
    float thetas[8];
    for (int i = 0; i < 8; i++) {
        batch_in[i] = (rac_vec2){(float)i, (float)(i + 1)};
        thetas[i] = i * 0.1f;
    }
    rac_rotate_batch(batch_in, thetas, batch_out, 8);
    /* inner / outer */
    (void)rac_inner(batch_in, batch_in, 8);
    float outer[64];
    rac_outer(batch_in, batch_in, outer, 8, 8);
    HIT("rotate_batch / inner / outer");

    /* Softmax + exp + tanh */
    float sm_in[4] = {1, 2, 3, 4}, sm_out[4];
    rac_softmax(sm_in, sm_out, 4);
    rac_softmax_batch(sm_in, sm_out, 1, 4);
    (void)rac_exp(0.5f);
    (void)rac_tanh(0.5f);
    HIT("softmax / softmax_batch / exp / tanh");

    /* SGEMM alpha/beta paths + error handling */
    rac_config cfg2 = rac_default_config();
    float mA[4] = {1,2,3,4}, mB[4] = {5,6,7,8}, mC[4] = {1,1,1,1};
    rac_sgemm(mA, mB, mC, 2, 2, 2, 2.0f, 0.5f, &cfg2);   /* beta != 0, 1 */
    rac_sgemm(mA, mB, mC, 2, 2, 2, 1.0f, 0.0f, &cfg2);   /* beta == 0 */
    rac_sgemm(mA, mB, mC, 2, 2, 2, 1.0f, 1.0f, &cfg2);   /* beta == 1 */
    if (rac_sgemm(NULL, mB, mC, 2, 2, 2, 1.0f, 0.0f, &cfg2) != RAC_ERR_NULL_PTR) return 5;
    if (rac_sgemm(mA, mB, mC, 0, 2, 2, 1.0f, 0.0f, &cfg2) != RAC_ERR_INVALID_DIM) return 5;

    /* Fused linear: every activation + NULL bias + error paths */
    float fin[4]  = {0.5f, -0.5f, 1.0f, -1.0f};
    float fwt[8]  = {1, 0, 0, 0, 0, 1, 0, 0};
    float fbias[2] = {0.1f, -0.1f};
    float fout[2];
    rac_fused_linear(fin, fwt, fbias, fout, 1, 2, 4, RAC_ACT_NONE, &cfg2);
    rac_fused_linear(fin, fwt, fbias, fout, 1, 2, 4, RAC_ACT_RELU, &cfg2);
    rac_fused_linear(fin, fwt, fbias, fout, 1, 2, 4, RAC_ACT_GELU, &cfg2);
    rac_fused_linear(fin, fwt, fbias, fout, 1, 2, 4, RAC_ACT_SILU, &cfg2);
    rac_fused_linear(fin, fwt, NULL,   fout, 1, 2, 4, RAC_ACT_NONE, &cfg2);  /* NULL bias */
    if (rac_fused_linear(NULL, fwt, NULL, fout, 1, 2, 4, RAC_ACT_NONE, &cfg2) != RAC_ERR_NULL_PTR) return 6;
    if (rac_fused_linear(fin,  fwt, NULL, fout, 0, 2, 4, RAC_ACT_NONE, &cfg2) != RAC_ERR_INVALID_DIM) return 6;

    /* Batch activations */
    float ax[8] = {-2,-1,0,1,2,3,-3,4}, ay[8];
    rac_relu(ax, ay, 8);
    rac_gelu(ax, ay, 8);
    rac_silu(ax, ay, 8);
    HIT("sgemm alpha/beta / fused_linear / batch activations");

    /* Non-default config path */
    rac_config cfg3 = {.num_threads = 2, .tile_size = 32, .cordic_iters = 12};
    rac_matmul(mA, mB, mC, 2, 2, 2, &cfg3);
    /* Fall-through paths: NULL cfg, zero tile size (_get_tile defaults to 64). */
    rac_matmul(mA, mB, mC, 2, 2, 2, NULL);
    rac_config cfg4 = {.num_threads = 0, .tile_size = 0, .cordic_iters = 0};
    rac_matmul(mA, mB, mC, 2, 2, 2, &cfg4);
    rac_fused_linear(fin, fwt, fbias, fout, 1, 2, 4, RAC_ACT_NONE, NULL);
    HIT("non-default config + NULL cfg fallbacks");

    printf("\n  total coverage hits: %d\n", hits);
    return 0;
}
