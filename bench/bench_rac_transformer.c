/*
 * bench_rac_transformer.c — RAC single-layer transformer inference bench
 * Pinnacle Quantum Group — April 2026
 *
 * GOAL: a minimal, reproducible RAC transformer decoder-layer bench that
 * measures prefill + single-token decode latency on synthetic weights.
 * Numbers are directly comparable to llama.cpp / tinygrad running the
 * same model shape on the same hardware — see bench_harness.sh for the
 * side-by-side runner.
 *
 * MODEL: single decoder layer, TinyLlama-shaped:
 *   d_model   = 512          (hidden size)
 *   n_heads   = 8            (attention heads)
 *   d_head    = 64           (d_model / n_heads)
 *   d_ff      = 1536         (FFN intermediate, ~3x d_model)
 *   vocab     = unused at this scale — we measure compute, not lookup
 *
 * Two decode scenarios:
 *   1. PREFILL  — 128-token context forward pass (compute-bound, matmul)
 *   2. DECODE   — 1 token at a time, KV-cache warm (memory-bound)
 *
 * Weights are deterministic random (xavier-ish) so run-to-run results
 * are stable across frameworks on the same hardware.
 *
 * BUILD:
 *   cc -O3 -march=native -mavx2 -mfma -fopenmp \
 *      bench_rac_transformer.c -L../lib -lrac -lm -o bench_rac_tx
 *
 * RUN:
 *   ./bench_rac_tx              # default iterations
 *   ./bench_rac_tx 200 50       # prefill_iters decode_iters
 */

#define _POSIX_C_SOURCE 200112L
#include "rac_cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Config (override at build time or edit here) ──────────────────────── */

#ifndef TX_D_MODEL
#define TX_D_MODEL  512
#endif
#ifndef TX_N_HEADS
#define TX_N_HEADS  8
#endif
#ifndef TX_D_FF
#define TX_D_FF     1536
#endif
#ifndef TX_PREFILL_TOKENS
#define TX_PREFILL_TOKENS 128
#endif

#define TX_D_HEAD    (TX_D_MODEL / TX_N_HEADS)

/* ── Timing helpers ────────────────────────────────────────────────────── */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Lightweight xorshift RNG so the bench is deterministic. */
static uint32_t rng_state = 0xC0FFEEu;
static float rand_xavier(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state <<  5;
    /* Uniform[-1,1] / sqrt(d_model) — enough for a valid matmul benchmark. */
    return (((float)rng_state / 2147483648.0f) - 1.0f) *
           (1.0f / sqrtf((float)TX_D_MODEL));
}

/* ── Weights ───────────────────────────────────────────────────────────── */

typedef struct {
    /* Attention */
    float *W_q;   /* [d, d]       */
    float *W_k;   /* [d, d]       */
    float *W_v;   /* [d, d]       */
    float *W_o;   /* [d, d]       */
    /* FFN (gated, SwiGLU-style) */
    float *W_g;   /* [d_ff, d]    */
    float *W_u;   /* [d_ff, d]    */
    float *W_d;   /* [d, d_ff]    */
    /* RMSNorm scales */
    float *rms_att;  /* [d] */
    float *rms_ffn;  /* [d] */
} tx_weights;

static float *xalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(float)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return (float *)p;
}

static void init_weights(tx_weights *w) {
    size_t dd = (size_t)TX_D_MODEL * TX_D_MODEL;
    size_t df = (size_t)TX_D_FF    * TX_D_MODEL;
    w->W_q = xalloc(dd);
    w->W_k = xalloc(dd);
    w->W_v = xalloc(dd);
    w->W_o = xalloc(dd);
    w->W_g = xalloc(df);
    w->W_u = xalloc(df);
    w->W_d = xalloc(df);
    w->rms_att = xalloc(TX_D_MODEL);
    w->rms_ffn = xalloc(TX_D_MODEL);
    for (size_t i = 0; i < dd; i++) { w->W_q[i] = rand_xavier(); w->W_k[i] = rand_xavier(); w->W_v[i] = rand_xavier(); w->W_o[i] = rand_xavier(); }
    for (size_t i = 0; i < df; i++) { w->W_g[i] = rand_xavier(); w->W_u[i] = rand_xavier(); w->W_d[i] = rand_xavier(); }
    for (int i = 0; i < TX_D_MODEL; i++) { w->rms_att[i] = 1.0f; w->rms_ffn[i] = 1.0f; }
}

static void free_weights(tx_weights *w) {
    free(w->W_q); free(w->W_k); free(w->W_v); free(w->W_o);
    free(w->W_g); free(w->W_u); free(w->W_d);
    free(w->rms_att); free(w->rms_ffn);
}

/* ── Transformer primitives (inline — not yet in rac_cpu lib) ──────────── */

/* RMSNorm: x' = x * scale[i] / sqrt(mean(x²) + eps). Vectorizable. */
static void rmsnorm(const float *x, const float *scale, float *out,
                    int T, int d) {
    const float eps = 1e-5f;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float *row = x + (size_t)t * d;
        float ss = 0.0f;
        for (int i = 0; i < d; i++) ss += row[i] * row[i];
        float inv = 1.0f / sqrtf(ss / d + eps);
        float *orow = out + (size_t)t * d;
        for (int i = 0; i < d; i++) orow[i] = row[i] * scale[i] * inv;
    }
}

/* Scaled dot-product attention, causal, T × d_head.
 * q, k, v: [T, d_head]. out: [T, d_head]. Temp scratch: [T, T] scores. */
static void attention_head(const float *q, const float *k, const float *v,
                           float *out, float *scores_scratch,
                           int T, int d_head) {
    const float scale = 1.0f / sqrtf((float)d_head);

    /* scores[i,j] = q[i]·k[j] * scale, with causal mask j > i */
    for (int i = 0; i < T; i++) {
        for (int j = 0; j <= i; j++) {
            float s = 0.0f;
            for (int h = 0; h < d_head; h++) s += q[i*d_head + h] * k[j*d_head + h];
            scores_scratch[i*T + j] = s * scale;
        }
        for (int j = i + 1; j < T; j++) scores_scratch[i*T + j] = -1e30f;
    }

    /* Row-wise softmax */
    for (int i = 0; i < T; i++) {
        rac_softmax(scores_scratch + i*T, scores_scratch + i*T, T);
    }

    /* out[i] = sum_j scores[i,j] * v[j] */
    for (int i = 0; i < T; i++) {
        for (int h = 0; h < d_head; h++) out[i*d_head + h] = 0.0f;
        for (int j = 0; j < T; j++) {
            float s = scores_scratch[i*T + j];
            for (int h = 0; h < d_head; h++) out[i*d_head + h] += s * v[j*d_head + h];
        }
    }
}

/* ── Decoder layer forward pass ────────────────────────────────────────── */

/* x: [T, d_model] — residual stream. Overwritten in place with layer output. */
static void decoder_forward(float *x, const tx_weights *w,
                            float *norm_buf, float *qkv_buf,
                            float *att_out, float *scores_scratch,
                            float *ffn_in, float *ffn_gate, float *ffn_up,
                            int T) {
    rac_config cfg = rac_default_config();

    /* ── Attention block ── */
    rmsnorm(x, w->rms_att, norm_buf, T, TX_D_MODEL);

    float *Q = qkv_buf;
    float *K = qkv_buf +     (size_t)T * TX_D_MODEL;
    float *V = qkv_buf + 2 * (size_t)T * TX_D_MODEL;
    rac_fused_linear(norm_buf, w->W_q, NULL, Q,
                     T, TX_D_MODEL, TX_D_MODEL, RAC_ACT_NONE, &cfg);
    rac_fused_linear(norm_buf, w->W_k, NULL, K,
                     T, TX_D_MODEL, TX_D_MODEL, RAC_ACT_NONE, &cfg);
    rac_fused_linear(norm_buf, w->W_v, NULL, V,
                     T, TX_D_MODEL, TX_D_MODEL, RAC_ACT_NONE, &cfg);

    /* Per-head attention */
    for (int h = 0; h < TX_N_HEADS; h++) {
        /* gather head slice (stride-based view into Q/K/V) */
        float *qh = Q + (size_t)h * TX_D_HEAD;
        float *kh = K + (size_t)h * TX_D_HEAD;
        float *vh = V + (size_t)h * TX_D_HEAD;
        /* Repack contiguous for this head */
        static float qc[8192], kc[8192], vc[8192], oc[8192];  /* T≤128, d_head≤64 */
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < TX_D_HEAD; i++) {
                qc[t*TX_D_HEAD + i] = qh[(size_t)t * TX_D_MODEL + i];
                kc[t*TX_D_HEAD + i] = kh[(size_t)t * TX_D_MODEL + i];
                vc[t*TX_D_HEAD + i] = vh[(size_t)t * TX_D_MODEL + i];
            }
        }
        attention_head(qc, kc, vc, oc, scores_scratch, T, TX_D_HEAD);
        /* Scatter back into att_out at head offset */
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < TX_D_HEAD; i++) {
                att_out[(size_t)t * TX_D_MODEL + h * TX_D_HEAD + i] =
                    oc[t*TX_D_HEAD + i];
            }
        }
    }

    /* Output projection + residual */
    rac_fused_linear(att_out, w->W_o, NULL, norm_buf,
                     T, TX_D_MODEL, TX_D_MODEL, RAC_ACT_NONE, &cfg);
    for (size_t i = 0; i < (size_t)T * TX_D_MODEL; i++) x[i] += norm_buf[i];

    /* ── FFN block (SwiGLU) ── */
    rmsnorm(x, w->rms_ffn, ffn_in, T, TX_D_MODEL);
    rac_fused_linear(ffn_in, w->W_g, NULL, ffn_gate,
                     T, TX_D_FF, TX_D_MODEL, RAC_ACT_SILU, &cfg);
    rac_fused_linear(ffn_in, w->W_u, NULL, ffn_up,
                     T, TX_D_FF, TX_D_MODEL, RAC_ACT_NONE, &cfg);
    for (size_t i = 0; i < (size_t)T * TX_D_FF; i++) ffn_gate[i] *= ffn_up[i];
    rac_fused_linear(ffn_gate, w->W_d, NULL, norm_buf,
                     T, TX_D_MODEL, TX_D_FF, RAC_ACT_NONE, &cfg);
    for (size_t i = 0; i < (size_t)T * TX_D_MODEL; i++) x[i] += norm_buf[i];
}

/* ── Bench ─────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int prefill_iters = (argc > 1) ? atoi(argv[1]) : 30;
    int decode_iters  = (argc > 2) ? atoi(argv[2]) : 100;
    int T = TX_PREFILL_TOKENS;

    printf("RAC single-layer transformer inference bench\n");
    printf("  d_model=%d  n_heads=%d  d_head=%d  d_ff=%d  prefill_T=%d\n",
           TX_D_MODEL, TX_N_HEADS, TX_D_HEAD, TX_D_FF, T);

    tx_weights w; init_weights(&w);

    /* Working buffers */
    float *x       = xalloc((size_t)T * TX_D_MODEL);
    float *norm    = xalloc((size_t)T * TX_D_MODEL);
    float *qkv     = xalloc(3 * (size_t)T * TX_D_MODEL);
    float *att_out = xalloc((size_t)T * TX_D_MODEL);
    float *scores  = xalloc((size_t)T * T);
    float *ffn_in  = xalloc((size_t)T * TX_D_MODEL);
    float *ffn_g   = xalloc((size_t)T * TX_D_FF);
    float *ffn_u   = xalloc((size_t)T * TX_D_FF);

    for (size_t i = 0; i < (size_t)T * TX_D_MODEL; i++) x[i] = rand_xavier();

    /* Warm up (JIT icache, OMP team spin-up, caches) */
    decoder_forward(x, &w, norm, qkv, att_out, scores, ffn_in, ffn_g, ffn_u, T);

    /* ── Scenario A: prefill-128 ── */
    double t0 = now_sec();
    for (int it = 0; it < prefill_iters; it++) {
        decoder_forward(x, &w, norm, qkv, att_out, scores,
                        ffn_in, ffn_g, ffn_u, T);
    }
    double s_pre = now_sec() - t0;
    double ms_pre = s_pre * 1000.0 / prefill_iters;
    double tps_pre = (T * prefill_iters) / s_pre;

    /* ── Scenario B: single-token decode (T=1) ── */
    /* (We're measuring compute only — no real KV cache here; the decode
     *  bench is representative of memory-bound matmul at T=1.) */
    int T1 = 1;
    for (size_t i = 0; i < (size_t)T1 * TX_D_MODEL; i++) x[i] = rand_xavier();
    decoder_forward(x, &w, norm, qkv, att_out, scores, ffn_in, ffn_g, ffn_u, T1);

    t0 = now_sec();
    for (int it = 0; it < decode_iters; it++) {
        decoder_forward(x, &w, norm, qkv, att_out, scores,
                        ffn_in, ffn_g, ffn_u, T1);
    }
    double s_dec = now_sec() - t0;
    double ms_dec = s_dec * 1000.0 / decode_iters;
    double tps_dec = decode_iters / s_dec;

    /* ── FLOPs accounting (per token) ──
     *   attn: 4 · d² (QKVO projections) + 2·T·d (scores) + 2·T·d (v·s)
     *   ffn:  3 · d·d_ff (gate, up, down) + activation
     */
    double flops_per_tok_prefill =
          4.0 * (double)TX_D_MODEL * TX_D_MODEL          /* Q,K,V,O */
        + 4.0 * (double)T * TX_D_MODEL                    /* scores + v·s */
        + 3.0 * (double)TX_D_MODEL * TX_D_FF;             /* FFN */
    double gflops_prefill = flops_per_tok_prefill * 2.0 * T * prefill_iters
                            / s_pre / 1e9;
    double gflops_decode  = flops_per_tok_prefill * 2.0 * T1 * decode_iters
                            / s_dec / 1e9;

    printf("\n── RAC results ─────────────────────────────────────\n");
    printf("  prefill T=%d:   %7.2f ms/layer   %8.1f tok/s   %7.1f GFLOPS\n",
           T,  ms_pre, tps_pre, gflops_prefill);
    printf("  decode  T=1:    %7.2f ms/layer   %8.1f tok/s   %7.1f GFLOPS\n",
           ms_dec, tps_dec, gflops_decode);

    printf("\n── Framework comparison hooks ──────────────────────\n");
    printf("  llama.cpp:  see bench/configs/llama_cpp.yaml  (fill in, then\n");
    printf("              ./bench/run_llama_cpp.sh to emit comparable numbers)\n");
    printf("  tinygrad:   see bench/configs/tinygrad.yaml   (fill in, then\n");
    printf("              python bench/run_tinygrad.py)\n");
    printf("  Aggregate:  ./bench/bench_harness.sh  (runs all three, diffs)\n");

    free_weights(&w);
    free(x); free(norm); free(qkv); free(att_out); free(scores);
    free(ffn_in); free(ffn_g); free(ffn_u);
    return 0;
}
