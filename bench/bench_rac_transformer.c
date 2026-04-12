/*
 * bench_rac_transformer.c — RAC single-layer transformer inference bench
 * Pinnacle Quantum Group — April 2026
 *
 * A reproducible RAC transformer decoder-layer bench measuring prefill +
 * single-token decode latency. Runs on synthetic xavier weights by
 * default; with --safetensors=PATH it loads real weights from a HF
 * checkpoint so the three comparison frameworks see identical tensors.
 *
 * MODEL CONFIG (runtime-selectable):
 *   Default (tiny)        : d=512  heads=8  d_ff=1536  (compile-time fast path)
 *   --config tinyllama    : d=2048 heads=32 d_ff=5632  (TinyLlama-1.1B shape)
 *
 * SCENARIOS:
 *   PREFILL  — prefill_T tokens, compute-bound
 *   DECODE   — 1 token at a time, memory-bound
 *
 * USAGE:
 *   ./bench_rac_tx                          # tiny shape, synthetic weights
 *   ./bench_rac_tx --config tinyllama       # TinyLlama shape, synthetic weights
 *   ./bench_rac_tx --safetensors=PATH       # TinyLlama shape, real layer-0 weights
 *                                             from a HF-downloaded file
 *   ./bench_rac_tx --prefill 32 --decode 50 # override iteration counts
 *   ./bench_rac_tx --layer 5                # which layer to load with --safetensors
 *
 * Fetch weights first via bench/fetch_model.py:
 *   MODEL_DIR=$(python3 bench/fetch_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0)
 *   ./bench_rac_tx --safetensors "$MODEL_DIR/model.safetensors"
 */

#define _POSIX_C_SOURCE 200112L
#include "rac_cpu.h"
#include "safetensors_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Timing ────────────────────────────────────────────────────────────── */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Runtime config ────────────────────────────────────────────────────── */

typedef struct {
    int d_model;
    int n_heads;
    int n_kv_heads;      /* for GQA; equal to n_heads for MHA */
    int d_head;
    int d_ff;
    int prefill_T;
    int prefill_iters;
    int decode_iters;
    int layer_idx;       /* which layer to load from safetensors */
    const char *safetensors_path;
    const char *config_name;
} tx_cfg;

static void cfg_apply_preset(tx_cfg *c, const char *name) {
    if (!strcmp(name, "tiny")) {
        c->d_model = 512; c->n_heads = 8; c->n_kv_heads = 8;
        c->d_ff = 1536;
    } else if (!strcmp(name, "tinyllama")) {
        c->d_model = 2048; c->n_heads = 32; c->n_kv_heads = 4;  /* GQA */
        c->d_ff = 5632;
    } else {
        fprintf(stderr, "unknown preset '%s' (use 'tiny' or 'tinyllama')\n", name);
        exit(1);
    }
    c->d_head = c->d_model / c->n_heads;
    c->config_name = name;
}

/* ── Weights + buffers ─────────────────────────────────────────────────── */

typedef struct {
    float *W_q, *W_k, *W_v, *W_o;
    float *W_g, *W_u, *W_d;
    float *rms_att, *rms_ffn;
} tx_weights;

static float *xalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(float)) != 0) {
        fprintf(stderr, "alloc failed (%zu floats)\n", n); exit(1);
    }
    return (float *)p;
}

static uint32_t rng_state = 0xC0FFEEu;
static float rand_xavier(int d_model) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state <<  5;
    return (((float)rng_state / 2147483648.0f) - 1.0f) *
           (1.0f / sqrtf((float)d_model));
}

static void init_weights_synthetic(tx_weights *w, const tx_cfg *c) {
    size_t d = c->d_model;
    size_t dk = (size_t)c->n_kv_heads * c->d_head;  /* GQA smaller K,V */
    w->W_q = xalloc(d * d);
    w->W_k = xalloc(d * dk);
    w->W_v = xalloc(d * dk);
    w->W_o = xalloc(d * d);
    w->W_g = xalloc((size_t)c->d_ff * d);
    w->W_u = xalloc((size_t)c->d_ff * d);
    w->W_d = xalloc((size_t)d * c->d_ff);
    w->rms_att = xalloc(d);
    w->rms_ffn = xalloc(d);
    for (size_t i = 0; i < d*d;  i++) { w->W_q[i] = rand_xavier(c->d_model); w->W_o[i] = rand_xavier(c->d_model); }
    for (size_t i = 0; i < d*dk; i++) { w->W_k[i] = rand_xavier(c->d_model); w->W_v[i] = rand_xavier(c->d_model); }
    for (size_t i = 0; i < (size_t)c->d_ff * d; i++) { w->W_g[i] = rand_xavier(c->d_model); w->W_u[i] = rand_xavier(c->d_model); }
    for (size_t i = 0; i < (size_t)d * c->d_ff; i++) { w->W_d[i] = rand_xavier(c->d_model); }
    for (int i = 0; i < c->d_model; i++) { w->rms_att[i] = 1.0f; w->rms_ffn[i] = 1.0f; }
}

/* Load layer-L from a Llama-style safetensors checkpoint. Tensor names
 * follow the HF transformers convention:
 *   model.layers.L.self_attn.{q,k,v,o}_proj.weight
 *   model.layers.L.mlp.{gate,up,down}_proj.weight
 *   model.layers.L.{input_layernorm,post_attention_layernorm}.weight
 * Every weight is stored as [out_features, in_features] in HF. We keep
 * the same layout since rac_fused_linear expects weight[N, K] row-major. */
static int init_weights_safetensors(tx_weights *w, const tx_cfg *c,
                                    const char *path) {
    char err[256];
    st_file *f = st_open(path, err);
    if (!f) { fprintf(stderr, "safetensors open failed: %s\n", err); return -1; }

    /* Allocate based on runtime config (caller should have picked
     * a matching --config preset). */
    size_t d  = c->d_model;
    size_t dk = (size_t)c->n_kv_heads * c->d_head;
    w->W_q = xalloc(d * d);
    w->W_k = xalloc(d * dk);
    w->W_v = xalloc(d * dk);
    w->W_o = xalloc(d * d);
    w->W_g = xalloc((size_t)c->d_ff * d);
    w->W_u = xalloc((size_t)c->d_ff * d);
    w->W_d = xalloc((size_t)d * c->d_ff);
    w->rms_att = xalloc(d);
    w->rms_ffn = xalloc(d);

    char buf[ST_MAX_NAME];
    #define LOAD(name, ptr, expect_numel) do {                                \
        snprintf(buf, sizeof(buf), "model.layers.%d." name, c->layer_idx);   \
        const st_tensor *t = st_find(f, buf);                                \
        if (!t) { fprintf(stderr, "missing tensor: %s\n", buf); st_close(f); return -1; } \
        size_t nel = st_numel(t);                                            \
        if (nel != (size_t)(expect_numel)) {                                 \
            fprintf(stderr, "%s shape mismatch: got %zu expected %zu\n",     \
                    buf, nel, (size_t)(expect_numel));                       \
            st_close(f); return -1;                                          \
        }                                                                    \
        if (st_to_f32(f, t, ptr) != 0) {                                     \
            fprintf(stderr, "dtype not supported for %s (%s)\n", buf,        \
                    st_dtype_name(t->dtype));                                \
            st_close(f); return -1;                                          \
        }                                                                    \
    } while (0)

    LOAD("self_attn.q_proj.weight", w->W_q, d * d);
    LOAD("self_attn.k_proj.weight", w->W_k, d * dk);
    LOAD("self_attn.v_proj.weight", w->W_v, d * dk);
    LOAD("self_attn.o_proj.weight", w->W_o, d * d);
    LOAD("mlp.gate_proj.weight",    w->W_g, (size_t)c->d_ff * d);
    LOAD("mlp.up_proj.weight",      w->W_u, (size_t)c->d_ff * d);
    LOAD("mlp.down_proj.weight",    w->W_d, (size_t)d * c->d_ff);
    LOAD("input_layernorm.weight",          w->rms_att, d);
    LOAD("post_attention_layernorm.weight", w->rms_ffn, d);
    #undef LOAD

    printf("  loaded layer %d from %s (dtype auto-converted to F32)\n",
           c->layer_idx, path);
    st_close(f);
    return 0;
}

static void free_weights(tx_weights *w) {
    free(w->W_q); free(w->W_k); free(w->W_v); free(w->W_o);
    free(w->W_g); free(w->W_u); free(w->W_d);
    free(w->rms_att); free(w->rms_ffn);
}

/* ── Transformer primitives ────────────────────────────────────────────── */

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

/* Scaled dot-product attention for a single (possibly-replicated) head.
 * OpenMP-parallel over T for each of the three phases — dominates the
 * tinyllama prefill wall-clock at T=128, nh=32. */
static void attention_head(const float *q, const float *k, const float *v,
                           float *out, float *scratch,
                           int T, int d_head) {
    const float scale = 1.0f / sqrtf((float)d_head);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < T; i++) {
        float *row = scratch + (size_t)i * T;
        for (int j = 0; j <= i; j++) {
            float s = 0.0f;
            for (int h = 0; h < d_head; h++) s += q[i*d_head+h] * k[j*d_head+h];
            row[j] = s * scale;
        }
        for (int j = i+1; j < T; j++) row[j] = -1e30f;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < T; i++) rac_softmax(scratch + i*T, scratch + i*T, T);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < T; i++) {
        float *orow = out + (size_t)i * d_head;
        for (int h = 0; h < d_head; h++) orow[h] = 0.0f;
        for (int j = 0; j < T; j++) {
            float s = scratch[i*T+j];
            for (int h = 0; h < d_head; h++) orow[h] += s * v[j*d_head+h];
        }
    }
}

/* ── Forward pass ──────────────────────────────────────────────────────── */

typedef struct {
    float *norm, *Q, *K, *V, *att_out, *scratch;
    float *ffn_in, *ffn_g, *ffn_u;
    float *qc, *kc, *vc, *oc;  /* per-head contig buffers */
} tx_bufs;

static void bufs_alloc(tx_bufs *b, const tx_cfg *c, int T) {
    size_t d = c->d_model;
    size_t dk = (size_t)c->n_kv_heads * c->d_head;
    b->norm    = xalloc((size_t)T * d);
    b->Q       = xalloc((size_t)T * d);
    b->K       = xalloc((size_t)T * dk);
    b->V       = xalloc((size_t)T * dk);
    b->att_out = xalloc((size_t)T * d);
    b->scratch = xalloc((size_t)T * T);
    b->ffn_in  = xalloc((size_t)T * d);
    b->ffn_g   = xalloc((size_t)T * c->d_ff);
    b->ffn_u   = xalloc((size_t)T * c->d_ff);
    b->qc      = xalloc((size_t)T * c->d_head);
    b->kc      = xalloc((size_t)T * c->d_head);
    b->vc      = xalloc((size_t)T * c->d_head);
    b->oc      = xalloc((size_t)T * c->d_head);
}
static void bufs_free(tx_bufs *b) {
    free(b->norm); free(b->Q); free(b->K); free(b->V); free(b->att_out);
    free(b->scratch); free(b->ffn_in); free(b->ffn_g); free(b->ffn_u);
    free(b->qc); free(b->kc); free(b->vc); free(b->oc);
}

static void decoder_forward(float *x, const tx_weights *w, tx_bufs *b,
                            const tx_cfg *c, int T) {
    rac_config cfg = rac_default_config();
    int d = c->d_model, dh = c->d_head;
    int nq = c->n_heads, nkv = c->n_kv_heads;
    int gqa_repeat = nq / nkv;

    /* Attention block */
    rmsnorm(x, w->rms_att, b->norm, T, d);
    rac_fused_linear(b->norm, w->W_q, NULL, b->Q, T, d, d, RAC_ACT_NONE, &cfg);
    rac_fused_linear(b->norm, w->W_k, NULL, b->K, T, nkv*dh, d, RAC_ACT_NONE, &cfg);
    rac_fused_linear(b->norm, w->W_v, NULL, b->V, T, nkv*dh, d, RAC_ACT_NONE, &cfg);

    for (int h = 0; h < nq; h++) {
        int kv_h = h / gqa_repeat;              /* GQA: multiple q-heads share k/v */
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < dh; i++) {
                b->qc[t*dh + i] = b->Q[(size_t)t * d + h * dh + i];
                b->kc[t*dh + i] = b->K[(size_t)t * nkv * dh + kv_h * dh + i];
                b->vc[t*dh + i] = b->V[(size_t)t * nkv * dh + kv_h * dh + i];
            }
        }
        attention_head(b->qc, b->kc, b->vc, b->oc, b->scratch, T, dh);
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < dh; i++) {
                b->att_out[(size_t)t * d + h * dh + i] = b->oc[t*dh + i];
            }
        }
    }

    rac_fused_linear(b->att_out, w->W_o, NULL, b->norm, T, d, d, RAC_ACT_NONE, &cfg);
    for (size_t i = 0; i < (size_t)T * d; i++) x[i] += b->norm[i];

    /* FFN block (SwiGLU) */
    rmsnorm(x, w->rms_ffn, b->ffn_in, T, d);
    rac_fused_linear(b->ffn_in, w->W_g, NULL, b->ffn_g, T, c->d_ff, d, RAC_ACT_SILU, &cfg);
    rac_fused_linear(b->ffn_in, w->W_u, NULL, b->ffn_u, T, c->d_ff, d, RAC_ACT_NONE, &cfg);
    for (size_t i = 0; i < (size_t)T * c->d_ff; i++) b->ffn_g[i] *= b->ffn_u[i];
    rac_fused_linear(b->ffn_g, w->W_d, NULL, b->norm, T, d, c->d_ff, RAC_ACT_NONE, &cfg);
    for (size_t i = 0; i < (size_t)T * d; i++) x[i] += b->norm[i];
}

/* ── CLI ───────────────────────────────────────────────────────────────── */

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s [options]\n"
        "  --config {tiny|tinyllama}    model shape preset (default: tiny)\n"
        "  --safetensors PATH           load layer weights from HF checkpoint\n"
        "                                 (implies --config tinyllama)\n"
        "  --layer N                    which layer to load (default: 0)\n"
        "  --prefill N                  prefill T tokens (default: 128)\n"
        "  --prefill-iters N            # prefill passes (default: 30)\n"
        "  --decode-iters N             # decode passes (default: 100)\n"
        "  -h, --help                   this help\n"
        "\n"
        "Fetch weights via:\n"
        "  MODEL_DIR=$(python3 bench/fetch_model.py \\\n"
        "                --model TinyLlama/TinyLlama-1.1B-Chat-v1.0)\n"
        "  %s --safetensors \"$MODEL_DIR/model.safetensors\"\n",
        argv0, argv0);
}

int main(int argc, char **argv) {
    tx_cfg c = {0};
    c.prefill_T     = 128;
    c.prefill_iters = 30;
    c.decode_iters  = 100;
    c.layer_idx     = 0;
    cfg_apply_preset(&c, "tiny");

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i+1 < argc) cfg_apply_preset(&c, argv[++i]);
        else if (!strcmp(argv[i], "--safetensors") && i+1 < argc) {
            c.safetensors_path = argv[++i];
            cfg_apply_preset(&c, "tinyllama");
        }
        else if (!strcmp(argv[i], "--layer") && i+1 < argc)        c.layer_idx     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefill") && i+1 < argc)      c.prefill_T     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefill-iters") && i+1 < argc)c.prefill_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--decode-iters") && i+1 < argc) c.decode_iters  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    printf("RAC single-layer transformer inference bench\n");
    printf("  config=%s  d_model=%d  n_heads=%d  n_kv_heads=%d  d_head=%d  d_ff=%d\n"
           "  prefill_T=%d  prefill_iters=%d  decode_iters=%d\n",
           c.config_name, c.d_model, c.n_heads, c.n_kv_heads, c.d_head, c.d_ff,
           c.prefill_T, c.prefill_iters, c.decode_iters);
    if (c.safetensors_path) {
        printf("  weights: layer %d from %s\n", c.layer_idx, c.safetensors_path);
    } else {
        printf("  weights: synthetic xavier (deterministic, seed=0xC0FFEE)\n");
    }

    tx_weights w = {0};
    if (c.safetensors_path) {
        if (init_weights_safetensors(&w, &c, c.safetensors_path) != 0) return 2;
    } else {
        init_weights_synthetic(&w, &c);
    }

    int T = c.prefill_T;
    tx_bufs b; bufs_alloc(&b, &c, T);
    float *x = xalloc((size_t)T * c.d_model);
    for (size_t i = 0; i < (size_t)T * c.d_model; i++) x[i] = rand_xavier(c.d_model);

    decoder_forward(x, &w, &b, &c, T);   /* warmup */

    double t0 = now_sec();
    for (int it = 0; it < c.prefill_iters; it++)
        decoder_forward(x, &w, &b, &c, T);
    double s_pre = now_sec() - t0;
    double ms_pre  = s_pre * 1000.0 / c.prefill_iters;
    double tps_pre = (T * c.prefill_iters) / s_pre;

    /* Decode scenario: rebuild buffers for T=1 */
    bufs_free(&b); free(x);
    int T1 = 1;
    bufs_alloc(&b, &c, T1);
    x = xalloc((size_t)T1 * c.d_model);
    for (size_t i = 0; i < (size_t)T1 * c.d_model; i++) x[i] = rand_xavier(c.d_model);
    decoder_forward(x, &w, &b, &c, T1);  /* warmup */

    t0 = now_sec();
    for (int it = 0; it < c.decode_iters; it++)
        decoder_forward(x, &w, &b, &c, T1);
    double s_dec = now_sec() - t0;
    double ms_dec = s_dec * 1000.0 / c.decode_iters;
    double tps_dec = c.decode_iters / s_dec;

    double flops_per_tok =
          2.0 * c.d_model * c.d_model                    /* Q */
        + 2.0 * c.d_model * (c.n_kv_heads * c.d_head)    /* K */
        + 2.0 * c.d_model * (c.n_kv_heads * c.d_head)    /* V */
        + 2.0 * c.d_model * c.d_model                    /* O */
        + 4.0 * T * c.d_model                             /* attn scores+sum */
        + 3.0 * 2.0 * c.d_model * c.d_ff;                 /* FFN gate+up+down */
    double gflops_pre = flops_per_tok * T  * c.prefill_iters / s_pre / 1e9;
    double gflops_dec = flops_per_tok * T1 * c.decode_iters  / s_dec / 1e9;

    printf("\n── RAC results ─────────────────────────────────────\n");
    printf("  prefill T=%d:   %7.2f ms/layer   %8.1f tok/s   %7.1f GFLOPS\n",
           T,  ms_pre, tps_pre, gflops_pre);
    printf("  decode  T=1:    %7.2f ms/layer   %8.1f tok/s   %7.1f GFLOPS\n",
           ms_dec, tps_dec, gflops_dec);

    printf("\n── Framework comparison hooks ──────────────────────\n");
    printf("  llama.cpp:  see bench/configs/llama_cpp.yaml  (fill in paths)\n");
    printf("  tinygrad:   see bench/configs/tinygrad.yaml   (fill in paths)\n");
    printf("  Fetch HF weights:  python3 bench/fetch_model.py --model REPO_ID\n");

    free_weights(&w);
    bufs_free(&b);
    free(x);
    return 0;
}
