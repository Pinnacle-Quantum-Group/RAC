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
#include "rac_hal.h"          /* HAL auto-dispatches to rac_sgemm_avx2 / */
                               /* rac_fused_linear_avx2 / asm micro-kernels */
#include "rac_q8_0.h"         /* Q8_0 block format + AVX2 GEMV */
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
    int n_layers;        /* number of decoder layers in --full-model mode */
    int prefill_T;
    int prefill_iters;
    int decode_iters;
    int layer_idx;       /* which layer to load from safetensors (single-layer mode) */
    int full_model;      /* 0 = single-layer bench, 1 = whole stack + KV cache */
    int quant_q8_0;      /* if set, quantize weights to Q8_0 after load (decode path) */
    const char *safetensors_path;
    const char *config_name;
} tx_cfg;

static void cfg_apply_preset(tx_cfg *c, const char *name) {
    if (!strcmp(name, "tiny")) {
        c->d_model = 512; c->n_heads = 8; c->n_kv_heads = 8;
        c->d_ff = 1536; c->n_layers = 4;
    } else if (!strcmp(name, "tinyllama")) {
        c->d_model = 2048; c->n_heads = 32; c->n_kv_heads = 4;  /* GQA */
        c->d_ff = 5632; c->n_layers = 22;
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

    /* Optional Q8_0 shadow copies of the linear-layer weights. NULL when
     * --q8_0 is not set. Each pointer is a flat array of rac_q8_0_block
     * structs in the same [N, K/32] row-major layout as the f32 matrix.
     * Used only on the M==1 GEMV path (decode). Prefill still uses f32
     * weights through the AVX2 micro-kernel. */
    rac_q8_0_block *Wq_q8, *Wk_q8, *Wv_q8, *Wo_q8;
    rac_q8_0_block *Wg_q8, *Wu_q8, *Wd_q8;
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
    free(w->Wq_q8); free(w->Wk_q8); free(w->Wv_q8); free(w->Wo_q8);
    free(w->Wg_q8); free(w->Wu_q8); free(w->Wd_q8);
}

/*
 * Produce Q8_0 shadow copies of each linear-layer weight matrix. Called
 * once per layer after f32 load when --q8_0 is requested. Memory cost
 * is ~0.25x the f32 arrays.
 */
static rac_q8_0_block *q8_alloc_and_quantize(const float *src, int N, int K) {
    size_t blocks = rac_q8_0_blocks((size_t)N * K);
    rac_q8_0_block *dst = NULL;
    /* posix_memalign for 64-byte alignment — matches xalloc() elsewhere.
     * The AVX2 kernel uses _mm_loadu / _mm256_loadu (unaligned) but
     * aligned data is still faster on Zen / Skylake. */
    if (posix_memalign((void **)&dst, 64, blocks * sizeof(rac_q8_0_block)) != 0 || !dst) {
        fprintf(stderr, "q8_0 alloc failed (%zu blocks)\n", blocks);
        exit(1);
    }
    rac_q8_0_quantize_matrix(src, dst, N, K);
    return dst;
}

static void weights_quantize_q8_0(tx_weights *w, const tx_cfg *c) {
    int d  = c->d_model;
    int dk = c->n_kv_heads * c->d_head;
    w->Wq_q8 = q8_alloc_and_quantize(w->W_q, d,          d);
    w->Wk_q8 = q8_alloc_and_quantize(w->W_k, dk,         d);
    w->Wv_q8 = q8_alloc_and_quantize(w->W_v, dk,         d);
    w->Wo_q8 = q8_alloc_and_quantize(w->W_o, d,          d);
    w->Wg_q8 = q8_alloc_and_quantize(w->W_g, c->d_ff,    d);
    w->Wu_q8 = q8_alloc_and_quantize(w->W_u, c->d_ff,    d);
    w->Wd_q8 = q8_alloc_and_quantize(w->W_d, d,          c->d_ff);
}

/*
 * Bench-side wrapper picking the right fused_linear variant per shape.
 *
 *   M >= 2 (prefill):  rac_hal_fused_linear -> rac_fused_linear_avx2
 *     uses the tuned Zen3/AVX512 micro-kernel. Parallelizes over M via
 *     MR=8 tiles, transpose cost amortized across the tile rows.
 *
 *   M == 1 (decode GEMV): do NOT fall through to rac_fused_linear —
 *     that kernel parallelizes across M, so at M=1 only a single thread
 *     does work while the others idle (one-core-at-100% symptom). And
 *     do NOT route to rac_hal_fused_linear — it transposes the [N,K]
 *     weight into a fresh malloc every call (46 MB for TinyLlama FFN).
 *     Instead use an in-bench parallel-N GEMV: each thread computes a
 *     contiguous chunk of output rows via a vectorized dot product.
 *     All cores busy, no transpose, no malloc.
 */
static inline float _bench_apply_act(float s, rac_activation act) {
    switch (act) {
        case RAC_ACT_RELU: return s > 0.0f ? s : 0.0f;
        case RAC_ACT_GELU: return 0.5f * s * (1.0f + erff(s * 0.7071067811865f));
        case RAC_ACT_SILU: return s / (1.0f + expf(-s));
        default:           return s;
    }
}

static inline void bench_gemv(
    const float *input,       /* [K] */
    const float *weight,      /* [N, K] row-major */
    const float *bias,        /* [N] or NULL */
    float *output,            /* [N] */
    int N, int K, rac_activation act)
{
    /* Parallel across N. Each thread gets ~N/nproc rows. Inner loop is
     * a K-element dot product, auto-vectorized to AVX2/AVX-512 FMA
     * under -march=native. No transpose. */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j++) {
        const float *w_row = weight + (size_t)j * K;
        float s = 0.0f;
        #pragma omp simd reduction(+:s)
        for (int k = 0; k < K; k++) s += input[k] * w_row[k];
        if (bias) s += bias[j];
        output[j] = _bench_apply_act(s, act);
    }
}

static inline void bench_fused_linear(
    const float *input, const float *weight, const float *bias,
    float *output, int M, int N, int K, rac_activation act)
{
    if (M == 1) {
        bench_gemv(input, weight, bias, output, N, K, act);
    } else {
        rac_hal_fused_linear(input, weight, bias, output, M, N, K, act);
    }
}

/*
 * Q8_0-aware variant. When weight_q8 is non-NULL and M==1, we take the
 * dedicated Q8_0 GEMV path (4x less weight bandwidth than f32 — this is
 * where the pitch meets llama.cpp's Q8_0 decode numbers head-on).
 * Otherwise falls through to the f32 path.
 */
static inline void bench_fused_linear_q8(
    const float *input, const float *weight_f32,
    const rac_q8_0_block *weight_q8,
    const float *bias,
    float *output, int M, int N, int K, rac_activation act)
{
    if (M == 1 && weight_q8 != NULL) {
        rac_q8_0_gemv(input, weight_q8, bias, output, N, K, act);
        return;
    }
    bench_fused_linear(input, weight_f32, bias, output, M, N, K, act);
}

/* ── Transformer primitives ────────────────────────────────────────────── */

static void rmsnorm(const float *x, const float *scale, float *out,
                    int T, int d) {
    const float eps = 1e-5f;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float *row = x + (size_t)t * d;
        float ss = 0.0f;
        /* SIMD-reduce the sum of squares. #pragma omp simd lowers to an
         * 8-wide vpmulps+vaddps loop on AVX2 targets. */
        #pragma omp simd reduction(+:ss)
        for (int i = 0; i < d; i++) ss += row[i] * row[i];
        float inv = 1.0f / sqrtf(ss / d + eps);
        float *orow = out + (size_t)t * d;
        #pragma omp simd
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
            #pragma omp simd reduction(+:s)
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
        #pragma omp simd
        for (int h = 0; h < d_head; h++) orow[h] = 0.0f;
        for (int j = 0; j < T; j++) {
            float s = scratch[i*T+j];
            #pragma omp simd
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
    int d = c->d_model, dh = c->d_head;
    int nq = c->n_heads, nkv = c->n_kv_heads;
    int gqa_repeat = nq / nkv;

    /* Attention block */
    rmsnorm(x, w->rms_att, b->norm, T, d);
    bench_fused_linear(b->norm, w->W_q, NULL, b->Q, T, d, d, RAC_ACT_NONE);
    bench_fused_linear(b->norm, w->W_k, NULL, b->K, T, nkv*dh, d, RAC_ACT_NONE);
    bench_fused_linear(b->norm, w->W_v, NULL, b->V, T, nkv*dh, d, RAC_ACT_NONE);

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

    bench_fused_linear(b->att_out, w->W_o, NULL, b->norm, T, d, d, RAC_ACT_NONE);
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < (size_t)T * d; i++) x[i] += b->norm[i];

    /* FFN block (SwiGLU). Gate/up projection uses the AVX2 fused kernel
     * with RAC_ACT_SILU baked in; the elementwise gate*up and final
     * residual add are SIMD-reduced. */
    rmsnorm(x, w->rms_ffn, b->ffn_in, T, d);
    bench_fused_linear(b->ffn_in, w->W_g, NULL, b->ffn_g, T, c->d_ff, d, RAC_ACT_SILU);
    bench_fused_linear(b->ffn_in, w->W_u, NULL, b->ffn_u, T, c->d_ff, d, RAC_ACT_NONE);
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < (size_t)T * c->d_ff; i++) b->ffn_g[i] *= b->ffn_u[i];
    bench_fused_linear(b->ffn_g, w->W_d, NULL, b->norm, T, d, c->d_ff, RAC_ACT_NONE);
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < (size_t)T * d; i++) x[i] += b->norm[i];
}

/* ── Full-model: multi-layer weights + KV cache ────────────────────────── */
/*
 * Full-model path: load all n_layers decoder blocks, run prefill across
 * every layer, then a KV-cache-aware decode loop. This is the number that
 * compares apples-to-apples with llama-bench's "full-model tok/s" — it's
 * the wall time for one token end-to-end through the whole stack, not a
 * single layer in isolation.
 *
 * KV cache layout per layer:
 *   K[T_max, n_kv_heads, d_head], V[T_max, n_kv_heads, d_head]
 *   row-major, stride = n_kv_heads * d_head per timestep.
 */

typedef struct {
    float **K;         /* per-layer [T_max * nkv * d_head] */
    float **V;
    int n_layers;
    int T_max;
    int cur_len;       /* positions filled so far, grows from 0 */
} kv_cache;

static kv_cache *kv_alloc(int n_layers, int T_max, int n_kv_heads, int d_head) {
    kv_cache *kv = (kv_cache *)calloc(1, sizeof(kv_cache));
    if (!kv) { fprintf(stderr, "kv_alloc: oom\n"); exit(1); }
    kv->n_layers = n_layers;
    kv->T_max    = T_max;
    kv->cur_len  = 0;
    kv->K = (float **)calloc(n_layers, sizeof(float *));
    kv->V = (float **)calloc(n_layers, sizeof(float *));
    size_t per_layer = (size_t)T_max * n_kv_heads * d_head;
    for (int L = 0; L < n_layers; L++) {
        kv->K[L] = xalloc(per_layer);
        kv->V[L] = xalloc(per_layer);
    }
    return kv;
}

static void kv_reset(kv_cache *kv) { if (kv) kv->cur_len = 0; }

static void kv_free(kv_cache *kv) {
    if (!kv) return;
    for (int L = 0; L < kv->n_layers; L++) { free(kv->K[L]); free(kv->V[L]); }
    free(kv->K); free(kv->V); free(kv);
}

/*
 * Causal attention with strided K/V access into the KV cache.
 * Reads K[j * stride + kv_h * d_head + h] and V[...] directly — no copy.
 * T_q = query length (new tokens this step), T_kv = cache total length.
 * q_start_pos = absolute position of the first query token.
 */
static void attention_head_causal(
    const float *q_contig,      /* [T_q, d_head] */
    const float *k_cache, const float *v_cache,
    int kv_h, int nkv, int d_head,
    float *out, float *scratch,
    int T_q, int T_kv, int q_start_pos)
{
    const float scale = 1.0f / sqrtf((float)d_head);
    int stride = nkv * d_head;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < T_q; i++) {
        int abs_i = q_start_pos + i;
        float *row = scratch + (size_t)i * T_kv;
        const float *q_row = q_contig + (size_t)i * d_head;
        for (int j = 0; j < T_kv; j++) {
            if (j > abs_i) { row[j] = -1e30f; continue; }
            const float *k_row = k_cache + (size_t)j * stride + (size_t)kv_h * d_head;
            float s = 0.0f;
            /* SIMD-reduce to a vector FMA across d_head. For TinyLlama
             * d_head=64 this lowers to exactly 8 vfmadd231ps instructions. */
            #pragma omp simd reduction(+:s)
            for (int h = 0; h < d_head; h++) s += q_row[h] * k_row[h];
            row[j] = s * scale;
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < T_q; i++) rac_softmax(scratch + (size_t)i * T_kv,
                                               scratch + (size_t)i * T_kv, T_kv);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < T_q; i++) {
        float *orow = out + (size_t)i * d_head;
        /* Accumulate V weighted by the softmax row. Vectorized FMA over
         * d_head, which is also the dimension the store walks. */
        #pragma omp simd
        for (int h = 0; h < d_head; h++) orow[h] = 0.0f;
        for (int j = 0; j < T_kv; j++) {
            float s = scratch[(size_t)i * T_kv + j];
            const float *v_row = v_cache + (size_t)j * stride + (size_t)kv_h * d_head;
            #pragma omp simd
            for (int h = 0; h < d_head; h++) orow[h] += s * v_row[h];
        }
    }
}

/*
 * Cache-aware decoder forward. Computes Q for T_new tokens, writes new
 * K/V straight into the KV cache at position kv->cur_len, runs causal
 * attention over the full cache [0, cur_len + T_new). Does NOT advance
 * cur_len — the caller does that after all layers finish this step so
 * every layer sees the same cache extent.
 */
static void decoder_forward_kv(float *x, const tx_weights *w, tx_bufs *b,
                                const tx_cfg *c, int T_new,
                                kv_cache *kv, int layer_idx) {
    int d = c->d_model, dh = c->d_head;
    int nq = c->n_heads, nkv = c->n_kv_heads;
    int gqa_repeat = nq / nkv;
    int start = kv->cur_len;
    int total = start + T_new;

    rmsnorm(x, w->rms_att, b->norm, T_new, d);
    bench_fused_linear_q8(b->norm, w->W_q, w->Wq_q8, NULL, b->Q, T_new, d, d, RAC_ACT_NONE);

    /* Write K/V directly into the cache at the new slots. */
    float *K_slot = kv->K[layer_idx] + (size_t)start * nkv * dh;
    float *V_slot = kv->V[layer_idx] + (size_t)start * nkv * dh;
    bench_fused_linear_q8(b->norm, w->W_k, w->Wk_q8, NULL, K_slot, T_new, nkv*dh, d, RAC_ACT_NONE);
    bench_fused_linear_q8(b->norm, w->W_v, w->Wv_q8, NULL, V_slot, T_new, nkv*dh, d, RAC_ACT_NONE);

    /* Per query head attention over the full cache. */
    for (int h = 0; h < nq; h++) {
        int kv_h = h / gqa_repeat;
        for (int t = 0; t < T_new; t++) {
            for (int i = 0; i < dh; i++) {
                b->qc[t*dh + i] = b->Q[(size_t)t * d + h * dh + i];
            }
        }
        attention_head_causal(b->qc, kv->K[layer_idx], kv->V[layer_idx],
                              kv_h, nkv, dh,
                              b->oc, b->scratch,
                              T_new, total, start);
        for (int t = 0; t < T_new; t++) {
            for (int i = 0; i < dh; i++) {
                b->att_out[(size_t)t * d + h * dh + i] = b->oc[t*dh + i];
            }
        }
    }

    bench_fused_linear_q8(b->att_out, w->W_o, w->Wo_q8, NULL, b->norm, T_new, d, d, RAC_ACT_NONE);
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < (size_t)T_new * d; i++) x[i] += b->norm[i];

    /* FFN (SwiGLU) — HAL-dispatched fused kernels, SIMD elementwise. */
    rmsnorm(x, w->rms_ffn, b->ffn_in, T_new, d);
    bench_fused_linear_q8(b->ffn_in, w->W_g, w->Wg_q8, NULL, b->ffn_g, T_new, c->d_ff, d, RAC_ACT_SILU);
    bench_fused_linear_q8(b->ffn_in, w->W_u, w->Wu_q8, NULL, b->ffn_u, T_new, c->d_ff, d, RAC_ACT_NONE);
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < (size_t)T_new * c->d_ff; i++) b->ffn_g[i] *= b->ffn_u[i];
    bench_fused_linear_q8(b->ffn_g, w->W_d, w->Wd_q8, NULL, b->norm, T_new, d, c->d_ff, RAC_ACT_NONE);
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < (size_t)T_new * d; i++) x[i] += b->norm[i];
}

/* Full-stack forward over all layers. Advances kv->cur_len at the end. */
static void full_model_forward(float *x, tx_weights *layers, int n_layers,
                                tx_bufs *b, const tx_cfg *c, int T_new,
                                kv_cache *kv) {
    for (int L = 0; L < n_layers; L++) {
        decoder_forward_kv(x, &layers[L], b, c, T_new, kv, L);
    }
    kv->cur_len += T_new;
}

/* Load all N decoder layers from a Llama-style safetensors checkpoint. */
static int init_weights_safetensors_full(tx_weights *layers, tx_cfg *c,
                                          const char *path) {
    /* Re-open once per layer is wasteful but the single-layer path already
     * did that. Here we stash the file handle and reuse it. */
    char err[256];
    st_file *f = st_open(path, err);
    if (!f) { fprintf(stderr, "safetensors open failed: %s\n", err); return -1; }

    size_t d  = c->d_model;
    size_t dk = (size_t)c->n_kv_heads * c->d_head;
    char buf[ST_MAX_NAME];

    for (int L = 0; L < c->n_layers; L++) {
        tx_weights *w = &layers[L];
        w->W_q = xalloc(d * d);
        w->W_k = xalloc(d * dk);
        w->W_v = xalloc(d * dk);
        w->W_o = xalloc(d * d);
        w->W_g = xalloc((size_t)c->d_ff * d);
        w->W_u = xalloc((size_t)c->d_ff * d);
        w->W_d = xalloc((size_t)d * c->d_ff);
        w->rms_att = xalloc(d);
        w->rms_ffn = xalloc(d);

        #define LOAD_L(name, ptr, expect_numel) do {                                \
            snprintf(buf, sizeof(buf), "model.layers.%d." name, L);                \
            const st_tensor *t = st_find(f, buf);                                  \
            if (!t) { fprintf(stderr, "missing tensor: %s\n", buf); st_close(f); return -1; } \
            size_t nel = st_numel(t);                                              \
            if (nel != (size_t)(expect_numel)) {                                   \
                fprintf(stderr, "%s shape mismatch: got %zu expected %zu\n",       \
                        buf, nel, (size_t)(expect_numel));                         \
                st_close(f); return -1;                                            \
            }                                                                      \
            if (st_to_f32(f, t, ptr) != 0) {                                       \
                fprintf(stderr, "dtype not supported for %s (%s)\n", buf,          \
                        st_dtype_name(t->dtype));                                  \
                st_close(f); return -1;                                            \
            }                                                                      \
        } while (0)

        LOAD_L("self_attn.q_proj.weight", w->W_q, d * d);
        LOAD_L("self_attn.k_proj.weight", w->W_k, d * dk);
        LOAD_L("self_attn.v_proj.weight", w->W_v, d * dk);
        LOAD_L("self_attn.o_proj.weight", w->W_o, d * d);
        LOAD_L("mlp.gate_proj.weight",    w->W_g, (size_t)c->d_ff * d);
        LOAD_L("mlp.up_proj.weight",      w->W_u, (size_t)c->d_ff * d);
        LOAD_L("mlp.down_proj.weight",    w->W_d, (size_t)d * c->d_ff);
        LOAD_L("input_layernorm.weight",          w->rms_att, d);
        LOAD_L("post_attention_layernorm.weight", w->rms_ffn, d);
        #undef LOAD_L
    }
    printf("  loaded %d layers from %s\n", c->n_layers, path);
    st_close(f);
    return 0;
}

static void init_weights_synthetic_full(tx_weights *layers, const tx_cfg *c) {
    for (int L = 0; L < c->n_layers; L++) {
        init_weights_synthetic(&layers[L], c);
    }
}

/* Resize bufs to support T_q and T_kv bounds. For --full-model we need
 * scratch of size max(T_q) * max(T_kv). Safest: alloc at prefill_T for Q,
 * and T_max = prefill_T + decode_iters for K/V. */
static void bufs_alloc_for_full_model(tx_bufs *b, const tx_cfg *c,
                                       int T_q_max, int T_kv_max) {
    size_t d = c->d_model;
    b->norm    = xalloc((size_t)T_q_max * d);
    b->Q       = xalloc((size_t)T_q_max * d);
    b->K       = NULL;   /* KV cache owns the real K / V storage */
    b->V       = NULL;
    b->att_out = xalloc((size_t)T_q_max * d);
    b->scratch = xalloc((size_t)T_q_max * T_kv_max);
    b->ffn_in  = xalloc((size_t)T_q_max * d);
    b->ffn_g   = xalloc((size_t)T_q_max * c->d_ff);
    b->ffn_u   = xalloc((size_t)T_q_max * c->d_ff);
    b->qc      = xalloc((size_t)T_q_max * c->d_head);
    b->kc      = NULL;
    b->vc      = NULL;
    b->oc      = xalloc((size_t)T_q_max * c->d_head);
}
static void bufs_free_full_model(tx_bufs *b) {
    free(b->norm); free(b->Q); free(b->att_out);
    free(b->scratch); free(b->ffn_in); free(b->ffn_g); free(b->ffn_u);
    free(b->qc); free(b->oc);
}

/* ── CLI ───────────────────────────────────────────────────────────────── */

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s [options]\n"
        "  --config {tiny|tinyllama}    model shape preset (default: tiny)\n"
        "  --safetensors PATH           load layer weights from HF checkpoint\n"
        "                                 (implies --config tinyllama)\n"
        "  --layer N                    which layer to load (single-layer mode)\n"
        "  --full-model                 run ALL decoder layers + KV cache\n"
        "                                 (apples-to-apples with llama-bench)\n"
        "  --q8_0                       quantize linear weights to Q8_0 after\n"
        "                                 load (block=32 int8 + fp16 scale,\n"
        "                                 matches llama.cpp). Decode GEMV uses\n"
        "                                 4x less weight bandwidth.\n"
        "  --prefill N                  prefill T tokens (default: 128)\n"
        "  --prefill-iters N            # prefill passes (default: 30)\n"
        "  --decode-iters N             # decode passes (default: 100)\n"
        "  -h, --help                   this help\n"
        "\n"
        "Modes\n"
        "  Default:        time a single decoder layer in isolation. Useful\n"
        "                  for microbenchmarks + kernel tuning.\n"
        "  --full-model:   time a forward pass through every layer with a\n"
        "                  KV cache between decode steps. This is the metric\n"
        "                  that compares directly to llama-bench's\n"
        "                  full-model tok/s — no \"divide by n_layers\"\n"
        "                  fiction on either side.\n"
        "\n"
        "Fetch weights via:\n"
        "  MODEL_DIR=$(python3 bench/fetch_model.py \\\n"
        "                --model TinyLlama/TinyLlama-1.1B-Chat-v1.0)\n"
        "  %s --safetensors \"$MODEL_DIR/model.safetensors\" --full-model\n",
        argv0, argv0);
}

static int run_full_model(tx_cfg *c) {
    /* Allocate and populate weights for all layers. */
    tx_weights *layers = (tx_weights *)calloc(c->n_layers, sizeof(tx_weights));
    if (!layers) { fprintf(stderr, "layer alloc failed\n"); return 2; }
    if (c->safetensors_path) {
        if (init_weights_safetensors_full(layers, c, c->safetensors_path) != 0) {
            free(layers); return 2;
        }
    } else {
        init_weights_synthetic_full(layers, c);
    }

    /* Optional Q8_0 shadow-quantize of all linear layer weights. One-time
     * pass (a few seconds for TinyLlama on 32 threads). Adds ~0.25x memory
     * overhead on top of f32 weights. Decode GEMVs then use the Q8_0
     * variant; prefill still uses the f32 AVX2 micro-kernel. */
    if (c->quant_q8_0) {
        printf("  quantizing weights to Q8_0 (decode path) ...\n");
        double q_t0 = now_sec();
        for (int L = 0; L < c->n_layers; L++) {
            weights_quantize_q8_0(&layers[L], c);
        }
        printf("  Q8_0 quantization complete (%.2fs, 4x weight bandwidth reduction on decode)\n",
               now_sec() - q_t0);
    }

    int T_prompt = c->prefill_T;
    int T_max    = T_prompt + c->decode_iters + 16;  /* headroom for warmup */

    tx_bufs b;
    bufs_alloc_for_full_model(&b, c, /*T_q_max=*/T_prompt, /*T_kv_max=*/T_max);
    kv_cache *kv = kv_alloc(c->n_layers, T_max, c->n_kv_heads, c->d_head);

    /* Residual stream buffer sized for the larger of the two passes. */
    float *x_prompt = xalloc((size_t)T_prompt * c->d_model);
    float *x_decode = xalloc((size_t)c->d_model);

    /* ── Warmup: one prefill pass end-to-end. ────────────────────────── */
    for (size_t i = 0; i < (size_t)T_prompt * c->d_model; i++)
        x_prompt[i] = rand_xavier(c->d_model);
    kv_reset(kv);
    full_model_forward(x_prompt, layers, c->n_layers, &b, c, T_prompt, kv);

    /* ── Prefill benchmark: full stack, T prompt tokens, per iteration. ── */
    double t0 = now_sec();
    for (int it = 0; it < c->prefill_iters; it++) {
        for (size_t i = 0; i < (size_t)T_prompt * c->d_model; i++)
            x_prompt[i] = rand_xavier(c->d_model);
        kv_reset(kv);
        full_model_forward(x_prompt, layers, c->n_layers, &b, c, T_prompt, kv);
    }
    double s_pre = now_sec() - t0;
    double ms_pre  = s_pre * 1000.0 / c->prefill_iters;
    double tps_pre = (T_prompt * c->prefill_iters) / s_pre;

    /* ── Decode benchmark: KV-cache already filled by the last prefill.  */
    /*     Time N single-token forward passes that append to the cache.   */
    /*     The cache is reset once, prefill warms it to length T_prompt,  */
    /*     then we measure pure decode throughput.                        */
    kv_reset(kv);
    for (size_t i = 0; i < (size_t)T_prompt * c->d_model; i++)
        x_prompt[i] = rand_xavier(c->d_model);
    full_model_forward(x_prompt, layers, c->n_layers, &b, c, T_prompt, kv);

    t0 = now_sec();
    for (int it = 0; it < c->decode_iters; it++) {
        for (int i = 0; i < c->d_model; i++) x_decode[i] = rand_xavier(c->d_model);
        full_model_forward(x_decode, layers, c->n_layers, &b, c, 1, kv);
        if (kv->cur_len + 1 >= kv->T_max) break;   /* safety */
    }
    double s_dec = now_sec() - t0;
    double ms_dec = s_dec * 1000.0 / c->decode_iters;
    double tps_dec = c->decode_iters / s_dec;

    /* ── GFLOPS accounting ───────────────────────────────────────────── */
    double flops_per_tok_per_layer =
          2.0 * c->d_model * c->d_model                    /* Q */
        + 2.0 * c->d_model * (c->n_kv_heads * c->d_head)   /* K */
        + 2.0 * c->d_model * (c->n_kv_heads * c->d_head)   /* V */
        + 2.0 * c->d_model * c->d_model                    /* O */
        + 3.0 * 2.0 * c->d_model * c->d_ff;                /* FFN */
    /* Attention compute varies by step — use prompt T as upper bound. */
    double flops_per_tok = flops_per_tok_per_layer * c->n_layers;
    double gflops_pre = flops_per_tok * T_prompt * c->prefill_iters / s_pre / 1e9;
    double gflops_dec = flops_per_tok * c->decode_iters / s_dec / 1e9;

    printf("\n── RAC results (full-model, %d layers, KV cache, %s decode) ──────────\n",
           c->n_layers, c->quant_q8_0 ? "Q8_0" : "F32");
    printf("  prefill T=%d:   %7.2f ms/token   %8.2f tok/s   %7.1f GFLOPS\n",
           T_prompt, ms_pre / T_prompt, tps_pre, gflops_pre);
    printf("  decode  T=1:    %7.2f ms/token   %8.2f tok/s   %7.1f GFLOPS\n",
           ms_dec, tps_dec, gflops_dec);
    printf("  (these compare directly to llama-bench pp128/tg128 "
           "full-model tok/s — no divide-by-layers)\n");

    /* Clean up. */
    for (int L = 0; L < c->n_layers; L++) free_weights(&layers[L]);
    free(layers);
    bufs_free_full_model(&b);
    kv_free(kv);
    free(x_prompt); free(x_decode);
    return 0;
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
        else if (!strcmp(argv[i], "--full-model"))                 c.full_model    = 1;
        else if (!strcmp(argv[i], "--q8_0") || !strcmp(argv[i], "--q8")) c.quant_q8_0  = 1;
        else if (!strcmp(argv[i], "--layer") && i+1 < argc)        c.layer_idx     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefill") && i+1 < argc)      c.prefill_T     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefill-iters") && i+1 < argc)c.prefill_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--decode-iters") && i+1 < argc) c.decode_iters  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    /* Probe CPU features + cache topology + thread count once, so every
     * downstream rac_hal_* call routes to the AVX2/Zen3/AVX512 asm
     * micro-kernels with the right tile size. Without this init, calls
     * fall through to the scalar rac_cpu.c path. */
    rac_hal_init();
    rac_hal_print_profile();

    printf("RAC %stransformer inference bench%s\n",
           c.full_model ? "FULL-MODEL " : "single-layer ",
           c.quant_q8_0 ? " [Q8_0 decode]" : "");
    printf("  config=%s  d_model=%d  n_heads=%d  n_kv_heads=%d  d_head=%d  d_ff=%d"
           "  n_layers=%d\n  prefill_T=%d  prefill_iters=%d  decode_iters=%d\n",
           c.config_name, c.d_model, c.n_heads, c.n_kv_heads, c.d_head, c.d_ff,
           c.n_layers, c.prefill_T, c.prefill_iters, c.decode_iters);
    if (c.safetensors_path) {
        if (c.full_model)
            printf("  weights: all %d layers from %s\n", c.n_layers, c.safetensors_path);
        else
            printf("  weights: layer %d from %s\n", c.layer_idx, c.safetensors_path);
    } else {
        printf("  weights: synthetic xavier (deterministic, seed=0xC0FFEE)\n");
    }

    if (c.full_model) {
        int rc = run_full_model(&c);
        rac_hal_shutdown();
        return rc;
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

    printf("\n── Framework comparison ────────────────────────────\n");
    printf("  To compare against llama.cpp + tinygrad on this hardware:\n");
    printf("    ./bench/configure.sh                       # status check + auto-fill configs\n");
    printf("    ./bench/bench_harness.sh --auto-install --auto-build\n");
    printf("  See bench/README.md for per-framework invocation.\n");

    free_weights(&w);
    bufs_free(&b);
    free(x);
    rac_hal_shutdown();
    return 0;
}
