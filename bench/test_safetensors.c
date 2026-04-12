/*
 * test_safetensors.c — BVT for the minimal safetensors reader
 * Pinnacle Quantum Group — April 2026
 *
 * Synthesizes a small safetensors file with known tensors in F32, F16,
 * and BF16 dtypes, then round-trips them back through st_to_f32 and
 * verifies values. Runs entirely in /tmp — no HF network access needed.
 */

#define _POSIX_C_SOURCE 200809L
#include "safetensors_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>

static int passed = 0, failed = 0;
#define CHECK(name, cond) do { \
    if (cond) { passed++; printf("  [PASS] %s\n", name); } \
    else      { failed++; printf("  [FAIL] %s\n", name); } \
} while(0)

/* Encode f32 -> bf16 (top 16 bits). */
static uint16_t f32_to_bf16(float f) {
    union { float f; uint32_t u; } u = { f };
    return (uint16_t)((u.u + 0x8000) >> 16);     /* round-to-nearest */
}

/* Encode f32 -> f16 (IEEE 754 binary16). */
static uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } u = { f };
    uint32_t x = u.u;
    uint32_t sign = (x >> 31) & 1;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | (0x1F << 10) | mant);
    return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | mant);
}

static void write_u64_le(FILE *fp, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)((v >> (8*i)) & 0xFF);
    fwrite(b, 1, 8, fp);
}

int main(void) {
    printf("safetensors reader BVT — Pinnacle Quantum Group\n");

    const char *path = "/tmp/rac_bench_test.safetensors";

    /* ── 1. Build a synthetic safetensors file ───────────────────────── */
    /*
     * 3 tensors, packed tightly:
     *   t_f32  F32  shape [4]   bytes  16
     *   t_f16  F16  shape [3]   bytes   6
     *   t_bf16 BF16 shape [2,2] bytes  8
     * Header is a JSON dict with those entries; offsets within the data
     * segment (NOT absolute file offsets).
     */
    float  f32_in [4] = {1.0f, -2.5f, 3.14159f, 1e6f};
    float  f16_src[3] = {0.5f, -1.0f, 2.0f};
    float  bf16_src[4] = {0.0f, 1.0f, -0.5f, 100.0f};

    uint16_t f16_data[3], bf16_data[4];
    for (int i = 0; i < 3; i++) f16_data[i]  = f32_to_f16(f16_src[i]);
    for (int i = 0; i < 4; i++) bf16_data[i] = f32_to_bf16(bf16_src[i]);

    /* Header JSON. Offsets are into the data segment: 0, 16, 22. */
    const char *hdr_json =
        "{"
        "\"__metadata__\":{\"format\":\"pt\"},"
        "\"t_f32\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]},"
        "\"t_f16\":{\"dtype\":\"F16\",\"shape\":[3],\"data_offsets\":[16,22]},"
        "\"t_bf16\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[22,30]}"
        "}";
    size_t hdr_len = strlen(hdr_json);

    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen failed\n"); return 1; }
    write_u64_le(fp, (uint64_t)hdr_len);
    fwrite(hdr_json, 1, hdr_len, fp);
    fwrite(f32_in,    sizeof(float),    4, fp);
    fwrite(f16_data,  sizeof(uint16_t), 3, fp);
    fwrite(bf16_data, sizeof(uint16_t), 4, fp);
    fclose(fp);

    /* ── 2. Open + parse ────────────────────────────────────────────── */
    char err[256] = {0};
    st_file *f = st_open(path, err);
    if (!f) {
        fprintf(stderr, "st_open failed: %s\n", err);
        return 1;
    }
    CHECK("st_open succeeded",              f != NULL);
    CHECK("parsed 3 tensors",               f->n_tensors == 3);

    const st_tensor *t32  = st_find(f, "t_f32");
    const st_tensor *t16  = st_find(f, "t_f16");
    const st_tensor *tbf  = st_find(f, "t_bf16");
    CHECK("st_find t_f32",     t32 != NULL);
    CHECK("st_find t_f16",     t16 != NULL);
    CHECK("st_find t_bf16",    tbf != NULL);
    CHECK("st_find missing",   st_find(f, "not_there") == NULL);

    CHECK("t_f32 dtype",       t32 && t32->dtype == ST_F32);
    CHECK("t_f16 dtype",       t16 && t16->dtype == ST_F16);
    CHECK("t_bf16 dtype",      tbf && tbf->dtype == ST_BF16);

    CHECK("t_f32 shape[0]=4",  t32 && t32->ndim == 1 && t32->shape[0] == 4);
    CHECK("t_bf16 shape=[2,2]",tbf && tbf->ndim == 2 && tbf->shape[0] == 2 && tbf->shape[1] == 2);
    CHECK("t_bf16 numel=4",    tbf && st_numel(tbf) == 4);

    /* ── 3. Decode ──────────────────────────────────────────────────── */
    float buf[16];
    CHECK("st_to_f32 F32",   st_to_f32(f, t32, buf) == 0);
    int ok = 1;
    for (int i = 0; i < 4; i++) if (fabsf(buf[i] - f32_in[i]) > 1e-6f) ok = 0;
    CHECK("F32 values identical",     ok);

    CHECK("st_to_f32 F16",   st_to_f32(f, t16, buf) == 0);
    ok = 1;
    for (int i = 0; i < 3; i++) if (fabsf(buf[i] - f16_src[i]) > 1e-3f) ok = 0;
    CHECK("F16 values within 1e-3",   ok);

    CHECK("st_to_f32 BF16",  st_to_f32(f, tbf, buf) == 0);
    ok = 1;
    for (int i = 0; i < 4; i++) {
        float want = bf16_src[i];
        /* BF16 has ~1e-2 rel precision at larger magnitudes */
        float tol = fmaxf(fabsf(want) * 0.01f, 1e-3f);
        if (fabsf(buf[i] - want) > tol) ok = 0;
    }
    CHECK("BF16 values within 1%",    ok);

    /* ── 4. Dump (visual inspection) ────────────────────────────────── */
    printf("\n  st_dump:\n");
    st_dump(f);

    st_close(f);
    unlink(path);

    printf("\npassed=%d failed=%d\n", passed, failed);
    printf("%s\n", failed == 0 ? "ALL SAFETENSORS BVT PASSED"
                               : "SAFETENSORS BVT FAILURES");
    return failed == 0 ? 0 : 1;
}
