/*
 * safetensors_reader.c — Minimal safetensors reader
 * Pinnacle Quantum Group — April 2026
 *
 * Hand-rolled parser tuned to the narrow schema we actually need:
 *   "name":{"dtype":"TYPE","shape":[...],"data_offsets":[START,END]}
 *
 * Good enough for every Llama-family safetensors checkpoint the bench
 * runs against. Not a general-purpose JSON parser.
 */

#define _POSIX_C_SOURCE 200809L
#include "safetensors_reader.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* ── Utilities ──────────────────────────────────────────────────────────── */

static void set_err(st_file *f, const char *fmt, ...) {
    if (!f) return;
    va_list args;
    va_start(args, fmt);
    vsnprintf(f->err, sizeof(f->err), fmt, args);
    va_end(args);
}

static st_dtype parse_dtype(const char *s, size_t n) {
    if (n == 3 && !strncmp(s, "F32",  3)) return ST_F32;
    if (n == 3 && !strncmp(s, "F16",  3)) return ST_F16;
    if (n == 4 && !strncmp(s, "BF16", 4)) return ST_BF16;
    if (n == 3 && !strncmp(s, "F64",  3)) return ST_F64;
    if (n == 2 && !strncmp(s, "I8",   2)) return ST_I8;
    if (n == 2 && !strncmp(s, "U8",   2)) return ST_U8;
    if (n == 3 && !strncmp(s, "I16",  3)) return ST_I16;
    if (n == 3 && !strncmp(s, "I32",  3)) return ST_I32;
    if (n == 3 && !strncmp(s, "I64",  3)) return ST_I64;
    if (n == 4 && !strncmp(s, "BOOL", 4)) return ST_BOOL;
    return ST_UNKNOWN;
}

const char *st_dtype_name(st_dtype d) {
    switch (d) {
        case ST_F32: return "F32";
        case ST_F16: return "F16";
        case ST_BF16: return "BF16";
        case ST_F64: return "F64";
        case ST_I8:  return "I8";
        case ST_U8:  return "U8";
        case ST_I16: return "I16";
        case ST_I32: return "I32";
        case ST_I64: return "I64";
        case ST_BOOL: return "BOOL";
        default: return "UNKNOWN";
    }
}

size_t st_numel(const st_tensor *t) {
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= (size_t)t->shape[i];
    return n;
}

/* ── JSON-ish scanner ───────────────────────────────────────────────────── */

/* Skip whitespace */
static const char *skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
        p++;
    return p;
}

/* Find matching close brace starting at p (which points to '{'). */
static const char *match_brace(const char *p, const char *end) {
    int depth = 0;
    int in_str = 0;
    for (; p < end; p++) {
        if (in_str) {
            if (*p == '\\' && p + 1 < end) { p++; continue; }
            if (*p == '"') in_str = 0;
            continue;
        }
        if (*p == '"') { in_str = 1; continue; }
        if (*p == '{') depth++;
        else if (*p == '}') {
            depth--;
            if (depth == 0) return p + 1;
        }
    }
    return NULL;
}

/* Extract quoted string starting at p (p points at opening ").
 * Returns pointer past closing " and writes length + pointer-to-content. */
static const char *parse_string(const char *p, const char *end,
                                const char **out_start, size_t *out_len) {
    if (p >= end || *p != '"') return NULL;
    p++;
    *out_start = p;
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) p += 2;
        else p++;
    }
    if (p >= end) return NULL;
    *out_len = (size_t)(p - *out_start);
    return p + 1;
}

/* Parse one integer, advancing p past it. */
static const char *parse_int(const char *p, const char *end, int64_t *out) {
    char buf[32];
    size_t n = 0;
    while (p < end && n + 1 < sizeof(buf) &&
           (*p == '-' || *p == '+' || isdigit((unsigned char)*p))) {
        buf[n++] = *p++;
    }
    buf[n] = '\0';
    if (n == 0) return NULL;
    *out = strtoll(buf, NULL, 10);
    return p;
}

/* ── Header parse ───────────────────────────────────────────────────────── */

static int parse_tensor_body(const char *body, const char *end,
                             st_tensor *t) {
    /* body points inside the {...} value object. Fields in any order. */
    const char *p = skip_ws(body, end);
    if (p >= end || *p != '{') return -1;
    p++;

    int have_dtype = 0, have_shape = 0, have_offsets = 0;
    while (p < end) {
        p = skip_ws(p, end);
        if (p >= end) return -1;
        if (*p == '}') { p++; break; }
        if (*p == ',') { p++; continue; }
        if (*p != '"') return -1;

        const char *key; size_t keylen;
        p = parse_string(p, end, &key, &keylen);
        if (!p) return -1;
        p = skip_ws(p, end);
        if (p >= end || *p != ':') return -1;
        p++;
        p = skip_ws(p, end);

        if (keylen == 5 && !strncmp(key, "dtype", 5)) {
            const char *v; size_t vlen;
            p = parse_string(p, end, &v, &vlen);
            if (!p) return -1;
            t->dtype = parse_dtype(v, vlen);
            have_dtype = 1;
        } else if (keylen == 5 && !strncmp(key, "shape", 5)) {
            if (*p != '[') return -1;
            p++;
            t->ndim = 0;
            while (p < end) {
                p = skip_ws(p, end);
                if (*p == ']') { p++; break; }
                if (*p == ',') { p++; continue; }
                int64_t v;
                p = parse_int(p, end, &v);
                if (!p) return -1;
                if (t->ndim < ST_MAX_NDIM) t->shape[t->ndim++] = v;
            }
            have_shape = 1;
        } else if (keylen == 12 && !strncmp(key, "data_offsets", 12)) {
            if (*p != '[') return -1;
            p++;
            int64_t a, b;
            p = skip_ws(p, end);
            p = parse_int(p, end, &a);
            if (!p) return -1;
            p = skip_ws(p, end);
            if (*p == ',') p++;
            p = skip_ws(p, end);
            p = parse_int(p, end, &b);
            if (!p) return -1;
            p = skip_ws(p, end);
            if (*p != ']') return -1;
            p++;
            t->offset = (size_t)a;    /* relative to data segment start */
            t->nbytes = (size_t)(b - a);
            have_offsets = 1;
        } else {
            /* Unknown key — skip value robustly. We only need to handle
             * string/array/object/number values. */
            p = skip_ws(p, end);
            if (*p == '"') {
                const char *s; size_t sl;
                p = parse_string(p, end, &s, &sl);
                if (!p) return -1;
            } else if (*p == '[') {
                /* skip to matching ] */
                int d = 0;
                while (p < end) {
                    if (*p == '[') d++;
                    else if (*p == ']') { d--; if (d == 0) { p++; break; } }
                    p++;
                }
            } else if (*p == '{') {
                p = match_brace(p, end);
                if (!p) return -1;
            } else {
                while (p < end && *p != ',' && *p != '}') p++;
            }
        }
    }
    return (have_dtype && have_shape && have_offsets) ? 0 : -1;
}

static int parse_header(st_file *f, const char *json, size_t json_len) {
    const char *p = json;
    const char *end = json + json_len;

    p = skip_ws(p, end);
    if (p >= end || *p != '{') {
        set_err(f, "header is not a JSON object");
        return -1;
    }
    p++;

    while (p < end) {
        p = skip_ws(p, end);
        if (p >= end) break;
        if (*p == '}') break;
        if (*p == ',') { p++; continue; }
        if (*p != '"') {
            set_err(f, "expected key at offset %ld", (long)(p - json));
            return -1;
        }

        const char *key; size_t keylen;
        p = parse_string(p, end, &key, &keylen);
        if (!p) { set_err(f, "unterminated key"); return -1; }
        p = skip_ws(p, end);
        if (p >= end || *p != ':') { set_err(f, "expected ':'"); return -1; }
        p++;
        p = skip_ws(p, end);

        /* __metadata__ is skipped — it's informational only. */
        if (keylen == 12 && !strncmp(key, "__metadata__", 12)) {
            if (*p == '{') {
                p = match_brace(p, end);
                if (!p) { set_err(f, "bad __metadata__ block"); return -1; }
            }
            continue;
        }

        if (f->n_tensors >= ST_MAX_TENSORS) {
            set_err(f, "too many tensors (>%d)", ST_MAX_TENSORS);
            return -1;
        }

        st_tensor *t = &f->tensors[f->n_tensors];
        size_t copy = keylen < ST_MAX_NAME - 1 ? keylen : ST_MAX_NAME - 1;
        memcpy(t->name, key, copy);
        t->name[copy] = '\0';

        /* Body is a {...} object */
        if (*p != '{') { set_err(f, "expected '{' for tensor body"); return -1; }
        const char *body_end = match_brace(p, end);
        if (!body_end) { set_err(f, "unterminated tensor body"); return -1; }
        if (parse_tensor_body(p, body_end, t) != 0) {
            set_err(f, "malformed tensor body for '%s'", t->name);
            return -1;
        }
        p = body_end;
        f->n_tensors++;
    }
    return 0;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

st_file *st_open(const char *path, char errout[256]) {
    st_file *f = calloc(1, sizeof(*f));
    if (!f) { if (errout) snprintf(errout, 256, "oom"); return NULL; }
    f->fd = -1;
    f->tensors = calloc(ST_MAX_TENSORS, sizeof(st_tensor));
    if (!f->tensors) { st_close(f); if (errout) snprintf(errout, 256, "oom"); return NULL; }

    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) {
        snprintf(f->err, sizeof(f->err), "open(%s): %s", path, strerror(errno));
        if (errout) snprintf(errout, 256, "%s", f->err);
        st_close(f);
        return NULL;
    }
    struct stat sb;
    if (fstat(f->fd, &sb) != 0) {
        snprintf(f->err, sizeof(f->err), "fstat: %s", strerror(errno));
        if (errout) snprintf(errout, 256, "%s", f->err);
        st_close(f);
        return NULL;
    }
    f->mmap_len = (size_t)sb.st_size;
    if (f->mmap_len < 8) {
        snprintf(f->err, sizeof(f->err), "file too small (%zu bytes)", f->mmap_len);
        if (errout) snprintf(errout, 256, "%s", f->err);
        st_close(f);
        return NULL;
    }
    f->mmap_base = mmap(NULL, f->mmap_len, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (f->mmap_base == MAP_FAILED) {
        snprintf(f->err, sizeof(f->err), "mmap: %s", strerror(errno));
        if (errout) snprintf(errout, 256, "%s", f->err);
        f->mmap_base = NULL;
        st_close(f);
        return NULL;
    }

    uint64_t header_len;
    memcpy(&header_len, f->mmap_base, 8);
    if (header_len > f->mmap_len - 8) {
        snprintf(f->err, sizeof(f->err),
                 "header_len %llu exceeds file size",
                 (unsigned long long)header_len);
        if (errout) snprintf(errout, 256, "%s", f->err);
        st_close(f);
        return NULL;
    }
    f->data_offset = 8 + header_len;

    /* Copy header to null-terminated buffer for ergonomics */
    f->header_copy = malloc(header_len + 1);
    if (!f->header_copy) {
        snprintf(f->err, sizeof(f->err), "oom header copy");
        if (errout) snprintf(errout, 256, "%s", f->err);
        st_close(f);
        return NULL;
    }
    memcpy(f->header_copy, (char *)f->mmap_base + 8, header_len);
    f->header_copy[header_len] = '\0';

    if (parse_header(f, f->header_copy, header_len) != 0) {
        if (errout) snprintf(errout, 256, "%s", f->err);
        st_close(f);
        return NULL;
    }
    return f;
}

void st_close(st_file *f) {
    if (!f) return;
    if (f->mmap_base) munmap(f->mmap_base, f->mmap_len);
    if (f->fd >= 0) close(f->fd);
    free(f->header_copy);
    free(f->tensors);
    free(f);
}

const st_tensor *st_find(const st_file *f, const char *name) {
    if (!f) return NULL;
    for (int i = 0; i < f->n_tensors; i++) {
        if (!strcmp(f->tensors[i].name, name)) return &f->tensors[i];
    }
    return NULL;
}

const void *st_data(const st_file *f, const st_tensor *t) {
    return (const char *)f->mmap_base + f->data_offset + t->offset;
}

/* ── dtype conversions ──────────────────────────────────────────────────── */

static inline float f16_to_f32(uint16_t h) {
    /* IEEE 754 binary16 → binary32 via bit manipulation. */
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            /* subnormal: normalize */
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            exp++;
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000 | (mant << 13);   /* inf or nan */
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } u = { f };
    return u.f;
}

static inline float bf16_to_f32(uint16_t b) {
    /* BF16 → F32: pad 16 zero bits on the right. */
    union { uint32_t u; float f; } u;
    u.u = ((uint32_t)b) << 16;
    return u.f;
}

int st_to_f32(const st_file *f, const st_tensor *t, float *out) {
    const void *src = st_data(f, t);
    size_t n = st_numel(t);

    switch (t->dtype) {
        case ST_F32:
            memcpy(out, src, n * sizeof(float));
            return 0;
        case ST_F16: {
            const uint16_t *p = (const uint16_t *)src;
            for (size_t i = 0; i < n; i++) out[i] = f16_to_f32(p[i]);
            return 0;
        }
        case ST_BF16: {
            const uint16_t *p = (const uint16_t *)src;
            for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(p[i]);
            return 0;
        }
        case ST_F64: {
            const double *p = (const double *)src;
            for (size_t i = 0; i < n; i++) out[i] = (float)p[i];
            return 0;
        }
        default:
            /* Integer / quantized types not supported in this minimal reader. */
            return -1;
    }
}

void st_dump(const st_file *f) {
    if (!f) { printf("(null st_file)\n"); return; }
    printf("st_file: %d tensors, data starts at offset %zu\n",
           f->n_tensors, f->data_offset);
    for (int i = 0; i < f->n_tensors; i++) {
        const st_tensor *t = &f->tensors[i];
        printf("  [%4d] %-60s %-5s shape=[", i, t->name, st_dtype_name(t->dtype));
        for (int d = 0; d < t->ndim; d++) {
            printf("%s%lld", d ? "," : "", (long long)t->shape[d]);
        }
        printf("] nbytes=%zu offset=%zu\n", t->nbytes, t->offset);
    }
}
