/*
 * safetensors_reader.h — Minimal safetensors reader for the RAC bench
 * Pinnacle Quantum Group — April 2026
 *
 * Reads the safetensors container format
 * (https://github.com/huggingface/safetensors):
 *
 *     [u64 header_len_le] [header_len bytes of JSON] [raw tensor data]
 *
 * JSON schema per tensor:
 *     {"dtype": "F32"|"F16"|"BF16"|"I8"|...,
 *      "shape": [N1, N2, ...],
 *      "data_offsets": [start, end]}
 *
 * The file is mmap'd; tensor pointers stay valid until st_close. For dtype
 * promotion to F32 call st_to_f32, which handles F32 (memcpy), F16, and
 * BF16 — the dtypes used by common Llama-family checkpoints. Integer
 * dtypes error out cleanly; we're not writing a quant runtime here.
 *
 * Not a production JSON parser — just enough to drive bench harnesses.
 */

#ifndef SAFETENSORS_READER_H
#define SAFETENSORS_READER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ST_MAX_TENSORS  8192
#define ST_MAX_NAME     256
#define ST_MAX_NDIM     8

typedef enum {
    ST_UNKNOWN = 0,
    ST_F32     = 1,
    ST_F16     = 2,
    ST_BF16    = 3,
    ST_F64     = 4,
    ST_I8      = 5,
    ST_U8      = 6,
    ST_I16     = 7,
    ST_I32     = 8,
    ST_I64     = 9,
    ST_BOOL    = 10,
} st_dtype;

typedef struct {
    char      name[ST_MAX_NAME];
    st_dtype  dtype;
    int       ndim;
    int64_t   shape[ST_MAX_NDIM];
    size_t    offset;    /* absolute offset into the file (past header) */
    size_t    nbytes;
} st_tensor;

typedef struct st_file {
    int          fd;
    void        *mmap_base;
    size_t       mmap_len;
    size_t       data_offset;   /* 8 + header_len */
    st_tensor   *tensors;
    int          n_tensors;
    char        *header_copy;   /* owned; null-terminated header JSON */
    char         err[256];
} st_file;

/* Open & parse the safetensors header. Returns NULL on failure; caller may
 * inspect `errout` (may be NULL). */
st_file *st_open(const char *path, char errout[256]);

/* Close and release all resources. Safe to call on NULL. */
void st_close(st_file *f);

/* Find tensor by exact name. Returns NULL if absent. */
const st_tensor *st_find(const st_file *f, const char *name);

/* Raw pointer to tensor bytes inside the mmap. Do NOT free. */
const void *st_data(const st_file *f, const st_tensor *t);

/* Convert tensor to float32 into caller-provided buffer.
 *   out must have space for prod(shape) * sizeof(float).
 * Returns 0 on success, -1 if dtype is unsupported (integers). */
int st_to_f32(const st_file *f, const st_tensor *t, float *out);

/* Count elements in tensor (product of shape). */
size_t st_numel(const st_tensor *t);

/* Human-readable dtype name. */
const char *st_dtype_name(st_dtype d);

/* List all tensor names to stdout (debugging). */
void st_dump(const st_file *f);

#ifdef __cplusplus
}
#endif

#endif /* SAFETENSORS_READER_H */
