/*
 * rac_blas.h — RAC BLAS API (Levels 1, 2, 3, single precision, row-major)
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * Standard BLAS-style single-precision (float) interface implemented on top
 * of RAC's CORDIC primitives. The API mirrors the Netlib reference BLAS
 * naming (sxxx) but with a few intentional differences:
 *
 *   1. Row-major storage by default (CBLAS_ORDER == CblasRowMajor).
 *      Leading dimensions follow the CBLAS convention (lda is the row
 *      stride for row-major storage).
 *   2. All names are namespaced with the rac_ prefix (rac_saxpy, ...).
 *   3. Transpose flags use rac_blas_op (NoTrans / Trans).
 *   4. UpLo / Diag / Side flags follow the CBLAS enum convention.
 *
 * This file only declares; rac_blas.c contains the OpenMP-parallelized
 * implementation. The Level-3 GEMM family forwards to rac_sgemm
 * (rac_cpu.c) when no transpose is requested, falling back to a
 * row-major reference kernel otherwise.
 *
 * Build: included automatically by lib/CMakeLists.txt (see RAC_BLAS option).
 */

#ifndef RAC_BLAS_H
#define RAC_BLAS_H

#include <stddef.h>
#include "rac_cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Enums (CBLAS-compatible numeric values) ─────────────────────────────── */

typedef enum {
    RAC_BLAS_NO_TRANS = 111,  /* CblasNoTrans */
    RAC_BLAS_TRANS    = 112,  /* CblasTrans   */
} rac_blas_op;

typedef enum {
    RAC_BLAS_UPPER = 121,     /* CblasUpper */
    RAC_BLAS_LOWER = 122,     /* CblasLower */
} rac_blas_uplo;

typedef enum {
    RAC_BLAS_NON_UNIT = 131,  /* CblasNonUnit */
    RAC_BLAS_UNIT     = 132,  /* CblasUnit    */
} rac_blas_diag;

typedef enum {
    RAC_BLAS_LEFT  = 141,     /* CblasLeft  */
    RAC_BLAS_RIGHT = 142,     /* CblasRight */
} rac_blas_side;

/* ── Level 1 BLAS ────────────────────────────────────────────────────────── */
/*
 * Vector-vector operations. All `incx` / `incy` parameters are signed int
 * strides in float elements (negative strides not supported — callers
 * should reverse the buffer themselves; matches the most common usage).
 */

/* y := alpha * x + y */
void rac_saxpy(int n, float alpha,
               const float *x, int incx,
               float *y, int incy);

/* dot := sum_i x_i * y_i */
float rac_sdot(int n,
               const float *x, int incx,
               const float *y, int incy);

/* nrm2 := sqrt(sum_i x_i^2) */
float rac_snrm2(int n, const float *x, int incx);

/* asum := sum_i |x_i| */
float rac_sasum(int n, const float *x, int incx);

/* return index of element with max |x_i| (0-based, BLAS reference is 1-based). */
int   rac_isamax(int n, const float *x, int incx);

/* x := alpha * x */
void  rac_sscal(int n, float alpha, float *x, int incx);

/* y := x */
void  rac_scopy(int n,
                const float *x, int incx,
                float *y, int incy);

/* swap(x, y) */
void  rac_sswap(int n,
                float *x, int incx,
                float *y, int incy);

/* Apply Givens rotation:
 *   x_i := c*x_i + s*y_i
 *   y_i := c*y_i - s*x_i
 */
void  rac_srot(int n,
               float *x, int incx,
               float *y, int incy,
               float c, float s);

/* Construct Givens rotation [c s; -s c] s.t. [c s; -s c] [a; b] = [r; 0].
 * Inputs a, b are overwritten with r and z (BLAS reference behavior).
 * c, s are written to *c and *s. Implementation uses RAC vectoring
 * internally (rac_polar) which is the natural CORDIC formulation.
 */
void  rac_srotg(float *a, float *b, float *c, float *s);

/* ── Level 2 BLAS ────────────────────────────────────────────────────────── */
/*
 * Matrix-vector operations. Row-major. lda is the leading dimension
 * (row stride) of the matrix in elements.
 */

/*
 * y := alpha * op(A) * x + beta * y
 *   A is M x N row-major (lda >= N when no transpose, lda >= N otherwise).
 *   When trans == RAC_BLAS_NO_TRANS:  x is length N, y is length M.
 *   When trans == RAC_BLAS_TRANS:     x is length M, y is length N.
 */
rac_status rac_sgemv(rac_blas_op trans,
                     int M, int N,
                     float alpha,
                     const float *A, int lda,
                     const float *x, int incx,
                     float beta,
                     float *y, int incy);

/*
 * A := alpha * x * y^T + A    (rank-1 update, row-major)
 *   A is M x N (lda >= N), x length M (incx), y length N (incy).
 */
rac_status rac_sger(int M, int N,
                    float alpha,
                    const float *x, int incx,
                    const float *y, int incy,
                    float *A, int lda);

/*
 * y := alpha * A * x + beta * y, A is N x N symmetric.
 *   uplo selects which triangle of A is referenced.
 */
rac_status rac_ssymv(rac_blas_uplo uplo,
                     int N,
                     float alpha,
                     const float *A, int lda,
                     const float *x, int incx,
                     float beta,
                     float *y, int incy);

/*
 * A := alpha * x * x^T + A, symmetric rank-1 update (only `uplo` triangle).
 */
rac_status rac_ssyr(rac_blas_uplo uplo,
                    int N,
                    float alpha,
                    const float *x, int incx,
                    float *A, int lda);

/*
 * A := alpha * (x * y^T + y * x^T) + A, symmetric rank-2 update.
 */
rac_status rac_ssyr2(rac_blas_uplo uplo,
                     int N,
                     float alpha,
                     const float *x, int incx,
                     const float *y, int incy,
                     float *A, int lda);

/*
 * x := op(A) * x, A is N x N triangular.
 */
rac_status rac_strmv(rac_blas_uplo uplo,
                     rac_blas_op trans,
                     rac_blas_diag diag,
                     int N,
                     const float *A, int lda,
                     float *x, int incx);

/*
 * Solve op(A) * x = b in-place (b passed via x). A is N x N triangular,
 * non-singular (no checks; caller responsible).
 */
rac_status rac_strsv(rac_blas_uplo uplo,
                     rac_blas_op trans,
                     rac_blas_diag diag,
                     int N,
                     const float *A, int lda,
                     float *x, int incx);

/* ── Level 3 BLAS ────────────────────────────────────────────────────────── */
/*
 * Matrix-matrix operations. Row-major. lda/ldb/ldc are leading dimensions
 * (row strides).
 */

/*
 * C := alpha * op(A) * op(B) + beta * C
 *   op(A) is M x K, op(B) is K x N, C is M x N.
 *   For NoTrans+NoTrans this dispatches to rac_sgemm (the cache-tiled,
 *   OpenMP-parallel kernel in rac_cpu.c) for performance. Other transpose
 *   combinations use a row-major reference kernel.
 */
rac_status rac_sgemm_ex(rac_blas_op transA, rac_blas_op transB,
                        int M, int N, int K,
                        float alpha,
                        const float *A, int lda,
                        const float *B, int ldb,
                        float beta,
                        float *C, int ldc);

/*
 * C := alpha * A * B + beta * C    (side == LEFT,  A is M x M symmetric)
 * C := alpha * B * A + beta * C    (side == RIGHT, A is N x N symmetric)
 */
rac_status rac_ssymm(rac_blas_side side, rac_blas_uplo uplo,
                     int M, int N,
                     float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float beta,
                     float *C, int ldc);

/*
 * C := alpha * op(A) * op(A)^T + beta * C, C is N x N symmetric.
 *   op(A) is N x K (NoTrans) or K x N (Trans).
 *   Only `uplo` triangle of C is updated.
 */
rac_status rac_ssyrk(rac_blas_uplo uplo, rac_blas_op trans,
                     int N, int K,
                     float alpha,
                     const float *A, int lda,
                     float beta,
                     float *C, int ldc);

/*
 * C := alpha * (op(A) * op(B)^T + op(B) * op(A)^T) + beta * C
 *   Symmetric rank-2K update.
 */
rac_status rac_ssyr2k(rac_blas_uplo uplo, rac_blas_op trans,
                      int N, int K,
                      float alpha,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float beta,
                      float *C, int ldc);

/*
 * B := alpha * op(A) * B    (side == LEFT,  A is M x M triangular)
 * B := alpha * B * op(A)    (side == RIGHT, A is N x N triangular)
 */
rac_status rac_strmm(rac_blas_side side, rac_blas_uplo uplo,
                     rac_blas_op trans, rac_blas_diag diag,
                     int M, int N,
                     float alpha,
                     const float *A, int lda,
                     float *B, int ldb);

/*
 * Solve op(A) * X = alpha * B  (side == LEFT)
 *       X * op(A) = alpha * B  (side == RIGHT)
 *   Result overwrites B. A is triangular, non-singular.
 */
rac_status rac_strsm(rac_blas_side side, rac_blas_uplo uplo,
                     rac_blas_op trans, rac_blas_diag diag,
                     int M, int N,
                     float alpha,
                     const float *A, int lda,
                     float *B, int ldb);

#ifdef __cplusplus
}
#endif

#endif /* RAC_BLAS_H */
