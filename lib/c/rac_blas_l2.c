/* rac_blas_l2.c — RAC BLAS Level 2 (single precision, row-major). PQG / Michael A. Doran Jr. — April 2026 */

#include "rac_blas.h"
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── rac_sgemv ──────────────────────────────────────────────────────────────
 * y := alpha * op(A) * x + beta * y
 *   A is M x N row-major, lda is row stride.
 *   NoTrans: x has length N (incx), y has length M (incy).
 *   Trans:   x has length M (incx), y has length N (incy).
 */
rac_status rac_sgemv(rac_blas_op trans,
                     int M, int N,
                     float alpha,
                     const float *A, int lda,
                     const float *x, int incx,
                     float beta,
                     float *y, int incy)
{
    if (!A || !x || !y)              return RAC_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || lda < N) return RAC_ERR_INVALID_DIM;

    const int y_len = (trans == RAC_BLAS_NO_TRANS) ? M : N;
    const int x_len = (trans == RAC_BLAS_NO_TRANS) ? N : M;

    /* Apply beta to y first. */
    if (beta == 0.0f) {
        #pragma omp parallel for if(y_len > 256)
        for (int i = 0; i < y_len; ++i) y[i * incy] = 0.0f;
    } else if (beta != 1.0f) {
        #pragma omp parallel for if(y_len > 256)
        for (int i = 0; i < y_len; ++i) y[i * incy] *= beta;
    }

    if (trans == RAC_BLAS_NO_TRANS) {
        /* y[i] += alpha * sum_j A[i,j] * x[j] */
        #pragma omp parallel for if(M > 64)
        for (int i = 0; i < M; ++i) {
            const float *Ai = A + (size_t)i * (size_t)lda;
            float acc = 0.0f;
            for (int j = 0; j < N; ++j) {
                acc = fmaf(Ai[j], x[j * incx], acc);
            }
            y[i * incy] = fmaf(alpha, acc, y[i * incy]);
        }
    } else { /* RAC_BLAS_TRANS */
        /* y[j] += alpha * sum_i A[i,j] * x[i] */
        #pragma omp parallel for if(N > 64)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int i = 0; i < M; ++i) {
                acc = fmaf(A[(size_t)i * (size_t)lda + j], x[i * incx], acc);
            }
            y[j * incy] = fmaf(alpha, acc, y[j * incy]);
        }
    }
    (void)x_len;
    return RAC_OK;
}

/* ── rac_sger ───────────────────────────────────────────────────────────────
 * A := alpha * x * y^T + A   (rank-1 update, row-major)
 *   A is M x N, x length M, y length N.
 */
rac_status rac_sger(int M, int N,
                    float alpha,
                    const float *x, int incx,
                    const float *y, int incy,
                    float *A, int lda)
{
    if (!A || !x || !y)              return RAC_ERR_NULL_PTR;
    if (M <= 0 || N <= 0 || lda < N) return RAC_ERR_INVALID_DIM;
    if (alpha == 0.0f) return RAC_OK;

    #pragma omp parallel for if(M > 64)
    for (int i = 0; i < M; ++i) {
        const float ax = alpha * x[i * incx];
        float *Ai = A + (size_t)i * (size_t)lda;
        for (int j = 0; j < N; ++j) {
            Ai[j] = fmaf(ax, y[j * incy], Ai[j]);
        }
    }
    return RAC_OK;
}

/* ── rac_ssymv ──────────────────────────────────────────────────────────────
 * y := alpha * A * x + beta * y, A is N x N symmetric (only `uplo` stored).
 *   For (i,j) outside the stored triangle, A_full[i,j] = A[j,i].
 */
rac_status rac_ssymv(rac_blas_uplo uplo,
                     int N,
                     float alpha,
                     const float *A, int lda,
                     const float *x, int incx,
                     float beta,
                     float *y, int incy)
{
    if (!A || !x || !y)     return RAC_ERR_NULL_PTR;
    if (N <= 0 || lda < N)  return RAC_ERR_INVALID_DIM;

    /* Apply beta to y first. */
    if (beta == 0.0f) {
        #pragma omp parallel for if(N > 256)
        for (int i = 0; i < N; ++i) y[i * incy] = 0.0f;
    } else if (beta != 1.0f) {
        #pragma omp parallel for if(N > 256)
        for (int i = 0; i < N; ++i) y[i * incy] *= beta;
    }

    if (uplo == RAC_BLAS_UPPER) {
        /* Stored triangle: j >= i. For j < i, use A[j,i]. */
        #pragma omp parallel for if(N > 64)
        for (int i = 0; i < N; ++i) {
            float acc = 0.0f;
            /* j < i: read transposed slot A[j*lda + i] */
            for (int j = 0; j < i; ++j) {
                acc = fmaf(A[(size_t)j * (size_t)lda + i], x[j * incx], acc);
            }
            /* j >= i: stored row */
            const float *Ai = A + (size_t)i * (size_t)lda;
            for (int j = i; j < N; ++j) {
                acc = fmaf(Ai[j], x[j * incx], acc);
            }
            y[i * incy] = fmaf(alpha, acc, y[i * incy]);
        }
    } else { /* RAC_BLAS_LOWER */
        /* Stored triangle: j <= i. For j > i, use A[j,i]. */
        #pragma omp parallel for if(N > 64)
        for (int i = 0; i < N; ++i) {
            float acc = 0.0f;
            const float *Ai = A + (size_t)i * (size_t)lda;
            /* j <= i: stored row */
            for (int j = 0; j <= i; ++j) {
                acc = fmaf(Ai[j], x[j * incx], acc);
            }
            /* j > i: read transposed slot A[j*lda + i] */
            for (int j = i + 1; j < N; ++j) {
                acc = fmaf(A[(size_t)j * (size_t)lda + i], x[j * incx], acc);
            }
            y[i * incy] = fmaf(alpha, acc, y[i * incy]);
        }
    }
    return RAC_OK;
}

/* ── rac_ssyr ───────────────────────────────────────────────────────────────
 * A := alpha * x * x^T + A, only `uplo` triangle (including diagonal) updated.
 */
rac_status rac_ssyr(rac_blas_uplo uplo,
                    int N,
                    float alpha,
                    const float *x, int incx,
                    float *A, int lda)
{
    if (!A || !x)           return RAC_ERR_NULL_PTR;
    if (N <= 0 || lda < N)  return RAC_ERR_INVALID_DIM;
    if (alpha == 0.0f) return RAC_OK;

    if (uplo == RAC_BLAS_UPPER) {
        #pragma omp parallel for if(N > 64)
        for (int i = 0; i < N; ++i) {
            const float ax = alpha * x[i * incx];
            float *Ai = A + (size_t)i * (size_t)lda;
            for (int j = i; j < N; ++j) {
                Ai[j] = fmaf(ax, x[j * incx], Ai[j]);
            }
        }
    } else { /* RAC_BLAS_LOWER */
        #pragma omp parallel for if(N > 64)
        for (int i = 0; i < N; ++i) {
            const float ax = alpha * x[i * incx];
            float *Ai = A + (size_t)i * (size_t)lda;
            for (int j = 0; j <= i; ++j) {
                Ai[j] = fmaf(ax, x[j * incx], Ai[j]);
            }
        }
    }
    return RAC_OK;
}

/* ── rac_ssyr2 ──────────────────────────────────────────────────────────────
 * A := alpha * (x*y^T + y*x^T) + A, only `uplo` triangle updated.
 */
rac_status rac_ssyr2(rac_blas_uplo uplo,
                     int N,
                     float alpha,
                     const float *x, int incx,
                     const float *y, int incy,
                     float *A, int lda)
{
    if (!A || !x || !y)     return RAC_ERR_NULL_PTR;
    if (N <= 0 || lda < N)  return RAC_ERR_INVALID_DIM;
    if (alpha == 0.0f) return RAC_OK;

    if (uplo == RAC_BLAS_UPPER) {
        #pragma omp parallel for if(N > 64)
        for (int i = 0; i < N; ++i) {
            const float ax = alpha * x[i * incx];
            const float ay = alpha * y[i * incy];
            float *Ai = A + (size_t)i * (size_t)lda;
            for (int j = i; j < N; ++j) {
                float t = fmaf(ax, y[j * incy], Ai[j]);
                Ai[j]   = fmaf(ay, x[j * incx], t);
            }
        }
    } else { /* RAC_BLAS_LOWER */
        #pragma omp parallel for if(N > 64)
        for (int i = 0; i < N; ++i) {
            const float ax = alpha * x[i * incx];
            const float ay = alpha * y[i * incy];
            float *Ai = A + (size_t)i * (size_t)lda;
            for (int j = 0; j <= i; ++j) {
                float t = fmaf(ax, y[j * incy], Ai[j]);
                Ai[j]   = fmaf(ay, x[j * incx], t);
            }
        }
    }
    return RAC_OK;
}

/* ── rac_strmv ──────────────────────────────────────────────────────────────
 * x := op(A) * x, A is N x N triangular (`uplo`), optional unit diagonal.
 *   In-place ordering matters; serial.
 */
rac_status rac_strmv(rac_blas_uplo uplo,
                     rac_blas_op trans,
                     rac_blas_diag diag,
                     int N,
                     const float *A, int lda,
                     float *x, int incx)
{
    if (!A || !x)           return RAC_ERR_NULL_PTR;
    if (N <= 0 || lda < N)  return RAC_ERR_INVALID_DIM;

    const int unit = (diag == RAC_BLAS_UNIT);

    if (trans == RAC_BLAS_NO_TRANS) {
        if (uplo == RAC_BLAS_LOWER) {
            /* result_i = sum_{j<=i} A[i,j] * x[j], with A[i,i]=1 if Unit.
             * Compute in DECREASING i so we don't overwrite x[j] (j<i) needed. */
            for (int i = N - 1; i >= 0; --i) {
                const float *Ai = A + (size_t)i * (size_t)lda;
                float acc = unit ? x[i * incx] : Ai[i] * x[i * incx];
                for (int j = 0; j < i; ++j) {
                    acc = fmaf(Ai[j], x[j * incx], acc);
                }
                x[i * incx] = acc;
            }
        } else { /* UPPER */
            /* result_i = sum_{j>=i} A[i,j] * x[j]. Compute INCREASING i. */
            for (int i = 0; i < N; ++i) {
                const float *Ai = A + (size_t)i * (size_t)lda;
                float acc = unit ? x[i * incx] : Ai[i] * x[i * incx];
                for (int j = i + 1; j < N; ++j) {
                    acc = fmaf(Ai[j], x[j * incx], acc);
                }
                x[i * incx] = acc;
            }
        }
    } else { /* TRANS */
        if (uplo == RAC_BLAS_LOWER) {
            /* A^T is upper. result_j = sum_{i>=j} A[i,j] * x[i].
             * Compute DECREASING j: x[j] is overwritten using x[i] with i>j, still untouched. */
            for (int j = N - 1; j >= 0; --j) {
                float acc = unit ? x[j * incx]
                                 : A[(size_t)j * (size_t)lda + j] * x[j * incx];
                for (int i = j + 1; i < N; ++i) {
                    acc = fmaf(A[(size_t)i * (size_t)lda + j], x[i * incx], acc);
                }
                x[j * incx] = acc;
            }
        } else { /* UPPER */
            /* A^T is lower. result_j = sum_{i<=j} A[i,j] * x[i].
             * Walk j DECREASING: x[i] for i<j is still original at this point. */
            for (int j = N - 1; j >= 0; --j) {
                float acc = unit ? x[j * incx]
                                 : A[(size_t)j * (size_t)lda + j] * x[j * incx];
                for (int i = 0; i < j; ++i) {
                    acc = fmaf(A[(size_t)i * (size_t)lda + j], x[i * incx], acc);
                }
                x[j * incx] = acc;
            }
        }
    }
    return RAC_OK;
}

/* ── rac_strsv ──────────────────────────────────────────────────────────────
 * Solve op(A) * x = b in place (b passed via x). A triangular non-singular.
 */
rac_status rac_strsv(rac_blas_uplo uplo,
                     rac_blas_op trans,
                     rac_blas_diag diag,
                     int N,
                     const float *A, int lda,
                     float *x, int incx)
{
    if (!A || !x)           return RAC_ERR_NULL_PTR;
    if (N <= 0 || lda < N)  return RAC_ERR_INVALID_DIM;

    const int unit = (diag == RAC_BLAS_UNIT);

    if (trans == RAC_BLAS_NO_TRANS) {
        if (uplo == RAC_BLAS_LOWER) {
            /* Forward substitution: x[i] = (x[i] - sum_{j<i} A[i,j]*x[j]) / A[i,i]. */
            for (int i = 0; i < N; ++i) {
                const float *Ai = A + (size_t)i * (size_t)lda;
                float s = x[i * incx];
                for (int j = 0; j < i; ++j) {
                    s = fmaf(-Ai[j], x[j * incx], s);
                }
                x[i * incx] = unit ? s : (s / Ai[i]);
            }
        } else { /* UPPER */
            /* Back substitution: x[i] = (x[i] - sum_{j>i} A[i,j]*x[j]) / A[i,i]. */
            for (int i = N - 1; i >= 0; --i) {
                const float *Ai = A + (size_t)i * (size_t)lda;
                float s = x[i * incx];
                for (int j = i + 1; j < N; ++j) {
                    s = fmaf(-Ai[j], x[j * incx], s);
                }
                x[i * incx] = unit ? s : (s / Ai[i]);
            }
        }
    } else { /* TRANS */
        if (uplo == RAC_BLAS_LOWER) {
            /* A^T is upper → back substitution on j.
             * x[j] = (x[j] - sum_{i>j} A[i,j]*x[i]) / A[j,j]. */
            for (int j = N - 1; j >= 0; --j) {
                float s = x[j * incx];
                for (int i = j + 1; i < N; ++i) {
                    s = fmaf(-A[(size_t)i * (size_t)lda + j], x[i * incx], s);
                }
                x[j * incx] = unit ? s : (s / A[(size_t)j * (size_t)lda + j]);
            }
        } else { /* UPPER */
            /* A^T is lower → forward substitution on j.
             * x[j] = (x[j] - sum_{i<j} A[i,j]*x[i]) / A[j,j]. */
            for (int j = 0; j < N; ++j) {
                float s = x[j * incx];
                for (int i = 0; i < j; ++i) {
                    s = fmaf(-A[(size_t)i * (size_t)lda + j], x[i * incx], s);
                }
                x[j * incx] = unit ? s : (s / A[(size_t)j * (size_t)lda + j]);
            }
        }
    }
    return RAC_OK;
}
