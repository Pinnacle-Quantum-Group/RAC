/* rac_blas_l3.c — RAC BLAS Level 3 (single precision, row-major). PQG / Michael A. Doran Jr. — April 2026 */

#include "rac_blas.h"
#include "rac_cpu.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Internal helpers ────────────────────────────────────────────────────── */

/* Scale C (M x N row-major, ldc stride) by beta. Handles beta == 0 specially. */
static void l3_scale_C(float *C, int M, int N, int ldc, float beta)
{
    if (beta == 1.0f) return;
    if (beta == 0.0f) {
        if (ldc == N) {
            memset(C, 0, (size_t)M * (size_t)N * sizeof(float));
        } else {
            for (int i = 0; i < M; ++i) {
                memset(&C[(size_t)i * ldc], 0, (size_t)N * sizeof(float));
            }
        }
        return;
    }
    for (int i = 0; i < M; ++i) {
        float *row = &C[(size_t)i * ldc];
        for (int j = 0; j < N; ++j) {
            row[j] *= beta;
        }
    }
}

/* Symmetric-matrix accessor. A is N x N stored in `uplo` triangle (with lda
 * row stride). Returns the logical full element A_full[i,j]. */
static inline float l3_sym_get(const float *A, int lda,
                               rac_blas_uplo uplo, int i, int j)
{
    if (uplo == RAC_BLAS_UPPER) {
        if (i <= j) return A[(size_t)i * lda + j];
        return A[(size_t)j * lda + i];
    } else {
        if (i >= j) return A[(size_t)i * lda + j];
        return A[(size_t)j * lda + i];
    }
}

/* Triangular-matrix accessor with unit-diagonal handling.
 * Returns 0 for entries outside the stored triangle, 1 on the diagonal when
 * diag == UNIT, otherwise the stored value. */
static inline float l3_tri_get(const float *A, int lda,
                               rac_blas_uplo uplo, rac_blas_diag diag,
                               int i, int j)
{
    if (i == j) {
        return (diag == RAC_BLAS_UNIT) ? 1.0f : A[(size_t)i * lda + j];
    }
    if (uplo == RAC_BLAS_UPPER) {
        return (i < j) ? A[(size_t)i * lda + j] : 0.0f;
    } else {
        return (i > j) ? A[(size_t)i * lda + j] : 0.0f;
    }
}

/* Stack/heap small-buffer helper. */
#define L3_SMALL_STACK 4096

/* ── rac_sgemm_ex ────────────────────────────────────────────────────────── */

rac_status rac_sgemm_ex(rac_blas_op transA, rac_blas_op transB,
                        int M, int N, int K,
                        float alpha,
                        const float *A, int lda,
                        const float *B, int ldb,
                        float beta,
                        float *C, int ldc)
{
    if (!A || !B || !C) return RAC_ERR_NULL_PTR;
    if (M < 0 || N < 0 || K < 0) return RAC_ERR_INVALID_DIM;
    if (M == 0 || N == 0) return RAC_OK;

    /* Fast path: NoTrans + contiguous strides → cache-tiled rac_sgemm. */
    if (transA == RAC_BLAS_NO_TRANS && transB == RAC_BLAS_NO_TRANS &&
        lda == K && ldb == N && ldc == N) {
        return rac_sgemm(A, B, C, M, N, K, alpha, beta, NULL);
    }

    /* Apply beta to C. */
    l3_scale_C(C, M, N, ldc, beta);

    if (K == 0 || alpha == 0.0f) return RAC_OK;

    const int tA = (transA == RAC_BLAS_NO_TRANS) ? 0 : 1;
    const int tB = (transB == RAC_BLAS_NO_TRANS) ? 0 : 1;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; ++i) {
        float *Ci = &C[(size_t)i * ldc];
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = tA ? A[(size_t)k * lda + i] : A[(size_t)i * lda + k];
                float b = tB ? B[(size_t)j * ldb + k] : B[(size_t)k * ldb + j];
                acc = fmaf(a, b, acc);
            }
            Ci[j] = fmaf(alpha, acc, Ci[j]);
        }
    }
    return RAC_OK;
}

/* ── rac_ssymm ───────────────────────────────────────────────────────────── */

rac_status rac_ssymm(rac_blas_side side, rac_blas_uplo uplo,
                     int M, int N,
                     float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float beta,
                     float *C, int ldc)
{
    if (!A || !B || !C) return RAC_ERR_NULL_PTR;
    if (M < 0 || N < 0) return RAC_ERR_INVALID_DIM;
    if (M == 0 || N == 0) return RAC_OK;

    l3_scale_C(C, M, N, ldc, beta);

    if (alpha == 0.0f) return RAC_OK;

    if (side == RAC_BLAS_LEFT) {
        /* C[i,j] += alpha * sum_k A_full(i,k) * B[k,j], A is M x M. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; ++i) {
            float *Ci = &C[(size_t)i * ldc];
            for (int j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < M; ++k) {
                    float a = l3_sym_get(A, lda, uplo, i, k);
                    float b = B[(size_t)k * ldb + j];
                    acc = fmaf(a, b, acc);
                }
                Ci[j] = fmaf(alpha, acc, Ci[j]);
            }
        }
    } else {
        /* RIGHT: C[i,j] += alpha * sum_k B[i,k] * A_full(k,j), A is N x N. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; ++i) {
            float *Ci = &C[(size_t)i * ldc];
            const float *Bi = &B[(size_t)i * ldb];
            for (int j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < N; ++k) {
                    float a = l3_sym_get(A, lda, uplo, k, j);
                    acc = fmaf(Bi[k], a, acc);
                }
                Ci[j] = fmaf(alpha, acc, Ci[j]);
            }
        }
    }
    return RAC_OK;
}

/* ── rac_ssyrk ───────────────────────────────────────────────────────────── */

rac_status rac_ssyrk(rac_blas_uplo uplo, rac_blas_op trans,
                     int N, int K,
                     float alpha,
                     const float *A, int lda,
                     float beta,
                     float *C, int ldc)
{
    if (!A || !C) return RAC_ERR_NULL_PTR;
    if (N < 0 || K < 0) return RAC_ERR_INVALID_DIM;
    if (N == 0) return RAC_OK;

    /* Scale only the referenced triangle of C by beta. */
    if (beta != 1.0f) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < N; ++i) {
            float *Ci = &C[(size_t)i * ldc];
            int j0 = (uplo == RAC_BLAS_UPPER) ? i : 0;
            int j1 = (uplo == RAC_BLAS_UPPER) ? N : (i + 1);
            if (beta == 0.0f) {
                for (int j = j0; j < j1; ++j) Ci[j] = 0.0f;
            } else {
                for (int j = j0; j < j1; ++j) Ci[j] *= beta;
            }
        }
    }

    if (K == 0 || alpha == 0.0f) return RAC_OK;

    const int tA = (trans == RAC_BLAS_NO_TRANS) ? 0 : 1;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        float *Ci = &C[(size_t)i * ldc];
        int j0 = (uplo == RAC_BLAS_UPPER) ? i : 0;
        int j1 = (uplo == RAC_BLAS_UPPER) ? N : (i + 1);
        for (int j = j0; j < j1; ++j) {
            float acc = 0.0f;
            if (!tA) {
                /* A is N x K: C[i,j] = alpha * sum_k A[i,k] * A[j,k] */
                const float *Ai = &A[(size_t)i * lda];
                const float *Aj = &A[(size_t)j * lda];
                for (int k = 0; k < K; ++k) {
                    acc = fmaf(Ai[k], Aj[k], acc);
                }
            } else {
                /* A is K x N: C[i,j] = alpha * sum_k A[k,i] * A[k,j] */
                for (int k = 0; k < K; ++k) {
                    float aki = A[(size_t)k * lda + i];
                    float akj = A[(size_t)k * lda + j];
                    acc = fmaf(aki, akj, acc);
                }
            }
            Ci[j] = fmaf(alpha, acc, Ci[j]);
        }
    }
    return RAC_OK;
}

/* ── rac_ssyr2k ──────────────────────────────────────────────────────────── */

rac_status rac_ssyr2k(rac_blas_uplo uplo, rac_blas_op trans,
                      int N, int K,
                      float alpha,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float beta,
                      float *C, int ldc)
{
    if (!A || !B || !C) return RAC_ERR_NULL_PTR;
    if (N < 0 || K < 0) return RAC_ERR_INVALID_DIM;
    if (N == 0) return RAC_OK;

    if (beta != 1.0f) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < N; ++i) {
            float *Ci = &C[(size_t)i * ldc];
            int j0 = (uplo == RAC_BLAS_UPPER) ? i : 0;
            int j1 = (uplo == RAC_BLAS_UPPER) ? N : (i + 1);
            if (beta == 0.0f) {
                for (int j = j0; j < j1; ++j) Ci[j] = 0.0f;
            } else {
                for (int j = j0; j < j1; ++j) Ci[j] *= beta;
            }
        }
    }

    if (K == 0 || alpha == 0.0f) return RAC_OK;

    const int tA = (trans == RAC_BLAS_NO_TRANS) ? 0 : 1;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        float *Ci = &C[(size_t)i * ldc];
        int j0 = (uplo == RAC_BLAS_UPPER) ? i : 0;
        int j1 = (uplo == RAC_BLAS_UPPER) ? N : (i + 1);
        for (int j = j0; j < j1; ++j) {
            float acc = 0.0f;
            if (!tA) {
                const float *Ai = &A[(size_t)i * lda];
                const float *Aj = &A[(size_t)j * lda];
                const float *Bi = &B[(size_t)i * ldb];
                const float *Bj = &B[(size_t)j * ldb];
                for (int k = 0; k < K; ++k) {
                    acc = fmaf(Ai[k], Bj[k], acc);
                    acc = fmaf(Bi[k], Aj[k], acc);
                }
            } else {
                for (int k = 0; k < K; ++k) {
                    float aki = A[(size_t)k * lda + i];
                    float akj = A[(size_t)k * lda + j];
                    float bki = B[(size_t)k * ldb + i];
                    float bkj = B[(size_t)k * ldb + j];
                    acc = fmaf(aki, bkj, acc);
                    acc = fmaf(bki, akj, acc);
                }
            }
            Ci[j] = fmaf(alpha, acc, Ci[j]);
        }
    }
    return RAC_OK;
}

/* ── rac_strmm ───────────────────────────────────────────────────────────── */

/* Compute one column of (op(A) * B) into tmp[0..M-1], B's column j gathered
 * on the fly. A is M x M triangular. Uses l3_tri_get (op-aware). */
static void l3_trmm_left_col(rac_blas_uplo uplo, rac_blas_op trans,
                             rac_blas_diag diag,
                             int M,
                             const float *A, int lda,
                             const float *B, int ldb,
                             int j, float *tmp)
{
    /* op(A)[i,k] = (trans==NoTrans) ? l3_tri_get(uplo, i, k)
     *                               : l3_tri_get(opposite uplo, k, i)
     * For NoTrans use l3_tri_get(uplo, i, k); for Trans, swap indices and
     * flip uplo so that the original triangle is the one referenced. */
    for (int i = 0; i < M; ++i) {
        float acc = 0.0f;
        for (int k = 0; k < M; ++k) {
            float a;
            if (trans == RAC_BLAS_NO_TRANS) {
                a = l3_tri_get(A, lda, uplo, diag, i, k);
            } else {
                /* op(A)[i,k] = A[k,i]. A[k,i] is a stored entry iff (k,i)
                 * lies in `uplo`. */
                if (i == k) {
                    a = (diag == RAC_BLAS_UNIT)
                            ? 1.0f
                            : A[(size_t)i * lda + i];
                } else if (uplo == RAC_BLAS_UPPER) {
                    /* (k,i) upper iff k <= i, so k < i for off-diag. */
                    a = (k < i) ? A[(size_t)k * lda + i] : 0.0f;
                } else {
                    a = (k > i) ? A[(size_t)k * lda + i] : 0.0f;
                }
            }
            acc = fmaf(a, B[(size_t)k * ldb + j], acc);
        }
        tmp[i] = acc;
    }
}

/* Compute one row of (B * op(A)) into tmp[0..N-1]. B is M x N, A is N x N. */
static void l3_trmm_right_row(rac_blas_uplo uplo, rac_blas_op trans,
                              rac_blas_diag diag,
                              int N,
                              const float *A, int lda,
                              const float *B, int ldb,
                              int i, float *tmp)
{
    const float *Bi = &B[(size_t)i * ldb];
    for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) {
            float a;
            if (trans == RAC_BLAS_NO_TRANS) {
                a = l3_tri_get(A, lda, uplo, diag, k, j);
            } else {
                /* op(A)[k,j] = A[j,k] — stored iff (j,k) in uplo. */
                if (k == j) {
                    a = (diag == RAC_BLAS_UNIT)
                            ? 1.0f
                            : A[(size_t)j * lda + j];
                } else if (uplo == RAC_BLAS_UPPER) {
                    a = (k > j) ? A[(size_t)j * lda + k] : 0.0f;
                } else {
                    a = (k < j) ? A[(size_t)j * lda + k] : 0.0f;
                }
            }
            acc = fmaf(Bi[k], a, acc);
        }
        tmp[j] = acc;
    }
}

rac_status rac_strmm(rac_blas_side side, rac_blas_uplo uplo,
                     rac_blas_op trans, rac_blas_diag diag,
                     int M, int N,
                     float alpha,
                     const float *A, int lda,
                     float *B, int ldb)
{
    if (!A || !B) return RAC_ERR_NULL_PTR;
    if (M < 0 || N < 0) return RAC_ERR_INVALID_DIM;
    if (M == 0 || N == 0) return RAC_OK;

    if (alpha == 0.0f) {
        for (int i = 0; i < M; ++i) {
            float *Bi = &B[(size_t)i * ldb];
            for (int j = 0; j < N; ++j) Bi[j] = 0.0f;
        }
        return RAC_OK;
    }

    if (side == RAC_BLAS_LEFT) {
        /* B := alpha * op(A) * B, parallelize over columns j. */
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            float stack_tmp[L3_SMALL_STACK];
            float *tmp;
            int heap = 0;
            if (M <= L3_SMALL_STACK) {
                tmp = stack_tmp;
            } else {
                tmp = (float *)malloc((size_t)M * sizeof(float));
                heap = 1;
            }
            if (tmp) {
#ifdef _OPENMP
                #pragma omp for schedule(static)
#endif
                for (int j = 0; j < N; ++j) {
                    l3_trmm_left_col(uplo, trans, diag, M,
                                     A, lda, B, ldb, j, tmp);
                    for (int i = 0; i < M; ++i) {
                        B[(size_t)i * ldb + j] = alpha * tmp[i];
                    }
                }
                if (heap) free(tmp);
            }
        }
    } else {
        /* B := alpha * B * op(A), parallelize over rows i. */
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            float stack_tmp[L3_SMALL_STACK];
            float *tmp;
            int heap = 0;
            if (N <= L3_SMALL_STACK) {
                tmp = stack_tmp;
            } else {
                tmp = (float *)malloc((size_t)N * sizeof(float));
                heap = 1;
            }
            if (tmp) {
#ifdef _OPENMP
                #pragma omp for schedule(static)
#endif
                for (int i = 0; i < M; ++i) {
                    l3_trmm_right_row(uplo, trans, diag, N,
                                      A, lda, B, ldb, i, tmp);
                    float *Bi = &B[(size_t)i * ldb];
                    for (int j = 0; j < N; ++j) {
                        Bi[j] = alpha * tmp[j];
                    }
                }
                if (heap) free(tmp);
            }
        }
    }

    return RAC_OK;
}

/* ── rac_strsm ───────────────────────────────────────────────────────────── */

/* Forward declare rac_strsv (defined in rac_blas_l2.c). The header already
 * declares it, but this comment documents the cross-TU dependency. */

rac_status rac_strsm(rac_blas_side side, rac_blas_uplo uplo,
                     rac_blas_op trans, rac_blas_diag diag,
                     int M, int N,
                     float alpha,
                     const float *A, int lda,
                     float *B, int ldb)
{
    if (!A || !B) return RAC_ERR_NULL_PTR;
    if (M < 0 || N < 0) return RAC_ERR_INVALID_DIM;
    if (M == 0 || N == 0) return RAC_OK;

    /* Scale B by alpha. */
    if (alpha != 1.0f) {
        if (alpha == 0.0f) {
            for (int i = 0; i < M; ++i) {
                float *Bi = &B[(size_t)i * ldb];
                for (int j = 0; j < N; ++j) Bi[j] = 0.0f;
            }
            return RAC_OK;
        }
        for (int i = 0; i < M; ++i) {
            float *Bi = &B[(size_t)i * ldb];
            for (int j = 0; j < N; ++j) Bi[j] *= alpha;
        }
    }

    if (side == RAC_BLAS_LEFT) {
        /* Solve op(A) * X = B, A is M x M. For each column j of B, gather
         * into temp of length M, run rac_strsv (incx=1), scatter back. */
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            float stack_tmp[L3_SMALL_STACK];
            float *tmp;
            int heap = 0;
            if (M <= L3_SMALL_STACK) {
                tmp = stack_tmp;
            } else {
                tmp = (float *)malloc((size_t)M * sizeof(float));
                heap = 1;
            }
            if (tmp) {
#ifdef _OPENMP
                #pragma omp for schedule(static)
#endif
                for (int j = 0; j < N; ++j) {
                    for (int i = 0; i < M; ++i) {
                        tmp[i] = B[(size_t)i * ldb + j];
                    }
                    (void)rac_strsv(uplo, trans, diag, M, A, lda, tmp, 1);
                    for (int i = 0; i < M; ++i) {
                        B[(size_t)i * ldb + j] = tmp[i];
                    }
                }
                if (heap) free(tmp);
            }
        }
    } else {
        /* Solve X * op(A) = B, A is N x N. For each row i of B (which is
         * already contiguous with incx=1), call rac_strsv with the
         * OPPOSITE op flag. Reasoning:
         *   X * op(A) = B  <=>  op(A)^T * X^T = B^T
         * Per-row that is op(A)^T x_row = b_row. Flipping the trans flag
         * achieves this; uplo stays unchanged because the stored triangle
         * of A is the SAME triangle of A^T's transpose-of-stored view —
         * but operationally rac_strsv reinterprets A according to (uplo,
         * trans), and for it to behave as op(A)^T we just flip trans. */
        rac_blas_op flipped = (trans == RAC_BLAS_NO_TRANS)
                                  ? RAC_BLAS_TRANS
                                  : RAC_BLAS_NO_TRANS;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; ++i) {
            float *Bi = &B[(size_t)i * ldb];
            (void)rac_strsv(uplo, flipped, diag, N, A, lda, Bi, 1);
        }
    }

    return RAC_OK;
}
