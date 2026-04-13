/* test_rac_blas_l3.c — BVT for rac_blas Level 3.
 * PQG / Michael A. Doran Jr. — April 2026 */

#include "rac_blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("PASS: %s\n", name); g_pass++; } \
    else { printf("FAIL: %s\n", name); g_fail++; } } while(0)

#define TOL 1e-3f

static float frand_det(int seed) {
    unsigned s = (unsigned)seed * 2654435761u + 1u;
    s = s * 1664525u + 1013904223u;
    return ((float)((s >> 8) & 0xFFFFFF) / (float)(1u << 24)) * 2.0f - 1.0f;
}
static void fill_random(float *a, int n, int seed) {
    for (int i = 0; i < n; i++) a[i] = frand_det(seed + i);
}
static float maxabs_diff(const float *a, const float *b, int n) {
    float m = 0; for (int i=0;i<n;i++) { float d=fabsf(a[i]-b[i]); if(d>m)m=d; }
    return m;
}

/* ------------------------------------------------------------------ */
/* Test 1: SGEMM NoTrans x NoTrans                                    */
/* ------------------------------------------------------------------ */
static void test_sgemm_NN(void) {
    const int M = 16, N = 24, K = 20;
    float *A      = malloc(sizeof(float) * M * K);
    float *B      = malloc(sizeof(float) * K * N);
    float *C      = malloc(sizeof(float) * M * N);
    float *C_ref  = malloc(sizeof(float) * M * N);
    float *C_orig = malloc(sizeof(float) * M * N);

    fill_random(A, M * K, 1001);
    fill_random(B, K * N, 2002);
    fill_random(C_orig, M * N, 3003);
    memcpy(C,     C_orig, sizeof(float) * M * N);
    memcpy(C_ref, C_orig, sizeof(float) * M * N);

    const float alpha = 1.3f, beta = 0.7f;

    /* Reference: naive row-major triple loop */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[i * K + k] * B[k * N + j];
            C_ref[i * N + j] = beta * C_orig[i * N + j] + alpha * acc;
        }
    }

    rac_sgemm_ex(RAC_BLAS_NO_TRANS, RAC_BLAS_NO_TRANS,
                 M, N, K,
                 alpha, A, K, B, N,
                 beta, C, N);

    float err = maxabs_diff(C, C_ref, M * N);
    CHECK(err < TOL, "sgemm NN");
    if (err >= TOL) printf("  max err = %g\n", err);

    free(A); free(B); free(C); free(C_ref); free(C_orig);
}

/* ------------------------------------------------------------------ */
/* Test 2: SGEMM Trans x NoTrans                                      */
/* A is stored as K x M (lda = M). op(A) is M x K.                    */
/* ------------------------------------------------------------------ */
static void test_sgemm_TN(void) {
    const int M = 16, N = 24, K = 20;
    float *A      = malloc(sizeof(float) * K * M); /* K x M storage */
    float *B      = malloc(sizeof(float) * K * N);
    float *C      = malloc(sizeof(float) * M * N);
    float *C_ref  = malloc(sizeof(float) * M * N);
    float *C_orig = malloc(sizeof(float) * M * N);

    fill_random(A, K * M, 1101);
    fill_random(B, K * N, 2102);
    fill_random(C_orig, M * N, 3103);
    memcpy(C,     C_orig, sizeof(float) * M * N);
    memcpy(C_ref, C_orig, sizeof(float) * M * N);

    const float alpha = 0.9f, beta = 1.1f;

    /* Reference: A_full[i,k] = A[k*M + i] */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[k * M + i] * B[k * N + j];
            C_ref[i * N + j] = beta * C_orig[i * N + j] + alpha * acc;
        }
    }

    rac_sgemm_ex(RAC_BLAS_TRANS, RAC_BLAS_NO_TRANS,
                 M, N, K,
                 alpha, A, M, B, N,
                 beta, C, N);

    float err = maxabs_diff(C, C_ref, M * N);
    CHECK(err < TOL, "sgemm TN");
    if (err >= TOL) printf("  max err = %g\n", err);

    free(A); free(B); free(C); free(C_ref); free(C_orig);
}

/* ------------------------------------------------------------------ */
/* Test 3: SGEMM NoTrans x Trans                                      */
/* B is stored as N x K (ldb = K). op(B) is K x N.                    */
/* ------------------------------------------------------------------ */
static void test_sgemm_NT(void) {
    const int M = 16, N = 24, K = 20;
    float *A      = malloc(sizeof(float) * M * K);
    float *B      = malloc(sizeof(float) * N * K); /* N x K storage */
    float *C      = malloc(sizeof(float) * M * N);
    float *C_ref  = malloc(sizeof(float) * M * N);
    float *C_orig = malloc(sizeof(float) * M * N);

    fill_random(A, M * K, 1201);
    fill_random(B, N * K, 2202);
    fill_random(C_orig, M * N, 3203);
    memcpy(C,     C_orig, sizeof(float) * M * N);
    memcpy(C_ref, C_orig, sizeof(float) * M * N);

    const float alpha = 1.0f, beta = 0.0f;

    /* Reference: B_full[k,j] = B[j*K + k] */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[i * K + k] * B[j * K + k];
            C_ref[i * N + j] = beta * C_orig[i * N + j] + alpha * acc;
        }
    }

    rac_sgemm_ex(RAC_BLAS_NO_TRANS, RAC_BLAS_TRANS,
                 M, N, K,
                 alpha, A, K, B, K,
                 beta, C, N);

    float err = maxabs_diff(C, C_ref, M * N);
    CHECK(err < TOL, "sgemm NT");
    if (err >= TOL) printf("  max err = %g\n", err);

    free(A); free(B); free(C); free(C_ref); free(C_orig);
}

/* ------------------------------------------------------------------ */
/* Test 4: SGEMM Trans x Trans                                        */
/* ------------------------------------------------------------------ */
static void test_sgemm_TT(void) {
    const int M = 16, N = 24, K = 20;
    float *A      = malloc(sizeof(float) * K * M); /* K x M */
    float *B      = malloc(sizeof(float) * N * K); /* N x K */
    float *C      = malloc(sizeof(float) * M * N);
    float *C_ref  = malloc(sizeof(float) * M * N);
    float *C_orig = malloc(sizeof(float) * M * N);

    fill_random(A, K * M, 1301);
    fill_random(B, N * K, 2302);
    fill_random(C_orig, M * N, 3303);
    memcpy(C,     C_orig, sizeof(float) * M * N);
    memcpy(C_ref, C_orig, sizeof(float) * M * N);

    const float alpha = 0.75f, beta = -0.5f;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[k * M + i] * B[j * K + k];
            C_ref[i * N + j] = beta * C_orig[i * N + j] + alpha * acc;
        }
    }

    rac_sgemm_ex(RAC_BLAS_TRANS, RAC_BLAS_TRANS,
                 M, N, K,
                 alpha, A, M, B, K,
                 beta, C, N);

    float err = maxabs_diff(C, C_ref, M * N);
    CHECK(err < TOL, "sgemm TT");
    if (err >= TOL) printf("  max err = %g\n", err);

    free(A); free(B); free(C); free(C_ref); free(C_orig);
}

/* ------------------------------------------------------------------ */
/* Test 5: SSYMM LEFT UPPER                                           */
/* C := alpha * A * B + beta * C, A is M x M symmetric (M==N here).   */
/* ------------------------------------------------------------------ */
static void test_ssymm_left_upper(void) {
    const int M = 12, N = 12;
    float *A      = malloc(sizeof(float) * M * M);
    float *A_full = malloc(sizeof(float) * M * M);
    float *B      = malloc(sizeof(float) * M * N);
    float *C      = malloc(sizeof(float) * M * N);
    float *C_ref  = malloc(sizeof(float) * M * N);
    float *C_orig = malloc(sizeof(float) * M * N);

    /* Fill A's upper triangle from random; set lower to sentinel -999. */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            if (i <= j) A[i * M + j] = frand_det(4001 + i * M + j);
            else        A[i * M + j] = -999.0f;
        }
    }
    /* Symmetric full matrix for reference */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            A_full[i * M + j] = (i <= j) ? A[i * M + j] : A[j * M + i];
        }
    }

    fill_random(B, M * N, 5002);
    fill_random(C_orig, M * N, 6003);
    memcpy(C,     C_orig, sizeof(float) * M * N);
    memcpy(C_ref, C_orig, sizeof(float) * M * N);

    const float alpha = 1.0f, beta = 0.5f;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < M; k++) acc += A_full[i * M + k] * B[k * N + j];
            C_ref[i * N + j] = beta * C_orig[i * N + j] + alpha * acc;
        }
    }

    rac_ssymm(RAC_BLAS_LEFT, RAC_BLAS_UPPER,
              M, N, alpha, A, M, B, N, beta, C, N);

    float err = maxabs_diff(C, C_ref, M * N);
    CHECK(err < TOL, "ssymm LEFT UPPER");
    if (err >= TOL) printf("  max err = %g\n", err);

    free(A); free(A_full); free(B); free(C); free(C_ref); free(C_orig);
}

/* ------------------------------------------------------------------ */
/* Test 6: SSYMM RIGHT LOWER                                          */
/* C := alpha * B * A + beta * C, A is N x N symmetric.               */
/* ------------------------------------------------------------------ */
static void test_ssymm_right_lower(void) {
    const int M = 12, N = 12;
    float *A      = malloc(sizeof(float) * N * N);
    float *A_full = malloc(sizeof(float) * N * N);
    float *B      = malloc(sizeof(float) * M * N);
    float *C      = malloc(sizeof(float) * M * N);
    float *C_ref  = malloc(sizeof(float) * M * N);
    float *C_orig = malloc(sizeof(float) * M * N);

    /* Fill A's lower triangle; sentinel above. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i >= j) A[i * N + j] = frand_det(7001 + i * N + j);
            else        A[i * N + j] = -999.0f;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_full[i * N + j] = (i >= j) ? A[i * N + j] : A[j * N + i];
        }
    }

    fill_random(B, M * N, 8002);
    fill_random(C_orig, M * N, 9003);
    memcpy(C,     C_orig, sizeof(float) * M * N);
    memcpy(C_ref, C_orig, sizeof(float) * M * N);

    const float alpha = 1.0f, beta = 0.5f;

    /* C[i,j] = beta*C_orig + alpha * sum_k B[i,k] * A_full[k,j] */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++) acc += B[i * N + k] * A_full[k * N + j];
            C_ref[i * N + j] = beta * C_orig[i * N + j] + alpha * acc;
        }
    }

    rac_ssymm(RAC_BLAS_RIGHT, RAC_BLAS_LOWER,
              M, N, alpha, A, N, B, N, beta, C, N);

    float err = maxabs_diff(C, C_ref, M * N);
    CHECK(err < TOL, "ssymm RIGHT LOWER");
    if (err >= TOL) printf("  max err = %g\n", err);

    free(A); free(A_full); free(B); free(C); free(C_ref); free(C_orig);
}

/* ------------------------------------------------------------------ */
/* Test 7: SSYRK UPPER NoTrans                                        */
/* C := alpha * A * A^T + beta * C, C is N x N symmetric.             */
/* A is N x K, only upper triangle of C updated.                      */
/* ------------------------------------------------------------------ */
static void test_ssyrk_upper_notrans(void) {
    const int N = 10, K = 8;
    float *A      = malloc(sizeof(float) * N * K);
    float *C      = malloc(sizeof(float) * N * N);
    float *C_ref  = malloc(sizeof(float) * N * N);

    fill_random(A, N * K, 11001);

    /* Sentinel lower, zero upper. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float v = (j >= i) ? 0.0f : -999.0f;
            C[i * N + j]     = v;
            C_ref[i * N + j] = v;
        }
    }

    const float alpha = 0.5f, beta = 0.0f;

    /* Reference for upper triangle only */
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[i * K + k] * A[j * K + k];
            C_ref[i * N + j] = alpha * acc;
        }
    }

    rac_ssyrk(RAC_BLAS_UPPER, RAC_BLAS_NO_TRANS,
              N, K, alpha, A, K, beta, C, N);

    /* Check upper triangle */
    float err = 0.0f;
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            float d = fabsf(C[i * N + j] - C_ref[i * N + j]);
            if (d > err) err = d;
        }
    }
    CHECK(err < TOL, "ssyrk UPPER NoTrans (upper)");
    if (err >= TOL) printf("  max err = %g\n", err);

    /* Check lower triangle sentinel preserved */
    int lower_ok = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            if (C[i * N + j] != -999.0f) { lower_ok = 0; break; }
        }
        if (!lower_ok) break;
    }
    CHECK(lower_ok, "ssyrk UPPER NoTrans (lower sentinel)");

    free(A); free(C); free(C_ref);
}

/* ------------------------------------------------------------------ */
/* Test 8: SSYR2K UPPER NoTrans                                       */
/* C := alpha * (A * B^T + B * A^T) + beta * C, upper triangle only.  */
/* ------------------------------------------------------------------ */
static void test_ssyr2k_upper_notrans(void) {
    const int N = 10, K = 8;
    float *A     = malloc(sizeof(float) * N * K);
    float *B     = malloc(sizeof(float) * N * K);
    float *C     = malloc(sizeof(float) * N * N);
    float *C_ref = malloc(sizeof(float) * N * N);

    fill_random(A, N * K, 12001);
    fill_random(B, N * K, 13002);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float v = (j >= i) ? 0.0f : -999.0f;
            C[i * N + j]     = v;
            C_ref[i * N + j] = v;
        }
    }

    const float alpha = 0.4f, beta = 0.0f;

    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[j * K + k] + B[i * K + k] * A[j * K + k];
            }
            C_ref[i * N + j] = alpha * acc;
        }
    }

    rac_ssyr2k(RAC_BLAS_UPPER, RAC_BLAS_NO_TRANS,
               N, K, alpha, A, K, B, K, beta, C, N);

    float err = 0.0f;
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            float d = fabsf(C[i * N + j] - C_ref[i * N + j]);
            if (d > err) err = d;
        }
    }
    CHECK(err < TOL, "ssyr2k UPPER NoTrans (upper)");
    if (err >= TOL) printf("  max err = %g\n", err);

    int lower_ok = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            if (C[i * N + j] != -999.0f) { lower_ok = 0; break; }
        }
        if (!lower_ok) break;
    }
    CHECK(lower_ok, "ssyr2k UPPER NoTrans (lower sentinel)");

    free(A); free(B); free(C); free(C_ref);
}

/* ------------------------------------------------------------------ */
/* Test 9: STRMM LEFT LOWER NoTrans NonUnit                           */
/* B := alpha * A * B, A is M x M lower triangular.                   */
/* ------------------------------------------------------------------ */
static void test_strmm_left_lower_notrans_nonunit(void) {
    const int M = 8, N = 10;
    float *A      = malloc(sizeof(float) * M * M);
    float *B      = malloc(sizeof(float) * M * N);
    float *B_orig = malloc(sizeof(float) * M * N);
    float *B_ref  = malloc(sizeof(float) * M * N);

    /* Lower-triangular A (zero above diag). */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            if (j <= i) A[i * M + j] = frand_det(14001 + i * M + j);
            else        A[i * M + j] = 0.0f;
        }
    }

    fill_random(B_orig, M * N, 15002);
    memcpy(B,     B_orig, sizeof(float) * M * N);
    memcpy(B_ref, B_orig, sizeof(float) * M * N);

    const float alpha = 1.0f;

    /* B_ref[i,j] = alpha * sum_{k<=i} A[i,k] * B_orig[k,j] */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k <= i; k++) acc += A[i * M + k] * B_orig[k * N + j];
            B_ref[i * N + j] = alpha * acc;
        }
    }

    rac_strmm(RAC_BLAS_LEFT, RAC_BLAS_LOWER,
              RAC_BLAS_NO_TRANS, RAC_BLAS_NON_UNIT,
              M, N, alpha, A, M, B, N);

    float err = maxabs_diff(B, B_ref, M * N);
    CHECK(err < TOL, "strmm LEFT LOWER NoTrans NonUnit");
    if (err >= TOL) printf("  max err = %g\n", err);

    free(A); free(B); free(B_orig); free(B_ref);
}

/* ------------------------------------------------------------------ */
/* Test 10: STRSM LEFT LOWER NoTrans NonUnit (round-trip)             */
/* Solve A * X = B where B = A * X_known; check recovered X.          */
/* ------------------------------------------------------------------ */
static void test_strsm_left_lower_round_trip(void) {
    const int M = 8, N = 6;
    float *A = malloc(sizeof(float) * M * M);
    float *X = malloc(sizeof(float) * M * N);
    float *B = malloc(sizeof(float) * M * N);

    /* Well-conditioned lower-tri A with positive diag in [1,2]. */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            if (j < i)      A[i * M + j] = 0.2f + 0.05f * (float)(i + j);
            else if (j == i) A[i * M + j] = 1.5f + 0.1f * (float)i;
            else            A[i * M + j] = 0.0f;
        }
    }

    fill_random(X, M * N, 16001);

    /* B = A * X using rac_sgemm_ex */
    rac_sgemm_ex(RAC_BLAS_NO_TRANS, RAC_BLAS_NO_TRANS,
                 M, N, M,
                 1.0f, A, M, X, N,
                 0.0f, B, N);

    rac_strsm(RAC_BLAS_LEFT, RAC_BLAS_LOWER,
              RAC_BLAS_NO_TRANS, RAC_BLAS_NON_UNIT,
              M, N, 1.0f, A, M, B, N);

    float err = maxabs_diff(B, X, M * N);
    CHECK(err < TOL, "strsm LEFT LOWER NoTrans NonUnit (round-trip)");
    if (err >= TOL) printf("  max err = %g\n", err);

    free(A); free(X); free(B);
}

/* ------------------------------------------------------------------ */
int main(void) {
    printf("=== rac_blas L3 BVT ===\n");
    test_sgemm_NN(); test_sgemm_TN(); test_sgemm_NT(); test_sgemm_TT();
    test_ssymm_left_upper(); test_ssymm_right_lower();
    test_ssyrk_upper_notrans(); test_ssyr2k_upper_notrans();
    test_strmm_left_lower_notrans_nonunit();
    test_strsm_left_lower_round_trip();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
