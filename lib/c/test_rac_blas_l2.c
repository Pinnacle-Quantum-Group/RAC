/* test_rac_blas_l2.c — BVT for rac_blas Level 2.
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

/* ── Test 1: sgemv NoTrans ──────────────────────────────────────────────── */
static void test_sgemv_notrans(void) {
    enum { M = 8, N = 12 };
    float A[M * N];
    float x[N], y[M], y_old[M], y_ref[M];
    const float alpha = 1.5f, beta = 0.25f;

    fill_random(A, M * N, 100);
    fill_random(x, N,     200);
    fill_random(y, M,     300);
    memcpy(y_old, y, sizeof(y));

    for (int i = 0; i < M; i++) {
        float s = 0.0f;
        for (int j = 0; j < N; j++) s += A[i * N + j] * x[j];
        y_ref[i] = beta * y_old[i] + alpha * s;
    }

    rac_sgemv(RAC_BLAS_NO_TRANS, M, N, alpha, A, N, x, 1, beta, y, 1);

    CHECK(maxabs_diff(y, y_ref, M) < 1e-4f, "sgemv NoTrans M=8 N=12");
}

/* ── Test 2: sgemv Trans ────────────────────────────────────────────────── */
static void test_sgemv_trans(void) {
    enum { M = 8, N = 12 };
    float A[M * N];
    float x[M], y[N], y_old[N], y_ref[N];
    const float alpha = 1.5f, beta = 0.25f;

    fill_random(A, M * N, 400);
    fill_random(x, M,     500);
    fill_random(y, N,     600);
    memcpy(y_old, y, sizeof(y));

    for (int j = 0; j < N; j++) {
        float s = 0.0f;
        for (int i = 0; i < M; i++) s += A[i * N + j] * x[i];
        y_ref[j] = beta * y_old[j] + alpha * s;
    }

    rac_sgemv(RAC_BLAS_TRANS, M, N, alpha, A, N, x, 1, beta, y, 1);

    CHECK(maxabs_diff(y, y_ref, N) < 1e-4f, "sgemv Trans M=8 N=12");
}

/* ── Test 3: sger ───────────────────────────────────────────────────────── */
static void test_sger(void) {
    enum { M = 6, N = 10 };
    float A[M * N], A_old[M * N], A_ref[M * N];
    float x[M], y[N];
    const float alpha = 0.7f;

    fill_random(A, M * N, 700);
    fill_random(x, M,     800);
    fill_random(y, N,     900);
    memcpy(A_old, A, sizeof(A));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A_ref[i * N + j] = A_old[i * N + j] + alpha * x[i] * y[j];
        }
    }

    rac_sger(M, N, alpha, x, 1, y, 1, A, N);

    CHECK(maxabs_diff(A, A_ref, M * N) < 1e-4f, "sger M=6 N=10 alpha=0.7");
}

/* ── Test 4: ssymv Upper ────────────────────────────────────────────────── */
static void test_ssymv_upper(void) {
    enum { N = 8 };
    float A[N * N], A_full[N * N];
    float x[N], y[N], y_ref[N];
    const float alpha = 1.0f, beta = 0.0f;

    /* Fill upper triangle (j>=i) with random, lower with sentinel garbage. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j >= i) A[i * N + j] = frand_det(1000 + i * N + j);
            else        A[i * N + j] = -999.0f;
        }
    }
    fill_random(x, N, 1100);
    fill_random(y, N, 1200);
    /* y_old not needed because beta=0.0 */

    /* Build full symmetric matrix from upper triangle. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_full[i * N + j] = (i <= j) ? A[i * N + j] : A[j * N + i];
        }
    }

    for (int i = 0; i < N; i++) {
        float s = 0.0f;
        for (int j = 0; j < N; j++) s += A_full[i * N + j] * x[j];
        y_ref[i] = beta * y[i] + alpha * s;
    }

    rac_ssymv(RAC_BLAS_UPPER, N, alpha, A, N, x, 1, beta, y, 1);

    CHECK(maxabs_diff(y, y_ref, N) < 1e-4f, "ssymv Upper N=8");
}

/* ── Test 5: ssymv Lower ────────────────────────────────────────────────── */
static void test_ssymv_lower(void) {
    enum { N = 8 };
    float A[N * N], A_full[N * N];
    float x[N], y[N], y_ref[N];
    const float alpha = 1.0f, beta = 0.0f;

    /* Fill lower triangle (j<=i) with random, upper with sentinel. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j <= i) A[i * N + j] = frand_det(1300 + i * N + j);
            else        A[i * N + j] = -999.0f;
        }
    }
    fill_random(x, N, 1400);
    fill_random(y, N, 1500);

    /* Build full symmetric matrix from lower triangle. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_full[i * N + j] = (i >= j) ? A[i * N + j] : A[j * N + i];
        }
    }

    for (int i = 0; i < N; i++) {
        float s = 0.0f;
        for (int j = 0; j < N; j++) s += A_full[i * N + j] * x[j];
        y_ref[i] = beta * y[i] + alpha * s;
    }

    rac_ssymv(RAC_BLAS_LOWER, N, alpha, A, N, x, 1, beta, y, 1);

    CHECK(maxabs_diff(y, y_ref, N) < 1e-4f, "ssymv Lower N=8");
}

/* ── Test 6: ssyr Upper ─────────────────────────────────────────────────── */
static void test_ssyr_upper(void) {
    enum { N = 8 };
    float A[N * N], A_old[N * N];
    float x[N];
    const float alpha = 0.5f;

    /* Sentinel pattern: upper random, lower = -999.0f (must remain untouched). */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j >= i) A[i * N + j] = frand_det(1600 + i * N + j);
            else        A[i * N + j] = -999.0f;
        }
    }
    memcpy(A_old, A, sizeof(A));
    fill_random(x, N, 1700);

    rac_ssyr(RAC_BLAS_UPPER, N, alpha, x, 1, A, N);

    int upper_ok = 1, lower_ok = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j >= i) {
                float ref = A_old[i * N + j] + alpha * x[i] * x[j];
                if (fabsf(A[i * N + j] - ref) >= 1e-4f) upper_ok = 0;
            } else {
                if (A[i * N + j] != -999.0f) lower_ok = 0;
            }
        }
    }
    CHECK(upper_ok, "ssyr Upper N=8 upper-triangle values");
    CHECK(lower_ok, "ssyr Upper N=8 lower-triangle untouched");
}

/* ── Test 7: ssyr2 Upper ────────────────────────────────────────────────── */
static void test_ssyr2_upper(void) {
    enum { N = 8 };
    float A[N * N], A_old[N * N];
    float x[N], y[N];
    const float alpha = 0.3f;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j >= i) A[i * N + j] = frand_det(1800 + i * N + j);
            else        A[i * N + j] = -999.0f;
        }
    }
    memcpy(A_old, A, sizeof(A));
    fill_random(x, N, 1900);
    fill_random(y, N, 2000);

    rac_ssyr2(RAC_BLAS_UPPER, N, alpha, x, 1, y, 1, A, N);

    int upper_ok = 1, lower_ok = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j >= i) {
                float ref = A_old[i * N + j]
                          + alpha * (x[i] * y[j] + y[i] * x[j]);
                if (fabsf(A[i * N + j] - ref) >= 1e-4f) upper_ok = 0;
            } else {
                if (A[i * N + j] != -999.0f) lower_ok = 0;
            }
        }
    }
    CHECK(upper_ok, "ssyr2 Upper N=8 upper-triangle values");
    CHECK(lower_ok, "ssyr2 Upper N=8 lower-triangle untouched");
}

/* ── Test 8: strmv Lower, NoTrans, NonUnit ──────────────────────────────── */
static void test_strmv_lower_notrans_nonunit(void) {
    enum { N = 8 };
    float A[N * N];
    float x[N], x_old[N], x_ref[N];

    /* Build random lower-triangular A (zeros above diagonal). */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j <= i) A[i * N + j] = frand_det(2100 + i * N + j);
            else        A[i * N + j] = 0.0f;
        }
    }
    fill_random(x, N, 700);
    memcpy(x_old, x, sizeof(x));

    /* x_ref[i] = sum_{j<=i} A[i,j] * x_old[j] */
    for (int i = 0; i < N; i++) {
        float s = 0.0f;
        for (int j = 0; j <= i; j++) s += A[i * N + j] * x_old[j];
        x_ref[i] = s;
    }

    rac_strmv(RAC_BLAS_LOWER, RAC_BLAS_NO_TRANS, RAC_BLAS_NON_UNIT,
              N, A, N, x, 1);

    CHECK(maxabs_diff(x, x_ref, N) < 1e-4f,
          "strmv Lower NoTrans NonUnit N=8");
}

/* ── Test 9: strsv Lower, NoTrans, NonUnit ──────────────────────────────── */
static void test_strsv_lower_notrans_nonunit(void) {
    enum { N = 8 };
    float A[N * N];
    float x_known[N], b[N];

    /* Well-conditioned lower-triangular: diag in [1.5, 2.2]; small off-diag. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j == i)      A[i * N + j] = 1.5f + 0.1f * (float)i;
            else if (j < i)  A[i * N + j] = 0.2f + 0.05f * (float)(i + j);
            else             A[i * N + j] = 0.0f;
        }
    }
    for (int i = 0; i < N; i++) x_known[i] = (float)(i + 1) * 0.5f;

    /* b[i] = sum_{j<=i} A[i,j] * x_known[j] */
    for (int i = 0; i < N; i++) {
        float s = 0.0f;
        for (int j = 0; j <= i; j++) s += A[i * N + j] * x_known[j];
        b[i] = s;
    }

    rac_strsv(RAC_BLAS_LOWER, RAC_BLAS_NO_TRANS, RAC_BLAS_NON_UNIT,
              N, A, N, b, 1);

    CHECK(maxabs_diff(b, x_known, N) < 1e-3f,
          "strsv Lower NoTrans NonUnit N=8");
}

/* ── Test 10: strsv Upper, NoTrans, NonUnit ─────────────────────────────── */
static void test_strsv_upper_notrans_nonunit(void) {
    enum { N = 8 };
    float A[N * N];
    float x_known[N], b[N];

    /* Well-conditioned upper-triangular: diag in [1.5, 2.2]; small off-diag. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j == i)      A[i * N + j] = 1.5f + 0.1f * (float)i;
            else if (j > i)  A[i * N + j] = 0.2f + 0.05f * (float)(i + j);
            else             A[i * N + j] = 0.0f;
        }
    }
    for (int i = 0; i < N; i++) x_known[i] = (float)(i + 1) * 0.5f;

    /* b[i] = sum_{j>=i} A[i,j] * x_known[j] */
    for (int i = 0; i < N; i++) {
        float s = 0.0f;
        for (int j = i; j < N; j++) s += A[i * N + j] * x_known[j];
        b[i] = s;
    }

    rac_strsv(RAC_BLAS_UPPER, RAC_BLAS_NO_TRANS, RAC_BLAS_NON_UNIT,
              N, A, N, b, 1);

    CHECK(maxabs_diff(b, x_known, N) < 1e-3f,
          "strsv Upper NoTrans NonUnit N=8");
}

/* ── main ───────────────────────────────────────────────────────────────── */
int main(void) {
    printf("RAC BLAS Level-2 BVT\n");
    printf("====================\n");

    test_sgemv_notrans();
    test_sgemv_trans();
    test_sger();
    test_ssymv_upper();
    test_ssymv_lower();
    test_ssyr_upper();
    test_ssyr2_upper();
    test_strmv_lower_notrans_nonunit();
    test_strsv_lower_notrans_nonunit();
    test_strsv_upper_notrans_nonunit();

    printf("--------------------\n");
    printf("Summary: %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
