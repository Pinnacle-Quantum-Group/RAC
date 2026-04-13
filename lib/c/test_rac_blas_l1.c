/* test_rac_blas_l1.c — BVT for rac_blas Level 1.
 * PQG / Michael A. Doran Jr. — April 2026 */

#include "rac_blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("PASS: %s\n", name); g_pass++; } \
    else { printf("FAIL: %s\n", name); g_fail++; } } while(0)

static void test_saxpy(void) {
    enum { N = 64 };
    const float alpha = 2.5f;
    float x[N], y[N], y_old[N];
    for (int i = 0; i < N; i++) {
        x[i]     = (float)((i % 7) - 3);
        y[i]     = (float)((i % 5) - 2);
        y_old[i] = y[i];
    }
    rac_saxpy(N, alpha, x, 1, y, 1);
    int ok = 1;
    for (int i = 0; i < N; i++) {
        float ref = alpha * x[i] + y_old[i];
        if (fabsf(y[i] - ref) >= 1e-5f) { ok = 0; break; }
    }
    CHECK(ok, "saxpy n=64 alpha=2.5");
}

static void test_sdot(void) {
    enum { N = 128 };
    float x[N], y[N];
    for (int i = 0; i < N; i++) {
        x[i] = sinf(0.1f * (float)i);
        y[i] = cosf(0.1f * (float)i);
    }
    float ref = 0.0f;
    for (int i = 0; i < N; i++) ref += x[i] * y[i];
    float got = rac_sdot(N, x, 1, y, 1);
    CHECK(fabsf(got - ref) < 1e-3f, "sdot n=128 sin/cos");
}

static void test_snrm2(void) {
    enum { N = 64 };
    float x[N];
    for (int i = 0; i < N; i++) x[i] = (float)(i + 1) * 0.1f;
    float s = 0.0f;
    for (int i = 0; i < N; i++) s += x[i] * x[i];
    float ref = sqrtf(s);
    float got = rac_snrm2(N, x, 1);
    CHECK(fabsf(got - ref) < 1e-4f, "snrm2 n=64 ramp");
}

static void test_sasum(void) {
    enum { N = 64 };
    float x[N];
    for (int i = 0; i < N; i++)
        x[i] = ((i % 2) ? 1.0f : -1.0f) * (float)(i + 1) * 0.1f;
    float ref = 0.0f;
    for (int i = 0; i < N; i++) ref += fabsf(x[i]);
    float got = rac_sasum(N, x, 1);
    CHECK(fabsf(got - ref) < 1e-5f, "sasum n=64 alternating");
}

static void test_isamax(void) {
    float x[10] = {0.1f, -0.5f, 0.3f, 99.0f, -2.0f,
                   1.5f, -3.5f, 0.0f, 50.0f, 0.7f};
    int got = rac_isamax(10, x, 1);
    CHECK(got == 3, "isamax index==3");
}

static void test_sscal(void) {
    enum { N = 8 };
    float x[N];
    for (int i = 0; i < N; i++) x[i] = (float)(i + 1);
    rac_sscal(N, 0.5f, x, 1);
    int ok = 1;
    for (int i = 0; i < N; i++) {
        float ref = (float)(i + 1) * 0.5f;
        if (fabsf(x[i] - ref) >= 1e-5f) { ok = 0; break; }
    }
    CHECK(ok, "sscal n=8 alpha=0.5");
}

static void test_scopy(void) {
    enum { N = 16 };
    float x[N], y[N];
    for (int i = 0; i < N; i++) {
        x[i] = (float)((i * 3) % 11) - 5.0f;
        y[i] = -7.0f;
    }
    rac_scopy(N, x, 1, y, 1);
    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (fabsf(y[i] - x[i]) >= 1e-5f) { ok = 0; break; }
    }
    CHECK(ok, "scopy n=16");
}

static void test_sswap(void) {
    enum { N = 8 };
    float x[N], y[N], x_old[N], y_old[N];
    for (int i = 0; i < N; i++) {
        x[i] = (float)i;
        y[i] = -(float)i;
        x_old[i] = x[i];
        y_old[i] = y[i];
    }
    rac_sswap(N, x, 1, y, 1);
    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (fabsf(x[i] - y_old[i]) >= 1e-5f ||
            fabsf(y[i] - x_old[i]) >= 1e-5f) { ok = 0; break; }
    }
    CHECK(ok, "sswap n=8");
}

static void test_srot(void) {
    enum { N = 8 };
    float x[N], y[N], x_old[N], y_old[N];
    for (int i = 0; i < N; i++) {
        x[i] = cosf(0.1f * (float)i);
        y[i] = sinf(0.1f * (float)i);
        x_old[i] = x[i];
        y_old[i] = y[i];
    }
    float c = cosf((float)M_PI / 3.0f);
    float s = sinf((float)M_PI / 3.0f);
    rac_srot(N, x, 1, y, 1, c, s);
    int ok = 1;
    for (int i = 0; i < N; i++) {
        float ref_x = c * x_old[i] + s * y_old[i];
        float ref_y = c * y_old[i] - s * x_old[i];
        if (fabsf(x[i] - ref_x) >= 1e-5f ||
            fabsf(y[i] - ref_y) >= 1e-5f) { ok = 0; break; }
    }
    CHECK(ok, "srot n=8 theta=pi/3");
}

static void test_srotg(void) {
    float a = 3.0f, b = 4.0f, c = 0.0f, s = 0.0f;
    rac_srotg(&a, &b, &c, &s);
    CHECK(fabsf(a - 5.0f) < 1e-4f, "srotg r==5");
    CHECK(fabsf(c - 0.6f) < 1e-4f, "srotg c==0.6");
    CHECK(fabsf(s - 0.8f) < 1e-4f, "srotg s==0.8");
}

int main(void) {
    printf("=== rac_blas L1 BVT ===\n");
    test_saxpy(); test_sdot(); test_snrm2(); test_sasum(); test_isamax();
    test_sscal(); test_scopy(); test_sswap(); test_srot(); test_srotg();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
