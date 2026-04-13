/* rac_blas_l1.c — RAC BLAS Level 1 (single precision, row-major). PQG / Michael A. Doran Jr. — April 2026 */

#include "rac_blas.h"
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* y := alpha * x + y */
void rac_saxpy(int n, float alpha,
               const float *x, int incx,
               float *y, int incy)
{
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        #pragma omp parallel for if(n > 1024)
#endif
        for (int i = 0; i < n; ++i)
            y[i] = fmaf(alpha, x[i], y[i]);
    } else {
        for (int i = 0; i < n; ++i)
            y[i * incy] = fmaf(alpha, x[i * incx], y[i * incy]);
    }
}

/* dot := sum_i x_i * y_i */
float rac_sdot(int n,
               const float *x, int incx,
               const float *y, int incy)
{
    if (n <= 0) return 0.0f;
    float acc = 0.0f;
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; ++i)
            acc = fmaf(x[i], y[i], acc);
    } else {
        for (int i = 0; i < n; ++i)
            acc = fmaf(x[i * incx], y[i * incy], acc);
    }
    return acc;
}

/* nrm2 := sqrt(sum_i x_i^2)  (Lawson-Hanson scaled accumulation) */
float rac_snrm2(int n, const float *x, int incx)
{
    if (n <= 0) return 0.0f;
    if (n == 1) return fabsf(x[0]);

    float scale = 0.0f, ssq = 1.0f;
    for (int i = 0; i < n; ++i) {
        float xi = (incx == 1) ? x[i] : x[i * incx];
        if (xi != 0.0f) {
            float ax = fabsf(xi);
            if (scale < ax) {
                float r = scale / ax;
                ssq = 1.0f + ssq * r * r;
                scale = ax;
            } else {
                float r = ax / scale;
                ssq += r * r;
            }
        }
    }
    return scale * sqrtf(ssq);
}

/* asum := sum_i |x_i| */
float rac_sasum(int n, const float *x, int incx)
{
    if (n <= 0) return 0.0f;
    float acc = 0.0f;
    if (incx == 1) {
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:acc) if(n > 1024)
#endif
        for (int i = 0; i < n; ++i)
            acc += fabsf(x[i]);
    } else {
        for (int i = 0; i < n; ++i)
            acc += fabsf(x[i * incx]);
    }
    return acc;
}

/* return 0-based index of element with max |x_i| */
int rac_isamax(int n, const float *x, int incx)
{
    if (n <= 0) return 0;
    int   imax = 0;
    float vmax = fabsf(x[0]);
    if (incx == 1) {
        for (int i = 1; i < n; ++i) {
            float a = fabsf(x[i]);
            if (a > vmax) { vmax = a; imax = i; }
        }
    } else {
        for (int i = 1; i < n; ++i) {
            float a = fabsf(x[i * incx]);
            if (a > vmax) { vmax = a; imax = i; }
        }
    }
    return imax;
}

/* x := alpha * x */
void rac_sscal(int n, float alpha, float *x, int incx)
{
    if (n <= 0) return;
    if (incx == 1) {
#ifdef _OPENMP
        #pragma omp parallel for if(n > 1024)
#endif
        for (int i = 0; i < n; ++i) x[i] *= alpha;
    } else {
        for (int i = 0; i < n; ++i) x[i * incx] *= alpha;
    }
}

/* y := x */
void rac_scopy(int n,
               const float *x, int incx,
               float *y, int incy)
{
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        memcpy(y, x, (size_t)n * sizeof(float));
    } else {
        for (int i = 0; i < n; ++i)
            y[i * incy] = x[i * incx];
    }
}

/* swap(x, y) */
void rac_sswap(int n,
               float *x, int incx,
               float *y, int incy)
{
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        #pragma omp parallel for if(n > 1024)
#endif
        for (int i = 0; i < n; ++i) {
            float t = x[i]; x[i] = y[i]; y[i] = t;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            float t = x[i * incx];
            x[i * incx] = y[i * incy];
            y[i * incy] = t;
        }
    }
}

/* Apply Givens rotation: x_i := c*x_i + s*y_i; y_i := c*y_i - s*x_i */
void rac_srot(int n,
              float *x, int incx,
              float *y, int incy,
              float c, float s)
{
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        #pragma omp parallel for if(n > 1024)
#endif
        for (int i = 0; i < n; ++i) {
            float tmp = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = tmp;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            float xi = x[i * incx], yi = y[i * incy];
            float tmp = c * xi + s * yi;
            y[i * incy] = c * yi - s * xi;
            x[i * incx] = tmp;
        }
    }
}

/* Construct Givens rotation (Lawson-Hanson). Overwrites *a=r, *b=z. */
void rac_srotg(float *a, float *b, float *c, float *s)
{
    float roe   = (fabsf(*a) > fabsf(*b)) ? *a : *b;
    float scale = fabsf(*a) + fabsf(*b);
    float r, z;

    if (scale == 0.0f) {
        *c = 1.0f; *s = 0.0f; r = 0.0f; z = 0.0f;
    } else {
        float aa = *a / scale, bb = *b / scale;
        r = scale * sqrtf(aa * aa + bb * bb);
        r = copysignf(r, roe);
        *c = *a / r;
        *s = *b / r;
        z = 1.0f;
        if (fabsf(*a) > fabsf(*b))      z = *s;
        else if (*c != 0.0f)            z = 1.0f / *c;
    }
    *a = r;
    *b = z;
}
