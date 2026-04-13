//! BLAS Level 2 — matrix-vector ops. PQG / Michael A. Doran Jr. — April 2026
//!
//! Single-precision (f32), row-major. `lda` is row stride in elements.
//! Mirrors the C API in `lib/c/rac_blas.h`.
//!
//! Implemented:
//!   sgemv, sger, ssymv, ssyr, ssyr2, strmv, strsv

use super::{Op, UpLo, Diag};
use crate::{RacError, Result};

#[inline(always)]
fn nz(s: usize) -> usize { if s == 0 { 1 } else { s } }

#[inline(always)]
fn vec_min_len(n: usize, inc: usize) -> usize {
    if n == 0 { 0 } else { (n - 1) * inc + 1 }
}

// ---------- sgemv -----------------------------------------------------------

/// y := alpha * op(A) * x + beta * y.
pub fn sgemv(
    trans: Op, m: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    x: &[f32], incx: usize,
    beta: f32,
    y: &mut [f32], incy: usize,
) -> Result<()> {
    if m == 0 || n == 0 { return Ok(()); }
    let incx = nz(incx); let incy = nz(incy);
    if lda < n { return Err(RacError::InvalidDimension); }
    if a.len() < (m - 1) * lda + n { return Err(RacError::InvalidDimension); }

    let (xlen, ylen) = match trans {
        Op::NoTrans => (n, m),
        Op::Trans   => (m, n),
    };
    if x.len() < vec_min_len(xlen, incx) { return Err(RacError::InvalidDimension); }
    if y.len() < vec_min_len(ylen, incy) { return Err(RacError::InvalidDimension); }

    // beta * y
    if beta == 0.0 {
        for i in 0..ylen { y[i * incy] = 0.0; }
    } else if beta != 1.0 {
        for i in 0..ylen { y[i * incy] *= beta; }
    }
    if alpha == 0.0 { return Ok(()); }

    match trans {
        Op::NoTrans => {
            for i in 0..m {
                let row = &a[i * lda .. i * lda + n];
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc = (row[j] * x[j * incx]) + acc;
                }
                y[i * incy] += alpha * acc;
            }
        }
        Op::Trans => {
            for j in 0..n {
                let mut acc = 0.0f32;
                for i in 0..m {
                    acc = (a[i * lda + j] * x[i * incx]) + acc;
                }
                y[j * incy] += alpha * acc;
            }
        }
    }
    Ok(())
}

// ---------- sger ------------------------------------------------------------

/// A := alpha * x * y^T + A   (rank-1 update, M×N row-major).
pub fn sger(
    m: usize, n: usize, alpha: f32,
    x: &[f32], incx: usize,
    y: &[f32], incy: usize,
    a: &mut [f32], lda: usize,
) -> Result<()> {
    if m == 0 || n == 0 || alpha == 0.0 { return Ok(()); }
    let incx = nz(incx); let incy = nz(incy);
    if lda < n { return Err(RacError::InvalidDimension); }
    if a.len() < (m - 1) * lda + n { return Err(RacError::InvalidDimension); }
    if x.len() < vec_min_len(m, incx) { return Err(RacError::InvalidDimension); }
    if y.len() < vec_min_len(n, incy) { return Err(RacError::InvalidDimension); }

    for i in 0..m {
        let xi = alpha * x[i * incx];
        let row = &mut a[i * lda .. i * lda + n];
        for j in 0..n {
            row[j] += xi * y[j * incy];
        }
    }
    Ok(())
}

// ---------- ssymv -----------------------------------------------------------

#[inline(always)]
fn sym_get(a: &[f32], lda: usize, i: usize, j: usize, uplo: UpLo) -> f32 {
    let stored = match uplo {
        UpLo::Upper => i <= j,
        UpLo::Lower => i >= j,
    };
    if stored { a[i * lda + j] } else { a[j * lda + i] }
}

/// y := alpha * A * x + beta * y, A is N×N symmetric (only `uplo` triangle read).
pub fn ssymv(
    uplo: UpLo, n: usize, alpha: f32,
    a: &[f32], lda: usize,
    x: &[f32], incx: usize,
    beta: f32,
    y: &mut [f32], incy: usize,
) -> Result<()> {
    if n == 0 { return Ok(()); }
    let incx = nz(incx); let incy = nz(incy);
    if lda < n { return Err(RacError::InvalidDimension); }
    if a.len() < (n - 1) * lda + n { return Err(RacError::InvalidDimension); }
    if x.len() < vec_min_len(n, incx) { return Err(RacError::InvalidDimension); }
    if y.len() < vec_min_len(n, incy) { return Err(RacError::InvalidDimension); }

    if beta == 0.0 {
        for i in 0..n { y[i * incy] = 0.0; }
    } else if beta != 1.0 {
        for i in 0..n { y[i * incy] *= beta; }
    }
    if alpha == 0.0 { return Ok(()); }

    for i in 0..n {
        let mut acc = 0.0f32;
        for j in 0..n {
            acc = (sym_get(a, lda, i, j, uplo) * x[j * incx]) + acc;
        }
        y[i * incy] += alpha * acc;
    }
    Ok(())
}

// ---------- ssyr ------------------------------------------------------------

/// A := alpha * x * x^T + A, only `uplo` triangle (incl. diagonal) updated.
pub fn ssyr(
    uplo: UpLo, n: usize, alpha: f32,
    x: &[f32], incx: usize,
    a: &mut [f32], lda: usize,
) -> Result<()> {
    if n == 0 || alpha == 0.0 { return Ok(()); }
    let incx = nz(incx);
    if lda < n { return Err(RacError::InvalidDimension); }
    if a.len() < (n - 1) * lda + n { return Err(RacError::InvalidDimension); }
    if x.len() < vec_min_len(n, incx) { return Err(RacError::InvalidDimension); }

    for i in 0..n {
        let xi = alpha * x[i * incx];
        match uplo {
            UpLo::Upper => {
                for j in i..n {
                    a[i * lda + j] += xi * x[j * incx];
                }
            }
            UpLo::Lower => {
                for j in 0..=i {
                    a[i * lda + j] += xi * x[j * incx];
                }
            }
        }
    }
    Ok(())
}

// ---------- ssyr2 -----------------------------------------------------------

/// A := alpha * (x*y^T + y*x^T) + A, only `uplo` triangle updated.
pub fn ssyr2(
    uplo: UpLo, n: usize, alpha: f32,
    x: &[f32], incx: usize,
    y: &[f32], incy: usize,
    a: &mut [f32], lda: usize,
) -> Result<()> {
    if n == 0 || alpha == 0.0 { return Ok(()); }
    let incx = nz(incx); let incy = nz(incy);
    if lda < n { return Err(RacError::InvalidDimension); }
    if a.len() < (n - 1) * lda + n { return Err(RacError::InvalidDimension); }
    if x.len() < vec_min_len(n, incx) { return Err(RacError::InvalidDimension); }
    if y.len() < vec_min_len(n, incy) { return Err(RacError::InvalidDimension); }

    for i in 0..n {
        let xi = alpha * x[i * incx];
        let yi = alpha * y[i * incy];
        match uplo {
            UpLo::Upper => {
                for j in i..n {
                    a[i * lda + j] += xi * y[j * incy] + yi * x[j * incx];
                }
            }
            UpLo::Lower => {
                for j in 0..=i {
                    a[i * lda + j] += xi * y[j * incy] + yi * x[j * incx];
                }
            }
        }
    }
    Ok(())
}

// ---------- strmv -----------------------------------------------------------

#[inline(always)]
fn diag_val(a: &[f32], lda: usize, i: usize, diag: Diag) -> f32 {
    match diag {
        Diag::Unit => 1.0,
        Diag::NonUnit => a[i * lda + i],
    }
}

/// x := op(A) * x, A is N×N triangular.
pub fn strmv(
    uplo: UpLo, trans: Op, diag: Diag, n: usize,
    a: &[f32], lda: usize,
    x: &mut [f32], incx: usize,
) -> Result<()> {
    if n == 0 { return Ok(()); }
    let incx = nz(incx);
    if lda < n { return Err(RacError::InvalidDimension); }
    if a.len() < (n - 1) * lda + n { return Err(RacError::InvalidDimension); }
    if x.len() < vec_min_len(n, incx) { return Err(RacError::InvalidDimension); }

    match (trans, uplo) {
        // NoTrans + Lower: result_i = sum_{j<=i} A[i,j]*x[j]. Iterate i = n-1 down to 0.
        (Op::NoTrans, UpLo::Lower) => {
            for ii in (0..n).rev() {
                let i = ii;
                let mut acc = diag_val(a, lda, i, diag) * x[i * incx];
                for j in 0..i {
                    acc = (a[i * lda + j] * x[j * incx]) + acc;
                }
                x[i * incx] = acc;
            }
        }
        // NoTrans + Upper: result_i = sum_{j>=i} A[i,j]*x[j]. Iterate i = 0..n.
        (Op::NoTrans, UpLo::Upper) => {
            for i in 0..n {
                let mut acc = diag_val(a, lda, i, diag) * x[i * incx];
                for j in (i+1)..n {
                    acc = (a[i * lda + j] * x[j * incx]) + acc;
                }
                x[i * incx] = acc;
            }
        }
        // Trans + Lower (A^T is upper-relative): result_j = sum_{i>=j} A[i,j]*x[i]. j = n-1..0.
        (Op::Trans, UpLo::Lower) => {
            for jj in (0..n).rev() {
                let j = jj;
                let mut acc = diag_val(a, lda, j, diag) * x[j * incx];
                for i in (j+1)..n {
                    acc = (a[i * lda + j] * x[i * incx]) + acc;
                }
                x[j * incx] = acc;
            }
        }
        // Trans + Upper: result_j = sum_{i<=j} A[i,j]*x[i]. j = 0..n.
        (Op::Trans, UpLo::Upper) => {
            for j in 0..n {
                let mut acc = diag_val(a, lda, j, diag) * x[j * incx];
                for i in 0..j {
                    acc = (a[i * lda + j] * x[i * incx]) + acc;
                }
                x[j * incx] = acc;
            }
        }
    }
    Ok(())
}

// ---------- strsv -----------------------------------------------------------

/// Solve op(A) x = b in place (b passed via x). A is N×N triangular non-singular.
pub fn strsv(
    uplo: UpLo, trans: Op, diag: Diag, n: usize,
    a: &[f32], lda: usize,
    x: &mut [f32], incx: usize,
) -> Result<()> {
    if n == 0 { return Ok(()); }
    let incx = nz(incx);
    if lda < n { return Err(RacError::InvalidDimension); }
    if a.len() < (n - 1) * lda + n { return Err(RacError::InvalidDimension); }
    if x.len() < vec_min_len(n, incx) { return Err(RacError::InvalidDimension); }

    match (trans, uplo) {
        // NoTrans + Lower (forward sub): x[i] = (x[i] - sum_{j<i} A[i,j]*x[j]) / A[i,i]
        (Op::NoTrans, UpLo::Lower) => {
            for i in 0..n {
                let mut s = x[i * incx];
                for j in 0..i {
                    s -= a[i * lda + j] * x[j * incx];
                }
                if matches!(diag, Diag::NonUnit) { s /= a[i * lda + i]; }
                x[i * incx] = s;
            }
        }
        // NoTrans + Upper (back sub): x[i] = (x[i] - sum_{j>i} A[i,j]*x[j]) / A[i,i]
        (Op::NoTrans, UpLo::Upper) => {
            for ii in (0..n).rev() {
                let i = ii;
                let mut s = x[i * incx];
                for j in (i+1)..n {
                    s -= a[i * lda + j] * x[j * incx];
                }
                if matches!(diag, Diag::NonUnit) { s /= a[i * lda + i]; }
                x[i * incx] = s;
            }
        }
        // Trans + Lower: x[j] = (x[j] - sum_{i>j} A[i,j]*x[i]) / A[j,j]
        (Op::Trans, UpLo::Lower) => {
            for jj in (0..n).rev() {
                let j = jj;
                let mut s = x[j * incx];
                for i in (j+1)..n {
                    s -= a[i * lda + j] * x[i * incx];
                }
                if matches!(diag, Diag::NonUnit) { s /= a[j * lda + j]; }
                x[j * incx] = s;
            }
        }
        // Trans + Upper: x[j] = (x[j] - sum_{i<j} A[i,j]*x[i]) / A[j,j]
        (Op::Trans, UpLo::Upper) => {
            for j in 0..n {
                let mut s = x[j * incx];
                for i in 0..j {
                    s -= a[i * lda + j] * x[i * incx];
                }
                if matches!(diag, Diag::NonUnit) { s /= a[j * lda + j]; }
                x[j * incx] = s;
            }
        }
    }
    Ok(())
}

// ---------- tests -----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[test]
    fn test_sgemv_no_trans_3x4() {
        // A = [[1,2,3,4],[5,6,7,8],[9,10,11,12]], x=[1,1,1,1] -> y = [10, 26, 42]
        let a = [1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0,
                 9.0, 10.0, 11.0, 12.0];
        let x = [1.0; 4];
        let mut y = [0.0f32; 3];
        sgemv(Op::NoTrans, 3, 4, 1.0, &a, 4, &x, 1, 0.0, &mut y, 1).unwrap();
        let exp = [10.0, 26.0, 42.0];
        assert!(maxabs(&y, &exp) < 1e-5, "got {:?}", y);
    }

    #[test]
    fn test_sgemv_alpha_beta() {
        let a = [1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0];
        let x = [1.0, 1.0, 1.0, 1.0];
        let mut y = [10.0f32, 20.0];
        sgemv(Op::NoTrans, 2, 4, 2.0, &a, 4, &x, 1, 0.5, &mut y, 1).unwrap();
        // y0 = 0.5*10 + 2*(10) = 25; y1 = 0.5*20 + 2*(26) = 62
        assert!((y[0] - 25.0).abs() < 1e-4);
        assert!((y[1] - 62.0).abs() < 1e-4);
    }

    #[test]
    fn test_sgemv_trans() {
        let a = [1.0, 2.0,
                 3.0, 4.0,
                 5.0, 6.0]; // 3x2
        let x = [1.0, 1.0, 1.0]; // length 3 (m)
        let mut y = [0.0f32; 2]; // length 2 (n)
        sgemv(Op::Trans, 3, 2, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1).unwrap();
        // y[0] = 1+3+5=9; y[1] = 2+4+6=12
        assert!((y[0] - 9.0).abs() < 1e-5);
        assert!((y[1] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_sger_rank1() {
        let mut a = [0.0f32; 6]; // 2x3
        let x = [1.0, 2.0];
        let y = [10.0, 20.0, 30.0];
        sger(2, 3, 1.0, &x, 1, &y, 1, &mut a, 3).unwrap();
        // expected: [[10,20,30],[20,40,60]]
        let exp = [10.0, 20.0, 30.0, 20.0, 40.0, 60.0];
        assert!(maxabs(&a, &exp) < 1e-5);
    }

    #[test]
    fn test_ssymv_upper_matches_full() {
        // Stored upper triangle. Build full sym for reference.
        let n = 4;
        // upper:
        //   1 2 3 4
        //   . 5 6 7
        //   . . 8 9
        //   . . . 10
        let a = [1.0, 2.0, 3.0, 4.0,
                 0.0, 5.0, 6.0, 7.0,
                 0.0, 0.0, 8.0, 9.0,
                 0.0, 0.0, 0.0, 10.0];
        // Full symmetric for the reference:
        let full = [1.0, 2.0, 3.0, 4.0,
                    2.0, 5.0, 6.0, 7.0,
                    3.0, 6.0, 8.0, 9.0,
                    4.0, 7.0, 9.0, 10.0];
        let x = [1.0, 1.0, 1.0, 1.0];
        let mut y = [0.0f32; 4];
        let mut ref_y = [0.0f32; 4];
        ssymv(UpLo::Upper, n, 1.0, &a, n, &x, 1, 0.0, &mut y, 1).unwrap();
        sgemv(Op::NoTrans, n, n, 1.0, &full, n, &x, 1, 0.0, &mut ref_y, 1).unwrap();
        assert!(maxabs(&y, &ref_y) < 1e-5, "y={:?} ref={:?}", y, ref_y);
    }

    #[test]
    fn test_ssyr_upper_only_touches_chosen_triangle() {
        let n = 3;
        let mut a = [
            0.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,  // sentinel below diag
            -2.0, -3.0, 0.0,
        ];
        let x = [1.0, 2.0, 3.0];
        ssyr(UpLo::Upper, n, 1.0, &x, 1, &mut a, n).unwrap();
        // upper: A[0,0]=1, A[0,1]=2, A[0,2]=3, A[1,1]=4, A[1,2]=6, A[2,2]=9
        assert!((a[0] - 1.0).abs() < 1e-5);
        assert!((a[1] - 2.0).abs() < 1e-5);
        assert!((a[2] - 3.0).abs() < 1e-5);
        assert!((a[4] - 4.0).abs() < 1e-5);
        assert!((a[5] - 6.0).abs() < 1e-5);
        assert!((a[8] - 9.0).abs() < 1e-5);
        // sentinels untouched
        assert!((a[3] - (-1.0)).abs() < 1e-5);
        assert!((a[6] - (-2.0)).abs() < 1e-5);
        assert!((a[7] - (-3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_strmv_lower_no_trans_non_unit() {
        // L = [[1,0,0],[2,3,0],[4,5,6]], x = [1,1,1]
        // result = [1, 2+3, 4+5+6] = [1, 5, 15]
        let a = [1.0, 0.0, 0.0,
                 2.0, 3.0, 0.0,
                 4.0, 5.0, 6.0];
        let mut x = [1.0f32, 1.0, 1.0];
        strmv(UpLo::Lower, Op::NoTrans, Diag::NonUnit, 3, &a, 3, &mut x, 1).unwrap();
        let exp = [1.0, 5.0, 15.0];
        assert!(maxabs(&x, &exp) < 1e-5);
    }

    #[test]
    fn test_strsv_lower_round_trip() {
        // L with diag in [1,2], known x, b = L*x via strmv, then strsv recovers x.
        let n = 5;
        let mut l = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..=i {
                l[i * n + j] = if i == j { 1.5 + (i as f32) * 0.1 } else { 0.2 + 0.05 * (i + j) as f32 };
            }
        }
        let x_known = [1.0f32, -2.0, 3.0, -4.0, 5.0];
        let mut b = x_known;
        strmv(UpLo::Lower, Op::NoTrans, Diag::NonUnit, n, &l, n, &mut b, 1).unwrap();
        strsv(UpLo::Lower, Op::NoTrans, Diag::NonUnit, n, &l, n, &mut b, 1).unwrap();
        assert!(maxabs(&b, &x_known) < 1e-3, "recovered={:?} expected={:?}", b, x_known);
    }

    #[test]
    fn test_invalid_lda_returns_error() {
        let a = [1.0f32; 4];
        let x = [1.0f32; 4];
        let mut y = [0.0f32; 4];
        // n=4 but lda=2 (< n) should error.
        let r = sgemv(Op::NoTrans, 4, 4, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1);
        assert_eq!(r, Err(RacError::InvalidDimension));
    }
}
