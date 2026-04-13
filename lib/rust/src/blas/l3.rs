//! BLAS Level 3 — matrix-matrix ops. PQG / Michael A. Doran Jr. — April 2026
//!
//! Single-precision (f32), row-major. lda/ldb/ldc are row strides in elements.
//! Mirrors the C API in `lib/c/rac_blas.h`.
//!
//! Implemented:
//!   sgemm_ex, ssymm, ssyrk, ssyr2k, strmm, strsm

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::{Op, UpLo, Diag, Side};
use crate::{RacError, Result, Config};

#[inline(always)]
fn scale_c(c: &mut [f32], m: usize, n: usize, ldc: usize, beta: f32) {
    if beta == 1.0 { return; }
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] = 0.0;
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] *= beta;
            }
        }
    }
}

#[inline(always)]
fn sym_get(a: &[f32], lda: usize, i: usize, j: usize, uplo: UpLo) -> f32 {
    let stored = match uplo {
        UpLo::Upper => i <= j,
        UpLo::Lower => i >= j,
    };
    if stored { a[i * lda + j] } else { a[j * lda + i] }
}

#[inline(always)]
fn tri_get(a: &[f32], lda: usize, i: usize, j: usize, uplo: UpLo, diag: Diag) -> f32 {
    if i == j {
        return match diag { Diag::Unit => 1.0, Diag::NonUnit => a[i * lda + i] };
    }
    let in_tri = match uplo {
        UpLo::Upper => i < j,
        UpLo::Lower => i > j,
    };
    if in_tri { a[i * lda + j] } else { 0.0 }
}

// ---------- sgemm_ex --------------------------------------------------------

/// C := alpha * op(A) * op(B) + beta * C
pub fn sgemm_ex(
    trans_a: Op, trans_b: Op,
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) -> Result<()> {
    if m == 0 || n == 0 { return Ok(()); }

    // Validate
    let (a_rows, a_cols) = match trans_a {
        Op::NoTrans => (m, k), Op::Trans => (k, m),
    };
    let (b_rows, b_cols) = match trans_b {
        Op::NoTrans => (k, n), Op::Trans => (n, k),
    };
    if lda < a_cols { return Err(RacError::InvalidDimension); }
    if ldb < b_cols { return Err(RacError::InvalidDimension); }
    if ldc < n { return Err(RacError::InvalidDimension); }
    if k > 0 {
        if a.len() < (a_rows - 1) * lda + a_cols { return Err(RacError::InvalidDimension); }
        if b.len() < (b_rows - 1) * ldb + b_cols { return Err(RacError::InvalidDimension); }
    }
    if c.len() < (m - 1) * ldc + n { return Err(RacError::InvalidDimension); }

    // Fast path: contiguous, no transpose -> existing tiled SGEMM
    if matches!(trans_a, Op::NoTrans) && matches!(trans_b, Op::NoTrans)
        && lda == k && ldb == n && ldc == n {
        return crate::matmul::sgemm(a, b, c, m, n, k, alpha, beta, &Config::default());
    }

    scale_c(c, m, n, ldc, beta);
    if alpha == 0.0 || k == 0 { return Ok(()); }

    let a_get = |i: usize, kk: usize| -> f32 {
        match trans_a { Op::NoTrans => a[i * lda + kk], Op::Trans => a[kk * lda + i] }
    };
    let b_get = |kk: usize, j: usize| -> f32 {
        match trans_b { Op::NoTrans => b[kk * ldb + j], Op::Trans => b[j * ldb + kk] }
    };

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc = (a_get(i, kk) * b_get(kk, j)) + acc;
            }
            c[i * ldc + j] += alpha * acc;
        }
    }
    Ok(())
}

// ---------- ssymm -----------------------------------------------------------

/// LEFT:  C := alpha * A * B + beta * C, A is M×M symmetric.
/// RIGHT: C := alpha * B * A + beta * C, A is N×N symmetric.
pub fn ssymm(
    side: Side, uplo: UpLo,
    m: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) -> Result<()> {
    if m == 0 || n == 0 { return Ok(()); }
    let a_dim = match side { Side::Left => m, Side::Right => n };
    if lda < a_dim { return Err(RacError::InvalidDimension); }
    if ldb < n { return Err(RacError::InvalidDimension); }
    if ldc < n { return Err(RacError::InvalidDimension); }
    if a.len() < (a_dim - 1) * lda + a_dim { return Err(RacError::InvalidDimension); }
    if b.len() < (m - 1) * ldb + n { return Err(RacError::InvalidDimension); }
    if c.len() < (m - 1) * ldc + n { return Err(RacError::InvalidDimension); }

    scale_c(c, m, n, ldc, beta);
    if alpha == 0.0 { return Ok(()); }

    match side {
        Side::Left => {
            // C[i,j] += alpha * sum_k A_full(i,k) * B[k,j]
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for kk in 0..m {
                        acc = (sym_get(a, lda, i, kk, uplo) * b[kk * ldb + j]) + acc;
                    }
                    c[i * ldc + j] += alpha * acc;
                }
            }
        }
        Side::Right => {
            // C[i,j] += alpha * sum_k B[i,k] * A_full(k,j)
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for kk in 0..n {
                        acc = (b[i * ldb + kk] * sym_get(a, lda, kk, j, uplo)) + acc;
                    }
                    c[i * ldc + j] += alpha * acc;
                }
            }
        }
    }
    Ok(())
}

// ---------- ssyrk -----------------------------------------------------------

/// C := alpha * op(A) * op(A)^T + beta * C, only `uplo` triangle of C touched.
pub fn ssyrk(
    uplo: UpLo, trans: Op,
    n: usize, k: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) -> Result<()> {
    if n == 0 { return Ok(()); }
    let (a_rows, a_cols) = match trans { Op::NoTrans => (n, k), Op::Trans => (k, n) };
    if lda < a_cols { return Err(RacError::InvalidDimension); }
    if ldc < n { return Err(RacError::InvalidDimension); }
    if k > 0 && a.len() < (a_rows - 1) * lda + a_cols { return Err(RacError::InvalidDimension); }
    if c.len() < (n - 1) * ldc + n { return Err(RacError::InvalidDimension); }

    // Scale C in chosen triangle only
    if beta != 1.0 {
        for i in 0..n {
            let (jstart, jend) = match uplo {
                UpLo::Upper => (i, n),
                UpLo::Lower => (0, i + 1),
            };
            for j in jstart..jend {
                c[i * ldc + j] = if beta == 0.0 { 0.0 } else { c[i * ldc + j] * beta };
            }
        }
    }
    if alpha == 0.0 || k == 0 { return Ok(()); }

    let a_row = |i: usize, kk: usize| -> f32 {
        match trans { Op::NoTrans => a[i * lda + kk], Op::Trans => a[kk * lda + i] }
    };

    for i in 0..n {
        let (jstart, jend) = match uplo {
            UpLo::Upper => (i, n),
            UpLo::Lower => (0, i + 1),
        };
        for j in jstart..jend {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc = (a_row(i, kk) * a_row(j, kk)) + acc;
            }
            c[i * ldc + j] += alpha * acc;
        }
    }
    Ok(())
}

// ---------- ssyr2k ----------------------------------------------------------

/// C := alpha*(op(A)*op(B)^T + op(B)*op(A)^T) + beta*C, `uplo` triangle only.
pub fn ssyr2k(
    uplo: UpLo, trans: Op,
    n: usize, k: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) -> Result<()> {
    if n == 0 { return Ok(()); }
    let (rows, cols) = match trans { Op::NoTrans => (n, k), Op::Trans => (k, n) };
    if lda < cols { return Err(RacError::InvalidDimension); }
    if ldb < cols { return Err(RacError::InvalidDimension); }
    if ldc < n { return Err(RacError::InvalidDimension); }
    if k > 0 {
        if a.len() < (rows - 1) * lda + cols { return Err(RacError::InvalidDimension); }
        if b.len() < (rows - 1) * ldb + cols { return Err(RacError::InvalidDimension); }
    }
    if c.len() < (n - 1) * ldc + n { return Err(RacError::InvalidDimension); }

    if beta != 1.0 {
        for i in 0..n {
            let (jstart, jend) = match uplo {
                UpLo::Upper => (i, n),
                UpLo::Lower => (0, i + 1),
            };
            for j in jstart..jend {
                c[i * ldc + j] = if beta == 0.0 { 0.0 } else { c[i * ldc + j] * beta };
            }
        }
    }
    if alpha == 0.0 || k == 0 { return Ok(()); }

    let a_row = |i: usize, kk: usize| -> f32 {
        match trans { Op::NoTrans => a[i * lda + kk], Op::Trans => a[kk * lda + i] }
    };
    let b_row = |i: usize, kk: usize| -> f32 {
        match trans { Op::NoTrans => b[i * ldb + kk], Op::Trans => b[kk * ldb + i] }
    };

    for i in 0..n {
        let (jstart, jend) = match uplo {
            UpLo::Upper => (i, n),
            UpLo::Lower => (0, i + 1),
        };
        for j in jstart..jend {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc = (a_row(i, kk) * b_row(j, kk)) + acc;
                acc = (b_row(i, kk) * a_row(j, kk)) + acc;
            }
            c[i * ldc + j] += alpha * acc;
        }
    }
    Ok(())
}

// ---------- strmm -----------------------------------------------------------

/// LEFT:  B := alpha * op(A) * B,  A is M×M triangular.
/// RIGHT: B := alpha * B * op(A),  A is N×N triangular.
pub fn strmm(
    side: Side, uplo: UpLo, trans: Op, diag: Diag,
    m: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &mut [f32], ldb: usize,
) -> Result<()> {
    if m == 0 || n == 0 { return Ok(()); }
    let a_dim = match side { Side::Left => m, Side::Right => n };
    if lda < a_dim { return Err(RacError::InvalidDimension); }
    if ldb < n { return Err(RacError::InvalidDimension); }
    if a.len() < (a_dim - 1) * lda + a_dim { return Err(RacError::InvalidDimension); }
    if b.len() < (m - 1) * ldb + n { return Err(RacError::InvalidDimension); }

    if alpha == 0.0 {
        for i in 0..m { for j in 0..n { b[i * ldb + j] = 0.0; } }
        return Ok(());
    }

    // op_a(i,j) -> entry of op(A) at (i,j). If trans, swap indices in tri_get.
    let op_a = |i: usize, j: usize| -> f32 {
        match trans {
            Op::NoTrans => tri_get(a, lda, i, j, uplo, diag),
            Op::Trans   => tri_get(a, lda, j, i, uplo, diag),
        }
    };

    match side {
        Side::Left => {
            // For each column j of B:  newcol = op(A) * col;  col := alpha * newcol.
            let mut col = vec![0.0f32; m];
            let mut newcol = vec![0.0f32; m];
            for j in 0..n {
                for i in 0..m { col[i] = b[i * ldb + j]; }
                for i in 0..m {
                    let mut acc = 0.0f32;
                    for kk in 0..m {
                        acc = (op_a(i, kk) * col[kk]) + acc;
                    }
                    newcol[i] = acc;
                }
                for i in 0..m { b[i * ldb + j] = alpha * newcol[i]; }
            }
        }
        Side::Right => {
            // For each row i of B:  newrow = row * op(A);  row := alpha * newrow.
            let mut row = vec![0.0f32; n];
            let mut newrow = vec![0.0f32; n];
            for i in 0..m {
                for j in 0..n { row[j] = b[i * ldb + j]; }
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for kk in 0..n {
                        acc = (row[kk] * op_a(kk, j)) + acc;
                    }
                    newrow[j] = acc;
                }
                for j in 0..n { b[i * ldb + j] = alpha * newrow[j]; }
            }
        }
    }
    Ok(())
}

// ---------- strsm -----------------------------------------------------------

fn flip_op(op: Op) -> Op {
    match op { Op::NoTrans => Op::Trans, Op::Trans => Op::NoTrans }
}

/// LEFT:  Solve op(A) * X = alpha * B, result in B.  A is M×M triangular.
/// RIGHT: Solve X * op(A) = alpha * B, result in B.  A is N×N triangular.
pub fn strsm(
    side: Side, uplo: UpLo, trans: Op, diag: Diag,
    m: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &mut [f32], ldb: usize,
) -> Result<()> {
    if m == 0 || n == 0 { return Ok(()); }
    let a_dim = match side { Side::Left => m, Side::Right => n };
    if lda < a_dim { return Err(RacError::InvalidDimension); }
    if ldb < n { return Err(RacError::InvalidDimension); }
    if a.len() < (a_dim - 1) * lda + a_dim { return Err(RacError::InvalidDimension); }
    if b.len() < (m - 1) * ldb + n { return Err(RacError::InvalidDimension); }

    // Scale B by alpha
    if alpha != 1.0 {
        for i in 0..m { for j in 0..n { b[i * ldb + j] *= alpha; } }
    }

    match side {
        Side::Left => {
            // For each column j of B, run strsv on the column.
            let mut col = vec![0.0f32; m];
            for j in 0..n {
                for i in 0..m { col[i] = b[i * ldb + j]; }
                super::l2::strsv(uplo, trans, diag, m, a, lda, &mut col, 1)?;
                for i in 0..m { b[i * ldb + j] = col[i]; }
            }
        }
        Side::Right => {
            // For each row i of B, transpose-trick: x*op(A)=b  <=>  op(A)^T x^T = b^T.
            // -> call strsv with (uplo unchanged, trans flipped, diag unchanged).
            let mut row = vec![0.0f32; n];
            let trans_eff = flip_op(trans);
            for i in 0..m {
                for j in 0..n { row[j] = b[i * ldb + j]; }
                super::l2::strsv(uplo, trans_eff, diag, n, a, lda, &mut row, 1)?;
                for j in 0..n { b[i * ldb + j] = row[j]; }
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

    fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
        // Tiny LCG so tests are deterministic without a dep.
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n).map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }).collect()
    }

    #[test]
    fn test_sgemm_ex_no_trans_matches_matmul() {
        let m = 8; let n = 6; let k = 12;
        let a = rand_vec(m * k, 1);
        let b = rand_vec(k * n, 2);
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        sgemm_ex(Op::NoTrans, Op::NoTrans, m, n, k, 1.0, &a, k, &b, n, 0.0, &mut c1, n).unwrap();
        crate::matmul::matmul(&a, &b, &mut c2, m, n, k, &Config::default()).unwrap();
        assert!(maxabs(&c1, &c2) < 1e-3, "diff={}", maxabs(&c1, &c2));
    }

    #[test]
    fn test_sgemm_ex_trans_a() {
        let m = 5; let n = 4; let k = 3;
        // A_t is K×M (trans means we treat input as K x M, conceptual A is M x K)
        let a_t = rand_vec(k * m, 11); // stored K rows of M elements
        let b = rand_vec(k * n, 12);
        // Reference: build A = transpose(a_t)
        let mut a_full = vec![0.0f32; m * k];
        for i in 0..m { for j in 0..k { a_full[i * k + j] = a_t[j * m + i]; } }
        let mut c1 = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        sgemm_ex(Op::Trans, Op::NoTrans, m, n, k, 1.0, &a_t, m, &b, n, 0.0, &mut c1, n).unwrap();
        crate::matmul::matmul(&a_full, &b, &mut c_ref, m, n, k, &Config::default()).unwrap();
        assert!(maxabs(&c1, &c_ref) < 1e-3);
    }

    #[test]
    fn test_sgemm_ex_trans_b() {
        let m = 4; let n = 5; let k = 3;
        let a = rand_vec(m * k, 21);
        let b_t = rand_vec(n * k, 22); // B^T is N×K
        let mut b_full = vec![0.0f32; k * n];
        for i in 0..k { for j in 0..n { b_full[i * n + j] = b_t[j * k + i]; } }
        let mut c1 = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        sgemm_ex(Op::NoTrans, Op::Trans, m, n, k, 1.0, &a, k, &b_t, k, 0.0, &mut c1, n).unwrap();
        crate::matmul::matmul(&a, &b_full, &mut c_ref, m, n, k, &Config::default()).unwrap();
        assert!(maxabs(&c1, &c_ref) < 1e-3);
    }

    #[test]
    fn test_sgemm_ex_trans_both() {
        let m = 4; let n = 3; let k = 5;
        let a_t = rand_vec(k * m, 31);
        let b_t = rand_vec(n * k, 32);
        let mut a_full = vec![0.0f32; m * k];
        for i in 0..m { for j in 0..k { a_full[i * k + j] = a_t[j * m + i]; } }
        let mut b_full = vec![0.0f32; k * n];
        for i in 0..k { for j in 0..n { b_full[i * n + j] = b_t[j * k + i]; } }
        let mut c1 = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        sgemm_ex(Op::Trans, Op::Trans, m, n, k, 1.0, &a_t, m, &b_t, k, 0.0, &mut c1, n).unwrap();
        crate::matmul::matmul(&a_full, &b_full, &mut c_ref, m, n, k, &Config::default()).unwrap();
        assert!(maxabs(&c1, &c_ref) < 1e-3);
    }

    #[test]
    fn test_ssymm_left_upper() {
        let n = 4; let m = n;
        let a_upper = [
            1.0, 2.0, 3.0, 4.0,
            0.0, 5.0, 6.0, 7.0,
            0.0, 0.0, 8.0, 9.0,
            0.0, 0.0, 0.0, 10.0];
        let a_full = [
            1.0, 2.0, 3.0, 4.0,
            2.0, 5.0, 6.0, 7.0,
            3.0, 6.0, 8.0, 9.0,
            4.0, 7.0, 9.0, 10.0];
        let b: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let mut c1 = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        ssymm(Side::Left, UpLo::Upper, m, n, 1.0, &a_upper, n, &b, n, 0.0, &mut c1, n).unwrap();
        crate::matmul::matmul(&a_full, &b, &mut c_ref, m, n, n, &Config::default()).unwrap();
        assert!(maxabs(&c1, &c_ref) < 1e-3);
    }

    #[test]
    fn test_ssyrk_upper_no_trans_only_touches_chosen_triangle() {
        let n = 4; let k = 3;
        let a = rand_vec(n * k, 41);
        let mut c = vec![-99.0f32; n * n]; // sentinel everywhere
        // Initialize chosen (upper) triangle to 0
        for i in 0..n { for j in i..n { c[i * n + j] = 0.0; } }
        ssyrk(UpLo::Upper, Op::NoTrans, n, k, 1.0, &a, k, 0.0, &mut c, n).unwrap();
        // Verify: upper triangle == sum_k a[i,k]*a[j,k]; lower triangle still -99 sentinel.
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    assert!((c[i * n + j] - (-99.0)).abs() < 1e-5,
                        "lower (i={i},j={j}) should be sentinel, got {}", c[i * n + j]);
                } else {
                    let mut e = 0.0f32;
                    for kk in 0..k { e += a[i * k + kk] * a[j * k + kk]; }
                    assert!((c[i * n + j] - e).abs() < 1e-3,
                        "upper (i={i},j={j}) got {} expected {}", c[i * n + j], e);
                }
            }
        }
    }

    #[test]
    fn test_strmm_left_lower_no_trans_non_unit() {
        let m = 3; let n = 2;
        let a = [1.0, 0.0, 0.0,
                 2.0, 3.0, 0.0,
                 4.0, 5.0, 6.0];
        let mut b = [1.0f32, 1.0,
                     1.0, 1.0,
                     1.0, 1.0];
        strmm(Side::Left, UpLo::Lower, Op::NoTrans, Diag::NonUnit,
              m, n, 1.0, &a, m, &mut b, n).unwrap();
        // Expected (each col): [1, 5, 15]
        let exp = [1.0, 1.0,
                   5.0, 5.0,
                   15.0, 15.0];
        assert!(maxabs(&b, &exp) < 1e-5);
    }

    #[test]
    fn test_strsm_left_lower_round_trip() {
        let m = 4; let n = 3;
        // Build well-conditioned lower triangular A.
        let mut a = vec![0.0f32; m * m];
        for i in 0..m {
            for j in 0..=i {
                a[i * m + j] = if i == j { 1.5 + i as f32 * 0.1 } else { 0.2 + 0.05 * (i + j) as f32 };
            }
        }
        let x_known = rand_vec(m * n, 51);
        // Compute B = A * X via strmm
        let mut b = x_known.clone();
        strmm(Side::Left, UpLo::Lower, Op::NoTrans, Diag::NonUnit,
              m, n, 1.0, &a, m, &mut b, n).unwrap();
        // Solve back
        strsm(Side::Left, UpLo::Lower, Op::NoTrans, Diag::NonUnit,
              m, n, 1.0, &a, m, &mut b, n).unwrap();
        assert!(maxabs(&b, &x_known) < 1e-3,
            "diff={}", maxabs(&b, &x_known));
    }
}
