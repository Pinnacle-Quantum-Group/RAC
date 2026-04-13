//! BLAS Level 1 — vector-vector ops. PQG / Michael A. Doran Jr. — April 2026
//!
//! Single-precision (f32) BLAS-1: saxpy, sdot, snrm2, sasum, isamax,
//! sscal, scopy, sswap, srot, srotg.
//!
//! Conventions:
//! - `incx`, `incy` are positive `usize` strides; `0` is treated as `1`.
//! - `n == 0` is a no-op (returns `0.0` for accumulator-style functions).
//! - If a slice is shorter than required, iteration is silently capped — no panics.
//! - Pure safe Rust; no `unsafe`, no `f32::mul_add` (for `no_std` portability).

// ---------- helpers ----------------------------------------------------------

#[inline(always)]
fn norm_stride(inc: usize) -> usize {
    if inc == 0 { 1 } else { inc }
}

/// Maximum number of stride-`inc` elements addressable in a slice of `len`.
#[inline(always)]
fn capacity(len: usize, inc: usize) -> usize {
    if len == 0 { 0 } else { (len - 1) / inc + 1 }
}

// ---------- SAXPY: y := alpha*x + y -----------------------------------------

pub fn saxpy(n: usize, alpha: f32, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
    if n == 0 || alpha == 0.0 { return; }
    let incx = norm_stride(incx);
    let incy = norm_stride(incy);
    let n = n.min(capacity(x.len(), incx)).min(capacity(y.len(), incy));
    if incx == 1 && incy == 1 {
        for i in 0..n { y[i] = (alpha * x[i]) + y[i]; }
    } else {
        for i in 0..n {
            let xi = x[i * incx];
            y[i * incy] = (alpha * xi) + y[i * incy];
        }
    }
}

// ---------- SDOT: sum(x_i * y_i) --------------------------------------------

pub fn sdot(n: usize, x: &[f32], incx: usize, y: &[f32], incy: usize) -> f32 {
    if n == 0 { return 0.0; }
    let incx = norm_stride(incx);
    let incy = norm_stride(incy);
    let n = n.min(capacity(x.len(), incx)).min(capacity(y.len(), incy));
    let mut acc = 0.0f32;
    if incx == 1 && incy == 1 {
        for i in 0..n { acc = (x[i] * y[i]) + acc; }
    } else {
        for i in 0..n { acc = (x[i * incx] * y[i * incy]) + acc; }
    }
    acc
}

// ---------- SNRM2: sqrt(sum x_i^2) — Lawson-Hanson scaled accumulation ------

pub fn snrm2(n: usize, x: &[f32], incx: usize) -> f32 {
    if n == 0 { return 0.0; }
    let incx = norm_stride(incx);
    let n = n.min(capacity(x.len(), incx));
    let mut scale = 0.0f32;
    let mut ssq = 1.0f32;
    for i in 0..n {
        let xi = x[i * incx];
        if xi != 0.0 {
            let ax = xi.abs();
            if scale < ax {
                let r = scale / ax;
                ssq = 1.0 + ssq * (r * r);
                scale = ax;
            } else {
                let r = ax / scale;
                ssq += r * r;
            }
        }
    }
    if scale == 0.0 { 0.0 } else { scale * ssq.sqrt() }
}

// ---------- SASUM: sum |x_i| ------------------------------------------------

pub fn sasum(n: usize, x: &[f32], incx: usize) -> f32 {
    if n == 0 { return 0.0; }
    let incx = norm_stride(incx);
    let n = n.min(capacity(x.len(), incx));
    let mut acc = 0.0f32;
    if incx == 1 {
        for i in 0..n { acc += x[i].abs(); }
    } else {
        for i in 0..n { acc += x[i * incx].abs(); }
    }
    acc
}

// ---------- ISAMAX: argmax_i |x_i| (0-based) --------------------------------

pub fn isamax(n: usize, x: &[f32], incx: usize) -> usize {
    if n == 0 { return 0; }
    let incx = norm_stride(incx);
    let n = n.min(capacity(x.len(), incx));
    if n == 0 { return 0; }
    let mut best_idx = 0usize;
    let mut best_val = x[0].abs();
    for i in 1..n {
        let v = x[i * incx].abs();
        if v > best_val { best_val = v; best_idx = i; }
    }
    best_idx
}

// ---------- SSCAL: x := alpha*x ---------------------------------------------

pub fn sscal(n: usize, alpha: f32, x: &mut [f32], incx: usize) {
    if n == 0 { return; }
    let incx = norm_stride(incx);
    let n = n.min(capacity(x.len(), incx));
    if incx == 1 {
        for i in 0..n { x[i] *= alpha; }
    } else {
        for i in 0..n { x[i * incx] *= alpha; }
    }
}

// ---------- SCOPY: y := x ---------------------------------------------------

pub fn scopy(n: usize, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
    if n == 0 { return; }
    let incx = norm_stride(incx);
    let incy = norm_stride(incy);
    let n = n.min(capacity(x.len(), incx)).min(capacity(y.len(), incy));
    if incx == 1 && incy == 1 {
        y[..n].copy_from_slice(&x[..n]);
    } else {
        for i in 0..n { y[i * incy] = x[i * incx]; }
    }
}

// ---------- SSWAP: x <-> y --------------------------------------------------

pub fn sswap(n: usize, x: &mut [f32], incx: usize, y: &mut [f32], incy: usize) {
    if n == 0 { return; }
    let incx = norm_stride(incx);
    let incy = norm_stride(incy);
    let n = n.min(capacity(x.len(), incx)).min(capacity(y.len(), incy));
    if incx == 1 && incy == 1 {
        for i in 0..n { let t = x[i]; x[i] = y[i]; y[i] = t; }
    } else {
        for i in 0..n {
            let xi = x[i * incx];
            let yi = y[i * incy];
            x[i * incx] = yi;
            y[i * incy] = xi;
        }
    }
}

// ---------- SROT: Givens rotation in place ----------------------------------
//   x' =  c*x + s*y
//   y' = -s*x + c*y

pub fn srot(n: usize, x: &mut [f32], incx: usize, y: &mut [f32], incy: usize, c: f32, s: f32) {
    if n == 0 { return; }
    let incx = norm_stride(incx);
    let incy = norm_stride(incy);
    let n = n.min(capacity(x.len(), incx)).min(capacity(y.len(), incy));
    if incx == 1 && incy == 1 {
        for i in 0..n {
            let xi = x[i];
            let yi = y[i];
            x[i] = (c * xi) + (s * yi);
            y[i] = (c * yi) + (-s * xi);
        }
    } else {
        for i in 0..n {
            let xi = x[i * incx];
            let yi = y[i * incy];
            x[i * incx] = (c * xi) + (s * yi);
            y[i * incy] = (c * yi) + (-s * xi);
        }
    }
}

// ---------- SROTG: construct a Givens rotation (Lawson-Hanson) --------------
//   In : a, b
//   Out: a := r, b := z, c, s

pub fn srotg(a: &mut f32, b: &mut f32, c: &mut f32, s: &mut f32) {
    let aa = a.abs();
    let bb = b.abs();
    let roe = if aa > bb { *a } else { *b };
    let scale = aa + bb;
    let r_out: f32;
    let z_out: f32;
    if scale == 0.0 {
        *c = 1.0;
        *s = 0.0;
        r_out = 0.0;
        z_out = 0.0;
    } else {
        let na = *a / scale;
        let nb = *b / scale;
        let r = (scale * (na * na + nb * nb).sqrt()).copysign(roe);
        *c = *a / r;
        *s = *b / r;
        z_out = if aa > bb { *s }
                else if *c != 0.0 { 1.0 / *c }
                else { 1.0 };
        r_out = r;
    }
    *a = r_out;
    *b = z_out;
}

// ---------- tests -----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    const EPS: f32 = 1e-5;

    #[test]
    fn test_saxpy_unit() {
        let x = [1.0f32, 2.0, 3.0];
        let mut y = [10.0f32, 20.0, 30.0];
        saxpy(3, 2.0, &x, 1, &mut y, 1);
        assert_eq!(y, [12.0, 24.0, 36.0]);
    }

    #[test]
    fn test_sdot_unit() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let y = [5.0f32, 6.0, 7.0, 8.0];
        assert!((sdot(4, &x, 1, &y, 1) - 70.0).abs() < EPS);
    }

    #[test]
    fn test_snrm2_unit() {
        let x = [3.0f32, 4.0];
        assert!((snrm2(2, &x, 1) - 5.0).abs() < EPS);
    }

    #[test]
    fn test_sasum_unit() {
        let x = [1.0f32, -2.0, 3.0, -4.0];
        assert!((sasum(4, &x, 1) - 10.0).abs() < EPS);
    }

    #[test]
    fn test_isamax() {
        let x = [1.0f32, -5.0, 3.0, 4.0];
        assert_eq!(isamax(4, &x, 1), 1);
    }

    #[test]
    fn test_sscal_unit() {
        let mut x = [1.0f32, 2.0, 3.0];
        sscal(3, 0.5, &mut x, 1);
        assert!((x[0] - 0.5).abs() < EPS);
        assert!((x[1] - 1.0).abs() < EPS);
        assert!((x[2] - 1.5).abs() < EPS);
    }

    #[test]
    fn test_scopy_sswap() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut y = [0.0f32; 4];
        scopy(4, &x, 1, &mut y, 1);
        assert_eq!(y, [1.0, 2.0, 3.0, 4.0]);

        let mut a = [10.0f32, 20.0, 30.0];
        let mut b = [100.0f32, 200.0, 300.0];
        sswap(3, &mut a, 1, &mut b, 1);
        assert_eq!(a, [100.0, 200.0, 300.0]);
        assert_eq!(b, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_srot_pi3() {
        let c = 0.5f32;                        // cos(pi/3)
        let s = (3.0f32).sqrt() / 2.0;          // sin(pi/3)
        let mut x = [1.0f32, 2.0, 3.0];
        let mut y = [4.0f32, 5.0, 6.0];
        let xr = [c*x[0] + s*y[0], c*x[1] + s*y[1], c*x[2] + s*y[2]];
        let yr = [c*y[0] - s*x[0], c*y[1] - s*x[1], c*y[2] - s*x[2]];
        srot(3, &mut x, 1, &mut y, 1, c, s);
        for i in 0..3 {
            assert!((x[i] - xr[i]).abs() < 1e-5);
            assert!((y[i] - yr[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_srotg_3_4() {
        let mut a = 3.0f32;
        let mut b = 4.0f32;
        let mut c = 0.0f32;
        let mut s = 0.0f32;
        srotg(&mut a, &mut b, &mut c, &mut s);
        // |b|>|a| ⇒ roe=b=+4 ⇒ r = +5
        assert!((a - 5.0).abs() < 1e-5, "r != 5: got {}", a);
        assert!((c - 0.6).abs() < 1e-5, "c != 0.6: got {}", c);
        assert!((s - 0.8).abs() < 1e-5, "s != 0.8: got {}", s);
    }
}
