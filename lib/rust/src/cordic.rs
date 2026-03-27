//! CORDIC rotation primitives — all 17 RAC operations

use crate::{K_INV, ITERS, K_HYP_INV, RAC_PI};

/// CORDIC arctangent lookup table: atan(2^-i) for i = 0..15
const ATAN_TABLE: [f32; ITERS] = [
    0.78539816, 0.46364761, 0.24497866, 0.12435499,
    0.06241881, 0.03123983, 0.01562373, 0.00781234,
    0.00390623, 0.00195312, 0.00097656, 0.00048828,
    0.00024414, 0.00012207, 0.00006104, 0.00003052,
];

/// Hyperbolic CORDIC atanh table: atanh(2^-i) for i = 1..16
const ATANH_TABLE: [f32; ITERS] = [
    0.54930614, 0.25541281, 0.12565721, 0.06258157,
    0.03126017, 0.01562627, 0.00781265, 0.00390626,
    0.00195313, 0.00097656, 0.00048828, 0.00024414,
    0.00012207, 0.00006104, 0.00003052, 0.00001526,
];

const HYPER_ITER_MAP: [usize; ITERS] = [1,2,3,4,4,5,6,7,8,9,10,11,12,13,13,14];

/// 2D vector
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self { Self { x, y } }

    #[inline]
    pub fn norm(self) -> f32 { norm(self) }

    #[inline]
    pub fn normalized(self) -> Self { normalize(self) }

    #[inline]
    pub fn rotate(self, theta: f32) -> Self { rotate(self, theta) }

    #[inline]
    pub fn project(self, theta: f32) -> f32 { project(self, theta) }
}

/* ── Core CORDIC rotation ───────────────────────────────────────────────── */

#[inline]
fn cordic_rotate_raw(v: Vec2, theta: f32, iters: usize) -> Vec2 {
    let (mut x, mut y, mut angle) = (v.x, v.y, theta);
    let mut scale = 1.0f32;

    for i in 0..iters {
        let d = if angle >= 0.0 { 1.0f32 } else { -1.0f32 };
        let xn = x - d * y * scale;
        let yn = y + d * x * scale;
        angle -= d * ATAN_TABLE[i];
        x = xn;
        y = yn;
        scale *= 0.5;
    }
    Vec2 { x, y }
}

#[inline]
fn cordic_hyperbolic(x_in: f32, y_in: f32, z_in: f32) -> Vec2 {
    let (mut x, mut y, mut z) = (x_in, y_in, z_in);
    let mut scale = 0.5f32;

    for i in 0..ITERS {
        let d = if z >= 0.0 { 1.0f32 } else { -1.0f32 };
        let xn = x + d * y * scale;
        let yn = y + d * x * scale;
        z -= d * ATANH_TABLE[HYPER_ITER_MAP[i] - 1];
        x = xn;
        y = yn;
        if i != 3 && i != 12 { scale *= 0.5; }
    }
    Vec2 { x, y }
}

/* ── 1. Core rotation ───────────────────────────────────────────────────── */

/// Gain-compensated rotation. Output magnitude == input magnitude.
#[inline]
pub fn rotate(v: Vec2, theta: f32) -> Vec2 {
    let comp = Vec2::new(v.x * K_INV, v.y * K_INV);
    cordic_rotate_raw(comp, theta, ITERS)
}

/// Raw rotation (no gain compensation). Output magnitude scaled by K.
#[inline]
pub fn rotate_raw(v: Vec2, theta: f32) -> Vec2 {
    cordic_rotate_raw(v, theta, ITERS)
}

/// Scalar projection: v.x*cos(theta) + v.y*sin(theta)
#[inline]
pub fn project(v: Vec2, theta: f32) -> f32 {
    let (s, c) = theta.sin_cos();
    v.x.mul_add(c, v.y * s)
}

/* ── 2. Polar / vectoring ───────────────────────────────────────────────── */

/// Cartesian to polar via CORDIC vectoring.
#[inline]
pub fn polar(v: Vec2) -> (f32, f32) {
    let (mut x, mut y, mut z) = (v.x, v.y, 0.0f32);
    let mut scale = 1.0f32;

    for i in 0..ITERS {
        let d = if y < 0.0 { 1.0f32 } else { -1.0f32 };
        let xn = x - d * y * scale;
        let yn = y + d * x * scale;
        z += d * ATAN_TABLE[i];
        x = xn; y = yn;
        scale *= 0.5;
    }
    (x * K_INV, z) // (magnitude, angle)
}

/// Euclidean norm via CORDIC.
#[inline]
pub fn norm(v: Vec2) -> f32 { polar(v).0 }

/// Unit vector via fused vectoring + rotation.
#[inline]
pub fn normalize(v: Vec2) -> Vec2 {
    let (_, angle) = polar(v);
    let (s, c) = angle.sin_cos();
    Vec2::new(c, s)
}

/* ── 3. Dot product / similarity ────────────────────────────────────────── */

#[inline]
pub fn dot(a: Vec2, b: Vec2) -> f32 {
    let (ma, aa) = polar(a);
    let (mb, ab) = polar(b);
    ma * mb * (aa - ab).cos()
}

#[inline]
pub fn coherence(a: Vec2, b: Vec2) -> f32 {
    let (_, aa) = polar(a);
    let (_, ab) = polar(b);
    (aa - ab).cos()
}

/* ── 4. Complex / DSP ───────────────────────────────────────────────────── */

#[inline]
pub fn complex_mul(a: Vec2, b: Vec2) -> Vec2 {
    let (mb, ab) = polar(b);
    let r = rotate(a, ab);
    Vec2::new(r.x * mb, r.y * mb)
}

pub fn dct(x: &[f32], out: &mut [f32]) {
    let n = x.len();
    for k in 0..n {
        let mut sum = 0.0f32;
        for i in 0..n {
            let theta = RAC_PI * (2 * i + 1) as f32 * k as f32 / (2 * n) as f32;
            sum += project(Vec2::new(x[i], 0.0), theta);
        }
        out[k] = sum;
    }
}

/* ── 5. Hyperbolic / activations ────────────────────────────────────────── */

#[inline]
pub fn exp(x: f32) -> f32 {
    cordic_hyperbolic(K_HYP_INV, K_HYP_INV, x).x
}

#[inline]
pub fn tanh(x: f32) -> f32 {
    let r = cordic_hyperbolic(K_HYP_INV, 0.0, x);
    r.y / r.x
}

pub fn softmax(x: &[f32], out: &mut [f32]) {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (i, &xi) in x.iter().enumerate() {
        out[i] = exp(xi - max_val);
        sum += out[i];
    }
    let inv = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }

    #[test]
    fn test_rotate_identity() {
        let v = rotate(Vec2::new(1.0, 0.0), 0.0);
        assert!(approx(v.x, 1.0, 0.02));
        assert!(approx(v.y, 0.0, 0.02));
    }

    #[test]
    fn test_rotate_90() {
        let v = rotate(Vec2::new(1.0, 0.0), core::f32::consts::FRAC_PI_2);
        assert!(approx(v.x, 0.0, 0.02));
        assert!(approx(v.y, 1.0, 0.02));
    }

    #[test]
    fn test_magnitude_preservation() {
        let v = rotate(Vec2::new(3.0, 4.0), 1.23);
        let mag = (v.x * v.x + v.y * v.y).sqrt();
        assert!(approx(mag, 5.0, 0.05));
    }

    #[test]
    fn test_project() {
        assert!(approx(project(Vec2::new(1.0, 0.0), 0.0), 1.0, 0.01));
        assert!(approx(project(Vec2::new(1.0, 0.0), RAC_PI), -1.0, 0.01));
    }

    #[test]
    fn test_polar() {
        let (mag, angle) = polar(Vec2::new(3.0, 4.0));
        assert!(approx(mag, 5.0, 0.05));
        assert!(approx(angle, (4.0f32).atan2(3.0), 0.02));
    }

    #[test]
    fn test_exp() {
        assert!(approx(exp(0.0), 1.0, 0.02));
        assert!(approx(exp(1.0), core::f32::consts::E, 0.1));
    }

    #[test]
    fn test_tanh() {
        assert!(approx(tanh(0.0), 0.0, 0.02));
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        softmax(&x, &mut out);
        let sum: f32 = out.iter().sum();
        assert!(approx(sum, 1.0, 0.02));
    }

    #[test]
    fn test_dot_parallel() {
        let d = dot(Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0));
        assert!(approx(d, 1.0, 0.02));
    }

    #[test]
    fn test_dot_orthogonal() {
        let d = dot(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0));
        assert!(approx(d, 0.0, 0.02));
    }

    #[test]
    fn test_coherence() {
        assert!(approx(coherence(Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0)), 1.0, 0.02));
        assert!(approx(coherence(Vec2::new(1.0, 0.0), Vec2::new(-1.0, 0.0)), -1.0, 0.02));
    }
}
