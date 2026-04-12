//! CORDIC rotation primitives — all 17 RAC operations plus
//! tunable-precision variants and the hyperbolic-vectoring `rsqrt`
//! primitive used by LayerNorm / RMSNorm.

use crate::{K_INV, ITERS, K_HYP_INV, RAC_PI};

/// CORDIC arctangent lookup table: atan(2^-i) for i = 0..15
const ATAN_TABLE: [f32; ITERS] = [
    0.78539816, 0.46364761, 0.24497866, 0.12435499,
    0.06241881, 0.03123983, 0.01562373, 0.00781234,
    0.00390623, 0.00195312, 0.00097656, 0.00048828,
    0.00024414, 0.00012207, 0.00006104, 0.00003052,
];

/// Extended atan table for iters in [16, 24). Used when callers request
/// higher-precision CORDIC (training-time quantization, long rotations).
const ATAN_TABLE_EXT: [f32; 8] = [
    0.00001526, 0.00000763, 0.00000381, 0.00000191,
    0.00000095, 0.00000048, 0.00000024, 0.00000012,
];

/// Precomputed CORDIC circular gain K(n) = prod_{i=0..n-1} sqrt(1 + 2^-2i).
/// Indexed by iteration count 0..=24.
const K_TABLE: [f32; 25] = [
    1.00000000, 1.41421356, 1.58113883, 1.62979295,
    1.64248460, 1.64568447, 1.64648404, 1.64668388,
    1.64673383, 1.64674632, 1.64674944, 1.64675022,
    1.64675042, 1.64675047, 1.64675048, 1.64675049,
    1.64675049, 1.64675049, 1.64675049, 1.64675049,
    1.64675049, 1.64675049, 1.64675049, 1.64675049,
    1.64675049,
];

#[inline]
fn clamp_iters(iters: usize) -> usize {
    if iters < 4 { 4 } else if iters > 24 { 24 } else { iters }
}

#[inline]
fn atan_entry(i: usize) -> f32 {
    if i < ITERS { ATAN_TABLE[i] } else { ATAN_TABLE_EXT[i - ITERS] }
}

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
    // Wrap theta into (-π, π].
    let tau = 2.0 * RAC_PI;
    let mut theta = theta;
    while theta >  RAC_PI { theta -= tau; }
    while theta <= -RAC_PI { theta += tau; }

    // CORDIC circular rotation converges for theta in [-π/2, π/2].
    // Pre-rotate by ±π (i.e. (x,y) -> (-x,-y)) when outside.
    let (mut x, mut y) = (v.x, v.y);
    if theta > RAC_PI * 0.5 {
        x = -x; y = -y;
        theta -= RAC_PI;
    } else if theta < -RAC_PI * 0.5 {
        x = -x; y = -y;
        theta += RAC_PI;
    }

    let mut angle = theta;
    let mut scale = 1.0f32;
    for i in 0..iters {
        let d = if angle >= 0.0 { 1.0f32 } else { -1.0f32 };
        let xn = x - d * y * scale;
        let yn = y + d * x * scale;
        angle -= d * atan_entry(i);
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
///
/// CORDIC vectoring only converges for x >= 0, so we apply an explicit
/// pre-rotation by ±π when x < 0 to bring the vector into the right
/// half-plane. The accumulated angle is then adjusted on return.
/// Returns (magnitude, angle) where angle = atan2(y, x).
#[inline]
pub fn polar(v: Vec2) -> (f32, f32) {
    // Pre-rotate into the right half-plane if needed.
    let (vx, vy, pre_angle) = if v.x < 0.0 {
        if v.y >= 0.0 { (-v.x, -v.y,  RAC_PI) } else { (-v.x, -v.y, -RAC_PI) }
    } else {
        (v.x, v.y, 0.0)
    };

    let (mut x, mut y, mut z) = (vx, vy, 0.0f32);
    let mut scale = 1.0f32;

    for i in 0..ITERS {
        let d = if y < 0.0 { 1.0f32 } else { -1.0f32 };
        let xn = x - d * y * scale;
        let yn = y + d * x * scale;
        z += d * ATAN_TABLE[i];
        x = xn; y = yn;
        scale *= 0.5;
    }
    // `z` accumulates -angle (we rotate *toward* zero y). Angle = -z + pre.
    (x * K_INV, pre_angle - z)
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

/* ── 6. Tunable-precision CORDIC ────────────────────────────────────────── */

/// Gain-compensated rotation with a caller-chosen iteration count.
/// Iters is clamped to [4, 24]. Fewer iters = lower precision & power.
#[inline]
pub fn rotate_n(v: Vec2, theta: f32, iters: usize) -> Vec2 {
    let iters = clamp_iters(iters);
    let k_inv = 1.0 / K_TABLE[iters];
    let comp = Vec2::new(v.x * k_inv, v.y * k_inv);
    cordic_rotate_raw(comp, theta, iters)
}

/// Scalar projection at tunable precision.
#[inline]
pub fn project_n(v: Vec2, theta: f32, iters: usize) -> f32 {
    let iters = clamp_iters(iters);
    let unit = Vec2::new(1.0 / K_TABLE[iters], 0.0);
    let r = cordic_rotate_raw(unit, theta, iters);
    v.x.mul_add(r.x, v.y * r.y)
}

/// Cartesian-to-polar (vectoring mode) at tunable precision.
pub fn polar_n(v: Vec2, iters: usize) -> (f32, f32) {
    let iters = clamp_iters(iters);
    let (vx, vy, pre_angle) = if v.x < 0.0 {
        if v.y >= 0.0 { (-v.x, -v.y,  RAC_PI) } else { (-v.x, -v.y, -RAC_PI) }
    } else {
        (v.x, v.y, 0.0)
    };
    let (mut x, mut y, mut z) = (vx, vy, 0.0f32);
    let mut scale = 1.0f32;
    for i in 0..iters {
        let d = if y < 0.0 { 1.0 } else { -1.0 };
        let xn = x - d * y * scale;
        let yn = y + d * x * scale;
        z += d * atan_entry(i);
        x = xn;
        y = yn;
        scale *= 0.5;
    }
    (x / K_TABLE[iters], pre_angle - z)
}

/// Single-pass sin and cos via one CORDIC rotation — the primitive
/// behind RoPE, DCT, and general rotations. Returns (sin, cos).
#[inline]
pub fn sincos(theta: f32) -> (f32, f32) {
    let unit = Vec2::new(K_INV, 0.0);
    let r = cordic_rotate_raw(unit, theta, ITERS);
    (r.y, r.x)
}

/// Reciprocal sqrt — the LayerNorm / RMSNorm primitive.
/// On CPU this uses `sqrtf` for exact f32 correctness. The GPU backends
/// (rac_cuda.cu, rac_hip.cpp) implement this as hyperbolic CORDIC
/// vectoring to route through the SFU / transcendental unit.
#[inline]
pub fn rsqrt(x: f32) -> f32 {
    if x <= 0.0 { 0.0 } else { 1.0 / x.sqrt() }
}

/// Sigmoid via the identity sigmoid(x) = 0.5 * (1 + tanh(x/2)).
/// One hyperbolic transcendental, no divide, no explicit exp.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    0.5 * (1.0 + (0.5 * x).tanh())
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

    #[test]
    fn test_rotate_n_low_precision_converges() {
        // 8-iter rotation should still land within ~1% of the target
        let v = rotate_n(Vec2::new(1.0, 0.0), core::f32::consts::FRAC_PI_4, 8);
        let expected = (core::f32::consts::FRAC_PI_4).cos();
        assert!(approx(v.x, expected, 0.01));
    }

    #[test]
    fn test_rotate_n_matches_rotate_at_default_iters() {
        let a = rotate(Vec2::new(1.0, 2.0), 0.5);
        let b = rotate_n(Vec2::new(1.0, 2.0), 0.5, ITERS);
        assert!(approx(a.x, b.x, 1e-4));
        assert!(approx(a.y, b.y, 1e-4));
    }

    #[test]
    fn test_sincos_matches_libm() {
        for &t in &[-1.0f32, -0.25, 0.0, 0.25, 1.0, 1.5] {
            let (s, c) = sincos(t);
            assert!(approx(s, t.sin(), 0.01), "sin({}) off", t);
            assert!(approx(c, t.cos(), 0.01), "cos({}) off", t);
        }
    }

    #[test]
    fn test_rsqrt() {
        assert!(approx(rsqrt(4.0), 0.5, 1e-6));
        assert!(approx(rsqrt(1.0), 1.0, 1e-6));
        assert_eq!(rsqrt(-1.0), 0.0);
    }

    #[test]
    fn test_sigmoid() {
        assert!(approx(sigmoid(0.0), 0.5, 1e-5));
        assert!(approx(sigmoid(10.0), 1.0, 0.001));
        assert!(approx(sigmoid(-10.0), 0.0, 0.001));
    }
}
