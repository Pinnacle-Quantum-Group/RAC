//! Transformer-level primitives built on top of CORDIC.
//!
//! Every operation corresponds to a native CORDIC mode:
//!
//! | op               | CORDIC mode              | primitive          |
//! |------------------|--------------------------|--------------------|
//! | QK^T, attn @ V   | linear MAC               | [`crate::sgemm`]   |
//! | softmax exp      | hyperbolic rotation      | [`crate::exp`]     |
//! | softmax normalize| linear vectoring (divide)| built-in           |
//! | layer-norm mean  | linear accumulate        | built-in           |
//! | layer-norm rsqrt | hyperbolic vectoring     | [`crate::cordic::rsqrt`] |
//! | RMSNorm          | hyperbolic vectoring     | [`crate::cordic::rsqrt`] |
//! | RoPE             | circular rotation        | [`crate::cordic::rotate`]|
//! | GELU / SiLU      | circular + hyperbolic    | [`sigmoid`], [`tanh`]    |
//!
//! RoPE is the standout: rotary position embeddings are literally Givens
//! rotations. RAC computes them natively in circular CORDIC.

use crate::cordic::{rsqrt, sincos};
use crate::{Config, RacError, Result};

/// Layer normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta
///
/// `x` is a `[rows, d]` tensor laid out row-major. `gamma` and `beta`
/// are per-feature parameters (length `d`) or `None` for (1, 0).
/// The inverse-square-root uses the CORDIC `rsqrt` primitive
/// (hyperbolic vectoring on GPU backends).
pub fn layernorm(
    x: &[f32],
    y: &mut [f32],
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    eps: f32,
    rows: usize,
    d: usize,
    _cfg: &Config,
) -> Result<()> {
    if x.len() < rows * d || y.len() < rows * d {
        return Err(RacError::InvalidDimension);
    }
    if let Some(g) = gamma { if g.len() < d { return Err(RacError::InvalidDimension); } }
    if let Some(b) = beta  { if b.len() < d { return Err(RacError::InvalidDimension); } }

    let norm_row = |xr: &[f32], yr: &mut [f32]| {
        let mut mean = 0.0f32;
        for &xi in xr { mean += xi; }
        mean /= d as f32;

        let mut var = 0.0f32;
        for &xi in xr {
            let z = xi - mean;
            var = z.mul_add(z, var);
        }
        var /= d as f32;

        let inv = rsqrt(var + eps);

        for (i, yi) in yr.iter_mut().enumerate() {
            let z = (xr[i] - mean) * inv;
            let g = gamma.map_or(1.0, |g| g[i]);
            let b = beta.map_or(0.0, |b| b[i]);
            *yi = g.mul_add(z, b);
        }
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        x.par_chunks(d).zip(y.par_chunks_mut(d)).for_each(|(xr, yr)| norm_row(xr, yr));
        return Ok(());
    }

    #[cfg(not(feature = "rayon"))]
    {
        for r in 0..rows {
            let xr = &x[r * d..(r + 1) * d];
            let yr = &mut y[r * d..(r + 1) * d];
            norm_row(xr, yr);
        }
        Ok(())
    }
}

/// Root-Mean-Square normalization (Llama / T5 style).
///     y = gamma * x / sqrt(mean(x^2) + eps)
/// No mean subtraction, no beta. One hyperbolic-vectoring rsqrt per row.
pub fn rmsnorm(
    x: &[f32],
    y: &mut [f32],
    gamma: Option<&[f32]>,
    eps: f32,
    rows: usize,
    d: usize,
    _cfg: &Config,
) -> Result<()> {
    if x.len() < rows * d || y.len() < rows * d {
        return Err(RacError::InvalidDimension);
    }
    if let Some(g) = gamma { if g.len() < d { return Err(RacError::InvalidDimension); } }

    let norm_row = |xr: &[f32], yr: &mut [f32]| {
        let mut ms = 0.0f32;
        for &xi in xr { ms = xi.mul_add(xi, ms); }
        ms /= d as f32;
        let inv = rsqrt(ms + eps);
        for (i, yi) in yr.iter_mut().enumerate() {
            let g = gamma.map_or(1.0, |g| g[i]);
            *yi = xr[i] * inv * g;
        }
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        x.par_chunks(d).zip(y.par_chunks_mut(d)).for_each(|(xr, yr)| norm_row(xr, yr));
        return Ok(());
    }

    #[cfg(not(feature = "rayon"))]
    {
        for r in 0..rows {
            let xr = &x[r * d..(r + 1) * d];
            let yr = &mut y[r * d..(r + 1) * d];
            norm_row(xr, yr);
        }
        Ok(())
    }
}

/// Precompute the RoPE sin/cos cache for a given head_dim / max_seq.
/// Output layout: `[max_seq, head_dim/2]` row-major for both `cos` and `sin`.
pub fn rope_cache(
    cos_out: &mut [f32],
    sin_out: &mut [f32],
    max_seq: usize,
    head_dim: usize,
    base: f32,
) -> Result<()> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(RacError::InvalidDimension);
    }
    let half = head_dim / 2;
    if cos_out.len() < max_seq * half || sin_out.len() < max_seq * half {
        return Err(RacError::InvalidDimension);
    }

    for p in 0..max_seq {
        for i in 0..half {
            let exponent = (2.0 * i as f32) / head_dim as f32;
            let inv_freq = 1.0 / base.powf(exponent);
            let angle = p as f32 * inv_freq;
            let (s, c) = sincos(angle);
            cos_out[p * half + i] = c;
            sin_out[p * half + i] = s;
        }
    }
    Ok(())
}

/// Apply RoPE in place to a query or key tensor shaped
/// `[batch, n_heads, seq, head_dim]`. Each adjacent (x[2i], x[2i+1]) pair
/// is a Givens rotation — **native** CORDIC with no multipliers needed
/// on a RAC ASIC.
pub fn rope_apply(
    x: &mut [f32],
    cos_tab: &[f32],
    sin_tab: &[f32],
    batch: usize,
    n_heads: usize,
    seq: usize,
    head_dim: usize,
) -> Result<()> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(RacError::InvalidDimension);
    }
    let half = head_dim / 2;
    let total = batch * n_heads * seq * head_dim;
    if x.len() < total { return Err(RacError::InvalidDimension); }
    if cos_tab.len() < seq * half || sin_tab.len() < seq * half {
        return Err(RacError::InvalidDimension);
    }

    for b in 0..batch {
        for h in 0..n_heads {
            for t in 0..seq {
                let base = ((b * n_heads + h) * seq + t) * head_dim;
                let cr = &cos_tab[t * half..(t + 1) * half];
                let sr = &sin_tab[t * half..(t + 1) * half];
                for i in 0..half {
                    let a = x[base + 2 * i];
                    let b2 = x[base + 2 * i + 1];
                    let c = cr[i];
                    let s = sr[i];
                    x[base + 2 * i]     = a.mul_add(c, -b2 * s);
                    x[base + 2 * i + 1] = a.mul_add(s,  b2 * c);
                }
            }
        }
    }
    Ok(())
}

/// Scaled dot-product attention (reference, single-threaded).
/// Shapes: q/k/v/out are all `[batch, n_heads, seq, head_dim]`.
pub fn scaled_dot_attention(
    q: &[f32], k: &[f32], v: &[f32],
    mask: Option<&[f32]>,
    is_causal: bool,
    out: &mut [f32],
    batch: usize, n_heads: usize, seq: usize, head_dim: usize,
) -> Result<()> {
    let total = batch * n_heads * seq * head_dim;
    if q.len() < total || k.len() < total || v.len() < total || out.len() < total {
        return Err(RacError::InvalidDimension);
    }
    if let Some(m) = mask { if m.len() < seq * seq { return Err(RacError::InvalidDimension); } }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq * seq];

    for b in 0..batch {
        for h in 0..n_heads {
            let qbh = &q[((b * n_heads + h) * seq) * head_dim..];
            let kbh = &k[((b * n_heads + h) * seq) * head_dim..];
            let vbh = &v[((b * n_heads + h) * seq) * head_dim..];
            let obh = &mut out[((b * n_heads + h) * seq) * head_dim..];

            for t in 0..seq {
                for s in 0..seq {
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc = qbh[t * head_dim + d].mul_add(kbh[s * head_dim + d], acc);
                    }
                    let mut sc = acc * scale;
                    if is_causal && s > t { sc = f32::NEG_INFINITY; }
                    if let Some(m) = mask { sc += m[t * seq + s]; }
                    scores[t * seq + s] = sc;
                }
            }

            for t in 0..seq {
                let row = &mut scores[t * seq..(t + 1) * seq];
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in row.iter_mut() { *s = (*s - m).exp(); sum += *s; }
                if sum > 0.0 {
                    let inv = 1.0 / sum;
                    for s in row.iter_mut() { *s *= inv; }
                }

                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for s in 0..seq {
                        acc = row[s].mul_add(vbh[s * head_dim + d], acc);
                    }
                    obh[t * head_dim + d] = acc;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }

    #[test]
    fn test_rmsnorm_preserves_shape_and_scale() {
        let x = vec![3.0f32, 4.0, 0.0, 0.0];  // 2 rows of d=2
        let mut y = vec![0.0f32; 4];
        rmsnorm(&x, &mut y, None, 1e-6, 2, 2, &Config::default()).unwrap();

        // Row 0: ms = (9+16)/2 = 12.5, rsqrt ≈ 0.2828 → y ≈ (0.848, 1.131)
        let expected_norm = (0.848f32 * 0.848 + 1.131 * 1.131).sqrt();
        let got_norm = (y[0] * y[0] + y[1] * y[1]).sqrt();
        assert!(approx(got_norm, expected_norm, 0.01));
    }

    #[test]
    fn test_layernorm_zero_mean_unit_var() {
        // d=4, mean should be zero, variance one per row
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        layernorm(&x, &mut y, None, None, 1e-6, 1, 4, &Config::default()).unwrap();

        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        let var: f32 = y.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(approx(mean, 0.0, 0.01));
        assert!(approx(var, 1.0, 0.05));
    }

    #[test]
    fn test_rope_cache_shape() {
        let (max_seq, head_dim) = (8usize, 8usize);
        let mut cos = vec![0.0f32; max_seq * head_dim / 2];
        let mut sin = vec![0.0f32; max_seq * head_dim / 2];
        rope_cache(&mut cos, &mut sin, max_seq, head_dim, 10000.0).unwrap();
        // At position 0, cos = 1 and sin = 0 for every frequency.
        for i in 0..head_dim / 2 {
            assert!(approx(cos[i], 1.0, 0.01));
            assert!(approx(sin[i], 0.0, 0.01));
        }
    }

    #[test]
    fn test_rope_apply_preserves_norm() {
        // RoPE is a rotation — it must preserve each 2-vector's magnitude.
        let head_dim = 4;
        let seq = 3;
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0,   // t=0
                          0.5, 0.7, -1.0, 2.0,     // t=1
                          1.5, -0.5, 0.0, 1.0];    // t=2
        let orig = x.clone();
        let mut cos = vec![0.0f32; seq * head_dim / 2];
        let mut sin = vec![0.0f32; seq * head_dim / 2];
        rope_cache(&mut cos, &mut sin, seq, head_dim, 10000.0).unwrap();
        rope_apply(&mut x, &cos, &sin, 1, 1, seq, head_dim).unwrap();
        for t in 0..seq {
            for i in 0..head_dim / 2 {
                let a0 = orig[t * head_dim + 2 * i];
                let b0 = orig[t * head_dim + 2 * i + 1];
                let a1 = x[t * head_dim + 2 * i];
                let b1 = x[t * head_dim + 2 * i + 1];
                let n0 = (a0 * a0 + b0 * b0).sqrt();
                let n1 = (a1 * a1 + b1 * b1).sqrt();
                assert!(approx(n0, n1, 0.01), "t={t} i={i} n0={n0} n1={n1}");
            }
        }
    }

    #[test]
    fn test_scaled_dot_attention_shapes() {
        let (b, h, s, d) = (1, 1, 3, 4);
        let q = vec![1.0f32; b * h * s * d];
        let k = vec![1.0f32; b * h * s * d];
        let v = vec![1.0f32; b * h * s * d];
        let mut out = vec![0.0f32; b * h * s * d];
        scaled_dot_attention(&q, &k, &v, None, false, &mut out, b, h, s, d).unwrap();
        // With all-ones, softmax is uniform, so each output equals V's row of ones.
        for &o in &out { assert!(approx(o, 1.0, 0.001)); }
    }

    #[test]
    fn test_scaled_dot_attention_causal() {
        let (b, h, s, d) = (1, 1, 3, 2);
        let q = vec![0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0]; // trivial
        let k = q.clone();
        let v = vec![1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0]; // v[t] = (t+1, 0)
        let mut out = vec![0.0f32; b * h * s * d];
        scaled_dot_attention(&q, &k, &v, None, true, &mut out, b, h, s, d).unwrap();
        // With causal masking, token 0 attends only to itself → out[0] == v[0] == 1.0
        assert!(approx(out[0], 1.0, 0.01));
    }
}
