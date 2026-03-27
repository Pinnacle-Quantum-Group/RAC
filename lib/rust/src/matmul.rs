//! Matrix multiply — cache-tiled SGEMM with Rayon parallelism

use crate::{Config, RacError, Result, activation::Activation};

/// C = alpha * A @ B + beta * C
/// Row-major layout. Rayon-parallelized over M, cache-tiled over K and N.
pub fn sgemm(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
    alpha: f32, beta: f32,
    cfg: &Config,
) -> Result<()> {
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return Err(RacError::InvalidDimension);
    }

    let tile = cfg.tile_size;

    // Apply beta
    if beta == 0.0 {
        c.iter_mut().for_each(|x| *x = 0.0);
    } else if beta != 1.0 {
        c.iter_mut().for_each(|x| *x *= beta);
    }

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        // Parallel over row tiles
        c.par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, c_row)| {
                for j0 in (0..n).step_by(tile) {
                    let jmax = (j0 + tile).min(n);
                    for k0 in (0..k).step_by(tile) {
                        let kmax = (k0 + tile).min(k);
                        for j in j0..jmax {
                            let mut sum = 0.0f32;
                            for kk in k0..kmax {
                                sum = a[i * k + kk].mul_add(b[kk * n + j], sum);
                            }
                            c_row[j] = (alpha * sum).mul_add(1.0, c_row[j]);
                        }
                    }
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for i0 in (0..m).step_by(tile) {
            for j0 in (0..n).step_by(tile) {
                for k0 in (0..k).step_by(tile) {
                    let imax = (i0 + tile).min(m);
                    let jmax = (j0 + tile).min(n);
                    let kmax = (k0 + tile).min(k);

                    for i in i0..imax {
                        for j in j0..jmax {
                            let mut sum = 0.0f32;
                            for kk in k0..kmax {
                                sum = a[i * k + kk].mul_add(b[kk * n + j], sum);
                            }
                            c[i * n + j] = alpha.mul_add(sum, c[i * n + j]);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// C = A @ B  (convenience wrapper)
pub fn matmul(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
    cfg: &Config,
) -> Result<()> {
    sgemm(a, b, c, m, n, k, 1.0, 0.0, cfg)
}

/// Fused linear: output = act(input @ weight^T + bias)
/// weight is [n, k] row-major (out_features x in_features).
pub fn fused_linear(
    input: &[f32],    // [m, k]
    weight: &[f32],   // [n, k]
    bias: Option<&[f32]>,  // [n] or None
    output: &mut [f32],    // [m, n]
    m: usize, n: usize, k: usize,
    act: Activation,
    cfg: &Config,
) -> Result<()> {
    if input.len() < m * k || weight.len() < n * k || output.len() < m * n {
        return Err(RacError::InvalidDimension);
    }

    let apply_act = |x: f32| -> f32 {
        match act {
            Activation::None => x,
            Activation::ReLU => x.max(0.0),
            Activation::GELU => x * 0.5 * (1.0 + libm::erff(x * 0.7071067811865)),
            Activation::SiLU => x / (1.0 + (-x).exp()),
        }
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        output.par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, out_row)| {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    let tile = cfg.tile_size;
                    for k0 in (0..k).step_by(tile) {
                        let kmax = (k0 + tile).min(k);
                        for kk in k0..kmax {
                            sum = input[i * k + kk].mul_add(weight[j * k + kk], sum);
                        }
                    }
                    if let Some(b) = bias { sum += b[j]; }
                    out_row[j] = apply_act(sum);
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum = input[i * k + kk].mul_add(weight[j * k + kk], sum);
                }
                if let Some(b) = bias { sum += b[j]; }
                output[i * n + j] = apply_act(sum);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        let a = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
        let b = [3.0f32, 4.0, 5.0, 6.0]; // 2x2
        let mut c = [0.0f32; 4];

        matmul(&a, &b, &mut c, 2, 2, 2, &Config::default()).unwrap();

        assert!((c[0] - 3.0).abs() < 0.01);
        assert!((c[1] - 4.0).abs() < 0.01);
        assert!((c[2] - 5.0).abs() < 0.01);
        assert!((c[3] - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_matmul_3x4() {
        // A: 2x3, B: 3x2
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0f32; 4]; // 2x2

        matmul(&a, &b, &mut c, 2, 2, 3, &Config::default()).unwrap();

        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!((c[0] - 58.0).abs() < 0.01);
        assert!((c[1] - 64.0).abs() < 0.01);
        assert!((c[2] - 139.0).abs() < 0.01);
        assert!((c[3] - 154.0).abs() < 0.01);
    }

    #[test]
    fn test_sgemm_alpha_beta() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 0.0, 0.0, 1.0]; // identity
        let mut c = [10.0f32, 20.0, 30.0, 40.0];

        // c = 2.0 * A @ I + 0.5 * C_old
        sgemm(&a, &b, &mut c, 2, 2, 2, 2.0, 0.5, &Config::default()).unwrap();

        // c[0] = 2.0*1.0 + 0.5*10.0 = 7.0
        assert!((c[0] - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_fused_linear_relu() {
        let input = [1.0f32, -1.0]; // 1x2
        let weight = [1.0f32, 1.0, -1.0, 1.0]; // 2x2 (out=2, in=2)
        let bias = [0.0f32, -5.0];
        let mut output = [0.0f32; 2]; // 1x2

        fused_linear(&input, &weight, Some(&bias), &mut output,
                     1, 2, 2, Activation::ReLU, &Config::default()).unwrap();

        // out[0] = relu(1*1 + (-1)*1 + 0) = relu(0) = 0
        // out[1] = relu(1*(-1) + (-1)*1 + (-5)) = relu(-7) = 0
        assert!((output[0] - 0.0).abs() < 0.01);
        assert!((output[1] - 0.0).abs() < 0.01);
    }
}
