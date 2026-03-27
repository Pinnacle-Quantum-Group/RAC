//! Batch activation functions with Rayon parallelism

/// Activation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    None = 0,
    ReLU = 1,
    GELU = 2,
    SiLU = 3,
}

pub fn relu(x: &[f32], out: &mut [f32]) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        x.par_iter().zip(out.par_iter_mut())
            .for_each(|(&xi, o)| *o = xi.max(0.0));
        return;
    }

    #[cfg(not(feature = "rayon"))]
    for (i, &xi) in x.iter().enumerate() {
        out[i] = xi.max(0.0);
    }
}

pub fn gelu(x: &[f32], out: &mut [f32]) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        x.par_iter().zip(out.par_iter_mut())
            .for_each(|(&xi, o)| {
                *o = xi * 0.5 * (1.0 + libm::erff(xi * 0.7071067811865));
            });
        return;
    }

    #[cfg(not(feature = "rayon"))]
    for (i, &xi) in x.iter().enumerate() {
        out[i] = xi * 0.5 * (1.0 + libm::erff(xi * 0.7071067811865));
    }
}

pub fn silu(x: &[f32], out: &mut [f32]) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        x.par_iter().zip(out.par_iter_mut())
            .for_each(|(&xi, o)| *o = xi / (1.0 + (-xi).exp()));
        return;
    }

    #[cfg(not(feature = "rayon"))]
    for (i, &xi) in x.iter().enumerate() {
        out[i] = xi / (1.0 + (-xi).exp());
    }
}

pub fn softmax_batch(x: &[f32], out: &mut [f32], batch: usize, n: usize) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        x.par_chunks(n).zip(out.par_chunks_mut(n))
            .for_each(|(xi, oi)| {
                crate::cordic::softmax(xi, oi);
            });
        return;
    }

    #[cfg(not(feature = "rayon"))]
    for b in 0..batch {
        crate::cordic::softmax(&x[b*n..(b+1)*n], &mut out[b*n..(b+1)*n]);
    }
}
