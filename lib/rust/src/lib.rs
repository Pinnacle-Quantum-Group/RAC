//! # RAC — Rotation-Accumulate Compute Primitives (Rust)
//!
//! Multiply-free CORDIC-based linear algebra library.
//!
//! All 17 RAC primitives plus production extensions:
//! - SGEMM with cache tiling and Rayon parallelism
//! - Fused linear layer (matmul + bias + activation)
//! - Batch activations (ReLU, GELU, SiLU, softmax)
//!
//! # Example
//! ```
//! use rac::{matmul, Config};
//!
//! let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
//! let b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2x2
//! let mut c = vec![0.0f32; 4];           // 2x2
//!
//! matmul(&a, &b, &mut c, 2, 2, 2, &Config::default());
//! assert!((c[0] - 19.0).abs() < 0.01);  // 1*5 + 2*7
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod cordic;
pub mod matmul;
pub mod activation;
#[cfg(feature = "ffi")]
pub mod ffi;

use core::f32::consts::PI;

// Re-exports
pub use cordic::{Vec2, rotate, rotate_raw, project, polar, norm, normalize, dot, coherence};
pub use cordic::{complex_mul, exp, tanh, dct, softmax};
pub use matmul::{sgemm, matmul, fused_linear};
pub use activation::{Activation, relu, gelu, silu, softmax_batch};

/// CORDIC constants
pub const K_INV: f32 = 0.60725;
pub const K: f32 = 1.64676;
pub const ITERS: usize = 16;
pub const ITERS_FAST: usize = 12;
pub const K_HYP_INV: f32 = 0.82816;
pub const RAC_PI: f32 = PI;

/// Configuration for RAC operations
#[derive(Debug, Clone)]
pub struct Config {
    pub num_threads: usize,
    pub tile_size: usize,
    pub cordic_iters: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            num_threads: 0, // auto
            tile_size: 64,
            cordic_iters: ITERS,
        }
    }
}

/// Error type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RacError {
    NullPointer,
    InvalidDimension,
    AllocationFailed,
}

impl core::fmt::Display for RacError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RacError::NullPointer => write!(f, "null pointer"),
            RacError::InvalidDimension => write!(f, "invalid dimension"),
            RacError::AllocationFailed => write!(f, "allocation failed"),
        }
    }
}

pub type Result<T> = core::result::Result<T, RacError>;
