//! # RAC BLAS — Single-precision, row-major BLAS surface
//!
//! Mirrors the C API in `lib/c/rac_blas.h`.
//!
//! - **Level 1** (vector-vector): [`saxpy`], [`sdot`], [`snrm2`], [`sasum`],
//!   [`isamax`], [`sscal`], [`scopy`], [`sswap`], [`srot`], [`srotg`].
//! - **Level 2** (matrix-vector): [`sgemv`], [`sger`], [`ssymv`], [`ssyr`],
//!   [`ssyr2`], [`strmv`], [`strsv`].
//! - **Level 3** (matrix-matrix): [`sgemm_ex`], [`ssymm`], [`ssyrk`],
//!   [`ssyr2k`], [`strmm`], [`strsm`].
//!
//! ## Conventions
//! - Row-major storage; `lda`, `ldb`, `ldc` are row strides in `f32` elements.
//! - `incx`, `incy` are positive `usize` strides. Negative strides not supported.
//! - L2/L3 functions return `Result<(), crate::RacError>`.
//! - L1 functions return native types (no error path) to match BLAS reference.
//! - In-place ops modify the output buffer (named `y`, `a`, `c`, `b`, `x`).

pub mod l1;
pub mod l2;
pub mod l3;

/// Transpose flag. Values match CBLAS `CblasNoTrans` / `CblasTrans`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    NoTrans = 111,
    Trans = 112,
}

/// Triangle selector. Values match CBLAS `CblasUpper` / `CblasLower`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpLo {
    Upper = 121,
    Lower = 122,
}

/// Diagonal flag. Values match CBLAS `CblasNonUnit` / `CblasUnit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diag {
    NonUnit = 131,
    Unit = 132,
}

/// Side flag for symmetric/triangular L3 ops. Values match CBLAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left = 141,
    Right = 142,
}

// Re-export the public surface at the `blas::` level.
pub use l1::{saxpy, sdot, snrm2, sasum, isamax, sscal, scopy, sswap, srot, srotg};
pub use l2::{sgemv, sger, ssymv, ssyr, ssyr2, strmv, strsv};
pub use l3::{sgemm_ex, ssymm, ssyrk, ssyr2k, strmm, strsm};
