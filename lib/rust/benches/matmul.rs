//! Criterion benches for RAC primitives — matmul, transformer ops, and
//! the tunable-precision CORDIC sweep.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rac::{cordic, matmul, transformer, Config, Vec2};

fn bench_matmul(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul");
    for &sz in &[64usize, 256, 512] {
        let a = vec![0.5f32; sz * sz];
        let b = vec![0.25f32; sz * sz];
        let mut out = vec![0.0f32; sz * sz];
        let cfg = Config::default();
        g.bench_with_input(BenchmarkId::from_parameter(sz), &sz, |bn, &n| {
            bn.iter(|| {
                matmul(&a, &b, &mut out, n, n, n, &cfg).unwrap();
                black_box(&out);
            });
        });
    }
    g.finish();
}

fn bench_rotate_precision(c: &mut Criterion) {
    let mut g = c.benchmark_group("rotate_precision_sweep");
    for &iters in &[4usize, 8, 12, 16, 20, 24] {
        g.bench_with_input(BenchmarkId::from_parameter(iters), &iters, |bn, &n| {
            let v = Vec2::new(1.0, 0.0);
            bn.iter(|| {
                let r = cordic::rotate_n(black_box(v), black_box(0.5), n);
                black_box(r);
            });
        });
    }
    g.finish();
}

fn bench_rmsnorm(c: &mut Criterion) {
    let mut g = c.benchmark_group("rmsnorm");
    let cfg = Config::default();
    for &sz in &[128usize, 1024, 4096] {
        let x = vec![0.5f32; sz];
        let mut y = vec![0.0f32; sz];
        g.bench_with_input(BenchmarkId::from_parameter(sz), &sz, |bn, &d| {
            bn.iter(|| {
                transformer::rmsnorm(&x, &mut y, None, 1e-6, 1, d, &cfg).unwrap();
                black_box(&y);
            });
        });
    }
    g.finish();
}

fn bench_rope(c: &mut Criterion) {
    let mut g = c.benchmark_group("rope_apply");
    for &(t, h, d) in &[(64usize, 8usize, 64usize), (512, 16, 64), (1024, 16, 128)] {
        let total = t * h * d;
        let mut x = vec![0.5f32; total];
        let half = d / 2;
        let mut cos = vec![0.0f32; t * half];
        let mut sin = vec![0.0f32; t * half];
        transformer::rope_cache(&mut cos, &mut sin, t, d, 10000.0).unwrap();
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("T{}_H{}_D{}", t, h, d)),
            &(t, h, d),
            |bn, &(tt, hh, dd)| {
                bn.iter(|| {
                    transformer::rope_apply(&mut x, &cos, &sin, 1, hh, tt, dd).unwrap();
                    black_box(&x);
                });
            },
        );
    }
    g.finish();
}

criterion_group!(benches, bench_matmul, bench_rotate_precision, bench_rmsnorm, bench_rope);
criterion_main!(benches);
