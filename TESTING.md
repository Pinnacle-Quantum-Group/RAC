# RAC Testing Pyramid — BVT / DVT / E2E / Perf / Coverage

Every primitive shipped by RAC — CORDIC core, transformer primitives, matmul,
activations, normalization — is validated at four levels, mirrored across C,
Rust, and PyTorch.

```
                  ┌────────────────┐
                  │ perf harness   │  throughput / latency / precision-vs-cost
                  ├────────────────┤
                  │ E2E            │  multi-op flows, transformer blocks
                  ├────────────────┤
                  │ DVT            │  sweep correctness, gradient flow, edge cases
                  ├────────────────┤
                  │ BVT            │  smoke / API surface / basic math
                  └────────────────┘
```

## C library — `lib/c`

| Layer | File                          | Checks | Runtime |
|-------|-------------------------------|--------|---------|
| BVT   | `test_rac_lib_bvt.c`          | 70+    | < 2 s   |
| DVT   | `test_rac_lib_dvt.c`          | 85+    | ~ 30 s  |
| E2E   | `test_rac_lib_e2e.c`          | 15+    | ~ 60 s  |
| Cov   | `lib/Testing/cov_transformer.c` | drives gcov to 100 % |

Build + run:
```bash
cd lib && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRAC_ENGINE=OFF
make -j$(nproc)
./test_rac_lib_bvt && ./test_rac_lib_dvt && ./test_rac_lib_e2e
```

New in v0.3 (transformer primitives):

- BVT-12 tunable-precision CORDIC (iters 4..24)
- BVT-13 LayerNorm / RMSNorm
- BVT-14 RoPE (cache + apply)
- BVT-15 Scaled dot-product attention
- DVT-10 iter-count sweep with ~2^-iters convergence check
- DVT-11 rsqrt / sigmoid / sincos full-sweep parity vs libm
- DVT-12 LayerNorm / RMSNorm sweeps + gamma / beta scaling + error paths
- DVT-13 RoPE rotation properties (pair magnitude preservation, t=0 identity, odd-dim rejection)
- DVT-14 Attention correctness (causal, uniform, error paths)
- E2E-7 Transformer-stack integration (QKV → RoPE → attention → RMSNorm → FFN)
- E2E-8 Transformer-op throughput benchmark (GB/s)
- E2E-9 CORDIC iter-count vs worst-case error table

## Rust crate — `lib/rust`

28 inline `#[test]` functions in `src/cordic.rs`, `src/matmul.rs`, and
`src/transformer.rs`. Benches via Criterion in `benches/matmul.rs`.

```bash
cd lib/rust
cargo test --lib            # unit + integration tests
cargo bench --no-run        # compile benches
cargo bench -- matmul       # run a specific bench group
```

## PyTorch extension — `rac_torch`

| Layer | File                  | Count |
|-------|-----------------------|-------|
| BVT   | `test_rac_bvt.py`     | ~60   |
| DVT   | `test_rac_dvt.py`     | ~130  |
| E2E   | `test_rac_e2e.py`     | ~30   |
| Perf  | `test_rac_perf.py`    | throughput harness |

New in v0.3:
- BVT-15 tunable-precision knob (`rac_set_precision`, clamping, roundtrip)
- BVT-16 RACRoPE / RACRMSNorm / RACLayerNorm / RACRoPEAttention / RACLlamaBlock forwards
- DVT-27 RoPE rotation invariance + explicit positions + grad flow
- DVT-28 RMSNorm vs reference implementation
- DVT-29 RACLayerNorm vs `torch.nn.LayerNorm` (weight-copied) parity
- DVT-30 RACRoPEAttention forward+backward + causal prefix invariance
- DVT-31 RACLlamaBlock 60-step training loss decrease
- DVT-32 Precision-knob roundtrip
- E2E-18 Llama-block training convergence (loss decrease > 15 %)
- E2E-19 Transformer throughput bench (RoPE, RMSNorm, LayerNorm)
- E2E-20 Precision-knob delta measurement

Run:
```bash
cd rac_torch
python3 test_rac_bvt.py
python3 test_rac_dvt.py
python3 test_rac_e2e.py
python3 test_rac_perf.py --device cuda
```

## Coverage

### C

Target: `lib/c/rac_cpu.c` — 100.0 % line coverage verified.

```bash
lib/coverage.sh c       # gcov + (optional) lcov HTML → build-cov/c/
```

The `lib/Testing/cov_transformer.c` driver exercises every public symbol
in `rac_cpu.c`, including every branch of the error-handling paths,
every activation in `rac_fused_linear`, every CORDIC iteration clamp,
the pre-rotation branches in `rac_polar` / `rac_polar_n`, and the NULL-cfg
fallback in `_get_tile`.

### Rust

Target: 28 / 28 unit tests passing. With `cargo-llvm-cov` installed
(`cargo install cargo-llvm-cov`) the script emits HTML + lcov.info:

```bash
lib/coverage.sh rust    # llvm-cov → build-cov/rust/
```

### Python

With `coverage` installed (`pip install coverage`) the script runs BVT,
DVT, E2E under `coverage run` and emits an HTML report:

```bash
lib/coverage.sh python  # coverage.py → build-cov/python/
```

### All languages at once

```bash
lib/coverage.sh         # runs c + rust + python
```

## Test hygiene

- Tests are deterministic (seeded RNGs, PyTorch `manual_seed`).
- Numerical tolerances scale with CORDIC iteration count: `tol ≈ 2^-iters`
  with a float32 floor of ~1e-5.
- New primitives are validated against both analytic identity checks
  (magnitude preservation for RoPE, sum-to-1 for softmax, zero-mean /
  unit-variance for LayerNorm) and against libm / `torch.nn` reference
  implementations where applicable.
