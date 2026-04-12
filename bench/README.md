## RAC transformer inference — three-way bench

Side-by-side single-layer transformer inference: **RAC** (this library)
vs **llama.cpp** vs **tinygrad**, same shape, same hardware, same
prompt/decode scenarios.

**Current status:** the RAC side runs end-to-end. The other two are
stubbed — fill in `configs/llama_cpp.yaml` and `configs/tinygrad.yaml`
with local paths/models, then re-run.

### Model shape (start small)

```
d_model   = 512
n_heads   = 8           (d_head = 64)
d_ff      = 1536        (~3× d_model, SwiGLU gated)
layers    = 1           (single decoder block)
```

TinyLlama-shaped at 1 / 22 the depth. Small enough to load fast, large
enough that the GEMM cost dominates and differences between frameworks
surface cleanly.

### Scenarios

| scenario | T | bottleneck | measures |
|---|---|---|---|
| **prefill** | 128 | compute-bound (big GEMM) | peak GFLOPS |
| **decode** | 1   | memory-bound (T=1 GEMV) | token latency |

### Running

**RAC:**
```bash
cd build/
cmake --build . --target bench_rac_transformer
./bench_rac_transformer               # default: 30 prefill iters, 100 decode iters
./bench_rac_transformer 100 200       # custom iteration counts
```

**llama.cpp:** fill in `configs/llama_cpp.yaml`, then:
```bash
./bench/run_llama_cpp.sh              # TODO: will be added once config is populated
```

**tinygrad:** fill in `configs/tinygrad.yaml`, then:
```bash
python bench/run_tinygrad.py          # TODO: will be added once config is populated
```

### Fair-comparison notes

- **Precision:** RAC is FP32. llama.cpp defaults to Q4_K_M; tinygrad to FP16. The harness reports each framework at each precision it supports so you can pick a like-for-like comparison.
- **Threading:** pin each framework to the same physical core count (16 on your 5950X). RAC uses OpenMP; llama.cpp uses `-t`; tinygrad uses `OMP_NUM_THREADS` + `TINYGRAD_CLANG_PARALLEL=1`.
- **Warmup:** each framework gets one untimed warmup pass before the measurement loop.
- **Layer scaling:** for framework comparisons at whole-model scale, divide by `n_layers_divisor` (22 for TinyLlama-1.1B) to get per-layer numbers.

### What to expect (rough guidance)

On a Ryzen 9 5950X running FP32:

| framework | prefill (tokens/s/layer) | decode (ms/layer) |
|---|---|---|
| RAC (CORDIC) | ~400–600 | ~0.3–0.8 |
| llama.cpp (F32, cBLAS) | ~1500–3000 | ~0.1–0.3 |
| tinygrad (CLANG/F32) | ~200–800 | ~0.5–2.0 |

RAC's competitive advantage isn't FP32-on-x86-silicon (where tensor-oriented kernels beat CORDIC). The point of this bench is to establish **where RAC sits today on commodity silicon**, so that the FPGA/ASIC numbers have a ground-truth baseline to improve on.
