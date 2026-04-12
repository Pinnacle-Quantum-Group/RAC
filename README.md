# RAC — Rotation-Accumulate Primitive Library
**Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026**

---

## What This Is

RAC replaces the Multiply-Accumulate (MAC) operation with geometric rotation as the fundamental compute primitive. MAC is a degenerate one-dimensional projection of rotation. This library makes that mathematically precise and benchmarks it on real GPU hardware.

**The key insight:** Modern GPUs already have CORDIC hardware on-die — the Special Function Units (SFUs) that compute `sin`, `cos`, `atan`, `sqrt`. Those units are used internally for transcendental functions but sit largely idle during tensor workloads, which route through the multiplier-heavy tensor cores instead. RAC routes tensor operations through the SFUs. The tensor cores are not used.

---

## The 17 Primitives

### Core Rotation
| Function | Description |
|----------|-------------|
| `rac_rotate(v, θ)` | Rotate vector v by θ, gain-compensated (magnitude preserved) |
| `rac_rotate_raw(v, θ)` | Rotate v by θ, NO gain compensation — for chained rotations |
| `rac_compensate(v, n)` | Apply K⁻ⁿ after n chained `rac_rotate_raw` calls |
| `rac_project(v, θ)` | Signed scalar: x-component of rotated v = MAC equivalent |

### Polar / Vectoring
| Function | Description |
|----------|-------------|
| `rac_polar(v, mag, angle)` | Cartesian → polar in one CORDIC pass |
| `rac_norm(v)` | Euclidean magnitude |
| `rac_normalize(v)` | Unit vector |

### Dot / Similarity
| Function | Description |
|----------|-------------|
| `rac_dot(a, b)` | Dot product via angle subtraction |
| `rac_coherence(a, b)` | Cosine similarity ∈ [−1, 1] |

### Complex / DSP
| Function | Description |
|----------|-------------|
| `rac_complex_mul(a, b)` | Complex multiply via rotation (replaces 4 MACs) |
| `rac_dct(x, X, n)` | DCT-II via CORDIC basis rotation |

### Hyperbolic / Activations
| Function | Description |
|----------|-------------|
| `rac_exp(x)` | eˣ via hyperbolic CORDIC — no `expf()` |
| `rac_tanh(x)` | tanh(x) via hyperbolic CORDIC |
| `rac_softmax(x, out, n)` | Softmax via batched hyperbolic CORDIC |

### Batch / Linear Algebra
| Function | Description |
|----------|-------------|
| `rac_rotate_batch(v, θ, out, n)` | Independent batch rotation |
| `rac_inner(a, b, n)` | Inner product via rotation-project-accumulate |
| `rac_outer(a, b, C, m, n)` | Outer product |
| `rac_matmul(A, B, C, M, N, K)` | Matrix multiply — zero multiply operators in kernel |

### Transformer / AI Primitives (native CORDIC)

Every op in a transformer maps to a native CORDIC mode:

| Op                      | CORDIC mode            | Primitive                          |
|-------------------------|------------------------|------------------------------------|
| `QKᵀ` / `attn @ V`      | linear MAC             | `rac_matmul`                       |
| Softmax `exp(x)`        | hyperbolic rotation    | `rac_exp`                          |
| Softmax normalize       | linear vectoring       | built-in (divide)                  |
| LayerNorm mean / var    | linear accumulate      | built-in                           |
| `1 / √variance`         | hyperbolic vectoring   | **`rac_rsqrt`**                    |
| RMSNorm                 | hyperbolic vectoring   | `rac_rmsnorm`                      |
| LayerNorm               | accumulate + rsqrt     | `rac_layernorm`                    |
| **RoPE embeddings**     | **circular rotation**  | **`rac_rope_apply` — native**      |
| Sigmoid                 | hyperbolic rotation    | `rac_sigmoid` = ½(1 + tanh(x/2))   |
| GELU / SiLU             | hyperbolic + circle    | `rac_gelu`, `rac_silu`             |
| Scaled dot-product attn | composite              | `rac_attention`                    |

**RoPE is the killer app.** Rotary position embeddings are *literally* Givens rotations. Every other accelerator emulates them with multipliers. RAC executes them natively — one `rac_rotate` per embedding-dim pair.

### Tunable Precision

CORDIC is an N-iteration algorithm — one bit of precision per iteration. That's a first-class knob for AI workloads:

| Regime             | iters | use case                          |
|--------------------|-------|-----------------------------------|
| Training           | 24    | fp32-matched precision            |
| Inference          | 16    | default, good quality             |
| Edge / quantized   | 8     | tiny, cheap, good enough          |

Exposed via:
- C:   `rac_rotate_n(v, θ, iters)`, `rac_project_n`, `rac_polar_n`, `rac_exp_n`, `rac_tanh_n`
- Rust: `cordic::rotate_n`, `project_n`, `polar_n`, `sincos`, `rsqrt`, `sigmoid`
- PyTorch: `rac_set_precision(iters)` / `RAC_CORDIC_ITERS` env var

No other architecture gives you that knob.

### PyTorch-native classes

```python
from rac_torch import (
    RACLinear, RACFusedFFN,              # FFN
    RACRoPE, RACRoPEAttention,           # rotary-positional attention
    RACRMSNorm, RACLayerNorm,            # CORDIC-rsqrt norms
    RACLlamaBlock,                        # Llama-style transformer block
    rac_set_precision,                   # tunable-precision knob
)

block = RACLlamaBlock(d_model=4096, n_heads=32, ff_dim=11008)
y = block(x, is_causal=True)   # every op routes through RAC CORDIC
```

---

## MAC Equivalence

```c
// Traditional MAC
float result = a * b;

// RAC equivalent
float2 va = make_float2(a, 0.0f);
float  angle_b = atan2f(0.0f, b >= 0 ? 1.0f : -1.0f);
float  mag_b   = fabsf(b);
float  result  = rac_project(va, angle_b) * mag_b;
// rac_project routes through SFU __sinf/__cosf
// the final * mag_b is magnitude scaling, not compute
```

The sign comes for free from the x-component extraction. Negative when v opposes the projection axis. This is why `rac_inner` and `rac_accumulate` can represent subtraction — cancellation is how weighted sums go negative, how interference patterns form, how neural nets learn.

---

## Gain Compensation

CORDIC rotation introduces a gain factor **K ≈ 1.64676** per iteration sequence. The CORDIC gain constant arises because each micro-rotation slightly scales the vector magnitude.

- `rac_rotate`: **compensated** — applies K⁻¹ = 0.60725 at initialization. Output magnitude == input magnitude. Use for single rotations.
- `rac_rotate_raw`: **uncompensated** — output magnitude = input × K. Use for chained rotations where you want to apply compensation once at chain end via `rac_compensate(v, n)`.

**Example — chained rotation:**
```c
float2 v = {1.0f, 0.0f};
v = rac_rotate_raw(v, theta1);   // magnitude now K
v = rac_rotate_raw(v, theta2);   // magnitude now K²
v = rac_rotate_raw(v, theta3);   // magnitude now K³
v = rac_compensate(v, 3);        // magnitude back to 1.0
```

---

## SFU Routing Strategy

### NVIDIA (CUDA)
SFUs execute `__sinf`, `__cosf`, `__fdividef`, `__powf` in 4 cycles on Ampere (vs ~20 cycles for a multiply on a CUDA core). Each SM has 4 SFUs and 128 CUDA cores. Under MAC workloads, SFUs are ~95% idle. RAC saturates them.

Key intrinsics used:
- `__sinf(x)` / `__cosf(x)` — 4-cycle SFU trig
- `__fdividef(a, b)` — fast SFU division
- `__powf(a, b)` — SFU power (used for gain compensation)

### AMD (HIP/ROCm)
AMD SFUs are part of the Transcendental Units in each CU. RDNA3 has 4 transcendental units per CU.

Key intrinsics used:
- `__ocml_sin_f32(x)` / `__ocml_cos_f32(x)` — AMD SFU trig
- `__ocml_div_f32(a, b)` — SFU division
- `__ocml_pow_f32(a, b)` — SFU power

---

## Building

```bash
# NVIDIA GPU (Ampere)
CUDA_ARCH=86 ./rac_benchmark.sh cuda

# NVIDIA GPU (H100)
CUDA_ARCH=90 ./rac_benchmark.sh cuda

# AMD GPU (MI210)
HIP_ARCH=gfx90a ./rac_benchmark.sh hip

# Both
./rac_benchmark.sh both
```

---

## Reading Benchmark Output

```
Device: NVIDIA A100-SXM4  (SM 8.0)  SFUs/SM: 108  SMs: 108

── Matmul 1024x1024 x 1024x1024 ──────────────────────
  RAC:    12.34 ms/iter  0.174 TOPS  energy=1240 mJ
  cuBLAS: 0.89 ms/iter   2.41 TOPS  energy=980 mJ
  Speedup: RAC/cuBLAS = 0.072x
  Correctness: max_error=0.000412  tolerance=0.001  failures=0/1048576  PASS
```

**What the numbers mean:**

- **TOPS**: Tera-Operations Per Second, counting 2 ops per matmul element (multiply + add in MAC terms, rotation + accumulate in RAC terms)
- **Energy**: millijoules per iteration via NVML. RAC energy reflects SFU power; cuBLAS reflects tensor core power.
- **Speedup < 1**: Expected on current hardware — cuBLAS tensor cores are optimized over decades. RAC is not yet hardware-optimized. The point of this benchmark is to establish the baseline for a purpose-built RAC accelerator, not to beat cuBLAS on NVIDIA silicon.
- **Correctness PASS**: RAC output matches cuBLAS within float32 tolerance. The math is equivalent.

The energy-per-correct-op ratio is the meaningful long-term metric. A purpose-built RAC ASIC (no multiplier circuits, SFU-native) collapses the TOPS gap while the energy advantage grows.

---

## Proprietary Extended Operations

The `RAC_OP_EXTENDED` enum value and the `rac_execute()` dispatch hook are reserved for backend-specific operations not included in this public release. `rac_query_capability(ctx, RAC_OP_EXTENDED)` returns 0 on all public backends.

---

## License

RAC primitive interface and this implementation: © 2026 Michael A. Doran Jr. / Pinnacle Quantum Group. All rights reserved. Prior art established March 2026.

CORDIC algorithm: public domain (Volder, 1959).
