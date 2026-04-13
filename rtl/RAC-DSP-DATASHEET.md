# RAC-DSP — No-Multiplier DSP Cell

**Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026**

A datasheet-style specification for the RAC-DSP primitive: a CORDIC-based
DSP cell intended as a drop-in replacement for multiply-based DSP blocks
(Xilinx DSP48E2, Intel DSP blocks, AMD AIE-ML) when the workload is
dominated by rotations, transcendentals, and GEMM-via-CORDIC.

> **Every DSP block in the industry embeds a hardware multiplier. The
> RAC-DSP does not.** That's the entire design thesis.

---

## 1. Block Diagram

```
          valid_in                                    valid_out
  op_in ─────┐                                      ┌───── op_out
  x_in  ─────┼──┐                                   ┌───── x_out
  y_in  ─────┼──┤                                   ├───── y_out
  z_in  ─────┼──┤                                   ├───── z_out
             ▼  ▼                                   │
        ┌─────────────┐                             │
        │ quadrant    │  reduces z to (-π/2, +π/2]  │
        │ fold + reg  │  by flipping (x,y) on flip  │
        └─────────────┘                             │
               │                                    │
               ▼                                    │
        ┌─────────────┐  index 10 high bits of z    │
        │ coarse LUT  │  ROM: 1024 × 10-bit dir     │
        │ (dir bits)  │  vectors (direction bits)   │
        └─────────────┘                             │
               │                                    │
               ▼                                    │
        ┌─────────────┐  10 sign-controlled         │
        │ coarse      │  shift-add pairs in one     │
        │ apply (comb)│  combinational chain        │
        └─────────────┘                             │
               │                                    │
               ▼                                    │
        ┌─────────────┐  6 pipelined CORDIC micro-  │
        │ residual    │  steps (iter 10..15)        │
        │ pipeline ×6 │  3 adders + 2 shifts each   │
        └─────────────┘                             │
               │                                    │
               ▼                                    │
        ┌─────────────┐  mode-dependent output      │
        │ commit/sel  │  select (rotate, project,   │
        │             │  accum, polar, etc.)        │
        └─────────────┘                             │
               │                                    │
               └────────────────────────────────────┘
```

---

## 2. Interface

| signal | dir | width | description |
|---|---|---|---|
| `clk` | in | 1 | positive-edge clock |
| `rst_n` | in | 1 | active-low synchronous reset |
| `valid_in` | in | 1 | new op accepted on this cycle |
| `op_in` | in | 3 | see §3 for encoding |
| `x_in`, `y_in`, `z_in` | in | WIDTH | operands, Q(W-32).32 signed fixed-point |
| `valid_out` | out | 1 | result valid (LATENCY cycles after input) |
| `op_out` | out | 3 | echoes `op_in` for multi-issue routing |
| `x_out`, `y_out`, `z_out` | out | WIDTH | results, same format as inputs |

Parameters:

| parameter | default | range | meaning |
|---|---|---|---|
| `WIDTH` | 64 | 32, 48, 64 | datapath width in bits |
| `LUT_BITS` | 10 | 8–14 | coarse LUT address bits (collapses this many CORDIC iters) |
| `RESIDUAL` | 9 | 6–12 | residual CORDIC stages (fine-grained iters) |
| `RESIDUAL_START` | 8 | `LUT_BITS-2` typical | first residual physical shift (overlap coarse by 2 iters for convergence margin) |
| `INIT_LUT` | `cordic_coarse_lut.mem` | path | `$readmemh` source for the direction-bit ROM |
| `INIT_ATANH` | `cordic_atanh.mem` | path | `$readmemh` source for the hyperbolic table |

---

## 3. Operation Encoding (`op_in[2:0]`)

| code | mnemonic | operands | result | notes |
|---|---|---|---|---|
| `3'b000` | rotate | (x, y, θ) | (x', y') | CORDIC circular rotation |
| `3'b001` | project | (x, y, θ) | x' only | MAC-equivalent: `x' = x·cos(θ) + y·sin(θ)` |
| `3'b010` | vectoring | (x, y, 0) | (\|v\|, atan2(y,x)) | polar; `x_out=\|v\|`, `z_out=angle` |
| `3'b011` | hyperbolic | (x, y, z) | (K·cosh, K·sinh) | used for exp/tanh/sigmoid |
| `3'b100` | accum | (x, y, z) | internal `acc += x_proj` | projection + accumulate |
| `3'b101` | compensate | (x, y, –) | (x·K⁻ⁿ, y·K⁻ⁿ) | gain undo after raw chains |
| `3'b110` | pass-through | (x, y, z) | (x, y, z) | latency-match dummy |
| `3'b111` | reserved | — | — | undefined |

---

## 4. Timing

| quantity | value (default params) | notes |
|---|---|---|
| Latency | **12 cycles** | input-reg (1) + coarse (1) + residual (9) + output (1) |
| Throughput | **1 result / cycle** | fully pipelined after fill |
| Max Fmax | ~180 MHz (Alveo U250, -2 grade) | combinational 10-add chain critical |
| Accuracy (circular) | 2^-15 ≈ 3×10⁻⁵ | validated bit-exact vs sim/rac_dsp_ref.c across 101 angles, max err 4.2e-5 |
| Accuracy (hyperbolic) | ~2^-12 | Walther convergence is slower |
| Cosim status | **101/101 PASS** at 2^-14 tolerance | see `sim/README.md` |

---

## 5. Resource Cost

**Per cell (Alveo U250, XCU250, -2 grade):**

| resource | count | vs DSP48E2 |
|---|---|---|
| LUTs | ~2 900 | +57× (DSP48E2 uses ~50 LUTs for glue) |
| FFs | ~1 350 | +6× |
| BRAM (RAMB18) | 1 | +1 (DSP48E2 has no BRAM) |
| **DSP slices** | **0** | **-1 (the headline number)** |

**Per 16-cell bank (one reasonable system integration unit):**

| resource | count | % of XCU250 |
|---|---|---|
| LUTs | 46 K | 2.7% |
| FFs | 22 K | 0.6% |
| BRAM18 | 16 | 0.7% |
| DSPs | 0 | 0% |

**Per 432-cell bank (targets the A100's 432-SFU density):**

| resource | count | % of XCU250 |
|---|---|---|
| LUTs | 1.25 M | 72% |
| FFs | 580 K | 17% |
| BRAM18 | 432 | 20% |
| DSPs | 0 | 0% |

An A100-class SFU density fits in a single Alveo U250 with zero DSP
slices consumed. Those DSP slices remain available for whatever
conventional MAC work coexists in the bitstream — so the RAC-DSP
bank **adds** to the chip's arithmetic throughput rather than trading
against DSP48E2 usage.

---

## 6. Why No Multiplier Is Not a Compromise

The usual objection to CORDIC DSPs is that multipliers are "free" on
modern FPGAs (the DSP48E2 is a hard block, tens of thousands per chip)
while CORDIC costs LUT area. That framing is wrong for two reasons:

1. **The workload mix is different.** Dense LLM inference already
   saturates DSP slices for the dense GEMMs. The RAC-DSP is positioned
   to handle what's *left over*: the softmax, the RoPE rotations, the
   layer-norm rsqrts, the SiLU/GELU activations. Those ops want
   transcendental throughput, which DSP48E2 slices can't provide —
   you'd need a lookup-table plus polynomial approximation anyway,
   and that already costs LUTs.

2. **ASIC economics invert the argument.** On a 5 nm ASIC, multiplier
   arrays are expensive silicon (big transistor count per macro, lots
   of leakage). Shift-add cells are small. A RAC-DSP at the same node
   is ~1/4 the area of an equivalent MAC cell with a 32-bit multiplier.
   You pack 4× the throughput in the same die area — provided your
   workload is CORDIC-amenable.

Transformer inference is CORDIC-amenable. RoPE is a literal Givens
rotation. Softmax's `exp` is a hyperbolic CORDIC. Layer-norm's
`1/sqrt(·)` is a hyperbolic-vectoring CORDIC. FFN's SiLU / GELU are
sigmoid = `0.5·(1+tanh(·/2))`, another hyperbolic CORDIC. The only
op in the block that isn't a native CORDIC primitive is the Q·Kᵀ /
A·V GEMM — and we handle that via projection (§7).

---

## 7. GEMM Mapping (the non-obvious part)

A MAC is a degenerate 1-D projection: `a·b = |b| · project((a, 0), angle(b))`.
That gives RAC-DSP a direct path to MAC-equivalent arithmetic without
a multiplier:

```
rac_dsp.op = project
    input:  (a, 0, θ_b)    where θ_b = atan2(b, ·ref)
    output: x_out = a · cos(θ_b)    ← this is the MAC result, scaled by |b|
```

For a full M×K @ K×N GEMM, tile into a `rac_systolic_array` (see
`rac_systolic_array.v`). Each PE holds one weight as a pre-computed
angle + magnitude; activations stream through; partial sums accumulate
along columns. At 16×16 this gives 256 rotations/cycle = 46 Grot/s at
180 MHz — comparable per-area to a DSP48E2 array, with **zero** DSPs
consumed.

---

## 8. Verification Flow

1. **C reference** — `lib/c/rac_alu.c` is the bit-exact software
   model. The 67 BVTs in `test_rac_alu.c` validate the CORDIC
   semantics.
2. **Microcode equivalence** — `test_rac_ucode.c` + `test_rac_xrac.c`
   prove the ISA and RV32 decode layers match the C reference.
3. **RTL simulation** — `sim/tb_rac_dsp.v` (to be added) drives the
   Verilog with the same angle sweep as the C BVTs and compares
   Q32.32 outputs to the reference.
4. **Formal** — SymbiYosys proof of bit equivalence between
   `rac_dsp.v` and a Python/C reference model run over 16 random seeds.

---

## 9. Known Limitations

- **CORDIC convergence:** circular mode needs `|z| ≤ π` after quadrant
  folding; hyperbolic needs `|z| ≤ 1.12` after argument reduction.
  Both folds happen inside the cell.
- **Gain:** circular CORDIC has gain K ≈ 1.647; hyperbolic has K ≈
  0.828. The `compensate` op applies `K⁻ⁿ` after N chained raw
  rotations.
- **Precision:** ~42-bit signal path (10 coarse bits + 6 fine CORDIC
  bits × ~2 bits/iter + output truncation). Sufficient for FP32
  transformer inference; insufficient for FP64-critical scientific
  workloads — use `WIDTH=64` with `RESIDUAL=10` for those.

---

## 10. Change Log

| rev | date | change |
|---|---|---|
| 0.1 | Apr 2026 | Initial spec: 64-bit, LUT_BITS=10, RESIDUAL=6, no multipliers |
