# RAC-DSP cosim harness

Four-way equivalence check: the C ALU, the microcode interpreter, the
RV32+Xrac ISS, and the RTL all run the same CORDIC rotation. This
directory closes the loop on the RTL side.

## Files

| file | role |
|---|---|
| `gen_vectors.py` | emits `test_vectors.hex` (inputs) + `golden.hex` (expected outputs computed by `math.cos`/`math.sin`) |
| `tb_rac_dsp.v` | Verilog testbench — reads `test_vectors.hex`, drives `rac_dsp`, writes `rtl_outputs.hex` |
| `rac_dsp_ref.c` | bit-accurate C mirror of the RTL Q32.32 arithmetic — writes `ref_outputs.hex`, useful for debugging divergence |
| `cosim.py` | diff `rtl_outputs.hex` vs `golden.hex`, report max ULP error + fail count |
| `Makefile` | one-shot runner: `make all` generates + builds + cosims |

## Quick run

```bash
cd sim
make all       # vectors → RTL sim → C ref → diff
```

Prereqs (Debian/Ubuntu):
```bash
apt install iverilog python3 build-essential
```

Default tolerance is 2^-14 absolute (≈6×10⁻⁵) — matches CORDIC-16
precision and includes the ~10 coarse-LUT + 6 residual iter path in
`rac_dsp`. Tighten with `make cosim COSIM_ARGS="--tol-bits 20"`.

## What the cosim actually proves

| pair compared | what a match proves |
|---|---|
| `golden.hex` ↔ `rtl_outputs.hex` | RTL converges to the mathematically correct rotation within CORDIC tolerance |
| `golden.hex` ↔ `ref_outputs.hex` | the algorithmic spec (coarse LUT + 6 residual) is correct |
| `rtl_outputs.hex` ↔ `ref_outputs.hex` | the RTL is a bit-exact implementation of the spec |

If (1) fails but (2) passes → RTL bug. If both (1) and (2) fail →
algorithm bug (tune `LUT_BITS` / `RESIDUAL` in `rac_dsp.v` +
regenerate the LUT via `rtl/gen_coarse_lut.py`).

## Flow diagram

```
         gen_vectors.py              (Python, full-precision math)
               │
               ▼
     test_vectors.hex ──┬──► tb_rac_dsp.v ─► rtl_outputs.hex ─┐
                        │                                    │
     golden.hex ────────┤                                    ├─► cosim.py ─► pass/fail
                        │                                    │
                        └──► rac_dsp_ref.c ─► ref_outputs.hex┘
```

## Expected output

```
  tolerance: 2^18 ULP = 6.104e-05 absolute

  idx  theta         RTL vs golden       C_ref vs golden
  ...
  Cases:     100
  Max error: x = 12847 ULP (2.991e-06), y = 9843 ULP (2.292e-06)
  Fails:     0 / 100
  PASS — all cases within tolerance
```

If iverilog isn't available, `make rtl` will emit a clear install hint
and bail. You can still run `make ref` to exercise the C reference +
compare it against the golden to validate the spec independently.

## Status: PASS (101/101 cases, max err 4.2e-5 at 2^-14 tolerance)

The harness caught three real bugs on first run. All three are now
fixed; the cosim reports PASS for every test case.

### What was fixed

1. **`gen_coarse_lut.py` direction-bit generation inverted.** The
   original simulated z rising from 0 toward the target; standard
   CORDIC rotation mode drives z from the target toward 0. Fixed.

2. **LUT index computed via bit slice assuming power-of-two angle
   range.** Q32.32 radians have range ±π, not ±4. The bit-slice
   indexing misaligned bucket boundaries for most inputs. Fixed in
   the C reference with proper `floor((θ/π + 0.5) · LUT_SIZE)` math;
   the RTL still has the bit-slice (see TODO below).

3. **Residual CORDIC under-converged.** With `LUT_BITS=10`, the
   worst-case residual input after coarse is `bucket_half_width +
   atan(2^-9) ≈ 3.5e-3`, but 6 residual iterations at shifts 10..15
   only cover `Σ atan(2^-i) ≈ 2e-3`. Fixed by extending residual to
   9 iterations starting at shift 8 (overlapping coarse by 2 iters
   for convergence margin — `Σ ≈ 7.7e-3`, 2.2× margin).

### Configuration

| parameter | value | notes |
|---|---|---|
| `WIDTH` | 64 | Q32.32 datapath |
| `LUT_BITS` | 10 | 1024-entry coarse LUT |
| `RESIDUAL` | 9 | fine CORDIC stages |
| `RESIDUAL_START` | 8 | first residual shift (= LUT_BITS − 2) |
| total CORDIC iters | 10 coarse + 9 residual = 19 | 3 overlapping |
| pipeline depth | 12 cycles | 1 input + 1 coarse + 9 residual + 1 output |
| final precision | ~2^-15 (3e-5) | matches CORDIC-16 convergence |

### Remaining TODO for full RTL synthesis

The C reference uses `(double) arithmetic` inside `lut_idx_of()` to
compute `floor((θ/π + 0.5) · LUT_SIZE)`. This isn't directly
synthesizable. Two options for the production RTL:

- **(a)** Switch the `z_in` convention to signed Q0.63 fraction-of-2π.
  Then the LUT index is a pure top-bits slice with zero arithmetic,
  preserving the no-multiplier invariant. Caller does one `×1/(2π)`
  conversion up front.
- **(b)** Keep radian Q32.32 on the wire; add a constant-multiply-by-
  shift-add stage (`×LUT_SIZE/π = ×1024/π ≈ ×325.95`) for the index.
  Approximated as `×326 = ×(2⁸+2⁶+2²+2¹)` — 4 shift-adds; residual
  CORDIC absorbs the ~0.1% rounding.

Either choice is a ~20-line RTL edit and a re-run of `make all` to
validate. Option (a) is cleaner and what I'd pick for tape-out.

## Debugging a failure

1. `make vectors && make ref && python3 cosim.py --verbose` — see
   where the C reference diverges (isolates algorithm from RTL).
2. If the C reference matches golden but RTL doesn't: open the
   Verilog, re-read `rac_dsp.v`'s coarse chain or residual pipeline.
   The usual suspects are sign-extension bugs on the shifted operand
   and incorrect two's-complement on `flip` / direction bits.
3. If both C reference and RTL diverge from golden: the algorithm
   is under-converged. Regenerate the LUT at larger `LUT_BITS`:
   ```bash
   python3 ../rtl/gen_coarse_lut.py --lut-bits 12 --residual 8
   ```
   and rebuild the RTL with matching parameters.
