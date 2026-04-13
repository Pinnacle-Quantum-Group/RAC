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

## Known issue (caught by this very harness)

Running `make all` today produces a FAIL. Good — that's the harness doing its
job. The failure isolates a real bug in `rac_dsp.v`'s LUT indexing:

```verilog
// rac_dsp.v, coarse stage:
wire [LUT_BITS-1:0] z_idx = z_r[WIDTH-2 -: LUT_BITS] + (LUT_SIZE/2);
```

This assumes the top `LUT_BITS` of `z_r` span the full ±π input range in a
power-of-two unit, but Q32.32 radians have a range of ±π ≈ ±3.14 — *not*
±4 or some other power-of-two. Bucket boundaries misalign and most test
angles route to the wrong LUT entry.

**Two fixes are possible, and the harness is the right place to validate
whichever we pick:**

1. **Fraction-of-2π input units.** Re-interpret `z_in` as signed Q0.63
   representing θ / (2π), so the full signed range maps to ±π naturally
   and the LUT index becomes a pure bit slice. Caller does one `×1/(2π)`
   conversion up front, then everything downstream is shift-add.

2. **Radian-to-index scaling stage.** Keep Q32.32 radians on the wire,
   add a constant-multiply-by-shift-add stage that scales `z_r` by
   `LUT_SIZE/π`. Synthesizes to ~4 shift-adds (1024/π ≈ 325.95 ≈
   2⁸+2⁶+2²+2¹); residual CORDIC absorbs the rounding error.

The cosim harness makes either fix trivial to validate: tweak `rac_dsp.v`,
re-`make all`, watch the FAIL count drop to zero. This is the entire
point of having bit-level co-simulation — the first time you run it, it
exposes whatever assumptions you quietly baked in.

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
