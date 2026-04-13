#!/usr/bin/env python3
"""
gen_coarse_lut.py — emit rac_dsp.v's coarse direction-bit LUT.

For each of 1024 angle buckets in [-π/2, +π/2], precompute the 10-bit
CORDIC direction sequence (d_0 .. d_9) that a reference circular-rotation
CORDIC would take. Hardware then applies those 10 shift-adds
combinationally — no multipliers required.

Outputs two .mem files in $readmemh format:
  cordic_coarse_lut.mem   — 1024 lines, each a hex 10-bit dir vector
  cordic_atanh.mem        — 6 lines, each a Q32.32 atanh(2^-i) constant
                            for i in {10, 11, 12, 13, 14, 15}

Usage:
  python3 rtl/gen_coarse_lut.py                 # writes to rtl/
  python3 rtl/gen_coarse_lut.py --lut-bits 12   # larger LUT (4096 buckets)
"""

from __future__ import annotations
import argparse
import math
import pathlib


def atan(x):
    return math.atan(x)


def cordic_dirs_for_angle(target: float, n_iters: int) -> int:
    """Return an n_iters-bit int encoding the CORDIC direction sequence
    that rotates a vector by `target` radians.

    CORDIC rotation-mode convention: z starts at `target` and is driven
    toward zero. At iter i we pick d_i = sign(z) and update
    z ← z - d_i · atan(2^-i). Bit i of the returned int is 1 iff d_i = +1.

    Applying the same direction bits to (x, y) via
        d = +1:  x' = x - y·2^-i,  y' = y + x·2^-i
        d = -1:  x' = x + y·2^-i,  y' = y - x·2^-i
    rotates (x, y) by `target` radians (with the standard CORDIC K gain)."""
    z = target
    bits = 0
    for i in range(n_iters):
        step = atan(2 ** -i)
        if z >= 0.0:
            z -= step
            bits |= (1 << i)
        else:
            z += step
    return bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut-bits", type=int, default=10,
                    help="coarse LUT address bits (default 10 → 1024 entries)")
    ap.add_argument("--residual", type=int, default=9,
                    help="residual CORDIC stages (default 9, matches rac_dsp.v)")
    ap.add_argument("--residual-start", type=int, default=None,
                    help="first residual physical shift. Default = "
                         "LUT_BITS-2 (overlap coarse by 2 for convergence "
                         "margin — see RAC-DSP-DATASHEET.md §4)")
    ap.add_argument("--out-dir", default=str(pathlib.Path(__file__).parent))
    args = ap.parse_args()

    if args.residual_start is None:
        args.residual_start = args.lut_bits - 2

    out_dir = pathlib.Path(args.out_dir)
    lut_size = 1 << args.lut_bits

    # ── Coarse direction-bit LUT ───────────────────────────────────────
    # Buckets cover [-π/2, +π/2]. Bucket k's CENTER angle is
    #   (k - lut_size/2 + 0.5) · π / lut_size
    # so every angle within ±(π/2)/lut_size of a bucket center lands in
    # that bucket. Using bucket centers (not edges) means the residual
    # CORDIC converges quickly — typical residual |z| < π/(2·lut_size).
    lut_path = out_dir / "cordic_coarse_lut.mem"
    with open(lut_path, "w") as f:
        f.write("// rac_dsp coarse direction-bit LUT\n")
        f.write(f"// lut_bits={args.lut_bits} → {lut_size} entries\n")
        f.write(f"// each line = {args.lut_bits}-bit CORDIC direction sequence\n")
        f.write("// target angle = bucket center, so residual converges fast\n")
        hex_digits = max(1, (args.lut_bits + 3) // 4)
        for k in range(lut_size):
            theta = (k - lut_size / 2 + 0.5) * math.pi / lut_size
            d = cordic_dirs_for_angle(theta, args.lut_bits)
            f.write(f"{d:0{hex_digits}x}\n")
    print(f"wrote {lut_path} ({lut_size} entries)")

    # ── atan / atanh ROMs in Q0.63 fraction-of-π ───────────────────────
    # Encoding: stored value × π = angle in radians.
    #   q = round(atan(2^-i) / π · 2^63)
    # The z-pipeline inside rac_dsp operates in this fraction-of-π scale;
    # keeping the constants in the same units is what makes the whole
    # CORDIC multiplier-free at the gate level.

    def q063_frac_pi(x: float) -> int:
        scaled = int(round(x / math.pi * (1 << 63)))
        if scaled < 0:
            scaled += (1 << 64)
        return scaled & ((1 << 64) - 1)

    # Circular atan ROM — sized to cover both coarse and residual.
    atan_rom_size = args.residual_start + args.residual
    atan_path = out_dir / "cordic_atan.mem"
    with open(atan_path, "w") as f:
        f.write("// rac_dsp circular atan constants — Q0.63 fraction-of-π\n")
        f.write(f"// {atan_rom_size} entries, one per iter (shift = index)\n")
        f.write("// stored_value * π = atan(2^-i) in radians\n")
        for i in range(atan_rom_size):
            f.write(f"{q063_frac_pi(math.atan(2.0 ** -i)):016x}\n")
    print(f"wrote {atan_path} ({atan_rom_size} entries, shifts 0..{atan_rom_size-1})")

    # Residual atanh ROM — Walther hyperbolic, same fraction-of-π units.
    atanh_path = out_dir / "cordic_atanh.mem"
    with open(atanh_path, "w") as f:
        f.write("// rac_dsp residual atanh constants — Q0.63 fraction-of-π\n")
        f.write(f"// residual_start={args.residual_start} residual={args.residual}\n")
        for stage in range(args.residual):
            shift = args.residual_start + stage
            f.write(f"{q063_frac_pi(math.atanh(2.0 ** -shift)):016x}\n")
    print(f"wrote {atanh_path} ({args.residual} entries, shifts {args.residual_start}..{args.residual_start+args.residual-1})")

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print(f"Integration:")
    print(f"  rac_dsp #(.LUT_BITS({args.lut_bits}), "
          f".RESIDUAL({args.residual}),")
    print(f"           .INIT_LUT(\"cordic_coarse_lut.mem\"),")
    print(f"           .INIT_ATANH(\"cordic_atanh.mem\"))")


if __name__ == "__main__":
    main()
