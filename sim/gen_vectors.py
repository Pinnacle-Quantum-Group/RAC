#!/usr/bin/env python3
"""
gen_vectors.py — generate test vectors + golden outputs for rac_dsp cosim.

Writes three files into the sim/ directory:

  test_vectors.hex   One hex line per test case: "op xxxx yyyy zzzz"
                     (op = 3-bit op_in; x/y/z = 64-bit Q32.32 signed)
                     Consumed by tb_rac_dsp.v via $readmemh.

  golden.hex         One hex line per test case: "xxxx yyyy zzzz"
                     The mathematically-correct rotation result computed
                     in Python with full-precision math.cos / math.sin.
                     Compared against RTL output by cosim.py.

  vectors.csv        Human-readable form of the same data for debugging.

The test set:
  - 32 angle sweep    : θ ∈ [-π, +π] in π/16 steps, (x, y) = (1, 0)
  - 32 magnitudes     : θ = π/4 fixed, |v| ∈ {0.1, 0.2, ..., 3.2}
  - 32 PRNG-seeded    : reproducible uniform θ ∈ (-π, π], |v| ∈ (0, 4)
  - 4  edge cases     : θ = 0, ±π/2, ±π exactly
"""

from __future__ import annotations
import argparse
import math
import pathlib
import random
import struct


HERE = pathlib.Path(__file__).resolve().parent


def q3232(f: float) -> int:
    """Convert float to signed 64-bit Q32.32 integer (round-half-even).
    Used for x_in, y_in, x_gold, y_gold (Cartesian coordinates)."""
    scaled = round(f * (1 << 32))
    # Clamp to int64 range
    if scaled >  (1 << 63) - 1: scaled =  (1 << 63) - 1
    if scaled < -(1 << 63):     scaled = -(1 << 63)
    # Two's complement encoding as unsigned 64-bit for hex emission
    if scaled < 0:
        scaled = scaled + (1 << 64)
    return scaled & ((1 << 64) - 1)


def q063_frac_pi(theta: float) -> int:
    """Convert angle in radians to signed 64-bit Q0.63 fraction-of-π.
    z_signed_int / 2^63 = θ / π. Used for z_in (and z_gold = 0 after
    rotation drives z to zero in RAC-DSP's encoding)."""
    scaled = round(theta / math.pi * (1 << 63))
    # Wrap mod 2^64 so θ just beyond ±π maps to the opposite extreme
    scaled &= (1 << 64) - 1
    return scaled


def from_q3232(u: int) -> float:
    """Unsigned 64-bit two's complement → float."""
    if u >= (1 << 63):
        u = u - (1 << 64)
    return u / (1 << 32)


# CORDIC gain for 16 iterations: K = prod sqrt(1 + 2^-2i), i=0..15
# Same value whether the first 10 iters are run explicitly or via the
# coarse LUT; the LUT's direction bits ARE those 10 iterations folded
# into a single combinational chain, so the magnitude expansion is
# identical. Pre-scaling by K_INV at the caller is what keeps CORDIC
# output magnitude equal to input magnitude.
K_CORDIC = 1.6467602581210655
K_INV    = 1.0 / K_CORDIC              # ≈ 0.6072529350088813


def make_case(op: int, x: float, y: float, theta: float) -> dict:
    """Build one test case dict with op + Q32.32 inputs + golden outputs.

    x_in, y_in are pre-scaled by K_INV — the convention rac_alu_rotate
    uses: drive the CORDIC with K_INV·v, so the K-fold gain the 16-iter
    sequence introduces lands the output back on the true rotation of v.
    Golden is the true mathematical rotation result (NOT K-scaled)."""
    cx = math.cos(theta)
    sx = math.sin(theta)
    gx = x * cx - y * sx
    gy = x * sx + y * cx
    gz = 0.0   # rotation mode drives z to zero

    # Pre-scaled inputs — this is what the TB drives into the RTL.
    x_pre = x * K_INV
    y_pre = y * K_INV

    return {
        "op":      op,
        "x_in":    q3232(x_pre),
        "y_in":    q3232(y_pre),
        "z_in":    q063_frac_pi(theta),   # Q0.63 fraction-of-π
        "x_gold":  q3232(gx),
        "y_gold":  q3232(gy),
        "z_gold":  q063_frac_pi(gz),      # z drives to 0, same encoding
        "x_f":     x, "y_f": y, "theta_f": theta,
        "gx_f":    gx, "gy_f": gy,
    }


def build_vectors(seed=0xC0FFEE) -> list[dict]:
    rng = random.Random(seed)
    cases = []

    # Sweep: 32 angles, (1, 0)
    for k in range(32):
        theta = -math.pi + (2 * math.pi * k / 32)
        cases.append(make_case(0b000, 1.0, 0.0, theta))

    # Magnitudes at θ = π/4
    for k in range(1, 33):
        r = k * 0.1
        cases.append(make_case(0b000, r, 0.0, math.pi / 4))

    # PRNG
    for _ in range(32):
        theta = rng.uniform(-math.pi, math.pi)
        r = rng.uniform(0.1, 4.0)
        # Decompose to (x, y) with random direction
        phi = rng.uniform(0, 2 * math.pi)
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        cases.append(make_case(0b000, x, y, theta))

    # Edge cases
    for theta in (0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi + 1e-12):
        cases.append(make_case(0b000, 1.0, 0.0, theta))

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0xC0FFEE)
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    cases = build_vectors(args.seed)

    # test_vectors.hex: one line per case
    #   Format: "op x_hex y_hex z_hex"
    #   op is a single hex digit; each Q32.32 field is 16 hex chars.
    with open(out / "test_vectors.hex", "w") as f:
        f.write(f"// rac_dsp cosim test vectors — {len(cases)} cases\n")
        f.write("// format: op x_q3232 y_q3232 z_q3232\n")
        for c in cases:
            f.write(f"{c['op']:x} "
                    f"{c['x_in']:016x} "
                    f"{c['y_in']:016x} "
                    f"{c['z_in']:016x}\n")

    # golden.hex: same ordering, just the expected outputs
    with open(out / "golden.hex", "w") as f:
        f.write(f"// rac_dsp cosim golden outputs — {len(cases)} cases\n")
        f.write("// format: x_q3232 y_q3232 z_q3232\n")
        for c in cases:
            f.write(f"{c['x_gold']:016x} "
                    f"{c['y_gold']:016x} "
                    f"{c['z_gold']:016x}\n")

    # CSV for humans
    with open(out / "vectors.csv", "w") as f:
        f.write("idx,op,x_in,y_in,theta,gx,gy\n")
        for i, c in enumerate(cases):
            f.write(f"{i},{c['op']},{c['x_f']:.10f},{c['y_f']:.10f},"
                    f"{c['theta_f']:.10f},{c['gx_f']:.10f},{c['gy_f']:.10f}\n")

    print(f"  {len(cases)} test cases:")
    print(f"    {out / 'test_vectors.hex'}")
    print(f"    {out / 'golden.hex'}")
    print(f"    {out / 'vectors.csv'}")


if __name__ == "__main__":
    main()
