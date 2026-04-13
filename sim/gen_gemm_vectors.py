#!/usr/bin/env python3
"""
gen_gemm_vectors.py — test vectors for the rac_systolic_array GEMM cosim.

Produces a small matrix-vector multiply test case:
  y[c] = Σ_r  W[r,c] * x[r]      (N×N weights, N-element input/output)

Emits four files under sim/:
  gemm_weights.hex   N² Q0.63 fraction-of-π angles = acos(W[r,c])
                      (one per line, row-major r·N+c)
  gemm_input.hex     N   Q32.32 signed values = K_INV * x[r]
                      (pre-scaled so post-CORDIC K gain recovers x·W)
  gemm_golden.hex    N   Q32.32 signed values of golden y[c] = Σ W·x
  gemm.csv           human-readable CSV of W, x, y for debugging

Weight encoding rationale
  Each PE computes x_proj = x_in * cos(pe_angle) * K_CORDIC
  With pe_angle = acos(W) and x_in pre-scaled by K_INV = 1/K_CORDIC,
  x_proj = (x_real * K_INV) * W * K_CORDIC = x_real * W.
  Column sum → y[c] = Σ_r x[r] * W[r,c].

  Constraint: |W[r,c]| ≤ 1 (acos domain). Generator picks |W| ≤ 0.5 to
  keep some headroom and make sign flips likely across the test set.

Usage:
  python3 gen_gemm_vectors.py                # N=4 default
  python3 gen_gemm_vectors.py --n 16         # full 16×16 array
  python3 gen_gemm_vectors.py --seed 0x42    # different PRNG seed
"""

from __future__ import annotations
import argparse
import math
import pathlib
import random


HERE = pathlib.Path(__file__).resolve().parent


K_CORDIC = 1.6467602581210655        # 16-iter circular CORDIC gain
K_INV    = 1.0 / K_CORDIC             # ≈ 0.60725


def q3232(f: float) -> int:
    """float → signed 64-bit Q32.32 two's-complement, as unsigned hex."""
    scaled = round(f * (1 << 32))
    if scaled >  (1 << 63) - 1: scaled =  (1 << 63) - 1
    if scaled < -(1 << 63):     scaled = -(1 << 63)
    if scaled < 0:
        scaled += (1 << 64)
    return scaled & ((1 << 64) - 1)


def q063_frac_pi(theta: float) -> int:
    """radians → signed Q0.63 fraction-of-π (z_signed / 2^63 = θ / π)."""
    scaled = round(theta / math.pi * (1 << 63))
    scaled &= (1 << 64) - 1
    return scaled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int,               default=4)
    ap.add_argument("--seed", type=lambda s: int(s,0), default=0xDEADBEEF)
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    N = args.n

    # Random W with |entries| ≤ 0.5
    W = [[rng.uniform(-0.5, 0.5) for _ in range(N)] for _ in range(N)]
    # Random x with |entries| ≤ 0.5
    x = [rng.uniform(-0.5, 0.5) for _ in range(N)]
    # Golden y[c] = Σ_r W[r][c] * x[r]
    y = [sum(W[r][c] * x[r] for r in range(N)) for c in range(N)]

    out = pathlib.Path(args.out_dir)

    with open(out / "gemm_weights.hex", "w") as f:
        f.write(f"// gemm weight angles — N={N}, {N*N} entries, "
                f"row-major (r*N+c)\n")
        f.write("// each = Q0.63 fraction-of-π: acos(W[r][c]) / π · 2^63\n")
        for r in range(N):
            for c in range(N):
                angle = math.acos(W[r][c])      # acos returns [0, π]
                f.write(f"{q063_frac_pi(angle):016x}  // W[{r}][{c}]={W[r][c]:+.6f}\n")

    with open(out / "gemm_input.hex", "w") as f:
        f.write(f"// gemm input vector — N={N} entries\n")
        f.write(f"// each = K_INV · x[r] in Q32.32 (pre-scaled for CORDIC gain)\n")
        for r, xi in enumerate(x):
            f.write(f"{q3232(xi * K_INV):016x}  // x[{r}]={xi:+.6f}\n")

    with open(out / "gemm_golden.hex", "w") as f:
        f.write(f"// gemm golden output — N={N} entries\n")
        f.write("// each = y[c] = Σ_r W[r][c] · x[r] in Q32.32\n")
        for c, yc in enumerate(y):
            f.write(f"{q3232(yc):016x}  // y[{c}]={yc:+.6f}\n")

    with open(out / "gemm.csv", "w") as f:
        f.write("# N=" + str(N) + "\n")
        f.write("# weights row-major:\n")
        for r in range(N):
            f.write(",".join(f"{v:+.6f}" for v in W[r]) + "\n")
        f.write("# input x:\n")
        f.write(",".join(f"{v:+.6f}" for v in x) + "\n")
        f.write("# golden y:\n")
        f.write(",".join(f"{v:+.6f}" for v in y) + "\n")

    print(f"  N={N}  seed={args.seed:#x}")
    print(f"  weights:   {out / 'gemm_weights.hex'}  ({N*N} angles)")
    print(f"  input:     {out / 'gemm_input.hex'}    ({N} values)")
    print(f"  golden:    {out / 'gemm_golden.hex'}   ({N} values)")
    print(f"  csv debug: {out / 'gemm.csv'}")


if __name__ == "__main__":
    main()
