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
    ap.add_argument("--n",    type=int,               default=4,
                    help="matrix dimension N (weights are N×N)")
    ap.add_argument("--k",    type=int,               default=1,
                    help="batch size K (# of input vectors). K>1 emits "
                         "gemm_input_batch.hex with K·N values; TB runs "
                         "a matrix-matrix multiply Y = W · X[:, 0..K-1]")
    ap.add_argument("--seed", type=lambda s: int(s,0), default=0xDEADBEEF)
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    N, K = args.n, args.k

    # Random W with |entries| ≤ 0.5
    W = [[rng.uniform(-0.5, 0.5) for _ in range(N)] for _ in range(N)]
    # Random X (N × K, stored as K vectors of length N)
    X = [[rng.uniform(-0.5, 0.5) for _ in range(N)] for _ in range(K)]
    # Golden Y[k][c] = Σ_r W[r][c] * X[k][r]
    Y = [[sum(W[r][c] * X[k][r] for r in range(N)) for c in range(N)]
         for k in range(K)]

    out = pathlib.Path(args.out_dir)

    with open(out / "gemm_weights.hex", "w") as f:
        f.write(f"// gemm weight angles — N={N}, {N*N} entries, "
                f"row-major (r*N+c)\n")
        f.write("// each = Q0.63 fraction-of-π: acos(W[r][c]) / π · 2^63\n")
        for r in range(N):
            for c in range(N):
                angle = math.acos(W[r][c])
                f.write(f"{q063_frac_pi(angle):016x}"
                        f"  // W[{r}][{c}]={W[r][c]:+.6f}\n")

    # Single-vector input (always written, K=1 default)
    with open(out / "gemm_input.hex", "w") as f:
        f.write(f"// gemm input vector — N={N} entries (first X column)\n")
        f.write("// each = K_INV · x[r] in Q32.32 (pre-scaled for CORDIC gain)\n")
        for r, xi in enumerate(X[0]):
            f.write(f"{q3232(xi * K_INV):016x}  // x[{r}]={xi:+.6f}\n")

    # Batch input for matrix-matrix mode; remove any stale batch file
    # from a previous K>1 run when the current K==1 (otherwise the TB
    # auto-detects batch mode and uses wrong dimensions).
    batch_path = out / "gemm_input_batch.hex"
    if K > 1:
        with open(batch_path, "w") as f:
            f.write(f"// gemm batch input — K={K} vectors × N={N} entries\n")
            f.write("// row-major (k*N + r), pre-scaled by K_INV\n")
            for k in range(K):
                for r, xi in enumerate(X[k]):
                    f.write(f"{q3232(xi * K_INV):016x}"
                            f"  // X[{k}][{r}]={xi:+.6f}\n")
    else:
        if batch_path.exists():
            batch_path.unlink()

    # Golden output (K·N values)
    with open(out / "gemm_golden.hex", "w") as f:
        f.write(f"// gemm golden output — K={K} × N={N} entries\n")
        f.write("// each = Y[k][c] = Σ_r W[r][c] · X[k][r] in Q32.32\n")
        for k in range(K):
            for c, yc in enumerate(Y[k]):
                f.write(f"{q3232(yc):016x}  // Y[{k}][{c}]={yc:+.6f}\n")

    with open(out / "gemm.csv", "w") as f:
        f.write(f"# N={N} K={K}\n")
        f.write("# weights row-major:\n")
        for r in range(N):
            f.write(",".join(f"{v:+.6f}" for v in W[r]) + "\n")
        f.write("# input batch (K rows, N cols):\n")
        for k in range(K):
            f.write(",".join(f"{v:+.6f}" for v in X[k]) + "\n")
        f.write("# golden Y (K rows, N cols):\n")
        for k in range(K):
            f.write(",".join(f"{v:+.6f}" for v in Y[k]) + "\n")

    print(f"  N={N} K={K}  seed={args.seed:#x}")
    print(f"  weights:         {out / 'gemm_weights.hex'}  ({N*N} angles)")
    if K > 1:
        print(f"  input (batch):   {out / 'gemm_input_batch.hex'}  "
              f"({K*N} values = {K} × {N})")
    print(f"  input (single):  {out / 'gemm_input.hex'}     ({N} values)")
    print(f"  golden:          {out / 'gemm_golden.hex'}    "
          f"({K*N} values)")
    print(f"  csv debug:       {out / 'gemm.csv'}")


if __name__ == "__main__":
    main()
