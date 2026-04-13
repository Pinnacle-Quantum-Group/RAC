#!/usr/bin/env python3
"""
cosim_gemm.py — diff rac_systolic_array RTL output against golden.

Inputs:
  gemm_golden.hex         mathematically correct y = Σ W·x (from gen_gemm_vectors.py)
  rtl_gemm_outputs.hex    output of tb_rac_systolic_array.v
  ref_gemm_outputs.hex    optional, from rac_systolic_ref

Passes if every y[c] in the RTL output is within tolerance of the
golden, where tolerance scales with N (CORDIC-16 error accumulates
across the N-element column sum).

Usage:
  python3 cosim_gemm.py
  python3 cosim_gemm.py --tol-bits 12 --verbose
"""

from __future__ import annotations
import argparse
import pathlib
import sys


HERE = pathlib.Path(__file__).resolve().parent


def parse_hex_lines(path: pathlib.Path):
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        # Take first whitespace token; strip any trailing comment
        tok = s.split()[0]
        try:
            rows.append(int(tok, 16))
        except ValueError:
            continue
    return rows


def q_to_float(q: int) -> float:
    if q >= (1 << 63):
        q -= (1 << 64)
    return q / (1 << 32)


def abs_q_diff(a: int, b: int) -> int:
    sa = a - (1 << 64) if a >= (1 << 63) else a
    sb = b - (1 << 64) if b >= (1 << 63) else b
    return abs(sa - sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--tol-bits", type=int, default=12,
                    help="max per-element ULP error = 2^(32-tol_bits). "
                         "Default 12 → 2.4e-4 abs, allowing N=4 CORDIC-16 "
                         "accumulation headroom.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    paths = {
        "golden": d / "gemm_golden.hex",
        "rtl":    d / "rtl_gemm_outputs.hex",
        "ref":    d / "ref_gemm_outputs.hex",
    }

    missing = [k for k, p in paths.items() if not p.exists() and k != "ref"]
    if missing:
        for m in missing:
            print(f"  MISSING: {paths[m]}", file=sys.stderr)
        print(f"  run:  make gemm", file=sys.stderr)
        return 2

    golden = parse_hex_lines(paths["golden"])
    rtl    = parse_hex_lines(paths["rtl"])
    ref    = parse_hex_lines(paths["ref"]) if paths["ref"].exists() else None

    if len(rtl) != len(golden):
        print(f"  rtl ({len(rtl)}) / golden ({len(golden)}) length mismatch",
              file=sys.stderr)
        return 2

    tol_ulp = 1 << (32 - args.tol_bits)
    print(f"  tolerance: 2^{32 - args.tol_bits} ULP = "
          f"{1.0 * tol_ulp / (1 << 32):.3e} absolute\n")

    header = f"  {'idx':>3}  {'golden':>12}  {'RTL':>12}  {'err(ULP)':>12}"
    if ref:
        header += f"  {'c_ref':>12}"
    print(header)
    fail = 0
    max_err = 0
    n = len(golden)
    for i in range(n):
        g = golden[i]
        r = rtl[i]
        err = abs_q_diff(r, g)
        max_err = max(max_err, err)
        ok = err <= tol_ulp
        if not ok:
            fail += 1
        if args.verbose or not ok:
            tag = "OK " if ok else "FAIL"
            extra = ""
            if ref and i < len(ref):
                extra = f"  c_ref={q_to_float(ref[i]):+.6f}"
            print(f"  [{i}] golden={q_to_float(g):+.6f}  "
                  f"rtl={q_to_float(r):+.6f}  err={err:>9d}  [{tag}]{extra}")

    print()
    print(f"  Cases:     {n}")
    print(f"  Max error: {max_err} ULP ({q_to_float(max_err):.3e})")
    print(f"  Fails:     {fail} / {n}")
    if fail == 0:
        print("  PASS — all cases within tolerance")
        return 0
    print("  FAIL — tighten --tol-bits or inspect above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
