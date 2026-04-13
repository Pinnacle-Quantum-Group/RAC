#!/usr/bin/env python3
"""
cosim.py — diff rac_dsp RTL outputs against golden (and C reference).

Reads:
  test_vectors.hex   input cases driven into RTL + reference
  golden.hex         mathematically-correct results (math.cos/sin)
  rtl_outputs.hex    written by tb_rac_dsp.v via $fwrite
  ref_outputs.hex    written by rac_dsp_ref.c   [optional]

Produces a pass/fail report + ULP-error histogram. Q32.32 ULP is
2^-32 ≈ 2.3e-10 of the represented value.

Exit codes:
  0  all within tolerance
  1  max error exceeded (cosim FAIL)
  2  missing/unreadable input file(s)

Usage:
  python3 sim/cosim.py                  # default tolerance 2^-14 (~0.6e-4)
  python3 sim/cosim.py --tol-bits 20    # stricter (CORDIC-converged)
  python3 sim/cosim.py --verbose        # per-case error detail
"""

from __future__ import annotations
import argparse
import pathlib
import sys


HERE = pathlib.Path(__file__).resolve().parent


def parse_hex_lines(path: pathlib.Path, n_fields: int):
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        parts = s.split()
        if len(parts) < n_fields:
            continue
        # Strip trailing comments within the line
        parts = [p for p in parts if not p.startswith("//")]
        try:
            rows.append([int(p, 16) for p in parts[:n_fields]])
        except ValueError:
            continue
    return rows


def q_to_float(q: int) -> float:
    if q >= (1 << 63):
        q -= (1 << 64)
    return q / (1 << 32)


def abs_q_diff(a: int, b: int) -> int:
    """Absolute difference in Q32.32 ULP (= LSBs)."""
    sa = a - (1 << 64) if a >= (1 << 63) else a
    sb = b - (1 << 64) if b >= (1 << 63) else b
    return abs(sa - sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--tol-bits", type=int, default=14,
                    help="max tolerated ULP error is 2^(32-tol_bits). "
                         "Default 14 → ~6e-5 absolute, matches CORDIC-16 precision.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    paths = {
        "vectors":   d / "test_vectors.hex",
        "golden":    d / "golden.hex",
        "rtl":       d / "rtl_outputs.hex",
        "ref":       d / "ref_outputs.hex",
    }

    missing = [k for k, p in paths.items() if not p.exists()]
    # 'ref' is optional
    critical = [m for m in missing if m != "ref"]
    if critical:
        for m in critical:
            print(f"  MISSING: {paths[m]}", file=sys.stderr)
        print(f"  generate with:  "
              f"python3 {HERE/'gen_vectors.py'} && "
              f"iverilog -g2012 -o tb tb_rac_dsp.v ../rtl/rac_dsp.v && vvp tb",
              file=sys.stderr)
        return 2

    vectors = parse_hex_lines(paths["vectors"], 4)   # op + xyz
    golden  = parse_hex_lines(paths["golden"],  3)
    rtl     = parse_hex_lines(paths["rtl"],     3)
    ref     = parse_hex_lines(paths["ref"],     3) if paths["ref"].exists() else None

    if len(vectors) != len(golden):
        print(f"  vectors ({len(vectors)}) / golden ({len(golden)}) "
              "length mismatch — regenerate", file=sys.stderr)
        return 2
    if len(rtl) != len(golden):
        print(f"  rtl_outputs ({len(rtl)}) / golden ({len(golden)}) "
              "length mismatch — likely a testbench capture bug",
              file=sys.stderr)

    tol_ulp = 1 << (32 - args.tol_bits)
    print(f"  tolerance: 2^{32 - args.tol_bits} ULP = "
          f"{1.0 * tol_ulp / (1 << 32):.3e} absolute\n")

    header = "  idx  theta         RTL vs golden       "
    if ref is not None:
        header += "C_ref vs golden"
    print(header)

    fail_count = 0
    max_err_x  = 0
    max_err_y  = 0
    n = min(len(vectors), len(golden), len(rtl))

    for i in range(n):
        _, _, _, z_in = vectors[i]
        gx, gy, _ = golden[i]
        rx, ry, _ = rtl[i]
        ex = abs_q_diff(rx, gx)
        ey = abs_q_diff(ry, gy)
        max_err_x = max(max_err_x, ex)
        max_err_y = max(max_err_y, ey)

        line_ok = (ex <= tol_ulp and ey <= tol_ulp)
        if not line_ok:
            fail_count += 1

        if args.verbose or not line_ok:
            theta = q_to_float(z_in)
            tag   = "OK " if line_ok else "FAIL"
            msg   = f"  {i:3d}  θ={theta:+9.5f}  [{tag}] x_err={ex:>8d} y_err={ey:>8d}"
            if ref is not None and i < len(ref):
                cx, cy, _ = ref[i]
                ex_c = abs_q_diff(cx, gx)
                ey_c = abs_q_diff(cy, gy)
                msg += f"   | c_ref x_err={ex_c:>8d} y_err={ey_c:>8d}"
            print(msg)

    print()
    print(f"  Cases:     {n}")
    print(f"  Max error: x = {max_err_x} ULP ({q_to_float(max_err_x):.3e}),"
          f" y = {max_err_y} ULP ({q_to_float(max_err_y):.3e})")
    print(f"  Fails:     {fail_count} / {n}")
    if fail_count == 0:
        print("  PASS — all cases within tolerance")
        return 0
    else:
        print("  FAIL — tighten --tol-bits or inspect failing indices above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
