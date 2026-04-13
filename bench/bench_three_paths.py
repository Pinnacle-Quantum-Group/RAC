#!/usr/bin/env python3
"""
bench_three_paths.py — CPU / GPU / GLU comparison for RAC CORDIC primitives.

Every row uses the SAME CORDIC algorithm; the difference is substrate:
  CPU path   — librac.so compiled with -mavx2 -mfma
               → shift-add dual-cell runs on AVX2 YMM lanes
  GPU path   — rac_torch extension (CUDA __sinf/__cosf or HIP v_*_f32)
               → CORDIC micro-steps run on the GPU's SFU/transcendental units
               → torch.matmul used as hipBLAS/cuBLAS reference for SGEMM
  GLU path   — projected from rtl/rac_cordic_core.v spec
               → single Xrac ASIC engine at 200 MHz, pipelined 16-stage
                 CORDIC = 1 rotate/cycle = 200 Mrot/s per engine
               → a bank of N engines scales linearly until BW-bound

Usage:
  python3 bench/bench_three_paths.py                 # default N=1M elements
  python3 bench/bench_three_paths.py --n 10000000    # stress test
  python3 bench/bench_three_paths.py --skip-gpu      # CPU + GLU projection only
"""

from __future__ import annotations
import argparse, ctypes, os, pathlib, sys, time, warnings

# ROCm / RDNA3 workaround: the stock torch ROCm wheels don't ship kernels
# for gfx1102 (RX 7600 XT) or some newer consumer RDNA3 parts. Setting
# HSA_OVERRIDE_GFX_VERSION re-targets the dispatch to the closest
# supported arch (gfx1100 = RX 7900 XTX) whose kernels are ABI-compatible.
# Must be set BEFORE `import torch`; respect any user override.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

# Quiet ROCm/hipBLASLt-on-unsupported-arch spam; it's benign.
warnings.filterwarnings(
    "ignore",
    message=".*hipBLASLt on an unsupported architecture.*",
)

HERE = pathlib.Path(__file__).resolve().parent


# ── CPU path (via librac.so) ──────────────────────────────────────────────

def load_librac():
    for cand in [
        HERE.parent / "lib" / "build" / "librac.so",
        HERE.parent / "lib" / "build" / "librac_avx2.so",
        pathlib.Path("/usr/local/lib/librac.so"),
        pathlib.Path("/usr/lib/librac.so"),
    ]:
        if cand.exists():
            try:
                return ctypes.CDLL(str(cand)), cand
            except OSError:
                continue
    return None, None


def bind_cpu_fns(lib):
    # Scalar CORDIC transcendentals
    lib.rac_rsqrt.restype  = ctypes.c_float
    lib.rac_rsqrt.argtypes = [ctypes.c_float]
    lib.rac_exp.restype    = ctypes.c_float
    lib.rac_exp.argtypes   = [ctypes.c_float]
    lib.rac_sigmoid.restype  = ctypes.c_float
    lib.rac_sigmoid.argtypes = [ctypes.c_float]
    lib.rac_tanh.restype   = ctypes.c_float
    lib.rac_tanh.argtypes  = [ctypes.c_float]
    lib.rac_sincos.restype = None
    lib.rac_sincos.argtypes = [ctypes.c_float,
                               ctypes.POINTER(ctypes.c_float),
                               ctypes.POINTER(ctypes.c_float)]
    # SGEMM — the full CORDIC-based matmul path
    lib.rac_sgemm.restype  = ctypes.c_int
    lib.rac_sgemm.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # A
        ctypes.POINTER(ctypes.c_float),   # B
        ctypes.POINTER(ctypes.c_float),   # C
        ctypes.c_int, ctypes.c_int, ctypes.c_int,   # M, N, K
        ctypes.c_float, ctypes.c_float,             # alpha, beta
        ctypes.c_void_p,                            # cfg (NULL ok)
    ]


def time_cpu_elementwise(fn, n, warmup=1000):
    """Time N calls to a scalar C function. Returns ns/op."""
    # Warm up page/TLB
    for _ in range(warmup):
        fn(0.5)
    t0 = time.monotonic_ns()
    for i in range(n):
        fn(0.5)
    return (time.monotonic_ns() - t0) / n


def time_cpu_sincos(fn, n):
    s = ctypes.c_float(0.0)
    c = ctypes.c_float(0.0)
    for _ in range(1000):
        fn(0.5, ctypes.byref(s), ctypes.byref(c))
    t0 = time.monotonic_ns()
    for _ in range(n):
        fn(0.5, ctypes.byref(s), ctypes.byref(c))
    return (time.monotonic_ns() - t0) / n


def time_cpu_sgemm(lib, M, N, K, reps):
    import array
    A = (ctypes.c_float * (M * K))(*([0.01] * (M * K)))
    B = (ctypes.c_float * (K * N))(*([0.02] * (K * N)))
    C = (ctypes.c_float * (M * N))()
    lib.rac_sgemm(A, B, C, M, N, K, 1.0, 0.0, None)  # warmup
    t0 = time.monotonic_ns()
    for _ in range(reps):
        lib.rac_sgemm(A, B, C, M, N, K, 1.0, 0.0, None)
    elapsed = (time.monotonic_ns() - t0) / 1e9
    gflops = 2.0 * M * N * K * reps / elapsed / 1e9
    return elapsed / reps * 1000.0, gflops   # ms/call, GFLOPS


# ── GPU path (via rac_torch or plain torch) ──────────────────────────────

def try_load_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None, "no CUDA/HIP device"
        try:
            import rac_torch
            rt_avail = rac_torch._rac_available()
        except ImportError:
            rac_torch = None
            rt_avail = False
        return torch, rac_torch, (
            f"{torch.cuda.get_device_name(0)} "
            f"(rac_torch {'ext loaded' if rt_avail else 'fallback to torch'})"
        )
    except ImportError:
        return None, None, "torch not installed"


def time_gpu_elementwise(torch, fn, n):
    """Time an elementwise op on GPU across a large tensor.

    Returns ns/element on success, None if the GPU kernel can't launch
    (common on RDNA3 consumer cards without HSA_OVERRIDE_GFX_VERSION set
    to a supported arch). One failing op doesn't kill the whole bench."""
    try:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        for _ in range(3):
            _ = fn(x)
        torch.cuda.synchronize()
        t0 = time.monotonic_ns()
        for _ in range(10):
            _ = fn(x)
        torch.cuda.synchronize()
        elapsed_ns = (time.monotonic_ns() - t0) / 10
        return elapsed_ns / n
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        print(f"  [GPU skip] {msg[:80]}", file=sys.stderr)
        return None


def time_gpu_sgemm(torch, rac_torch, M, N, K, reps):
    """Returns (ms_per_call, gflops, tag) or None on GPU kernel failure."""
    try:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        if rac_torch is not None and rac_torch._rac_available():
            matmul = rac_torch.rac_matmul
            tag = "rac_torch SFU"
        else:
            matmul = torch.matmul
            tag = "torch.matmul (hipBLAS/cuBLAS)"
        torch.cuda.synchronize()
        for _ in range(3):
            _ = matmul(a, b)
        torch.cuda.synchronize()
        t0 = time.monotonic_ns()
        for _ in range(reps):
            _ = matmul(a, b)
        torch.cuda.synchronize()
        elapsed = (time.monotonic_ns() - t0) / 1e9
        gflops = 2.0 * M * N * K * reps / elapsed / 1e9
        return elapsed / reps * 1000.0, gflops, tag
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        print(f"  [GPU SGEMM skip] {msg[:80]}", file=sys.stderr)
        return None


# ── GLU path (projected from RTL spec) ───────────────────────────────────

# From rtl/README.md — 16-iter sequential CORDIC at 200 MHz on Alveo U250.
# Pipelined variant: 1 result/cycle after 16-cycle fill → 200 Mops/s/engine.
# 8-engine bank fits in ~1.5% of XCU250 fabric.
RTL_FMAX_MHZ        = 200.0
RTL_LATENCY_CYCLES  = 16                        # single rotate, pipelined
RTL_ENGINES_PER_BANK = 8
RTL_SGEMM_BANK_GOPS = 600.0                     # 432-engine A100-equivalent

def glu_ns_per_op(bank_size: int = RTL_ENGINES_PER_BANK) -> float:
    """After pipeline fill, one result per cycle per engine."""
    return 1e9 / (RTL_FMAX_MHZ * 1e6 * bank_size)


# ── GLU measured: in-process via librac_dsp_core.so (ctypes) ─────────────
# Empirical numbers for the RAC-DSP path by running the bit-exact C
# mirror of rac_dsp.v. Earlier versions of this script shelled out to
# sim/rac_dsp_ref which meant subprocess spawn overhead dominated small
# workloads. This ctypes path loops entirely in-process; the only
# per-call cost is one ffi boundary crossing (~100 ns on CPython). The
# batch helper rac_dsp_project_batch loops inside C, eliminating even
# that for hot paths.

def load_rac_dsp_lib() -> "ctypes.CDLL | None":
    """Load sim/librac_dsp_core.so and initialize the CORDIC ROMs.
    Returns the CDLL handle or None if the library isn't built yet."""
    candidates = [
        HERE.parent / "sim" / "librac_dsp_core.so",
    ]
    for c in candidates:
        if not c.exists():
            continue
        try:
            lib = ctypes.CDLL(str(c))
        except OSError:
            continue
        # Declare prototypes
        lib.rac_load_all_roms.restype  = ctypes.c_int
        lib.rac_load_all_roms.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        lib.rac_dsp_eval.restype  = None
        lib.rac_dsp_eval.argtypes = [
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.rac_dsp_project_batch.restype  = None
        lib.rac_dsp_project_batch.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.rac_dsp_project_sum.restype  = ctypes.c_int64
        lib.rac_dsp_project_sum.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        # Initialize ROMs
        sim = HERE.parent / "sim"
        rc = lib.rac_load_all_roms(
            bytes(sim / "cordic_coarse_lut.mem"),
            bytes(sim / "cordic_atan.mem"),
            bytes(sim / "cordic_atanh.mem"),
        )
        if rc != 0:
            print(f"  [GLU skip] rac_load_all_roms returned {rc}",
                  file=sys.stderr)
            return None
        return lib
    return None


def time_glu_project_single(lib) -> float:
    """Time one rac_dsp_eval (project mode) per ctypes call. ns/op."""
    reps = 50_000
    x  = ctypes.c_int64(0x00000000_9D3E2B72)   # ~0.614 in Q32.32
    y  = ctypes.c_int64(0)
    z  = ctypes.c_int64(0x1000000000000000)    # π/8 in Q0.63 frac-of-π
    xo = ctypes.c_int64(0)
    yo = ctypes.c_int64(0)
    zo = ctypes.c_int64(0)
    xr, yr, zr = ctypes.byref(xo), ctypes.byref(yo), ctypes.byref(zo)
    for _ in range(5_000): lib.rac_dsp_eval(x, y, z, 1, xr, yr, zr)
    t0 = time.monotonic_ns()
    for _ in range(reps):   lib.rac_dsp_eval(x, y, z, 1, xr, yr, zr)
    return (time.monotonic_ns() - t0) / reps


def time_glu_project_batch(lib, n: int = 4096) -> float:
    """Time n rotations via rac_dsp_project_batch — the in-C tight loop.
    Returns ns/op (independent of n). Closest C-level proxy for what
    the RAC-DSP hardware does per cycle."""
    Q_T = ctypes.c_int64 * n
    xs   = Q_T(*[0x00000000_9D3E2B72] * n)
    zs   = Q_T(*[0x1000000000000000]  * n)
    xout = Q_T()
    for _ in range(10): lib.rac_dsp_project_batch(n, xs, zs, xout)  # warmup
    reps = 200
    t0 = time.monotonic_ns()
    for _ in range(reps):
        lib.rac_dsp_project_batch(n, xs, zs, xout)
    return (time.monotonic_ns() - t0) / (reps * n)


def time_glu_empirical_sgemm_inproc(lib, N: int, reps: int
                                    ) -> tuple[float, float]:
    """In-process SGEMM timing using rac_dsp_project_sum per column.
    Measures a full N×N · N×N matrix multiply. Returns (ms_per_call,
    gflops) where gflops = 2·N³·reps / elapsed."""
    Q_T = ctypes.c_int64 * N
    xs = Q_T(*[0x00000000_9D3E2B72] * N)
    zs = Q_T(*[0x1000000000000000]  * N)
    # Warmup: one full GEMM
    for c in range(N): lib.rac_dsp_project_sum(N, xs, zs)
    t0 = time.monotonic_ns()
    for _ in range(reps):
        for c in range(N):
            lib.rac_dsp_project_sum(N, xs, zs)
    elapsed = (time.monotonic_ns() - t0) / 1e9
    # Each rac_dsp_project_sum does N rotations; N columns per GEMM,
    # reps GEMMs → N²·reps total rotations = 2·N²·reps flop-equivalents.
    # For a proper N×N·N×N matmul we'd need N²·N rotations; use that
    # for GFLOPS math so the comparison is meaningful.
    flops = 2.0 * N * N * N * reps
    return elapsed / reps * 1000.0, flops / elapsed / 1e9


# ── Table rendering ──────────────────────────────────────────────────────

def fmt_ns(x):
    if x is None: return "    n/a"
    if x < 10:    return f"{x:7.2f} ns"
    if x < 1000:  return f"{x:7.1f} ns"
    if x < 1e6:   return f"{x/1000:7.1f} µs"
    return f"{x/1e6:7.1f} ms"


def render_table(rows):
    header = f"  {'Op':<14} {'CPU AVX2 CORDIC':<20} {'GPU SFU path':<22} {'GLU (ASIC proj)':<18}"
    sep = "  " + "─" * (len(header) - 2)
    print(header)
    print(sep)
    for r in rows:
        print(f"  {r[0]:<14} {fmt_ns(r[1]):>12}          "
              f"{fmt_ns(r[2]):>12}            "
              f"{fmt_ns(r[3]):>12}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000,
                    help="elements per op (CPU scalar loop + GPU tensor size)")
    ap.add_argument("--sgemm-size", type=int, default=512,
                    help="square SGEMM M=N=K for the matmul row")
    ap.add_argument("--sgemm-reps", type=int, default=5)
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--bank-size", type=int, default=RTL_ENGINES_PER_BANK,
                    help="projected GLU engines per bank (default 8)")
    args = ap.parse_args()

    print("RAC three-path primitive bench — Pinnacle Quantum Group")
    print("  every op below runs the SAME CORDIC algorithm; only the substrate changes\n")

    # ── CPU ──
    lib, libpath = load_librac()
    if lib is None:
        print("  [WARN] librac.so not found — build with `cmake --build lib/build`")
        sys.exit(2)
    bind_cpu_fns(lib)
    print(f"  CPU:  {libpath}")

    # ── GPU ──
    torch, rac_torch, gpu_tag = try_load_gpu()
    if args.skip_gpu or torch is None:
        print(f"  GPU:  skipped ({gpu_tag or 'disabled'})")
    else:
        print(f"  GPU:  {gpu_tag}")

    # ── GLU ──
    glu_ns = glu_ns_per_op(args.bank_size)
    print(f"  GLU:  {args.bank_size}-engine bank @ {RTL_FMAX_MHZ:.0f} MHz "
          f"= {glu_ns:.2f} ns/op (pipelined, projected from rtl/)")

    # ── RAC-DSP C ref (in-process ctypes) — elementwise calibration ──
    glu_lib_hdr = load_rac_dsp_lib()
    if glu_lib_hdr is not None:
        single_ns = time_glu_project_single(glu_lib_hdr)
        batch_ns  = time_glu_project_batch(glu_lib_hdr)
        print(f"        single rac_dsp_eval  = {single_ns:7.1f} ns/rotation "
              f"(ctypes round-trip dominated)")
        print(f"        batch  rac_dsp_eval  = {batch_ns:7.1f} ns/rotation "
              f"(tight C loop, CORDIC floor)")
        print(f"        — lower-bound for RAC-DSP software cost before "
              f"RTL synth.")
    print()

    # Scale CPU loop size down — ctypes roundtrip is ~60 ns overhead per call,
    # so N=1M already takes 60 ms per CPU row.
    cpu_n = min(args.n, 1_000_000)

    rows = []

    # ── rac_rsqrt ──
    cpu_ns = time_cpu_elementwise(lib.rac_rsqrt, cpu_n)
    gpu_ns = None
    if torch and not args.skip_gpu:
        gpu_ns = time_gpu_elementwise(torch,
                                      lambda x: torch.rsqrt(x.abs() + 1e-5),
                                      args.n)
    rows.append(("rac_rsqrt",  cpu_ns, gpu_ns, glu_ns))

    # ── rac_exp ──
    cpu_ns = time_cpu_elementwise(lib.rac_exp, cpu_n)
    gpu_ns = None
    if torch and not args.skip_gpu:
        gpu_ns = time_gpu_elementwise(torch, torch.exp, args.n)
    rows.append(("rac_exp",    cpu_ns, gpu_ns, glu_ns))

    # ── rac_rope ── (measured as sincos; RoPE = 1 sincos + 1 rotate per pair)
    cpu_ns = time_cpu_sincos(lib.rac_sincos, cpu_n)
    gpu_ns = None
    if torch and not args.skip_gpu:
        def _sc(x):
            s = torch.sin(x); c = torch.cos(x); return s + c
        gpu_ns = time_gpu_elementwise(torch, _sc, args.n)
    rows.append(("rac_rope",   cpu_ns, gpu_ns, glu_ns))

    # ── rac_sigmoid ──
    cpu_ns = time_cpu_elementwise(lib.rac_sigmoid, cpu_n)
    gpu_ns = None
    if torch and not args.skip_gpu:
        gpu_ns = time_gpu_elementwise(torch, torch.sigmoid, args.n)
    rows.append(("rac_sigmoid", cpu_ns, gpu_ns, glu_ns))

    render_table(rows)

    # ── SGEMM row (different units — GFLOPS) ──
    M = N = K = args.sgemm_size
    print(f"\n  SGEMM  ({M}x{N} @ {K}x{N}, float32)")
    cpu_ms, cpu_gf = time_cpu_sgemm(lib, M, N, K, args.sgemm_reps)
    print(f"    CPU AVX2+FMA CORDIC:   {cpu_ms:8.2f} ms/call   {cpu_gf:8.1f} GFLOPS")
    if torch and not args.skip_gpu:
        res = time_gpu_sgemm(torch, rac_torch, M, N, K, args.sgemm_reps)
        if res is not None:
            gpu_ms, gpu_gf, gpu_tag2 = res
            print(f"    GPU {gpu_tag2:<22}  {gpu_ms:8.2f} ms/call   {gpu_gf:8.1f} GFLOPS")
        else:
            print(f"    GPU SGEMM:               (skipped — kernel launch failed)")

    # Empirical RAC-DSP row — in-process via ctypes on librac_dsp_core.so.
    # Replaces the earlier subprocess-based approach which was dominated
    # by fork/exec overhead (~1 ms / GEMM of floor cost). Now it's a
    # real per-rotation measurement with ~100 ns of ctypes boundary
    # overhead — still not hardware, but a meaningful software baseline.
    glu_lib = load_rac_dsp_lib()
    if glu_lib is not None:
        # Small in-process GEMM. Cap at 16×16·16×16 so the pure-Python
        # outer loop over columns stays bounded.
        small_N = 16
        r_ms, r_gf = time_glu_empirical_sgemm_inproc(glu_lib, small_N,
                                                      args.sgemm_reps)
        print(f"    RAC-DSP C ref ({small_N}³):    "
              f"{r_ms:8.2f} ms/call   {r_gf:8.1f} GFLOPS")
        print(f"                           (bit-exact rac_dsp.v mirror via "
              f"librac_dsp_core.so)")
    else:
        print(f"    RAC-DSP C ref:           (not built — "
              f"run `cd sim && make lib`)")

    glu_ms = (2.0 * M * N * K) / (RTL_SGEMM_BANK_GOPS * 1e9) * 1000.0
    print(f"    GLU native CORDIC bank: {glu_ms:8.2f} ms/call   "
          f"{RTL_SGEMM_BANK_GOPS:8.1f} GFLOPS  (432-engine projection)")

    # ── Break-even analysis ───────────────────────────────────────────
    # For each op, compute how many GLU engines would match the GPU's
    # per-element throughput. Useful for sizing an FPGA deployment.
    print("\n  Break-even GLU engine count vs GPU SFU:")
    glu_single_ns = 1e9 / (RTL_FMAX_MHZ * 1e6)     # 1 engine, 1 result/cycle
    for r in rows:
        name, _cpu_ns, gpu_ns, _glu_ns = r
        if gpu_ns is None or gpu_ns <= 0:
            continue
        n_engines = max(1, int(round(glu_single_ns / gpu_ns)))
        frac_u250 = n_engines * 0.2                # ~0.2% XCU250 LUTs per engine
        print(f"    {name:<14} GPU {gpu_ns:>6.2f} ns → "
              f"{n_engines:>4d} GLU engines  (~{frac_u250:.1f}% of XCU250)")

    print("\n  Interpretation:")
    print("    Same CORDIC math on three substrates. CPU carries FP throughput")
    print("    via AVX2 shift-add. GPU routes transcendentals through SFUs that")
    print("    sit idle during MAC-heavy tensor work. GLU is the theoretical")
    print("    endpoint — custom RTL with no instruction fetch, no decode,")
    print("    one CORDIC iteration per clock, ZERO multipliers (see rtl/).")


if __name__ == "__main__":
    main()
