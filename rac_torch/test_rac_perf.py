#!/usr/bin/env python3
"""
test_rac_perf.py — Performance / throughput harness for RAC transformer ops
Pinnacle Quantum Group — April 2026

Measures:
  - RACRoPE throughput vs manual rotary
  - RACRMSNorm vs torch-native fused rsqrt
  - RACLayerNorm vs torch.nn.LayerNorm
  - RACLlamaBlock step time at several (batch, seq, d_model) shapes
  - Tunable-precision cost delta (iters 4..24)

Usage:
    python test_rac_perf.py
    python test_rac_perf.py --device cpu   # force CPU even with CUDA
"""

import argparse
import sys
import time

import torch

from rac_torch import (
    RACRoPE, RACRMSNorm, RACLayerNorm, RACLlamaBlock,
    rac_set_precision, rac_get_precision,
    rac_set_mode, rac_get_mode,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--iters", type=int, default=20)
    return p.parse_args()


def bench(fn, iters, device):
    # Warm-up
    for _ in range(5):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def safe_bench(name, fn, iters, device):
    """Run a bench; print error and return None rather than aborting the run."""
    try:
        return bench(fn, iters, device)
    except Exception as e:
        print(f"  [SKIP] {name}: {type(e).__name__}: {str(e).splitlines()[0][:90]}")
        return None


def _device_is_healthy(dev: str) -> bool:
    if dev == "cpu":
        return True
    try:
        a = torch.randn(4, 4, device=dev)
        b = torch.matmul(a, a)
        if dev == "cuda":
            torch.cuda.synchronize()
        _ = b.sum().item()
        return True
    except Exception as e:
        print(f"[WARN] {dev} device probe failed: {type(e).__name__}: "
              f"{str(e).splitlines()[0][:100]}")
        return False


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not _device_is_healthy(device):
        print(f"[WARN] falling back to CPU")
        device = "cpu"
    print(f"Device: {device}    iters/bench: {args.iters}")
    print("=" * 66)

    # ── RoPE ───────────────────────────────────────────────────────────
    print("\n── RACRoPE throughput ──")
    rope = RACRoPE(head_dim=64, max_seq_len=2048).to(device)
    for (B, H, T, D) in [(1, 8, 128, 64), (2, 16, 512, 64), (2, 32, 1024, 64)]:
        q = torch.randn(B, H, T, D, device=device)
        k = torch.randn_like(q)
        t = safe_bench(f"RoPE B={B} H={H} T={T}", lambda: rope(q, k), args.iters, device)
        if t is None: continue
        bytes_moved = 2 * q.numel() * q.element_size() * 2  # Q+K read+write
        gbps = bytes_moved / (t * 1e6)
        print(f"  B={B} H={H} T={T} D={D}:  {t:7.3f} ms   {gbps:6.2f} GB/s")

    # ── RMSNorm ────────────────────────────────────────────────────────
    print("\n── RACRMSNorm vs torch.nn.LayerNorm (proxy) ──")
    for d in [1024, 4096, 8192]:
        rms = RACRMSNorm(d).to(device)
        ref = torch.nn.LayerNorm(d).to(device)
        x = torch.randn(64, d, device=device)
        t_rac = safe_bench(f"RMS d={d}", lambda: rms(x), args.iters, device)
        t_ref = safe_bench(f"LN d={d}",  lambda: ref(x), args.iters, device)
        if t_rac is None or t_ref is None: continue
        print(f"  d={d}:  RMS={t_rac:6.3f}ms    LayerNorm={t_ref:6.3f}ms    "
              f"ratio={t_rac/t_ref:.2f}x")

    # ── LayerNorm ──────────────────────────────────────────────────────
    print("\n── RACLayerNorm vs torch.nn.LayerNorm ──")
    for d in [1024, 4096]:
        rln = RACLayerNorm(d).to(device)
        tln = torch.nn.LayerNorm(d).to(device)
        x = torch.randn(64, d, device=device)
        t_rac = safe_bench(f"RACLN d={d}", lambda: rln(x), args.iters, device)
        t_ref = safe_bench(f"torchLN d={d}", lambda: tln(x), args.iters, device)
        if t_rac is None or t_ref is None: continue
        print(f"  d={d}:  RACLN={t_rac:6.3f}ms    torchLN={t_ref:6.3f}ms    "
              f"ratio={t_rac/t_ref:.2f}x")

    # ── Llama block ───────────────────────────────────────────────────
    print("\n── RACLlamaBlock step time ──")
    for (B, T, D, H) in [(1, 128, 256, 8), (2, 512, 512, 16), (2, 1024, 1024, 16)]:
        block = RACLlamaBlock(
            d_model=D, n_heads=H, ff_dim=4 * D, max_seq_len=T
        ).to(device)
        x = torch.randn(B, T, D, device=device)
        t = safe_bench(f"Llama B={B} T={T} D={D}",
                        lambda: block(x, is_causal=True), args.iters, device)
        if t is None: continue
        params = sum(p.numel() for p in block.parameters())
        print(f"  B={B} T={T} D={D} H={H}:  {t:7.3f} ms   params={params/1e6:.2f}M")

    # ── Compute-mode A/B ──────────────────────────────────────────────
    # Three kernel paths: FAST (sign-XOR FMA, hardware multiplier),
    # CORDIC (iterative rotation port from rac_cuda.cu), SHIFTADD
    # (integer ALU only, no FP multiplier). Same matmul input, very
    # different silicon utilization.
    print("\n── Compute-mode A/B (RACLlamaBlock step, iters=16) ──")
    block = RACLlamaBlock(d_model=256, n_heads=8, ff_dim=1024, max_seq_len=128).to(device)
    x = torch.randn(2, 64, 256, device=device)
    rac_set_precision(16)
    for mode in ('fast', 'cordic', 'shiftadd'):
        rac_set_mode(mode)
        t = safe_bench(f"mode={mode}",
                        lambda: block(x, is_causal=True), args.iters, device)
        if t is None: continue
        print(f"  mode={mode:9s}  {t:7.3f} ms    (get={rac_get_mode()})")
    rac_set_mode('fast')

    # ── Tunable precision (mode=shiftadd shows the real cycle curve) ──
    print("\n── Tunable-precision sweep (RACLlamaBlock, mode=shiftadd) ──")
    rac_set_mode('shiftadd')
    for iters in [4, 8, 12, 16, 20, 24]:
        rac_set_precision(iters)
        t = safe_bench(f"iters={iters}",
                        lambda: block(x, is_causal=True), args.iters, device)
        if t is None: continue
        print(f"  iters={iters:2d}:  {t:7.3f} ms    (get={rac_get_precision()})")
    rac_set_precision(24); rac_set_mode('fast')

    print("\nperf harness done.")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"perf harness failed: {e}")
        sys.exit(1)
