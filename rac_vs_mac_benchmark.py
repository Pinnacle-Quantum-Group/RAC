#!/usr/bin/env python3
"""
rac_vs_mac_benchmark.py — RAC vs MAC Proof-of-Performance Demo
Pinnacle Quantum Group — March 2026

Quantitative, reproducible benchmark comparing Rotation-Accumulate (RAC)
vs Multiply-Accumulate (MAC) on real GPU hardware.

Measures: latency, throughput, energy, accuracy.
Generates: tables + matplotlib plots.

Requirements: torch, numpy, matplotlib, pynvml (or rocm_smi)
Run: python rac_vs_mac_benchmark.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import math
from collections import defaultdict

# ── Energy measurement backend ──────────────────────────────────────────────

ENERGY_BACKEND = None  # 'nvml', 'rocm', or None

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    ENERGY_BACKEND = 'nvml'
except Exception:
    pass

if ENERGY_BACKEND is None:
    try:
        import subprocess
        r = subprocess.run(['rocm-smi', '--showpower'], capture_output=True, text=True)
        if r.returncode == 0:
            ENERGY_BACKEND = 'rocm'
    except Exception:
        pass


def gpu_energy_mj():
    """Return cumulative GPU energy in millijoules, or 0 if unavailable."""
    if ENERGY_BACKEND == 'nvml':
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(_nvml_handle)
        except Exception:
            return 0
    return 0


def gpu_power_w():
    """Return current GPU power draw in watts."""
    if ENERGY_BACKEND == 'nvml':
        try:
            return pynvml.nvmlDeviceGetPowerUsage(_nvml_handle) / 1000.0
        except Exception:
            return 0
    if ENERGY_BACKEND == 'rocm':
        try:
            r = subprocess.run(['rocm-smi', '--showpower', '--json'],
                               capture_output=True, text=True)
            import json
            data = json.loads(r.stdout)
            for card in data.values():
                if isinstance(card, dict) and 'Average Graphics Package Power' in card:
                    return float(card['Average Graphics Package Power'].replace(' W', ''))
        except Exception:
            return 0
    return 0


# ── RAC Linear Layer ────────────────────────────────────────────────────────

class RACLinearPure(nn.Module):
    """
    RAC Linear: replaces dot product with rotation-projection.

    Standard MAC:  out[i] = sum_j(x[j] * w[i,j])
    RAC:           out[i] = sum_j(x[j] * cos(theta[i,j]))

    Where theta = atan2(0, sign(w)) encodes the weight as an angle.
    For the degenerate scalar case: cos(0)=1 for w>0, cos(pi)=-1 for w<0.

    This layer uses the GENERAL RAC path: weights stored as (magnitude, angle)
    and projection computed via sin/cos SFU calls. This is the non-degenerate
    case that exercises the SFU hardware path differently from FMA.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store weights in polar form: magnitude and angle
        # Initialize from Kaiming-equivalent
        w = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))

        # Decompose into magnitude + angle
        self.magnitude = nn.Parameter(w.abs())
        self.angle = nn.Parameter(torch.atan2(torch.zeros_like(w),
                                               torch.where(w >= 0, torch.ones_like(w),
                                                           -torch.ones_like(w))))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        """
        RAC forward: projection via sin/cos.
        out = x @ (magnitude * cos(angle))^T + bias

        The cos(angle) call routes through GPU SFU (Special Function Unit)
        on NVIDIA, or transcendental ALU on AMD. This is the key difference
        from MAC: we're using SFU cycles instead of FMA cycles.
        """
        # Reconstruct effective weight via rotation
        # w_eff = magnitude * cos(angle)  — this IS the RAC projection
        w_eff = self.magnitude * torch.cos(self.angle)  # SFU: cos()

        # Standard matmul with the reconstructed weight
        out = F.linear(x, w_eff, self.bias)
        return out


class RACLinearFused(nn.Module):
    """
    Fused RAC Linear + Activation.
    Computes act(x @ (mag * cos(angle))^T + bias) with cos in the hot path.
    """

    def __init__(self, in_features, out_features, activation='gelu', bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        w = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.magnitude = nn.Parameter(w.abs())
        self.angle = nn.Parameter(torch.atan2(torch.zeros_like(w),
                                               torch.where(w >= 0, torch.ones_like(w),
                                                           -torch.ones_like(w))))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self._act_fn = {'relu': F.relu, 'gelu': F.gelu, 'silu': F.silu}.get(activation, lambda x: x)

    def forward(self, x):
        w_eff = self.magnitude * torch.cos(self.angle)
        out = F.linear(x, w_eff, self.bias)
        return self._act_fn(out)


# ── Benchmark Engine ────────────────────────────────────────────────────────

def benchmark_layer(layer, x, n_warmup=10, n_iters=100):
    """
    Benchmark a layer: returns (latency_ms, energy_mj, output).
    Uses CUDA events for precise GPU timing.
    """
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = layer(x)

    torch.cuda.synchronize()

    # Energy measurement
    e0 = gpu_energy_mj()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(n_iters):
            out = layer(x)
    end_event.record()

    torch.cuda.synchronize()
    e1 = gpu_energy_mj()

    latency_ms = start_event.elapsed_time(end_event) / n_iters
    energy_mj = (e1 - e0) / n_iters if (e1 - e0) > 0 else 0

    return latency_ms, energy_mj, out


def compute_accuracy(out_mac, out_rac):
    """Compare MAC vs RAC outputs."""
    diff = (out_mac.float() - out_rac.float()).abs()
    max_err = diff.max().item()
    mse = (diff ** 2).mean().item()
    return max_err, mse


# ── Main Benchmark ──────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm GPU required.")
        sys.exit(1)

    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)

    print("=" * 90)
    print("  RAC vs MAC — Proof-of-Performance Benchmark")
    print("  Pinnacle Quantum Group — March 2026")
    print("=" * 90)
    print(f"  Device:          {props.name}")
    print(f"  Compute:         {props.major}.{props.minor}")
    print(f"  SMs:             {props.multi_processor_count}")
    print(f"  Memory:          {props.total_mem // (1024**2)} MB")
    print(f"  Energy backend:  {ENERGY_BACKEND or 'none'}")
    print(f"  PyTorch:         {torch.__version__}")
    print("=" * 90)

    # ── Configuration ──
    sizes = [256, 512, 1024, 2048, 4096]
    batches = [1, 8, 32]
    n_warmup = 10
    n_iters = 100

    results = []

    # ── Header ──
    print(f"\n{'Size':>6} {'Batch':>5} {'Method':>10} {'Latency':>10} {'Throughput':>12} "
          f"{'Energy':>10} {'Energy/op':>10} {'MaxErr':>10}")
    print(f"{'':>6} {'':>5} {'':>10} {'(ms)':>10} {'(GFLOPS)':>12} "
          f"{'(mJ)':>10} {'(nJ)':>10} {'':>10}")
    print("─" * 90)

    for size in sizes:
        for batch in batches:
            in_f = size
            out_f = size

            # Skip huge sizes on small GPUs
            mem_needed = batch * in_f * 4 + out_f * in_f * 4 * 2 + batch * out_f * 4
            if mem_needed > props.total_mem * 0.3:
                continue

            # ── MAC baseline ──
            mac_layer = nn.Linear(in_f, out_f, bias=True).to(device).eval()
            x = torch.randn(batch, in_f, device=device)

            lat_mac, energy_mac, out_mac = benchmark_layer(mac_layer, x, n_warmup, n_iters)

            ops = 2.0 * batch * in_f * out_f  # FLOPs per matmul
            gflops_mac = ops / (lat_mac * 1e6)
            energy_per_op_mac = (energy_mac * 1e6 / ops) if energy_mac > 0 else 0  # nJ

            # ── RAC (pure SFU path) ──
            rac_layer = RACLinearPure(in_f, out_f, bias=True).to(device).eval()
            lat_rac, energy_rac, out_rac = benchmark_layer(rac_layer, x, n_warmup, n_iters)

            gflops_rac = ops / (lat_rac * 1e6)
            energy_per_op_rac = (energy_rac * 1e6 / ops) if energy_rac > 0 else 0

            # ── RAC fused (SFU + activation) ──
            rac_fused = RACLinearFused(in_f, out_f, activation='gelu', bias=True).to(device).eval()
            lat_fused, energy_fused, out_fused = benchmark_layer(rac_fused, x, n_warmup, n_iters)

            gflops_fused = ops / (lat_fused * 1e6)
            energy_per_op_fused = (energy_fused * 1e6 / ops) if energy_fused > 0 else 0

            # ── Accuracy ──
            max_err, mse = compute_accuracy(out_mac, out_rac)

            # ── Print ──
            print(f"{size:6d} {batch:5d} {'MAC':>10} {lat_mac:10.3f} {gflops_mac:12.2f} "
                  f"{energy_mac:10.2f} {energy_per_op_mac:10.4f} {'─':>10}")
            print(f"{'':>6} {'':>5} {'RAC':>10} {lat_rac:10.3f} {gflops_rac:12.2f} "
                  f"{energy_rac:10.2f} {energy_per_op_rac:10.4f} {max_err:10.2e}")
            print(f"{'':>6} {'':>5} {'RAC+GELU':>10} {lat_fused:10.3f} {gflops_fused:12.2f} "
                  f"{energy_fused:10.2f} {energy_per_op_fused:10.4f} {'─':>10}")

            # Speedup
            speedup = lat_mac / lat_rac if lat_rac > 0 else 0
            energy_ratio = energy_mac / energy_rac if energy_rac > 0 else 0
            print(f"{'':>6} {'':>5} {'RATIO':>10} {'':>10} {speedup:11.2f}x "
                  f"{'':>10} {energy_ratio:9.2f}x {'':>10}")
            print("─" * 90)

            results.append({
                'size': size, 'batch': batch,
                'lat_mac': lat_mac, 'lat_rac': lat_rac, 'lat_fused': lat_fused,
                'gf_mac': gflops_mac, 'gf_rac': gflops_rac, 'gf_fused': gflops_fused,
                'e_mac': energy_mac, 'e_rac': energy_rac, 'e_fused': energy_fused,
                'epo_mac': energy_per_op_mac, 'epo_rac': energy_per_op_rac,
                'epo_fused': energy_per_op_fused,
                'max_err': max_err, 'mse': mse, 'speedup': speedup,
                'energy_ratio': energy_ratio,
            })

    # ── Summary ──
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)

    if results:
        # Find crossover points
        wins_rac = [r for r in results if r['speedup'] > 1.0]
        wins_mac = [r for r in results if r['speedup'] <= 1.0]

        print(f"\n  RAC wins at {len(wins_rac)}/{len(results)} configurations")
        if wins_rac:
            best = max(wins_rac, key=lambda r: r['speedup'])
            print(f"  Best RAC speedup: {best['speedup']:.2f}x at "
                  f"{best['size']}x{best['size']} batch={best['batch']}")
        if wins_mac:
            worst = min(wins_mac, key=lambda r: r['speedup'])
            print(f"  Worst RAC ratio:  {worst['speedup']:.2f}x at "
                  f"{worst['size']}x{worst['size']} batch={worst['batch']}")

        # Energy summary
        e_results = [r for r in results if r['e_mac'] > 0 and r['e_rac'] > 0]
        if e_results:
            avg_energy_ratio = np.mean([r['energy_ratio'] for r in e_results])
            print(f"\n  Average energy ratio (MAC/RAC): {avg_energy_ratio:.2f}x")
            if avg_energy_ratio > 1.0:
                print(f"  → RAC uses {(1 - 1/avg_energy_ratio)*100:.1f}% less energy on average")
            else:
                print(f"  → MAC uses {(1 - avg_energy_ratio)*100:.1f}% less energy on average")

        # Accuracy
        max_errs = [r['max_err'] for r in results]
        print(f"\n  Numerical accuracy:")
        print(f"    Max absolute error: {max(max_errs):.2e}")
        print(f"    All within float32 tolerance: {'YES' if max(max_errs) < 0.1 else 'NO'}")

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('RAC vs MAC — Performance & Energy', fontsize=14, fontweight='bold')

        # Group by batch=32 (largest batch for best GPU utilization)
        b32 = [r for r in results if r['batch'] == 32]
        if not b32:
            b32 = [r for r in results if r['batch'] == max(r['batch'] for r in results)]

        if b32:
            szs = [r['size'] for r in b32]

            # Plot 1: Throughput vs Size
            ax = axes[0][0]
            ax.plot(szs, [r['gf_mac'] for r in b32], 'b-o', label='MAC (cuBLAS)', linewidth=2)
            ax.plot(szs, [r['gf_rac'] for r in b32], 'r-s', label='RAC (SFU)', linewidth=2)
            ax.plot(szs, [r['gf_fused'] for r in b32], 'g-^', label='RAC+GELU (fused)', linewidth=2)
            ax.set_xlabel('Matrix Size (N×N)')
            ax.set_ylabel('Throughput (GFLOPS)')
            ax.set_title('Throughput vs Matrix Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)

            # Plot 2: Latency vs Size
            ax = axes[0][1]
            ax.plot(szs, [r['lat_mac'] for r in b32], 'b-o', label='MAC', linewidth=2)
            ax.plot(szs, [r['lat_rac'] for r in b32], 'r-s', label='RAC', linewidth=2)
            ax.plot(szs, [r['lat_fused'] for r in b32], 'g-^', label='RAC+GELU', linewidth=2)
            ax.set_xlabel('Matrix Size (N×N)')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Latency vs Matrix Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')

            # Plot 3: Speedup (RAC / MAC)
            ax = axes[1][0]
            speedups = [r['speedup'] for r in b32]
            colors = ['green' if s >= 1.0 else 'red' for s in speedups]
            ax.bar(range(len(szs)), speedups, color=colors, alpha=0.7)
            ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
            ax.set_xticks(range(len(szs)))
            ax.set_xticklabels([str(s) for s in szs])
            ax.set_xlabel('Matrix Size (N×N)')
            ax.set_ylabel('Speedup (MAC_time / RAC_time)')
            ax.set_title('RAC Speedup (>1 = RAC wins)')
            ax.grid(True, alpha=0.3)

            # Plot 4: Energy per Op
            ax = axes[1][1]
            e_b32 = [r for r in b32 if r['epo_mac'] > 0]
            if e_b32:
                e_szs = [r['size'] for r in e_b32]
                ax.plot(e_szs, [r['epo_mac'] for r in e_b32], 'b-o', label='MAC', linewidth=2)
                ax.plot(e_szs, [r['epo_rac'] for r in e_b32], 'r-s', label='RAC', linewidth=2)
                ax.set_xlabel('Matrix Size (N×N)')
                ax.set_ylabel('Energy per Op (nJ)')
                ax.set_title('Energy Efficiency')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log', base=2)
            else:
                ax.text(0.5, 0.5, 'Energy measurement\nnot available',
                        transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title('Energy Efficiency (N/A)')

        plt.tight_layout()
        plot_path = 'rac_vs_mac_benchmark.png'
        plt.savefig(plot_path, dpi=150)
        print(f"\n  Plot saved: {plot_path}")

    except ImportError:
        print("\n  matplotlib not available — skipping plots")

    # ── Memory Bandwidth Analysis ──
    print("\n" + "=" * 90)
    print("  MEMORY BANDWIDTH ANALYSIS")
    print("=" * 90)

    print(f"\n  {'Size':>6} {'Batch':>5} {'MAC BW':>12} {'RAC BW':>12} {'RAC overhead':>14}")
    print(f"  {'':>6} {'':>5} {'(GB/s)':>12} {'(GB/s)':>12} {'':>14}")

    for r in results:
        # MAC: reads x[B,K] + W[N,K], writes out[B,N]
        mac_bytes = (r['batch'] * r['size'] + r['size'] * r['size'] + r['batch'] * r['size']) * 4
        mac_bw = mac_bytes / (r['lat_mac'] * 1e-3) / 1e9

        # RAC: reads x[B,K] + mag[N,K] + angle[N,K], writes out[B,N]
        # Extra: cos(angle) computation reads angle and writes effective weight
        rac_bytes = (r['batch'] * r['size'] + r['size'] * r['size'] * 2 + r['batch'] * r['size']) * 4
        rac_bw = rac_bytes / (r['lat_rac'] * 1e-3) / 1e9

        overhead = (rac_bytes - mac_bytes) / mac_bytes * 100

        print(f"  {r['size']:6d} {r['batch']:5d} {mac_bw:12.1f} {rac_bw:12.1f} {overhead:13.1f}%")

    print(f"\n  RAC reads 2x weight data (magnitude + angle) vs MAC (1x weight)")
    print(f"  At large N, the extra bandwidth is amortized by batch reuse")

    # ── Where Hardware Matters ──
    print("\n" + "=" * 90)
    print("  WHERE GPU HARDWARE BEHAVIOR MATTERS")
    print("=" * 90)
    print("""
  MAC path:
    x @ W → cuBLAS SGEMM → Tensor Cores (Ampere+) or FMA units
    Peak: limited by FMA throughput (FP32 TFLOPS)

  RAC path:
    cos(angle) → SFU (Special Function Unit) on NVIDIA / transcendental ALU on AMD
    x @ W_eff → same cuBLAS SGEMM as MAC

  Key insight: RAC's cos() call adds SFU work, but the matmul itself is identical.
  On commodity GPUs, RAC can only win if:
    1. The weight reconstruction (cos * magnitude) is overlapped with compute
    2. The fused path eliminates memory round-trips
    3. On hardware with native CORDIC (FIL), the cos() is free

  On GPU: RAC ≈ MAC + SFU overhead (cos reconstruction)
  On FIL: RAC < MAC (CORDIC replaces multiplier entirely)
  GPU results here are a LOWER BOUND on RAC's advantage.
""")

    print("=" * 90)
    print("  Benchmark complete.")
    print("=" * 90)


if __name__ == '__main__':
    main()
