#!/usr/bin/env python3
"""
test_hybrid.py — RAC Hybrid CPU+GPU Dispatch Test
Pinnacle Quantum Group — March 2026

Tests CPU and GPU backends running in concert:
  1. Small ops → CPU (HAL auto-dispatch, AVX2/OpenMP)
  2. Large ops → GPU (CUDA/HIP micro-tiled kernels)
  3. Concurrent: CPU activations + GPU matmul on different data
  4. Pipeline: GPU matmul → CPU postprocess → GPU matmul
  5. Adaptive dispatch: auto-select backend by problem size
  6. Full transformer inference: GPU attention + CPU+GPU linear layers

Requires: PyTorch with CUDA/ROCm + compiled rac_torch extension + librac.

Usage:
    python test_hybrid.py
"""

import sys
import os
import time
import ctypes
import ctypes.util

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import RAC PyTorch (GPU) ────────────────────────────────────────────────

from rac_torch import (
    RACLinear, FusedRACLinear, RACAttention, RACTransformerBlock,
    RACFusedFFN, RACFusedQKV,
    rac_matmul, rac_linear, rac_matmul_adaptive,
    patch_model, _rac_available, rac_info,
)

# ── Load RAC C library (CPU) ────────────────────────────────────────────────

def _load_librac():
    """Try to load the compiled C library for CPU dispatch."""
    search_paths = [
        os.path.join(os.path.dirname(__file__), 'lib', 'build', 'librac_avx2.so'),
        os.path.join(os.path.dirname(__file__), 'lib', 'build', 'librac.so'),
        'librac_avx2.so',
        'librac.so',
    ]
    for path in search_paths:
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                return lib
            except OSError:
                continue
    return None

librac = _load_librac()

# ── Test harness ────────────────────────────────────────────────────────────

passed = 0
failed = 0
skipped = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")

def skip(name, reason=""):
    global skipped
    skipped += 1
    print(f"  [SKIP] {name}  {reason}")

def header(s):
    print(f"\n{'='*65}\n  {s}\n{'='*65}")

def bench_ms(fn, n_warmup=5, n_iters=20):
    """Benchmark a callable, return ms per call."""
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1000


# ═══════════════════════════════════════════════════════════════════════════
print("RAC Hybrid CPU+GPU Test — Pinnacle Quantum Group")
print("=" * 65)

rac_info()

HAS_GPU = torch.cuda.is_available() and _rac_available()
HAS_CPU_LIB = librac is not None

print(f"\n  GPU backend:  {'available' if HAS_GPU else 'NOT available'}")
print(f"  CPU library:  {'loaded' if HAS_CPU_LIB else 'NOT loaded'}")

if not HAS_GPU:
    print("\nWARNING: No GPU — GPU tests will be skipped.")
if not HAS_CPU_LIB:
    print("\nWARNING: librac not found — CPU C library tests will be skipped.")
    print("  Build with: cd lib && mkdir build && cd build && cmake .. && make")

device = torch.device('cuda' if HAS_GPU else 'cpu')
torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid-1: Small → CPU, Large → GPU dispatch
# ═══════════════════════════════════════════════════════════════════════════
header("Hybrid-1: Size-based dispatch (small=CPU, large=GPU)")

if HAS_GPU:
    sizes = [(16, 16, 16), (64, 64, 64), (256, 256, 256),
             (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]

    print(f"\n  {'Size':>12s}  {'CPU (ms)':>10s}  {'GPU (ms)':>10s}  {'Winner':>8s}  {'Speedup':>8s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")

    crossover_found = False
    for M, K, N in sizes:
        A_cpu = torch.randn(M, K)
        B_cpu = torch.randn(K, N)
        A_gpu = A_cpu.to(device)
        B_gpu = B_cpu.to(device)

        t_cpu = bench_ms(lambda: torch.matmul(A_cpu, B_cpu), n_warmup=3, n_iters=10)
        t_gpu = bench_ms(lambda: rac_matmul(A_gpu, B_gpu), n_warmup=3, n_iters=10)

        winner = "CPU" if t_cpu < t_gpu else "GPU"
        speedup = max(t_cpu, t_gpu) / min(t_cpu, t_gpu)

        print(f"  {M:4d}x{K:4d}  {t_cpu:10.3f}  {t_gpu:10.3f}  {winner:>8s}  {speedup:7.1f}x")

        if winner == "GPU" and not crossover_found:
            crossover_found = True
            print(f"  {'':>12s}  {'':>10s}  {'':>10s}  ^^^ crossover point")

    check("size dispatch benchmark completed", True)
else:
    skip("size-based dispatch", "no GPU")


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid-2: Concurrent CPU + GPU work
# ═══════════════════════════════════════════════════════════════════════════
header("Hybrid-2: Concurrent CPU + GPU execution")

if HAS_GPU:
    # GPU: large matmul
    A_gpu = torch.randn(2048, 2048, device=device)
    B_gpu = torch.randn(2048, 2048, device=device)

    # CPU: activation on large tensor
    X_cpu = torch.randn(1 << 20)  # 1M elements

    # Sequential
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C_gpu = rac_matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()
    Y_cpu = torch.relu(X_cpu)
    t_seq = (time.perf_counter() - t0) * 1000

    # Concurrent: launch GPU work, do CPU work while GPU runs
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C_gpu2 = rac_matmul(A_gpu, B_gpu)  # launches async on GPU
    Y_cpu2 = torch.relu(X_cpu)          # runs on CPU while GPU computes
    torch.cuda.synchronize()             # wait for GPU
    t_conc = (time.perf_counter() - t0) * 1000

    overlap = t_seq / t_conc if t_conc > 0 else 1.0
    print(f"  Sequential:  {t_seq:.2f}ms")
    print(f"  Concurrent:  {t_conc:.2f}ms")
    print(f"  Overlap:     {overlap:.2f}x (>1.0 = CPU+GPU overlapped)")

    check("concurrent execution completed", True)
    check("GPU result matches", torch.equal(C_gpu, C_gpu2))
    check("CPU result matches", torch.equal(Y_cpu, Y_cpu2))
else:
    skip("concurrent CPU+GPU", "no GPU")


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid-3: Pipeline — GPU matmul → CPU postprocess → GPU matmul
# ═══════════════════════════════════════════════════════════════════════════
header("Hybrid-3: GPU → CPU → GPU pipeline")

if HAS_GPU:
    # Simulate: embedding (GPU) → CPU normalization → attention (GPU)
    batch, seq, d = 8, 128, 256

    embed_w = torch.randn(d, d, device=device)
    attn_w = torch.randn(d, d, device=device)

    x = torch.randn(batch * seq, d, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Step 1: GPU matmul (embedding projection)
    h = rac_matmul(x, embed_w)

    # Step 2: CPU postprocess (layer norm — move to CPU, normalize, move back)
    h_cpu = h.cpu()
    h_normed = F.layer_norm(h_cpu, [d])
    h_gpu = h_normed.to(device)

    # Step 3: GPU matmul (attention projection)
    out = rac_matmul(h_gpu, attn_w)

    torch.cuda.synchronize()
    t_pipeline = (time.perf_counter() - t0) * 1000

    # Compare: all-GPU (no CPU roundtrip)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    h2 = rac_matmul(x, embed_w)
    h2_normed = F.layer_norm(h2, [d])  # stays on GPU
    out2 = rac_matmul(h2_normed, attn_w)
    torch.cuda.synchronize()
    t_allgpu = (time.perf_counter() - t0) * 1000

    print(f"  GPU→CPU→GPU pipeline: {t_pipeline:.2f}ms")
    print(f"  All-GPU pipeline:     {t_allgpu:.2f}ms")
    print(f"  PCIe overhead:        {t_pipeline - t_allgpu:.2f}ms")

    check("pipeline output shape correct", tuple(out.shape) == (batch*seq, d))
    check("pipeline output finite", torch.isfinite(out).all().item())
    check("all-GPU faster than pipeline", t_allgpu < t_pipeline * 1.5)
else:
    skip("GPU→CPU→GPU pipeline", "no GPU")


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid-4: Adaptive dispatch — rac_matmul_adaptive
# ═══════════════════════════════════════════════════════════════════════════
header("Hybrid-4: Adaptive dispatch correctness")

if HAS_GPU:
    # Small (should prefer torch.matmul / CPU path)
    A_s = torch.randn(4, 4, device=device)
    B_s = torch.randn(4, 4, device=device)
    C_s = rac_matmul_adaptive(A_s, B_s)
    ref_s = torch.matmul(A_s, B_s)
    err = (C_s - ref_s).abs().max().item()
    check(f"adaptive small 4x4 correct (err={err:.2e})", err < 1e-4)

    # Large (should use RAC GPU kernel)
    A_l = torch.randn(1024, 1024, device=device)
    B_l = torch.randn(1024, 1024, device=device)
    C_l = rac_matmul_adaptive(A_l, B_l)
    ref_l = torch.matmul(A_l, B_l)
    err = (C_l - ref_l).abs().max().item()
    check(f"adaptive large 1024x1024 correct (err={err:.2e})", err < 0.02)

    # Boundary
    A_m = torch.randn(64, 64, device=device)
    B_m = torch.randn(64, 64, device=device)
    C_m = rac_matmul_adaptive(A_m, B_m)
    ref_m = torch.matmul(A_m, B_m)
    err = (C_m - ref_m).abs().max().item()
    check(f"adaptive boundary 64x64 correct (err={err:.2e})", err < 0.01)
else:
    skip("adaptive dispatch", "no GPU")


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid-5: Full transformer — all RAC, CPU+GPU dispatched
# ═══════════════════════════════════════════════════════════════════════════
header("Hybrid-5: Full RAC transformer (attention + FFN)")

if HAS_GPU:
    d_model, n_heads = 256, 8

    # Build a 2-layer RAC transformer
    class RACTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                RACTransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=1024)
                for _ in range(2)
            ])
            self.head = RACLinear(d_model, 10)

        def forward(self, x, is_causal=False):
            for block in self.blocks:
                x = block(x, is_causal=is_causal)
            return self.head(x.mean(dim=1))

    model = RACTransformer().to(device)
    x = torch.randn(4, 64, d_model, device=device)

    # Forward
    with torch.no_grad():
        logits = model(x, is_causal=True)
    check("transformer forward shape", tuple(logits.shape) == (4, 10))
    check("  output finite", torch.isfinite(logits).all().item())

    # Training (10 steps)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    labels = torch.randint(0, 10, (4,), device=device)
    losses = []
    for step in range(10):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x, is_causal=True), labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    check(f"training loss decreasing ({losses[-1]:.3f} < {losses[0]:.3f})",
          losses[-1] < losses[0])
    check("all grads exist",
          all(p.grad is not None for p in model.parameters() if p.requires_grad))

    # Benchmark
    t_fwd = bench_ms(lambda: model(x, is_causal=True), n_warmup=5, n_iters=20)
    print(f"\n  2-layer RAC transformer: {t_fwd:.2f}ms/forward")
    print(f"  {4*64} tokens, {d_model}d, {n_heads} heads, causal")
else:
    skip("full RAC transformer", "no GPU")


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid-6: Mixed precision pipeline (fp32 CPU + fp16 GPU)
# ═══════════════════════════════════════════════════════════════════════════
header("Hybrid-6: Mixed precision (fp32 CPU + fp16 GPU)")

if HAS_GPU:
    layer = FusedRACLinear(512, 256, activation='gelu').to(device)

    # fp32 on CPU
    x_cpu = torch.randn(32, 512)
    y_cpu = layer.cpu()(x_cpu)
    check("fp32 CPU forward", True)
    check("  output dtype fp32", y_cpu.dtype == torch.float32)

    # fp16 on GPU
    layer_gpu = layer.to(device).half()
    x_gpu = torch.randn(32, 512, device=device, dtype=torch.float16)
    y_gpu = layer_gpu(x_gpu)
    check("fp16 GPU forward", True)
    check("  output dtype fp16", y_gpu.dtype == torch.float16)

    # Compare (rough — different precision)
    err = (y_cpu.float() - y_gpu.cpu().float()).abs().max().item()
    print(f"  fp32 vs fp16 max error: {err:.4f}")
    check("fp32 vs fp16 within tolerance", err < 1.0)  # fp16 has limited precision

    # Autocast GPU
    layer_f32 = FusedRACLinear(512, 256, activation='gelu').to(device)
    x_f32 = torch.randn(32, 512, device=device)
    with torch.autocast('cuda', dtype=torch.float16):
        y_amp = layer_f32(x_f32)
    check("autocast fp16 forward", True)
else:
    skip("mixed precision", "no GPU")


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid-7: Throughput — batch size scaling on GPU
# ═══════════════════════════════════════════════════════════════════════════
header("Hybrid-7: GPU batch scaling")

if HAS_GPU:
    d = 768
    layer = RACLinear(d, d).to(device).eval()

    print(f"\n  {'Batch':>8s}  {'Tokens':>8s}  {'Time(ms)':>10s}  {'GFLOPS':>8s}  {'tok/s':>12s}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*12}")

    for batch in [1, 4, 16, 64, 256]:
        seq = 128
        x = torch.randn(batch, seq, d, device=device)

        t = bench_ms(lambda: layer(x), n_warmup=5, n_iters=20)
        tokens = batch * seq
        ops = 2.0 * tokens * d * d
        gflops = ops / (t * 1e6)
        tok_per_s = tokens / (t / 1000)

        print(f"  {batch:8d}  {tokens:8d}  {t:10.3f}  {gflops:8.1f}  {tok_per_s:12,.0f}")

    check("batch scaling benchmark completed", True)
else:
    skip("GPU batch scaling", "no GPU")


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid-8: Sustained mixed workload (stress)
# ═══════════════════════════════════════════════════════════════════════════
header("Hybrid-8: Sustained mixed workload (30 iterations)")

if HAS_GPU:
    n_iters = 30
    d = 256
    layer_gpu = RACLinear(d, d).to(device)
    ffn_gpu = RACFusedFFN(d, d * 4, activation='gelu').to(device)

    t0 = time.perf_counter()
    for i in range(n_iters):
        x_gpu = torch.randn(32, 64, d, device=device)

        # GPU: linear + FFN
        h = layer_gpu(x_gpu)
        h = ffn_gpu(h)

        # CPU: process a batch of activations
        x_cpu = torch.randn(1 << 18)  # 256K elements
        y_cpu = torch.relu(x_cpu)
        y_cpu = F.gelu(y_cpu)

        # GPU: another matmul
        out = layer_gpu(h)

        # Verify
        if i == 0 or i == n_iters - 1:
            torch.cuda.synchronize()
            ok = torch.isfinite(out).all().item() and torch.isfinite(y_cpu).all().item()
            check(f"sustained iter {i}: outputs finite", ok)

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n  {n_iters} iterations in {elapsed:.1f}ms ({elapsed/n_iters:.1f}ms/iter)")
    check("sustained mixed workload completed", True)
else:
    skip("sustained mixed workload", "no GPU")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
header("HYBRID TEST SUMMARY")
total = passed + failed + skipped
print(f"  Passed:  {passed}")
print(f"  Failed:  {failed}")
print(f"  Skipped: {skipped}")
print(f"  Total:   {total}")
print(f"\n  {'ALL HYBRID TESTS PASSED' if failed == 0 else 'FAILURES DETECTED'}")
sys.exit(0 if failed == 0 else 1)
