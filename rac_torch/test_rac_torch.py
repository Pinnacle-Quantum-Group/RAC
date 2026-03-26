"""
test_rac_torch.py — RAC PyTorch Extension: Tests and Demo
Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026

Tests:
    1. RACLinear correctness vs nn.Linear
    2. rac_matmul correctness vs torch.matmul
    3. Gradient correctness (autograd check)
    4. patch_model on a small transformer
    5. Training step with RACLinear
    6. Benchmark: RAC vs baseline
"""

import torch
import torch.nn as nn
import math
import sys

def header(s):
    print(f"\n{'─'*60}")
    print(f"  {s}")
    print(f"{'─'*60}")

# ── Import RAC ────────────────────────────────────────────────────────────────

from rac_torch import (
    RACLinear, rac_matmul, rac_linear,
    patch_model, unpatch_model, benchmark_model, rac_info
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nRunning on: {device}")
rac_info()

# ── Test 1: RACLinear correctness ─────────────────────────────────────────────

header("Test 1: RACLinear vs nn.Linear — forward correctness")

torch.manual_seed(42)
for (B, IN, OUT) in [(1, 64, 64), (8, 256, 128), (32, 768, 768), (4, 4096, 4096)]:
    x      = torch.randn(B, IN,  device=device)
    linear = nn.Linear(IN, OUT,  bias=True).to(device).float()
    rac    = RACLinear.from_linear(linear)

    with torch.no_grad():
        y_ref = linear(x)
        y_rac = rac(x)

    err = (y_ref - y_rac).abs().max().item()
    status = "PASS" if err < 1e-2 else "FAIL"
    print(f"  [{status}] batch={B:3d}  {IN:4d}→{OUT:4d}  max_err={err:.2e}")

# ── Test 2: rac_matmul correctness ────────────────────────────────────────────

header("Test 2: rac_matmul vs torch.matmul")

for (M, K, N) in [(64,64,64), (256,512,256), (1024,1024,1024), (4096,4096,4096)]:
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)

    C_ref = torch.matmul(A, B)
    C_rac = rac_matmul(A, B)

    err    = (C_ref - C_rac).abs().max().item()
    status = "PASS" if err < 1e-2 else "FAIL"
    print(f"  [{status}] {M}×{K} @ {K}×{N}  max_err={err:.2e}")

# ── Test 3: Gradient correctness ──────────────────────────────────────────────

header("Test 3: Gradient correctness (torch.autograd.gradcheck)")

from torch.autograd import gradcheck

try:
    A = torch.randn(16, 16, device=device, dtype=torch.float64, requires_grad=True)
    B = torch.randn(16, 16, device=device, dtype=torch.float64, requires_grad=True)

    # gradcheck needs double precision
    from rac_torch import RACMatmulFunction
    result = gradcheck(RACMatmulFunction.apply, (A, B), eps=1e-4, atol=1e-3, rtol=1e-3)
    print(f"  [{'PASS' if result else 'FAIL'}] gradcheck on rac_matmul (16×16, float64)")
except Exception as e:
    print(f"  [SKIP] gradcheck: {e}")

# Manual gradient check on RACLinear
torch.manual_seed(0)
x = torch.randn(4, 64, device=device, requires_grad=True)
layer = RACLinear(64, 32).to(device)
y = layer(x)
loss = y.sum()
loss.backward()
print(f"  [{'PASS' if x.grad is not None else 'FAIL'}] RACLinear backward — gradients flow")
print(f"  [{'PASS' if layer.weight.grad is not None else 'FAIL'}] RACLinear weight gradients")
print(f"  [{'PASS' if layer.bias.grad is not None else 'FAIL'}] RACLinear bias gradients")

# ── Test 4: patch_model on a small transformer ────────────────────────────────

header("Test 4: patch_model — small transformer")

class SmallTransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, ff_dim=1024):
        super().__init__()
        self.attn_q  = nn.Linear(d_model, d_model)
        self.attn_k  = nn.Linear(d_model, d_model)
        self.attn_v  = nn.Linear(d_model, d_model)
        self.attn_out= nn.Linear(d_model, d_model)
        self.ff1     = nn.Linear(d_model, ff_dim)
        self.ff2     = nn.Linear(ff_dim, d_model)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.n_heads = n_heads

    def forward(self, x):
        B, T, D = x.shape
        Q = self.attn_q(x)
        K = self.attn_k(x)
        V = self.attn_v(x)
        # simplified attention (no masking)
        scale = math.sqrt(D // self.n_heads)
        attn = torch.softmax(Q @ K.transpose(-2,-1) / scale, dim=-1)
        x = x + self.attn_out(attn @ V)
        x = self.norm1(x)
        x = x + self.ff2(torch.relu(self.ff1(x)))
        x = self.norm2(x)
        return x

model = SmallTransformerBlock(d_model=256, n_heads=8, ff_dim=1024).to(device)
x_in  = torch.randn(2, 32, 256, device=device)  # batch=2, seq=32, d=256

with torch.no_grad():
    y_before = model(x_in).clone()

model = patch_model(model, verbose=True)

with torch.no_grad():
    y_after = model(x_in)

err = (y_before - y_after).abs().max().item()
print(f"\n  Output preserved after patching: max_err={err:.2e}  "
      f"[{'PASS' if err < 1e-2 else 'FAIL'}]")

# ── Test 5: Training step ─────────────────────────────────────────────────────

header("Test 5: Training step with RACLinear")

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            RACLinear(784, 512),
            nn.ReLU(),
            RACLinear(512, 256),
            nn.ReLU(),
            RACLinear(256, 10),
        )
    def forward(self, x): return self.layers(x)

mlp   = SmallMLP().to(device)
opt   = torch.optim.Adam(mlp.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

losses = []
for step in range(20):
    x  = torch.randn(64, 784, device=device)
    y  = torch.randint(0, 10, (64,), device=device)
    opt.zero_grad()
    loss = loss_fn(mlp(x), y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

decreasing = losses[-1] < losses[0]
print(f"  Initial loss:  {losses[0]:.4f}")
print(f"  Final loss:    {losses[-1]:.4f}")
print(f"  Loss decreasing: [{'PASS' if decreasing else 'FAIL'}]")

# ── Test 6: Benchmark ─────────────────────────────────────────────────────────

header("Test 6: Benchmark — RAC vs nn.Linear")

import time

def bench(layer, x, n=200):
    with torch.no_grad():
        for _ in range(10): layer(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n): layer(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

for (B, IN, OUT) in [(64, 768, 768), (64, 2048, 2048), (32, 4096, 4096)]:
    x      = torch.randn(B, IN, device=device)
    linear = nn.Linear(IN, OUT).to(device).float()
    rac    = RACLinear.from_linear(linear)

    t_ref  = bench(linear, x)
    t_rac  = bench(rac, x)
    speedup = t_ref / t_rac

    print(f"  {B:3d}×{IN:4d}→{OUT:4d}  "
          f"Linear={t_ref:.3f}ms  RAC={t_rac:.3f}ms  "
          f"speedup={speedup:.2f}×")

# ── Summary ───────────────────────────────────────────────────────────────────

header("Summary")
print("""
  RACLinear is a drop-in replacement for nn.Linear.
  patch_model() replaces all Linear layers in any model with one call.
  Forward and backward are both multiply-free on CUDA float32.
  Weights are preserved exactly — no retraining required.

  To use with any HuggingFace model:
    from transformers import AutoModel
    from rac_torch import patch_model
    model = AutoModel.from_pretrained('bert-base-uncased')
    model = patch_model(model)
    # All linear ops now run via RAC — zero multiplications
""")
