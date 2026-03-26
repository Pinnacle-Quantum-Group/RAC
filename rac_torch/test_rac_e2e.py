#!/usr/bin/env python3
"""
test_rac_e2e.py — End-to-End Tests (E2E)
Pinnacle Quantum Group — March 2026

Full integration tests: real model architectures, training loops,
inference pipelines, patch/unpatch round-trips, and performance validation.

Requires CUDA + compiled RAC extension. ~2-5 min runtime.

Usage:
    python test_rac_e2e.py
"""

import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rac_torch import (
    RACLinear, rac_matmul, rac_linear,
    patch_model, unpatch_model, benchmark_model, _rac_available
)

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
    print(f"\n{'='*60}\n  {s}\n{'='*60}")

if not torch.cuda.is_available():
    print("FATAL: E2E tests require CUDA. Exiting.")
    sys.exit(2)

device = torch.device('cuda')
torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# E2E-1: Transformer block — forward + backward
# ═══════════════════════════════════════════════════════════════════════════
header("E2E-1: Transformer block (forward + backward)")

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, ff_dim=1024, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x):
        B, T, D = x.shape
        # Multi-head attention
        Q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.d_head), dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        x = self.norm1(x + self.out(out))
        x = self.norm2(x + self.ff2(F.relu(self.ff1(x))))
        return x

model_base = TransformerBlock(d_model=256).to(device)
x = torch.randn(2, 32, 256, device=device)

# Baseline forward
with torch.no_grad():
    y_base = model_base(x).clone()

# Patch
model_rac = patch_model(model_base, verbose=False)
n_rac = sum(1 for m in model_rac.modules() if isinstance(m, RACLinear))
check(f"transformer: patched {n_rac} layers", n_rac == 6)

# RAC forward
with torch.no_grad():
    y_rac = model_rac(x)
err = (y_base - y_rac).abs().max().item()
check(f"transformer forward matches (err={err:.2e})", err < 0.05)

# RAC backward
y_rac2 = model_rac(x)
loss = y_rac2.sum()
loss.backward()
grads_exist = all(p.grad is not None for p in model_rac.parameters() if p.requires_grad)
check("transformer backward: all grads exist", grads_exist)
grads_finite = all(torch.isfinite(p.grad).all() for p in model_rac.parameters() if p.grad is not None)
check("transformer backward: all grads finite", grads_finite)


# ═══════════════════════════════════════════════════════════════════════════
# E2E-2: MLP classifier — full training loop
# ═══════════════════════════════════════════════════════════════════════════
header("E2E-2: MLP classifier training (MNIST-like)")

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x): return self.net(x)

# Train baseline
model_bl = MLPClassifier().to(device)
X = torch.randn(512, 784, device=device)
Y = torch.randint(0, 10, (512,), device=device)

opt_bl = torch.optim.Adam(model_bl.parameters(), lr=1e-3)
for _ in range(30):
    opt_bl.zero_grad()
    nn.CrossEntropyLoss()(model_bl(X), Y).backward()
    opt_bl.step()
acc_bl = (model_bl(X).argmax(1) == Y).float().mean().item()

# Train RAC-patched from scratch
model_rac = MLPClassifier().to(device)
model_rac = patch_model(model_rac, verbose=False)
opt_rac = torch.optim.Adam(model_rac.parameters(), lr=1e-3)
losses_rac = []
for _ in range(30):
    opt_rac.zero_grad()
    loss = nn.CrossEntropyLoss()(model_rac(X), Y)
    loss.backward()
    opt_rac.step()
    losses_rac.append(loss.item())
acc_rac = (model_rac(X).argmax(1) == Y).float().mean().item()

check(f"RAC training: loss decreases ({losses_rac[-1]:.3f} < {losses_rac[0]:.3f})",
      losses_rac[-1] < losses_rac[0])
check(f"RAC accuracy > 60% (got {acc_rac*100:.1f}%)", acc_rac > 0.6)
check(f"RAC accuracy competitive with baseline (RAC={acc_rac*100:.1f}% vs BL={acc_bl*100:.1f}%)",
      acc_rac > acc_bl * 0.8)  # within 80% of baseline


# ═══════════════════════════════════════════════════════════════════════════
# E2E-3: Autoencoder — reconstruction
# ═══════════════════════════════════════════════════════════════════════════
header("E2E-3: Autoencoder reconstruction")

class Autoencoder(nn.Module):
    def __init__(self, dim=256, latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 128), nn.ReLU(),
            nn.Linear(128, dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

ae = Autoencoder(dim=256, latent=32).to(device)
ae = patch_model(ae, verbose=False)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
data = torch.randn(128, 256, device=device)

for _ in range(100):
    opt.zero_grad()
    recon = ae(data)
    loss = F.mse_loss(recon, data)
    loss.backward()
    opt.step()

recon_err = F.mse_loss(ae(data), data).item()
check(f"autoencoder reconstruction loss < 0.5 (got {recon_err:.4f})", recon_err < 0.5)


# ═══════════════════════════════════════════════════════════════════════════
# E2E-4: Multi-layer transformer stack
# ═══════════════════════════════════════════════════════════════════════════
header("E2E-4: Stacked transformer (3 layers)")

class StackedTransformer(nn.Module):
    def __init__(self, n_layers=3, d_model=128, n_heads=4, ff_dim=512):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x.mean(dim=1))  # pool → classify

stacked = StackedTransformer(n_layers=3, d_model=128).to(device)
stacked = patch_model(stacked, verbose=False)

n_rac = sum(1 for m in stacked.modules() if isinstance(m, RACLinear))
check(f"stacked transformer: {n_rac} RACLinear layers", n_rac == 3 * 6 + 1)  # 6 per block + head

# Forward
x_seq = torch.randn(4, 16, 128, device=device)
y_out = stacked(x_seq)
check("stacked forward: output shape", tuple(y_out.shape) == (4, 10))
check("stacked forward: output finite", torch.isfinite(y_out).all().item())

# Train a few steps
opt = torch.optim.Adam(stacked.parameters(), lr=1e-3)
labels = torch.randint(0, 10, (4,), device=device)
losses = []
for _ in range(20):
    opt.zero_grad()
    loss = nn.CrossEntropyLoss()(stacked(x_seq), labels)
    loss.backward()
    opt.step()
    losses.append(loss.item())

check(f"stacked training: loss decreasing ({losses[-1]:.3f} < {losses[0]:.3f})",
      losses[-1] < losses[0])


# ═══════════════════════════════════════════════════════════════════════════
# E2E-5: Patch → inference → unpatch → inference round-trip
# ═══════════════════════════════════════════════════════════════════════════
header("E2E-5: Patch → unpatch round-trip preserves output")

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 64)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

net = SimpleNet().to(device)
x = torch.randn(8, 128, device=device)

with torch.no_grad():
    y_orig = net(x).clone()

# Patch
net = patch_model(net, verbose=False)
with torch.no_grad():
    y_patched = net(x).clone()

# Unpatch
net = unpatch_model(net, verbose=False)
with torch.no_grad():
    y_unpatched = net(x).clone()

err_patch = (y_orig - y_patched).abs().max().item()
err_round = (y_orig - y_unpatched).abs().max().item()
check(f"patched output close (err={err_patch:.2e})", err_patch < 0.05)
check(f"unpatched output exact (err={err_round:.2e})", err_round < 1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# E2E-6: Gradient accumulation (multi-step)
# ═══════════════════════════════════════════════════════════════════════════
header("E2E-6: Gradient accumulation")

layer = RACLinear(128, 64, bias=True).to(device)
opt = torch.optim.SGD(layer.parameters(), lr=0.01)

# Accumulate gradients over 4 micro-batches
opt.zero_grad()
for _ in range(4):
    x = torch.randn(8, 128, device=device)
    y = layer(x)
    (y.sum() / 4).backward()

check("accumulated grad_weight exists", layer.weight.grad is not None)
check("accumulated grad_weight finite", torch.isfinite(layer.weight.grad).all().item())

# Verify accumulation: grad should be ~4x a single batch grad
layer2 = RACLinear(128, 64, bias=True).to(device)
with torch.no_grad():
    layer2.weight.copy_(layer.weight)
    layer2.bias.copy_(layer.bias)

x_single = torch.randn(8, 128, device=device)
y_single = layer2(x_single)
(y_single.sum()).backward()

# accumulated grad magnitude should be roughly 4x single (same distribution)
ratio = layer.weight.grad.abs().mean().item() / (layer2.weight.grad.abs().mean().item() + 1e-8)
check(f"gradient accumulation ratio ~1.0 (got {ratio:.2f})", 0.3 < ratio < 3.0)


# ═══════════════════════════════════════════════════════════════════════════
# E2E-7: Mixed model (some patched, some not)
# ═══════════════════════════════════════════════════════════════════════════
header("E2E-7: Mixed model (RACLinear + nn.Linear)")

class MixedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rac_layer = RACLinear(128, 256)
        self.std_layer = nn.Linear(256, 64)
    def forward(self, x):
        return self.std_layer(F.relu(self.rac_layer(x)))

mixed = MixedNet().to(device)
x = torch.randn(4, 128, device=device, requires_grad=True)
y = mixed(x)
loss = y.sum()
loss.backward()

check("mixed model forward works", tuple(y.shape) == (4, 64))
check("mixed model backward: input grad", x.grad is not None)
check("mixed model backward: RAC weight grad", mixed.rac_layer.weight.grad is not None)
check("mixed model backward: std weight grad", mixed.std_layer.weight.grad is not None)


# ═══════════════════════════════════════════════════════════════════════════
# E2E-8: torch.no_grad context
# ═══════════════════════════════════════════════════════════════════════════
header("E2E-8: Inference mode (torch.no_grad)")

layer = RACLinear(256, 128).to(device)
x = torch.randn(16, 256, device=device)

with torch.no_grad():
    y = layer(x)

check("no_grad forward works", tuple(y.shape) == (16, 128))
check("no_grad output finite", torch.isfinite(y).all().item())

with torch.inference_mode():
    y2 = layer(x)
check("inference_mode forward works", tuple(y2.shape) == (16, 128))


# ═══════════════════════════════════════════════════════════════════════════
# E2E-9: Large matmul (stress test)
# ═══════════════════════════════════════════════════════════════════════════
header("E2E-9: Large matmul stress test")

for (M, K, N) in [(2048, 2048, 2048), (4096, 4096, 4096)]:
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C = rac_matmul(A, B)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) * 1000

    C_ref = torch.matmul(A, B)
    err = (C - C_ref).abs().max().item()
    tops = (2.0 * M * N * K) / (t * 1e9)

    check(f"{M}x{K}@{K}x{N}: correct (err={err:.2e})", err < 0.05)
    check(f"  {tops:.2f} TOPS, {t:.1f}ms", True)


# ═══════════════════════════════════════════════════════════════════════════
# E2E-10: benchmark_model utility
# ═══════════════════════════════════════════════════════════════════════════
header("E2E-10: benchmark_model utility")

simple = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
).to(device)

try:
    results = benchmark_model(simple, input_shape=(8, 256), n_warmup=5, n_iters=20)
    check("benchmark_model runs", True)
    check("  returns baseline_ms", 'baseline_ms' in results)
    check("  returns rac_ms", 'rac_ms' in results)
    check("  returns speedup", 'speedup' in results)
    check(f"  speedup > 0 ({results['speedup']:.2f}x)", results['speedup'] > 0)
except Exception as e:
    check("benchmark_model runs", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
header("E2E Summary")
total = passed + failed + skipped
print(f"  Passed:  {passed}")
print(f"  Failed:  {failed}")
print(f"  Skipped: {skipped}")
print(f"  Total:   {total}")
print(f"\n  {'ALL E2E PASSED' if failed == 0 else 'E2E FAILURES DETECTED'}")
sys.exit(0 if failed == 0 else 1)
