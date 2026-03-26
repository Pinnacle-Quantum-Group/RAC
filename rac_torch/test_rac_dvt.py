#!/usr/bin/env python3
"""
test_rac_dvt.py — Design Verification Tests (DVT)
Pinnacle Quantum Group — March 2026

Thorough correctness and numerical validation of all RAC primitives.
Covers forward/backward accuracy, edge cases, numerical stability,
gradient correctness, and dtype/shape handling.

Requires CUDA. Should complete in <5min.

Usage:
    python test_rac_dvt.py
"""

import sys
import math
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

from rac_torch import (
    RACLinear, RACMatmulFunction, RACLinearFunction,
    rac_matmul, rac_linear, patch_model, unpatch_model, _rac_available
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

def require_cuda():
    if not torch.cuda.is_available():
        print("FATAL: DVT requires CUDA. Exiting.")
        sys.exit(2)

require_cuda()
device = torch.device('cuda')
torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# DVT-1: rac_matmul forward correctness sweep
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-1: rac_matmul forward correctness (shape sweep)")

shapes = [
    # (M, K, N) — square, rectangular, small, large, non-power-of-2
    (1, 1, 1),
    (1, 64, 1),
    (1, 1, 64),
    (16, 16, 16),
    (32, 64, 128),
    (128, 64, 32),
    (63, 65, 67),      # non-power-of-2
    (127, 255, 131),   # primes-ish
    (256, 256, 256),
    (512, 1024, 512),
    (1024, 1024, 1024),
    (1, 4096, 1),      # degenerate: single row/col
    (4096, 1, 4096),   # degenerate: K=1
]

for (M, K, N) in shapes:
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    C_ref = torch.matmul(A, B)
    C_rac = rac_matmul(A, B)
    err = (C_ref - C_rac).abs().max().item()
    # Tolerance scales with K (accumulation length)
    tol = max(1e-2, K * 1e-5)
    check(f"matmul {M}x{K} @ {K}x{N}  (err={err:.2e})", err < tol, f"tol={tol:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# DVT-2: RACLinear forward correctness with bias/no-bias
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-2: RACLinear forward correctness (bias & no-bias)")

for bias in [True, False]:
    for (B, IN, OUT) in [(1, 64, 64), (8, 128, 256), (32, 768, 768), (4, 1024, 512)]:
        linear = nn.Linear(IN, OUT, bias=bias).to(device).float()
        rac = RACLinear.from_linear(linear)
        x = torch.randn(B, IN, device=device)
        with torch.no_grad():
            y_ref = linear(x)
            y_rac = rac(x)
        err = (y_ref - y_rac).abs().max().item()
        check(f"Linear {B}x{IN}→{OUT} bias={bias} (err={err:.2e})", err < 1e-2)


# ═══════════════════════════════════════════════════════════════════════════
# DVT-3: 3D/4D batch dimensions (transformer shapes)
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-3: Batch dimensions (3D, 4D)")

for shape in [(2, 32, 256), (1, 1, 64), (4, 16, 128)]:
    linear = nn.Linear(shape[-1], 512, bias=True).to(device)
    rac = RACLinear.from_linear(linear)
    x = torch.randn(*shape, device=device)
    with torch.no_grad():
        y_ref = linear(x)
        y_rac = rac(x)
    err = (y_ref - y_rac).abs().max().item()
    check(f"3D input {shape}→512  (err={err:.2e})", err < 1e-2)

# 4D
for shape in [(2, 8, 16, 128),]:
    linear = nn.Linear(128, 256, bias=True).to(device)
    rac = RACLinear.from_linear(linear)
    x = torch.randn(*shape, device=device)
    with torch.no_grad():
        y_ref = linear(x)
        y_rac = rac(x)
    err = (y_ref - y_rac).abs().max().item()
    check(f"4D input {shape}→256  (err={err:.2e})", err < 1e-2)


# ═══════════════════════════════════════════════════════════════════════════
# DVT-4: Gradient correctness — matmul
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-4: Gradient correctness — rac_matmul")

for (M, K, N) in [(8, 8, 8), (16, 32, 16), (4, 64, 8)]:
    A = torch.randn(M, K, device=device, dtype=torch.float64, requires_grad=True)
    B = torch.randn(K, N, device=device, dtype=torch.float64, requires_grad=True)
    try:
        result = gradcheck(RACMatmulFunction.apply, (A, B), eps=1e-4, atol=1e-3, rtol=1e-3)
        check(f"gradcheck matmul {M}x{K}@{K}x{N}", result)
    except Exception as e:
        check(f"gradcheck matmul {M}x{K}@{K}x{N}", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-5: Gradient correctness — RACLinear
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-5: Gradient correctness — RACLinear")

for bias in [True, False]:
    for (IN, OUT) in [(64, 32), (128, 128), (256, 64)]:
        layer = RACLinear(IN, OUT, bias=bias).to(device)
        x = torch.randn(4, IN, device=device, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        check(f"grad_input exists ({IN}→{OUT} bias={bias})", x.grad is not None)
        check(f"grad_weight exists ({IN}→{OUT} bias={bias})", layer.weight.grad is not None)
        if bias:
            check(f"grad_bias exists ({IN}→{OUT})", layer.bias.grad is not None)

        # Verify gradient shapes
        if x.grad is not None:
            check(f"grad_input shape", tuple(x.grad.shape) == (4, IN))
        if layer.weight.grad is not None:
            check(f"grad_weight shape", tuple(layer.weight.grad.shape) == (OUT, IN))

        # Reset grads
        x.grad = None
        layer.zero_grad()


# ═══════════════════════════════════════════════════════════════════════════
# DVT-6: Gradient numerical accuracy (manual finite diff)
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-6: Gradient numerical accuracy (finite difference)")

M, K, N = 16, 16, 16
eps = 1e-3

A = torch.randn(M, K, device=device, requires_grad=True)
B = torch.randn(K, N, device=device, requires_grad=True)
C = rac_matmul(A, B)
C.sum().backward()
grad_A_analytic = A.grad.clone()

# Finite difference for A[0,0]
A_plus = A.data.clone(); A_plus[0, 0] += eps
A_minus = A.data.clone(); A_minus[0, 0] -= eps
fd = (rac_matmul(A_plus, B.data).sum() - rac_matmul(A_minus, B.data).sum()) / (2 * eps)
rel_err = abs(fd.item() - grad_A_analytic[0, 0].item()) / (abs(fd.item()) + 1e-8)
check(f"finite diff grad_A[0,0] rel_err={rel_err:.2e}", rel_err < 0.05)


# ═══════════════════════════════════════════════════════════════════════════
# DVT-7: Numerical stability — large/small values
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-7: Numerical stability")

# Large values
A_big = torch.randn(64, 64, device=device) * 1000.0
B_big = torch.randn(64, 64, device=device) * 1000.0
C_rac = rac_matmul(A_big, B_big)
C_ref = torch.matmul(A_big, B_big)
rel_err = ((C_rac - C_ref).abs() / (C_ref.abs() + 1e-6)).max().item()
check(f"large values (1000x scale) rel_err={rel_err:.2e}", rel_err < 0.01)
check("  no NaN in output", not torch.isnan(C_rac).any().item())
check("  no Inf in output", not torch.isinf(C_rac).any().item())

# Small values
A_small = torch.randn(64, 64, device=device) * 1e-4
B_small = torch.randn(64, 64, device=device) * 1e-4
C_rac = rac_matmul(A_small, B_small)
check("small values (1e-4 scale) no NaN", not torch.isnan(C_rac).any().item())
check("  no Inf", not torch.isinf(C_rac).any().item())

# Mixed signs
A_mixed = torch.randn(64, 64, device=device)
A_mixed[:32, :] *= -1
B_mixed = torch.randn(64, 64, device=device)
C_rac = rac_matmul(A_mixed, B_mixed)
C_ref = torch.matmul(A_mixed, B_mixed)
err = (C_rac - C_ref).abs().max().item()
check(f"mixed signs (err={err:.2e})", err < 0.01)

# Zero input
A_zero = torch.zeros(64, 64, device=device)
B_rand = torch.randn(64, 64, device=device)
C_rac = rac_matmul(A_zero, B_rand)
check("zero A → zero output", C_rac.abs().max().item() < 1e-6)

# Identity-like
A_eye = torch.eye(64, device=device)
B_data = torch.randn(64, 64, device=device)
C_rac = rac_matmul(A_eye, B_data)
err = (C_rac - B_data).abs().max().item()
check(f"identity A → output == B  (err={err:.2e})", err < 1e-2)


# ═══════════════════════════════════════════════════════════════════════════
# DVT-8: Contiguity handling
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-8: Non-contiguous tensor handling")

A = torch.randn(64, 128, device=device)
B = torch.randn(128, 64, device=device)

# Transposed (non-contiguous)
A_t = A.t()  # now 128x64, non-contiguous
B_t = B.t()  # now 64x128, non-contiguous
try:
    C1 = rac_matmul(A_t.contiguous(), B_t.contiguous())
    check("contiguous inputs work", True)
except Exception as e:
    check("contiguous inputs work", False, str(e)[:80])

# Sliced (non-contiguous)
A_slice = torch.randn(128, 128, device=device)[:64, :]
check("sliced tensor is non-contiguous", not A_slice.is_contiguous())
try:
    C2 = rac_matmul(A_slice, torch.randn(128, 32, device=device))
    check("non-contiguous A handled", True)
except Exception as e:
    check("non-contiguous A handled", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-9: Weight preservation through patch/unpatch cycle
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-9: Weight preservation (patch → unpatch)")

linear_orig = nn.Linear(256, 128, bias=True).to(device)
w_orig = linear_orig.weight.data.clone()
b_orig = linear_orig.bias.data.clone()

class SingleLayer(nn.Module):
    def __init__(self, l): super().__init__(); self.fc = l
    def forward(self, x): return self.fc(x)

model = SingleLayer(linear_orig)
model = patch_model(model, verbose=False)
check("after patch: is RACLinear", isinstance(model.fc, RACLinear))
check("after patch: weights match", torch.equal(model.fc.weight.data, w_orig))
check("after patch: bias match", torch.equal(model.fc.bias.data, b_orig))

model = unpatch_model(model, verbose=False)
check("after unpatch: is nn.Linear", isinstance(model.fc, nn.Linear))
check("after unpatch: weights match", torch.equal(model.fc.weight.data, w_orig))
check("after unpatch: bias match", torch.equal(model.fc.bias.data, b_orig))


# ═══════════════════════════════════════════════════════════════════════════
# DVT-10: Training convergence
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-10: Training convergence")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            RACLinear(64, 128), nn.ReLU(),
            RACLinear(128, 64), nn.ReLU(),
            RACLinear(64, 10),
        )
    def forward(self, x): return self.net(x)

mlp = MLP().to(device)
opt = torch.optim.SGD(mlp.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Generate fixed training data
X_train = torch.randn(256, 64, device=device)
Y_train = torch.randint(0, 10, (256,), device=device)

losses = []
for epoch in range(50):
    opt.zero_grad()
    loss = loss_fn(mlp(X_train), Y_train)
    loss.backward()
    opt.step()
    losses.append(loss.item())

check("loss decreases over 50 epochs", losses[-1] < losses[0])
check(f"final loss < initial ({losses[-1]:.3f} < {losses[0]:.3f})",
      losses[-1] < losses[0] * 0.5)

# Check loss is monotonically decreasing (roughly)
n_increases = sum(1 for i in range(1, len(losses)) if losses[i] > losses[i-1] + 0.1)
check(f"mostly decreasing (only {n_increases} increases)", n_increases < 10)


# ═══════════════════════════════════════════════════════════════════════════
# DVT-11: Determinism
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-11: Determinism")

A = torch.randn(256, 256, device=device)
B = torch.randn(256, 256, device=device)
C1 = rac_matmul(A, B)
C2 = rac_matmul(A, B)
check("same input → same output (deterministic)", torch.equal(C1, C2))


# ═══════════════════════════════════════════════════════════════════════════
# DVT-12: Alpha/Beta semantics (kernel level)
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-12: Matmul output consistency")

# Verify C = A @ B via both rac_matmul and RACLinear match
A = torch.randn(32, 128, device=device)
W = torch.randn(64, 128, device=device)  # weight: (out, in)
b = torch.randn(64, device=device)

y_fn = rac_linear(A, W, b)
y_manual = rac_matmul(A, W.t()) + b
err = (y_fn - y_manual).abs().max().item()
check(f"rac_linear == rac_matmul(A, W.T) + b  (err={err:.2e})", err < 1e-2)


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
header("DVT Summary")
total = passed + failed + skipped
print(f"  Passed:  {passed}")
print(f"  Failed:  {failed}")
print(f"  Skipped: {skipped}")
print(f"  Total:   {total}")
print(f"\n  {'ALL DVT PASSED' if failed == 0 else 'DVT FAILURES DETECTED'}")
sys.exit(0 if failed == 0 else 1)
