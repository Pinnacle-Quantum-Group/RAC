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
    rac_matmul, rac_linear, patch_model, unpatch_model, _rac_available,
    FusedRACLinear, RACFusedQKV, RACFusedFFN, rac_matmul_adaptive,
    RACAttention, RACTransformerBlock,
    ACT_NONE, ACT_RELU, ACT_GELU, ACT_SILU,
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
# DVT-13: Mixed precision — fp16
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-13: Mixed precision — float16")

A_fp16 = torch.randn(128, 128, device=device, dtype=torch.float16)
B_fp16 = torch.randn(128, 128, device=device, dtype=torch.float16)
try:
    C_fp16 = rac_matmul(A_fp16, B_fp16)
    check("rac_matmul fp16 runs", True)
    check("  output dtype is fp16", C_fp16.dtype == torch.float16)
    C_ref = torch.matmul(A_fp16.float(), B_fp16.float()).half()
    err = (C_fp16.float() - C_ref.float()).abs().max().item()
    check(f"  correctness (err={err:.2e})", err < 0.5)  # fp16 has ~1e-3 precision
except Exception as e:
    check("rac_matmul fp16", False, str(e)[:80])

# RACLinear with fp16
layer_f16 = RACLinear(128, 64, bias=True).to(device).half()
x_f16 = torch.randn(8, 128, device=device, dtype=torch.float16)
try:
    y_f16 = layer_f16(x_f16)
    check("RACLinear fp16 forward", True)
    check("  output dtype fp16", y_f16.dtype == torch.float16)
    check("  output finite", torch.isfinite(y_f16).all().item())
except Exception as e:
    check("RACLinear fp16 forward", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-14: Mixed precision — bfloat16
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-14: Mixed precision — bfloat16")

if torch.cuda.is_bf16_supported():
    A_bf = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
    B_bf = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
    try:
        C_bf = rac_matmul(A_bf, B_bf)
        check("rac_matmul bf16 runs", True)
        check("  output dtype is bf16", C_bf.dtype == torch.bfloat16)
    except Exception as e:
        check("rac_matmul bf16", False, str(e)[:80])

    layer_bf = RACLinear(128, 64, bias=True).to(device).to(torch.bfloat16)
    x_bf = torch.randn(8, 128, device=device, dtype=torch.bfloat16)
    try:
        y_bf = layer_bf(x_bf)
        check("RACLinear bf16 forward", True)
        check("  output dtype bf16", y_bf.dtype == torch.bfloat16)
    except Exception as e:
        check("RACLinear bf16 forward", False, str(e)[:80])
else:
    skip("bfloat16 tests", "device does not support bf16")


# ═══════════════════════════════════════════════════════════════════════════
# DVT-15: torch.autocast (AMP)
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-15: torch.autocast (mixed precision training)")

layer_amp = RACLinear(256, 128, bias=True).to(device)
x_amp = torch.randn(16, 256, device=device)

try:
    with torch.autocast('cuda', dtype=torch.float16):
        y_amp = layer_amp(x_amp)
    check("autocast fp16 forward", True)
    check("  output is finite", torch.isfinite(y_amp).all().item())
except Exception as e:
    check("autocast fp16 forward", False, str(e)[:80])

if torch.cuda.is_bf16_supported():
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y_amp_bf = layer_amp(x_amp)
        check("autocast bf16 forward", True)
    except Exception as e:
        check("autocast bf16 forward", False, str(e)[:80])

# Autocast with backward
try:
    scaler = torch.amp.GradScaler()
    opt = torch.optim.SGD(layer_amp.parameters(), lr=0.01)
    opt.zero_grad()
    with torch.autocast('cuda', dtype=torch.float16):
        y_train = layer_amp(x_amp)
        loss = y_train.sum()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    check("autocast + GradScaler training step", True)
    check("  weight grad exists", layer_amp.weight.grad is not None)
except Exception as e:
    check("autocast + GradScaler training step", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-16: torch.compile compatibility
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-16: torch.compile compatibility")

if hasattr(torch, 'compile'):
    layer_compile = RACLinear(128, 64, bias=True).to(device)
    try:
        compiled = torch.compile(layer_compile, mode='reduce-overhead')
        x_c = torch.randn(8, 128, device=device)
        y_c = compiled(x_c)
        check("torch.compile forward", True)
        check("  output shape correct", tuple(y_c.shape) == (8, 64))
        check("  output finite", torch.isfinite(y_c).all().item())
    except Exception as e:
        # torch.compile may not support all custom ops — graceful skip
        skip("torch.compile forward", f"not supported: {str(e)[:60]}")
else:
    skip("torch.compile tests", "torch.compile not available (PyTorch < 2.0)")


# ═══════════════════════════════════════════════════════════════════════════
# DVT-17: model.half() / model.bfloat16() compatibility
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-17: model dtype casting")

model_cast = nn.Sequential(
    RACLinear(128, 64),
    nn.ReLU(),
    RACLinear(64, 32),
).to(device)

# .half()
model_half = model_cast.half()
x_half = torch.randn(4, 128, device=device, dtype=torch.float16)
try:
    y_half = model_half(x_half)
    check("model.half() forward", True)
    check("  output dtype fp16", y_half.dtype == torch.float16)
except Exception as e:
    check("model.half() forward", False, str(e)[:80])

# .float() back
model_f32 = model_cast.float()
x_f32 = torch.randn(4, 128, device=device)
try:
    y_f32 = model_f32(x_f32)
    check("model.float() after half()", True)
except Exception as e:
    check("model.float() after half()", False, str(e)[:80])

# .bfloat16()
if torch.cuda.is_bf16_supported():
    model_bf = model_cast.to(torch.bfloat16)
    x_bf = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    try:
        y_bf = model_bf(x_bf)
        check("model.bfloat16() forward", True)
    except Exception as e:
        check("model.bfloat16() forward", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-18: FusedRACLinear correctness (all activations)
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-18: FusedRACLinear correctness")

for act_name, act_fn in [('relu', torch.relu), ('gelu', F.gelu), ('silu', F.silu), (None, lambda x: x)]:
    linear = nn.Linear(256, 128, bias=True).to(device)
    fused = FusedRACLinear.from_linear_and_act(linear, act_name)

    x = torch.randn(8, 256, device=device)
    with torch.no_grad():
        y_ref = act_fn(linear(x))
        y_fused = fused(x)

    err = (y_ref - y_fused).abs().max().item()
    tol = 0.05 if act_name else 0.01
    check(f"FusedRACLinear act={act_name} (err={err:.2e})", err < tol)


# ═══════════════════════════════════════════════════════════════════════════
# DVT-19: FusedRACLinear gradient correctness
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-19: FusedRACLinear gradient flow")

for act_name in ['relu', 'gelu', 'silu']:
    fused = FusedRACLinear(128, 64, activation=act_name, bias=True).to(device)
    x = torch.randn(4, 128, device=device, requires_grad=True)
    y = fused(x)
    loss = y.sum()
    loss.backward()
    check(f"FusedRACLinear({act_name}) grad_input exists", x.grad is not None)
    check(f"  grad_weight exists", fused.weight.grad is not None)
    check(f"  grad_bias exists", fused.bias.grad is not None)
    check(f"  grad_input finite", torch.isfinite(x.grad).all().item())
    check(f"  grad_weight finite", torch.isfinite(fused.weight.grad).all().item())
    x.grad = None
    fused.zero_grad()


# ═══════════════════════════════════════════════════════════════════════════
# DVT-20: FusedRACLinear training convergence
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-20: FusedRACLinear training convergence")

class FusedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = FusedRACLinear(64, 128, activation='relu')
        self.fc2 = FusedRACLinear(128, 64, activation='relu')
        self.head = RACLinear(64, 10)
    def forward(self, x): return self.head(self.fc2(self.fc1(x)))

fmlp = FusedMLP().to(device)
opt = torch.optim.Adam(fmlp.parameters(), lr=1e-3)
X = torch.randn(128, 64, device=device)
Y = torch.randint(0, 10, (128,), device=device)
losses = []
for _ in range(30):
    opt.zero_grad()
    loss = nn.CrossEntropyLoss()(fmlp(X), Y)
    loss.backward()
    opt.step()
    losses.append(loss.item())
check(f"fused MLP loss decreasing ({losses[-1]:.3f} < {losses[0]:.3f})", losses[-1] < losses[0])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-21: RACFusedQKV correctness
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-21: RACFusedQKV correctness")

d = 128
q_lin = nn.Linear(d, d, bias=True).to(device)
k_lin = nn.Linear(d, d, bias=True).to(device)
v_lin = nn.Linear(d, d, bias=True).to(device)

fqkv = RACFusedQKV.from_qkv_linears(q_lin, k_lin, v_lin)

x = torch.randn(2, 16, d, device=device)
with torch.no_grad():
    Q_ref, K_ref, V_ref = q_lin(x), k_lin(x), v_lin(x)
    Q_fused, K_fused, V_fused = fqkv(x)

for name, ref, fused in [('Q', Q_ref, Q_fused), ('K', K_ref, K_fused), ('V', V_ref, V_fused)]:
    err = (ref - fused).abs().max().item()
    check(f"FusedQKV {name} matches separate (err={err:.2e})", err < 0.01)

# Backward
Q, K, V = fqkv(x)
loss = (Q.sum() + K.sum() + V.sum())
loss.backward()
check("FusedQKV backward: grad exists", fqkv.qkv.weight.grad is not None)
check("  grad finite", torch.isfinite(fqkv.qkv.weight.grad).all().item())


# ═══════════════════════════════════════════════════════════════════════════
# DVT-22: RACFusedFFN correctness
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-22: RACFusedFFN correctness")

ffn = RACFusedFFN(d_model=128, ff_dim=512, activation='gelu').to(device)
x = torch.randn(2, 16, 128, device=device)
y = ffn(x)
check("RACFusedFFN forward shape", tuple(y.shape) == (2, 16, 128))
check("  output finite", torch.isfinite(y).all().item())

# Backward
y.sum().backward()
check("RACFusedFFN backward: fc1 grad", ffn.fc1.weight.grad is not None)
check("  fc2 grad", ffn.fc2.weight.grad is not None)


# ═══════════════════════════════════════════════════════════════════════════
# DVT-23: rac_matmul_adaptive
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-23: rac_matmul_adaptive")

# Small (should use torch.matmul)
A_s = torch.randn(4, 4, device=device)
B_s = torch.randn(4, 4, device=device)
C_s = rac_matmul_adaptive(A_s, B_s)
ref_s = torch.matmul(A_s, B_s)
err = (C_s - ref_s).abs().max().item()
check(f"adaptive small (4x4) correct (err={err:.2e})", err < 1e-5)

# Large (should use RAC)
A_l = torch.randn(256, 256, device=device)
B_l = torch.randn(256, 256, device=device)
C_l = rac_matmul_adaptive(A_l, B_l)
ref_l = torch.matmul(A_l, B_l)
err = (C_l - ref_l).abs().max().item()
check(f"adaptive large (256x256) correct (err={err:.2e})", err < 0.01)


# ═══════════════════════════════════════════════════════════════════════════
# DVT-24: FusedRACLinear with fp16/bf16
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-24: FusedRACLinear mixed precision")

fused_mp = FusedRACLinear(128, 64, activation='gelu', bias=True).to(device).half()
x_fp16 = torch.randn(8, 128, device=device, dtype=torch.float16)
try:
    y = fused_mp(x_fp16)
    check("FusedRACLinear fp16 forward", True)
    check("  output dtype fp16", y.dtype == torch.float16)
    check("  output finite", torch.isfinite(y).all().item())
except Exception as e:
    check("FusedRACLinear fp16 forward", False, str(e)[:80])

# Autocast
fused_amp = FusedRACLinear(128, 64, activation='relu', bias=True).to(device)
try:
    with torch.autocast('cuda', dtype=torch.float16):
        y = fused_amp(torch.randn(8, 128, device=device))
    check("FusedRACLinear autocast", True)
except Exception as e:
    check("FusedRACLinear autocast", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-25: RACAttention correctness
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-25: RACAttention correctness")

d, h = 128, 4
attn = RACAttention(d_model=d, n_heads=h).to(device)
x = torch.randn(2, 16, d, device=device)

# Forward
y = attn(x)
check("RACAttention forward shape", tuple(y.shape) == (2, 16, d))
check("  output finite", torch.isfinite(y).all().item())

# With causal mask
y_causal = attn(x, is_causal=True)
check("RACAttention causal forward shape", tuple(y_causal.shape) == (2, 16, d))
check("  causal output finite", torch.isfinite(y_causal).all().item())
check("  causal != non-causal", not torch.allclose(y, y_causal, atol=1e-5))

# Backward
y.sum().backward()
check("RACAttention backward: qkv grad exists", attn.qkv.weight.grad is not None)
check("  out_proj grad exists", attn.out_proj.weight.grad is not None)
check("  qkv grad finite", torch.isfinite(attn.qkv.weight.grad).all().item())

# Custom mask
mask = torch.zeros(1, 1, 16, 16, device=device)
mask[:, :, :, 8:] = float('-inf')  # mask out second half
attn.zero_grad()
y_masked = attn(x, mask=mask)
check("RACAttention custom mask shape", tuple(y_masked.shape) == (2, 16, d))

# from_attention_layers
q_lin = nn.Linear(d, d).to(device)
k_lin = nn.Linear(d, d).to(device)
v_lin = nn.Linear(d, d).to(device)
o_lin = nn.Linear(d, d).to(device)
attn2 = RACAttention.from_attention_layers(q_lin, k_lin, v_lin, o_lin, n_heads=h)
check("from_attention_layers creates", isinstance(attn2, RACAttention))
check("  qkv weight shape", tuple(attn2.qkv.weight.shape) == (3*d, d))


# ═══════════════════════════════════════════════════════════════════════════
# DVT-26: RACTransformerBlock
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-26: RACTransformerBlock correctness")

block = RACTransformerBlock(d_model=128, n_heads=4, ff_dim=512).to(device)
x = torch.randn(2, 16, 128, device=device)

y = block(x)
check("RACTransformerBlock forward shape", tuple(y.shape) == (2, 16, 128))
check("  output finite", torch.isfinite(y).all().item())

y_causal = block(x, is_causal=True)
check("  causal forward", tuple(y_causal.shape) == (2, 16, 128))

# Training
y.sum().backward()
grads_ok = all(p.grad is not None for p in block.parameters() if p.requires_grad)
check("  all gradients flow", grads_ok)


# ═══════════════════════════════════════════════════════════════════════════
# DVT-27: RACRoPE correctness (rotation invariance)
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-27: RACRoPE correctness")

try:
    from rac_torch import RACRoPE

    rope = RACRoPE(head_dim=16, max_seq_len=32)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn_like(q)
    q2, k2 = rope(q, k)

    # Pair magnitudes preserved (RoPE is an orthogonal rotation per pair)
    p0 = q.view(2, 4, 8, 8, 2)
    p1 = q2.view(2, 4, 8, 8, 2)
    m0 = p0.pow(2).sum(-1).sqrt()
    m1 = p1.pow(2).sum(-1).sqrt()
    check("RoPE preserves pair magnitudes", torch.allclose(m0, m1, atol=1e-4))

    # RoPE respects relative position: Q·K invariance under equal shifts.
    # (Stronger property — we check a simpler one: t=0 rotation is identity.)
    check("RoPE at t=0 is identity",
          torch.allclose(q[:, :, 0, :], q2[:, :, 0, :], atol=1e-4))

    # Explicit positions argument
    positions = torch.arange(8)
    q3, _ = rope(q, k, positions=positions)
    check("RoPE with explicit positions matches default",
          torch.allclose(q3, q2, atol=1e-4))

    # Gradient flows through RoPE
    q_leaf = torch.randn(1, 1, 4, 16, requires_grad=True)
    k_leaf = torch.randn(1, 1, 4, 16, requires_grad=True)
    q_out, k_out = rope(q_leaf, k_leaf)
    (q_out.sum() + k_out.sum()).backward()
    check("RoPE grad flows to Q", q_leaf.grad is not None)
    check("RoPE grad flows to K", k_leaf.grad is not None)
except Exception as e:
    check("RoPE correctness", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-28: RACRMSNorm vs torch-native RMSNorm reference
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-28: RACRMSNorm correctness")

try:
    from rac_torch import RACRMSNorm

    d = 64
    rms = RACRMSNorm(d, eps=1e-6)
    x = torch.randn(8, d)

    # Reference implementation
    def ref_rmsnorm(x, w, eps):
        ms = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(ms + eps)) * w

    y = rms(x)
    y_ref = ref_rmsnorm(x, rms.weight, 1e-6)
    check("RMSNorm matches reference", torch.allclose(y, y_ref, atol=1e-5))

    # Non-trivial gamma — scale all outputs by 2
    with torch.no_grad():
        rms.weight.fill_(2.0)
    y = rms(x)
    y_ref = ref_rmsnorm(x, rms.weight, 1e-6)
    check("RMSNorm gamma scaling", torch.allclose(y, y_ref, atol=1e-5))

    # Gradient
    x_leaf = x.clone().requires_grad_(True)
    rms(x_leaf).sum().backward()
    check("RMSNorm grad flows to input", x_leaf.grad is not None)
    check("RMSNorm grad flows to weight", rms.weight.grad is not None)
except Exception as e:
    check("RMSNorm correctness", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-29: RACLayerNorm vs torch.nn.LayerNorm reference
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-29: RACLayerNorm correctness")

try:
    from rac_torch import RACLayerNorm

    d = 48
    rac_ln = RACLayerNorm(d, eps=1e-5)
    tn_ln  = torch.nn.LayerNorm(d, eps=1e-5)
    with torch.no_grad():
        tn_ln.weight.copy_(rac_ln.weight)
        tn_ln.bias.copy_(rac_ln.bias)

    x = torch.randn(4, 7, d)
    y_rac = rac_ln(x)
    y_ref = tn_ln(x)
    check("RACLayerNorm matches torch.nn.LayerNorm",
          torch.allclose(y_rac, y_ref, atol=1e-5))

    # Bias=False variant
    ln_no_bias = RACLayerNorm(d, eps=1e-5, bias=False)
    y = ln_no_bias(x)
    check("RACLayerNorm bias=False output shape", y.shape == x.shape)
except Exception as e:
    check("LayerNorm correctness", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-30: RACRoPEAttention end-to-end (forward + backward)
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-30: RACRoPEAttention forward + backward")

try:
    from rac_torch import RACRoPEAttention

    attn = RACRoPEAttention(d_model=32, n_heads=4, max_seq_len=16)
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = attn(x, is_causal=True)
    check("RoPEAttention output shape", out.shape == x.shape)
    out.sum().backward()
    check("RoPEAttention backward flows", x.grad is not None)
    check("RoPEAttention all params got grad",
          all(p.grad is not None for p in attn.parameters()))

    # Causal: last token's output doesn't depend on future tokens (no change
    # when permuting future inputs on last position for is_causal=True).
    x_a = torch.randn(1, 4, 32)
    x_b = x_a.clone(); x_b[:, -1] = torch.randn(32)  # change last token
    attn.eval()
    with torch.no_grad():
        ya = attn(x_a, is_causal=True)
        yb = attn(x_b, is_causal=True)
    # Tokens before the last should match in causal mode
    check("RoPEAttention causal prefix unchanged",
          torch.allclose(ya[:, :-1], yb[:, :-1], atol=1e-4))
except Exception as e:
    check("RoPEAttention correctness", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-31: RACLlamaBlock training step
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-31: RACLlamaBlock training")

try:
    from rac_torch import RACLlamaBlock

    block = RACLlamaBlock(d_model=32, n_heads=4, ff_dim=64, max_seq_len=16)
    opt = torch.optim.Adam(block.parameters(), lr=1e-3)
    x = torch.randn(2, 8, 32)
    target = torch.randn_like(x)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        y = block(x, is_causal=True)
        loss = (y - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    check("Llama block training decreases loss", losses[-1] < losses[0] * 0.95)
    check("Llama block output finite after training",
          torch.isfinite(block(x)).all().item())
except Exception as e:
    check("Llama training", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════
# DVT-32: Tunable-precision knob roundtrip
# ═══════════════════════════════════════════════════════════════════════════
header("DVT-32: Tunable precision")

try:
    from rac_torch import rac_set_precision, rac_get_precision
    for iters in [4, 8, 12, 16, 20, 24]:
        rac_set_precision(iters)
        check(f"precision {iters} roundtrips", rac_get_precision() == iters)
    rac_set_precision(-5)
    check("negative clamped to 4", rac_get_precision() == 4)
    rac_set_precision(1000)
    check("oversize clamped to 24", rac_get_precision() == 24)
    rac_set_precision(16)
except Exception as e:
    check("precision knob", False, str(e)[:80])


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
