#!/usr/bin/env python3
"""
test_rac_bvt.py — Build Verification Tests (BVT)
Pinnacle Quantum Group — March 2026

Quick smoke tests that validate basic build integrity and API surface.
Run after every build to catch regressions fast. Should complete in <30s.

Usage:
    python test_rac_bvt.py
"""

import sys
import math
import traceback

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
    print(f"\n{'─'*60}\n  {s}\n{'─'*60}")


# ═══════════════════════════════════════════════════════════════════════════
# BVT-1: Python imports
# ═══════════════════════════════════════════════════════════════════════════
header("BVT-1: Import verification")

try:
    import torch
    check("import torch", True)
except ImportError:
    check("import torch", False, "PyTorch not installed")
    print("FATAL: PyTorch required. Exiting.")
    sys.exit(1)

try:
    import torch.nn as nn
    check("import torch.nn", True)
except ImportError:
    check("import torch.nn", False)

try:
    from rac_torch import (
        RACLinear, RACMatmulFunction, RACLinearFunction,
        rac_matmul, rac_linear,
        patch_model, unpatch_model,
        benchmark_model, rac_info, _rac_available
    )
    check("import rac_torch (all public symbols)", True)
except ImportError as e:
    check("import rac_torch (all public symbols)", False, str(e))
    print("FATAL: Cannot import rac_torch. Exiting.")
    sys.exit(1)

try:
    import rac_cuda_ext
    check("import rac_cuda_ext (native extension)", True)
    NATIVE_AVAILABLE = True
except ImportError:
    skip("import rac_cuda_ext (native extension)", "extension not compiled")
    NATIVE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# BVT-2: Extension API surface
# ═══════════════════════════════════════════════════════════════════════════
header("BVT-2: Extension API surface")

if NATIVE_AVAILABLE:
    for fn in ['matmul_forward', 'matmul_backward', 'linear_forward', 'linear_backward']:
        check(f"rac_cuda_ext.{fn} exists", hasattr(rac_cuda_ext, fn))
else:
    skip("Extension API surface checks", "extension not compiled")


# ═══════════════════════════════════════════════════════════════════════════
# BVT-3: Class instantiation
# ═══════════════════════════════════════════════════════════════════════════
header("BVT-3: Class instantiation")

# RACLinear with bias
layer = RACLinear(64, 32, bias=True)
check("RACLinear(64, 32, bias=True) creates", True)
check("  .in_features == 64", layer.in_features == 64)
check("  .out_features == 32", layer.out_features == 32)
check("  .weight.shape == (32,64)", tuple(layer.weight.shape) == (32, 64))
check("  .bias is not None", layer.bias is not None)
check("  .bias.shape == (32,)", tuple(layer.bias.shape) == (32,))

# RACLinear without bias
layer_nb = RACLinear(128, 64, bias=False)
check("RACLinear(128, 64, bias=False) creates", True)
check("  .bias is None", layer_nb.bias is None)

# from_linear conversion
linear = nn.Linear(256, 128, bias=True)
rac = RACLinear.from_linear(linear)
check("RACLinear.from_linear preserves weight shape",
      tuple(rac.weight.shape) == tuple(linear.weight.shape))
check("RACLinear.from_linear preserves weights",
      torch.equal(rac.weight, linear.weight))
check("RACLinear.from_linear preserves bias",
      torch.equal(rac.bias, linear.bias))

# from_linear without bias
linear_nb = nn.Linear(256, 128, bias=False)
rac_nb = RACLinear.from_linear(linear_nb)
check("from_linear(no bias) preserves bias=None", rac_nb.bias is None)


# ═══════════════════════════════════════════════════════════════════════════
# BVT-4: CPU fallback forward pass
# ═══════════════════════════════════════════════════════════════════════════
header("BVT-4: CPU fallback forward pass")

torch.manual_seed(42)
x_cpu = torch.randn(4, 64)
layer_cpu = RACLinear(64, 32, bias=True)

try:
    y_cpu = layer_cpu(x_cpu)
    check("RACLinear forward on CPU succeeds", True)
    check("  output shape correct", tuple(y_cpu.shape) == (4, 32))
    check("  output is finite", torch.isfinite(y_cpu).all().item())
except Exception as e:
    check("RACLinear forward on CPU succeeds", False, str(e))

# rac_matmul CPU fallback
A_cpu = torch.randn(8, 16)
B_cpu = torch.randn(16, 4)
try:
    C_cpu = rac_matmul(A_cpu, B_cpu)
    check("rac_matmul CPU fallback", True)
    check("  shape correct", tuple(C_cpu.shape) == (8, 4))
    ref = torch.matmul(A_cpu, B_cpu)
    check("  matches torch.matmul", torch.allclose(C_cpu, ref, atol=1e-5))
except Exception as e:
    check("rac_matmul CPU fallback", False, str(e))

# rac_linear CPU fallback
try:
    y = rac_linear(x_cpu, layer_cpu.weight, layer_cpu.bias)
    check("rac_linear CPU fallback", True)
except Exception as e:
    check("rac_linear CPU fallback", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# BVT-5: CUDA forward pass (if available)
# ═══════════════════════════════════════════════════════════════════════════
header("BVT-5: CUDA forward pass")

HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    device = torch.device('cuda')

    x_gpu = torch.randn(4, 128, device=device)
    layer_gpu = RACLinear(128, 64, bias=True).to(device)
    try:
        y_gpu = layer_gpu(x_gpu)
        check("RACLinear forward on CUDA", True)
        check("  output on CUDA", y_gpu.is_cuda)
        check("  shape correct", tuple(y_gpu.shape) == (4, 64))
        check("  output is finite", torch.isfinite(y_gpu).all().item())
    except Exception as e:
        check("RACLinear forward on CUDA", False, str(e))

    # rac_matmul CUDA
    A_gpu = torch.randn(64, 64, device=device)
    B_gpu = torch.randn(64, 64, device=device)
    try:
        C_gpu = rac_matmul(A_gpu, B_gpu)
        check("rac_matmul CUDA", True)
        check("  output on CUDA", C_gpu.is_cuda)
    except Exception as e:
        check("rac_matmul CUDA", False, str(e))
else:
    skip("CUDA forward pass tests", "CUDA not available")


# ═══════════════════════════════════════════════════════════════════════════
# BVT-6: patch_model / unpatch_model
# ═══════════════════════════════════════════════════════════════════════════
header("BVT-6: patch_model / unpatch_model")

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = TinyModel()
check("TinyModel has 2 Linear layers",
      sum(1 for m in model.modules() if isinstance(m, nn.Linear)) == 2)

model = patch_model(model, verbose=False)
n_rac = sum(1 for m in model.modules() if isinstance(m, RACLinear))
check("patch_model converts Linear → RACLinear", n_rac == 2)

model = unpatch_model(model, verbose=False)
n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
n_rac_after = sum(1 for m in model.modules() if isinstance(m, RACLinear))
check("unpatch_model restores Linear", n_linear == 2 and n_rac_after == 0)


# ═══════════════════════════════════════════════════════════════════════════
# BVT-7: min_features threshold
# ═══════════════════════════════════════════════════════════════════════════
header("BVT-7: min_features threshold")

class MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.small = nn.Linear(8, 4)     # below threshold
        self.big = nn.Linear(256, 128)   # above threshold
    def forward(self, x):
        return self.big(x)

mm = MixedModel()
mm = patch_model(mm, verbose=False, min_features=64)
check("small layer (8→4) NOT patched", isinstance(mm.small, nn.Linear))
check("big layer (256→128) IS patched", isinstance(mm.big, RACLinear))


# ═══════════════════════════════════════════════════════════════════════════
# BVT-8: rac_info runs without error
# ═══════════════════════════════════════════════════════════════════════════
header("BVT-8: rac_info")

try:
    rac_info()
    check("rac_info() runs without error", True)
except Exception as e:
    check("rac_info() runs without error", False, str(e))

check("_rac_available() returns bool", isinstance(_rac_available(), bool))


# ═══════════════════════════════════════════════════════════════════════════
# BVT-9: extra_repr
# ═══════════════════════════════════════════════════════════════════════════
header("BVT-9: Module repr")

layer = RACLinear(512, 256)
r = repr(layer)
check("repr contains 'RAC'", 'RAC' in r)
check("repr contains '512'", '512' in r)
check("repr contains '256'", '256' in r)


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
header("BVT Summary")
total = passed + failed + skipped
print(f"  Passed:  {passed}")
print(f"  Failed:  {failed}")
print(f"  Skipped: {skipped}")
print(f"  Total:   {total}")
print(f"\n  {'ALL BVT PASSED' if failed == 0 else 'BVT FAILURES DETECTED'}")
sys.exit(0 if failed == 0 else 1)
