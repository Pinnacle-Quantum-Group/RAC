"""
rac_torch.py — RAC PyTorch Extension: Python Interface
Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026

Production-grade drop-in replacements for:
    torch.matmul / torch.mm      →  rac_matmul
    torch.nn.Linear              →  RACLinear
    Any nn.Module with Linear    →  patch_model(model)

Supports:
    - float32, float16, bfloat16 (auto-promotion to fp32 for compute)
    - torch.amp.autocast (mixed precision training)
    - torch.compile (PyTorch 2.x graph compilation)
    - Arbitrary batch dimensions (3D, 4D — transformers, etc.)
    - Full autograd backward (forward + backward are both RAC-native)

Usage:
    from rac_torch import RACLinear, rac_matmul, patch_model

    # Replace a single layer
    model.fc = RACLinear(768, 256)

    # Replace all Linear layers in a model
    model = patch_model(model)

    # Works with autocast
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional
import warnings

# ── Load compiled extension ──────────────────────────────────────────────────

try:
    import rac_cuda_ext as _rac
    _RAC_AVAILABLE = True
except ImportError:
    _RAC_AVAILABLE = False
    warnings.warn(
        "RAC CUDA extension not found. Run `pip install -e .` to compile. "
        "Falling back to torch.matmul.",
        RuntimeWarning, stacklevel=2
    )

def _rac_available() -> bool:
    return _RAC_AVAILABLE and torch.cuda.is_available()


# ── torch.compile compatibility ──────────────────────────────────────────────
# Mark RAC functions as non-decomposable for torch.compile.
# Without this, the compiler either errors or silently falls back to eager mode.

_compile_supported = hasattr(torch, 'compiler') and hasattr(torch.compiler, 'is_compiling')


# ── Autograd Function: matmul ────────────────────────────────────────────────

class RACMatmulFunction(Function):
    """
    torch.autograd.Function wrapping the RAC matmul kernel.

    Forward:  C = A @ B          via RAC micro-tiled kernel
    Backward: dA = dC @ B.T      via RAC NT kernel
              dB = A.T @ dC      via RAC TN kernel

    Supports float32, float16, bfloat16.
    fp16/bf16 are promoted to fp32 for compute, then cast back.
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(A, B)
        if _rac_available() and A.is_cuda and A.dtype in (torch.float32, torch.float16, torch.bfloat16):
            return _rac.matmul_forward(A, B)
        return torch.matmul(A.float(), B.float()).to(A.dtype)

    @staticmethod
    def backward(ctx, grad_C: torch.Tensor):
        A, B = ctx.saved_tensors
        grad_A = grad_B = None

        if _rac_available() and grad_C.is_cuda and grad_C.dtype in (torch.float32, torch.float16, torch.bfloat16):
            grads = _rac.matmul_backward(
                grad_C.contiguous(), A, B,
                ctx.needs_input_grad[0], ctx.needs_input_grad[1])
            if ctx.needs_input_grad[0]: grad_A = grads[0]
            if ctx.needs_input_grad[1]: grad_B = grads[1]
        else:
            gc = grad_C.float()
            if ctx.needs_input_grad[0]: grad_A = (gc @ B.float().t()).to(grad_C.dtype)
            if ctx.needs_input_grad[1]: grad_B = (A.float().t() @ gc).to(grad_C.dtype)

        return grad_A, grad_B


# ── Autograd Function: linear ────────────────────────────────────────────────

class RACLinearFunction(Function):
    """
    torch.autograd.Function wrapping the RAC linear kernel.

    Forward:  output = input @ weight.T + bias
    Backward: grad_input, grad_weight, grad_bias

    Supports arbitrary batch dimensions and mixed precision.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight, bias)
        ctx.has_bias = bias is not None

        if _rac_available() and input.is_cuda and input.dtype in (torch.float32, torch.float16, torch.bfloat16):
            bias_tensor = bias if bias is not None else torch.tensor([], device=input.device)
            return _rac.linear_forward(input, weight, bias_tensor)

        return F.linear(input.float(), weight.float(),
                        bias.float() if bias is not None else None).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if _rac_available() and grad_output.is_cuda and grad_output.dtype in (torch.float32, torch.float16, torch.bfloat16):
            grads = _rac.linear_backward(
                grad_output.contiguous(), input, weight, ctx.has_bias)
            if ctx.needs_input_grad[0]: grad_input  = grads[0]
            if ctx.needs_input_grad[1]: grad_weight = grads[1]
            if ctx.has_bias:            grad_bias   = grads[2]
        else:
            go = grad_output.float()
            w  = weight.float()
            inp = input.float()
            if ctx.needs_input_grad[0]:
                grad_input = (go @ w).to(grad_output.dtype)
            if ctx.needs_input_grad[1]:
                grad_weight = (go.reshape(-1, w.size(0)).t() @ inp.reshape(-1, w.size(1))).to(grad_output.dtype)
            if ctx.has_bias:
                grad_bias = go.reshape(-1, w.size(0)).sum(0).to(grad_output.dtype)

        return grad_input, grad_weight, grad_bias


# ── Register with torch.compile ──────────────────────────────────────────────
# This tells the compiler our custom ops are opaque and should not be traced through.

try:
    if hasattr(torch.library, 'custom_op'):
        pass  # PyTorch 2.4+ style — functions work as-is with allow_in_graph
    # For PyTorch 2.1-2.3, mark functions as non-decomposable
    if hasattr(torch._dynamo, 'allow_in_graph'):
        torch._dynamo.allow_in_graph(RACMatmulFunction)
        torch._dynamo.allow_in_graph(RACLinearFunction)
except Exception:
    pass  # Graceful degradation — eager mode still works


# ── Public API: functions ────────────────────────────────────────────────────

def rac_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for torch.matmul / torch.mm.
    Supports float32, float16, bfloat16. Falls back to torch.matmul on CPU.

    Example:
        C = rac_matmul(A, B)   # identical output to A @ B
    """
    return RACMatmulFunction.apply(A, B)


def rac_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Drop-in replacement for F.linear.
    Supports float32, float16, bfloat16.

    Example:
        out = rac_linear(x, self.weight, self.bias)
    """
    return RACLinearFunction.apply(input, weight, bias)


# ── RACLinear: drop-in nn.Linear replacement ────────────────────────────────

class RACLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using RAC.

    Identical interface to nn.Linear. Weight initialization, bias, and
    parameter shapes are unchanged. Only the compute kernel is replaced.

    Works with:
        - torch.autocast (mixed precision)
        - torch.compile (PyTorch 2.x)
        - DDP / FSDP (standard parameter handling)
        - model.half() / model.bfloat16()

    Example:
        layer = RACLinear(768, 256, bias=True)
        layer = RACLinear.from_linear(existing_linear_layer)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return RACLinearFunction.apply(input, self.weight, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> 'RACLinear':
        """Convert an existing nn.Linear to RACLinear, preserving weights exactly."""
        rac = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype
        )
        with torch.no_grad():
            rac.weight.copy_(linear.weight)
            if linear.bias is not None:
                rac.bias.copy_(linear.bias)
        return rac

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'backend=RAC')


# ── Model patching ───────────────────────────────────────────────────────────

def patch_model(
    model: nn.Module,
    verbose: bool = True,
    min_features: int = 64
) -> nn.Module:
    """
    Replace all nn.Linear layers in a model with RACLinear.
    Weights are preserved exactly. No retraining required.

    Works with any model architecture (BERT, GPT, Llama, ResNet, etc.)
    Compatible with model.half(), model.bfloat16(), and torch.autocast.

    Args:
        model:        Any nn.Module
        verbose:      Print a summary of replaced layers
        min_features: Skip layers smaller than this

    Example:
        from transformers import AutoModel
        model = AutoModel.from_pretrained('bert-base-uncased')
        model = patch_model(model)   # all Linear layers now use RAC
    """
    replaced = []

    def _replace(module: nn.Module, prefix: str = ''):
        for name, child in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name
            if isinstance(child, nn.Linear) and not isinstance(child, RACLinear):
                if (child.in_features >= min_features and
                    child.out_features >= min_features):
                    rac_layer = RACLinear.from_linear(child)
                    setattr(module, name, rac_layer)
                    replaced.append((full_name, child.in_features, child.out_features))
                elif verbose:
                    print(f'  skip  {full_name}: '
                          f'{child.in_features}->{child.out_features} (below min_features)')
            else:
                _replace(child, full_name)

    _replace(model)

    if verbose:
        print(f'patch_model: replaced {len(replaced)} Linear layers with RACLinear')
        for name, inf, outf in replaced:
            print(f'  +  {name}: {inf}->{outf}')

    return model


def unpatch_model(model: nn.Module, verbose: bool = True) -> nn.Module:
    """Reverse patch_model — restore all RACLinear layers to nn.Linear."""
    restored = []

    def _restore(module: nn.Module, prefix: str = ''):
        for name, child in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name
            if isinstance(child, RACLinear):
                linear = nn.Linear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype
                )
                with torch.no_grad():
                    linear.weight.copy_(child.weight)
                    if child.bias is not None:
                        linear.bias.copy_(child.bias)
                setattr(module, name, linear)
                restored.append(full_name)
            else:
                _restore(child, full_name)

    _restore(model)
    if verbose:
        print(f'unpatch_model: restored {len(restored)} RACLinear -> Linear')
    return model


# ── Benchmarking utility ────────────────────────────────────────────────────

def benchmark_model(
    model: nn.Module,
    input_shape: tuple,
    n_warmup: int = 10,
    n_iters: int = 100,
    device: str = 'cuda'
) -> dict:
    """
    Benchmark a model before and after RAC patching.
    Returns timing and speedup comparison.
    """
    import time

    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    def _time_model(m):
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = m(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = m(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iters * 1000

    baseline_ms  = _time_model(model)
    patched      = patch_model(model, verbose=False)
    rac_ms       = _time_model(patched)
    unpatch_model(patched, verbose=False)

    return {
        'baseline_ms': baseline_ms,
        'rac_ms':      rac_ms,
        'speedup':     baseline_ms / rac_ms if rac_ms > 0 else float('inf'),
        'n_iters':     n_iters
    }


# ── Info ────────────────────────────────────────────────────────────────────

def rac_info():
    """Print RAC extension status."""
    print("RAC PyTorch Extension — Pinnacle Quantum Group — March 2026")
    print(f"  CUDA available:      {torch.cuda.is_available()}")
    print(f"  RAC kernel loaded:   {_RAC_AVAILABLE}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  Device:              {props.name}")
        print(f"  Compute capability:  {props.major}.{props.minor}")
    print(f"  Backend:             {'RAC CUDA kernel' if _rac_available() else 'torch.matmul fallback'}")
    print(f"  Supported dtypes:    float32, float16, bfloat16")
    print(f"  torch.compile:       {'registered' if _compile_supported else 'eager only'}")
    print(f"  Mixed precision:     supported (auto-promotion to fp32)")
