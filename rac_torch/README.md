# rac_torch — RAC PyTorch Extension
**Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026**

Drop-in replacement for `nn.Linear` and `torch.matmul` using the RAC (Rotation-Accumulate) primitive. Zero multiplications in both forward and backward pass. Numerically equivalent to standard PyTorch within float32 tolerance.

---

## Install

```bash
# AMD ROCm
USE_ROCM=1 pip install -e .

# NVIDIA CUDA
pip install -e .
```

---

## Usage

### One-line model patching

```python
from rac_torch import patch_model

# Works with any model — HuggingFace, torchvision, custom
model = patch_model(model)
# All nn.Linear layers now use RAC. Weights preserved. No retraining needed.
```

### Drop-in layer replacement

```python
from rac_torch import RACLinear

# Identical interface to nn.Linear
layer = RACLinear(768, 256, bias=True)

# Convert from existing layer
layer = RACLinear.from_linear(existing_linear)
```

### Function-level replacement

```python
from rac_torch import rac_matmul, rac_linear

C   = rac_matmul(A, B)           # replaces torch.matmul / A @ B
out = rac_linear(x, weight, bias) # replaces F.linear
```

---

## What it does

`nn.Linear` computes `output = input @ weight.T + bias` using matrix multiply — thousands of multiplications per forward pass.

`RACLinear` computes the same result using the RAC micro-8x8 kernel — CORDIC rotation-accumulate, zero multiplications, routes through GPU Special Function Units (SFUs/transcendental units).

**Benchmark on AMD RX 7600 XT (gfx1102):**

| Matrix size | nn.Linear (hipBLAS) | RACLinear | Speedup |
|-------------|---------------------|-----------|---------|
| 256×256     | baseline            | 0.94×     | —       |
| 512×512     | baseline            | 1.30×     | **+30%** |
| 1024×1024   | baseline            | 1.94×     | **+94%** |
| 2048×2048   | baseline            | 2.24×     | **+124%** |
| 4096×4096   | baseline            | 2.48×     | **+148%** |

Zero multiplications. 100% correctness (max error < 5×10⁻⁴ at 4096²).

---

## Gradient support

Both forward and backward are RAC-native:

```
Forward:   C    = A @ B
Backward:  dA   = dC @ B.T    (RAC nt kernel)
           dB   = A.T @ dC    (RAC tn kernel)
```

The backward pass is mathematically identical to the forward pass — matrix multiply by transposed matrices — so RAC handles it with the same zero-multiplication kernels.

Training works out of the box:

```python
model = patch_model(model)
optimizer = torch.optim.Adam(model.parameters())

for x, y in dataloader:
    loss = criterion(model(x), y)
    loss.backward()    # RAC backward — zero multiplications
    optimizer.step()
```

---

## Fallback behavior

On CPU or non-float32 inputs, `RACLinear` falls back to `F.linear` automatically. The interface is identical regardless of backend.

```python
layer = RACLinear(256, 128)

# GPU float32 → RAC kernel (zero multiplications)
out = layer(x.cuda().float())

# CPU or float16 → torch fallback (transparent)
out = layer(x.cpu())
```

---

## Architecture

```
rac_torch.py          Python: autograd Functions, RACLinear, patch_model
rac_torch.cu          CUDA/HIP: micro-8x8 tiled kernels (NN, NT, TN)
setup.py              Build: auto-detects CUDA vs ROCm
test_rac_torch.py     Tests: correctness, gradients, training, benchmark
```

---

## Proprietary extended operations

`rac_cuda_ext` exposes only the public RAC primitive interface. Extended operations available on FIL hardware are not included in this release. See `RAC_OP_EXTENDED` in `rac.h`.

---

## License

© 2026 Michael A. Doran Jr. / Pinnacle Quantum Group. All rights reserved.  
Prior art established March 2026.  
CORDIC algorithm: public domain (Volder, 1959).
