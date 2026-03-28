#!/usr/bin/env python3
"""Minimal test: RAC kernel correctness via rac_cuda_ext"""
import torch
import sys

# Must be in a dir where rac_cuda_ext.so exists
try:
    import rac_cuda_ext
    print("Extension loaded OK")
except ImportError as e:
    print(f"FAILED: {e}")
    sys.exit(1)

device = 'cuda'

# Test 1: Identity matrix @ vector = vector
print("\n=== Test 1: Identity ===")
A = torch.eye(4, device=device, dtype=torch.float32)
B = torch.tensor([[1.0],[2.0],[3.0],[4.0]], device=device, dtype=torch.float32)
torch.cuda.synchronize()
C = rac_cuda_ext.matmul_forward(A, B)
torch.cuda.synchronize()
print(f"A:\n{A}")
print(f"B:\n{B}")
print(f"C (should be [1,2,3,4]):\n{C}")
print(f"Error: {(C - B).abs().max().item():.6f}")

# Test 2: Simple 2x2
print("\n=== Test 2: 2x2 ===")
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
torch.cuda.synchronize()
C = rac_cuda_ext.matmul_forward(A, B)
torch.cuda.synchronize()
C_ref = torch.matmul(A, B)
print(f"RAC:  {C}")
print(f"Ref:  {C_ref}")
print(f"Error: {(C - C_ref).abs().max().item():.6f}")

# Test 3: linear_forward (the actual path used by benchmark)
print("\n=== Test 3: linear_forward ===")
W = torch.randn(64, 128, device=device)
x = torch.randn(8, 128, device=device)
bias = torch.zeros(64, device=device)
torch.cuda.synchronize()
y_rac = rac_cuda_ext.linear_forward(x, W, bias)
torch.cuda.synchronize()
y_ref = torch.nn.functional.linear(x, W, bias)
err = (y_rac - y_ref).abs().max().item()
print(f"Shape: {y_rac.shape}")
print(f"Error: {err:.6f}")
print(f"{'PASS' if err < 0.01 else 'FAIL'}")

# Test 4: Larger
print("\n=== Test 4: 256x256 ===")
A = torch.randn(256, 256, device=device)
B = torch.randn(256, 256, device=device)
torch.cuda.synchronize()
C = rac_cuda_ext.matmul_forward(A, B)
torch.cuda.synchronize()
C_ref = torch.matmul(A, B)
err = (C - C_ref).abs().max().item()
print(f"Error: {err:.6f}")
print(f"{'PASS' if err < 0.1 else 'FAIL'}")
