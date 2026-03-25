Write CUDA and ROCm/HIP C code implementing the RAC (Rotation-Accumulate) primitive using on-die Special Function Units (SFUs) instead of tensor cores or multiplier hardware.
Context:
RAC replaces Multiply-Accumulate (MAC) with CORDIC-based geometric rotation. The key insight is that modern GPUs already have CORDIC hardware on-die in their Special Function Units (SFUs), used internally for sin/cos/atan/sqrt. RAC hijacks those units as a first-class compute primitive instead of routing tensor operations through multiplier-based tensor cores.
Implement the following four primitives in both CUDA and HIP:

rac_rotate(float2 v, float theta) — rotate vector v by angle theta using SFU __sinf/__cosf, returns float2
rac_project(float2 v, float theta) — rotate v by theta and return x-axis projection only (the MAC-degenerate case: equivalent to scalar multiply)
rac_accumulate(float2* vectors, float* thetas, int n) — batched rotation-accumulate across n input vectors, returns float2 sum
rac_matmul(float* A, float* B, float* C, int M, int N, int K) — matrix multiply expressed as rotation sequences using rac_project, no multiplications in the compute kernel

Requirements:

CUDA version uses __sinf, __cosf intrinsics which route to SFU hardware on NVIDIA
HIP version uses __ocml_sin_f32, __ocml_cos_f32 or equivalent AMD SFU intrinsics
Kernels must contain zero * multiply operators in the core compute path — all scaling via SFU trig calls
Include a benchmark harness that runs RAC matmul vs cuBLAS SGEMM (CUDA) and rocBLAS SGEMM (HIP) and reports: ops/sec, energy per op (via NVML / ROCm SMI), and wall time
Include a correctness check validating RAC matmul output matches cuBLAS/rocBLAS within floating point tolerance
Comments must explicitly note each location where a multiply would have appeared in a MAC implementation and what replaced it

Deliverables:

rac_cuda.cu — full CUDA implementation
rac_hip.cpp — full HIP/ROCm implementation
rac_benchmark.sh — build and run script for both
README.md — explaining the SFU routing strategy and how to read the benchmark output

