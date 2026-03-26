"""
setup.py — RAC PyTorch Extension Build
Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026

Build:
    pip install -e .                    # editable install
    python setup.py install             # standard install
    python setup.py build_ext --inplace # build in place for dev

CUDA:
    CUDA_HOME must be set, or nvcc must be on PATH.
    Override architectures: TORCH_CUDA_ARCH_LIST="7.0 8.0 9.0" pip install -e .

ROCm/HIP:
    Set USE_ROCM=1 or ensure hipcc is on PATH.
    The extension auto-detects ROCm via torch.version.hip.
"""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ── Backend detection ────────────────────────────────────────────────────────

IS_ROCM = (
    os.environ.get('USE_ROCM', '0') == '1' or
    (hasattr(torch.version, 'hip') and torch.version.hip is not None)
)

print(f"RAC build: {'ROCm/HIP' if IS_ROCM else 'CUDA'} backend")

# ── Compiler flags ───────────────────────────────────────────────────────────

# Allow users to control fast-math (disable for strict numerical reproducibility)
USE_FAST_MATH = os.environ.get('RAC_FAST_MATH', '1') == '1'

CUDA_FLAGS = ['-O3']
if USE_FAST_MATH:
    CUDA_FLAGS.append('--use_fast_math')

# User can override via TORCH_CUDA_ARCH_LIST (standard PyTorch convention)
if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
    CUDA_FLAGS += [
        '-gencode', 'arch=compute_70,code=sm_70',   # V100
        '-gencode', 'arch=compute_75,code=sm_75',   # T4
        '-gencode', 'arch=compute_80,code=sm_80',   # A100
        '-gencode', 'arch=compute_86,code=sm_86',   # RTX 30xx
        '-gencode', 'arch=compute_89,code=sm_89',   # RTX 40xx
        '-gencode', 'arch=compute_90,code=sm_90',   # H100
        '-gencode', 'arch=compute_90,code=compute_90',  # PTX forward-compat
    ]

HIP_FLAGS = ['-O3']
if USE_FAST_MATH:
    HIP_FLAGS.append('-ffast-math')
HIP_FLAGS += [
    '--offload-arch=gfx908',    # MI100
    '--offload-arch=gfx90a',    # MI210
    '--offload-arch=gfx942',    # MI300X
    '--offload-arch=gfx1100',   # RX 7900
    '--offload-arch=gfx1102',   # RX 7600 XT (validated)
]

# ── Extension definition ────────────────────────────────────────────────────

ext = CUDAExtension(
    name='rac_cuda_ext',
    sources=['rac_torch.cu'],
    extra_compile_args={
        'cxx': ['-O3', '-std=c++17'],
        'nvcc': HIP_FLAGS if IS_ROCM else CUDA_FLAGS,
    },
    define_macros=[('__HIP__', '1')] if IS_ROCM else [],
)

# ── Package setup ───────────────────────────────────────────────────────────

setup(
    name='rac_torch',
    version='0.2.0',
    description='RAC: Rotation-Accumulate PyTorch Extension — multiply-free compute primitive',
    author='Michael A. Doran Jr.',
    author_email='mike@pinnacle-quantum.com',
    url='https://github.com/Pinnacle-Quantum-Group/RAC',
    license='Proprietary — (c) 2026 Pinnacle Quantum Group',
    py_modules=['rac_torch'],
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.9',
    install_requires=['torch>=2.0'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
)
