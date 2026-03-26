"""
setup.py — RAC PyTorch Extension Build
Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026

Build:
    pip install -e .                    # editable install
    python setup.py install             # standard install
    python setup.py build_ext --inplace # build in place for dev

CUDA:
    CUDA_HOME must be set, or nvcc must be on PATH.

ROCm/HIP:
    Set USE_ROCM=1 or ensure hipcc is on PATH.
    The extension auto-detects ROCm via torch.version.hip.
"""

import os
import sys
import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension, CUDAExtension, CppExtension
)

# ── Backend detection ────────────────────────────────────────────────────────

IS_ROCM = (
    os.environ.get('USE_ROCM', '0') == '1' or
    (hasattr(torch.version, 'hip') and torch.version.hip is not None)
)

print(f"RAC build: {'ROCm/HIP' if IS_ROCM else 'CUDA'} backend")

# ── Compiler flags ────────────────────────────────────────────────────────────

COMMON_FLAGS = [
    '-O3',
    '--use_fast_math' if not IS_ROCM else '-ffast-math',
]

CUDA_FLAGS = COMMON_FLAGS + [
    '-gencode', 'arch=compute_80,code=sm_80',   # A100
    '-gencode', 'arch=compute_86,code=sm_86',   # RTX 30xx
    '-gencode', 'arch=compute_89,code=sm_89',   # RTX 40xx
    '-gencode', 'arch=compute_90,code=sm_90',   # H100
]

HIP_FLAGS = COMMON_FLAGS + [
    '--offload-arch=gfx1102',   # RX 7600 XT (validated)
    '--offload-arch=gfx1100',   # RX 7900
    '--offload-arch=gfx90a',    # MI210
    '--offload-arch=gfx908',    # MI100
]

# ── Extension definition ──────────────────────────────────────────────────────

ext = CUDAExtension(
    name='rac_cuda_ext',
    sources=['rac_torch.cu'],
    extra_compile_args={
        'cxx': ['-O3', '-std=c++17'],
        'nvcc': HIP_FLAGS if IS_ROCM else CUDA_FLAGS,
    },
    define_macros=[('__HIP__', '1')] if IS_ROCM else [],
)

# ── Package setup ─────────────────────────────────────────────────────────────

setup(
    name='rac_torch',
    version='0.1.0',
    description='RAC: Rotation-Accumulate PyTorch Extension — multiply-free compute primitive',
    author='Michael A. Doran Jr.',
    author_email='mike@pinnacle-quantum.com',
    url='https://github.com/Pinnacle-Quantum-Group/RAC',
    license='Proprietary — © 2026 Pinnacle Quantum Group',
    py_modules=['rac_torch'],
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.9',
    install_requires=['torch>=2.0'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
)
