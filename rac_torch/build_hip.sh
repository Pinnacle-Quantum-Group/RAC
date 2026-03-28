#!/bin/bash
# build_hip.sh — Build RAC PyTorch extension for ROCm/HIP
# Compiles kernels with hipcc, bindings with g++, links together.
#
# Usage: cd rac_torch && bash build_hip.sh

set -e

TORCH_DIR=$(python3 -c "import torch; print(torch.__path__[0])")
PY_EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
HIPCC=${HIPCC:-/usr/bin/hipcc}
GFX=${GFX_ARCH:-gfx1100}  # gfx1100 for RX 7900, gfx1102 for RX 7600 XT

echo "RAC PyTorch Extension — HIP Build"
echo "  HIPCC:    ${HIPCC}"
echo "  Arch:     ${GFX}"
echo "  Torch:    ${TORCH_DIR}"
echo "  Output:   rac_cuda_ext${PY_EXT}"
echo ""

# Step 1: Compile HIP kernels → .o (no PyTorch headers!)
echo "── Compiling HIP kernels..."
${HIPCC} -O3 \
    --offload-arch=${GFX} \
    -fPIC -c \
    rac_kernels.hip \
    -o rac_kernels.o

echo "   rac_kernels.o OK"

# Step 2: Compile pybind11 bindings → .o (g++, no hipcc needed)
echo "── Compiling PyTorch bindings..."
g++ -O3 -std=c++17 -fPIC -DUSE_ROCM -D__HIP_PLATFORM_AMD__ -c \
    -I${TORCH_DIR}/include \
    -I${TORCH_DIR}/include/torch/csrc/api/include \
    -I$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
    -I/usr/include \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    rac_torch_bind.cpp \
    -o rac_torch_bind.o

echo "   rac_torch_bind.o OK"

# Step 3: Link everything with hipcc (handles device code + host code)
echo "── Linking (hipcc handles device + host)..."
${HIPCC} --offload-arch=${GFX} \
    -shared \
    rac_kernels.o rac_torch_bind.o \
    -L${TORCH_DIR}/lib \
    -L/opt/rocm-7.1.1/lib \
    -ltorch -ltorch_hip -lc10 -lc10_hip -ltorch_python \
    -Wl,-rpath,${TORCH_DIR}/lib \
    -Wl,-rpath,/opt/rocm-7.1.1/lib \
    -o rac_cuda_ext${PY_EXT}

echo "   rac_cuda_ext${PY_EXT} OK"
echo ""
echo "Done. Test: python3 -c 'import rac_cuda_ext; print(\"RAC extension loaded!\")'"
