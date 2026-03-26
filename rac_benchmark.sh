#!/bin/bash
# rac_benchmark.sh — Build and run RAC benchmark
# Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026

set -e

TARGET=${1:-cuda}   # cuda | hip | both

CUDA_ARCH=${CUDA_ARCH:-86}   # default: Ampere (RTX 3xxx / A100)
                              # 80 = A100, 86 = RTX 30xx, 89 = RTX 40xx, 90 = H100

HIP_ARCH=${HIP_ARCH:-gfx1100}  # default: RDNA3 (RX 7xxx)
                                 # gfx906=MI50, gfx908=MI100, gfx90a=MI210, gfx1100=RX7900

echo "================================================"
echo " RAC Benchmark Build & Run"
echo " Pinnacle Quantum Group — March 2026"
echo "================================================"

build_cuda() {
    echo ""
    echo "── Building CUDA ────────────────────────────────"
    echo "   Architecture: sm_${CUDA_ARCH}"

    nvcc -O3 \
        -arch=sm_${CUDA_ARCH} \
        --use_fast_math \
        -I. \
        rac_benchmark.cu rac_cuda.cu \
        -lcublas -lnvml \
        -o rac_benchmark_cuda

    echo "   Build: OK → rac_benchmark_cuda"
}

run_cuda() {
    echo ""
    echo "── Running CUDA benchmark ───────────────────────"
    ./rac_benchmark_cuda 2>&1 | tee rac_benchmark_cuda.log
    echo ""
    echo "   Log saved: rac_benchmark_cuda.log"
}

build_hip() {
    echo ""
    echo "── Building HIP/ROCm ────────────────────────────"
    echo "   Architecture: ${HIP_ARCH}"

    hipcc -O3 \
        --offload-arch=${HIP_ARCH} \
        -I. \
        rac_benchmark.cu rac_hip.cpp \
        -lrocblas -D__HIP_PLATFORM_AMD__ \
        -o rac_benchmark_hip

    echo "   Build: OK → rac_benchmark_hip"
}

run_hip() {
    echo ""
    echo "── Running HIP benchmark ────────────────────────"
    ./rac_benchmark_hip 2>&1 | tee rac_benchmark_hip.log
    echo ""
    echo "   Log saved: rac_benchmark_hip.log"
}

case $TARGET in
    cuda)
        build_cuda
        run_cuda
        ;;
    hip)
        build_hip
        run_hip
        ;;
    both)
        build_cuda
        build_hip
        run_cuda
        run_hip
        ;;
    build-cuda)
        build_cuda
        ;;
    build-hip)
        build_hip
        ;;
    *)
        echo "Usage: $0 [cuda|hip|both|build-cuda|build-hip]"
        echo "  CUDA_ARCH=86  ./rac_benchmark.sh cuda     # Ampere"
        echo "  CUDA_ARCH=90  ./rac_benchmark.sh cuda     # H100"
        echo "  HIP_ARCH=gfx90a ./rac_benchmark.sh hip    # MI210"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo " Done"
echo "================================================"
