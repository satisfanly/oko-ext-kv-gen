#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/cuda/bin/:$PATH"  

LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-../llama.cpp-s}

CUDA12_ROOT="${CUDA12_ROOT:-/opt/cuda-12.8}"
BUILD_DIR="${LLAMA_CPP_DIR}/build"

NVCC="${CUDA12_ROOT}/bin/nvcc"

if [[ ! -x "${NVCC}" ]]; then
    echo "ERROR: CUDA 12 nvcc not found at:"
    echo "  ${NVCC}"
    echo
    echo "Set CUDA12_ROOT if CUDA 12.8 is installed elsewhere."
    exit 1
fi

echo "Using CUDA toolkit:"
"${NVCC}" --version
echo

# CUDA 12.8 officially supports GCC <= 14.
#
# Prefer GCC 14 if installed. On bleeding-edge distributions such as Arch,
# fall back to the system compiler and tell nvcc to permit the unsupported
# host compiler.
CMAKE_HOST_COMPILER_ARGS=()
CMAKE_CUDA_EXTRA_FLAGS=""

if command -v gcc-14 >/dev/null 2>&1 &&
   command -v g++-14 >/dev/null 2>&1; then

    echo "Using GCC 14 as CUDA host compiler:"
    gcc-14 --version | head -n1

    CMAKE_HOST_COMPILER_ARGS=(
        -DCMAKE_C_COMPILER=gcc-14
        -DCMAKE_CXX_COMPILER=g++-14
        -DCMAKE_CUDA_HOST_COMPILER=g++-14
    )
else
    echo "WARNING: GCC 14 not found."
    echo "Using system compiler with nvcc --allow-unsupported-compiler."
    echo

    CMAKE_CUDA_EXTRA_FLAGS="--allow-unsupported-compiler"
fi

export PATH="${CUDA12_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA12_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Avoid reusing a CMake cache created with CUDA 13 / another compiler.
rm -rf "${BUILD_DIR}"

cmake -S "${LLAMA_CPP_DIR}" -B "${BUILD_DIR}" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    -DCUDAToolkit_ROOT="${CUDA12_ROOT}" \
    '-DCMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual' \
    -DCMAKE_CUDA_FLAGS="${CMAKE_CUDA_EXTRA_FLAGS}" \
    "${CMAKE_HOST_COMPILER_ARGS[@]}"

cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo
echo "Built GGML CUDA shared backend under:"
echo "  ${BUILD_DIR}"
echo

find "${BUILD_DIR}" -maxdepth 4 -type f \
    \( \
        -name 'libggml.so' \
        -o -name 'libggml-base.so' \
        -o -name 'libggml-cpu.so' \
        -o -name 'libggml-cuda.so' \
    \) \
    -print
