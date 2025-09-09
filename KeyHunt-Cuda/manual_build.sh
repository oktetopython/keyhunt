#!/bin/bash

# Manual build script to bypass make timestamp issues
set -euo pipefail

# Resolve project root (supports WSL and native Linux)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR"

# Detect CUDA path if not provided
: "${CUDA_PATH:=${CUDA:-}}"
if [[ -z "${CUDA_PATH}" ]]; then
  for c in /usr/local/cuda /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda-12.4 /usr/local/cuda-12.3 /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda-11.8 /usr/local/cuda-11.7; do
    if [[ -d "$c" && -x "$c/bin/nvcc" ]]; then CUDA_PATH="$c"; break; fi
  done
fi
if [[ -z "${CUDA_PATH}" ]]; then
  NVCC_PATH=$(command -v nvcc || true)
  if [[ -n "${NVCC_PATH}" ]]; then CUDA_PATH=$(dirname "$(dirname "${NVCC_PATH}")"); fi
fi
if [[ -z "${CUDA_PATH}" ]]; then
  echo "ERROR: CUDA not found. Please export CUDA_PATH." >&2
  exit 1
fi
if [[ ! -d "${CUDA_PATH}/lib64" ]]; then
  echo "WARNING: ${CUDA_PATH}/lib64 not found; cudart link may fail" >&2
fi

# Create obj directories if they don't exist
mkdir -p obj obj/GPU obj/hash

# Auto-detect GPU compute capability (CCAP)
# NEW: Auto-detect CCAP via nvidia-smi, allow parameter override, default to 86
CCAP="${1:-}"
if [[ -z "${CCAP}" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)
    if [[ -n "${cap:-}" ]]; then
      CCAP=$(echo "$cap" | tr -d ' .' )
    fi
  fi
fi
if [[ -z "${CCAP}" ]]; then CCAP=86; fi
if ! [[ "$CCAP" =~ ^[0-9]+$ ]]; then
  echo "WARNING: Invalid CCAP '$CCAP', fallback to 86" >&2
  CCAP=86
fi
echo "Using CUDA_PATH=${CUDA_PATH}, target sm_${CCAP}"

# NVCC detection
NVCC_BIN="${CUDA_PATH}/bin/nvcc"
if ! [[ -x "$NVCC_BIN" ]]; then
  NVCC_BIN=$(command -v nvcc || true)
fi
if [[ -z "${NVCC_BIN}" ]]; then
  echo "ERROR: nvcc not found; required to compile GPU/GPUEngine.cu" >&2
  exit 2
fi

# Common compile flags
CXXFLAGS=( -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I"${CUDA_PATH}/include" )
GENCODE_ARGS=( -gencode "arch=compute_${CCAP},code=sm_${CCAP}" )

# Compile all source files manually
echo "Compiling source files..."

# Host C++ sources
g++ "${CXXFLAGS[@]}" -o obj/Base58.o -c Base58.cpp
g++ "${CXXFLAGS[@]}" -o obj/IntGroup.o -c IntGroup.cpp
g++ "${CXXFLAGS[@]}" -o obj/Main.o -c Main.cpp
g++ "${CXXFLAGS[@]}" -o obj/Bloom.o -c Bloom.cpp
g++ "${CXXFLAGS[@]}" -o obj/Random.o -c Random.cpp
g++ "${CXXFLAGS[@]}" -o obj/Timer.o -c Timer.cpp
g++ "${CXXFLAGS[@]}" -o obj/Int.o -c Int.cpp
g++ "${CXXFLAGS[@]}" -o obj/IntMod.o -c IntMod.cpp
g++ "${CXXFLAGS[@]}" -o obj/Point.o -c Point.cpp
g++ "${CXXFLAGS[@]}" -o obj/SECP256K1.o -c SECP256K1.cpp
g++ "${CXXFLAGS[@]}" -o obj/KeyHunt.o -c KeyHunt.cpp
g++ "${CXXFLAGS[@]}" -o obj/GPU/GPUGenerate.o -c GPU/GPUGenerate.cpp
g++ "${CXXFLAGS[@]}" -o obj/hash/ripemd160.o -c hash/ripemd160.cpp
g++ "${CXXFLAGS[@]}" -o obj/hash/sha256.o -c hash/sha256.cpp
g++ "${CXXFLAGS[@]}" -o obj/hash/sha512.o -c hash/sha512.cpp
g++ "${CXXFLAGS[@]}" -o obj/hash/ripemd160_sse.o -c hash/ripemd160_sse.cpp
g++ "${CXXFLAGS[@]}" -o obj/hash/sha256_sse.o -c hash/sha256_sse.cpp
g++ "${CXXFLAGS[@]}" -o obj/hash/keccak160.o -c hash/keccak160.cpp
g++ "${CXXFLAGS[@]}" -o obj/GmpUtil.o -c GmpUtil.cpp
g++ "${CXXFLAGS[@]}" -o obj/CmdParse.o -c CmdParse.cpp

# Device CUDA source (match Makefile settings)
"${NVCC_BIN}" -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin g++ -m64 -O2 -I"${CUDA_PATH}/include" "${GENCODE_ARGS[@]}" \
  -o obj/GPU/GPUEngine.o -c GPU/GPUEngine.cu

# Link all object files
echo "Linking object files..."
# Prefer glob if it expands; otherwise use find to list .o files
set +e
GLOB_LIST=( obj/*.o obj/GPU/*.o obj/hash/*.o )
EXPANDED_LIST=()
for f in "${GLOB_LIST[@]}"; do
  if [[ -e "$f" ]]; then EXPANDED_LIST+=("$f"); fi

done
set -e
if [[ ${#EXPANDED_LIST[@]} -eq 0 ]]; then
  mapfile -t EXPANDED_LIST < <(find obj -type f -name '*.o' -print)
fi
if [[ ${#EXPANDED_LIST[@]} -eq 0 ]]; then
  echo "ERROR: No object files found under obj/; compile step may have failed" >&2
  exit 3
fi

# Build output name reflecting min arch if provided
OUT=keyhunt-cuda
if [[ -n "${CCAP:-}" ]]; then OUT+="$CCAP"; fi

# Link with GMP and CUDA runtime
LDFLAGS=( -lpthread -L"${CUDA_PATH}/lib64" -lcudart -lgmp )

g++ "${EXPANDED_LIST[@]}" "${LDFLAGS[@]}" -o "$OUT"

echo "Build complete! Checking binary..."
ls -la "$OUT" 2>/dev/null || echo "Binary not found"

# Show version if supported
if [[ -x "$OUT" ]]; then
  "$OUT" --version || true
fi