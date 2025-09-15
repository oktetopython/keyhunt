#!/bin/bash

# Build script for Keyhunt-CUDA
# Handles dependencies and configuration validation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Keyhunt-CUDA Build System${NC}"
echo "=========================="

# Check system requirements
echo -e "${YELLOW}Checking system requirements...${NC}"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA compiler (nvcc) not found${NC}"
    echo "Please install CUDA Toolkit 11.0 or later"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "Found CUDA version: $CUDA_VERSION"

# Check CMake version
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake not found${NC}"
    echo "Please install CMake 3.18 or later"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | grep "cmake version" | sed -n 's/cmake version \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "Found CMake version: $CMAKE_VERSION"

# Check for libsecp256k1
if ! pkg-config --exists libsecp256k1 2>/dev/null; then
    echo -e "${YELLOW}Warning: libsecp256k1-dev may not be installed${NC}"
    echo "For CPU validation reference, install with:"
    echo "  sudo apt-get install libsecp256k1-dev"
    echo "Note: System will continue with validation disabled"
fi

# Check for local GoogleTest
GOOGLETEST_PATH="specs/001-keyhunt-cuda-puzzle/src/googleTest"
if [ -d "$GOOGLETEST_PATH" ]; then
    echo "Found local GoogleTest at: $GOOGLETEST_PATH"
else
    echo -e "${YELLOW}Warning: Local GoogleTest not found${NC}"
    echo "Will attempt to download from internet during build"
fi

# Check for GPU devices
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo "Found $GPU_COUNT CUDA device(s)"
    nvidia-smi -L
else
    echo -e "${YELLOW}Warning: nvidia-smi not found${NC}"
fi

echo -e "${GREEN}System requirements check complete${NC}"
echo ""

# Build configuration
BUILD_TYPE=${1:-Release}
BUILD_DIR="build"
INSTALL_PREFIX=${2:-install}

echo -e "${YELLOW}Build configuration:${NC}"
echo "  Build type: $BUILD_TYPE"
echo "  Build directory: $BUILD_DIR" 
echo "  Install prefix: $INSTALL_PREFIX"
echo ""

# Create build directory
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DBUILD_TESTS=ON \
    -DENABLE_AGGRESSIVE_OPTIMIZATIONS=ON

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed${NC}"
    exit 1
fi

# Build
echo -e "${YELLOW}Building Keyhunt-CUDA...${NC}"
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo ""

# Show build results
echo -e "${YELLOW}Build results:${NC}"
ls -la bin/

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo "Run './bin/keyhunt --help' to get started"
echo "Run 'make test' to execute the test suite"