# Keyhunt-CUDA Development Guide

This is a GPU-accelerated Bitcoin private key search system for Bitcoin puzzle challenges with scientific validation requirements.

## Build System

### Prerequisites
- NVIDIA GPU with Compute Capability ≥ 7.5 (Turing architecture or newer)
- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler
- libsecp256k1-dev (for CPU validation reference)
- NCCL (optional, for multi-GPU support)

### Build Commands
```bash
# Quick build using provided script
./build.sh

# Manual CMake build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DENABLE_AGGRESSIVE_OPTIMIZATIONS=ON
make -j$(nproc)

# Run tests
make test
# or
ctest

# Run specific test binary
./bin/setup_test
```

### Build Configuration
- **Release builds require libsecp256k1-dev** for scientific validation
- CUDA architectures: 75;80;86;89;90 (Turing and newer)
- Aggressive optimizations enabled by default (`--maxrregcount=64 --use_fast_math`)
- Local GoogleTest integration from `specs/001-keyhunt-cuda-puzzle/src/googleTest`

## Architecture Overview

### Core Components
- **src/KeyhuntCore/**: Main application source
  - **ecc/**: Elliptic curve cryptography CUDA kernels
  - **scan/**: Private key range scanning framework
  - **compare/**: Address generation and comparison logic
  - **gpu/**: Multi-GPU coordination and NCCL integration
  - **models/**: Data models (PrivateKeyRange, TargetAddress, CheckpointData)
  - **cli/**: Command-line interface implementation
  - **utils/**: Utility functions and helpers

### External Dependencies
- **src/BitCrack@**: BitCrack integration (symlink)
- **src/CudaBrainSecp@**: CudaBrainSecp integration (symlink)

### Test Structure
- **tests/contract/**: API contract tests
- **tests/integration/**: Integration tests
- **tests/validation/**: Scientific validation tests
- **tests/unit/**: Unit tests

### Data Management
- **data/config/**: Configuration files
- **data/checkpoint/**: Checkpoint data for resume functionality
- **data/logs/**: Experimental logs
- **data/results/**: Search results and reports

## Development Workflow

### Scientific Validation Requirements
- All GPU computations validated against CPU reference (libsecp256k1)
- Precision threshold: <1e-10 relative error
- Performance targets: >1000M keys/s (Turing), >4000M keys/s (Hopper)
- GPU utilization target: >90%

### Implementation Phases
1. ECC kernel cleanup and validation (基于CudaBrainSecp)
2. Scanning framework foundation (参考BitCrack)
3. Address generation and comparison (融合BitCrack比对逻辑)
4. Performance optimization
5. Multi-GPU support
6. Multi-target and batch processing
7. Validation and documentation
8. Integration testing and source code fusion validation

### Multi-GPU Configuration
- NCCL-based coordination when available
- Fallback to CUDA streams for basic multi-GPU
- Maximum supported devices: 8 GPUs
- Configuration controlled via CMake options

## Key Files and Locations

### Build Configuration
- `CMakeLists.txt`: Main build configuration with CUDA architecture targets
- `build.sh`: Automated build script with dependency checking
- `src/KeyhuntCore/config.h.in`: Configuration template

### Main Entry Point
- `src/KeyhuntCore/main.cpp`: CLI entry point with command structure

### Test Setup
- `tests/setup_test.cpp`: Basic test runner
- Local GoogleTest at `specs/001-keyhunt-cuda-puzzle/src/googleTest`

### Documentation
- `README.md`: Project overview and structure
- `T002_COMPLETION_SUMMARY.md`: Recent build system improvements
- `docs/source-fusion/README.md`: Source fusion documentation

## Performance and Optimization

### CUDA Optimization Flags
- `--use_fast_math`: Enabled for aggressive optimizations
- `--maxrregcount=64`: Register count limitation
- Architecture-specific compilation for optimal performance

### Validation System
- CPU/GPU consistency validation using libsecp256k1
- Scientific precision requirements enforced
- Performance benchmarking integrated into test suite

## Development Notes

### Code Integration Strategy
The project integrates code from BitCrack and CudaBrainSecp through symlinks, requiring careful coordination of ECC implementations and scanning logic.

### Multi-Architecture Support
Build system automatically detects and compiles for available CUDA architectures, with fallback support for systems without NCCL.