# Source Code Fusion Architecture Setup

## Overview

This directory contains documentation for the Source Code Fusion Architecture (源码融合法) used in Keyhunt-CUDA. This approach follows the "Standing on Giants' Shoulders" intelligent fusion strategy by systematically extracting and integrating proven components from high-quality existing implementations.

## Source Libraries

### CudaBrainSecp Source Library (`src/CudaBrainSecp/`)
**Purpose**: Provides clean, maintainable, high-performance ECC kernel implementations
**Key Components**:
- `CPU/SECP256K1.cpp/.h` - CPU reference implementation
- `CPU/Point.cpp/.h` - Elliptic curve point operations
- `CPU/Int.cpp/.h` - 256-bit integer arithmetic
- `GPU/GPUSecp.cu/.h` - CUDA ECC kernels
- `GPU/GPUMath.h` - GPU mathematical operations

**Extraction Strategy**: Extract core ECC functions like `scalar_mul`, `point_add`, `point_double` for integration into KeyhuntCore ECC module.

### BitCrack Source Library (`src/BitCrack/`)
**Purpose**: Provides mature multi-GPU scanning framework, range scanning, and address comparison logic
**Key Components**:
- `CudaKeySearchDevice/` - GPU device management and kernel execution
- `KeyFinderLib/` - Core scanning algorithms and range management
- `AddressUtil/` - Bitcoin address generation and Base58 encoding
- `CryptoUtil/` - Hash functions (SHA256, RIPEMD160)
- `DeviceManager.cpp` - Multi-GPU coordination framework

**Extraction Strategy**: Borrow range scanning concepts, GPU resource optimization, and multi-GPU framework design for KeyhuntCore scanning module.

## Fusion Architecture Design

```
KeyhuntCore/ (Custom framework skeleton)
├── ecc/          # ECC kernels extracted from CudaBrainSecp
│   ├── secp256k1.cu/.h      # Main ECC interface
│   ├── secp256k1_math.cu    # 256-bit modular arithmetic (extracted+optimized)
│   ├── secp256k1_point.cu   # Elliptic curve point operations (extracted+optimized)
│   └── secp256k1_cpu.cpp    # CPU reference implementation
├── scan/         # Scanning framework inspired by BitCrack design
│   ├── scanner.cu/.h        # Range scanning logic (borrowed+restructured)
│   └── checkpoint.cpp       # Checkpoint system (custom)
├── compare/      # Address comparison module fusing BitCrack logic
│   ├── hash.cu/.h          # Address generation pipeline (extracted+optimized)
│   └── bloom_filter.cu     # GPU Bloom Filter (borrowed+improved)
└── utils/        # Custom utility modules
    ├── logger.cpp          # Logging system
    ├── timer.cpp           # Performance timing
    └── config.cpp          # Configuration management
```

## Execution Method

### 1. Code Archaeology Analysis
Deep analysis of source code structures to identify core functions:
- Map function dependencies and call graphs
- Identify performance-critical code paths
- Document mathematical algorithms and implementations

### 2. Precision Extraction
Extract key functions with scientific validation:
- Copy relevant `.cu` files to `KeyhuntCore/` modules
- Preserve mathematical accuracy and performance characteristics
- Maintain original algorithm logic while adapting interfaces

### 3. Code Migration
Systematic file copying and adaptation:
- Extract `.cu`/`.cpp` files to appropriate KeyhuntCore modules
- Refactor headers and namespaces for CPU/GPU consistency
- Adapt build system integration with CMake

### 4. Interface Reconstruction
Unified API design across all modules:
- Consistent error handling and return codes
- Unified logging and debugging interfaces
- Standard configuration and parameter passing

### 5. Scientific Validation
Use libsecp256k1 as CPU reference for bit-level consistency verification:
- CPU/GPU result comparison with <1e-10 precision tolerance
- Mathematical property validation (group laws, field operations)
- Edge case and boundary condition testing

## Setup Status

### T006 Completion Status ✅

**Source Fusion Directories Setup**:
- [x] Created symbolic links to reference implementations
  - `src/CudaBrainSecp/` → `specs/001-keyhunt-cuda-puzzle/src/CudaBrainSecp/`
  - `src/BitCrack/` → `specs/001-keyhunt-cuda-puzzle/src/BitCrack/`
- [x] Verified access to all reference source files
- [x] Created documentation framework for fusion analysis
- [x] Established read-only access pattern as specified in CLAUDE.md

**Key Benefits**:
1. **Easy Access**: Reference implementations available directly in `src/` for analysis
2. **Read-Only Safety**: Symbolic links prevent accidental modification of reference code
3. **Clear Separation**: KeyhuntCore development separate from reference analysis
4. **Scientific Rigor**: Structured approach to component extraction and validation

## Next Steps (Phase 3.2: Contract Tests)

The next phase requires implementing contract tests (T007-T015) before any core implementation begins. This follows the Test-Driven Development approach with scientific validation requirements.

**CRITICAL**: All tests must be written and must FAIL before ANY implementation begins.

## Usage Guidelines

### For Code Analysis
```bash
# Analyze CudaBrainSecp ECC implementation
ls src/CudaBrainSecp/CPU/     # CPU reference algorithms
ls src/CudaBrainSecp/GPU/     # CUDA kernel implementations

# Analyze BitCrack scanning framework
ls src/BitCrack/CudaKeySearchDevice/    # GPU device management
ls src/BitCrack/KeyFinderLib/           # Core scanning logic
```

### For Extraction Work
- **DO NOT** modify files in `src/CudaBrainSecp/` or `src/BitCrack/`
- **DO** copy relevant code to `src/KeyhuntCore/` modules
- **DO** document extraction decisions and modifications
- **DO** validate extracted code against CPU references

### For Scientific Validation
- All extracted GPU code must pass CPU/GPU consistency testing
- Mathematical properties must be validated against libsecp256k1
- Performance benchmarks must meet or exceed reference implementations

## Architecture Validation

This setup enables the Source Code Fusion approach by:
1. ✅ Providing easy access to proven implementations
2. ✅ Maintaining clear separation between reference and implementation
3. ✅ Enabling systematic analysis and extraction workflow
4. ✅ Supporting scientific validation methodology
5. ✅ Preserving original implementations for benchmarking

The source fusion directories are now ready for Phase 3.2 contract test implementation.