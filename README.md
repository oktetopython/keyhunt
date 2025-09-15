# Keyhunt-CUDA Scientific Research System

GPU-accelerated Bitcoin private key search system for Bitcoin puzzle challenges with scientific validation.

## Project Structure

```
src/KeyhuntCore/           # Main application source
├── ecc/                   # Elliptic curve cryptography kernels
├── scan/                  # Private key range scanning framework
├── compare/               # Address generation and comparison
├── utils/                 # Utility functions and helpers
├── models/                # Data models and entities
├── gpu/                   # Multi-GPU coordination
└── cli/                   # Command-line interface

tests/                     # Test-driven development
├── contract/              # API contract tests
├── integration/           # Integration tests
├── validation/            # Scientific validation tests
└── unit/                  # Unit tests

data/                      # Runtime data and configuration
├── config/                # Configuration files
├── checkpoint/            # Checkpoint data for resume
├── logs/                  # Experimental logs
└── results/               # Search results and reports

docs/                      # Documentation
├── api/                   # API documentation
├── architecture/          # System architecture
├── performance/           # Performance analysis
└── user/                  # User guides
```

## Implementation Phases

- **Phase 1**: ECC kernel cleanup and validation (基于CudaBrainSecp)
- **Phase 2**: Scanning framework foundation (参考BitCrack)
- **Phase 3**: Address generation and comparison (融合BitCrack比对逻辑)
- **Phase 4**: Performance optimization
- **Phase 5**: Multi-GPU support
- **Phase 6**: Multi-target and batch processing
- **Phase 7**: Validation and documentation
- **Phase 8**: Integration testing and source code fusion validation

## Requirements

- NVIDIA GPU with Compute Capability ≥ 7.5 (Turing architecture or newer)
- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler
- libsecp256k1-dev for CPU validation reference

## Performance Targets

- **Turing Architecture**: >1000M keys/s
- **Hopper Architecture**: >4000M keys/s
- **GPU Utilization**: >90%
- **Validation Precision**: <1e-10 relative error

## Getting Started

See `docs/user/quickstart.md` for detailed setup and usage instructions.

## Scientific Validation

All GPU computations are validated against CPU reference implementations using libsecp256k1 to ensure scientific accuracy and reproducible results.