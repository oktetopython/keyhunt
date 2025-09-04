# Unified ECC Interface

A unified, configuration-driven elliptic curve cryptography interface that provides seamless integration between different ECC implementations (gECC, KeyHunt, LightweightECC).

## Features

- **Unified API**: Single interface for all ECC operations
- **Configuration-Driven**: Runtime backend selection and optimization
- **Multiple Backends**: Support for CPU and GPU implementations
- **Batch Operations**: Efficient batch processing capabilities
- **Mathematical Library**: Comprehensive math utilities
- **Performance Monitoring**: Built-in performance tracking
- **Security Features**: Configurable security settings

## Architecture

```
Unified ECC Interface
├── Core Components
│   ├── IECCOperations (Abstract Interface)
│   ├── ECCFactory (Backend Factory)
│   ├── ConfigManager (Configuration System)
│   └── Math Library (BigInt, Number Theory, etc.)
├── Backends
│   ├── LightweightECC (CPU Optimized)
│   ├── gECC (CUDA Implementation)
│   └── KeyHunt (Hybrid CPU/GPU)
└── Utilities
    ├── Configuration Builder
    ├── Performance Monitor
    └── Memory Management
```

## Quick Start

### Basic Usage

```cpp
#include "unified_ecc/include/ecc_interface.h"

using namespace UnifiedECC;

// Configure and create ECC instance
ECCConfig config;
config.backend = ECCBackend::AUTO; // Auto-select best backend
config.enable_gpu = true;

auto ecc = ECCFactory::create(config);
ecc->initialize(Curves::secp256k1);

// Perform operations
ECPoint G = ecc->get_generator();
UInt256 scalar(42, 0, 0, 0);
ECPoint result = ecc->scalar_mul(scalar, G);
```

### Configuration Management

```cpp
#include "unified_ecc/include/config_manager.h"

// Use preset configurations
configure_for_performance();  // For maximum speed
configure_for_security();     // For security-focused
configure_for_development();  // For development/debugging

// Or build custom configuration
ConfigBuilder()
    .backend("cuda_gecc")
    .enable_gpu(true)
    .batch_size(2048)
    .enable_performance_monitoring(true)
    .build();
```

### Batch Operations

```cpp
// Prepare batch data
std::vector<UInt256> scalars = {scalar1, scalar2, scalar3};
std::vector<ECPoint> points = {point1, point2, point3};

// Perform batch scalar multiplication
std::vector<ECPoint> results = ecc->batch_scalar_mul(scalars, points);
```

## Supported Backends

| Backend | Type | GPU Support | Performance | Security |
|---------|------|-------------|-------------|----------|
| LightweightECC | CPU | No | High | High |
| gECC | CUDA | Yes | Very High | Medium |
| KeyHunt | Hybrid | Yes | High | Medium |
| CPU_Generic | CPU | No | Medium | High |

## Configuration Options

### Backend Selection
```cpp
config.backend = ECCBackend::AUTO;           // Auto-select
config.backend = ECCBackend::CPU_OPTIMIZED;  // LightweightECC
config.backend = ECCBackend::CUDA_GECC;      // gECC
config.backend = ECCBackend::CUDA_KEYHUNT;   // KeyHunt
```

### Performance Settings
```cpp
config.enable_batch_ops = true;
config.batch_size = 1024;
config.enable_multithreading = true;
config.thread_count = 8;
```

### GPU Settings
```cpp
config.enable_gpu = true;
config.gpu_device_id = 0;
config.gpu_memory_limit = 1024; // MB
```

### Security Settings
```cpp
config.enable_secure_memory = true;
config.random_seed = 0; // 0 = system random
```

## Mathematical Operations

### Big Integer Arithmetic
```cpp
#include "unified_ecc/include/math_common.h"

using namespace UnifiedECC::Math;

// Big integer operations
uint64_t limbs_a[4] = {1, 2, 3, 4};
uint64_t limbs_b[4] = {5, 6, 7, 8};
uint64_t result[4];

BigIntUtils::add(result, limbs_a, limbs_b, 4);
BigIntUtils::mul(result, limbs_a, limbs_b, 4);
```

### Number Theory
```cpp
// GCD, modular inverse, primality testing
uint64_t gcd = NumberTheory::gcd(48, 18); // = 6
uint64_t inv = NumberTheory::mod_inverse(7, 13); // = 2
bool is_prime = NumberTheory::is_prime_fermat(17); // = true
```

### Bit Operations
```cpp
// Bit manipulation utilities
bool bit = BitUtils::get_bit(limbs, 42);
BitUtils::set_bit(limbs, 42, true);
int bit_length = BitUtils::bit_length(limbs, 4);
```

## Building and Testing

### Prerequisites
- C++17 compatible compiler
- CMake 3.10+ (optional)
- CUDA toolkit (for GPU backends)

### Build Commands
```bash
# Build all components
make all

# Build specific tests
make test_unified_ecc    # Main ECC interface test
make test_math           # Math library test
make run_config_test     # Configuration system test

# Run tests
./test_unified_ecc
./test_math_common
./test_config
```

### Integration with Existing Code

To integrate the unified interface with existing ECC code:

1. **Replace direct backend calls** with unified interface calls
2. **Use configuration system** for backend selection
3. **Leverage batch operations** for improved performance
4. **Utilize performance monitoring** for optimization

### Example Migration

**Before (direct LightweightECC usage):**
```cpp
LightweightECC::ECOp ec_op;
LightweightECC::CurveParams params = {/*...*/};
ec_op.init(params);
LightweightECC::Point result = ec_op.scalar_mul(k, P);
```

**After (unified interface):**
```cpp
auto ecc = ECCFactory::create(ECCConfig{});
ecc->initialize(Curves::secp256k1);
ECPoint result = ecc->scalar_mul(k, P);
```

## Performance Optimization

### Backend Selection Guidelines

- **Use AUTO** for most applications (automatic optimization)
- **Use CPU_OPTIMIZED** for security-critical applications
- **Use CUDA backends** for high-performance computing
- **Use CPU_GENERIC** for development and testing

### Batch Processing

Enable batch operations for multiple scalar multiplications:
```cpp
config.enable_batch_ops = true;
config.batch_size = 1024; // Adjust based on workload
```

### Memory Management

Use secure memory for sensitive operations:
```cpp
config.enable_secure_memory = true;
// Memory will be securely zeroed after use
```

## Security Considerations

### Memory Security
- Enable `enable_secure_memory` for sensitive key operations
- Memory is automatically zeroed after use
- Prevents key material from remaining in memory

### Backend Security
- CPU backends provide better side-channel resistance
- GPU backends offer higher performance but may have side-channel vulnerabilities
- Use CPU_OPTIMIZED for cryptographic key operations

### Random Number Generation
- Configure random seed for reproducible testing
- Use system entropy for production keys
- Validate random number quality for cryptographic use

## Troubleshooting

### Common Issues

1. **Backend not available**
   - Check if required libraries are installed
   - Verify GPU availability for CUDA backends
   - Use CPU backends as fallback

2. **Configuration validation errors**
   - Check configuration values against supported ranges
   - Use `ConfigManager::validate()` to identify issues
   - Review validation error messages

3. **Performance issues**
   - Enable batch operations for multiple operations
   - Adjust batch size based on workload
   - Consider GPU acceleration if available

### Debug Information

Enable debug logging:
```cpp
config.enable_performance_monitoring = true;
config.log_level = "debug";
```

## API Reference

### Core Classes

- `IECCOperations`: Abstract ECC operations interface
- `ECCFactory`: Backend factory and management
- `ConfigManager`: Configuration management system
- `ConfigBuilder`: Fluent configuration builder

### Mathematical Classes

- `BigIntUtils`: Big integer arithmetic utilities
- `NumberTheory`: Number theory functions
- `BitUtils`: Bit manipulation utilities
- `PerformanceMonitor`: Performance measurement tools

### Data Types

- `UInt256`: 256-bit unsigned integer
- `ECPoint`: Elliptic curve point
- `ECCParams`: Elliptic curve parameters
- `ECCConfig`: ECC configuration structure

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility
5. Validate performance impact

## License

This project integrates multiple ECC implementations with their respective licenses:
- Unified Interface: MIT License
- LightweightECC: BSD License
- gECC: Custom license
- KeyHunt: Custom license

## Roadmap

### Phase 3: Performance Optimization (Current)
- [x] Implement advanced scalar multiplication algorithms
- [ ] Optimize CUDA kernels
- [ ] Add comprehensive performance benchmarks

### Phase 4: Quality Assurance
- [ ] Complete unit test coverage
- [ ] API documentation generation
- [ ] Continuous integration setup

### Future Enhancements
- Additional curve support (Ed25519, Curve25519)
- Hardware acceleration (Intel IPP, ARM Cryptography Extensions)
- Distributed computing support
- Formal verification of critical operations