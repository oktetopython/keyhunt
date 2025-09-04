# KeyHunt-Cuda

A high-performance CUDA-accelerated Bitcoin private key search tool optimized for modern NVIDIA GPUs.

## 🚀 Features

- **GPU Acceleration**: Harness the power of NVIDIA GPUs for Bitcoin private key search
- **Universal GPU Support**: Compatible with RTX 20xx, 30xx, and 40xx series (Compute Capability 7.5-9.0)
- **Multiple Search Modes**: Address search, X-point search, and Ethereum address search
- **High Performance**: Achieve 1000+ Mk/s on modern GPUs
- **Optimized Code**: Zero compilation warnings, clean architecture, minimal code duplication
- **Easy Setup**: Automated GPU detection and optimal compilation recommendations

## 📋 Requirements

### Hardware
- **NVIDIA GPU**: GTX 16xx series or newer (Compute Capability 7.5+)
- **RAM**: 4GB+ recommended
- **Storage**: 1GB free space

### Software
- **CUDA Toolkit**: 11.0 or newer
- **GCC/G++**: 7.5 or newer
- **Make**: GNU Make
- **Git**: For cloning the repository

### Supported Operating Systems
- **Linux**: Ubuntu 18.04+, CentOS 7+, other distributions
- **Windows**: Windows 10/11 with WSL2 or native CUDA
- **macOS**: Limited support (CPU only)

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-repo/KeyHunt-Cuda.git
cd KeyHunt-Cuda
```

### 2. Install Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential libgmp-dev
```

#### CentOS/RHEL
```bash
sudo yum groupinstall "Development Tools"
sudo yum install gmp-devel
```

### 3. Install CUDA Toolkit
Download and install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## 🔧 Compilation

### Quick Start (Recommended)
```bash
# Auto-detect your GPU and get compilation recommendations
./scripts/detect_gpu.sh

# Follow the recommended compilation command from the script output
```

### Manual Compilation Options

#### Option 1: Single GPU Architecture (Faster compilation, optimized)
```bash
# For RTX 20xx/GTX 16xx series
make clean && make gpu=1 CCAP=75 all

# For RTX 30xx series
make clean && make gpu=1 CCAP=86 all

# For RTX 40xx series
make clean && make gpu=1 CCAP=90 all
```

#### Option 2: Multi-GPU Architecture (Universal compatibility)
```bash
# Works on all supported GPUs (RTX 20xx-40xx)
make clean && make gpu=1 MULTI_GPU=1 all
```

#### Debug Build
```bash
make clean && make gpu=1 debug=1 CCAP=86 all
```

### GPU Compatibility Guide

| GPU Series | Compute Capability | Recommended Build | Typical Performance |
|------------|-------------------|-------------------|-------------------|
| GTX 16xx | 7.5 | `CCAP=75` | 800-1000 Mk/s |
| RTX 20xx | 7.5 | `CCAP=75` | 1200-1500 Mk/s |
| RTX 30xx | 8.6 | `CCAP=86` | 1500-2200 Mk/s |
| RTX 40xx | 8.9/9.0 | `CCAP=90` | 2000-3500 Mk/s |

## 🎯 Usage

### Basic Usage
```bash
# Search for a specific Bitcoin address
./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range START:END TARGET_ADDRESS

# Example: Search Bitcoin puzzle 40
./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range e9ae493300:e9ae493400 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv
```

### Command Line Options

#### Required Parameters
- `-g`: Enable GPU mode
- `--gpui N`: GPU device index (0 for first GPU)
- `--mode MODE`: Search mode (ADDRESS, XPOINT, ETH)
- `--coin TYPE`: Cryptocurrency type (BTC, ETH)
- `--range START:END`: Search range in hexadecimal
- `TARGET`: Target address or public key

#### Optional Parameters
- `--comp`: Search compressed addresses only
- `--uncomp`: Search uncompressed addresses only
- `--both`: Search both compressed and uncompressed
- `-t N`: Number of CPU threads (default: auto)
- `--gpugridsize NxM`: Custom GPU grid size
- `--rkey N`: Random key mode
- `--maxfound N`: Maximum results to find
- `-o FILE`: Output file (default: Found.txt)

### Search Modes

#### 1. Address Search (MODE: ADDRESS)
Search for Bitcoin addresses:
```bash
./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --comp --range 1:FFFFFFFF 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2
```

#### 2. X-Point Search (MODE: XPOINT)
Search for public key X-coordinates:
```bash
./KeyHunt -g --gpui 0 --mode XPOINT --coin BTC --comp --range 1:FFFFFFFF 50929b74c1a04954b78b4b6035e97a5e078a5a0f28ec96d547bfee9ace803ac0
```

#### 3. Ethereum Search (MODE: ETH)
Search for Ethereum addresses:
```bash
./KeyHunt -g --gpui 0 --mode ADDRESS --coin ETH --range 1:FFFFFFFF 0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6
```

## 📊 Performance Optimization

### GPU Settings
```bash
# Auto grid size (recommended)
./KeyHunt -g --gpui 0 --gpugridsize -1x128 [other options]

# Custom grid size for fine-tuning
./KeyHunt -g --gpui 0 --gpugridsize 256x256 [other options]
```

### Multi-GPU Setup
```bash
# Use multiple GPUs
./KeyHunt -g --gpui 0,1,2 [other options]
```

### Performance Tips
1. **Use single GPU builds** for maximum performance on specific hardware
2. **Adjust grid size** based on your GPU's SM count
3. **Monitor GPU temperature** and ensure adequate cooling
4. **Use compressed mode** for faster Bitcoin address search
5. **Optimize search ranges** to avoid unnecessary computation

## 📁 Project Structure

```
KeyHunt-Cuda/
├── src/                    # Source code
├── GPU/                    # GPU kernels and CUDA code
├── hash/                   # Hash algorithm implementations
├── tests/                  # Test files and verification scripts
├── debug/                  # Debug utilities
├── scripts/                # Build and utility scripts
├── docs/                   # Documentation
├── examples/               # Usage examples
├── Makefile               # Build configuration
└── README.md              # This file
```

## 🔍 Examples

### Bitcoin Puzzle Solving
```bash
# Puzzle 40 (solved example)
./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --comp --range e9ae493300:e9ae493400 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv

# Expected output: Private key E9AE4933D6
```

### Random Key Search
```bash
# Search with random starting points
./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --comp --rkey 1000000 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2
```

### Range Search
```bash
# Search specific range
./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --comp --range 8000000000000000:FFFFFFFFFFFFFFFF 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2
```

## 🐛 Troubleshooting

### Common Issues

#### Compilation Errors
```bash
# CUDA not found
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# GMP library not found
sudo apt install libgmp-dev  # Ubuntu/Debian
sudo yum install gmp-devel   # CentOS/RHEL
```

#### Runtime Errors
```bash
# No CUDA device found
nvidia-smi  # Check if GPU is detected
sudo nvidia-modprobe  # Load NVIDIA kernel module

# Out of memory
# Reduce grid size or use smaller batch sizes
./KeyHunt -g --gpui 0 --gpugridsize 128x128 [other options]
```

#### Performance Issues
```bash
# Low performance
# 1. Use single GPU build for your architecture
make clean && make gpu=1 CCAP=86 all

# 2. Check GPU utilization
nvidia-smi -l 1

# 3. Adjust grid size
./KeyHunt -g --gpui 0 --gpugridsize 256x256 [other options]
```

## 📚 Documentation

- **[GPU Compatibility Guide](docs/GPU_COMPATIBILITY_GUIDE.md)**: Detailed GPU support information
- **[Code Quality Improvements](docs/CODE_QUALITY_IMPROVEMENTS.md)**: Technical improvements documentation
- **[Build System](docs/BUILD_SYSTEM.md)**: Advanced build configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Users are responsible for complying with all applicable laws and regulations. The authors are not responsible for any misuse of this software.

## 🙏 Acknowledgments

- NVIDIA for CUDA toolkit and documentation
- Bitcoin community for cryptographic standards
- Contributors and testers

## 🔧 Advanced Configuration

### Environment Variables
```bash
# CUDA paths (if not in default location)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Compiler settings
export NVCC_CCBIN=/usr/bin/g++-9  # Specify GCC version
```

### Custom Build Options
```bash
# Debug build with verbose output
make clean && make gpu=1 debug=1 CCAP=86 VERBOSE=1 all

# Static linking
make clean && make gpu=1 CCAP=86 STATIC=1 all

# Specific CUDA architecture
make clean && make gpu=1 CCAP=75 GENCODE="-gencode arch=compute_75,code=sm_75" all
```

## 📊 Benchmarking

### Performance Testing
```bash
# Benchmark your GPU
./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --comp --range 1:10000 --benchmark

# Compare different grid sizes
for size in "128x128" "256x256" "512x512"; do
    echo "Testing grid size: $size"
    ./KeyHunt -g --gpui 0 --gpugridsize $size --mode ADDRESS --coin BTC --comp --range 1:1000000 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2
done
```

### Expected Performance (Mk/s)
- **GTX 1660 Ti**: 800-1000 Mk/s
- **RTX 2070**: 1000-1300 Mk/s
- **RTX 2080 Ti**: 1200-1500 Mk/s
- **RTX 3060**: 900-1200 Mk/s
- **RTX 3070**: 1300-1600 Mk/s
- **RTX 3080**: 1500-2000 Mk/s
- **RTX 3090**: 1800-2200 Mk/s
- **RTX 4070**: 1200-1600 Mk/s
- **RTX 4080**: 1800-2400 Mk/s
- **RTX 4090**: 2500-3500 Mk/s

## 🔐 Security Considerations

### Safe Usage
- **Never share private keys** found during legitimate research
- **Use secure systems** for key generation and storage
- **Verify results** before claiming any funds
- **Follow legal guidelines** in your jurisdiction

### Best Practices
- Run on isolated systems when possible
- Use hardware wallets for any valuable keys
- Keep detailed logs of search parameters
- Verify tool integrity before use

## 📞 Support

### Getting Help
1. **Check Documentation**: Review docs/ directory first
2. **Search Issues**: Look for similar problems on GitHub
3. **GPU Detection**: Run `./scripts/detect_gpu.sh` for hardware info
4. **System Info**: Include GPU model, CUDA version, and OS in reports

### Reporting Issues
Include the following information:
- GPU model and compute capability
- CUDA toolkit version
- Operating system and version
- Compilation command used
- Full error message or unexpected behavior
- Steps to reproduce the issue

### Community Resources
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community support
- **Documentation**: Comprehensive guides in docs/ directory

---

**Happy hunting! 🔍⚡**

*Remember: This tool is for educational and research purposes. Always comply with applicable laws and use responsibly.*
