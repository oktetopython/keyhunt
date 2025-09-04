# KeyHunt-Cuda Project

A high-performance CUDA-accelerated Bitcoin private key search tool optimized for modern NVIDIA GPUs.

[中文版本](README_CN.md) | **English Version**

## 🚀 Features

- **GPU Acceleration**: Harness the power of NVIDIA GPUs for Bitcoin private key search
- **Universal GPU Support**: Compatible with RTX 20xx, 30xx, and 40xx series (Compute Capability 7.5-9.0)
- **Multiple Search Modes**: Address search, X-point search, and Ethereum address search
- **High Performance**: Achieve 1000+ Mk/s on modern GPUs
- **Optimized Code**: Zero compilation warnings, clean architecture, minimal code duplication
- **Easy Setup**: Automated GPU detection and optimal compilation recommendations

## 📋 Quick Start

### 1. System Requirements
- **NVIDIA GPU**: GTX 16xx series or newer (Compute Capability 7.5+)
- **CUDA Toolkit**: 11.0 or newer
- **GCC/G++**: 7.5 or newer
- **RAM**: 4GB+ recommended

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/keyhunt-cuda.git
cd keyhunt-cuda

# Install dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install build-essential libgmp-dev

# Auto-detect GPU and compile
cd keyhuntcuda/KeyHunt-Cuda
./scripts/detect_gpu.sh
# Follow the recommended compilation command
```

### 3. Quick Test
```bash
# Test with Bitcoin puzzle 40 (known solution)
./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range e9ae493300:e9ae493400 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv

# Expected result: Should find private key E9AE4933D6
```

## 🔧 Compilation Options

### Single GPU Architecture (Recommended)
```bash
# For RTX 20xx/GTX 16xx series
make clean && make gpu=1 CCAP=75 all

# For RTX 30xx series
make clean && make gpu=1 CCAP=86 all

# For RTX 40xx series
make clean && make gpu=1 CCAP=90 all
```

### Multi-GPU Architecture (Universal)
```bash
# Works on all supported GPUs (RTX 20xx-40xx)
make clean && make gpu=1 MULTI_GPU=1 all
```

## 🎯 Usage Examples

### Basic Address Search
```bash
./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range 1000000000:2000000000 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2
```

### X-Point Search
```bash
./KeyHunt -g --gpui 0 --mode XPOINT --coin BTC --range 1:FFFFFFFF 50929b74c1a04954b78b4b6035e97a5e078a5a0f28ec96d547bfee9ace803ac0
```

### Ethereum Address Search
```bash
./KeyHunt -g --gpui 0 --mode ADDRESS --coin ETH --range 1:FFFFFFFF 0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6
```

## 📊 Performance

| GPU Model | Compute Capability | Typical Performance |
|-----------|-------------------|-------------------|
| GTX 1660 Ti | 7.5 | 800-1000 Mk/s |
| RTX 2080 Ti | 7.5 | 1200-1500 Mk/s |
| RTX 3080 | 8.6 | 1500-2000 Mk/s |
| RTX 3090 | 8.6 | 1800-2200 Mk/s |
| RTX 4080 | 9.0 | 1800-2400 Mk/s |
| RTX 4090 | 9.0 | 2500-3500 Mk/s |

## 📁 Project Structure

```
keyhunt/
├── README.md                    # This file
├── README_CN.md                 # Chinese version
├── keyhuntcuda/                 # Main project directory
│   └── KeyHunt-Cuda/           # Source code and binaries
│       ├── README.md           # Detailed technical documentation
│       ├── docs/               # Documentation
│       ├── scripts/            # Build and utility scripts
│       ├── tests/              # Test files
│       ├── examples/           # Usage examples
│       └── GPU/                # CUDA kernels
└── gECC-main/                  # gECC integration (optional)
```

## 🛠️ Advanced Features

### GPU Detection
```bash
# Automatic GPU detection and build recommendations
cd keyhuntcuda/KeyHunt-Cuda
./scripts/detect_gpu.sh
```

### Performance Optimization
```bash
# Custom grid size for fine-tuning
./KeyHunt -g --gpui 0 --gpugridsize 256x256 [other options]

# Multi-GPU setup
./KeyHunt -g --gpui 0,1,2 [other options]
```

### Example Scripts
```bash
# Run interactive examples
cd keyhuntcuda/KeyHunt-Cuda
./examples/example_searches.sh
```

## 🔍 Troubleshooting

### Common Issues

#### CUDA Not Found
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### GPU Not Detected
```bash
nvidia-smi  # Check if GPU is detected
sudo nvidia-modprobe  # Load NVIDIA kernel module
```

#### Low Performance
```bash
# Use single GPU build for your architecture
make clean && make gpu=1 CCAP=86 all

# Check GPU utilization
nvidia-smi -l 1
```

## 📚 Documentation

- **[Technical Documentation](keyhuntcuda/KeyHunt-Cuda/README.md)**: Comprehensive usage guide
- **[Quick Start Guide](keyhuntcuda/KeyHunt-Cuda/docs/QUICK_START.md)**: Get started in 5 minutes
- **[GPU Compatibility](keyhuntcuda/KeyHunt-Cuda/docs/GPU_COMPATIBILITY_GUIDE.md)**: GPU support details
- **[Build System](keyhuntcuda/KeyHunt-Cuda/docs/BUILD_SYSTEM.md)**: Advanced compilation options
- **[Code Quality Report](keyhuntcuda/KeyHunt-Cuda/docs/CODE_QUALITY_IMPROVEMENTS.md)**: Technical improvements

## 🎯 Recent Improvements

### Code Quality Enhancements
- ✅ Eliminated 3,550+ lines of duplicate code
- ✅ Achieved zero compilation warnings
- ✅ Improved project organization with structured directories
- ✅ Enhanced GPU compatibility (CC 7.5-9.0)
- ✅ Created comprehensive documentation

### Performance Optimizations
- ✅ Unified error handling patterns
- ✅ Optimized memory access patterns
- ✅ Reduced compilation time by 15%
- ✅ Maintained 1200+ Mk/s GPU performance

## ⚠️ Important Notes

- **Educational Use Only**: This tool is for research and educational purposes
- **Legal Compliance**: Ensure you comply with all applicable laws
- **Security**: Never share private keys found during legitimate research
- **Verification**: Always verify results before taking any action

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](keyhuntcuda/LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for CUDA toolkit and documentation
- Bitcoin community for cryptographic standards
- All contributors who have improved this project

## 📞 Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check the docs/ directory
- **Performance**: Use GPU detection script for optimization

---

**Happy hunting! 🔍⚡**

*For detailed technical information, see the [main documentation](keyhuntcuda/KeyHunt-Cuda/README.md).*
