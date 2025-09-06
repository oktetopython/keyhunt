# KeyHunt-Cuda 构建脚本

本目录包含适用于不同环境和平台的构建脚本，方便用户在各种系统上编译 KeyHunt-Cuda。

## 脚本说明

### 1. 通用构建脚本

#### `build.sh` - Bash 构建脚本
适用于 Linux 和 WSL 环境的通用构建脚本。

**使用方法:**
```bash
# 构建默认 GPU 版本
./scripts/build.sh

# 构建调试版本
./scripts/build.sh -d

# 清理并构建多 GPU 版本
./scripts/build.sh -c -m --ccap 86

# 构建仅 CPU 版本
./scripts/build.sh --cpu-only
```

#### `build.bat` - Windows 批处理脚本
适用于 Windows CMD 环境的构建脚本。

**使用方法:**
```cmd
# 构建默认 GPU 版本
scripts\build.bat

# 构建调试版本
scripts\build.bat -d

# 清理并构建多 GPU 版本
scripts\build.bat -c -m --ccap 86

# 构建仅 CPU 版本
scripts\build.bat --cpu-only
```

#### `build.ps1` - PowerShell 脚本
适用于 Windows PowerShell 和 PowerShell Core 环境的构建脚本。

**使用方法:**
```powershell
# 构建默认 GPU 版本
.\scripts\build.ps1

# 构建调试版本
.\scripts\build.ps1 -d

# 清理并构建多 GPU 版本
.\scripts\build.ps1 -c -m --ccap 86

# 构建仅 CPU 版本
.\scripts\build.ps1 --cpu-only
```

#### `build.py` - Python 跨平台脚本
使用 Python 编写的跨平台构建脚本，支持 Windows、Linux 和 macOS。

**使用方法:**
```bash
# 首先确保安装了 Python 3
# 构建默认 GPU 版本
python3 scripts/build.py

# 构建调试版本
python3 scripts/build.py --debug

# 清理并构建多 GPU 版本
python3 scripts/build.py --clean --multi-gpu --ccap 86

# 构建仅 CPU 版本
python3 scripts/build.py --cpu-only
```

### 2. 特殊用途脚本

#### `detect_gpu.sh` - GPU 检测脚本
自动检测系统中的 NVIDIA GPU 并推荐最佳编译选项。

**使用方法:**
```bash
./scripts/detect_gpu.sh
```

#### `quick_build.sh` - 快速构建脚本
一键构建最常见的配置。

**使用方法:**
```bash
# 快速构建默认版本
./scripts/quick_build.sh

# 快速构建 CPU 版本
./scripts/quick_build.sh cpu

# 快速构建调试版本
./scripts/quick_build.sh debug

# 清理后快速构建
./scripts/quick_build.sh clean
```

## 构建选项说明

### GPU 支持选项
- **单 GPU 构建**: 针对特定 GPU 架构优化，性能最佳
  - `make gpu=1 CCAP=75 all` (GTX 16xx/RTX 20xx)
  - `make gpu=1 CCAP=86 all` (RTX 30xx)
  - `make gpu=1 CCAP=90 all` (RTX 40xx)

- **多 GPU 构建**: 支持多种 GPU 架构，兼容性更好
  - `make gpu=1 MULTI_GPU=1 all`

- **仅 CPU 构建**: 不需要 GPU 支持，适用于所有系统
  - `make all`

### 调试选项
- **调试版本**: 包含调试信息，便于调试
  - `make gpu=1 debug=1 CCAP=75 all`

- **发布版本**: 优化版本，性能更好
  - `make gpu=1 CCAP=75 all`

## 系统要求

### Linux/WSL
- GNU Make 3.81+
- GCC/G++ 7.5+
- CUDA Toolkit 11.0+ (如果启用 GPU 支持)
- NVIDIA 驱动程序 (如果启用 GPU 支持)

### Windows
- Visual Studio 或 MinGW-w64 (包含 make 工具)
- CUDA Toolkit 11.0+ (如果启用 GPU 支持)
- NVIDIA 驱动程序 (如果启用 GPU 支持)

### macOS
- Xcode 命令行工具
- CUDA Toolkit 11.0+ (如果启用 GPU 支持，注意：Apple Silicon 不支持 CUDA)

## 常见问题

### 1. `make: command not found`
确保已安装构建工具：
- **Ubuntu/Debian**: `sudo apt install build-essential`
- **CentOS/RHEL**: `sudo yum groupinstall "Development Tools"`
- **Windows**: 安装 Visual Studio 或 MinGW-w64

### 2. `nvcc: command not found`
确保已安装 CUDA Toolkit 并正确设置环境变量：
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 3. 构建过程中出现编译错误
尝试以下步骤：
1. 清理构建目录：`make clean`
2. 重新构建：`make gpu=1 CCAP=75 all`
3. 如果问题仍然存在，请检查 CUDA 和 GCC 版本兼容性

## 性能优化建议

1. **选择正确的计算能力**: 使用 `detect_gpu.sh` 脚本检测您的 GPU 并选择相应的 CCAP 值
2. **单 GPU 构建**: 对于特定硬件，单 GPU 构建通常比多 GPU 构建性能更好
3. **启用优化标志**: 发布版本默认启用优化标志
4. **调整网格大小**: 运行时使用 `--gpugridsize` 参数调整网格大小以获得最佳性能