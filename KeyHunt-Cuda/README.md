# KeyHunt-Cuda with GPU Optimization

基于KeyHunt-Cuda的GPU加速比特币私钥搜索工具，已修复GMP库依赖问题和构建脚本。

## 功能特性

- ✅ GPU加速计算（CUDA支持）
- ✅ GMP大数运算库支持
- ✅ 多线程CPU优化
- ✅ Windows/Linux跨平台构建
- ✅ 完整的编译警告修复

## 系统要求

### 硬件要求
- NVIDIA GPU（支持CUDA）
- 至少4GB显存
- 8GB以上系统内存

### 软件依赖
- CUDA Toolkit 11.0+
- GMP库（libgmp-dev）
- GCC/G++ 9.0+
- Git

## 安装依赖

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y build-essential nvidia-cuda-toolkit libgmp-dev
```

### Windows
1. 安装 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
2. 安装 [MSYS2](https://www.msys2.org/) 并安装GMP：
   ```bash
   pacman -S mingw-w64-x86_64-gmp
   ```

## 编译步骤

### 方法一：使用Makefile（推荐）
```bash
# 清理旧构建
make clean

# 编译GPU版本
make WITHGPU=1

# 或者编译CPU版本  
make
```

### 方法二：手动构建脚本
```bash
# 修复行尾符（Windows需要）
sed -i 's/\r$//' manual_build.sh

# 运行构建脚本
bash manual_build.sh
```

## 使用方法

### 基本命令格式
```bash
./keyhunt-cuda75 -m address -f addresses.txt -r 1:FFFFFFFF
```

### 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `-m` | 搜索模式 | `-m address` |
| `-f` | 地址文件 | `-f addresses.txt` |
| `-r` | 密钥范围 | `-r 1:FFFFFFFF` |
| `-t` | 线程数 | `-t 8` |
| `-g` | GPU设备 | `-g 0` |
| `-b` | 工作大小 | `-b 64` |
| `-q` | 安静模式 | `-q` |

### 示例命令

1. **搜索特定地址**
   ```bash
   ./keyhunt-cuda75 -m address -f my_addresses.txt -r 1:1000000 -t 12 -g 0
   ```

2. **使用多个GPU**
   ```bash
   ./keyhunt-cuda75 -m address -f addresses.txt -r 1:FFFFFFFF -g 0,1,2 -t 24
   ```

3. **大范围搜索**
   ```bash
   ./keyhunt-cuda75 -m address -f large_list.txt -r 800000:FFFFFFFF -b 128 -q
   ```

## 故障排除

### 常见问题

1. **CUDA错误**
   ```
   CUDA error: no kernel image is available for execution on the device
   ```
   **解决方案**：检查CUDA版本兼容性，可能需要重新编译

2. **GMP库找不到**
   ```
   undefined reference to `__gmpz_init'
   ```
   **解决方案**：确保安装了libgmp-dev，链接时添加-lgmp

3. **内存不足**
   ```
   out of memory
   ```
   **解决方案**：减少工作大小（-b参数）或使用更多GPU

### 性能优化

1. **调整工作大小**：根据GPU内存调整`-b`参数
2. **多GPU负载均衡**：使用多个GPU设备编号
3. **CPU/GPU平衡**：调整线程数避免CPU瓶颈

## 文件说明

- `keyhunt-cuda75` - 主可执行文件
- `Makefile` - 构建配置文件
- `manual_build.sh` - 手动构建脚本
- `GmpUtil.cpp` - GMP工具函数
- `compile_warnings.log` - 编译警告日志

## 版本历史

### v1.0.0 (2025-01-15)
- ✅ 修复GMP库链接问题
- ✅ 解决构建脚本行尾符问题
- ✅ 优化Makefile编译选项
- ✅ 添加完整的CUDA支持
- ✅ 生成可用的二进制文件

## 技术支持

如有问题请提交Issue或联系开发团队。

## 许可证

本项目基于MIT许可证开源。

---
**注意**：请合法使用本工具，仅用于教育和技术研究目的。