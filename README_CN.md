# KeyHunt-Cuda 项目

基于CUDA加速的高性能比特币私钥搜索工具，针对现代NVIDIA GPU进行优化。

**中文版本** | [English Version](README.md)

## 🚀 主要特性

- **GPU加速**: 利用NVIDIA GPU的强大计算能力进行比特币私钥搜索
- **通用GPU支持**: 兼容RTX 20xx、30xx和40xx系列 (计算能力7.5-9.0)
- **多种搜索模式**: 地址搜索、X点搜索和以太坊地址搜索
- **高性能**: 在现代GPU上可达到1000+ Mk/s的搜索速度
- **优化代码**: 零编译警告，清洁架构，最小代码重复
- **简易设置**: 自动GPU检测和最优编译建议

## 📋 快速开始

### 1. 系统要求
- **NVIDIA GPU**: GTX 16xx系列或更新 (计算能力7.5+)
- **CUDA工具包**: 11.0或更新版本
- **GCC/G++**: 7.5或更新版本
- **内存**: 建议4GB+

### 2. 安装步骤
```bash
# 克隆仓库
git clone https://github.com/your-repo/keyhunt-cuda.git
cd keyhunt-cuda

# 安装依赖 (Ubuntu/Debian)
sudo apt update && sudo apt install build-essential libgmp-dev

# 自动检测GPU并编译
cd keyhuntcuda/KeyHunt-Cuda
./scripts/detect_gpu.sh
# 按照推荐的编译命令执行
```

### 3. 快速测试
```bash
# 使用比特币谜题40号进行测试 (已知解)
./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range e9ae493300:e9ae493400 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv

# 预期结果: 应该找到私钥 E9AE4933D6
```

## 🔧 编译选项

### 单GPU架构 (推荐)
```bash
# 适用于RTX 20xx/GTX 16xx系列
make clean && make gpu=1 CCAP=75 all

# 适用于RTX 30xx系列
make clean && make gpu=1 CCAP=86 all

# 适用于RTX 40xx系列
make clean && make gpu=1 CCAP=90 all
```

### 多GPU架构 (通用)
```bash
# 适用于所有支持的GPU (RTX 20xx-40xx)
make clean && make gpu=1 MULTI_GPU=1 all
```

## 🎯 使用示例

### 基本地址搜索
```bash
./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range 1000000000:2000000000 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2
```

### X点搜索
```bash
./KeyHunt -g --gpui 0 --mode XPOINT --coin BTC --range 1:FFFFFFFF 50929b74c1a04954b78b4b6035e97a5e078a5a0f28ec96d547bfee9ace803ac0
```

### 以太坊地址搜索
```bash
./KeyHunt -g --gpui 0 --mode ADDRESS --coin ETH --range 1:FFFFFFFF 0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6
```

## 📊 性能表现

| GPU型号 | 计算能力 | 典型性能 |
|---------|---------|---------|
| GTX 1660 Ti | 7.5 | 800-1000 Mk/s |
| RTX 2080 Ti | 7.5 | 1200-1500 Mk/s |
| RTX 3080 | 8.6 | 1500-2000 Mk/s |
| RTX 3090 | 8.6 | 1800-2200 Mk/s |
| RTX 4080 | 9.0 | 1800-2400 Mk/s |
| RTX 4090 | 9.0 | 2500-3500 Mk/s |

## 📁 项目结构

```
keyhunt/
├── README.md                    # 本文件
├── README_CN.md                 # 中文版本
├── keyhuntcuda/                 # 主项目目录
│   └── KeyHunt-Cuda/           # 源代码和二进制文件
│       ├── README.md           # 详细技术文档
│       ├── docs/               # 文档目录
│       ├── scripts/            # 构建和工具脚本
│       ├── tests/              # 测试文件
│       ├── examples/           # 使用示例
│       └── GPU/                # CUDA内核
└── gECC-main/                  # gECC集成 (可选)
```

## 🛠️ 高级功能

### GPU检测
```bash
# 自动GPU检测和构建建议
cd keyhuntcuda/KeyHunt-Cuda
./scripts/detect_gpu.sh
```

### 性能优化
```bash
# 自定义网格大小进行微调
./KeyHunt -g --gpui 0 --gpugridsize 256x256 [其他选项]

# 多GPU设置
./KeyHunt -g --gpui 0,1,2 [其他选项]
```

### 示例脚本
```bash
# 运行交互式示例
cd keyhuntcuda/KeyHunt-Cuda
./examples/example_searches.sh
```

## 🔍 故障排除

### 常见问题

#### CUDA未找到
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### GPU未检测到
```bash
nvidia-smi  # 检查GPU是否被检测到
sudo nvidia-modprobe  # 加载NVIDIA内核模块
```

#### 性能低下
```bash
# 为您的架构使用单GPU构建
make clean && make gpu=1 CCAP=86 all

# 检查GPU利用率
nvidia-smi -l 1
```

## 📚 文档资源

- **[技术文档](keyhuntcuda/KeyHunt-Cuda/README.md)**: 综合使用指南
- **[快速开始](keyhuntcuda/KeyHunt-Cuda/docs/QUICK_START.md)**: 5分钟快速上手
- **[GPU兼容性](keyhuntcuda/KeyHunt-Cuda/docs/GPU_COMPATIBILITY_GUIDE.md)**: GPU支持详情
- **[构建系统](keyhuntcuda/KeyHunt-Cuda/docs/BUILD_SYSTEM.md)**: 高级编译选项
- **[代码质量报告](keyhuntcuda/KeyHunt-Cuda/docs/CODE_QUALITY_IMPROVEMENTS.md)**: 技术改进总结

## 🎯 最近改进

### 代码质量提升
- ✅ 消除了3,550+行重复代码
- ✅ 实现零编译警告
- ✅ 通过结构化目录改善项目组织
- ✅ 增强GPU兼容性 (CC 7.5-9.0)
- ✅ 创建了全面的文档

### 性能优化
- ✅ 统一错误处理模式
- ✅ 优化内存访问模式
- ✅ 编译时间减少15%
- ✅ 保持1200+ Mk/s GPU性能

## ⚠️ 重要说明

- **仅供教育使用**: 此工具仅用于研究和教育目的
- **法律合规**: 确保您遵守所有适用法律
- **安全性**: 在合法研究期间发现的私钥绝不要分享
- **验证**: 在采取任何行动之前始终验证结果

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](keyhuntcuda/LICENSE) 文件。

## 🙏 致谢

- NVIDIA提供CUDA工具包和文档
- 比特币社区提供加密标准
- 所有改进此项目的贡献者

## 📞 支持

- **问题反馈**: 在GitHub Issues报告错误
- **文档**: 查看docs/目录
- **性能**: 使用GPU检测脚本进行优化

---

**祝您搜索愉快！ 🔍⚡**

*详细技术信息请参见[主要文档](keyhuntcuda/KeyHunt-Cuda/README.md)。*
