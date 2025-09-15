# Keyhunt-CUDA Scientific Research System - iFlow Context

## 项目概述

Keyhunt-CUDA 是一个基于GPU加速的高性能比特币私钥扫描系统，专门用于解决比特币谜题挑战（Bitcoin Puzzle Challenges）。该项目采用科学研究方法，实现了模块化的CUDA加速框架，支持多GPU并行扫描、地址比对和性能优化。

### 核心技术特点
- **GPU加速**: 基于CUDA的secp256k1椭圆曲线运算
- **模块化架构**: ECC、扫描、比对三大核心模块
- **科学验证**: CPU/GPU一致性验证，确保计算正确性
- **高性能**: 支持多GPU并行，Bloom过滤器优化
- **可扩展**: 模块化设计，便于功能扩展和性能调优

### 项目状态
- **当前分支**: 001-keyhunt-cuda-puzzle
- **完成进度**: 28% (26/92任务已完成)
- **主要成就**: T002 GoogleTest集成完成，T026 PrivateKeyRange模型实现并通过12个单元测试
- **构建状态**: CMake配置完成，支持CUDA 11.0+和C++17

## 项目结构

```
PuzzleKeyhunt/keyhuntcuda/
├── src/KeyhuntCore/          # 核心实现
│   ├── ecc/                  # ECC运算模块 (secp256k1.cu, secp256k1.h)
│   ├── scan/                 # 扫描框架 (scanner.cu, scanner.h)
│   ├── compare/              # 地址比对 (hash.cu, hash.h)
│   ├── utils/                # 工具模块 (logger.cpp, timer.cpp, file_io.cpp)
│   ├── models/               # 数据模型 (PrivateKeyRange.cpp, TargetAddress.cpp)
│   ├── gpu/                  # GPU协调 (coordinator.cpp, load_balancer.cu)
│   ├── cli/                  # 命令行接口 (keyhunt_cli.cpp)
│   ├── config.h.in           # 配置模板
│   ├── CMakeLists.txt        # CMake构建配置
│   └── main.cpp              # 主程序入口
├── tests/                    # 测试驱动开发
│   ├── contract/             # API合同测试 (8个端点测试)
│   ├── integration/          # 集成测试 (5个用户故事验证)
│   ├── validation/           # 科学验证测试 (CPU/GPU一致性)
│   ├── unit/                 # 单元测试 (PrivateKeyRange等)
│   └── setup_test.cpp        # 构建验证测试
├── data/                     # 运行时数据和配置
│   ├── config/               # 配置文件
│   ├── checkpoint/           # 检查点数据
│   ├── logs/                 # 实验日志
│   └── results/              # 搜索结果和报告
├── docs/                     # 文档目录
│   ├── api/                  # API文档
│   ├── architecture/         # 系统架构
│   ├── performance/          # 性能分析
│   └── user/                 # 用户指南
├── specs/                    # 规格说明
│   └── 001-keyhunt-cuda-puzzle/
│       ├── spec.md           # 功能规格
│       ├── plan.md           # 实施计划
│       ├── tasks.md          # 任务列表
│       ├── data-model.md     # 数据模型
│       └── src/              # 源码融合参考
├── build/                    # 构建输出目录
├── cmake/                    # CMake模块
├── include/                  # 头文件目录
├── CMakeLists.txt            # 顶层CMake配置
├── build.sh                  # 构建脚本
├── LICENSE                   # 许可证
├── README.md                 # 项目文档
└── T002_COMPLETION_SUMMARY.md # 任务完成总结
```

## 构建和运行

### 环境要求
- **GPU**: NVIDIA GPU with Compute Capability ≥ 7.5 (Turing架构或更新)
- **CUDA**: CUDA Toolkit 11.0+
- **构建工具**: CMake 3.18+
- **编译器**: C++17兼容编译器
- **依赖**: libsecp256k1-dev (CPU验证参考，推荐)
- **多GPU**: NCCL (可选，用于多GPU通信)

### 快速构建
```bash
# 使用构建脚本（推荐）
./build.sh

# 手动构建
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
make -j$(nproc)
```

### 运行方式
```bash
# 查看帮助
./build/bin/keyhunt --help

# 查看版本信息
./build/bin/keyhunt --version

# 列出可用GPU设备
./build/bin/keyhunt --list-gpus

# 运行科学验证测试
./build/bin/keyhunt --validate

# 运行测试套件
cd build && make test

# 运行性能基准测试
./build/bin/keyhunt --benchmark
```

### 构建选项
```bash
# 调试构建
cmake .. -DCMAKE_BUILD_TYPE=Debug

# 发布构建（默认）
cmake .. -DCMAKE_BUILD_TYPE=Release

# 启用测试
cmake .. -DBUILD_TESTS=ON

# 启用基准测试
cmake .. -DBUILD_BENCHMARKS=ON

# 启用激进优化
cmake .. -DENABLE_AGGRESSIVE_OPTIMIZATIONS=ON
```

## 开发规范

### 代码组织原则
1. **模块化**: 每个模块职责单一，接口清晰
2. **CUDA分离**: .cu文件包含CUDA代码，.cpp文件包含主机代码
3. **命名空间**: 使用`keyhunt::`命名空间，子模块使用子命名空间
4. **错误处理**: 使用日志系统记录错误，异常安全设计
5. **TDD方法**: 测试先行，所有实现必须先有失败的测试

### 文件命名约定
- **GPU内核文件**: `*.cu` (如`secp256k1.cu`, `scanner.cu`)
- **头文件**: `*.h` (如`secp256k1.h`, `scanner.h`)
- **主机代码**: `*.cpp` (如`logger.cpp`, `timer.cpp`)
- **配置文件**: `*.txt` (如`config.txt`, `private_ranges.txt`)
- **测试文件**: `test_*.cpp` (如`test_privatekey_range_unit.cpp`)

### 测试驱动开发
- **合同测试**: 先创建API合同测试，必须失败
- **集成测试**: 验证模块间协作和用户故事
- **科学验证**: CPU/GPU一致性验证，精度<1e-10
- **单元测试**: 针对每个模块的独立测试
- **测试顺序**: Contract → Integration → Validation → Unit

### 配置管理
- **主配置**: `data/config/config.txt`
- **私钥范围**: `data/config/private_ranges.txt`
- **目标地址**: `data/config/target_addresses.txt`
- **检查点**: `data/checkpoint/checkpoint.dat`
- **实验日志**: `data/logs/experimental_log.json`
- **验证报告**: `data/results/validation_report.json`

### 性能调优指南
- **GPU配置**: blocksPerGrid=30-60, threadsPerBlock=256
- **内存优化**: 使用pinned memory，启用memory pool
- **批处理**: 调整maxBatchSize基于GPU内存
- **多GPU**: 设置CUDA_VISIBLE_DEVICES环境变量
- **性能目标**: Turing >1000M keys/s, Hopper >4000M keys/s

## 核心模块说明

### ECC模块 (ecc/)
负责secp256k1椭圆曲线运算，包括：
- 私钥到公钥的标量乘法
- 预计算表生成
- CPU/GPU一致性验证
- Jacobian坐标系优化
- 科学精度验证 (<1e-10相对误差)

### 扫描模块 (scan/)
负责私钥范围扫描，包括：
- 线程/块分配逻辑
- 检查点/恢复功能
- 进度跟踪和统计
- 批量处理优化
- 多GPU负载均衡

### 比对模块 (compare/)
负责地址生成和比对，包括：
- P2PKH、P2SH、Bech32地址生成
- Hash160计算 (SHA256 + RIPEMD160)
- Bloom过滤器多目标比对
- Base58编码/解码
- 地址验证

### 工具模块 (utils/)
提供通用功能，包括：
- 日志系统（分级、文件轮转、彩色输出）
- 性能计时（CPU、CUDA、命名计时器）
- 文件I/O（文本、二进制、CSV、配置）
- 性能分析器和监控

### 数据模型 (models/)
核心数据结构，包括：
- **PrivateKeyRange**: 私钥范围管理，进度跟踪
- **TargetAddress**: 目标地址管理
- **CheckpointData**: 检查点数据
- **GPUConfiguration**: GPU配置管理
- **ValidationReport**: 验证报告

## 科学研究方法

### 验证机制
1. **CPU/GPU一致性**: 所有GPU计算结果与CPU实现对比验证
2. **地址验证**: 生成的地址通过多种方法验证正确性
3. **统计监控**: 持续监控性能指标和错误率
4. **可重现性**: 所有操作都是确定性和可重现的

### 性能分析
- 使用Nsight Compute分析GPU内核性能
- 收集keys/s、GPU利用率、内存带宽等指标
- 支持性能报告生成和导出
- 提供基准测试框架

### 源码融合架构
项目采用源码融合方法，结合了：
- **CudaBrainSecp**: 高性能ECC内核实现
- **BitCrack**: 成熟的扫描框架
- **KeyhuntCore**: 自定义科学验证框架

## 扩展开发

### 添加新功能
1. **新ECC算法**: 扩展`ecc/secp256k1.cu`
2. **新地址类型**: 修改`compare/hash.cu`
3. **新扫描策略**: 更新`scan/scanner.cu`
4. **新工具功能**: 添加到`utils/`目录

### 测试开发
- **单元测试**: 针对每个模块独立测试
- **集成测试**: 验证模块间协作
- **性能测试**: 基准测试和回归测试
- **验证测试**: CPU/GPU结果一致性

### 性能优化
- **内核优化**: 调整warp大小、共享内存使用
- **内存优化**: 减少主机-设备数据传输
- **算法优化**: 批处理、预计算、并行化
- **多GPU优化**: 负载均衡、通信优化

## 任务管理

### 当前任务状态
- **已完成**: T001-T002 (项目设置、GoogleTest集成)
- **已完成**: T026 (PrivateKeyRange模型)
- **已完成**: T085 (PrivateKeyRange单元测试)
- **进行中**: T003-T006 (科学验证工具配置)
- **待开始**: T007-T015 (合同测试)

### 任务执行顺序
1. **阶段3.1**: 项目设置和依赖 (T001-T006)
2. **阶段3.2**: 合同测试 (T007-T015) - 必须先失败
3. **阶段3.3**: 科学验证测试 (T016-T020) - 必须先失败
4. **阶段3.4**: 集成测试 (T021-T025)
5. **阶段3.5**: 核心实体模型 (T026-T031)
6. **阶段3.6**: ECC内核基础 (T032-T039)
7. **阶段3.7**: 扫描框架 (T040-T045)
8. **阶段3.8**: 地址生成和比对 (T046-T051)

## 故障排除

### 常见问题
1. **CUDA未找到**: 确保CUDA toolkit安装并配置PATH
2. **内存不足**: 减少批处理大小或pointsPerThread
3. **编译错误**: 检查CUDA架构兼容性
4. **性能问题**: 使用nvidia-smi监控GPU利用率
5. **验证失败**: 检查libsecp256k1-dev安装

### 调试工具
```bash
# 调试构建
cmake .. -DCMAKE_BUILD_TYPE=Debug

# 详细输出
make VERBOSE=1

# CUDA设备信息
./build/bin/keyhunt --list-gpus

# 运行特定测试
./build/bin/unit_test --gtest_filter="PrivateKeyRangeTest.ValidRangeConstruction"
```

### 性能监控
```bash
# GPU使用率监控
watch -n 1 nvidia-smi

# 性能基准测试
./build/bin/keyhunt --benchmark

# 启用性能分析
./build/bin/keyhunt --enable-profiling=true
```

## 注意事项

### 安全警告
- 本项目仅供教育和研究用途
- 请负责任地使用，遵守相关法律法规
- 私钥信息敏感，注意数据安全
- 不要在不安全的环境中运行

### 性能期望
- 实际性能取决于GPU硬件配置
- 大范围扫描需要大量时间和计算资源
- 建议先在小范围测试验证正确性
- 使用检查点功能防止进度丢失

### 科学诚信
- 所有实验结果应可重现和验证
- 遵循学术诚信原则
- 正确引用参考实现和算法来源
- 提供完整的实验数据和分析

## 技术栈详情

### 核心技术
- **语言**: C++17, CUDA 11.0+
- **数学库**: libsecp256k1 (CPU参考), CGBN (大整数运算)
- **构建系统**: CMake 3.18+
- **测试框架**: GoogleTest
- **多GPU**: NCCL (可选)
- **架构支持**: Turing (7.5), Ampere (8.0), Hopper (9.0)

### 依赖管理
- **必需**: CUDA Toolkit, CMake, C++17编译器
- **推荐**: libsecp256k1-dev (科学验证)
- **可选**: NCCL (多GPU优化)
- **测试**: GoogleTest (本地集成)

### 平台支持
- **主要**: Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA Turing架构及以上
- **内存**: 建议8GB+ GPU内存
- **存储**: SSD用于检查点和日志

这个IFLOW.md文件为未来的iFlow交互提供了完整的项目上下文，包括技术架构、构建方法、开发规范和最佳实践。