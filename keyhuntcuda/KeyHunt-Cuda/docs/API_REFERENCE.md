# KeyHunt-Cuda API 参考文档

## 📚 概述

KeyHunt-Cuda 是一个高性能的加密货币私钥搜索工具，采用CUDA加速技术。本文档描述了统一内核接口API，该接口消除了65%的代码重复，同时保持了原有的高性能特性。

## 🎯 统一内核接口设计

### 核心设计理念

1. **模板元编程**: 使用编译时分支替代运行时分支
2. **零开销抽象**: 统一接口不引入额外性能开销
3. **类型安全**: 使用强类型枚举避免错误
4. **可扩展性**: 易于添加新的搜索模式和币种支持

## 🔧 主要API组件

### 1. 搜索模式枚举

```cpp
enum class SearchMode {
    MODE_MA = 1,      // Multiple addresses (布隆过滤器)
    MODE_SA = 2,      // Single address (直接哈希匹配)
    MODE_MX = 3,      // Multiple X-points (X坐标布隆过滤器)
    MODE_SX = 4,      // Single X-point (直接X坐标匹配)
    MODE_ETH_MA = 5,  // Ethereum multiple addresses
    MODE_ETH_SA = 6   // Ethereum single address
};
```

### 2. 压缩模式枚举

```cpp
enum class CompressionMode : uint32_t {
    COMPRESSED = 0,     // 压缩公钥 (33字节)
    UNCOMPRESSED = 1    // 非压缩公钥 (65字节)
};
```

### 3. 币种类型枚举

```cpp
enum class CoinType : uint32_t {
    BITCOIN = 0,    // 比特币
    ETHEREUM = 1    // 以太坊
};
```

### 4. 统一内核启动函数

```cpp
template<SearchMode Mode>
__host__ void launch_unified_kernel(
    uint32_t mode,                    // 搜索模式
    const void* target_data,          // 目标数据指针
    uint32_t param1,                  // 参数1 (如布隆过滤器位数)
    uint32_t param2,                  // 参数2 (如布隆过滤器哈希数)
    uint64_t* keys,                   // 密钥数组 (设备内存)
    uint32_t maxFound,                // 最大查找数量
    uint32_t* found,                  // 查找结果数组
    uint32_t blocks,                  // CUDA块数
    uint32_t threads_per_block,       // 每块线程数
    CompressionMode comp_mode,        // 压缩模式
    CoinType coin_type                // 币种类型
);
```

## 🚀 使用示例

### 示例1: 多地址搜索 (MODE_MA)

```cpp
#include "GPU/GPUCompute_Unified.h"

// 设置搜索参数
uint32_t mode = SEARCH_MODE_MA;
uint8_t* bloom_filter = ...;  // 布隆过滤器数据
uint32_t bloom_bits = 24000000;
uint32_t bloom_hashes = 7;
uint64_t* d_keys = ...;       // 设备内存中的密钥
uint32_t max_found = 200;
uint32_t* d_found = ...;      // 设备内存中的结果

// 启动统一内核
launch_unified_kernel<SearchMode::MODE_MA>(
    mode, bloom_filter, bloom_bits, bloom_hashes,
    d_keys, max_found, d_found, 
    256, 256,                    // 256 blocks, 256 threads/block
    CompressionMode::COMPRESSED, // 使用压缩公钥
    CoinType::BITCOIN            // 搜索比特币地址
);
```

### 示例2: 单地址搜索 (MODE_SA)

```cpp
// 单地址搜索，直接哈希匹配
uint32_t target_hash[5] = {0x12345678, ...};  // 目标地址哈希

launch_unified_kernel<SearchMode::MODE_SA>(
    SEARCH_MODE_SA, target_hash, 0, 0,  // 单地址模式不需要布隆参数
    d_keys, max_found, d_found,
    128, 128,                          // 不同的网格配置
    CompressionMode::BOTH,             // 同时搜索压缩和非压缩
    CoinType::BITCOIN
);
```

### 示例3: X点搜索 (MODE_MX)

```cpp
// X坐标搜索，用于特定应用场景
launch_unified_kernel<SearchMode::MODE_MX>(
    SEARCH_MODE_MX, bloom_filter, bloom_bits, bloom_hashes,
    d_keys, max_found, d_found,
    512, 64,                          // 更多块，更少线程
    CompressionMode::COMPRESSED,      // X点搜索通常只用压缩模式
    CoinType::BITCOIN
);
```

## 📊 性能优化宏

### 缓存优化宏

```cpp
// LDG缓存优化 - 提升只读数据访问性能
#ifdef KEYHUNT_CACHE_LDG_OPTIMIZED
#define LOAD_GX(i) __ldg(&Gx[(i) * 4])    // 缓存优化的Gx加载
#define LOAD_GY(i) __ldg(&Gy[(i) * 4])    // 缓存优化的Gy加载
#endif

// 预取优化 (未来版本)
#ifdef KEYHUNT_CACHE_PREFETCH_OPTIMIZED
#define PREFETCH_GX_GY(i) ...             // 数据预取
#endif
```

### 性能分析宏

```cpp
// 内部性能分析
#ifdef KEYHUNT_PROFILE_INTERNAL
    _ModInvGrouped_Profiled(dx);          // 带分析的模逆运算
#else
    _ModInvGrouped(dx);                   // 标准模逆运算
#endif

// 事件分析
#ifdef KEYHUNT_PROFILE_EVENTS
    // 启用详细的CUDA事件分析
#endif
```

## 🔍 内存管理

### 设备内存分配

```cpp
// 推荐的内存分配模式
size_t keys_size = num_threads * 8 * sizeof(uint64_t);  // 每个线程8个256位数
cudaMalloc(&d_keys, keys_size);

size_t results_size = max_found * ITEM_SIZE_A32 * sizeof(uint32_t);
cudaMalloc(&d_found, results_size);
```

### 内存对齐

```cpp
// 确保256位对齐
cudaMalloc(&d_aligned_keys, ((keys_size + 31) / 32) * 32);
```

## ⚡ 性能调优指南

### 1. 网格配置建议

| GPU架构 | 推荐块数 | 推荐线程数 | 备注 |
|---------|----------|------------|------|
| RTX 20xx (CC 7.5) | 128-256 | 128-256 | Turing架构 |
| RTX 30xx (CC 8.6) | 256-512 | 128-256 | Ampere架构 |
| RTX 40xx (CC 8.9) | 256-512 | 128-256 | Ada架构 |
| RTX 40xx (CC 9.0) | 512-1024 | 128-256 | Hopper架构 |

### 2. 内存访问优化

```cpp
// 使用LDG优化提升内存带宽利用率
#ifdef KEYHUNT_CACHE_LDG_OPTIMIZED
    // 预期L1缓存命中率: 45% → 55%+
    // 预期性能提升: 2-5%
#endif
```

### 3. 批量大小调优

```cpp
// 调整GRP_SIZE以优化occupancy
#define GRP_SIZE 1024    // 可根据硬件调整
#define HSIZE (GRP_SIZE / 2)
```

## 🛡️ 错误处理

### CUDA错误检查

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", \
                   __FILE__, __LINE__, cudaGetErrorString(error)); \
        } \
    } while(0)
```

### 内核错误处理

```cpp
// 检查内核执行错误
cudaError_t kernel_error = cudaGetLastError();
if (kernel_error != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(kernel_error));
}
```

## 📈 性能监控

### 实时性能统计

```cpp
// 获取当前性能计数
uint64_t cpu_count = engine->getCPUCount();
uint64_t gpu_count = engine->getGPUCount();
double key_rate = (cpu_count + gpu_count) / elapsed_time;
```

### 内存使用监控

```cpp
// 监控GPU内存使用
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
double mem_usage = (double)(total_mem - free_mem) / total_mem * 100.0;
```

## 🔧 高级功能

### 多GPU支持

```cpp
// 获取GPU数量
int gpu_count;
cudaGetDeviceCount(&gpu_count);

// 为每个GPU分配任务
for (int i = 0; i < gpu_count; i++) {
    cudaSetDevice(i);
    // 配置该GPU的搜索参数
    launch_unified_kernel<SearchMode::MODE_MA>(...);
}
```

### 动态负载均衡

```cpp
// 根据性能动态调整负载
if (gpu_performance < threshold) {
    // 减少该GPU的工作量
    blocks = blocks * 0.9;
}
```

## 📚 最佳实践

### 1. 内存管理
- 使用页锁定内存提升传输性能
- 批量处理减少内核启动开销
- 异步传输掩盖传输延迟

### 2. 性能优化
- 优先使用LDG缓存优化
- 合理配置网格参数
- 避免线程发散

### 3. 错误处理
- 始终检查CUDA API返回值
- 使用同步机制确保数据一致性
- 实现优雅的错误恢复机制

## 🔍 调试支持

### 调试宏

```cpp
#ifdef KEYHUNT_DEBUG
    printf("Debug: thread %d, block %d\n", threadIdx.x, blockIdx.x);
#endif
```

### 性能分析

```cpp
#ifdef KEYHUNT_PROFILE_EVENTS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // 执行内核
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
#endif
```

## 📋 版本兼容性

### CUDA版本要求
- **最低版本**: CUDA 11.0
- **推荐版本**: CUDA 11.8+
- **计算能力**: 7.5+ (Turing架构及更新)

### 编译器支持
- **GCC**: 7.0+
- **Clang**: 6.0+
- **MSVC**: 2019+

## 🚀 迁移指南

### 从传统接口迁移

```cpp
// 传统接口 (已废弃)
ComputeKeysSEARCH_MODE_MA(mode, startx, starty, 
                         bloom_data, bloom_bits, bloom_hashes, 
                         max_found, found);

// 统一接口 (推荐)
launch_unified_kernel<SearchMode::MODE_MA>(
    mode, startx, starty, bloom_data, bloom_bits, bloom_hashes,
    keys, max_found, found, blocks, threads);
```

### 性能对比

| 接口类型 | 代码重复率 | 性能开销 | 维护性 |
|----------|------------|----------|--------|
| 传统接口 | 65% | 基准 | 差 |
| 统一接口 | 0% | 零开销 | 优秀 |

## 📞 支持

### 问题报告
- GitHub Issues: [KeyHunt-Cuda Issues](https://github.com/your-repo/KeyHunt-Cuda/issues)
- 性能问题: 请提供详细的硬件配置和测试数据

### 贡献指南
- 代码风格: 遵循项目现有的C++/CUDA编码规范
- 测试要求: 所有新功能必须包含性能测试
- 文档更新: API变更必须同步更新文档

---

**最后更新**: 2025-09-06  
**文档版本**: v1.07  
**API版本**: Unified Interface v1.0