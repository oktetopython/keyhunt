# KeyHunt-Cuda 开发者快速入门指南

## 🚀 5分钟快速开始

本指南帮助开发者快速理解和使用KeyHunt-Cuda的统一内核接口，从零开始构建和优化GPU加速的私钥搜索应用。

## 📋 前置要求

### 硬件要求
- **GPU**: NVIDIA GPU，计算能力7.5+ (Turing架构或更新)
- **内存**: 至少8GB GPU内存
- **CPU**: 支持SSSE3指令集

### 软件要求
- **操作系统**: Linux (推荐Ubuntu 20.04+)
- **CUDA Toolkit**: 11.8+
- **编译器**: GCC 7.0+ 或 Clang 6.0+
- **依赖库**: GMP (GNU Multiple Precision Arithmetic Library)

### 验证环境
```bash
# 检查CUDA安装
nvcc --version

# 检查GPU计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 检查GMP库
ldconfig -p | grep gmp
```

## 🔧 构建项目

### 1. 克隆项目
```bash
git clone https://github.com/your-repo/KeyHunt-Cuda.git
cd KeyHunt-Cuda
```

### 2. 编译项目
```bash
# 基础编译 (CPU only)
make all

# GPU加速编译 (推荐)
make gpu=1 CCAP=75 all

# 启用LDG缓存优化
make gpu=1 CCAP=75 NVCCFLAGS="-DKEYHUNT_CACHE_LDG_OPTIMIZED" all

# 调试模式
make gpu=1 CCAP=75 debug=1 all
```

### 3. 验证构建
```bash
# 检查可执行文件
ls -la KeyHunt

# 查看帮助信息
./KeyHunt --help
```

## 🎯 统一内核接口快速入门

### 基本概念

#### 1. 搜索模式
```cpp
enum class SearchMode {
    MODE_MA = 1,      // 多地址布隆过滤器搜索
    MODE_SA = 2,      // 单地址直接哈希匹配
    MODE_MX = 3,      // 多X坐标布隆过滤器搜索
    MODE_SX = 4,      // 单X坐标直接匹配
    MODE_ETH_MA = 5,  // 以太坊多地址搜索
    MODE_ETH_SA = 6   // 以太坊单地址搜索
};
```

#### 2. 压缩模式
```cpp
enum class CompressionMode : uint32_t {
    COMPRESSED = 0,     // 压缩公钥 (33字节)
    UNCOMPRESSED = 1    // 非压缩公钥 (65字节)
};
```

#### 3. 币种类型
```cpp
enum class CoinType : uint32_t {
    BITCOIN = 0,    // 比特币
    ETHEREUM = 1    // 以太坊
};
```

### 快速示例

#### 示例1: 多地址搜索 (5分钟完成)

```cpp
#include "GPU/GPUCompute_Unified.h"
#include <cuda_runtime.h>
#include <iostream>

int main() {
    // 1. 设置搜索参数
    const int NUM_THREADS = 65536;  // 256 blocks * 256 threads
    const int MAX_FOUND = 100;
    
    // 2. 准备布隆过滤器数据
    uint8_t bloom_data[1024] = {0};  // 示例布隆过滤器
    uint32_t bloom_bits = 8192;
    uint32_t bloom_hashes = 3;
    
    // 3. 分配设备内存
    uint64_t* d_keys;
    uint32_t* d_found;
    size_t keys_size = NUM_THREADS * 8 * sizeof(uint64_t);
    size_t found_size = MAX_FOUND * ITEM_SIZE_A32 * sizeof(uint32_t);
    
    cudaMalloc(&d_keys, keys_size);
    cudaMalloc(&d_found, found_size);
    cudaMemset(d_found, 0, found_size);
    
    // 4. 启动统一内核
    launch_unified_kernel<SearchMode::MODE_MA>(
        SEARCH_MODE_MA,           // 搜索模式
        bloom_data,               // 布隆过滤器数据
        bloom_bits,               // 布隆过滤器位数
        bloom_hashes,             // 布隆过滤器哈希数
        d_keys,                   // 密钥数组
        MAX_FOUND,                // 最大查找数量
        d_found,                  // 结果数组
        256, 256,                 // 网格配置: 256 blocks, 256 threads/block
        CompressionMode::COMPRESSED,  // 使用压缩公钥
        CoinType::BITCOIN         // 搜索比特币地址
    );
    
    // 5. 等待内核完成
    cudaDeviceSynchronize();
    
    // 6. 检查结果
    uint32_t h_found[MAX_FOUND * ITEM_SIZE_A32];
    cudaMemcpy(h_found, d_found, found_size, cudaMemcpyDeviceToHost);
    
    int num_found = h_found[0];
    std::cout << "Found " << num_found << " addresses" << std::endl;
    
    // 7. 清理资源
    cudaFree(d_keys);
    cudaFree(d_found);
    
    return 0;
}
```

#### 示例2: 单地址搜索 (3分钟完成)

```cpp
// 单地址搜索更简单
uint32_t target_hash[5] = {0x12345678, 0x9ABCDEF0, ...};  // 目标地址哈希

launch_unified_kernel<SearchMode::MODE_SA>(
    SEARCH_MODE_SA,           // 单地址模式
    target_hash,              // 直接传入目标哈希
    0, 0,                     // 单地址模式不需要布隆参数
    d_keys, MAX_FOUND, d_found,
    128, 128,                 // 较小的网格配置
    CompressionMode::BOTH,    // 同时搜索压缩和非压缩
    CoinType::BITCOIN
);
```

## 📊 性能调优

### 网格配置优化

#### 根据GPU架构选择配置
```cpp
// RTX 30xx系列 (Ampere)
const int BLOCKS = 320;      // 68 SM * 4-5 blocks/SM
const int THREADS = 256;     // 256 threads/block

// RTX 20xx系列 (Turing)  
const int BLOCKS = 256;      // 46 SM * 5-6 blocks/SM
const int THREADS = 256;     // 256 threads/block

// 通用配置
const int BLOCKS = 256;      // 安全默认值
const int THREADS = 256;     // 良好平衡
```

### 内存优化

#### 使用LDG缓存优化
```cpp
// 在编译时启用LDG优化
#define KEYHUNT_CACHE_LDG_OPTIMIZED

// 这将自动使用__ldg()指令访问只读数据
// 预期L1缓存命中率: 45% → 55%+
// 预期性能提升: 2-5%
```

#### 内存对齐
```cpp
// 确保256位对齐
size_t keys_size = num_threads * 8 * sizeof(uint64_t);
size_t aligned_size = ((keys_size + 31) / 32) * 32;

cudaMalloc(&d_keys, aligned_size);
```

## 🛠️ 错误处理

### CUDA错误检查
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// 使用示例
CUDA_CHECK(cudaMalloc(&d_keys, keys_size));
CUDA_CHECK(cudaMemcpy(d_keys, h_keys, keys_size, cudaMemcpyHostToDevice));
```

### 内核错误检查
```cpp
// 启动内核后检查错误
launch_unified_kernel<SearchMode::MODE_MA>(...);

cudaError_t kernel_error = cudaGetLastError();
if (kernel_error != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernel_error) << std::endl;
    return -1;
}

// 等待完成并检查执行错误
cudaError_t sync_error = cudaDeviceSynchronize();
if (sync_error != cudaSuccess) {
    std::cerr << "Kernel execution failed: " << cudaGetErrorString(sync_error) << std::endl;
    return -1;
}
```

## 📈 性能监控

### 实时性能统计
```cpp
// 获取性能计数
uint64_t cpu_count = engine.getCPUCount();
uint64_t gpu_count = engine.getGPUCount();
double elapsed_time = timer.getElapsedTime();
double key_rate = (cpu_count + gpu_count) / elapsed_time;

std::cout << "Performance: " << key_rate / 1000000.0 << " Mk/s" << std::endl;
```

### 内存使用监控
```cpp
// 监控GPU内存使用
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
double mem_usage = (double)(total_mem - free_mem) / total_mem * 100.0;

std::cout << "GPU memory usage: " << mem_usage << "%" << std::endl;
```

## 🔍 调试技巧

### 启用调试信息
```cpp
// 编译时启用调试
make gpu=1 CCAP=75 debug=1 all

// 代码中添加调试输出
#ifdef KEYHUNT_DEBUG
    printf("Thread %d, Block %d: processing key...\n", 
           threadIdx.x, blockIdx.x);
#endif
```

### 使用CUDA-GDB调试
```bash
# 编译调试版本
nvcc -G -g -O0 kernel.cu -o debug_kernel

# 使用CUDA-GDB调试
cuda-gdb ./debug_kernel
(gdb) run
(gdb) bt  # 查看调用栈
```

## 🚀 高级功能

### 多GPU支持
```cpp
int gpu_count;
cudaGetDeviceCount(&gpu_count);
std::cout << "Found " << gpu_count << " GPUs" << std::endl;

// 为每个GPU分配任务
for (int gpu = 0; gpu < gpu_count; gpu++) {
    cudaSetDevice(gpu);
    
    // 分配该GPU的内存
    cudaMalloc(&d_keys[gpu], keys_size);
    
    // 启动该GPU的内核
    launch_unified_kernel<SearchMode::MODE_MA>(...);
}
```

### 异步执行
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// 异步内存传输
cudaMemcpyAsync(d_keys, h_keys, keys_size, 
                cudaMemcpyHostToDevice, stream);

// 异步内核启动
launch_unified_kernel<SearchMode::MODE_MA>(...);

// 等待完成
cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

## 📚 完整示例项目

### 项目结构
```
my_keyhunt_project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── keyhunt_engine.cpp
│   └── keyhunt_engine.h
├── include/
│   └── KeyHunt-Cuda/  # 符号链接到KeyHunt-Cuda目录
└── build/
```

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.18)
project(MyKeyHunt)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(${CMAKE_SOURCE_DIR}/include)

cuda_add_executable(my_keyhunt 
    src/main.cpp
    src/keyhunt_engine.cpp
)

target_link_libraries(my_keyhunt 
    ${CUDA_LIBRARIES}
    gmp
)
```

### main.cpp
```cpp
#include "keyhunt_engine.h"

int main() {
    KeyHuntEngine engine;
    
    // 配置搜索参数
    engine.setSearchMode(SearchMode::MODE_MA);
    engine.setTargetFile("addresses.txt");
    engine.setRange("8000000000000000", "FFFFFFFFFFFFFF");
    engine.setGPU(true);
    engine.setGridSize(256, 256);
    
    // 开始搜索
    engine.start();
    
    // 等待完成
    engine.waitForCompletion();
    
    return 0;
}
```

## 📋 故障排除

### 常见问题1: 编译错误
```bash
# 错误: undefined reference to `__gmpz_init'
# 解决: 安装GMP开发库
sudo apt-get install libgmp-dev

# 错误: CUDA architecture not supported
# 解决: 检查GPU计算能力并调整CCAP
nvidia-smi --query-gpu=compute_cap --format=csv
```

### 常见问题2: 运行时错误
```bash
# 错误: out of memory
# 解决: 减少线程数量或分批处理
# 调整 BLOCKS 和 THREADS 参数

# 错误: invalid device function
# 解决: 检查计算能力设置是否正确
make gpu=1 CCAP=86 all  # RTX 30xx系列
```

### 常见问题3: 性能问题
```bash
# 性能低于预期
# 解决步骤:
1. 检查是否启用了LDG优化
2. 验证网格配置是否合适
3. 使用Nsight Compute分析瓶颈
4. 检查GPU温度和功耗限制
```

## 📞 获取帮助

### 文档资源
- 📖 [API参考文档](API_REFERENCE.md)
- 🎯 [性能优化指南](PERFORMANCE_OPTIMIZATION_GUIDE.md)
- 🔧 [完整项目示例](https://github.com/your-repo/KeyHunt-Cuda-Examples)

### 社区支持
- 💬 [GitHub Discussions](https://github.com/your-repo/KeyHunt-Cuda/discussions)
- 🐛 [Issue报告](https://github.com/your-repo/KeyHunt-Cuda/issues)
- 📧 邮件: support@keyhunt-cuda.org

---

**恭喜！** 🎉 您已经完成了KeyHunt-Cuda统一内核接口的快速入门。现在您可以开始构建自己的高性能私钥搜索应用了！

**下一步**: 查看[API参考文档](API_REFERENCE.md)了解更多高级功能，或使用[性能优化指南](PERFORMANCE_OPTIMIZATION_GUIDE.md)进一步提升性能。

**最后更新**: 2025-09-06  
**指南版本**: v1.07  
**预计学习时间**: 30分钟