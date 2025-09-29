#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstdint>

namespace KeyHuntConstants {
    // 版本信息
    constexpr const char* VERSION = "1.0.8";
    
    // 默认配置值
    constexpr uint32_t DEFAULT_MAX_FOUND = 65536;  // 1024 * 64
    constexpr uint64_t DEFAULT_RANGE_END = 0xFFFFFFFFFFFFULL;
    constexpr size_t STACK_SIZE = 49152;
    
    // 数组和计算常量 (避免与GPUEngine.h中的宏冲突)
    constexpr uint32_t BYTE_ORDER_PATTERN = 0x0123;
    constexpr uint32_t DEFAULT_ITEM_SIZE_A = 256;
    constexpr uint32_t DEFAULT_ITEM_SIZE_X = 384;
    constexpr uint32_t DEFAULT_ITEM_SIZE_A32 = DEFAULT_ITEM_SIZE_A / 32;
    constexpr uint32_t DEFAULT_ITEM_SIZE_X32 = DEFAULT_ITEM_SIZE_X / 32;
    
    // 椭圆曲线计算常量
    constexpr uint32_t ELLIPTIC_CURVE_GROUP_SIZE = 2048;  // 1024*2
    constexpr uint32_t ELLIPTIC_CURVE_HALF_GROUP_SIZE = ELLIPTIC_CURVE_GROUP_SIZE / 2 - 1;  // 1023
    
    // 搜索模式常量 (避免与GPUEngine.h中的宏冲突)
    constexpr uint32_t DEFAULT_SEARCH_MODE_MA = 1;   // Multiple Addresses
    constexpr uint32_t DEFAULT_SEARCH_MODE_SA = 2;   // Single Address
    constexpr uint32_t DEFAULT_SEARCH_MODE_MX = 3;   // Multiple X-points
    constexpr uint32_t DEFAULT_SEARCH_MODE_SX = 4;   // Single X-point
    
    // 压缩模式常量 (避免与GPUEngine.h中的宏冲突)
    constexpr uint32_t DEFAULT_SEARCH_COMPRESSED = 0;
    constexpr uint32_t DEFAULT_SEARCH_UNCOMPRESSED = 1;
    constexpr uint32_t DEFAULT_SEARCH_BOTH = 2;
    
    // 币种类型常量 (避免与GPUEngine.h中的宏冲突)
    constexpr uint32_t DEFAULT_COIN_BTC = 1;
    constexpr uint32_t DEFAULT_COIN_ETH = 2;
    
    // GPU相关常量
    constexpr int DEFAULT_GPU_THREADS_PER_BLOCK = 128;
    constexpr int DEFAULT_GPU_AUTO_GRID_FLAG = -1;
    
    // 内存相关常量
    constexpr size_t DEFAULT_GPU_MEMORY_ALIGNMENT = 32;
    constexpr size_t CUDA_STACK_SIZE = 49152;  // Stack size for CUDA kernels
    
    // 计算相关常量
    constexpr size_t GROUP_HALF_SIZE = ELLIPTIC_CURVE_GROUP_SIZE / 2;
    constexpr size_t DX_ARRAY_SIZE = GROUP_HALF_SIZE + 1;
    
    // 文件和输出常量
    constexpr const char* DEFAULT_OUTPUT_FILE = "Found.txt";
    constexpr size_t MAX_LINE_LENGTH = 256;
    
    // 性能常量
    constexpr int DEFAULT_MULTIPROCESSOR_MULTIPLIER = 8;
    constexpr int CPU_SLEEP_INTERVAL_MS = 1;  // CPU sleep interval for async operations
}

// 为保持向后兼容性，定义宏别名
#define KEYHUNT_ITEM_SIZE_A KeyHuntConstants::DEFAULT_ITEM_SIZE_A
#define KEYHUNT_ITEM_SIZE_X KeyHuntConstants::DEFAULT_ITEM_SIZE_X
#define KEYHUNT_ITEM_SIZE_A32 KeyHuntConstants::DEFAULT_ITEM_SIZE_A32
#define KEYHUNT_ITEM_SIZE_X32 KeyHuntConstants::DEFAULT_ITEM_SIZE_X32

// 椭圆曲线计算宏别名
#define GRP_SIZE KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE
#define HSIZE KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE

#define KEYHUNT_SEARCH_MODE_MA KeyHuntConstants::DEFAULT_SEARCH_MODE_MA
#define KEYHUNT_SEARCH_MODE_SA KeyHuntConstants::DEFAULT_SEARCH_MODE_SA
#define KEYHUNT_SEARCH_MODE_MX KeyHuntConstants::DEFAULT_SEARCH_MODE_MX
#define KEYHUNT_SEARCH_MODE_SX KeyHuntConstants::DEFAULT_SEARCH_MODE_SX

#define KEYHUNT_SEARCH_COMPRESSED KeyHuntConstants::DEFAULT_SEARCH_COMPRESSED
#define KEYHUNT_SEARCH_UNCOMPRESSED KeyHuntConstants::DEFAULT_SEARCH_UNCOMPRESSED
#define KEYHUNT_SEARCH_BOTH KeyHuntConstants::DEFAULT_SEARCH_BOTH

#define KEYHUNT_COIN_BTC KeyHuntConstants::DEFAULT_COIN_BTC
#define KEYHUNT_COIN_ETH KeyHuntConstants::DEFAULT_COIN_ETH

#endif // CONSTANTS_H