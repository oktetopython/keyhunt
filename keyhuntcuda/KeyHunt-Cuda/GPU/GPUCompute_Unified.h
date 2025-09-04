/*
 * KeyHunt-Cuda 统一GPU计算模块
 * 
 * 目标: 消除65%的代码重复，统一CUDA内核接口
 * 作者: AI Agent - Expert-CUDA-C++-Architect
 * 日期: 2025-08-30
 * 
 * 设计原则:
 * 1. 使用模板元编程统一不同搜索模式
 * 2. 编译时分支替代运行时分支，保持性能
 * 3. 保持原有算法逻辑不变，确保正确性
 */

#ifndef GPU_COMPUTE_UNIFIED_H
#define GPU_COMPUTE_UNIFIED_H

#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUCompute.h"  // 包含原始的检查函数声明
#include "GPUMemoryOptimized.h"  // 内存访问优化
#include "GPUProfiler.h"  // 设备侧性能分析
#include "GPUCacheOptimizer.h"  // L1缓存优化

// 搜索模式枚举
enum class SearchMode : uint32_t {
    MODE_MA = 0,  // Multiple Addresses (布隆过滤器)
    MODE_SA = 1,  // Single Address (单个哈希)
    MODE_MX = 2,  // Multiple X-points
    MODE_SX = 3,  // Single X-point
    MODE_ETH_MA = 4,  // Ethereum Multiple Addresses
    MODE_ETH_SA = 5   // Ethereum Single Address
};

// 压缩模式枚举
enum class CompressionMode : uint32_t {
    COMPRESSED = 0,
    UNCOMPRESSED = 1
};

// 币种类型枚举
enum class CoinType : uint32_t {
    BITCOIN = 0,
    ETHEREUM = 1
};

// 统一的检查函数模板特化声明
template<SearchMode Mode>
__device__ __forceinline__ void unified_check_hash(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint32_t param1, uint32_t param2,
    uint32_t maxFound, uint32_t* out);

// MA模式特化 - 多地址布隆过滤器检查
template<>
__device__ __forceinline__ void unified_check_hash<SearchMode::MODE_MA>(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint32_t bloom_bits, uint32_t bloom_hashes,
    uint32_t maxFound, uint32_t* out)
{
    const uint8_t* bloomLookUp = static_cast<const uint8_t*>(target_data);
    CheckHashSEARCH_MODE_MA(mode, px, py, incr, bloomLookUp, bloom_bits, bloom_hashes, maxFound, out);
}

// SA模式特化 - 单地址哈希检查
template<>
__device__ __forceinline__ void unified_check_hash<SearchMode::MODE_SA>(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint32_t param1, uint32_t param2,
    uint32_t maxFound, uint32_t* out)
{
    const uint32_t* hash160 = static_cast<const uint32_t*>(target_data);
    CheckHashSEARCH_MODE_SA(mode, px, py, incr, hash160, maxFound, out);
}

// MX模式特化 - 多X坐标检查
template<>
__device__ __forceinline__ void unified_check_hash<SearchMode::MODE_MX>(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint32_t bloom_bits, uint32_t bloom_hashes,
    uint32_t maxFound, uint32_t* out)
{
    const uint8_t* bloomLookUp = static_cast<const uint8_t*>(target_data);
    CheckPubSEARCH_MODE_MX(mode, px, py, incr, bloomLookUp, bloom_bits, bloom_hashes, maxFound, out);
}

// SX模式特化 - 单X坐标检查
template<>
__device__ __forceinline__ void unified_check_hash<SearchMode::MODE_SX>(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint32_t param1, uint32_t param2,
    uint32_t maxFound, uint32_t* out)
{
    const uint32_t* xpoint = static_cast<const uint32_t*>(target_data);
    CheckPubSEARCH_MODE_SX(mode, px, py, incr, xpoint, maxFound, out);
}

// 以太坊MA模式特化
template<>
__device__ __forceinline__ void unified_check_hash<SearchMode::MODE_ETH_MA>(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint32_t bloom_bits, uint32_t bloom_hashes,
    uint32_t maxFound, uint32_t* out)
{
    const uint8_t* bloomLookUp = static_cast<const uint8_t*>(target_data);
    CheckHashSEARCH_ETH_MODE_MA(px, py, incr, bloomLookUp, bloom_bits, bloom_hashes, maxFound, out);
}

// 以太坊SA模式特化
template<>
__device__ __forceinline__ void unified_check_hash<SearchMode::MODE_ETH_SA>(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint32_t param1, uint32_t param2,
    uint32_t maxFound, uint32_t* out)
{
    const uint32_t* hash = static_cast<const uint32_t*>(target_data);
    CheckHashSEARCH_ETH_MODE_SA(px, py, incr, hash, maxFound, out);
}

// 统一的椭圆曲线计算核心函数
// 这个函数合并了所有重复的椭圆曲线计算逻辑
template<SearchMode Mode>
__device__ void unified_compute_keys_core(
    uint32_t mode, 
    uint64_t* startx, 
    uint64_t* starty,
    const void* target_data,
    uint32_t param1,  // bloom_bits 或其他参数
    uint32_t param2,  // bloom_hashes 或其他参数
    uint32_t maxFound, 
    uint32_t* out)
{
    // 统一的变量声明 - 消除重复
    uint64_t dx[GRP_SIZE / 2 + 1][4];
    uint64_t px[4];
    uint64_t py[4];
    uint64_t pyn[4];
    uint64_t sx[4];
    uint64_t sy[4];
    uint64_t dy[4];
    uint64_t _s[4];
    uint64_t _p2[4];

    // 统一的起始点加载逻辑 - 消除重复
    __syncthreads();
    Load256A(sx, startx);
    Load256A(sy, starty);
    Load256(px, sx);
    Load256(py, sy);

    // 多层次内存优化的delta x计算 - 减少非合并访问，提升L1缓存命中率
    uint32_t i;

    #ifdef KEYHUNT_CACHE_OPTIMIZED
    // L1缓存优化路径 - 使用__ldg和预取
    __shared__ uint64_t shared_sx[4];
    if (threadIdx.x == 0) {
        Load256(shared_sx, sx);
    }
    __syncthreads();

    // 缓存感知的计算
    compute_dx_cache_optimized(dx, shared_sx, HSIZE + 2);

    #elif defined(KEYHUNT_MEMORY_OPTIMIZED)
    // 内存访问优化路径 - 使用共享内存缓存
    __shared__ uint64_t shared_sx[4];
    if (threadIdx.x == 0) {
        Load256(shared_sx, sx);
    }
    __syncthreads();

    // 优化的访问模式
    for (i = 0; i < HSIZE; i++) {
        uint64_t temp_gx[4];
        load_Gx_Gy_cached(i, temp_gx, nullptr);
        ModSub256(dx[i], temp_gx, shared_sx);
    }
    uint64_t temp_gx[4];
    load_Gx_Gy_cached(i, temp_gx, nullptr);
    ModSub256(dx[i], temp_gx, shared_sx);     // For the first point
    ModSub256(dx[i + 1], _2Gnx, shared_sx);  // For the next center point

    #else
    // 原始访问模式
    for (i = 0; i < HSIZE; i++)
        ModSub256(dx[i], Gx + 4 * i, sx);
    ModSub256(dx[i], Gx + 4 * i, sx);     // For the first point
    ModSub256(dx[i + 1], _2Gnx, sx);      // For the next center point
    #endif

    // 统一的模逆计算 - 消除重复，支持性能分析
    #ifdef KEYHUNT_PROFILE_INTERNAL
    _ModInvGrouped_Profiled(dx);
    #else
    _ModInvGrouped(dx);
    #endif

    // 统一的起始点检查 - 使用模板特化
    unified_check_hash<Mode>(mode, px, py, GRP_SIZE / 2, target_data, param1, param2, maxFound, out);

    ModNeg256(pyn, py);

    // 多层次优化的主循环 - 减少DRAM带宽压力，提升L1缓存命中率
    for (i = 0; i < HSIZE; i++) {
        // P = StartPoint + i*G
        Load256(px, sx);
        Load256(py, sy);

        #ifdef KEYHUNT_CACHE_OPTIMIZED
        uint64_t temp_gx[4], temp_gy[4];
        load_Gx_Gy_cache_aware(i, temp_gx, temp_gy);
        compute_ec_point_add_profiled(px, py, temp_gx, temp_gy, dx[i]);
        #elif defined(KEYHUNT_MEMORY_OPTIMIZED)
        uint64_t temp_gx[4], temp_gy[4];
        load_Gx_Gy_cached(i, temp_gx, temp_gy);
        compute_ec_point_add_profiled(px, py, temp_gx, temp_gy, dx[i]);
        #else
        compute_ec_point_add_profiled(px, py, Gx + 4 * i, Gy + 4 * i, dx[i]);
        #endif

        unified_check_hash<Mode>(mode, px, py, GRP_SIZE / 2 + (i + 1), target_data, param1, param2, maxFound, out);

        // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
        Load256(px, sx);

        #ifdef KEYHUNT_CACHE_OPTIMIZED
        compute_ec_point_add_negative_profiled(px, py, pyn, temp_gx, temp_gy, dx[i]);
        #elif defined(KEYHUNT_MEMORY_OPTIMIZED)
        compute_ec_point_add_negative_profiled(px, py, pyn, temp_gx, temp_gy, dx[i]);
        #else
        compute_ec_point_add_negative_profiled(px, py, pyn, Gx + 4 * i, Gy + 4 * i, dx[i]);
        #endif

        unified_check_hash<Mode>(mode, px, py, GRP_SIZE / 2 - (i + 1), target_data, param1, param2, maxFound, out);
    }

    // 统一的边界点处理 - 消除重复
    // First point (startP - (GRP_SIZE/2)*G) - 内存优化
    Load256(px, sx);
    Load256(py, sy);

    #ifdef KEYHUNT_MEMORY_OPTIMIZED
    uint64_t temp_gx[4], temp_gy[4];
    load_Gx_Gy_cached(i, temp_gx, temp_gy);
    compute_ec_point_add_special(px, py, temp_gx, temp_gy, dx[i], true);
    #else
    compute_ec_point_add_special(px, py, Gx + 4 * i, Gy + 4 * i, dx[i], true);
    #endif

    unified_check_hash<Mode>(mode, px, py, 0, target_data, param1, param2, maxFound, out);

    i++;

    // Next start point (startP + GRP_SIZE*G)
    Load256(px, sx);
    Load256(py, sy);
    compute_ec_point_add(px, py, _2Gnx, _2Gny, dx[i + 1]);

    // 统一的起始点更新 - 消除重复
    __syncthreads();
    Store256A(startx, px);
    Store256A(starty, py);

    // 报告性能分析结果
    #ifdef KEYHUNT_PROFILE_INTERNAL
    report_timing_stats();
    #endif
}

// 统一的CUDA内核函数模板
template<SearchMode Mode, CompressionMode Comp, CoinType Coin>
__global__ void unified_compute_keys_kernel(
    uint32_t mode,
    const void* target_data,
    uint32_t param1,  // bloom_bits, bloom_hashes等
    uint32_t param2,
    uint64_t* keys,
    uint32_t maxFound,
    uint32_t* found)
{
    int xPtr = (blockIdx.x * blockDim.x) * 8;
    int yPtr = xPtr + 4 * blockDim.x;
    
    // 调用统一的核心计算函数
    unified_compute_keys_core<Mode>(mode, keys + xPtr, keys + yPtr, 
                                   target_data, param1, param2, maxFound, found);
}

// 便利的内核启动函数
template<SearchMode Mode>
__host__ void launch_unified_kernel(
    uint32_t mode,
    const void* target_data,
    uint32_t param1,
    uint32_t param2,
    uint64_t* keys,
    uint32_t maxFound,
    uint32_t* found,
    uint32_t blocks,
    uint32_t threads_per_block,
    CompressionMode comp_mode = CompressionMode::COMPRESSED,
    CoinType coin_type = CoinType::BITCOIN)
{
    if (comp_mode == CompressionMode::COMPRESSED) {
        if (coin_type == CoinType::BITCOIN) {
            unified_compute_keys_kernel<Mode, CompressionMode::COMPRESSED, CoinType::BITCOIN>
                <<<blocks, threads_per_block>>>(mode, target_data, param1, param2, keys, maxFound, found);
        } else {
            unified_compute_keys_kernel<Mode, CompressionMode::COMPRESSED, CoinType::ETHEREUM>
                <<<blocks, threads_per_block>>>(mode, target_data, param1, param2, keys, maxFound, found);
        }
    } else {
        if (coin_type == CoinType::BITCOIN) {
            unified_compute_keys_kernel<Mode, CompressionMode::UNCOMPRESSED, CoinType::BITCOIN>
                <<<blocks, threads_per_block>>>(mode, target_data, param1, param2, keys, maxFound, found);
        } else {
            unified_compute_keys_kernel<Mode, CompressionMode::UNCOMPRESSED, CoinType::ETHEREUM>
                <<<blocks, threads_per_block>>>(mode, target_data, param1, param2, keys, maxFound, found);
        }
    }
}

#endif // GPU_COMPUTE_UNIFIED_H
