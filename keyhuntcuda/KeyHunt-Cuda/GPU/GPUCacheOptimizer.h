/*
 * KeyHunt-Cuda 重新设计的缓存优化模块
 *
 * 目标: 基于性能测试报告，实施渐进式缓存优化
 * 作者: AI Agent - Expert-CUDA-C++-Architect
 * 日期: 2025-09-06
 *
 * 设计原则:
 * 1. 渐进式优化 - 从最简单的__ldg指令开始
 * 2. 数据驱动 - 基于实际profiling数据决策
 * 3. 零开销 - 避免额外的函数调用和同步
 * 4. 可回退 - 所有优化都通过宏控制
 */

#ifndef GPU_CACHE_OPTIMIZER_H
#define GPU_CACHE_OPTIMIZER_H

// 阶段1: LDG优化 - 仅使用__ldg指令访问只读数据
// 这是最安全的优化，零额外开销，预期2-5%性能提升
#ifdef KEYHUNT_CACHE_LDG_OPTIMIZED
#define LOAD_GX(i) __ldg(&Gx[(i) * 4])
#define LOAD_GY(i) __ldg(&Gy[(i) * 4])
#define LOAD_GX_COMPONENT(i, comp) __ldg(&Gx[(i) * 4 + (comp)])
#define LOAD_GY_COMPONENT(i, comp) __ldg(&Gy[(i) * 4 + (comp)])
#else
#define LOAD_GX(i) (Gx[(i) * 4])
#define LOAD_GY(i) (Gy[(i) * 4])
#define LOAD_GX_COMPONENT(i, comp) (Gx[(i) * 4 + (comp)])
#define LOAD_GY_COMPONENT(i, comp) (Gy[(i) * 4 + (comp)])
#endif

// 阶段2: 预取优化 (未来实施，需要验证阶段1效果)
// 在计算当前点的同时预取下一个点的数据
#ifdef KEYHUNT_CACHE_PREFETCH_OPTIMIZED
#define PREFETCH_GX_GY(i) \
    __prefetch_global_l1(&Gx[(i) * 4], 4); \
    __prefetch_global_l1(&Gy[(i) * 4], 4)
#define PREFETCH_GX_GY_COMPONENT(i, comp) \
    __prefetch_global_l1(&Gx[(i) * 4 + (comp)], 1); \
    __prefetch_global_l1(&Gy[(i) * 4 + (comp)], 1)
#else
#define PREFETCH_GX_GY(i)
#define PREFETCH_GX_GY_COMPONENT(i, comp)
#endif

// 阶段3: 访问模式优化 (深度分析后实施)
// 基于Nsight Compute分析结果优化内存访问模式
#ifdef KEYHUNT_CACHE_ACCESS_PATTERN_OPTIMIZED
// 需要基于实际分析数据设计具体实现
#endif

// 性能监控宏 - 用于跟踪优化效果
#ifdef KEYHUNT_CACHE_DEBUG
#define CACHE_DEBUG_PRINT(fmt, ...) printf("[CACHE] " fmt "\n", ##__VA_ARGS__)
#define CACHE_COUNTER_INC(counter) atomicAdd(&counter, 1)
#else
#define CACHE_DEBUG_PRINT(fmt, ...)
#define CACHE_COUNTER_INC(counter)
#endif

// 性能计数器 (用于分析优化效果)
#ifdef KEYHUNT_CACHE_PROFILE
extern __device__ uint32_t cache_ldg_count;
extern __device__ uint32_t cache_hit_count;
extern __device__ uint32_t cache_miss_count;

#define CACHE_PROFILE_LDG_ACCESS() CACHE_COUNTER_INC(cache_ldg_count)
#define CACHE_PROFILE_HIT() CACHE_COUNTER_INC(cache_hit_count)
#define CACHE_PROFILE_MISS() CACHE_COUNTER_INC(cache_miss_count)
#else
#define CACHE_PROFILE_LDG_ACCESS()
#define CACHE_PROFILE_HIT()
#define CACHE_PROFILE_MISS()
#endif

#endif // GPU_CACHE_OPTIMIZER_H
