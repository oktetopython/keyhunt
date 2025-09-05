/*
 * KeyHunt-Cuda 简化的L1缓存优化模块
 * 
 * 目标: 使用简单的__ldg指令优化只读数据访问
 * 作者: AI Assistant
 * 日期: 2025-09-05
 */

#ifndef GPU_CACHE_OPTIMIZER_H
#define GPU_CACHE_OPTIMIZER_H

// 简化的只读数据加载宏 - 使用__ldg指令
#ifdef KEYHUNT_SIMPLE_OPTIMIZED
#define LOAD_GX(i) __ldg(&Gx[(i) * 4])
#define LOAD_GY(i) __ldg(&Gy[(i) * 4])
#else
#define LOAD_GX(i) (Gx[(i) * 4])
#define LOAD_GY(i) (Gy[(i) * 4])
#endif

#endif // GPU_CACHE_OPTIMIZER_H
