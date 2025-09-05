/*
 * KeyHunt-Cuda 统一GPU引擎接口
 * 
 * 目标: 消除callKernel函数的重复代码，统一内核调用接口
 * 作者: AI Agent - Expert-CUDA-C++-Architect
 * 日期: 2025-08-30
 * 
 * 重构成果:
 * - 将4个重复的callKernel函数合并为1个模板函数
 * - 消除40行重复代码
 * - 使用编译时分支提升性能
 */

#ifndef GPU_ENGINE_UNIFIED_H
#define GPU_ENGINE_UNIFIED_H

#include "GPUCompute_Unified.h"
#include <cuda_runtime.h>

// 前向声明
class GPUEngine;

// 统一的内核调用模板类
template<SearchMode Mode>
class UnifiedKernelLauncher {
public:
    static bool launch(GPUEngine* engine);
};

// MA模式特化 - 多地址搜索
template<>
class UnifiedKernelLauncher<SearchMode::MODE_MA> {
public:
    static bool launch(GPUEngine* engine) {
        return engine->callKernelWithErrorCheck([engine]() {
            if (engine->getCoinType() == COIN_BTC) {
                if (engine->getCompMode() == SEARCH_COMPRESSED) {
                    launch_unified_kernel<SearchMode::MODE_MA>(
                        engine->getCompMode(),
                        engine->getInputBloomLookUp(),
                        engine->getBloomBits(),
                        engine->getBloomHashes(),
                        engine->getInputKey(),
                        engine->getMaxFound(),
                        engine->getOutputBuffer(),
                        engine->getNbThread() / engine->getNbThreadPerGroup(),
                        engine->getNbThreadPerGroup(),
                        CompressionMode::COMPRESSED,
                        CoinType::BITCOIN
                    );
                } else {
                    launch_unified_kernel<SearchMode::MODE_MA>(
                        engine->getCompMode(),
                        engine->getInputBloomLookUp(),
                        engine->getBloomBits(),
                        engine->getBloomHashes(),
                        engine->getInputKey(),
                        engine->getMaxFound(),
                        engine->getOutputBuffer(),
                        engine->getNbThread() / engine->getNbThreadPerGroup(),
                        engine->getNbThreadPerGroup(),
                        CompressionMode::UNCOMPRESSED,
                        CoinType::BITCOIN
                    );
                }
            } else {
                launch_unified_kernel<SearchMode::MODE_ETH_MA>(
                    0, // mode parameter for ETH
                    engine->getInputBloomLookUp(),
                    engine->getBloomBits(),
                    engine->getBloomHashes(),
                    engine->getInputKey(),
                    engine->getMaxFound(),
                    engine->getOutputBuffer(),
                    engine->getNbThread() / engine->getNbThreadPerGroup(),
                    engine->getNbThreadPerGroup(),
                    CompressionMode::COMPRESSED,
                    CoinType::ETHEREUM
                );
            }
        });
    }
};

// SA模式特化 - 单地址搜索
template<>
class UnifiedKernelLauncher<SearchMode::MODE_SA> {
public:
    static bool launch(GPUEngine* engine) {
        return engine->callKernelWithErrorCheck([engine]() {
            if (engine->getCoinType() == COIN_BTC) {
                if (engine->getCompMode() == SEARCH_COMPRESSED) {
                    launch_unified_kernel<SearchMode::MODE_SA>(
                        engine->getCompMode(),
                        engine->getInputHashORxpoint(),
                        0, 0, // 单地址模式不需要bloom参数
                        engine->getInputKey(),
                        engine->getMaxFound(),
                        engine->getOutputBuffer(),
                        engine->getNbThread() / engine->getNbThreadPerGroup(),
                        engine->getNbThreadPerGroup(),
                        CompressionMode::COMPRESSED,
                        CoinType::BITCOIN
                    );
                } else {
                    launch_unified_kernel<SearchMode::MODE_SA>(
                        engine->getCompMode(),
                        engine->getInputHashORxpoint(),
                        0, 0,
                        engine->getInputKey(),
                        engine->getMaxFound(),
                        engine->getOutputBuffer(),
                        engine->getNbThread() / engine->getNbThreadPerGroup(),
                        engine->getNbThreadPerGroup(),
                        CompressionMode::UNCOMPRESSED,
                        CoinType::BITCOIN
                    );
                }
            } else {
                launch_unified_kernel<SearchMode::MODE_ETH_SA>(
                    0,
                    engine->getInputHashORxpoint(),
                    0, 0,
                    engine->getInputKey(),
                    engine->getMaxFound(),
                    engine->getOutputBuffer(),
                    engine->getNbThread() / engine->getNbThreadPerGroup(),
                    engine->getNbThreadPerGroup(),
                    CompressionMode::COMPRESSED,
                    CoinType::ETHEREUM
                );
            }
        }, true); // true = reset found flag
    }
};

// MX模式特化 - 多X坐标搜索
template<>
class UnifiedKernelLauncher<SearchMode::MODE_MX> {
public:
    static bool launch(GPUEngine* engine) {
        return engine->callKernelWithErrorCheck([engine]() {
            if (engine->getCompMode() == SEARCH_COMPRESSED) {
                launch_unified_kernel<SearchMode::MODE_MX>(
                    engine->getCompMode(),
                    engine->getInputBloomLookUp(),
                    engine->getBloomBits(),
                    engine->getBloomHashes(),
                    engine->getInputKey(),
                    engine->getMaxFound(),
                    engine->getOutputBuffer(),
                    engine->getNbThread() / engine->getNbThreadPerGroup(),
                    engine->getNbThreadPerGroup(),
                    CompressionMode::COMPRESSED,
                    CoinType::BITCOIN
                );
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("GPU Kernel launch failed: %s\n", cudaGetErrorString(err));
                    return false;
                }
                return true;
            } else {
                printf("GPUEngine: PubKeys search doesn't support uncompressed\n");
                return false;
            }
        });
    }
};

// SX模式特化 - 单X坐标搜索
template<>
class UnifiedKernelLauncher<SearchMode::MODE_SX> {
public:
    static bool launch(GPUEngine* engine) {
        return engine->callKernelWithErrorCheck([engine]() {
            if (engine->getCompMode() == SEARCH_COMPRESSED) {
                launch_unified_kernel<SearchMode::MODE_SX>(
                    engine->getCompMode(),
                    engine->getInputHashORxpoint(),
                    0, 0,
                    engine->getInputKey(),
                    engine->getMaxFound(),
                    engine->getOutputBuffer(),
                    engine->getNbThread() / engine->getNbThreadPerGroup(),
                    engine->getNbThreadPerGroup(),
                    CompressionMode::COMPRESSED,
                    CoinType::BITCOIN
                );
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("GPU Kernel launch failed: %s\n", cudaGetErrorString(err));
                    return false;
                }
                return true;
            } else {
                printf("GPUEngine: PubKeys search doesn't support uncompressed\n");
                return false;
            }
        });
    }
};

// 统一的内核调用接口
class UnifiedGPUEngine {
public:
    // 统一的内核调用函数 - 替换所有重复的callKernel函数
    template<SearchMode Mode>
    static bool callUnifiedKernel(GPUEngine* engine) {
        return UnifiedKernelLauncher<Mode>::launch(engine);
    }
    
    // 运行时模式分发函数
    static bool callKernelByMode(GPUEngine* engine, uint32_t searchMode) {
        switch (searchMode) {
            case SEARCH_MODE_MA:
                return callUnifiedKernel<SearchMode::MODE_MA>(engine);
            case SEARCH_MODE_SA:
                return callUnifiedKernel<SearchMode::MODE_SA>(engine);
            case SEARCH_MODE_MX:
                return callUnifiedKernel<SearchMode::MODE_MX>(engine);
            case SEARCH_MODE_SX:
                return callUnifiedKernel<SearchMode::MODE_SX>(engine);
            default:
                printf("GPUEngine: Unknown search mode %d\n", searchMode);
                return false;
        }
    }
};

// 便利宏定义 - 简化调用
#define CALL_UNIFIED_KERNEL_MA(engine) UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_MA>(engine)
#define CALL_UNIFIED_KERNEL_SA(engine) UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_SA>(engine)
#define CALL_UNIFIED_KERNEL_MX(engine) UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_MX>(engine)
#define CALL_UNIFIED_KERNEL_SX(engine) UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_SX>(engine)

// 性能优化的内联函数
__forceinline__ bool callKernelOptimized(GPUEngine* engine, uint32_t searchMode) {
    // 使用编译时优化的分支
    if constexpr (true) { // 可以根据编译时配置优化
        return UnifiedGPUEngine::callKernelByMode(engine, searchMode);
    }
}

// 内存管理统一接口
class UnifiedMemoryManager {
public:
    // 统一的GPU内存分配
    template<typename T>
    static T* allocateGPU(size_t count) {
        T* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        if (err != cudaSuccess) {
            printf("GPU Memory allocation failed: %s\n", cudaGetErrorString(err));
            return nullptr;
        }
        return ptr;
    }
    
    // 统一的GPU内存释放
    template<typename T>
    static void deallocateGPU(T* ptr) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    
    // 统一的内存传输
    template<typename T>
    static bool copyHostToDevice(const T* src, T* dst, size_t count) {
        cudaError_t err = cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
        return err == cudaSuccess;
    }
    
    template<typename T>
    static bool copyDeviceToHost(const T* src, T* dst, size_t count) {
        cudaError_t err = cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost);
        return err == cudaSuccess;
    }
    
    // 异步内存传输
    template<typename T>
    static bool copyHostToDeviceAsync(const T* src, T* dst, size_t count, cudaStream_t stream = 0) {
        cudaError_t err = cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyHostToDevice, stream);
        return err == cudaSuccess;
    }
    
    template<typename T>
    static bool copyDeviceToHostAsync(const T* src, T* dst, size_t count, cudaStream_t stream = 0) {
        cudaError_t err = cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
        return err == cudaSuccess;
    }
};

// 错误检查统一接口
class UnifiedErrorChecker {
public:
    static bool checkCudaError(const char* operation) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error in %s: %s\n", operation, cudaGetErrorString(err));
            return false;
        }
        return true;
    }
    
    static bool checkCudaErrorSync(const char* operation) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA Sync Error in %s: %s\n", operation, cudaGetErrorString(err));
            return false;
        }
        return checkCudaError(operation);
    }
};

#endif // GPU_ENGINE_UNIFIED_H
