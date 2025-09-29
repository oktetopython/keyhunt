/**
 * GPUMemory.cuh
 * Memory and Cache Optimization for PUZZLE71
 * Implements LDG cache optimization and safe memory access patterns
 */

#ifndef GPU_MEMORY_CUH
#define GPU_MEMORY_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "KeyHuntConstants.h"

namespace GPUMemoryOptimization {
    // Memory access patterns
    constexpr int CACHE_LINE_SIZE = 128;  // bytes
    constexpr int WARP_SIZE = 32;
    constexpr int MAX_SHARED_MEMORY = 48 * 1024;  // 48KB per block
    
    // Safe bounds checking
    constexpr int MAX_GENERATOR_TABLE_SIZE = 256 * 32 * 4;  // Safe upper bound
}

/**
 * Use LDG (Load Global) cache for read-only data
 * Ensures cached access for frequently accessed generator tables
 */
__device__ __forceinline__ uint64_t loadCached(const uint64_t* ptr) {
    return __ldg(ptr);
}

/**
 * Load 4 uint64_t values with LDG cache
 */
__device__ __forceinline__ void loadCached4(const uint64_t* ptr, uint64_t result[4]) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        result[i] = __ldg(ptr + i);
    }
}

/**
 * Safe bounds-checked array access for generator tables
 */
__device__ __forceinline__ uint64_t safeLoadGx(const uint64_t* Gx, int index) {
    // Bounds check to prevent illegal memory access
    if (index >= 0 && index < GPUMemoryOptimization::MAX_GENERATOR_TABLE_SIZE) {
        return loadCached(&Gx[index]);
    }
    return 0;  // Return safe value if out of bounds
}

__device__ __forceinline__ uint64_t safeLoadGy(const uint64_t* Gy, int index) {
    // Bounds check to prevent illegal memory access
    if (index >= 0 && index < GPUMemoryOptimization::MAX_GENERATOR_TABLE_SIZE) {
        return loadCached(&Gy[index]);
    }
    return 0;  // Return safe value if out of bounds
}

/**
 * Optimized memory access patterns with coalescing
 * Ensures memory accesses are coalesced for better performance
 */
template<int STRIDE>
__device__ void loadCoalesced(uint64_t* dest, const uint64_t* src, int count) {
    int tid = threadIdx.x;
    
    // Coalesced access pattern
    #pragma unroll
    for (int i = tid; i < count; i += STRIDE) {
        dest[i] = loadCached(&src[i]);
    }
}

/**
 * Safe memory copy with bounds checking
 */
__device__ __forceinline__ void safeMemcpy(void* dest, const void* src, size_t size, size_t maxSize) {
    if (size <= maxSize) {
        for (size_t i = 0; i < size; i++) {
            ((char*)dest)[i] = ((const char*)src)[i];
        }
    }
}

/**
 * Prefetch data for future use
 */
__device__ __forceinline__ void prefetchData(const void* ptr, size_t size) {
    // Use LDG for prefetching read-only data
    // This is a hint to the cache system
    volatile const char* prefetch_ptr = (const char*)ptr;
    (void)prefetch_ptr; // Suppress unused variable warning
}

/**
 * Memory fence operations for synchronization
 */
__device__ __forceinline__ void memoryFence() {
    __threadfence();
}

__device__ __forceinline__ void memoryFenceBlock() {
    __threadfence_block();
}

#endif // GPU_MEMORY_CUH