/*
 * Phase 3.2 - Memory and Cache Optimization Module
 * Implements L1/L2 cache optimizations, shared memory layout,
 * and memory coalescing patterns for PUZZLE71
 */

#ifndef MEM_CACHE_OPT_CUH
#define MEM_CACHE_OPT_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// Cache hints for different access patterns
enum CacheHint {
    CACHE_STREAMING,     // Data used once (bypass cache)
    CACHE_PERSISTENT,    // Data reused frequently (keep in cache)
    CACHE_NORMAL        // Default caching behavior
};

/**
 * L2 Cache Configuration for optimal performance
 */
__device__ __forceinline__ void configure_l2_cache() {
    // Set L2 cache preference for persistent data
    // This helps keep generator tables in L2
    #if __CUDA_ARCH__ >= 800
    // For Ampere and newer GPUs
    cudaFuncSetAttribute(configure_l2_cache,
                        cudaFuncAttributePreferredSharedMemoryCarveout,
                        cudaSharedmemCarveoutMaxL1);
    #endif
}

/**
 * Prefetch data into L1/L2 cache
 */
template<CacheHint hint = CACHE_NORMAL>
__device__ __forceinline__ void cache_prefetch(const void* ptr, size_t size) {
    #if __CUDA_ARCH__ >= 700
    // Use CUDA-specific prefetch through read intrinsics
    // __builtin_prefetch is not available in device code
    switch(hint) {
        case CACHE_STREAMING:
            // Use streaming load to prefetch
            (void)__ldcs((const int*)ptr);
            break;
        case CACHE_PERSISTENT:
            // Use global load to prefetch to cache
            (void)__ldg((const int*)ptr);
            break;
        case CACHE_NORMAL:
        default:
            // Simple load to trigger cache prefetch
            volatile const int* vptr = (const int*)ptr;
            (void)*vptr;
            break;
    }
    #endif
}

/**
 * Load data with cache control
 */
template<typename T, CacheHint hint = CACHE_NORMAL>
__device__ __forceinline__ T load_cached(const T* ptr) {
    T value;
    
    #if __CUDA_ARCH__ >= 700
    switch(hint) {
        case CACHE_STREAMING:
            // Load with streaming hint (cg - cache global)
            value = __ldcs(ptr);  // Load streaming
            break;
        case CACHE_PERSISTENT:
            // Load with L1 cache hint (ca - cache all levels)
            value = __ldg(ptr);   // Load through texture cache (persistent)
            break;
        case CACHE_NORMAL:
        default:
            value = *ptr;
            break;
    }
    #else
    value = *ptr;
    #endif
    
    return value;
}

/**
 * Optimized shared memory layout to avoid bank conflicts
 * Banks are 32-bit wide, we need to avoid threads accessing same bank
 */
template<typename T, int BLOCK_SIZE>
struct SharedMemoryLayout {
    // Add padding to avoid bank conflicts
    static constexpr int PADDING = (sizeof(T) < 4) ? 1 : 0;
    static constexpr int STRIDE = BLOCK_SIZE + PADDING;
    
    T data[STRIDE];
    
    __device__ __forceinline__ T& operator[](int idx) {
        return data[idx];
    }
    
    __device__ __forceinline__ const T& operator[](int idx) const {
        return data[idx];
    }
};

/**
 * Coalesced memory access pattern for batch processing
 * Ensures consecutive threads access consecutive memory addresses
 */
template<typename T>
__device__ __forceinline__ void coalesced_load(
    T* dest,
    const T* src,
    int count,
    int thread_id,
    int block_size)
{
    // Each thread loads elements in a coalesced pattern
    for (int i = thread_id; i < count; i += block_size) {
        dest[i] = load_cached<T, CACHE_NORMAL>(&src[i]);
    }
}

/**
 * Vectorized memory operations for better throughput
 */
__device__ __forceinline__ void load_uint64x4(
    uint64_t dest[4],
    const uint64_t* src)
{
    // Use 128-bit loads when possible
    #if __CUDA_ARCH__ >= 700
    // Cast to uint4 for vectorized load
    uint4* dest_vec = reinterpret_cast<uint4*>(dest);
    const uint4* src_vec = reinterpret_cast<const uint4*>(src);
    *dest_vec = __ldg(src_vec);
    #else
    // Fallback to individual loads
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        dest[i] = src[i];
    }
    #endif
}

/**
 * Store with write-through cache policy
 */
template<typename T>
__device__ __forceinline__ void store_writethrough(T* dst, const T& value) {
    #if __CUDA_ARCH__ >= 700
    __stcs(dst, value);  // Store with cache streaming
    #else
    *dst = value;
    #endif
}

/**
 * Warp-level memory coalescing utilities
 */
namespace WarpMemory {
    
    // Load data cooperatively across warp
    template<typename T>
    __device__ __forceinline__ void warp_load(
        T* dest,
        const T* src,
        int count)
    {
        const int lane = threadIdx.x & 31;
        const int warp_size = 32;
        
        // Each lane loads its portion
        for (int i = lane; i < count; i += warp_size) {
            dest[i] = load_cached<T, CACHE_PERSISTENT>(&src[i]);
        }
    }
    
    // Store data cooperatively across warp
    template<typename T>
    __device__ __forceinline__ void warp_store(
        T* dest,
        const T* src,
        int count)
    {
        const int lane = threadIdx.x & 31;
        const int warp_size = 32;
        
        for (int i = lane; i < count; i += warp_size) {
            store_writethrough(&dest[i], src[i]);
        }
    }
}

/**
 * Optimized EC point structure with cache-friendly layout
 */
struct CacheOptimizedPoint {
    // Align to cache line boundary (128 bytes)
    __align__(128) uint64_t x[4];
    __align__(128) uint64_t y[4];
    
    __device__ __forceinline__ void load_from(const uint64_t* src_x, const uint64_t* src_y) {
        load_uint64x4(x, src_x);
        load_uint64x4(y, src_y);
    }
    
    __device__ __forceinline__ void store_to(uint64_t* dst_x, uint64_t* dst_y) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            store_writethrough(&dst_x[i], x[i]);
            store_writethrough(&dst_y[i], y[i]);
        }
    }
};

/**
 * Texture memory binding for generator tables (deprecated in newer CUDA)
 * Using __ldg intrinsics instead for better performance
 */
template<typename T>
class TextureCache {
private:
    const T* data;
    size_t size;
    
public:
    __device__ TextureCache(const T* ptr, size_t sz) : data(ptr), size(sz) {}
    
    __device__ __forceinline__ T operator[](int idx) const {
        #if __CUDA_ARCH__ >= 700
        return __ldg(&data[idx]);  // Load through texture cache
        #else
        return data[idx];
        #endif
    }
};

/**
 * Shared memory allocator with bank conflict avoidance
 */
template<int BLOCK_SIZE>
class SharedMemoryAllocator {
private:
    static uint8_t* pool;
    static size_t* offset;
    
public:
    template<typename T>
    __device__ static T* allocate(size_t count) {
        // Use external shared memory allocation
        extern __shared__ uint8_t shared_pool[];
        static __shared__ size_t shared_offset;
        
        // Align to avoid bank conflicts
        size_t alignment = (sizeof(T) >= 4) ? sizeof(T) : 4;
        size_t aligned_offset = (shared_offset + alignment - 1) & ~(alignment - 1);
        
        T* ptr = reinterpret_cast<T*>(&shared_pool[aligned_offset]);
        shared_offset = aligned_offset + count * sizeof(T);
        
        return ptr;
    }
    
    __device__ static void reset() {
        static __shared__ size_t shared_offset;
        if (threadIdx.x == 0) {
            shared_offset = 0;
        }
        __syncthreads();
    }
};

// Define static members - using external shared memory
template<int BLOCK_SIZE>
uint8_t* SharedMemoryAllocator<BLOCK_SIZE>::pool = nullptr;

template<int BLOCK_SIZE>
size_t* SharedMemoryAllocator<BLOCK_SIZE>::offset = nullptr;

/**
 * Memory fence operations for consistency
 */
__device__ __forceinline__ void memory_fence_block() {
    __threadfence_block();
}

__device__ __forceinline__ void memory_fence_device() {
    __threadfence();
}

/**
 * Additional utility functions for GPU kernels
 */
__device__ __forceinline__ void store_uint64x4(
    uint64_t* dest,
    const uint64_t src[4])
{
    // Use vectorized store when possible
    #if __CUDA_ARCH__ >= 700
    // Cast to uint4 for vectorized store
    uint4* dest_vec = reinterpret_cast<uint4*>(dest);
    const uint4* src_vec = reinterpret_cast<const uint4*>(src);
    __stcs(dest_vec, *src_vec);  // Store with cache streaming
    #else
    // Fallback to individual stores
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        dest[i] = src[i];
    }
    #endif
}

__device__ __forceinline__ void prefetch_l2(
    const void* ptr,
    size_t size)
{
    #if __CUDA_ARCH__ >= 700
    // Use global load intrinsic to prefetch to L2 cache
    // __builtin_prefetch is not available in device code
    const int* sptr = (const int*)ptr;
    (void)__ldg(sptr);
    #endif
}

/**
 * Batch memory operations with optimal access patterns
 */
namespace BatchMemory {
    
    // Copy with coalesced access
    template<typename T>
    __device__ void batch_copy(
        T* dest,
        const T* src,
        size_t count,
        int thread_id,
        int block_size)
    {
        // Each thread copies strided elements for coalescing
        for (size_t i = thread_id; i < count; i += block_size) {
            dest[i] = load_cached<T, CACHE_NORMAL>(&src[i]);
        }
        __syncthreads();
    }
    
    // Transpose for better access patterns
    template<typename T, int TILE_SIZE>
    __device__ void batch_transpose(
        T* dest,
        const T* src,
        int rows,
        int cols)
    {
        __shared__ T tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
        
        int x = blockIdx.x * TILE_SIZE + threadIdx.x;
        int y = blockIdx.y * TILE_SIZE + threadIdx.y;
        
        // Load tile cooperatively
        if (x < cols && y < rows) {
            tile[threadIdx.y][threadIdx.x] = src[y * cols + x];
        }
        __syncthreads();
        
        // Write transposed tile
        x = blockIdx.y * TILE_SIZE + threadIdx.x;
        y = blockIdx.x * TILE_SIZE + threadIdx.y;
        
        if (x < rows && y < cols) {
            dest[y * rows + x] = tile[threadIdx.x][threadIdx.y];
        }
    }
}

#endif // MEM_CACHE_OPT_CUH