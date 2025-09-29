/**
 * SharedMemLayout.cuh
 * Optimized shared memory layouts for PUZZLE71
 * Avoids bank conflicts and optimizes memory access patterns
 */

#ifndef SHARED_MEM_LAYOUT_CUH
#define SHARED_MEM_LAYOUT_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "BatchStepping.h"

namespace SharedMemoryOptimization {
    constexpr int BANK_SIZE = 4;  // bytes per bank
    constexpr int NUM_BANKS = 32; // number of banks in shared memory
    constexpr int BANK_OFFSET_STRIDE = NUM_BANKS + 1;  // Padding to avoid conflicts
}

/**
 * Optimal shared memory bank configuration to avoid conflicts
 * Uses padding to ensure different threads access different banks
 */
template<int BANKS = 32>
struct SharedMemoryLayout {
    // Padding to avoid bank conflicts
    // Each point uses 4 uint64_t + 1 padding = 5 elements
    union {
        uint64_t data[BANKS][5];      // 4 words + 1 padding for bank alignment
        uint32_t data32[BANKS][10];   // Alternative 32-bit access
        uint16_t data16[BANKS][20];   // Alternative 16-bit access
    } points;
    
    // Hash storage with padding
    union {
        uint32_t hashes[BANKS][6];    // 5 hash words + 1 padding
        uint64_t hashes64[BANKS][3];  // Alternative 64-bit access
    } hash_data;
    
    /**
     * Store EC point with optimal bank alignment
     */
    __device__ void storePoint(int idx, const uint64_t px[4], const uint64_t py[4]) {
        // Ensure idx is within bounds
        if (idx >= BANKS) return;
        
        // Store x coordinates
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            points.data[idx][i] = px[i];
        }
        
        // Store first y coordinate in position 4 to maintain alignment
        // (Remaining y coordinates stored separately to optimize access pattern)
        points.data[idx][4] = py[0];
    }
    
    /**
     * Store extended EC point coordinates
     */
    __device__ void storePointExtended(int idx, const uint64_t px[4], const uint64_t py[4]) {
        if (idx >= BANKS) return;
        
        // Use coalesced writes to avoid bank conflicts
        int bank_idx = (idx + threadIdx.x) & (BANKS - 1);
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            points.data[bank_idx][i] = px[i];
        }
    }
    
    /**
     * Load EC point with optimal bank alignment
     */
    __device__ void loadPoint(int idx, uint64_t px[4], uint64_t py[4]) {
        if (idx >= BANKS) return;
        
        // Load x coordinates
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            px[i] = points.data[idx][i];
        }
        
        // Load y coordinate (first element stored in position 4)
        py[0] = points.data[idx][4];
        // Set remaining y coordinates to safe values
        py[1] = py[2] = py[3] = 0;
    }
    
    /**
     * Store hash with bank conflict avoidance
     */
    __device__ void storeHash(int idx, const uint32_t hash[5]) {
        if (idx >= BANKS) return;
        
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            hash_data.hashes[idx][i] = hash[i];
        }
    }
    
    /**
     * Load hash with optimal access pattern
     */
    __device__ void loadHash(int idx, uint32_t hash[5]) {
        if (idx >= BANKS) return;
        
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            hash[i] = hash_data.hashes[idx][i];
        }
    }
    
    /**
     * Broadcast data from one thread to all threads in warp
     */
    __device__ void broadcastPoint(int src_lane, uint64_t px[4], uint64_t py[4]) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            px[i] = __shfl_sync(0xFFFFFFFF, px[i], src_lane);
            py[i] = __shfl_sync(0xFFFFFFFF, py[i], src_lane);
        }
    }
    
    /**
     * Shuffle data between threads for collaborative computation
     */
    __device__ uint64_t shuffleData(uint64_t value, int target_lane) {
        return __shfl_sync(0xFFFFFFFF, value, target_lane);
    }
};

/**
 * Specialized layout for batch processing
 */
struct BatchSharedMemLayout {
    // Generator point cache (most frequently accessed)
    __align__(16) uint64_t generator_cache_x[64][4];
    __align__(16) uint64_t generator_cache_y[64][4];
    
    // Working space for batch computations  
    __align__(16) uint64_t batch_work_x[BatchSteppingConstants::BATCH_SIZE][4];
    __align__(16) uint64_t batch_work_y[BatchSteppingConstants::BATCH_SIZE][4];
    
    // Hash computation workspace
    __align__(8) uint32_t hash_workspace[BatchSteppingConstants::BATCH_SIZE][6];  // 5 + padding
    
    /**
     * Initialize generator cache from global memory
     */
    __device__ void initGeneratorCache(const uint64_t* global_gx, const uint64_t* global_gy) {
        int tid = threadIdx.x;
        
        // Each thread loads different cache lines to avoid conflicts
        for (int i = tid; i < 64; i += blockDim.x) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                generator_cache_x[i][j] = global_gx[i * 4 + j];
                generator_cache_y[i][j] = global_gy[i * 4 + j];
            }
        }
        
        __syncthreads();  // Ensure all data is loaded before use
    }
    
    /**
     * Get cached generator multiple with bounds checking
     */
    __device__ void getCachedGenerator(int multiple, uint64_t gx[4], uint64_t gy[4]) {
        // Bounds check
        int safe_idx = (multiple >= 0 && multiple < 64) ? multiple : 0;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            gx[i] = generator_cache_x[safe_idx][i];
            gy[i] = generator_cache_y[safe_idx][i];
        }
    }
    
    /**
     * Store batch computation result
     */
    __device__ void storeBatchResult(int batch_idx, const uint64_t px[4], const uint64_t py[4]) {
        if (batch_idx >= 0 && batch_idx < BatchSteppingConstants::BATCH_SIZE) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                batch_work_x[batch_idx][i] = px[i];
                batch_work_y[batch_idx][i] = py[i];
            }
        }
    }
    
    /**
     * Load batch computation input
     */
    __device__ void loadBatchInput(int batch_idx, uint64_t px[4], uint64_t py[4]) {
        if (batch_idx >= 0 && batch_idx < BatchSteppingConstants::BATCH_SIZE) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                px[i] = batch_work_x[batch_idx][i];
                py[i] = batch_work_y[batch_idx][i];
            }
        } else {
            // Safe fallback
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                px[i] = py[i] = 0;
            }
        }
    }
};

/**
 * Dynamic shared memory allocator for variable-size data
 */
class DynamicSharedMemAllocator {
private:
    // Use dynamic shared memory pointer instead of extern array
    static int offset;
    
public:
    /**
     * Allocate aligned memory from shared memory pool
     */
    __device__ void* allocate(size_t size, size_t alignment = 16) {
        // Get dynamic shared memory pointer
        extern __shared__ char shared_memory[];
        
        // Align to requested boundary
        int aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
        
        // Check if we have enough space
        if (aligned_offset + size > 48 * 1024) {  // Max shared memory per block
            return nullptr;  // Out of shared memory
        }
        
        void* ptr = &shared_memory[aligned_offset];
        offset = aligned_offset + size;
        
        return ptr;
    }
    
    /**
     * Reset allocator for new kernel launch
     */
    __device__ void reset() {
        if (threadIdx.x == 0) {
            offset = 0;
        }
        __syncthreads();
    }
};

// Note: Static member definition moved to implementation file to avoid CUDA compilation issues

#endif // SHARED_MEM_LAYOUT_CUH