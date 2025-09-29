/**
 * GPUMathBatch.cu
 * Optimized batch modular inversion for PUZZLE71
 * Implements Montgomery batch inversion algorithm
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include "GPUMath.h"

// Montgomery batch inversion constants
#define BATCH_INV_SIZE 256  // Process 256 inversions at once
#define WARP_SIZE 32

/**
 * Montgomery batch inversion algorithm
 * Computes modular inverses of multiple elements efficiently
 * Based on Montgomery's trick: only one inversion + 3(n-1) multiplications
 */
__device__ void MontgomeryBatchInversion(
    uint64_t elements[][4],    // Input/Output: elements to invert
    uint32_t count,             // Number of elements
    uint64_t temp_storage[][4] // Temporary storage for products
) {
    if (count == 0) return;
    if (count == 1) {
        // Single element - use standard inversion
        uint64_t inv[5];
        Load256(inv, elements[0]);
        inv[4] = 0;
        _ModInv(inv);
        Load256(elements[0], inv);
        return;
    }
    
    // Step 1: Compute cumulative products
    // temp_storage[0] = elements[0]
    // temp_storage[1] = elements[0] * elements[1]
    // temp_storage[2] = elements[0] * elements[1] * elements[2]
    // ...
    Load256(temp_storage[0], elements[0]);
    
    for (uint32_t i = 1; i < count; i++) {
        _ModMult(temp_storage[i], temp_storage[i-1], elements[i]);
    }
    
    // Step 2: Compute inverse of final product
    uint64_t inv_product[5];
    Load256(inv_product, temp_storage[count-1]);
    inv_product[4] = 0;
    _ModInv(inv_product);
    
    if (_IsZero(inv_product)) {
        // One of the elements has no inverse
        // Mark all as zero (error case)
        for (uint32_t i = 0; i < count; i++) {
            elements[i][0] = 0;
            elements[i][1] = 0;
            elements[i][2] = 0;
            elements[i][3] = 0;
        }
        return;
    }
    
    // Step 3: Extract individual inverses by back-substitution
    // inv(elements[n-1]) = inv_product * temp_storage[n-2]
    // inv(elements[n-2]) = inv_product * elements[n-1] * temp_storage[n-3]
    // ...
    uint64_t current_inv[4];
    Load256(current_inv, inv_product);
    
    for (int32_t i = count - 1; i > 0; i--) {
        uint64_t element_inv[4];
        
        // elements[i]^-1 = current_inv * temp_storage[i-1]
        _ModMult(element_inv, current_inv, temp_storage[i-1]);
        
        // Update current_inv = current_inv * elements[i]
        uint64_t temp[4];
        Load256(temp, elements[i]);
        _ModMult(current_inv, current_inv, temp);
        
        // Store the inverse
        Load256(elements[i], element_inv);
    }
    
    // First element inverse
    Load256(elements[0], current_inv);
}

/**
 * Optimized batch inversion using shared memory
 * Processes multiple batches in parallel using warp-level parallelism
 */
__global__ void BatchModularInversion(
    uint64_t* elements,     // Global memory array of elements
    uint32_t num_batches,   // Number of batches to process
    uint32_t batch_size     // Elements per batch
) {
    // Dynamic shared memory for temporary storage
    extern __shared__ uint64_t shared_mem[];
    
    uint32_t tid = threadIdx.x;
    uint32_t batch_id = blockIdx.x;
    
    if (batch_id >= num_batches) return;
    
    // Each block processes one batch
    uint32_t batch_offset = batch_id * batch_size * 4;
    
    // Divide work among threads in the block
    uint32_t elements_per_thread = (batch_size + blockDim.x - 1) / blockDim.x;
    uint32_t start_idx = tid * elements_per_thread;
    uint32_t end_idx = min(start_idx + elements_per_thread, batch_size);
    
    // Local storage for this thread's elements
    uint64_t local_elements[16][4];  // Max 16 elements per thread
    uint64_t local_temp[16][4];
    
    // Load elements from global to local memory
    uint32_t local_count = 0;
    for (uint32_t i = start_idx; i < end_idx && local_count < 16; i++) {
        uint32_t global_idx = batch_offset + i * 4;
        local_elements[local_count][0] = elements[global_idx + 0];
        local_elements[local_count][1] = elements[global_idx + 1];
        local_elements[local_count][2] = elements[global_idx + 2];
        local_elements[local_count][3] = elements[global_idx + 3];
        local_count++;
    }
    
    // Perform batch inversion on local elements
    if (local_count > 0) {
        MontgomeryBatchInversion(local_elements, local_count, local_temp);
    }
    
    // Write results back to global memory
    local_count = 0;
    for (uint32_t i = start_idx; i < end_idx && local_count < 16; i++) {
        uint32_t global_idx = batch_offset + i * 4;
        elements[global_idx + 0] = local_elements[local_count][0];
        elements[global_idx + 1] = local_elements[local_count][1];
        elements[global_idx + 2] = local_elements[local_count][2];
        elements[global_idx + 3] = local_elements[local_count][3];
        local_count++;
    }
    
    __syncthreads();
}

/**
 * Warp-level optimized batch inversion
 * Uses warp shuffle operations for better performance
 */
__device__ void WarpBatchInversion(
    uint64_t elements[][4],
    uint32_t count
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    
    // Each lane processes different elements
    if (lane_id < count) {
        // Phase 1: Parallel prefix product using warp shuffles
        uint64_t my_element[4];
        Load256(my_element, elements[lane_id]);
        
        uint64_t prefix_product[4];
        Load256(prefix_product, my_element);
        
        // Compute prefix products using warp shuffle
        for (int offset = 1; offset < WARP_SIZE && offset < count; offset *= 2) {
            uint64_t received[4];
            
            // Shuffle to get value from lane_id - offset
            int source_lane = lane_id - offset;
            if (source_lane >= 0) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    received[i] = __shfl_sync(0xFFFFFFFF, prefix_product[i], source_lane);
                }
                
                // Multiply with received value
                uint64_t temp[4];
                _ModMult(temp, received, prefix_product);
                Load256(prefix_product, temp);
            }
        }
        
        // Phase 2: Compute inverse of final product
        if (lane_id == count - 1) {
            uint64_t inv[5];
            Load256(inv, prefix_product);
            inv[4] = 0;
            _ModInv(inv);
            Load256(prefix_product, inv);
        }
        
        // Broadcast inverse to all lanes
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            prefix_product[i] = __shfl_sync(0xFFFFFFFF, prefix_product[i], count - 1);
        }
        
        // Phase 3: Back-substitution to get individual inverses
        // This part needs sequential processing, done by leader lane
        if (lane_id == 0) {
            MontgomeryBatchInversion(elements, count, nullptr);
        }
    }
    
    __syncwarp();
}

/**
 * Hybrid batch inversion for large batches
 * Combines block-level and warp-level parallelism
 */
__global__ void HybridBatchModularInversion(
    uint64_t* elements,
    uint32_t total_elements
) {
    // Use shared memory for intermediate storage
    extern __shared__ uint64_t shared_storage[];
    
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t total_threads = blockDim.x * gridDim.x;
    
    // Each thread handles multiple elements
    const uint32_t elements_per_thread = (total_elements + total_threads - 1) / total_threads;
    const uint32_t start = tid * elements_per_thread;
    const uint32_t end = min(start + elements_per_thread, total_elements);
    
    if (start >= total_elements) return;
    
    // Process elements in chunks
    const uint32_t CHUNK_SIZE = 32;
    uint64_t chunk[CHUNK_SIZE][4];
    uint64_t temp[CHUNK_SIZE][4];
    
    for (uint32_t chunk_start = start; chunk_start < end; chunk_start += CHUNK_SIZE) {
        uint32_t chunk_end = min(chunk_start + CHUNK_SIZE, end);
        uint32_t chunk_size = chunk_end - chunk_start;
        
        // Load chunk
        for (uint32_t i = 0; i < chunk_size; i++) {
            uint32_t idx = (chunk_start + i) * 4;
            chunk[i][0] = elements[idx + 0];
            chunk[i][1] = elements[idx + 1];
            chunk[i][2] = elements[idx + 2];
            chunk[i][3] = elements[idx + 3];
        }
        
        // Process chunk
        if (chunk_size <= WARP_SIZE) {
            // Use warp-level optimization for small chunks
            WarpBatchInversion(chunk, chunk_size);
        } else {
            // Use standard batch inversion for larger chunks
            MontgomeryBatchInversion(chunk, chunk_size, temp);
        }
        
        // Store results
        for (uint32_t i = 0; i < chunk_size; i++) {
            uint32_t idx = (chunk_start + i) * 4;
            elements[idx + 0] = chunk[i][0];
            elements[idx + 1] = chunk[i][1];
            elements[idx + 2] = chunk[i][2];
            elements[idx + 3] = chunk[i][3];
        }
    }
}

/**
 * Launch configuration helper for batch inversion
 */
extern "C" void launchBatchModularInversion(
    uint64_t* d_elements,
    uint32_t num_elements,
    cudaStream_t stream
) {
    // Determine optimal launch configuration
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Use multiple blocks for large batches
    const uint32_t elements_per_block = 256;
    const uint32_t num_blocks = (num_elements + elements_per_block - 1) / elements_per_block;
    const uint32_t threads_per_block = min(256, prop.maxThreadsPerBlock);
    
    // Calculate shared memory size
    size_t shared_mem_size = elements_per_block * 4 * sizeof(uint64_t);
    
    if (num_elements <= WARP_SIZE) {
        // Small batch - use single warp
        HybridBatchModularInversion<<<1, WARP_SIZE, shared_mem_size, stream>>>(
            d_elements, num_elements
        );
    } else if (num_elements <= elements_per_block) {
        // Medium batch - use single block
        BatchModularInversion<<<1, threads_per_block, shared_mem_size, stream>>>(
            d_elements, 1, num_elements
        );
    } else {
        // Large batch - use multiple blocks
        HybridBatchModularInversion<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
            d_elements, num_elements
        );
    }
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Batch modular inversion kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}