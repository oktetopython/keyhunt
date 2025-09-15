/**
 * @file single_compare_kernels.cu
 * @brief CUDA kernel wrapper functions for single target comparison
 * @author KeyhuntCUDA Team
 * 
 * T049: Build single target address comparison in src/KeyhuntCore/compare/single_compare.cu
 * 
 * Provides C-style wrapper functions for CUDA kernels to bridge C++ and CUDA code.
 * These wrappers handle kernel parameter setup and launch configuration.
 */

#include "single_compare.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Import kernel declarations from single_compare.cu
namespace keyhunt {
namespace compare {

// External kernel declarations
__global__ void single_target_compare_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    size_t input_count
);

__global__ void single_target_compare_early_exit_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_found,
    uint32_t* match_index,
    size_t input_count
);

__global__ void single_target_compare_warp_optimized_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    size_t input_count
);

__global__ void single_target_compare_streaming_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    const uint64_t* private_key_indices,
    size_t input_count,
    uint64_t batch_offset
);

} // namespace compare
} // namespace keyhunt

// External C-style wrapper functions for kernel launches
extern "C" {

/**
 * @brief Set target hash160 in GPU constant memory
 * 
 * @param target_hash Array of 5 uint32_t words representing the 20-byte hash160
 * @return cudaError_t CUDA error code
 */
cudaError_t cuda_set_target_hash160(const uint32_t* target_hash) {
    // Copy to device constant memory symbol
    return cudaMemcpyToSymbol(keyhunt::compare::TARGET_HASH160, target_hash, 
                             5 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

/**
 * @brief Launch basic single target comparison kernel
 * 
 * @param hash160_input GPU pointer to input hash160 values (as uint32_t array)
 * @param match_results GPU pointer to output match results
 * @param input_count Number of hash160 values to process
 * @param grid_size CUDA grid dimensions
 * @param block_size CUDA block dimensions
 * @param stream CUDA stream (nullable)
 * @return cudaError_t CUDA error code
 */
cudaError_t cuda_launch_single_target_compare_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    size_t input_count,
    dim3 grid_size,
    dim3 block_size,
    cudaStream_t stream) {
    
    // Launch kernel
    if (stream) {
        keyhunt::compare::single_target_compare_kernel<<<grid_size, block_size, 0, stream>>>(
            hash160_input, match_results, input_count);
    } else {
        keyhunt::compare::single_target_compare_kernel<<<grid_size, block_size>>>(
            hash160_input, match_results, input_count);
    }
    
    // Check for kernel launch errors
    return cudaGetLastError();
}

/**
 * @brief Launch early exit single target comparison kernel
 * 
 * @param hash160_input GPU pointer to input hash160 values
 * @param match_found GPU pointer to global match found flag
 * @param match_index GPU pointer to match index storage
 * @param input_count Number of hash160 values to process
 * @param grid_size CUDA grid dimensions
 * @param block_size CUDA block dimensions
 * @param stream CUDA stream (nullable)
 * @return cudaError_t CUDA error code
 */
cudaError_t cuda_launch_single_target_compare_early_exit_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_found,
    uint32_t* match_index,
    size_t input_count,
    dim3 grid_size,
    dim3 block_size,
    cudaStream_t stream) {
    
    // Launch kernel
    if (stream) {
        keyhunt::compare::single_target_compare_early_exit_kernel<<<grid_size, block_size, 0, stream>>>(
            hash160_input, match_found, match_index, input_count);
    } else {
        keyhunt::compare::single_target_compare_early_exit_kernel<<<grid_size, block_size>>>(
            hash160_input, match_found, match_index, input_count);
    }
    
    return cudaGetLastError();
}

/**
 * @brief Launch warp-optimized single target comparison kernel
 * 
 * @param hash160_input GPU pointer to input hash160 values
 * @param match_results GPU pointer to output match results
 * @param input_count Number of hash160 values to process
 * @param grid_size CUDA grid dimensions
 * @param block_size CUDA block dimensions
 * @param stream CUDA stream (nullable)
 * @return cudaError_t CUDA error code
 */
cudaError_t cuda_launch_single_target_compare_warp_optimized_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    size_t input_count,
    dim3 grid_size,
    dim3 block_size,
    cudaStream_t stream) {
    
    if (stream) {
        keyhunt::compare::single_target_compare_warp_optimized_kernel<<<grid_size, block_size, 0, stream>>>(
            hash160_input, match_results, input_count);
    } else {
        keyhunt::compare::single_target_compare_warp_optimized_kernel<<<grid_size, block_size>>>(
            hash160_input, match_results, input_count);
    }
    
    return cudaGetLastError();
}

/**
 * @brief Launch streaming single target comparison kernel
 * 
 * @param hash160_input GPU pointer to input hash160 values
 * @param match_results GPU pointer to output match results
 * @param private_key_indices GPU pointer to private key indices
 * @param input_count Number of hash160 values to process
 * @param batch_offset Offset for this batch in the larger dataset
 * @param grid_size CUDA grid dimensions
 * @param block_size CUDA block dimensions
 * @param stream CUDA stream (nullable)
 * @return cudaError_t CUDA error code
 */
cudaError_t cuda_launch_single_target_compare_streaming_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    const uint64_t* private_key_indices,
    size_t input_count,
    uint64_t batch_offset,
    dim3 grid_size,
    dim3 block_size,
    cudaStream_t stream) {
    
    if (stream) {
        keyhunt::compare::single_target_compare_streaming_kernel<<<grid_size, block_size, 0, stream>>>(
            hash160_input, match_results, private_key_indices, input_count, batch_offset);
    } else {
        keyhunt::compare::single_target_compare_streaming_kernel<<<grid_size, block_size>>>(
            hash160_input, match_results, private_key_indices, input_count, batch_offset);
    }
    
    return cudaGetLastError();
}

/**
 * @brief Set target hash availability flag in constant memory
 * 
 * @param target_set Boolean flag indicating if target is set
 * @return cudaError_t CUDA error code
 */
cudaError_t cuda_set_target_hash_flag(bool target_set) {
    return cudaMemcpyToSymbol(keyhunt::compare::TARGET_HASH_SET, &target_set, 
                             sizeof(bool), 0, cudaMemcpyHostToDevice);
}

/**
 * @brief Get optimal grid size for given input count and device properties
 * 
 * @param input_count Number of hash160 values to process
 * @param device_id CUDA device ID
 * @param threads_per_block Desired threads per block
 * @param max_blocks_per_grid Maximum blocks per grid
 * @return dim3 Optimal grid dimensions
 */
dim3 cuda_calculate_optimal_grid_size(
    size_t input_count,
    int device_id,
    int threads_per_block,
    int max_blocks_per_grid) {
    
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, device_id);
    
    // Calculate blocks needed
    size_t blocks_needed = (input_count + threads_per_block - 1) / threads_per_block;
    
    // Limit to device capabilities and user constraints
    size_t max_blocks = std::min(
        static_cast<size_t>(device_props.maxGridSize[0]),
        static_cast<size_t>(max_blocks_per_grid)
    );
    
    size_t optimal_blocks = std::min(blocks_needed, max_blocks);
    
    return dim3(static_cast<unsigned int>(optimal_blocks));
}

/**
 * @brief Get optimal block size for single target comparison kernels
 * 
 * @param device_id CUDA device ID
 * @param kernel_type Type of kernel being used
 * @return dim3 Optimal block dimensions
 */
dim3 cuda_calculate_optimal_block_size(int device_id, int kernel_type) {
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, device_id);
    
    // Base block size on device capabilities
    int optimal_threads = device_props.maxThreadsPerBlock;
    
    // Adjust based on kernel characteristics
    switch (kernel_type) {
        case 0: // Basic kernel
            optimal_threads = std::min(optimal_threads, 512);
            break;
        case 1: // Early exit kernel
            optimal_threads = std::min(optimal_threads, 256);
            break;
        case 2: // Warp optimized kernel
            optimal_threads = std::min(optimal_threads, 1024);
            break;
        case 3: // Shared memory kernel
            optimal_threads = std::min(optimal_threads, 256);
            break;
        default:
            optimal_threads = 256;
            break;
    }
    
    // Ensure it's a multiple of warp size (32)
    optimal_threads = (optimal_threads / 32) * 32;
    
    return dim3(static_cast<unsigned int>(optimal_threads));
}

} // extern "C"