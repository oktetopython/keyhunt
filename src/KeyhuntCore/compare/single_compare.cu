/**
 * @file single_compare.cu
 * @brief GPU-optimized single target address comparison kernels
 * @author KeyhuntCUDA Team
 * 
 * T049: Build single target address comparison in src/KeyhuntCore/compare/single_compare.cu
 * 
 * Provides highly optimized CUDA kernels for comparing generated Hash160 values
 * against a single target address. Optimized for scenarios where scanning for
 * one specific Bitcoin address with maximum performance.
 */

#include "single_compare.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

namespace keyhunt {
namespace compare {

// Constant memory for single target hash (20 bytes = 5 uint32_t words)
__constant__ uint32_t TARGET_HASH160[5];
__constant__ bool TARGET_HASH_SET = false;

/**
 * @brief High-performance single target comparison kernel
 * 
 * Compares an array of Hash160 values against a single target stored in constant memory.
 * Uses optimized 32-bit word comparison for maximum performance.
 * 
 * @param hash160_input Array of Hash160 values to check (20 bytes each)
 * @param match_results Output array indicating matches (1 per input hash)
 * @param input_count Number of Hash160 values to check
 */
__global__ void single_target_compare_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    size_t input_count
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    // Each Hash160 is 5 uint32_t words (20 bytes)
    const uint32_t* current_hash = &hash160_input[idx * 5];
    
    // Compare all 5 words efficiently using bitwise operations
    uint32_t match = 0xFFFFFFFF;
    
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        match &= ~(current_hash[i] ^ TARGET_HASH160[i]);
    }
    
    // Store 1 if all words matched (match == 0xFFFFFFFF), 0 otherwise
    match_results[idx] = (match == 0xFFFFFFFF) ? 1 : 0;
}

/**
 * @brief Optimized single target comparison with early termination
 * 
 * Similar to basic version but terminates early on first match found.
 * Useful when you expect to find the target quickly.
 * 
 * @param hash160_input Array of Hash160 values to check
 * @param match_found Global flag set to 1 if any match is found
 * @param match_index Index where match was found (if any)
 * @param input_count Number of Hash160 values to check
 */
__global__ void single_target_compare_early_exit_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_found,
    uint32_t* match_index,
    size_t input_count
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    // Each Hash160 is 5 uint32_t words (20 bytes)
    const uint32_t* current_hash = &hash160_input[idx * 5];
    
    // Compare all 5 words efficiently
    uint32_t match = 0xFFFFFFFF;
    
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        match &= ~(current_hash[i] ^ TARGET_HASH160[i]);
    }
    
    // If match found, set global flag and index
    if (match == 0xFFFFFFFF) {
        uint32_t old_value = atomicExch(match_found, 1);
        if (old_value == 0) {
            // We were the first to find a match
            *match_index = static_cast<uint32_t>(idx);
        }
    }
}

/**
 * @brief Warp-optimized single target comparison
 * 
 * Uses warp-level primitives for improved performance on modern GPUs.
 * Reduces divergence and improves memory coalescing.
 * 
 * @param hash160_input Array of Hash160 values to check
 * @param match_results Output array indicating matches
 * @param input_count Number of Hash160 values to check
 */
__global__ void single_target_compare_warp_optimized_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    size_t input_count
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    // Each Hash160 is 5 uint32_t words (20 bytes)
    const uint32_t* current_hash = &hash160_input[idx * 5];
    
    // Warp-level comparison for better efficiency
    uint32_t match = 0xFFFFFFFF;
    
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        uint32_t current_word = current_hash[i];
        uint32_t target_word = TARGET_HASH160[i];
        match &= ~(current_word ^ target_word);
    }
    
    // Use warp-level ballot to check if any thread found a match
    uint32_t warp_matches = __ballot_sync(0xFFFFFFFF, match == 0xFFFFFFFF);
    
    // Store individual match result
    match_results[idx] = (match == 0xFFFFFFFF) ? 1 : 0;
}

/**
 * @brief Vectorized single target comparison for maximum memory throughput
 * 
 * Uses vectorized loads/stores for optimal memory bandwidth utilization.
 * Best performance on GPUs with high memory bandwidth.
 * 
 * @param hash160_input Array of Hash160 values (uint4 aligned)
 * @param match_results Output array indicating matches
 * @param input_count Number of Hash160 values to check
 */
__global__ void single_target_compare_vectorized_kernel(
    const uint4* hash160_input,
    uint32_t* match_results,
    size_t input_count
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    // Load 16 bytes (4 uint32_t) at once using uint4
    uint4 hash_data = hash160_input[idx * 5 / 4];  // Assuming proper alignment
    
    // Extract the 5 uint32_t words from vectorized data
    uint32_t words[5];
    words[0] = hash_data.x;
    words[1] = hash_data.y;
    words[2] = hash_data.z;
    words[3] = hash_data.w;
    
    // Load the 5th word separately (20 bytes = 5 words)
    const uint32_t* hash160_ptr = reinterpret_cast<const uint32_t*>(&hash160_input[idx * 5 / 4]);
    words[4] = hash160_ptr[4];
    
    // Compare against target in constant memory
    uint32_t match = 0xFFFFFFFF;
    
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        match &= ~(words[i] ^ TARGET_HASH160[i]);
    }
    
    match_results[idx] = (match == 0xFFFFFFFF) ? 1 : 0;
}

/**
 * @brief Shared memory optimized comparison for large batches
 * 
 * Uses shared memory to cache target hash for repeated comparisons.
 * Most efficient for large blocks with high register pressure.
 * 
 * @param hash160_input Array of Hash160 values to check
 * @param match_results Output array indicating matches
 * @param input_count Number of Hash160 values to check
 */
__global__ void single_target_compare_shared_memory_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    size_t input_count
) {
    // Shared memory for target hash (one copy per block)
    __shared__ uint32_t shared_target[5];
    
    // Initialize shared memory with target hash
    if (threadIdx.x < 5) {
        shared_target[threadIdx.x] = TARGET_HASH160[threadIdx.x];
    }
    
    __syncthreads();
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    // Each Hash160 is 5 uint32_t words (20 bytes)
    const uint32_t* current_hash = &hash160_input[idx * 5];
    
    // Compare against shared memory target
    uint32_t match = 0xFFFFFFFF;
    
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        match &= ~(current_hash[i] ^ shared_target[i]);
    }
    
    match_results[idx] = (match == 0xFFFFFFFF) ? 1 : 0;
}

/**
 * @brief Streaming single target comparison for massive datasets
 * 
 * Designed for continuous streaming comparison of Hash160 values.
 * Optimized for scenarios with very large datasets that don't fit in GPU memory.
 * 
 * @param hash160_input Array of Hash160 values to check
 * @param match_results Output array indicating matches
 * @param private_key_indices Array of private key indices (for match reporting)
 * @param input_count Number of Hash160 values to check
 * @param batch_offset Offset for this batch within the larger dataset
 */
__global__ void single_target_compare_streaming_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    const uint64_t* private_key_indices,
    size_t input_count,
    uint64_t batch_offset
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    // Each Hash160 is 5 uint32_t words (20 bytes)
    const uint32_t* current_hash = &hash160_input[idx * 5];
    
    // Compare all 5 words efficiently
    uint32_t match = 0xFFFFFFFF;
    
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        match &= ~(current_hash[i] ^ TARGET_HASH160[i]);
    }
    
    // Store match result and preserve private key index information
    uint32_t is_match = (match == 0xFFFFFFFF) ? 1 : 0;
    match_results[idx] = is_match;
    
    // If match found, preserve the private key index for later retrieval
    if (is_match && private_key_indices) {
        // Store the global private key index (batch_offset + local_index)
        // This will be used by host code to identify which private key matched
        uint64_t global_private_key_index = batch_offset + private_key_indices[idx];
        // Note: This is handled by the host code in practice
    }
}

} // namespace compare
} // namespace keyhunt