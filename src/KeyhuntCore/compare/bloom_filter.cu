/**
 * @file bloom_filter.cu
 * @brief GPU-accelerated multi-target address comparison with Bloom filter optimization
 * @author KeyhuntCUDA Team
 * 
 * T050: Create multi-target comparison with Bloom Filter in src/KeyhuntCore/compare/bloom_filter.cu
 * 
 * Provides highly optimized CUDA kernels for comparing generated Hash160 values
 * against multiple target addresses using Bloom filter for initial filtering
 * followed by exact matching for confirmed hits.
 */

#include "multi_target_compare.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

namespace keyhunt {
namespace compare {

// Constant memory for multi-target configuration
__constant__ uint32_t NUM_TARGETS;
__constant__ uint32_t MAX_TARGETS_CONSTANT_MEM = 64;  // Maximum targets in constant memory
__constant__ uint32_t TARGET_HASHES[64][5];           // Up to 64 targets in constant memory

// Bloom filter parameters in constant memory
__constant__ uint32_t* BLOOM_FILTER_DEVICE;          // Device pointer to Bloom filter
__constant__ uint32_t BLOOM_FILTER_SIZE_BITS;        // Size in bits
__constant__ uint32_t BLOOM_FILTER_SIZE_WORDS;       // Size in 32-bit words
__constant__ uint32_t BLOOM_FILTER_HASH_FUNCTIONS;   // Number of hash functions
__constant__ bool USE_BLOOM_FILTER;                  // Enable/disable Bloom filter

/**
 * @brief Compute hash function for Bloom filter
 * 
 * @param data Input data (hash160 as uint32_t array)
 * @param hash_func_index Hash function index (0 to num_hash_functions-1)
 * @return uint32_t Hash value
 */
__device__ __forceinline__ uint32_t bloom_hash_function(const uint32_t* data, uint32_t hash_func_index) {
    // Use different mixing for each hash function
    uint32_t hash = 0x811c9dc5;  // FNV-1a offset basis
    
    for (int i = 0; i < 5; ++i) {  // 5 words = 20 bytes (Hash160)
        uint32_t word = data[i];
        
        // Mix with hash function index
        word ^= (hash_func_index << (i * 4));
        
        // FNV-1a hash
        hash ^= word;
        hash *= 0x01000193;  // FNV-1a prime
        
        // Additional mixing for better distribution
        hash ^= hash >> 16;
        hash *= 0x85ebca6b;
        hash ^= hash >> 13;
        hash *= 0xc2b2ae35;
        hash ^= hash >> 16;
    }
    
    return hash;
}

/**
 * @brief Check if hash160 might be in Bloom filter
 * 
 * @param hash160_words Hash160 as array of 5 uint32_t words
 * @return bool True if might be present (could be false positive)
 */
__device__ __forceinline__ bool bloom_filter_check(const uint32_t* hash160_words) {
    if (!USE_BLOOM_FILTER || !BLOOM_FILTER_DEVICE) {
        return true;  // Skip Bloom filter check
    }
    
    for (uint32_t i = 0; i < BLOOM_FILTER_HASH_FUNCTIONS; ++i) {
        uint32_t hash = bloom_hash_function(hash160_words, i);
        uint32_t bit_index = hash % BLOOM_FILTER_SIZE_BITS;
        uint32_t word_index = bit_index / 32;
        uint32_t bit_offset = bit_index % 32;
        
        uint32_t word = BLOOM_FILTER_DEVICE[word_index];
        if ((word & (1U << bit_offset)) == 0) {
            return false;  // Definitely not present
        }
    }
    
    return true;  // Might be present (could be false positive)
}

/**
 * @brief Exact hash160 comparison against target list
 * 
 * @param hash160_words Hash160 as array of 5 uint32_t words
 * @param target_index Output: index of matching target (if found)
 * @return bool True if exact match found
 */
__device__ __forceinline__ bool exact_target_match(const uint32_t* hash160_words, uint32_t* target_index) {
    for (uint32_t i = 0; i < NUM_TARGETS && i < MAX_TARGETS_CONSTANT_MEM; ++i) {
        uint32_t match = 0xFFFFFFFF;
        
        #pragma unroll
        for (int j = 0; j < 5; ++j) {
            match &= ~(hash160_words[j] ^ TARGET_HASHES[i][j]);
        }
        
        if (match == 0xFFFFFFFF) {
            *target_index = i;
            return true;
        }
    }
    
    return false;
}

/**
 * @brief Basic multi-target comparison kernel with Bloom filter optimization
 * 
 * @param hash160_input Array of Hash160 values to check (as uint32_t arrays)
 * @param match_results Output array indicating matches (1 per input hash)
 * @param target_indices Output array indicating which target matched
 * @param input_count Number of Hash160 values to check
 */
__global__ void multi_target_compare_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    uint32_t* target_indices,
    size_t input_count
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    // Each Hash160 is 5 uint32_t words (20 bytes)
    const uint32_t* current_hash = &hash160_input[idx * 5];
    
    // Initialize results
    match_results[idx] = 0;
    target_indices[idx] = 0xFFFFFFFF;  // Invalid index
    
    // First stage: Bloom filter check (fast rejection)
    if (!bloom_filter_check(current_hash)) {
        return;  // Definitely not a target
    }
    
    // Second stage: Exact comparison against targets
    uint32_t target_index;
    if (exact_target_match(current_hash, &target_index)) {
        match_results[idx] = 1;
        target_indices[idx] = target_index;
    }
}

/**
 * @brief Warp-optimized multi-target comparison kernel
 * 
 * Uses warp-level primitives for better performance on modern GPUs.
 * 
 * @param hash160_input Array of Hash160 values to check
 * @param match_results Output array indicating matches
 * @param target_indices Output array indicating which target matched
 * @param input_count Number of Hash160 values to check
 */
__global__ void multi_target_compare_warp_optimized_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    uint32_t* target_indices,
    size_t input_count
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    const uint32_t* current_hash = &hash160_input[idx * 5];
    
    // Initialize results
    match_results[idx] = 0;
    target_indices[idx] = 0xFFFFFFFF;
    
    // Bloom filter check with warp-level optimization
    bool bloom_pass = bloom_filter_check(current_hash);
    
    // Use warp ballot to check if any thread passed Bloom filter
    uint32_t warp_bloom_mask = __ballot_sync(0xFFFFFFFF, bloom_pass);
    
    if (warp_bloom_mask == 0) {
        return;  // No thread in warp passed Bloom filter
    }
    
    if (!bloom_pass) {
        return;  // This thread didn't pass Bloom filter
    }
    
    // Exact comparison
    uint32_t target_index;
    bool exact_match = exact_target_match(current_hash, &target_index);
    
    if (exact_match) {
        match_results[idx] = 1;
        target_indices[idx] = target_index;
    }
    
    // Use warp ballot to count matches for statistics
    uint32_t warp_match_mask = __ballot_sync(0xFFFFFFFF, exact_match);
    // This can be used for performance monitoring
}

/**
 * @brief Early exit multi-target comparison kernel
 * 
 * Terminates on first match found across all threads.
 * 
 * @param hash160_input Array of Hash160 values to check
 * @param match_found Global flag set to 1 if any match is found
 * @param match_index Index where first match was found
 * @param target_index Index of target that matched
 * @param input_count Number of Hash160 values to check
 */
__global__ void multi_target_compare_early_exit_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_found,
    uint32_t* match_index,
    uint32_t* target_index,
    size_t input_count
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    // Early exit if another thread already found a match
    if (*match_found != 0) return;
    
    const uint32_t* current_hash = &hash160_input[idx * 5];
    
    // Bloom filter check
    if (!bloom_filter_check(current_hash)) {
        return;
    }
    
    // Exact comparison
    uint32_t found_target_index;
    if (exact_target_match(current_hash, &found_target_index)) {
        // Atomic update to ensure only first match is recorded
        uint32_t old_value = atomicExch(match_found, 1);
        if (old_value == 0) {
            // We were the first to find a match
            *match_index = static_cast<uint32_t>(idx);
            *target_index = found_target_index;
        }
    }
}

/**
 * @brief Streaming multi-target comparison kernel for large datasets
 * 
 * @param hash160_input Array of Hash160 values to check
 * @param match_results Output array indicating matches
 * @param target_indices Output array indicating which target matched
 * @param private_key_indices Array of private key indices
 * @param input_count Number of Hash160 values to check
 * @param batch_offset Offset for this batch within larger dataset
 */
__global__ void multi_target_compare_streaming_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    uint32_t* target_indices,
    const uint64_t* private_key_indices,
    size_t input_count,
    uint64_t batch_offset
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    const uint32_t* current_hash = &hash160_input[idx * 5];
    
    // Initialize results
    match_results[idx] = 0;
    target_indices[idx] = 0xFFFFFFFF;
    
    // Bloom filter check
    if (!bloom_filter_check(current_hash)) {
        return;
    }
    
    // Exact comparison
    uint32_t target_index;
    if (exact_target_match(current_hash, &target_index)) {
        match_results[idx] = 1;
        target_indices[idx] = target_index;
        
        // Private key indices are preserved for later retrieval
        // The host code will extract the private key based on batch_offset + idx
    }
}

/**
 * @brief Shared memory optimized multi-target comparison
 * 
 * Caches frequently accessed targets in shared memory for better performance.
 * 
 * @param hash160_input Array of Hash160 values to check
 * @param match_results Output array indicating matches
 * @param target_indices Output array indicating which target matched
 * @param input_count Number of Hash160 values to check
 */
__global__ void multi_target_compare_shared_memory_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    uint32_t* target_indices,
    size_t input_count
) {
    // Shared memory for caching targets (limited subset)
    __shared__ uint32_t shared_targets[32][5];  // Cache up to 32 targets
    __shared__ uint32_t shared_num_targets;
    
    // Initialize shared memory with targets
    if (threadIdx.x == 0) {
        shared_num_targets = min(NUM_TARGETS, 32U);
    }
    
    if (threadIdx.x < shared_num_targets * 5) {
        uint32_t target_idx = threadIdx.x / 5;
        uint32_t word_idx = threadIdx.x % 5;
        shared_targets[target_idx][word_idx] = TARGET_HASHES[target_idx][word_idx];
    }
    
    __syncthreads();
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    const uint32_t* current_hash = &hash160_input[idx * 5];
    
    // Initialize results
    match_results[idx] = 0;
    target_indices[idx] = 0xFFFFFFFF;
    
    // Bloom filter check
    if (!bloom_filter_check(current_hash)) {
        return;
    }
    
    // Exact comparison against shared memory targets
    for (uint32_t i = 0; i < shared_num_targets; ++i) {
        uint32_t match = 0xFFFFFFFF;
        
        #pragma unroll
        for (int j = 0; j < 5; ++j) {
            match &= ~(current_hash[j] ^ shared_targets[i][j]);
        }
        
        if (match == 0xFFFFFFFF) {
            match_results[idx] = 1;
            target_indices[idx] = i;
            return;  // Found match, exit early
        }
    }
    
    // If not found in shared memory, check remaining targets in constant memory
    if (NUM_TARGETS > 32) {
        for (uint32_t i = 32; i < NUM_TARGETS && i < MAX_TARGETS_CONSTANT_MEM; ++i) {
            uint32_t match = 0xFFFFFFFF;
            
            #pragma unroll
            for (int j = 0; j < 5; ++j) {
                match &= ~(current_hash[j] ^ TARGET_HASHES[i][j]);
            }
            
            if (match == 0xFFFFFFFF) {
                match_results[idx] = 1;
                target_indices[idx] = i;
                return;
            }
        }
    }
}

/**
 * @brief Batch processing kernel for multiple Hash160 comparisons
 * 
 * Processes multiple Hash160 values per thread for better memory efficiency.
 * 
 * @param hash160_input Array of Hash160 values to check
 * @param match_results Output array indicating matches
 * @param target_indices Output array indicating which target matched
 * @param input_count Number of Hash160 values to check
 * @param hashes_per_thread Number of hashes each thread processes
 */
__global__ void multi_target_compare_batch_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    uint32_t* target_indices,
    size_t input_count,
    uint32_t hashes_per_thread
) {
    const size_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * hashes_per_thread;
    
    for (uint32_t i = 0; i < hashes_per_thread; ++i) {
        size_t idx = base_idx + i;
        
        if (idx >= input_count) return;
        
        const uint32_t* current_hash = &hash160_input[idx * 5];
        
        // Initialize results
        match_results[idx] = 0;
        target_indices[idx] = 0xFFFFFFFF;
        
        // Bloom filter check
        if (!bloom_filter_check(current_hash)) {
            continue;
        }
        
        // Exact comparison
        uint32_t target_index;
        if (exact_target_match(current_hash, &target_index)) {
            match_results[idx] = 1;
            target_indices[idx] = target_index;
        }
    }
}

/**
 * @brief Statistics collection kernel
 * 
 * Collects performance statistics during multi-target comparison.
 * 
 * @param hash160_input Array of Hash160 values to check
 * @param match_results Output array indicating matches
 * @param target_indices Output array indicating which target matched
 * @param bloom_hits Output: number of Bloom filter hits
 * @param exact_matches Output: number of exact matches
 * @param input_count Number of Hash160 values to check
 */
__global__ void multi_target_compare_stats_kernel(
    const uint32_t* hash160_input,
    uint32_t* match_results,
    uint32_t* target_indices,
    uint64_t* bloom_hits,
    uint64_t* exact_matches,
    size_t input_count
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= input_count) return;
    
    const uint32_t* current_hash = &hash160_input[idx * 5];
    
    // Initialize results
    match_results[idx] = 0;
    target_indices[idx] = 0xFFFFFFFF;
    
    // Bloom filter check with statistics
    bool bloom_pass = bloom_filter_check(current_hash);
    
    if (bloom_pass) {
        atomicAdd(bloom_hits, 1);
        
        // Exact comparison
        uint32_t target_index;
        if (exact_target_match(current_hash, &target_index)) {
            match_results[idx] = 1;
            target_indices[idx] = target_index;
            atomicAdd(exact_matches, 1);
        }
    }
}

} // namespace compare
} // namespace keyhunt