/*
 * Optimized PUZZLE71 kernel implementation
 * High-performance version targeting 1000+ MKey/s
 */

#include "../cuda_fix.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include "GPUCompute.h"
#include "GPUHash.h"
#include "GPUMath.h"

// PUZZLE71 target address: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
// HASH160: 7ADCAE87EF9D7AACFB7F3D0AB30C7E4B8BCE16D9 (big endian)
// As uint32_t array (little endian for GPU comparison):
__device__ __constant__ uint32_t PUZZLE71_TARGET_HASH_OPT[5] = {
    0xD916CE8B,  // First 4 bytes 
    0x4E7C630A,  // Next 4 bytes
    0xB3D7FFFB,  // Next 4 bytes
    0xAC7A9DEF,  // Next 4 bytes
    0x87AEDC7A   // Last 4 bytes
};

// Optimized secp256k1 generator point (little-endian)
__device__ __constant__ uint64_t G_FAST[8] = {
    // G_X
    0x79BE667EF9DCBBACULL, 0x55A06295CE870B07ULL,
    0x029BFCDB2DCE28D9ULL, 0x9C47D08FFB10D4B8ULL,
    // G_Y  
    0x483ADA7726A3C465ULL, 0x5DA4FBFC0E1108A8ULL,
    0xFD17B448A6855419ULL, 0x9C47D08FFB10D4B8ULL
};

/**
 * Fast EC point addition using Montgomery ladder
 * Optimized for GPU with minimal branching
 */
__device__ __forceinline__ void fast_point_add(
    uint64_t p1x[4], uint64_t p1y[4], 
    const uint64_t p2x[4], const uint64_t p2y[4]) 
{
    // Use existing optimized functions from GPUMath.h
    uint64_t slope[4], temp1[4], temp2[4];
    
    // slope = (p2y - p1y) / (p2x - p1x)
    ModSub256(temp1, p2y, p1y);  // dy
    ModSub256(temp2, p2x, p1x);  // dx
    
    // Use Montgomery inversion for division
    uint64_t dx_inv[5] = {temp2[0], temp2[1], temp2[2], temp2[3], 0};
    _ModInv(dx_inv);
    _ModMult(slope, temp1, dx_inv);
    
    // x3 = slope^2 - p1x - p2x
    _ModSqr(temp1, slope);
    ModSub256(temp1, temp1, p1x);
    ModSub256(temp1, temp1, p2x);
    
    // y3 = slope * (p1x - x3) - p1y
    ModSub256(temp2, p1x, temp1);
    _ModMult(temp2, slope, temp2);
    ModSub256(temp2, temp2, p1y);
    
    // Update result
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        p1x[i] = temp1[i];
        p1y[i] = temp2[i];
    }
}

/**
 * Ultra-fast hash comparison (unrolled)
 */
__device__ __forceinline__ bool fast_hash_compare(const uint32_t hash[5]) {
    return (hash[0] == PUZZLE71_TARGET_HASH_OPT[0]) &
           (hash[1] == PUZZLE71_TARGET_HASH_OPT[1]) &
           (hash[2] == PUZZLE71_TARGET_HASH_OPT[2]) &
           (hash[3] == PUZZLE71_TARGET_HASH_OPT[3]) &
           (hash[4] == PUZZLE71_TARGET_HASH_OPT[4]);
}

/**
 * High-performance PUZZLE71 kernel
 * Processes many keys per thread with optimized batching
 */
__global__ void compute_keys_puzzle71_optimized(
    uint32_t mode, uint32_t* hash, uint64_t* keys, 
    uint32_t maxFound, uint32_t* found) 
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int BATCH_SIZE = 256;  // Process more keys per thread
    
    // Load starting point from global memory (should be k*G for some k)
    uint64_t px[4], py[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        px[i] = keys[tid * 8 + i];
        py[i] = keys[tid * 8 + 4 + i];
    }
    
    // Generator point for increments
    uint64_t gx[4] = {G_FAST[0], G_FAST[1], G_FAST[2], G_FAST[3]};
    uint64_t gy[4] = {G_FAST[4], G_FAST[5], G_FAST[6], G_FAST[7]};
    
    // Process batch of keys
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        // Compute hash directly using optimized functions
        uint8_t hash160_bytes[20];
        _GetHash160Comp(px, (py[0] & 1), hash160_bytes);
        
        // Fast comparison (no loops)
        uint32_t* hash_words = (uint32_t*)hash160_bytes;
        if (fast_hash_compare(hash_words)) {
            // Found match!
            uint32_t pos = atomicAdd(found, 1);
            if (pos < maxFound) {
                // Store result efficiently 
                uint32_t base_idx = 1 + pos * 16;
                hash[base_idx] = 0xDEADBEEF;     // Found marker
                hash[base_idx + 1] = tid;        // Thread ID
                hash[base_idx + 2] = batch;      // Batch number
                
                // Copy hash
                #pragma unroll
                for (int j = 0; j < 5; j++) {
                    hash[base_idx + 3 + j] = hash_words[j];
                }
                
                // Copy point (compressed format)
                #pragma unroll 
                for (int j = 0; j < 4; j++) {
                    hash[base_idx + 8 + j] = (uint32_t)(px[j] & 0xFFFFFFFF);
                    hash[base_idx + 12 + j] = (uint32_t)(px[j] >> 32);
                }
            }
            return; // Exit on first match
        }
        
        // Increment point: P = P + G (optimized)
        fast_point_add(px, py, gx, gy);
    }
    
    // Store updated position back
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        keys[tid * 8 + i] = px[i];
        keys[tid * 8 + 4 + i] = py[i];
    }
}

/**
 * Compressed mode wrapper for compatibility
 */
__global__ void compute_keys_comp_puzzle71_optimized(
    uint32_t mode, uint32_t* hash, uint64_t* keys, 
    uint32_t maxFound, uint32_t* found) 
{
    // Just call the optimized version (already handles compression)
    compute_keys_puzzle71_optimized(mode, hash, keys, maxFound, found);
}