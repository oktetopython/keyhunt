/*
 * PUZZLE71 specific kernel implementations - OPTIMIZED VERSION
 * High-performance implementation targeting 1000+ MKey/s
 */

#include "../cuda_fix.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include "GPUCompute.h"
#include "GPUHash.h"
#include "GPUMath.h"
#include "BatchModInverse.cuh"

// PUZZLE71 target address: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
// HASH160: 7ADCAE87EF9D7AACFB7F3D0AB30C7E4B8BCE16D9 (big endian)
// As uint32_t array (little endian for GPU comparison):
__device__ __constant__ uint32_t PUZZLE71_TARGET_HASH_LOCAL[5] = {
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

// Precomputed generator point multiples for fast access
extern __device__ uint64_t* Gx;
extern __device__ uint64_t* Gy;

// Also import the optimized generator functions
extern __device__ __forceinline__ void GetGeneratorMultiple(
    uint32_t multiple,
    uint64_t result_x[4],
    uint64_t result_y[4]);
extern __device__ void LoadGeneratorBatch(
    uint32_t start_multiple,
    uint32_t count,
    uint64_t batch_x[][4],
    uint64_t batch_y[][4]);

// Shared memory size for batch processing
#define SHARED_MEM_SIZE 8192

// Found flag for early exit (defined in GPUGlobals.cu)
extern __device__ int found_flag;

/**
 * Simplified PUZZLE71 kernel - direct algorithm without complex batch processing
 * Focuses on basic EC point computation and hash verification
 */
/**
 * Fast EC point addition using optimized batch inversion when possible
 * Falls back to single inversion for individual operations
 */
__device__ __forceinline__ void fast_point_add(
    uint64_t p1x[4], uint64_t p1y[4], 
    const uint64_t p2x[4], const uint64_t p2y[4]) 
{
    // Use existing optimized functions from GPUMath.h
    uint64_t slope[4], temp1[4], temp2[4];
    
    // slope = (p2y - p1y) / (p2x - p1x)
    ModSub256(temp1, (uint64_t*)p2y, (uint64_t*)p1y);  // dy
    ModSub256(temp2, (uint64_t*)p2x, (uint64_t*)p1x);  // dx
    
    // Use Montgomery inversion for division (single value)
    uint64_t dx_inv[5] = {temp2[0], temp2[1], temp2[2], temp2[3], 0};
    _ModInv(dx_inv);
    _ModMult(slope, temp1, dx_inv);
    
    // x3 = slope^2 - p1x - p2x
    _ModSqr(temp1, slope);
    ModSub256(temp1, temp1, (uint64_t*)p1x);
    ModSub256(temp1, temp1, (uint64_t*)p2x);
    
    // y3 = slope * (p1x - x3) - p1y
    ModSub256(temp2, (uint64_t*)p1x, temp1);
    _ModMult(temp2, slope, temp2);
    ModSub256(temp2, temp2, (uint64_t*)p1y);
    
    // Update result
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        p1x[i] = temp1[i];
        p1y[i] = temp2[i];
    }
}

/**
 * Batch-optimized EC point addition for processing multiple points
 * Uses true batch modular inversion for better performance
 */
__device__ __forceinline__ void batch_fast_point_add(
    uint64_t points_x[][4], uint64_t points_y[][4],
    const uint64_t add_x[4], const uint64_t add_y[4],
    uint32_t batch_size)
{
    // Use our new batch EC point addition function
    batchECPointAdd((uint64_t*)points_x, (uint64_t*)points_y, 
                   add_x, add_y, batch_size);
}

/**
 * Ultra-fast hash comparison (unrolled)
 */
__device__ __forceinline__ bool fast_hash_compare(const uint32_t hash[5]) {
    return (hash[0] == PUZZLE71_TARGET_HASH_LOCAL[0]) &
           (hash[1] == PUZZLE71_TARGET_HASH_LOCAL[1]) &
           (hash[2] == PUZZLE71_TARGET_HASH_LOCAL[2]) &
           (hash[3] == PUZZLE71_TARGET_HASH_LOCAL[3]) &
           (hash[4] == PUZZLE71_TARGET_HASH_LOCAL[4]);
}

/**
 * Advanced batch-processing PUZZLE71 kernel with true batch optimization
 * Uses batch modular inversion and warp-level cooperation
 */
__global__ void compute_keys_puzzle71_advanced(uint32_t mode, uint32_t* hash, uint64_t* keys, uint32_t maxFound, uint32_t* found) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    
    // Advanced batch processing with shared memory
    __shared__ uint64_t shared_points_x[32][4];  // Warp-sized batch
    __shared__ uint64_t shared_points_y[32][4];
    __shared__ uint64_t shared_inverses[32][4];
    __shared__ uint8_t shared_hashes[32][20];
    
    const int MEGA_BATCH_SIZE = 8;   // Process 8 mini-batches per kernel call
    const int MINI_BATCH_SIZE = 32;  // Use warp size for cooperation
    
    // Load starting point from global memory
    uint64_t px[4], py[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        px[i] = keys[tid * 8 + i];
        py[i] = keys[tid * 8 + 4 + i];
    }
    
    // Generator point for increments
    uint64_t gx[4] = {G_FAST[0], G_FAST[1], G_FAST[2], G_FAST[3]};
    uint64_t gy[4] = {G_FAST[4], G_FAST[5], G_FAST[6], G_FAST[7]};
    
    // Process multiple mega-batches
    for (int mega_batch = 0; mega_batch < MEGA_BATCH_SIZE; mega_batch++) {
        // Load current points to shared memory for batch processing
        shared_points_x[lane][0] = px[0]; shared_points_x[lane][1] = px[1];
        shared_points_x[lane][2] = px[2]; shared_points_x[lane][3] = px[3];
        shared_points_y[lane][0] = py[0]; shared_points_y[lane][1] = py[1];
        shared_points_y[lane][2] = py[2]; shared_points_y[lane][3] = py[3];
        
        __syncwarp();
        
        // Compute hashes for entire warp in parallel
        _GetHash160Comp(px, (py[0] & 1), &shared_hashes[lane][0]);
        
        __syncwarp();
        
        // Check all hashes in warp for matches
        uint32_t* my_hash = (uint32_t*)&shared_hashes[lane][0];
        if (fast_hash_compare(my_hash)) {
            // Found match!
            uint32_t pos = atomicAdd(found, 1);
            if (pos < maxFound) {
                // Store result efficiently 
                uint32_t base_idx = 1 + pos * 16;
                hash[base_idx] = 0xBEEFCAFE;     // Advanced found marker
                hash[base_idx + 1] = tid;        // Thread ID
                hash[base_idx + 2] = mega_batch; // Mega batch number
                
                // Copy hash
                #pragma unroll
                for (int j = 0; j < 5; j++) {
                    hash[base_idx + 3 + j] = my_hash[j];
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
        
        // Batch increment all points in warp using true batch processing
        // This is where the real optimization happens!
        if (lane == 0) {
            // Warp leader coordinates the batch point addition
            batchECPointAdd((uint64_t*)shared_points_x, (uint64_t*)shared_points_y,
                           gx, gy, MINI_BATCH_SIZE);
        }
        
        __syncwarp();
        
        // Load updated point back to registers
        px[0] = shared_points_x[lane][0]; px[1] = shared_points_x[lane][1];
        px[2] = shared_points_x[lane][2]; px[3] = shared_points_x[lane][3];
        py[0] = shared_points_y[lane][0]; py[1] = shared_points_y[lane][1];
        py[2] = shared_points_y[lane][2]; py[3] = shared_points_y[lane][3];
    }
    
    // Store updated position back
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        keys[tid * 8 + i] = px[i];
        keys[tid * 8 + 4 + i] = py[i];
    }
}

/**
 * Standard high-performance PUZZLE71 kernel (fallback version)
 * Maintains compatibility while still providing good performance
 */
__global__ void compute_keys_puzzle71(uint32_t mode, uint32_t* hash, uint64_t* keys, uint32_t maxFound, uint32_t* found) {
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

// ComputeKeysPUZZLE71 function is now defined in GPUCompute.h to avoid duplication

/**
 * Compressed mode wrapper for compatibility
 */
__global__ void compute_keys_comp_puzzle71(uint32_t mode, uint32_t* hash, uint64_t* keys, uint32_t maxFound, uint32_t* found) {
    // Delegate to the main kernel (already handles compression)
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
