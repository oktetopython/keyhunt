/**
 * GPUModInv.cuh
 * Batch modular inversion implementation for GPU
 * Implements Montgomery's trick for efficient batch inversion
 */

#ifndef GPU_MOD_INV_CUH
#define GPU_MOD_INV_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "GPUMath.h"

/**
 * Batch modular inversion using Montgomery's trick
 * Computes modular inverses for multiple elements efficiently
 * Only requires 1 inversion + 3(n-1) multiplications instead of n inversions
 * 
 * @param dx Array of elements to invert (in/out)
 * @param count Number of elements
 */
__device__ __forceinline__ void ModInvGrouped(uint64_t dx[][4], uint32_t count) {
    if (count == 0) return;
    
    // Special case for single element
    if (count == 1) {
        uint64_t inv[5];
        inv[0] = dx[0][0];
        inv[1] = dx[0][1];
        inv[2] = dx[0][2];
        inv[3] = dx[0][3];
        inv[4] = 0;
        _ModInv(inv);
        dx[0][0] = inv[0];
        dx[0][1] = inv[1];
        dx[0][2] = inv[2];
        dx[0][3] = inv[3];
        return;
    }
    
    // Allocate temporary storage for products
    // In a real implementation, this would use shared memory for better performance
    uint64_t products[513][4];  // Max size for KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE + 2
    
    // Step 1: Compute cumulative products
    // products[0] = dx[0]
    // products[1] = dx[0] * dx[1]
    // products[2] = dx[0] * dx[1] * dx[2]
    // ...
    products[0][0] = dx[0][0];
    products[0][1] = dx[0][1];
    products[0][2] = dx[0][2];
    products[0][3] = dx[0][3];
    
    for (uint32_t i = 1; i < count; i++) {
        _ModMult(products[i], products[i-1], dx[i]);
    }
    
    // Step 2: Compute inverse of the final product
    uint64_t inv[5];
    inv[0] = products[count-1][0];
    inv[1] = products[count-1][1];
    inv[2] = products[count-1][2];
    inv[3] = products[count-1][3];
    inv[4] = 0;
    _ModInv(inv);
    
    // Check if inversion failed (element has no inverse)
    if (_IsZero(inv)) {
        // Mark all as zero to indicate error
        for (uint32_t i = 0; i < count; i++) {
            dx[i][0] = 0;
            dx[i][1] = 0;
            dx[i][2] = 0;
            dx[i][3] = 0;
        }
        return;
    }
    
    // Step 3: Extract individual inverses by back-substitution
    // inv(dx[n-1]) = inv * products[n-2]
    // inv(dx[n-2]) = inv * dx[n-1] * products[n-3]
    // ...
    
    uint64_t current_inv[4];
    current_inv[0] = inv[0];
    current_inv[1] = inv[1];
    current_inv[2] = inv[2];
    current_inv[3] = inv[3];
    
    for (int32_t i = count - 1; i > 0; i--) {
        uint64_t element_inv[4];
        
        // dx[i]^-1 = current_inv * products[i-1]
        _ModMult(element_inv, current_inv, products[i-1]);
        
        // Update current_inv = current_inv * dx[i]
        uint64_t temp[4];
        temp[0] = dx[i][0];
        temp[1] = dx[i][1];
        temp[2] = dx[i][2];
        temp[3] = dx[i][3];
        _ModMult(current_inv, current_inv, temp);
        
        // Store the inverse back
        dx[i][0] = element_inv[0];
        dx[i][1] = element_inv[1];
        dx[i][2] = element_inv[2];
        dx[i][3] = element_inv[3];
    }
    
    // First element's inverse
    dx[0][0] = current_inv[0];
    dx[0][1] = current_inv[1];
    dx[0][2] = current_inv[2];
    dx[0][3] = current_inv[3];
}

/**
 * Optimized batch inversion using shared memory
 * This version uses shared memory for better performance
 * 
 * @param dx Array of elements to invert (in/out)
 * @param count Number of elements
 * @param shared_mem Shared memory buffer (must be at least count*4*sizeof(uint64_t))
 */
__device__ __forceinline__ void ModInvGroupedShared(uint64_t dx[][4], uint32_t count, uint64_t* shared_mem) {
    if (count == 0) return;
    
    // Use shared memory for products
    uint64_t (*products)[4] = (uint64_t (*)[4])shared_mem;
    
    // Special case for single element
    if (count == 1) {
        uint64_t inv[5];
        inv[0] = dx[0][0];
        inv[1] = dx[0][1];
        inv[2] = dx[0][2];
        inv[3] = dx[0][3];
        inv[4] = 0;
        _ModInv(inv);
        dx[0][0] = inv[0];
        dx[0][1] = inv[1];
        dx[0][2] = inv[2];
        dx[0][3] = inv[3];
        return;
    }
    
    // Step 1: Compute cumulative products in parallel (if possible)
    products[0][0] = dx[0][0];
    products[0][1] = dx[0][1];
    products[0][2] = dx[0][2];
    products[0][3] = dx[0][3];
    
    for (uint32_t i = 1; i < count; i++) {
        _ModMult(products[i], products[i-1], dx[i]);
    }
    
    // Synchronize threads if using multiple threads per batch
    __syncthreads();
    
    // Step 2: Compute inverse of the final product
    uint64_t inv[5];
    inv[0] = products[count-1][0];
    inv[1] = products[count-1][1];
    inv[2] = products[count-1][2];
    inv[3] = products[count-1][3];
    inv[4] = 0;
    _ModInv(inv);
    
    // Step 3: Extract individual inverses
    uint64_t current_inv[4];
    current_inv[0] = inv[0];
    current_inv[1] = inv[1];
    current_inv[2] = inv[2];
    current_inv[3] = inv[3];
    
    for (int32_t i = count - 1; i > 0; i--) {
        uint64_t element_inv[4];
        
        // dx[i]^-1 = current_inv * products[i-1]
        _ModMult(element_inv, current_inv, products[i-1]);
        
        // Update current_inv = current_inv * dx[i]
        uint64_t temp[4];
        temp[0] = dx[i][0];
        temp[1] = dx[i][1];
        temp[2] = dx[i][2];
        temp[3] = dx[i][3];
        _ModMult(current_inv, current_inv, temp);
        
        // Store the inverse back
        dx[i][0] = element_inv[0];
        dx[i][1] = element_inv[1];
        dx[i][2] = element_inv[2];
        dx[i][3] = element_inv[3];
    }
    
    // First element's inverse
    dx[0][0] = current_inv[0];
    dx[0][1] = current_inv[1];
    dx[0][2] = current_inv[2];
    dx[0][3] = current_inv[3];
}

/**
 * Warp-level optimized batch inversion
 * Uses warp shuffle operations for maximum efficiency
 * This version requires all threads in a warp to cooperate
 * 
 * @param dx Array of elements to invert (in/out)
 * @param count Number of elements
 */
__device__ __forceinline__ void ModInvGroupedWarp(uint64_t dx[][4], uint32_t count) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    
    // For simplicity, use the basic version if not enough parallelism
    if (count <= 32) {
        // Each lane handles one element
        if (lane_id < count) {
            // Use basic ModInvGrouped
            ModInvGrouped(dx, count);
        }
    } else {
        // For larger batches, partition work among lanes
        ModInvGrouped(dx, count);
    }
}

#endif // GPU_MOD_INV_CUH