/*
 * Batch Modular Inverse Implementation for GPU
 * Uses Montgomery's trick for efficient batch inversion
 */

#ifndef BATCH_MOD_INVERSE_CUH
#define BATCH_MOD_INVERSE_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "GPUMath.h"

/**
 * Montgomery's batch modular inverse algorithm
 * Computes modular inverses of multiple values efficiently
 * 
 * Algorithm:
 * 1. Compute cumulative products: p[i] = a[0] * a[1] * ... * a[i]
 * 2. Compute inverse of final product: inv_p = 1/p[n-1]
 * 3. Extract individual inverses using backward pass
 * 
 * Time complexity: O(n) inversions + 3n multiplications
 * vs naive O(n) inversions individually
 */
__device__ __forceinline__ void batchModInverse(
    uint64_t* values,      // Input: values to invert (modified)
    uint64_t* inverses,    // Output: modular inverses
    uint32_t count)        // Number of values to invert
{
    if (count == 0) return;
    if (count == 1) {
        // Single value - use regular inversion
        uint64_t temp[5] = {values[0], values[1], values[2], values[3], 0};
        _ModInv(temp);
        inverses[0] = temp[0]; inverses[1] = temp[1];
        inverses[2] = temp[2]; inverses[3] = temp[3];
        return;
    }
    
    // Temporary storage for cumulative products
    uint64_t products[32][4];  // Support up to 32 values per batch
    if (count > 32) count = 32;  // Safety limit
    
    // Forward pass: compute cumulative products
    // products[0] = values[0]
    products[0][0] = values[0]; products[0][1] = values[1];
    products[0][2] = values[2]; products[0][3] = values[3];
    
    for (uint32_t i = 1; i < count; i++) {
        // products[i] = products[i-1] * values[i]
        uint64_t current_value[4] = {
            values[i * 4], values[i * 4 + 1], 
            values[i * 4 + 2], values[i * 4 + 3]
        };
        _ModMult(products[i], products[i-1], current_value);
    }
    
    // Compute inverse of final product
    uint64_t inv_product[5] = {
        products[count-1][0], products[count-1][1],
        products[count-1][2], products[count-1][3], 0
    };
    _ModInv(inv_product);
    
    // Backward pass: extract individual inverses
    // inverses[i] = inv_product * products[i-1]
    // inv_product = inv_product * values[i]
    
    for (int i = count - 1; i > 0; i--) {
        // inverses[i] = inv_product * products[i-1]
        _ModMult(&inverses[i * 4], inv_product, products[i-1]);
        
        // Update inv_product for next iteration
        uint64_t current_value[4] = {
            values[i * 4], values[i * 4 + 1], 
            values[i * 4 + 2], values[i * 4 + 3]
        };
        _ModMult(inv_product, inv_product, current_value);
    }
    
    // inverses[0] = final inv_product
    inverses[0] = inv_product[0]; inverses[1] = inv_product[1];
    inverses[2] = inv_product[2]; inverses[3] = inv_product[3];
}

/**
 * Warp-optimized batch modular inverse
 * Uses cooperative processing across warp threads
 */
__device__ __forceinline__ void warpBatchModInverse(
    uint64_t* shared_values,   // Shared memory values
    uint64_t* shared_inverses, // Shared memory inverses
    uint32_t warp_size = 32)
{
    const int lane = threadIdx.x & 31;  // Warp lane ID
    
    // Each thread handles one value
    if (lane >= warp_size) return;
    
    uint64_t my_value[4] = {
        shared_values[lane * 4], shared_values[lane * 4 + 1],
        shared_values[lane * 4 + 2], shared_values[lane * 4 + 3]
    };
    
    // Warp-level prefix product computation
    uint64_t prefix_product[4];
    prefix_product[0] = my_value[0]; prefix_product[1] = my_value[1];
    prefix_product[2] = my_value[2]; prefix_product[3] = my_value[3];
    
    // Parallel prefix scan for cumulative products
    #pragma unroll
    for (int offset = 1; offset < warp_size; offset *= 2) {
        uint64_t temp[4];
        // Get value from lane (lane - offset)
        temp[0] = __shfl_up_sync(0xFFFFFFFF, prefix_product[0], offset);
        temp[1] = __shfl_up_sync(0xFFFFFFFF, prefix_product[1], offset);
        temp[2] = __shfl_up_sync(0xFFFFFFFF, prefix_product[2], offset);
        temp[3] = __shfl_up_sync(0xFFFFFFFF, prefix_product[3], offset);
        
        if (lane >= offset) {
            _ModMult(prefix_product, prefix_product, temp);
        }
    }
    
    // Thread 31 has the total product - compute its inverse
    uint64_t total_inverse[5];
    if (lane == 31) {
        total_inverse[0] = prefix_product[0]; total_inverse[1] = prefix_product[1];
        total_inverse[2] = prefix_product[2]; total_inverse[3] = prefix_product[3];
        total_inverse[4] = 0;
        _ModInv(total_inverse);
    }
    
    // Broadcast total inverse to all threads
    total_inverse[0] = __shfl_sync(0xFFFFFFFF, total_inverse[0], 31);
    total_inverse[1] = __shfl_sync(0xFFFFFFFF, total_inverse[1], 31);
    total_inverse[2] = __shfl_sync(0xFFFFFFFF, total_inverse[2], 31);
    total_inverse[3] = __shfl_sync(0xFFFFFFFF, total_inverse[3], 31);
    
    // Compute individual inverses
    uint64_t my_inverse[4];
    if (lane == 0) {
        // First thread: inverse = total_inverse
        my_inverse[0] = total_inverse[0]; my_inverse[1] = total_inverse[1];
        my_inverse[2] = total_inverse[2]; my_inverse[3] = total_inverse[3];
    } else {
        // Other threads: inverse = total_inverse / prefix_product[lane]
        uint64_t prev_prefix[4];
        prev_prefix[0] = __shfl_up_sync(0xFFFFFFFF, prefix_product[0], 1);
        prev_prefix[1] = __shfl_up_sync(0xFFFFFFFF, prefix_product[1], 1);
        prev_prefix[2] = __shfl_up_sync(0xFFFFFFFF, prefix_product[2], 1);
        prev_prefix[3] = __shfl_up_sync(0xFFFFFFFF, prefix_product[3], 1);
        
        _ModMult(my_inverse, total_inverse, prev_prefix);
    }
    
    // Store result in shared memory
    shared_inverses[lane * 4] = my_inverse[0];
    shared_inverses[lane * 4 + 1] = my_inverse[1]; 
    shared_inverses[lane * 4 + 2] = my_inverse[2];
    shared_inverses[lane * 4 + 3] = my_inverse[3];
}

/**
 * High-performance batch EC point addition using batch inverse
 * Processes multiple point additions efficiently
 */
__device__ __forceinline__ void batchECPointAdd(
    uint64_t* points_x,        // Input/Output: x coordinates [count][4]
    uint64_t* points_y,        // Input/Output: y coordinates [count][4]  
    const uint64_t* add_x,     // Point to add: x coordinate [4]
    const uint64_t* add_y,     // Point to add: y coordinate [4]
    uint32_t count)            // Number of points to process
{
    if (count == 0) return;
    
    // Batch compute slopes using batch inverse
    uint64_t dx_values[32 * 4];  // dx values for batch inversion
    uint64_t dy_values[32 * 4];  // dy values (numerators)
    uint64_t slopes[32 * 4];     // Computed slopes
    
    uint32_t batch_count = (count > 32) ? 32 : count;
    
    // Compute dx and dy for all points
    for (uint32_t i = 0; i < batch_count; i++) {
        // dx = add_x - points_x[i]
        ModSub256(&dx_values[i * 4], (uint64_t*)add_x, &points_x[i * 4]);
        
        // dy = add_y - points_y[i] 
        ModSub256(&dy_values[i * 4], (uint64_t*)add_y, &points_y[i * 4]);
    }
    
    // Batch invert all dx values
    uint64_t dx_inverses[32 * 4];
    batchModInverse(dx_values, dx_inverses, batch_count);
    
    // Compute slopes and new points
    for (uint32_t i = 0; i < batch_count; i++) {
        // slope = dy * dx^(-1)
        _ModMult(&slopes[i * 4], &dy_values[i * 4], &dx_inverses[i * 4]);
        
        // x_new = slope^2 - points_x[i] - add_x
        uint64_t slope_squared[4];
        _ModSqr(slope_squared, &slopes[i * 4]);
        
        uint64_t x_new[4];
        ModSub256(x_new, slope_squared, &points_x[i * 4]);
        ModSub256(x_new, x_new, (uint64_t*)add_x);
        
        // y_new = slope * (points_x[i] - x_new) - points_y[i]
        uint64_t x_diff[4];
        ModSub256(x_diff, &points_x[i * 4], x_new);
        
        uint64_t y_new[4];
        _ModMult(y_new, &slopes[i * 4], x_diff);
        ModSub256(y_new, y_new, &points_y[i * 4]);
        
        // Update points
        points_x[i * 4] = x_new[0]; points_x[i * 4 + 1] = x_new[1];
        points_x[i * 4 + 2] = x_new[2]; points_x[i * 4 + 3] = x_new[3];
        
        points_y[i * 4] = y_new[0]; points_y[i * 4 + 1] = y_new[1];
        points_y[i * 4 + 2] = y_new[2]; points_y[i * 4 + 3] = y_new[3];
    }
}

/**
 * Wrapper function for batch modular inverse with specific naming
 * Used by the batch stepping optimization code
 */
__device__ __forceinline__ void batch_mod_inverse_256(
    uint64_t* values,      // Input: flat array of values to invert
    uint64_t* inverses,    // Output: flat array of modular inverses
    uint32_t count)        // Number of 256-bit values to invert
{
    batchModInverse(values, inverses, count);
}

#endif // BATCH_MOD_INVERSE_CUH
