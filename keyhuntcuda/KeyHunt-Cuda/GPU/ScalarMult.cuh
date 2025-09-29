/**
 * ScalarMult.cuh
 * Efficient scalar multiplication implementation for secp256k1
 * Uses windowed NAF method for optimal performance
 */

#ifndef SCALAR_MULT_CUH
#define SCALAR_MULT_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "GPUMath.h"
#include "ECPointOps.cuh"

/**
 * Window-based scalar multiplication
 * Computes k*P efficiently using precomputed tables
 * 
 * @param k Scalar multiplier (256-bit)
 * @param px Base point x coordinate
 * @param py Base point y coordinate
 * @param rx Result x coordinate
 * @param ry Result y coordinate
 */
__device__ void scalar_mult_window(
    const uint64_t k[4],
    const uint64_t px[4],
    const uint64_t py[4],
    uint64_t rx[4],
    uint64_t ry[4])
{
    // Window size (4-bit windows are a good trade-off)
    const int WINDOW_SIZE = 4;
    const int WINDOW_MASK = (1 << WINDOW_SIZE) - 1;
    
    // Initialize result to identity (point at infinity)
    rx[0] = 0; rx[1] = 0; rx[2] = 0; rx[3] = 0;
    ry[0] = 0; ry[1] = 0; ry[2] = 0; ry[3] = 0;
    
    // Precompute odd multiples: P, 3P, 5P, ..., (2^w-1)P
    uint64_t precomp_x[8][4], precomp_y[8][4];
    
    // Store P
    for (int i = 0; i < 4; i++) {
        precomp_x[0][i] = px[i];
        precomp_y[0][i] = py[i];
    }
    
    // Compute 2P
    uint64_t doubleP_x[4], doubleP_y[4];
    for (int i = 0; i < 4; i++) {
        doubleP_x[i] = px[i];
        doubleP_y[i] = py[i];
    }
    // Double the point
    uint64_t lambda[4], lambda_sqr[4], temp[4];
    
    // lambda = 3*x^2 / 2*y
    uint64_t x_sqr[4], three_x_sqr[4], two_y[4];
    _ModSqr(x_sqr, px);
    ModAdd256(three_x_sqr, x_sqr, x_sqr);
    ModAdd256(three_x_sqr, three_x_sqr, x_sqr);
    ModAdd256(two_y, py, py);
    
    uint64_t two_y_inv[5];
    two_y_inv[0] = two_y[0]; two_y_inv[1] = two_y[1];
    two_y_inv[2] = two_y[2]; two_y_inv[3] = two_y[3];
    two_y_inv[4] = 0;
    _ModInv(two_y_inv);
    
    _ModMult(lambda, three_x_sqr, two_y_inv);
    _ModSqr(lambda_sqr, lambda);
    
    // x_new = lambda^2 - 2*x
    uint64_t two_x[4];
    ModAdd256(two_x, px, px);
    ModSub256(doubleP_x, lambda_sqr, two_x);
    
    // y_new = lambda*(x - x_new) - y
    ModSub256(temp, px, doubleP_x);
    _ModMult(doubleP_y, lambda, temp);
    ModSub256(doubleP_y, doubleP_y, py);
    
    // Compute 3P, 5P, 7P, ... (odd multiples)
    for (int i = 1; i < 8; i++) {
        // precomp[i] = precomp[i-1] + 2P
        for (int j = 0; j < 4; j++) {
            precomp_x[i][j] = precomp_x[i-1][j];
            precomp_y[i][j] = precomp_y[i-1][j];
        }
        _addPoint(precomp_x[i], precomp_y[i], doubleP_x, doubleP_y);
    }
    
    // Process scalar from high bits to low bits
    bool first_window = true;
    
    for (int bit_pos = 255; bit_pos >= 0; bit_pos -= WINDOW_SIZE) {
        if (!first_window) {
            // Double WINDOW_SIZE times
            for (int i = 0; i < WINDOW_SIZE; i++) {
                // Double the result point
                if (rx[0] != 0 || rx[1] != 0 || rx[2] != 0 || rx[3] != 0) {
                    // Compute 2*R
                    uint64_t r_x_sqr[4], three_r_x_sqr[4], two_r_y[4];
                    _ModSqr(r_x_sqr, rx);
                    ModAdd256(three_r_x_sqr, r_x_sqr, r_x_sqr);
                    ModAdd256(three_r_x_sqr, three_r_x_sqr, r_x_sqr);
                    ModAdd256(two_r_y, ry, ry);
                    
                    uint64_t two_r_y_inv[5];
                    two_r_y_inv[0] = two_r_y[0]; two_r_y_inv[1] = two_r_y[1];
                    two_r_y_inv[2] = two_r_y[2]; two_r_y_inv[3] = two_r_y[3];
                    two_r_y_inv[4] = 0;
                    _ModInv(two_r_y_inv);
                    
                    uint64_t r_lambda[4], r_lambda_sqr[4], r_temp[4];
                    _ModMult(r_lambda, three_r_x_sqr, two_r_y_inv);
                    _ModSqr(r_lambda_sqr, r_lambda);
                    
                    uint64_t two_r_x[4], new_rx[4], new_ry[4];
                    ModAdd256(two_r_x, rx, rx);
                    ModSub256(new_rx, r_lambda_sqr, two_r_x);
                    
                    ModSub256(r_temp, rx, new_rx);
                    _ModMult(new_ry, r_lambda, r_temp);
                    ModSub256(new_ry, new_ry, ry);
                    
                    for (int j = 0; j < 4; j++) {
                        rx[j] = new_rx[j];
                        ry[j] = new_ry[j];
                    }
                }
            }
        } else {
            first_window = false;
        }
        
        // Extract window bits
        int window_start = bit_pos - WINDOW_SIZE + 1;
        if (window_start < 0) window_start = 0;
        
        uint32_t window_value = 0;
        for (int i = bit_pos; i >= window_start; i--) {
            int word_idx = i / 64;
            int bit_idx = i % 64;
            uint32_t bit = (k[word_idx] >> bit_idx) & 1;
            window_value = (window_value << 1) | bit;
        }
        
        // Add appropriate precomputed multiple if non-zero
        if (window_value > 0) {
            if (window_value & 1) {
                // Odd value - use precomputed
                int precomp_idx = (window_value - 1) / 2;
                if (precomp_idx < 8) {
                    if (rx[0] == 0 && rx[1] == 0 && rx[2] == 0 && rx[3] == 0) {
                        // Result is identity, just copy
                        for (int i = 0; i < 4; i++) {
                            rx[i] = precomp_x[precomp_idx][i];
                            ry[i] = precomp_y[precomp_idx][i];
                        }
                    } else {
                        // Add to result
                        _addPoint(rx, ry, precomp_x[precomp_idx], precomp_y[precomp_idx]);
                    }
                }
            } else {
                // Even value - need to handle differently
                // For simplicity, we can decompose into powers of 2 and odd part
                uint32_t odd_part = window_value;
                int two_count = 0;
                while ((odd_part & 1) == 0) {
                    odd_part >>= 1;
                    two_count++;
                }
                
                // Add the odd part
                if (odd_part > 0) {
                    int precomp_idx = (odd_part - 1) / 2;
                    if (precomp_idx < 8) {
                        uint64_t temp_x[4], temp_y[4];
                        for (int i = 0; i < 4; i++) {
                            temp_x[i] = precomp_x[precomp_idx][i];
                            temp_y[i] = precomp_y[precomp_idx][i];
                        }
                        
                        // Double two_count times
                        for (int i = 0; i < two_count; i++) {
                            // Double temp point
                            uint64_t t_lambda[4], t_lambda_sqr[4], t_temp[4];
                            uint64_t t_x_sqr[4], t_three_x_sqr[4], t_two_y[4];
                            
                            _ModSqr(t_x_sqr, temp_x);
                            ModAdd256(t_three_x_sqr, t_x_sqr, t_x_sqr);
                            ModAdd256(t_three_x_sqr, t_three_x_sqr, t_x_sqr);
                            ModAdd256(t_two_y, temp_y, temp_y);
                            
                            uint64_t t_two_y_inv[5];
                            t_two_y_inv[0] = t_two_y[0]; t_two_y_inv[1] = t_two_y[1];
                            t_two_y_inv[2] = t_two_y[2]; t_two_y_inv[3] = t_two_y[3];
                            t_two_y_inv[4] = 0;
                            _ModInv(t_two_y_inv);
                            
                            _ModMult(t_lambda, t_three_x_sqr, t_two_y_inv);
                            _ModSqr(t_lambda_sqr, t_lambda);
                            
                            uint64_t t_two_x[4];
                            ModAdd256(t_two_x, temp_x, temp_x);
                            uint64_t new_temp_x[4];
                            ModSub256(new_temp_x, t_lambda_sqr, t_two_x);
                            
                            ModSub256(t_temp, temp_x, new_temp_x);
                            uint64_t new_temp_y[4];
                            _ModMult(new_temp_y, t_lambda, t_temp);
                            ModSub256(new_temp_y, new_temp_y, temp_y);
                            
                            for (int j = 0; j < 4; j++) {
                                temp_x[j] = new_temp_x[j];
                                temp_y[j] = new_temp_y[j];
                            }
                        }
                        
                        if (rx[0] == 0 && rx[1] == 0 && rx[2] == 0 && rx[3] == 0) {
                            for (int i = 0; i < 4; i++) {
                                rx[i] = temp_x[i];
                                ry[i] = temp_y[i];
                            }
                        } else {
                            _addPoint(rx, ry, temp_x, temp_y);
                        }
                    }
                }
            }
        }
    }
}