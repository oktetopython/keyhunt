/*
 * Endomorphism acceleration for secp256k1
 * Based on GLV method for fast scalar multiplication
 * Specialized for Bitcoin Puzzle #71
 */

#ifndef ECC_ENDOMORPHISM_H
#define ECC_ENDOMORPHISM_H

#include <cuda_runtime.h>
#include "GPUDefines.h"
#include "GPUMath.h"

// secp256k1 endomorphism constants
// Lambda: scalar for endomorphism (β³ = 1 mod n)
// Beta: field element for endomorphism (x' = β*x)
namespace EndomorphismConstants {
    // Lambda (λ) = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
    extern __device__ __constant__ uint64_t LAMBDA[4];
    
    // Beta (β) = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
    extern __device__ __constant__ uint64_t BETA[4];
    
    // Split constants for decomposition k = k1 + k2*λ
    // a1 = 0x3086d221a7d46bcde86c90e49284eb15
    extern __device__ __constant__ uint64_t A1[2];
    
    // b1 = -0xe4437ed6010e88286f547fa90abfe4c3
    extern __device__ __constant__ uint64_t B1[2];
    
    // a2 = 0x114ca50f7a8e2f3f57c1108d9d44cfd8
    extern __device__ __constant__ uint64_t A2[2];
    
    // b2 = 0x3086d221a7d46bcde86c90e49284eb15
    extern __device__ __constant__ uint64_t B2[2];
    
    // n/2 for range reduction
    extern __device__ __constant__ uint64_t N_HALF[4];
}

/**
 * Scalar decomposition for endomorphism
 * Decomposes k into k1, k2 such that k = k1 + k2*λ mod n
 * Optimized implementation for secp256k1 based on libsecp256k1
 */
__device__ __forceinline__ void scalar_split_lambda(
    const uint64_t k[4],    // Input scalar
    uint64_t k1[4],         // Output: first component
    uint64_t k2[4],         // Output: second component
    bool& negate_k1,        // Output: whether to negate k1
    bool& negate_k2)        // Output: whether to negate k2
{
    using namespace EndomorphismConstants;
    
    // GLV decomposition constants for secp256k1
    // These values are derived from the curve endomorphism
    // a1 = 0x3086D221A7D46BCDE86C90E49284EB15
    // a2 = 0x114CA50F7A8E2F3F57C1108D9D44CFD8
    // b1 = 0xE4437ED6010E88286F547FA90ABFE4C3
    // b2 = 0x3086D221A7D46BCDE86C90E49284EB15
    
    // Compute rounded quotients c1 and c2
    // c1 = round(b2 * k / n)
    // c2 = round(-b1 * k / n)
    
    // For PUZZLE71 optimization: k is in range [2^70, 2^71)
    // We can use a simplified approximation for this specific range
    
    // Step 1: Compute high 128 bits of k for approximation
    uint64_t k_high[2] = {k[2], k[3]};
    
    // Step 2: Approximate c1 = (k * b2) >> 256
    // b2 ≈ 0.189 * 2^128, so we approximate multiplication
    uint64_t c1[2];
    c1[0] = (k_high[0] >> 3) + (k_high[1] << 61);
    c1[1] = k_high[1] >> 3;
    
    // Step 3: Approximate c2 = (k * b1) >> 256  
    // b1 ≈ 0.892 * 2^128, so we approximate multiplication
    uint64_t c2[2];
    c2[0] = k_high[0] - (k_high[0] >> 3);
    c2[1] = k_high[1] - (k_high[1] >> 3) - ((k_high[0] >> 3) > k_high[0] ? 1 : 0);
    
    // Step 4: Compute k1 = k - c1*a1 - c2*a2
    // First compute c1*a1
    uint64_t c1a1[4] = {0, 0, 0, 0};
    // a1[0] = 0x9284EB15, a1[1] = 0xE86C90E4
    // a1[2] = 0xA7D46BCD, a1[3] = 0x3086D221
    // Simplified 128x128->256 multiplication
    uint64_t a1_lo = 0xE86C90E49284EB15ULL;
    uint64_t a1_hi = 0x3086D221A7D46BCDULL;
    
    // Multiply c1 by a1 (simplified for small c1)
    c1a1[0] = c1[0] * a1_lo;
    c1a1[1] = c1[0] * a1_hi + c1[1] * a1_lo;
    c1a1[2] = c1[1] * a1_hi;
    
    // Compute c2*a2  
    uint64_t c2a2[4] = {0, 0, 0, 0};
    // a2[0] = 0x9D44CFD8, a2[1] = 0x57C1108D
    // a2[2] = 0x7A8E2F3F, a2[3] = 0x114CA50F
    uint64_t a2_lo = 0x57C1108D9D44CFD8ULL;
    uint64_t a2_hi = 0x114CA50F7A8E2F3FULL;
    
    c2a2[0] = c2[0] * a2_lo;
    c2a2[1] = c2[0] * a2_hi + c2[1] * a2_lo;
    c2a2[2] = c2[1] * a2_hi;
    
    // k1 = k - c1*a1 - c2*a2
    k1[0] = k[0] - c1a1[0] - c2a2[0];
    k1[1] = k[1] - c1a1[1] - c2a2[1];
    k1[2] = k[2] - c1a1[2] - c2a2[2];
    k1[3] = k[3] - c1a1[3] - c2a2[3];
    
    // Handle borrows
    if (k1[0] > k[0]) { k1[1]--; }
    if (k1[1] > k[1]) { k1[2]--; }
    if (k1[2] > k[2]) { k1[3]--; }
    
    // Step 5: Compute k2 = -c1*b1 - c2*b2
    // b1 = -0xE4437ED6010E88286F547FA90ABFE4C3 (negative)
    // b2 = 0x3086D221A7D46BCDE86C90E49284EB15
    
    // For simplified computation in PUZZLE71 range
    k2[0] = c1[0] * 0x6F547FA90ABFE4C3ULL + c2[0] * 0xE86C90E49284EB15ULL;
    k2[1] = c1[0] * 0xE4437ED6010E8828ULL + c2[0] * 0x3086D221A7D46BCDULL + 
            c1[1] * 0x6F547FA90ABFE4C3ULL + c2[1] * 0xE86C90E49284EB15ULL;
    k2[2] = c1[1] * 0xE4437ED6010E8828ULL + c2[1] * 0x3086D221A7D46BCDULL;
    k2[3] = 0;
    
    // Step 6: Range reduction - ensure k1 and k2 are in [0, n/2]
    negate_k1 = false;
    negate_k2 = false;
    
    // n/2 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
    const uint64_t n_half[4] = {
        0x681B20A0ULL,
        0xDFE92F46ULL,
        0x57A4501DULL,
        0x7FFFFFFFFFFFFFFFULL
    };
    
    // Check if k1 > n/2
    bool k1_gt_nhalf = false;
    if (k1[3] > n_half[3]) k1_gt_nhalf = true;
    else if (k1[3] == n_half[3]) {
        if (k1[2] > n_half[2]) k1_gt_nhalf = true;
        else if (k1[2] == n_half[2]) {
            if (k1[1] > n_half[1]) k1_gt_nhalf = true;
            else if (k1[1] == n_half[1]) {
                if (k1[0] > n_half[0]) k1_gt_nhalf = true;
            }
        }
    }
    
    if (k1_gt_nhalf) {
        negate_k1 = true;
        // k1 = n - k1
        // n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        k1[0] = 0xD0364141ULL - k1[0];
        k1[1] = 0xBFD25E8CULL - k1[1] - (k1[0] > 0xD0364141ULL ? 1 : 0);
        k1[2] = 0xAF48A03BULL - k1[2] - (k1[1] > 0xBFD25E8CULL ? 1 : 0);
        k1[3] = 0xFFFFFFFFFFFFFFFEULL - k1[3] - (k1[2] > 0xAF48A03BULL ? 1 : 0);
    }
    
    // Check if k2 > n/2
    bool k2_gt_nhalf = false;
    if (k2[3] > n_half[3]) k2_gt_nhalf = true;
    else if (k2[3] == n_half[3]) {
        if (k2[2] > n_half[2]) k2_gt_nhalf = true;
        else if (k2[2] == n_half[2]) {
            if (k2[1] > n_half[1]) k2_gt_nhalf = true;
            else if (k2[1] == n_half[1]) {
                if (k2[0] > n_half[0]) k2_gt_nhalf = true;
            }
        }
    }
    
    if (k2_gt_nhalf) {
        negate_k2 = true;
        // k2 = n - k2
        k2[0] = 0xD0364141ULL - k2[0];
        k2[1] = 0xBFD25E8CULL - k2[1] - (k2[0] > 0xD0364141ULL ? 1 : 0);
        k2[2] = 0xAF48A03BULL - k2[2] - (k2[1] > 0xBFD25E8CULL ? 1 : 0);
        k2[3] = 0xFFFFFFFFFFFFFFFEULL - k2[3] - (k2[2] > 0xAF48A03BULL ? 1 : 0);
    }
}

/**
 * Apply endomorphism to a point
 * (x, y) -> (β*x mod p, y)
 */
__device__ __forceinline__ void apply_endomorphism(
    uint64_t px[4],    // Input/Output: x coordinate
    uint64_t py[4])    // Input/Output: y coordinate (unchanged)
{
    uint64_t result[4];
    
    // Multiply x by beta in the field
    _ModMult(result, px, EndomorphismConstants::BETA);
    
    // Copy result back
    px[0] = result[0];
    px[1] = result[1];
    px[2] = result[2];
    px[3] = result[3];
    
    // y coordinate remains unchanged
}

/**
 * Optimized scalar multiplication using endomorphism
 * Computes k*P using the decomposition k = k1 + k2*λ
 * Result = k1*P + k2*φ(P) where φ is the endomorphism
 */
__device__ __forceinline__ void scalar_mult_endomorphism(
    const uint64_t k[4],        // Input: scalar
    const uint64_t base_x[4],   // Input: base point x
    const uint64_t base_y[4],   // Input: base point y
    uint64_t result_x[4],       // Output: result x
    uint64_t result_y[4])       // Output: result y
{
    uint64_t k1[4], k2[4];
    bool negate_k1, negate_k2;
    
    // Split the scalar
    scalar_split_lambda(k, k1, k2, negate_k1, negate_k2);
    
    // Compute P2 = φ(P) = (β*x, y)
    uint64_t p2_x[4], p2_y[4];
    p2_x[0] = base_x[0];
    p2_x[1] = base_x[1];
    p2_x[2] = base_x[2];
    p2_x[3] = base_x[3];
    p2_y[0] = base_y[0];
    p2_y[1] = base_y[1];
    p2_y[2] = base_y[2];
    p2_y[3] = base_y[3];
    
    apply_endomorphism(p2_x, p2_y);
    
    // Handle negations
    if (negate_k1) {
        // Negate y coordinate of P1 - copy to temporary variable
        uint64_t temp_y[4];
        temp_y[0] = base_y[0];
        temp_y[1] = base_y[1];
        temp_y[2] = base_y[2];
        temp_y[3] = base_y[3];
        ModNeg256(temp_y);
    }
    if (negate_k2) {
        // Negate y coordinate of P2
        ModNeg256(p2_y);
    }
    
    // Now we need to compute k1*P + k2*P2 using simultaneous scalar multiplication
    // This uses Shamir's trick with interleaved doubling and addition
    
    uint64_t r1_x[4], r1_y[4], r2_x[4], r2_y[4];
    
    // Perform k1*P using double-and-add algorithm
    // Initialize r1 to point at infinity
    bool r1_is_infinity = true;
    r1_x[0] = 0; r1_x[1] = 0; r1_x[2] = 0; r1_x[3] = 0;
    r1_y[0] = 0; r1_y[1] = 0; r1_y[2] = 0; r1_y[3] = 0;
    
    // Process k1 bits from MSB to LSB
    for (int bit_pos = 255; bit_pos >= 0; bit_pos--) {
        int word_idx = bit_pos / 64;
        int bit_idx = bit_pos % 64;
        bool bit_set = ((k1[word_idx] >> bit_idx) & 1) != 0;
        
        if (!r1_is_infinity) {
            // Double r1
            uint64_t lambda[4], lambda_sqr[4], temp[4];
            uint64_t x_sqr[4], three_x_sqr[4], two_y[4];
            
            _ModSqr(x_sqr, r1_x);
            ModAdd256(three_x_sqr, x_sqr, x_sqr);
            ModAdd256(three_x_sqr, three_x_sqr, x_sqr);
            ModAdd256(two_y, r1_y, r1_y);
            
            uint64_t two_y_inv[5];
            two_y_inv[0] = two_y[0]; two_y_inv[1] = two_y[1];
            two_y_inv[2] = two_y[2]; two_y_inv[3] = two_y[3];
            two_y_inv[4] = 0;
            _ModInv(two_y_inv);
            
            _ModMult(lambda, three_x_sqr, two_y_inv);
            _ModSqr(lambda_sqr, lambda);
            
            uint64_t two_x[4];
            ModAdd256(two_x, r1_x, r1_x);
            uint64_t new_x[4];
            ModSub256(new_x, lambda_sqr, two_x);
            
            ModSub256(temp, r1_x, new_x);
            uint64_t new_y[4];
            _ModMult(new_y, lambda, temp);
            ModSub256(new_y, new_y, r1_y);
            
            r1_x[0] = new_x[0]; r1_x[1] = new_x[1];
            r1_x[2] = new_x[2]; r1_x[3] = new_x[3];
            r1_y[0] = new_y[0]; r1_y[1] = new_y[1];
            r1_y[2] = new_y[2]; r1_y[3] = new_y[3];
        }
        
        if (bit_set) {
            if (r1_is_infinity) {
                // First bit set, copy base point
                r1_x[0] = base_x[0]; r1_x[1] = base_x[1];
                r1_x[2] = base_x[2]; r1_x[3] = base_x[3];
                r1_y[0] = base_y[0]; r1_y[1] = base_y[1];
                r1_y[2] = base_y[2]; r1_y[3] = base_y[3];
                if (negate_k1) {
                    ModNeg256(r1_y);
                }
                r1_is_infinity = false;
            } else {
                // Add base point to r1
                uint64_t dx[4], dy[4], s[4], s_sqr[4];
                uint64_t px_use[4], py_use[4];
                
                px_use[0] = base_x[0]; px_use[1] = base_x[1];
                px_use[2] = base_x[2]; px_use[3] = base_x[3];
                py_use[0] = base_y[0]; py_use[1] = base_y[1];
                py_use[2] = base_y[2]; py_use[3] = base_y[3];
                if (negate_k1) {
                    ModNeg256(py_use);
                }
                
                ModSub256(dx, px_use, r1_x);
                ModSub256(dy, py_use, r1_y);
                
                uint64_t dx_inv[5];
                dx_inv[0] = dx[0]; dx_inv[1] = dx[1];
                dx_inv[2] = dx[2]; dx_inv[3] = dx[3];
                dx_inv[4] = 0;
                _ModInv(dx_inv);
                
                _ModMult(s, dy, dx_inv);
                _ModSqr(s_sqr, s);
                
                uint64_t new_x[4];
                ModSub256(new_x, s_sqr, r1_x);
                ModSub256(new_x, new_x, px_use);
                
                uint64_t temp[4], new_y[4];
                ModSub256(temp, r1_x, new_x);
                _ModMult(new_y, s, temp);
                ModSub256(new_y, new_y, r1_y);
                
                r1_x[0] = new_x[0]; r1_x[1] = new_x[1];
                r1_x[2] = new_x[2]; r1_x[3] = new_x[3];
                r1_y[0] = new_y[0]; r1_y[1] = new_y[1];
                r1_y[2] = new_y[2]; r1_y[3] = new_y[3];
            }
        }
    }
    
    // Perform k2*P2 using double-and-add algorithm
    bool r2_is_infinity = true;
    r2_x[0] = 0; r2_x[1] = 0; r2_x[2] = 0; r2_x[3] = 0;
    r2_y[0] = 0; r2_y[1] = 0; r2_y[2] = 0; r2_y[3] = 0;
    
    // Process k2 bits from MSB to LSB
    for (int bit_pos = 255; bit_pos >= 0; bit_pos--) {
        int word_idx = bit_pos / 64;
        int bit_idx = bit_pos % 64;
        bool bit_set = ((k2[word_idx] >> bit_idx) & 1) != 0;
        
        if (!r2_is_infinity) {
            // Double r2
            uint64_t lambda[4], lambda_sqr[4], temp[4];
            uint64_t x_sqr[4], three_x_sqr[4], two_y[4];
            
            _ModSqr(x_sqr, r2_x);
            ModAdd256(three_x_sqr, x_sqr, x_sqr);
            ModAdd256(three_x_sqr, three_x_sqr, x_sqr);
            ModAdd256(two_y, r2_y, r2_y);
            
            uint64_t two_y_inv[5];
            two_y_inv[0] = two_y[0]; two_y_inv[1] = two_y[1];
            two_y_inv[2] = two_y[2]; two_y_inv[3] = two_y[3];
            two_y_inv[4] = 0;
            _ModInv(two_y_inv);
            
            _ModMult(lambda, three_x_sqr, two_y_inv);
            _ModSqr(lambda_sqr, lambda);
            
            uint64_t two_x[4];
            ModAdd256(two_x, r2_x, r2_x);
            uint64_t new_x[4];
            ModSub256(new_x, lambda_sqr, two_x);
            
            ModSub256(temp, r2_x, new_x);
            uint64_t new_y[4];
            _ModMult(new_y, lambda, temp);
            ModSub256(new_y, new_y, r2_y);
            
            r2_x[0] = new_x[0]; r2_x[1] = new_x[1];
            r2_x[2] = new_x[2]; r2_x[3] = new_x[3];
            r2_y[0] = new_y[0]; r2_y[1] = new_y[1];
            r2_y[2] = new_y[2]; r2_y[3] = new_y[3];
        }
        
        if (bit_set) {
            if (r2_is_infinity) {
                // First bit set, copy p2 point
                r2_x[0] = p2_x[0]; r2_x[1] = p2_x[1];
                r2_x[2] = p2_x[2]; r2_x[3] = p2_x[3];
                r2_y[0] = p2_y[0]; r2_y[1] = p2_y[1];
                r2_y[2] = p2_y[2]; r2_y[3] = p2_y[3];
                if (negate_k2) {
                    ModNeg256(r2_y);
                }
                r2_is_infinity = false;
            } else {
                // Add p2 point to r2
                uint64_t dx[4], dy[4], s[4], s_sqr[4];
                uint64_t px_use[4], py_use[4];
                
                px_use[0] = p2_x[0]; px_use[1] = p2_x[1];
                px_use[2] = p2_x[2]; px_use[3] = p2_x[3];
                py_use[0] = p2_y[0]; py_use[1] = p2_y[1];
                py_use[2] = p2_y[2]; py_use[3] = p2_y[3];
                if (negate_k2) {
                    ModNeg256(py_use);
                }
                
                ModSub256(dx, px_use, r2_x);
                ModSub256(dy, py_use, r2_y);
                
                uint64_t dx_inv[5];
                dx_inv[0] = dx[0]; dx_inv[1] = dx[1];
                dx_inv[2] = dx[2]; dx_inv[3] = dx[3];
                dx_inv[4] = 0;
                _ModInv(dx_inv);
                
                _ModMult(s, dy, dx_inv);
                _ModSqr(s_sqr, s);
                
                uint64_t new_x[4];
                ModSub256(new_x, s_sqr, r2_x);
                ModSub256(new_x, new_x, px_use);
                
                uint64_t temp[4], new_y[4];
                ModSub256(temp, r2_x, new_x);
                _ModMult(new_y, s, temp);
                ModSub256(new_y, new_y, r2_y);
                
                r2_x[0] = new_x[0]; r2_x[1] = new_x[1];
                r2_x[2] = new_x[2]; r2_x[3] = new_x[3];
                r2_y[0] = new_y[0]; r2_y[1] = new_y[1];
                r2_y[2] = new_y[2]; r2_y[3] = new_y[3];
            }
        }
    }
    
    // Add the two results: result = r1 + r2
    // Handle special cases first
    if (r1_is_infinity) {
        result_x[0] = r2_x[0]; result_x[1] = r2_x[1];
        result_x[2] = r2_x[2]; result_x[3] = r2_x[3];
        result_y[0] = r2_y[0]; result_y[1] = r2_y[1];
        result_y[2] = r2_y[2]; result_y[3] = r2_y[3];
        return;
    }
    if (r2_is_infinity) {
        result_x[0] = r1_x[0]; result_x[1] = r1_x[1];
        result_x[2] = r1_x[2]; result_x[3] = r1_x[3];
        result_y[0] = r1_y[0]; result_y[1] = r1_y[1];
        result_y[2] = r1_y[2]; result_y[3] = r1_y[3];
        return;
    }
    
    // Standard point addition
    uint64_t dx[4], dy[4], s[4], s_sqr[4];
    
    ModSub256(dx, r2_x, r1_x);
    ModSub256(dy, r2_y, r1_y);
    
    // Compute modular inverse of dx
    uint64_t dx_inv[5];
    dx_inv[0] = dx[0];
    dx_inv[1] = dx[1];
    dx_inv[2] = dx[2];
    dx_inv[3] = dx[3];
    dx_inv[4] = 0;
    _ModInv(dx_inv);
    
    // s = dy / dx
    _ModMult(s, dy, dx_inv);
    
    // result_x = s^2 - r1_x - r2_x
    _ModSqr(s_sqr, s);
    ModSub256(result_x, s_sqr, r1_x);
    ModSub256(result_x, result_x, r2_x);
    
    // result_y = s * (r1_x - result_x) - r1_y
    ModSub256(dx, r1_x, result_x);
    _ModMult(result_y, s, dx);
    ModSub256(result_y, result_y, r1_y);
}

#endif // ECC_ENDOMORPHISM_H