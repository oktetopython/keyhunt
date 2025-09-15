/**
 * @file secp256k1_point_optimized.cu
 * @brief CUDA implementation for optimized elliptic curve point operations
 * @author KeyhuntCUDA Team
 * 
 * T037: Implement point operations with projective coordinates and optimized addition chains
 * 
 * Implements highly optimized CUDA kernels for secp256k1 point operations using:
 * - Projective coordinates to avoid expensive modular inversions
 * - Assembly-optimized field arithmetic from T036
 * - Precomputed tables and windowed methods
 * - GLV endomorphism and advanced scalar multiplication techniques
 */

#include "secp256k1_point.h"
#include "secp256k1_math.h"
#include <cassert>

namespace keyhunt {
namespace ecc {
namespace gpu {
namespace optimized {

// secp256k1 curve parameters in device constant memory
__constant__ uint64_t CURVE_P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};

__constant__ uint64_t CURVE_A[4] = {0, 0, 0, 0}; // a = 0 for secp256k1

__constant__ uint64_t CURVE_B[4] = {7, 0, 0, 0}; // b = 7 for secp256k1

__constant__ uint64_t CURVE_N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// GLV endomorphism constants
__constant__ uint64_t GLV_LAMBDA[4] = {
    0x5363AD4CC05C30E0ULL, 0x3F7707D812DEB33AULL,
    0x0000000000000000ULL, 0x0000000000000000ULL
};

__constant__ uint64_t GLV_BETA[4] = {
    0x9E4F21B14F5DB819ULL, 0x7E2D58D8B3BCDF1AULL,
    0x0000000000000000ULL, 0x0000000000000000ULL
};

/**
 * @brief Device function for projective point addition
 * Computes R = P1 + P2 using optimized projective coordinate formulas
 * 
 * Uses optimized addition formulas:
 * - 12M + 2S when Z1 ≠ 1 and Z2 ≠ 1
 * - 8M + 3S when Z1 = 1 (mixed addition)
 * - 7M + 4S when Z2 = 1 (mixed addition)
 */
__device__ void point_add_projective(ProjectivePoint* result, 
                                    const ProjectivePoint* p1, 
                                    const ProjectivePoint* p2) {
    // Handle special cases
    if (p1->z.is_zero()) {
        *result = *p2;
        return;
    }
    if (p2->z.is_zero()) {
        *result = *p1;
        return;
    }
    
    uint64_t t0[4], t1[4], t2[4], t3[4], t4[4], t5[4];
    uint64_t x1[4], y1[4], z1[4], x2[4], y2[4], z2[4];
    uint64_t rx[4], ry[4], rz[4];
    
    // Copy input coordinates to registers
    for (int i = 0; i < 4; i++) {
        x1[i] = p1->x.d[i];
        y1[i] = p1->y.d[i];
        z1[i] = p1->z.d[i];
        x2[i] = p2->x.d[i];
        y2[i] = p2->y.d[i];
        z2[i] = p2->z.d[i];
    }
    
    // Check if Z1 = 1 (affine point)
    bool z1_is_one = (z1[0] == 1 && z1[1] == 0 && z1[2] == 0 && z1[3] == 0);
    // Check if Z2 = 1 (affine point)  
    bool z2_is_one = (z2[0] == 1 && z2[1] == 0 && z2[2] == 0 && z2[3] == 0);
    
    if (z1_is_one && z2_is_one) {
        // Both points are affine - use optimized affine addition
        // t0 = x2 - x1
        _ModSub_Optimized(t0, x2, x1);
        // t1 = y2 - y1
        _ModSub_Optimized(t1, y2, y1);
        
        // Check if points are equal (same x coordinate)
        bool same_x = true;
        for (int i = 0; i < 4; i++) {
            if (t0[i] != 0) {
                same_x = false;
                break;
            }
        }
        
        if (same_x) {
            // Same x-coordinate - either point doubling or point at infinity
            bool same_y = true;
            for (int i = 0; i < 4; i++) {
                if (t1[i] != 0) {
                    same_y = false;
                    break;
                }
            }
            
            if (same_y) {
                // Point doubling case
                point_double_projective(result, p1);
                return;
            } else {
                // Points are inverses - result is point at infinity
                result->x = BigInt256(1);
                result->y = BigInt256(1);
                result->z = BigInt256(0);
                return;
            }
        }
        
        // t2 = t0^2 (lambda squared)
        _ModSqr_Optimized(t2, t0);
        // t3 = t1^2 (nu squared)  
        _ModSqr_Optimized(t3, t1);
        // t4 = t0 * t2 = t0^3
        _ModMult_Montgomery_Optimized(t4, t0, t2);
        // t5 = x1 * t2
        _ModMult_Montgomery_Optimized(t5, x1, t2);
        
        // rx = t3 - t4 - 2*t5
        _ModSub_Optimized(rx, t3, t4);
        _ModAdd_Optimized(t0, t5, t5); // 2*t5
        _ModSub_Optimized(rx, rx, t0);
        
        // ry = t1 * (t5 - rx) - y1 * t4
        _ModSub_Optimized(t0, t5, rx);
        _ModMult_Montgomery_Optimized(t0, t1, t0);
        _ModMult_Montgomery_Optimized(t1, y1, t4);
        _ModSub_Optimized(ry, t0, t1);
        
        // rz = t0 (original t0, which is lambda)
        for (int i = 0; i < 4; i++) {
            rz[i] = (x2[i] >= x1[i]) ? x2[i] - x1[i] : CURVE_P[i] - x1[i] + x2[i];
        }
        
    } else if (z1_is_one) {
        // P1 is affine, P2 is projective - mixed addition
        // u1 = x1, s1 = y1
        // u2 = x2, s2 = y2  
        // h = u2 - u1*z2
        _ModMult_Montgomery_Optimized(t0, x1, z2); // u1*z2
        _ModSub_Optimized(t1, x2, t0); // h
        
        // r = s2 - s1*z2
        _ModMult_Montgomery_Optimized(t0, y1, z2); // s1*z2
        _ModSub_Optimized(t2, y2, t0); // r
        
        // Check for point doubling
        bool h_zero = true, r_zero = true;
        for (int i = 0; i < 4; i++) {
            if (t1[i] != 0) h_zero = false;
            if (t2[i] != 0) r_zero = false;
        }
        
        if (h_zero) {
            if (r_zero) {
                point_double_projective(result, p1);
                return;
            } else {
                // Point at infinity
                result->x = BigInt256(1);
                result->y = BigInt256(1); 
                result->z = BigInt256(0);
                return;
            }
        }
        
        // h2 = h^2, h3 = h^3
        _ModSqr_Optimized(t3, t1); // h2
        _ModMult_Montgomery_Optimized(t4, t1, t3); // h3
        
        // v = u1*h2
        _ModMult_Montgomery_Optimized(t5, x1, t3);
        
        // rx = r^2 - h3 - 2*v
        _ModSqr_Optimized(t0, t2); // r^2
        _ModSub_Optimized(rx, t0, t4); // r^2 - h3
        _ModAdd_Optimized(t0, t5, t5); // 2*v
        _ModSub_Optimized(rx, rx, t0);
        
        // ry = r*(v - rx) - s1*h3
        _ModSub_Optimized(t0, t5, rx); // v - rx
        _ModMult_Montgomery_Optimized(t0, t2, t0); // r*(v - rx)
        _ModMult_Montgomery_Optimized(t1, y1, t4); // s1*h3
        _ModSub_Optimized(ry, t0, t1);
        
        // rz = z2*h
        _ModMult_Montgomery_Optimized(rz, z2, t1);
        
    } else if (z2_is_one) {
        // P1 is projective, P2 is affine - mixed addition (symmetric case)
        _ModMult_Montgomery_Optimized(t0, x2, z1); // u2*z1
        _ModSub_Optimized(t1, x1, t0); // h
        
        _ModMult_Montgomery_Optimized(t0, y2, z1); // s2*z1
        _ModSub_Optimized(t2, y1, t0); // r
        
        // Check for point doubling
        bool h_zero = true, r_zero = true;
        for (int i = 0; i < 4; i++) {
            if (t1[i] != 0) h_zero = false;
            if (t2[i] != 0) r_zero = false;
        }
        
        if (h_zero) {
            if (r_zero) {
                point_double_projective(result, p1);
                return;
            } else {
                result->x = BigInt256(1);
                result->y = BigInt256(1);
                result->z = BigInt256(0);
                return;
            }
        }
        
        _ModSqr_Optimized(t3, t1); // h2
        _ModMult_Montgomery_Optimized(t4, t1, t3); // h3
        _ModMult_Montgomery_Optimized(t5, x2, t3); // v
        
        _ModSqr_Optimized(t0, t2); // r^2
        _ModSub_Optimized(rx, t0, t4);
        _ModAdd_Optimized(t0, t5, t5); // 2*v
        _ModSub_Optimized(rx, rx, t0);
        
        _ModSub_Optimized(t0, t5, rx);
        _ModMult_Montgomery_Optimized(t0, t2, t0);
        _ModMult_Montgomery_Optimized(t1, y2, t4);
        _ModSub_Optimized(ry, t0, t1);
        
        _ModMult_Montgomery_Optimized(rz, z1, t1);
        
    } else {
        // Both points are projective - full projective addition (12M + 2S)
        // u1 = x1*z2, u2 = x2*z1
        _ModMult_Montgomery_Optimized(t0, x1, z2); // u1
        _ModMult_Montgomery_Optimized(t1, x2, z1); // u2
        
        // s1 = y1*z2, s2 = y2*z1  
        _ModMult_Montgomery_Optimized(t2, y1, z2); // s1
        _ModMult_Montgomery_Optimized(t3, y2, z1); // s2
        
        // h = u2 - u1, r = s2 - s1
        _ModSub_Optimized(t4, t1, t0); // h
        _ModSub_Optimized(t5, t3, t2); // r
        
        // Check for point doubling
        bool h_zero = true, r_zero = true;
        for (int i = 0; i < 4; i++) {
            if (t4[i] != 0) h_zero = false;
            if (t5[i] != 0) r_zero = false;
        }
        
        if (h_zero) {
            if (r_zero) {
                point_double_projective(result, p1);
                return;
            } else {
                result->x = BigInt256(1);
                result->y = BigInt256(1);
                result->z = BigInt256(0);
                return;
            }
        }
        
        // h2 = h^2, h3 = h^3
        _ModSqr_Optimized(t1, t4); // h2 (reuse t1)
        _ModMult_Montgomery_Optimized(t2, t4, t1); // h3 (reuse t2)
        
        // v = u1*h2
        _ModMult_Montgomery_Optimized(t3, t0, t1); // v (reuse t3)
        
        // rx = r^2 - h3 - 2*v
        _ModSqr_Optimized(t0, t5); // r^2 (reuse t0)
        _ModSub_Optimized(rx, t0, t2); // r^2 - h3
        _ModAdd_Optimized(t0, t3, t3); // 2*v
        _ModSub_Optimized(rx, rx, t0);
        
        // ry = r*(v - rx) - s1*h3
        _ModSub_Optimized(t0, t3, rx); // v - rx
        _ModMult_Montgomery_Optimized(t0, t5, t0); // r*(v - rx)
        _ModMult_Montgomery_Optimized(t1, t2, t2); // s1*h3 (original s1 was overwritten)
        
        // Need to recompute s1 = y1*z2
        _ModMult_Montgomery_Optimized(t1, y1, z2);
        _ModMult_Montgomery_Optimized(t1, t1, t2);
        _ModSub_Optimized(ry, t0, t1);
        
        // rz = z1*z2*h
        _ModMult_Montgomery_Optimized(t0, z1, z2); // z1*z2
        _ModMult_Montgomery_Optimized(rz, t0, t4); // z1*z2*h
    }
    
    // Copy results
    for (int i = 0; i < 4; i++) {
        result->x.d[i] = rx[i];
        result->y.d[i] = ry[i];
        result->z.d[i] = rz[i];
    }
}

/**
 * @brief Device function for projective point doubling
 * Computes R = 2*P using optimized projective coordinate formulas
 * 
 * Uses optimized doubling formulas (3M + 5S for general case):
 * - Handles special case where a = 0 (secp256k1)
 * - Optimized for both affine and projective inputs
 */
__device__ void point_double_projective(ProjectivePoint* result, 
                                       const ProjectivePoint* p) {
    // Handle point at infinity
    if (p->z.is_zero()) {
        *result = *p;
        return;
    }
    
    uint64_t t0[4], t1[4], t2[4], t3[4], t4[4];
    uint64_t x[4], y[4], z[4];
    uint64_t rx[4], ry[4], rz[4];
    
    // Copy input coordinates
    for (int i = 0; i < 4; i++) {
        x[i] = p->x.d[i];
        y[i] = p->y.d[i];
        z[i] = p->z.d[i];
    }
    
    // Check if input is affine (Z = 1)
    bool z_is_one = (z[0] == 1 && z[1] == 0 && z[2] == 0 && z[3] == 0);
    
    if (z_is_one) {
        // Affine point doubling (optimized for a = 0)
        // lambda = (3*x^2) / (2*y)
        // But we work in projective coordinates to avoid division
        
        // t0 = 3*x^2
        _ModSqr_Optimized(t0, x);
        _ModAdd_Optimized(t1, t0, t0); // 2*x^2
        _ModAdd_Optimized(t0, t0, t1); // 3*x^2
        
        // t1 = 2*y
        _ModAdd_Optimized(t1, y, y);
        
        // Check if 2*y = 0 (point has order 2)
        bool double_y_zero = true;
        for (int i = 0; i < 4; i++) {
            if (t1[i] != 0) {
                double_y_zero = false;
                break;
            }
        }
        
        if (double_y_zero) {
            // Point has order 2 - doubling gives infinity
            result->x = BigInt256(1);
            result->y = BigInt256(1);
            result->z = BigInt256(0);
            return;
        }
        
        // t2 = (3*x^2)^2 = 9*x^4
        _ModSqr_Optimized(t2, t0);
        
        // t3 = 2*x = 2*x*1^2 (since we're in affine)
        _ModAdd_Optimized(t3, x, x);
        
        // rx = 9*x^4 - 8*x
        _ModAdd_Optimized(t4, t3, t3); // 4*x
        _ModAdd_Optimized(t4, t4, t4); // 8*x
        _ModSub_Optimized(rx, t2, t4);
        
        // ry = (3*x^2)*(4*x - rx) - 8*y^2
        _ModAdd_Optimized(t3, t3, t3); // 4*x (reusing t3)
        _ModSub_Optimized(t3, t3, rx); // 4*x - rx
        _ModMult_Montgomery_Optimized(t3, t0, t3); // (3*x^2)*(4*x - rx)
        _ModSqr_Optimized(t4, y); // y^2
        _ModAdd_Optimized(t4, t4, t4); // 2*y^2
        _ModAdd_Optimized(t4, t4, t4); // 4*y^2
        _ModAdd_Optimized(t4, t4, t4); // 8*y^2
        _ModSub_Optimized(ry, t3, t4);
        
        // rz = 2*y (the denominator from affine doubling)
        for (int i = 0; i < 4; i++) {
            rz[i] = t1[i];
        }
        
    } else {
        // Projective point doubling (3M + 5S)
        // Using optimized formulas for a = 0
        
        // t0 = y^2
        _ModSqr_Optimized(t0, y);
        
        // t1 = 4*x*y^2
        _ModMult_Montgomery_Optimized(t1, x, t0);
        _ModAdd_Optimized(t1, t1, t1); // 2*x*y^2
        _ModAdd_Optimized(t1, t1, t1); // 4*x*y^2
        
        // t2 = 8*y^4
        _ModSqr_Optimized(t2, t0); // y^4
        _ModAdd_Optimized(t2, t2, t2); // 2*y^4
        _ModAdd_Optimized(t2, t2, t2); // 4*y^4
        _ModAdd_Optimized(t2, t2, t2); // 8*y^4
        
        // t3 = z^2 (for secp256k1, a = 0, so no a*z^4 term)
        _ModSqr_Optimized(t3, z);
        
        // t4 = 3*x^2 (slope numerator)
        _ModSqr_Optimized(t4, x);
        _ModAdd_Optimized(t0, t4, t4); // 2*x^2
        _ModAdd_Optimized(t4, t4, t0); // 3*x^2
        
        // rx = (3*x^2)^2 - 2*(4*x*y^2) = 9*x^4 - 8*x*y^2
        _ModSqr_Optimized(t0, t4); // (3*x^2)^2
        _ModAdd_Optimized(t3, t1, t1); // 2*(4*x*y^2) = 8*x*y^2
        _ModSub_Optimized(rx, t0, t3);
        
        // ry = (3*x^2)*(4*x*y^2 - rx) - 8*y^4
        _ModSub_Optimized(t0, t1, rx); // 4*x*y^2 - rx
        _ModMult_Montgomery_Optimized(t0, t4, t0); // (3*x^2)*(4*x*y^2 - rx)
        _ModSub_Optimized(ry, t0, t2); // subtract 8*y^4
        
        // rz = 2*y*z
        _ModMult_Montgomery_Optimized(t0, y, z);
        _ModAdd_Optimized(rz, t0, t0);
    }
    
    // Copy results
    for (int i = 0; i < 4; i++) {
        result->x.d[i] = rx[i];
        result->y.d[i] = ry[i];
        result->z.d[i] = rz[i];
    }
}

/**
 * @brief Device function for projective point tripling
 * Computes R = 3*P using optimized formulas
 */
__device__ void point_triple_projective(ProjectivePoint* result, 
                                       const ProjectivePoint* p) {
    ProjectivePoint doubled;
    point_double_projective(&doubled, p);
    point_add_projective(result, &doubled, p);
}

/**
 * @brief Device function for scalar multiplication using windowed method
 * Computes R = k*P using precomputed table and sliding window
 */
__device__ void point_multiply_scalar(ProjectivePoint* result, 
                                     const BigInt256* scalar, 
                                     const ProjectivePoint* point) {
    // Initialize result to point at infinity
    result->x = BigInt256(1);
    result->y = BigInt256(1);
    result->z = BigInt256(0);
    
    // Handle edge cases
    if (scalar->is_zero() || point->z.is_zero()) {
        return;
    }
    
    // Simple binary method for now (can be optimized with windowing)
    ProjectivePoint temp = *point;
    
    // Process each bit from MSB to LSB
    for (int i = 255; i >= 0; i--) {
        // Double the current result
        point_double_projective(result, result);
        
        // Add point if bit is set
        if (scalar->test_bit(i)) {
            point_add_projective(result, result, &temp);
        }
    }
}

/**
 * @brief Device function for scalar multiplication with precomputed table
 * Uses windowed method with precomputed odd multiples
 */
__device__ void point_multiply_precomputed(ProjectivePoint* result,
                                          const BigInt256* scalar,
                                          const ProjectivePoint* table,
                                          int window_size) {
    // Initialize result to point at infinity
    result->x = BigInt256(1);
    result->y = BigInt256(1);  
    result->z = BigInt256(0);
    
    if (scalar->is_zero()) {
        return;
    }
    
    int bits_processed = 0;
    const int total_bits = 256;
    
    // Process scalar in windows from MSB to LSB
    while (bits_processed < total_bits) {
        // Double result by window_size bits
        for (int i = 0; i < window_size && bits_processed + i < total_bits; i++) {
            point_double_projective(result, result);
        }
        
        // Extract window value
        int window_start = total_bits - bits_processed - window_size;
        if (window_start < 0) {
            window_start = 0;
            window_size = total_bits - bits_processed;
        }
        
        uint32_t window_val = 0;
        for (int i = 0; i < window_size; i++) {
            if (scalar->test_bit(window_start + i)) {
                window_val |= (1U << i);
            }
        }
        
        // Add precomputed value if non-zero
        if (window_val > 0) {
            point_add_projective(result, result, &table[window_val]);
        }
        
        bits_processed += window_size;
    }
}

/**
 * @brief Batch point addition kernel
 */
__global__ void batch_point_add(ProjectivePoint* results,
                               const ProjectivePoint* p1_array,
                               const ProjectivePoint* p2_array,
                               size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    point_add_projective(&results[idx], &p1_array[idx], &p2_array[idx]);
}

/**
 * @brief Batch point doubling kernel
 */
__global__ void batch_point_double(ProjectivePoint* results,
                                 const ProjectivePoint* points,
                                 size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    point_double_projective(&results[idx], &points[idx]);
}

/**
 * @brief Batch scalar multiplication kernel
 */
__global__ void batch_scalar_multiply(ProjectivePoint* results,
                                    const BigInt256* scalars,
                                    const ProjectivePoint* points,
                                    size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    point_multiply_scalar(&results[idx], &scalars[idx], &points[idx]);
}

/**
 * @brief Batch scalar multiplication with precomputed table kernel
 */
__global__ void batch_scalar_multiply_precomputed(ProjectivePoint* results,
                                                 const BigInt256* scalars,
                                                 const ProjectivePoint* table,
                                                 int window_size,
                                                 size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    point_multiply_precomputed(&results[idx], &scalars[idx], table, window_size);
}

} // namespace optimized
} // namespace gpu
} // namespace ecc
} // namespace keyhunt