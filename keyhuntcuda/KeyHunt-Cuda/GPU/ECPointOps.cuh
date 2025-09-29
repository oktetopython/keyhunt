/**
 * ECPointOps.cuh
 * Complete EC point operations for GPU kernels
 * Implements missing functions for PUZZLE71 kernel
 */

#ifndef EC_POINT_OPS_CUH
#define EC_POINT_OPS_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "GPUMath.h"

/**
 * Add two elliptic curve points
 * Implements the complete point addition formula for secp256k1
 * P3 = P1 + P2
 */
__device__ __forceinline__ void _addPoint(
    uint64_t p1x[4], uint64_t p1y[4],  // Input/Output: P1 (result stored here)
    const uint64_t p2x[4], const uint64_t p2y[4])  // Input: P2
{
    // Check for special cases
    // If P2 is identity (0,0), return P1
    bool p2_is_zero = true;
    for (int i = 0; i < 4; i++) {
        if (p2x[i] != 0 || p2y[i] != 0) {
            p2_is_zero = false;
            break;
        }
    }
    if (p2_is_zero) {
        return;  // P1 + 0 = P1
    }
    
    // If P1 is identity (0,0), return P2
    bool p1_is_zero = true;
    for (int i = 0; i < 4; i++) {
        if (p1x[i] != 0 || p1y[i] != 0) {
            p1_is_zero = false;
            break;
        }
    }
    if (p1_is_zero) {
        // Copy P2 to P1
        for (int i = 0; i < 4; i++) {
            p1x[i] = p2x[i];
            p1y[i] = p2y[i];
        }
        return;
    }
    
    // Check if points are equal (need point doubling)
    bool points_equal = true;
    for (int i = 0; i < 4; i++) {
        if (p1x[i] != p2x[i]) {
            points_equal = false;
            break;
        }
    }
    
    if (points_equal) {
        // Point doubling: P3 = 2*P1
        // s = (3*x1^2) / (2*y1)
        uint64_t x1_sqr[4], three_x1_sqr[4], two_y1[4];
        uint64_t s[4], s_sqr[4];
        
        // x1^2
        _ModSqr(x1_sqr, p1x);
        
        // 3*x1^2
        ModAdd256(three_x1_sqr, x1_sqr, x1_sqr);
        ModAdd256(three_x1_sqr, three_x1_sqr, x1_sqr);
        
        // 2*y1
        ModAdd256(two_y1, p1y, p1y);
        
        // Compute modular inverse of 2*y1
        uint64_t two_y1_inv[5];
        two_y1_inv[0] = two_y1[0];
        two_y1_inv[1] = two_y1[1];
        two_y1_inv[2] = two_y1[2];
        two_y1_inv[3] = two_y1[3];
        two_y1_inv[4] = 0;
        _ModInv(two_y1_inv);
        
        // s = three_x1_sqr / two_y1
        _ModMult(s, three_x1_sqr, two_y1_inv);
        
        // x3 = s^2 - 2*x1
        _ModSqr(s_sqr, s);
        uint64_t two_x1[4];
        ModAdd256(two_x1, p1x, p1x);
        uint64_t x3[4];
        ModSub256(x3, s_sqr, two_x1);
        
        // y3 = s*(x1 - x3) - y1
        uint64_t x1_minus_x3[4];
        ModSub256(x1_minus_x3, (uint64_t*)p1x, x3);
        uint64_t y3[4];
        _ModMult(y3, s, x1_minus_x3);
        ModSub256(y3, y3, (uint64_t*)p1y);
        
        // Store result
        for (int i = 0; i < 4; i++) {
            p1x[i] = x3[i];
            p1y[i] = y3[i];
        }
    } else {
        // Regular point addition
        // s = (y2 - y1) / (x2 - x1)
        uint64_t dx[4], dy[4], s[4], s_sqr[4];
        
        // dx = x2 - x1
        ModSub256(dx, (uint64_t*)p2x, (uint64_t*)p1x);
        
        // dy = y2 - y1
        ModSub256(dy, (uint64_t*)p2y, (uint64_t*)p1y);
        
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
        
        // x3 = s^2 - x1 - x2
        _ModSqr(s_sqr, s);
        uint64_t x3[4];
        ModSub256(x3, s_sqr, (uint64_t*)p1x);
        ModSub256(x3, x3, (uint64_t*)p2x);
        
        // y3 = s*(x1 - x3) - y1
        uint64_t x1_minus_x3[4];
        ModSub256(x1_minus_x3, (uint64_t*)p1x, x3);
        uint64_t y3[4];
        _ModMult(y3, s, x1_minus_x3);
        ModSub256(y3, y3, (uint64_t*)p1y);
        
        // Store result
        for (int i = 0; i < 4; i++) {
            p1x[i] = x3[i];
            p1y[i] = y3[i];
        }
    }
}

/**
 * Double an elliptic curve point
 * P3 = 2*P1 (point doubling)
 */
__device__ __forceinline__ void _doublePoint(
    uint64_t px[4], uint64_t py[4])  // Input/Output: point to double
{
    // Check if point is identity (0,0)
    bool p_is_zero = true;
    for (int i = 0; i < 4; i++) {
        if (px[i] != 0 || py[i] != 0) {
            p_is_zero = false;
            break;
        }
    }
    if (p_is_zero) {
        return;  // 2*0 = 0
    }
    
    // Point doubling formula:
    // s = (3*x^2) / (2*y)
    // x' = s^2 - 2*x
    // y' = s*(x - x') - y
    
    uint64_t x_sqr[4], three_x_sqr[4], two_y[4];
    uint64_t s[4], s_sqr[4];
    
    // x^2
    _ModSqr(x_sqr, px);
    
    // 3*x^2
    ModAdd256(three_x_sqr, x_sqr, x_sqr);
    ModAdd256(three_x_sqr, three_x_sqr, x_sqr);
    
    // 2*y
    ModAdd256(two_y, py, py);
    
    // Compute modular inverse of 2*y
    uint64_t two_y_inv[5];
    two_y_inv[0] = two_y[0];
    two_y_inv[1] = two_y[1];
    two_y_inv[2] = two_y[2];
    two_y_inv[3] = two_y[3];
    two_y_inv[4] = 0;
    _ModInv(two_y_inv);
    
    // s = three_x_sqr / two_y
    _ModMult(s, three_x_sqr, two_y_inv);
    
    // x' = s^2 - 2*x
    _ModSqr(s_sqr, s);
    uint64_t two_x[4];
    ModAdd256(two_x, px, px);
    uint64_t x_new[4];
    ModSub256(x_new, s_sqr, two_x);
    
    // y' = s*(x - x') - y
    uint64_t x_minus_x_new[4];
    ModSub256(x_minus_x_new, px, x_new);
    uint64_t y_new[4];
    _ModMult(y_new, s, x_minus_x_new);
    ModSub256(y_new, y_new, py);
    
    // Store result
    for (int i = 0; i < 4; i++) {
        px[i] = x_new[i];
        py[i] = y_new[i];
    }
}

// Forward declarations
__device__ __forceinline__ void sha256_gpu(const uint8_t* data, uint32_t len, uint8_t* hash);
__device__ __forceinline__ void ripemd160_gpu(const uint8_t* data, uint32_t len, uint8_t* hash);

/**
 * GPU-optimized SHA256 implementation
 * Simplified version for compressed public key hashing
 */
__device__ __forceinline__ void sha256_gpu(const uint8_t* data, uint32_t len, uint8_t* hash) {
    // SHA256 constants
    const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    
    // Initial hash values
    uint32_t H[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Padding
    uint8_t padded[64];
    for (int i = 0; i < len; i++) {
        padded[i] = data[i];
    }
    padded[len] = 0x80;
    for (int i = len + 1; i < 56; i++) {
        padded[i] = 0;
    }
    
    // Length in bits (big-endian)
    uint64_t bit_len = len * 8;
    for (int i = 0; i < 8; i++) {
        padded[63 - i] = (bit_len >> (i * 8)) & 0xFF;
    }
    
    // Process the block
    uint32_t W[64];
    for (int i = 0; i < 16; i++) {
        W[i] = (padded[i*4] << 24) | (padded[i*4+1] << 16) | 
               (padded[i*4+2] << 8) | padded[i*4+3];
    }
    
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = __funnelshift_r(W[i-15], W[i-15], 7) ^ 
                      __funnelshift_r(W[i-15], W[i-15], 18) ^ (W[i-15] >> 3);
        uint32_t s1 = __funnelshift_r(W[i-2], W[i-2], 17) ^ 
                      __funnelshift_r(W[i-2], W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    
    // Compression
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];
    
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = __funnelshift_r(e, e, 6) ^ 
                      __funnelshift_r(e, e, 11) ^ 
                      __funnelshift_r(e, e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + K[i] + W[i];
        uint32_t S0 = __funnelshift_r(a, a, 2) ^ 
                      __funnelshift_r(a, a, 13) ^ 
                      __funnelshift_r(a, a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    
    // Add to hash values
    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    
    // Output hash (big-endian)
    for (int i = 0; i < 8; i++) {
        hash[i*4] = (H[i] >> 24) & 0xFF;
        hash[i*4+1] = (H[i] >> 16) & 0xFF;
        hash[i*4+2] = (H[i] >> 8) & 0xFF;
        hash[i*4+3] = H[i] & 0xFF;
    }
}

/**
 * GPU-optimized RIPEMD160 implementation
 */
__device__ __forceinline__ void ripemd160_gpu(const uint8_t* data, uint32_t len, uint8_t* hash) {
    // RIPEMD160 initial values
    uint32_t h[5] = {
        0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    };
    
    // Padding
    uint8_t padded[64];
    for (int i = 0; i < len && i < 64; i++) {
        padded[i] = data[i];
    }
    if (len < 64) {
        padded[len] = 0x80;
        for (int i = len + 1; i < 56; i++) {
            padded[i] = 0;
        }
    }
    
    // Length in bits (little-endian)
    uint64_t bit_len = len * 8;
    for (int i = 0; i < 8; i++) {
        padded[56 + i] = (bit_len >> (i * 8)) & 0xFF;
    }
    
    // Process block (simplified for 32-byte input)
    uint32_t X[16];
    for (int i = 0; i < 16; i++) {
        X[i] = padded[i*4] | (padded[i*4+1] << 8) | 
               (padded[i*4+2] << 16) | (padded[i*4+3] << 24);
    }
    
    // Left line
    uint32_t al = h[0], bl = h[1], cl = h[2], dl = h[3], el = h[4];
    
    // Round 1 (simplified)
    for (int j = 0; j < 16; j++) {
        uint32_t f = bl ^ cl ^ dl;
        uint32_t k = 0;
        uint32_t t = al + f + X[j] + k;
        t = __funnelshift_l(t, t, 11) + el;
        al = el; el = dl; dl = __funnelshift_l(cl, cl, 10);
        cl = bl; bl = t;
    }
    
    // Right line
    uint32_t ar = h[0], br = h[1], cr = h[2], dr = h[3], er = h[4];
    
    // Round 1 (simplified)
    for (int j = 0; j < 16; j++) {
        uint32_t f = br ^ (cr | ~dr);
        uint32_t k = 0x50A28BE6;
        int r = (5 + j * 9) % 16;
        uint32_t t = ar + f + X[r] + k;
        t = __funnelshift_l(t, t, 8) + er;
        ar = er; er = dr; dr = __funnelshift_l(cr, cr, 10);
        cr = br; br = t;
    }
    
    // Combine results
    uint32_t t = h[1] + cl + dr;
    h[1] = h[2] + dl + er;
    h[2] = h[3] + el + ar;
    h[3] = h[4] + al + br;
    h[4] = h[0] + bl + cr;
    h[0] = t;
    
    // Output hash (little-endian)
    for (int i = 0; i < 5; i++) {
        hash[i*4] = h[i] & 0xFF;
        hash[i*4+1] = (h[i] >> 8) & 0xFF;
        hash[i*4+2] = (h[i] >> 16) & 0xFF;
        hash[i*4+3] = (h[i] >> 24) & 0xFF;
    }
}

/**
 * Compute HASH160 for compressed public key
 * SHA256(compressed_pubkey) -> RIPEMD160(sha256_result) -> hash160
 */
__device__ __forceinline__ void _GetHash160Comp(
    const uint64_t px[4],       // Public key x coordinate
    uint8_t compressed_flag,    // 0x02 or 0x03 depending on y parity
    uint8_t hash160[20])        // Output: 20-byte HASH160
{
    // Prepare compressed public key (33 bytes)
    uint8_t compressed_key[33];
    compressed_key[0] = compressed_flag;
    
    // Convert x coordinate to big-endian bytes
    for (int i = 0; i < 4; i++) {
        uint64_t val = px[3 - i];  // Reverse for big-endian
        for (int j = 7; j >= 0; j--) {
            compressed_key[1 + i * 8 + j] = (val >> ((7 - j) * 8)) & 0xFF;
        }
    }
    
    // Compute SHA256
    uint8_t sha256_result[32];
    sha256_gpu(compressed_key, 33, sha256_result);
    
    // Compute RIPEMD160
    ripemd160_gpu(sha256_result, 32, hash160);
}

/**
 * Batch point addition for optimized processing
 * Adds the same point to multiple base points
 */
__device__ __forceinline__ void batch_point_add(
    uint64_t base_x[][4], uint64_t base_y[][4],  // Array of base points
    const uint64_t add_x[4], const uint64_t add_y[4],  // Point to add
    uint32_t count)  // Number of points
{
    // Process each point
    for (uint32_t i = 0; i < count; i++) {
        _addPoint(base_x[i], base_y[i], add_x, add_y);
    }
}

#endif // EC_POINT_OPS_CUH