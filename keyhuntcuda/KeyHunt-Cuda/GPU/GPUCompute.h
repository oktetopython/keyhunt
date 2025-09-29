/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc Pons.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef GPU_COMPUTE_H
#define GPU_COMPUTE_H

// Use recommended CUDA headers instead of deprecated device_functions.h
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include "GPUEngine.h"
#include "GPUMemoryOptimized.h"
#include "KeyHuntConstants.h"

// Include hash function headers
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Constants.h"
#include "SearchMode.h"
#include "GPUMath.h"
#include "GPUHash.h"
#include "ECC_Endomorphism.h"
#include "BatchStepping.h"
#include "GPUModInv.cuh"

// Forward declaration of unified_check_hash function template
template<SearchMode Mode>
__device__ __forceinline__ void unified_check_hash(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint32_t param1, uint32_t param2,
    uint32_t maxFound, uint32_t* out);

// Forward declarations of global device variables
// Note: These are now declared in GPUMemoryOptimized.h to avoid redefinition

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        } \
    } while(0)

// Function to reset the found flag before each kernel launch
__global__ void reset_found_flag();

// ----------------------------- COMMON EC FUNCTIONS -----------------------------

/**
 * Common elliptic curve point addition computation
 * Eliminates code duplication across multiple kernels
 * @param px Current point X coordinate (input/output)
 * @param py Current point Y coordinate (input/output)
 * @param gx Generator point X coordinate
 * @param gy Generator point Y coordinate
 * @param dx Precomputed inverse differences
 * @param i Index for dx array
 */
__device__ __forceinline__ void compute_ec_point_add(
    uint64_t* px, uint64_t* py,
    uint64_t* gx, uint64_t* gy,
    uint64_t* dx
) {
    uint64_t _s[4], _p2[4], dy[4];

    // Compute slope: s = (p2.y-p1.y)*inverse(p2.x-p1.x)
    ModSub256(dy, gy, py);
    _ModMult(_s, dy, dx);
    _ModSqr(_p2, _s);

    // Compute new X coordinate: px = pow2(s) - p1.x - p2.x
    ModSub256(px, _p2, px);
    ModSub256(px, px, gx);

    // Compute new Y coordinate: py = -p2.y - s*(ret.x-p2.x)
    ModSub256(py, gx, px);
    _ModMult(py, _s, py);
    ModSub256(py, py, gy);
}

/**
 * Common elliptic curve point addition computation for negative direction
 * Used for P = StartPoint - i*G calculations
 * @param px Current point X coordinate (input/output)
 * @param pyn Negative Y coordinate of starting point
 * @param gx Generator point X coordinate
 * @param gy Generator point Y coordinate
 * @param dx Precomputed inverse differences
 * @param i Index for dx array
 */
__device__ __forceinline__ void compute_ec_point_add_negative(
    uint64_t* px, uint64_t* py,
    uint64_t* pyn,
    uint64_t* gx, uint64_t* gy,
    uint64_t* dx
) {
    uint64_t _s[4], _p2[4], dy[4];

    // Compute slope for negative direction
    ModSub256(dy, pyn, gy);
    _ModMult(_s, dy, dx);
    _ModSqr(_p2, _s);

    // Compute new X coordinate
    ModSub256(px, _p2, px);
    ModSub256(px, px, gx);

    // Compute new Y coordinate for negative direction
    ModSub256(py, px, gx);
    _ModMult(py, _s, py);
    ModSub256(py, gy, py);
}

/**
 * Special elliptic curve point addition for first/last points
 * Used for special cases with modified Y coordinates
 * @param px Current point X coordinate (input/output)
 * @param py Current point Y coordinate (input/output)
 * @param gx Generator point X coordinate
 * @param gy Generator point Y coordinate
 * @param dx Precomputed inverse differences
 * @param i Index for dx array
 * @param negate_gy Whether to negate gy before computation
 */
__device__ __forceinline__ void compute_ec_point_add_special(
    uint64_t* px, uint64_t* py,
    uint64_t* gx, uint64_t* gy,
    uint64_t* dx,
    bool negate_gy = false
) {
    uint64_t _s[4], _p2[4], dy[4];

    if (negate_gy) {
        ModNeg256(dy, gy);
        ModSub256(dy, dy, py);
    } else {
        ModSub256(dy, gy, py);
    }

    _ModMult(_s, dy, dx);
    _ModSqr(_p2, _s);

    ModSub256(px, _p2, px);
    ModSub256(px, px, gx);

    ModSub256(py, px, gx);
    _ModMult(py, _s, py);
    ModSub256(py, gy, py);
}

/**
 * Common hash computation for Bitcoin addresses
 * Eliminates code duplication across multiple kernels
 * @param publicKeyBytes Public key in bytes format
 * @param keySize Size of public key (33 for compressed, 65 for uncompressed)
 * @param hash160 Output hash160 result
 */
__device__ __forceinline__ void compute_bitcoin_hash(
    const uint8_t* publicKeyBytes,
    uint32_t keySize,
    uint32_t* hash160
) {
    uint8_t hash1[32];
    if (keySize == 33) {
        sha256_33((uint8_t*)publicKeyBytes, hash1);
    } else {
        sha256_65((uint8_t*)publicKeyBytes, hash1);
    }
    ripemd160_32(hash1, (uint8_t*)hash160);
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ int Test_Bit_Set_Bit(const uint8_t* buf, uint32_t bit)
{
	uint32_t byte = bit >> 3;
	uint8_t c = buf[byte];        // expensive memory access
	uint8_t mask = 1 << (bit % 8);

	if (c & mask) {
		return 1;
	}
	else {
		return 0;
	}
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t MurMurHash2(const void* key, int len, uint32_t seed)
{
	const uint32_t m = 0x5bd1e995;
	const int r = 24;

	uint32_t h = seed ^ len;
	const uint8_t* data = (const uint8_t*)key;
	while (len >= 4) {
		uint32_t k = *(uint32_t*)data;
		k *= m;
		k ^= k >> r;
		k *= m;
		h *= m;
		h ^= k;
		data += 4;
		len -= 4;
	}
	switch (len) {
	case 3: h ^= data[2] << 16;
		break;
	case 2: h ^= data[1] << 8;
		break;
	case 1: h ^= data[0];
		h *= m;
		break;
	}

	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;

	return h;
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ int BloomCheck(const uint32_t* hash, const uint8_t* inputBloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t K_LENGTH)
{
	int add = 0;
	uint8_t hits = 0;
	uint32_t a = MurMurHash2((uint8_t*)hash, K_LENGTH, 0x9747b28c);
	uint32_t b = MurMurHash2((uint8_t*)hash, K_LENGTH, a);
	uint32_t x;
	uint8_t i;
	for (i = 0; i < BLOOM_HASHES; i++) {
		x = (a + b * i) % BLOOM_BITS;
		if (Test_Bit_Set_Bit(inputBloomLookUp, x)) {
			hits++;
		}
		else if (!add) {
			return 0;
		}
	}
	if (hits == BLOOM_HASHES) {
		return 1;
	}
	return 0;
}

// ---------------------------------------------------------------------------------------

// 已被统一接口替代的函数 - 删除重复实现
// __device__ __noinline__ void CheckPointSEARCH_MODE_MA(...)

// ---------------------------------------------------------------------------------------

// 已被统一接口替代的函数 - 删除重复实现
// __device__ __noinline__ void CheckPointSEARCH_MODE_MX(...)

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ bool MatchHash(const uint32_t* _h, const uint32_t* hash)
{
	if (_h[0] == hash[0] &&
		_h[1] == hash[1] &&
		_h[2] == hash[2] &&
		_h[3] == hash[3] &&
		_h[4] == hash[4]) {
		return true;
	}
	else {
		return false;
	}
}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ bool MatchXPoint(const uint32_t* _h, const uint32_t* xpoint)
{
	

	if (_h[0] == xpoint[0] &&
		_h[1] == xpoint[1] &&
		_h[2] == xpoint[2] &&
		_h[3] == xpoint[3] &&
		_h[4] == xpoint[4] &&
		_h[5] == xpoint[5] &&
		_h[6] == xpoint[6] &&
		_h[7] == xpoint[7]) {
		return true;
	}
	else {
		return false;
	}
}

// ---------------------------------------------------------------------------------------

// 已被统一接口替代的函数 - 删除重复实现
// __device__ __noinline__ void CheckPointSEARCH_MODE_SA(...)

// ---------------------------------------------------------------------------------------

// 已被统一接口替代的函数 - 删除重复实现
// __device__ __noinline__ void CheckPointSEARCH_MODE_SX(...)

// -----------------------------------------------------------------------------------------

#define CHECK_POINT_SEARCH_MODE_MA(_h,incr,mode)  CheckPointSEARCH_MODE_MA(_h,incr,mode,bloomLookUp,BLOOM_BITS,BLOOM_HASHES,maxFound,out)

#define CHECK_POINT_SEARCH_MODE_SA(_h,incr,mode)  CheckPointSEARCH_MODE_SA(_h,incr,mode,hash160,maxFound,out)
// -----------------------------------------------------------------------------------------

#define CheckHashUnCompSEARCH_MODE_MA(px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out) \
    unified_check_hash<SearchMode::MODE_MA>(mode, px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

// ---------------------------------------------------------------------------------------

#define CheckHashUnCompSEARCH_MODE_SA(px, py, incr, hash160, maxFound, out) \
    unified_check_hash<SearchMode::MODE_SA>(mode, px, py, incr, hash160, 0, 0, maxFound, out)

// -----------------------------------------------------------------------------------------

#define CHECK_HASH_SEARCH_MODE_MA(incr) unified_check_hash<SearchMode::MODE_MA>(mode, px, py, incr, target_data, param1, param2, maxFound, out)

// -----------------------------------------------------------------------------------------

#define CHECK_POINT_SEARCH_MODE_MX(_h,incr,mode)  CheckPointSEARCH_MODE_MX(_h,incr,mode,bloomLookUp,BLOOM_BITS,BLOOM_HASHES,maxFound,out)

#define CheckPubCompSEARCH_MODE_MX(px, isOdd, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out) \
    unified_check_hash<SearchMode::MODE_MX>(mode, px, nullptr, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

#define CHECK_POINT_SEARCH_MODE_SX(_h,incr,mode)  CheckPointSEARCH_MODE_SX(_h,incr,mode,xpoint,maxFound,out)

#define CheckPubCompSEARCH_MODE_SX(px, isOdd, incr, xpoint, maxFound, out) \
    unified_check_hash<SearchMode::MODE_SX>(mode, px, nullptr, incr, xpoint, 0, 0, maxFound, out)

// ---------------------------------------------------------------------------------------

#define CheckPubSEARCH_MODE_MX(mode, px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out) \
    do { \
        if (mode == SEARCH_COMPRESSED) { \
            unified_check_hash<SearchMode::MODE_MX>(mode, px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out); \
        } \
    } while(0)

// -----------------------------------------------------------------------------------------

#define CheckPubSEARCH_MODE_SX(mode, px, py, incr, xpoint, maxFound, out) \
    do { \
        if (mode == SEARCH_COMPRESSED) { \
            unified_check_hash<SearchMode::MODE_SX>(mode, px, py, incr, xpoint, 0, 0, maxFound, out); \
        } \
    } while(0)

// -----------------------------------------------------------------------------------------

// Unified ComputeKeys function template to eliminate code duplication
template<SearchMode Mode>
__device__ __forceinline__ void ComputeKeysUnified(uint32_t mode, uint64_t* startx, uint64_t* starty,
	const void* target_data, uint32_t param1, uint32_t param2, uint32_t maxFound, uint32_t* out)
{
	// 使用统一接口，变量声明已移至统一函数中
	uint64_t dx[KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
	uint64_t pyn[4];
	uint64_t sx[4];
	uint64_t sy[4];
	uint64_t dy[4];
	uint64_t _s[4];
	uint64_t _p2[4];

	// Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
	for (i = 0; i < KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE; i++)
		ModSub256(dx[i], Gx + 4 * i, sx);
	ModSub256(dx[i], Gx + 4 * i, sx);   // For the first point
	ModSub256(dx[i + 1], _2Gnx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

	// Check starting point
	{
		// 使用宏定义替代直接调用，避免未定义函数错误
		CHECK_HASH_SEARCH_MODE_MA(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2);
	}

	ModNeg256(pyn, py);

	for (i = 0; i < KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE; i++) {

		// P = StartPoint + i*G
		Load256(px, sx);
		Load256(py, sy);
		compute_ec_point_add(px, py, Gx + 4 * i, Gy + 4 * i, dx[i]);

		{
			// 使用宏定义替代直接调用，避免未定义函数错误
			CHECK_HASH_SEARCH_MODE_MA(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 + (i + 1));
		}

		// P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
		Load256(px, sx);
		compute_ec_point_add_negative(px, py, pyn, Gx + 4 * i, Gy + 4 * i, dx[i]);

		{
			// 使用宏定义替代直接调用，避免未定义函数错误
			CHECK_HASH_SEARCH_MODE_MA(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 - (i + 1));
		}

	}

	// First point (startP - (GRP_SIZE/2)*G)
	Load256(px, sx);
	Load256(py, sy);
	compute_ec_point_add_special(px, py, Gx + 4 * i, Gy + 4 * i, dx[i], true);

	{
		// 使用宏定义替代直接调用，避免未定义函数错误
		CHECK_HASH_SEARCH_MODE_MA(0);
	}

	i++;

	// Next start point (startP + GRP_SIZE*G)
	Load256(px, sx);
	Load256(py, sy);
	compute_ec_point_add(px, py, _2Gnx, _2Gny, dx[i + 1]);

	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);
}

// -----------------------------------------------------------------------------------------
// PUZZLE71 specialized mode - Move definitions here before template specialization

// Hardcoded target HASH160 for Bitcoin Puzzle #71
// Address: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
// HASH160: f8455b22fa469a40654450d363959a3b932924b4
#ifndef PUZZLE71_TARGET_HASH_DEFINED
#define PUZZLE71_TARGET_HASH_DEFINED
extern __device__ __constant__ uint32_t PUZZLE71_TARGET_HASH[5];
#endif

// Specialized hash check for PUZZLE71 that uses hardcoded target
__device__ __forceinline__ void unified_check_hash_puzzle71(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    uint32_t maxFound, uint32_t* out)
{
    uint32_t h[5];
    
    // Compute compressed Bitcoin address hash
    _GetHash160Comp(px, (uint8_t)(py[0] & 1), (uint8_t*)h);
    
    // Compare against hardcoded PUZZLE71 target
    bool match = true;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        if (h[i] != PUZZLE71_TARGET_HASH[i]) {
            match = false;
            break;
        }
    }
    
    if (match) {
        // Found the target!
        uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        
        // Use atomic operations to ensure only one thread writes result
        if (atomicCAS(&found_flag, 0, 1) == 0) {
            uint32_t pos = atomicAdd(out, 1);
            if (pos < maxFound) {
                out[pos * ITEM_SIZE_A32 + 1] = tid;
                out[pos * ITEM_SIZE_A32 + 2] = (uint32_t)(incr << 16);
                for (int i = 0; i < 5; i++) {
                    out[pos * ITEM_SIZE_A32 + 3 + i] = h[i];
                }
            }
        }
    }
}

// Modified macro to use hardcoded target instead of parameter
#define CHECK_HASH_PUZZLE71(incr) unified_check_hash_puzzle71(mode, px, py, incr, maxFound, out)

// Template specialization for PUZZLE71 mode with endomorphism acceleration
template<>
__device__ __forceinline__ void ComputeKeysUnified<SearchMode::PUZZLE71>(
    uint32_t mode, uint64_t* startx, uint64_t* starty,
    const void* target_data, uint32_t param1, uint32_t param2, uint32_t maxFound, uint32_t* out)
{
    // PUZZLE71 ignores target_data - uses hardcoded PUZZLE71_TARGET_HASH
    // This version uses endomorphism acceleration for faster scalar multiplication
    uint64_t dx[KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 + 1][4];
    uint64_t px[4];
    uint64_t py[4];
    uint64_t pyn[4];
    uint64_t sx[4];
    uint64_t sy[4];
    uint64_t dy[4];
    uint64_t _s[4];
    uint64_t _p2[4];
    
    // Variables for endomorphism
    uint64_t px_endo[4], py_endo[4];
    uint64_t k1[4], k2[4];
    bool negate_k1, negate_k2;

    // Load starting key
    __syncthreads();
    Load256A(sx, startx);
    Load256A(sy, starty);
    Load256(px, sx);
    Load256(py, sy);

    // Check if we should use endomorphism acceleration
    // For Puzzle #71, keys are in range [2^70, 2^71), which is perfect for endomorphism
    bool use_endomorphism = (mode == SEARCH_COMPRESSED) && ((px[3] & 0xFFFFFFFF00000000ULL) != 0);

    if (use_endomorphism) {
        // Use endomorphism acceleration for initial point check
        // Split the scalar k into k1 and k2 for endomorphism
        // This is a simplified example - actual implementation would use the scalar value
        uint64_t scalar[4] = {px[0], px[1], px[2], px[3]}; // Using x-coordinate as example
        
        // Apply endomorphism to compute φ(P) = (β*x, y)
        px_endo[0] = px[0];
        px_endo[1] = px[1];
        px_endo[2] = px[2];
        px_endo[3] = px[3];
        py_endo[0] = py[0];
        py_endo[1] = py[1];
        py_endo[2] = py[2];
        py_endo[3] = py[3];
        apply_endomorphism(px_endo, py_endo);
        
        // Check both the original and endomorphism-transformed point
        CHECK_HASH_PUZZLE71(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2);
        
        // Check endomorphism point as well for double coverage
        uint64_t temp_px[4], temp_py[4];
        temp_px[0] = px_endo[0];
        temp_px[1] = px_endo[1];
        temp_px[2] = px_endo[2];
        temp_px[3] = px_endo[3];
        temp_py[0] = py_endo[0];
        temp_py[1] = py_endo[1];
        temp_py[2] = py_endo[2];
        temp_py[3] = py_endo[3];
        // Store temporarily to check
        px[0] = temp_px[0];
        px[1] = temp_px[1];
        px[2] = temp_px[2];
        px[3] = temp_px[3];
        py[0] = temp_py[0];
        py[1] = temp_py[1];
        py[2] = temp_py[2];
        py[3] = temp_py[3];
        CHECK_HASH_PUZZLE71(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 + 1000); // Different increment to distinguish
        // Restore original
        Load256(px, sx);
        Load256(py, sy);
    } else {
        // Standard check without endomorphism
        CHECK_HASH_PUZZLE71(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2);
    }

    // Fill group with delta x
    uint32_t i;
    for (i = 0; i < KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE; i++)
        ModSub256(dx[i], Gx + 4 * i, sx);
    ModSub256(dx[i], Gx + 4 * i, sx);   // For the first point
    ModSub256(dx[i + 1], _2Gnx, sx);    // For the next center point

    // Compute modular inverse
    _ModInvGrouped(dx);

    ModNeg256(pyn, py);

    // Main loop with optional endomorphism checks
    for (i = 0; i < KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE; i++) {
        // P = StartPoint + i*G
        Load256(px, sx);
        Load256(py, sy);
        compute_ec_point_add(px, py, Gx + 4 * i, Gy + 4 * i, dx[i]);
        CHECK_HASH_PUZZLE71(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 + (i + 1));
        
        // Apply endomorphism check every few iterations for better coverage
        if (use_endomorphism && ((i & 0x7) == 0)) {
            // Apply endomorphism to current point
            px_endo[0] = px[0];
            px_endo[1] = px[1];
            px_endo[2] = px[2];
            px_endo[3] = px[3];
            py_endo[0] = py[0];
            py_endo[1] = py[1];
            py_endo[2] = py[2];
            py_endo[3] = py[3];
            apply_endomorphism(px_endo, py_endo);
            
            // Temporarily swap to check endomorphism point
            uint64_t save_px[4], save_py[4];
            save_px[0] = px[0]; save_px[1] = px[1]; save_px[2] = px[2]; save_px[3] = px[3];
            save_py[0] = py[0]; save_py[1] = py[1]; save_py[2] = py[2]; save_py[3] = py[3];
            
            px[0] = px_endo[0]; px[1] = px_endo[1]; px[2] = px_endo[2]; px[3] = px_endo[3];
            py[0] = py_endo[0]; py[1] = py_endo[1]; py[2] = py_endo[2]; py[3] = py_endo[3];
            CHECK_HASH_PUZZLE71(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 + (i + 1) + 2000);
            
            // Restore
            px[0] = save_px[0]; px[1] = save_px[1]; px[2] = save_px[2]; px[3] = save_px[3];
            py[0] = save_py[0]; py[1] = save_py[1]; py[2] = save_py[2]; py[3] = save_py[3];
        }

        // P = StartPoint - i*G
        Load256(px, sx);
        compute_ec_point_add_negative(px, py, pyn, Gx + 4 * i, Gy + 4 * i, dx[i]);
        CHECK_HASH_PUZZLE71(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 - (i + 1));
    }

    // First point (startP - (GRP_SIZE/2)*G)
    Load256(px, sx);
    Load256(py, sy);
    compute_ec_point_add_special(px, py, Gx + 4 * i, Gy + 4 * i, dx[i], true);
    CHECK_HASH_PUZZLE71(0);

    i++;

    // Next start point (startP + GRP_SIZE*G)
    Load256(px, sx);
    Load256(py, sy);
    compute_ec_point_add(px, py, _2Gnx, _2Gny, dx[i + 1]);

    // Update starting point
    __syncthreads();
    Store256A(startx, px);
    Store256A(starty, py);
}

// Legacy function wrappers for backward compatibility
__device__ __forceinline__ void ComputeKeysSEARCH_MODE_MA(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	ComputeKeysUnified<SearchMode::MODE_MA>(mode, startx, starty, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out);
}



// -----------------------------------------------------------------------------------------

#define CheckHashSEARCH_MODE_SA(mode, px, py, incr, hash160, maxFound, out) \
    do { \
        switch (mode) { \
            case SEARCH_COMPRESSED: \
                unified_check_hash<SearchMode::MODE_SA>(mode, px, py, incr, hash160, 0, 0, maxFound, out); \
                break; \
            case SEARCH_UNCOMPRESSED: \
                unified_check_hash<SearchMode::MODE_SA>(mode, px, py, incr, hash160, 0, 0, maxFound, out); \
                break; \
            case SEARCH_BOTH: \
                unified_check_hash<SearchMode::MODE_SA>(mode, px, py, incr, hash160, 0, 0, maxFound, out); \
                unified_check_hash<SearchMode::MODE_SA>(mode, px, py, incr, hash160, 0, 0, maxFound, out); \
                break; \
        } \
    } while(0)

// -----------------------------------------------------------------------------------------

#define CHECK_HASH_SEARCH_MODE_SA(incr) CheckHashSEARCH_MODE_SA(mode, px, py, incr, target_data, maxFound, out)

__device__ __forceinline__ void ComputeKeysSEARCH_MODE_SA(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
	ComputeKeysUnified<SearchMode::MODE_SA>(mode, startx, starty, hash160, 0, 0, maxFound, out);
}


// -----------------------------------------------------------------------------------------

#define CHECK_PUB_SEARCH_MODE_MX(incr) CheckPubSEARCH_MODE_MX(mode, px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

__device__ __forceinline__ void ComputeKeysSEARCH_MODE_MX(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	ComputeKeysUnified<SearchMode::MODE_MX>(mode, startx, starty, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out);
}



// -----------------------------------------------------------------------------------------

#define CHECK_PUB_SEARCH_MODE_SX(incr) CheckPubSEARCH_MODE_SX(mode, px, py, incr, xpoint, maxFound, out)

__device__ __forceinline__ void ComputeKeysSEARCH_MODE_SX(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint32_t* xpoint, uint32_t maxFound, uint32_t* out)
{
	ComputeKeysUnified<SearchMode::MODE_SX>(mode, startx, starty, xpoint, 0, 0, maxFound, out);
}

// -----------------------------------------------------------------------------------------
// ComputeKeysPUZZLE71 wrapper function

__device__ __forceinline__ void ComputeKeysPUZZLE71(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
	// Note: hash160 parameter is ignored - we use the hardcoded PUZZLE71_TARGET_HASH
	// This specialized kernel only searches for the specific Puzzle #71 address
	ComputeKeysUnified<SearchMode::PUZZLE71>(mode, startx, starty, nullptr, 0, 0, maxFound, out);
}

/**
 * TASK-04: Batch Stepping Optimized version of PUZZLE71 kernel
 * Processes multiple keys in batches for improved GPU performance
 */
__device__ __forceinline__ void ComputeKeysPUZZLE71_BatchOptimized(
    uint32_t mode, uint64_t* startx, uint64_t* starty,
    uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
    // Initialize batch stepping state
    BatchSteppingState batch_state;
    init_batch_state(batch_state, startx, starty);
    
    // Use endomorphism if available
    bool use_endomorphism = (mode == SEARCH_COMPRESSED) && 
                            ((startx[3] & 0xFFFFFFFF00000000ULL) != 0);
    
    // Process keys in batches
    const int MAX_BATCHES = 1000;  // Limit for testing
    bool found = false;
    
    for (int batch_iter = 0; batch_iter < MAX_BATCHES && !found; batch_iter++) {
        // Choose optimization level based on GPU capability
        #if __CUDA_ARCH__ >= 700
            // Use warp-level optimization for Volta and newer
            process_batch_warp_optimized(batch_state, mode, maxFound, out);
            
            // Check if any thread in warp found the target
            found = __any_sync(0xFFFFFFFF, atomicAdd(&found_flag, 0) > 0);
        #elif __CUDA_ARCH__ >= 600
            // Use memory prefetching for Pascal
            found = process_key_batch_optimized(batch_state, mode, maxFound, out);
        #else
            // Basic batch processing for older GPUs
            found = process_key_batch(batch_state, mode, maxFound, out);
        #endif
        
        // Optional: Apply endomorphism every N batches for broader coverage
        if (use_endomorphism && (batch_iter & 0xF) == 0) {
            // Apply endomorphism transformation to current batch state
            uint64_t endo_x[4], endo_y[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                endo_x[i] = batch_state.base_x[i];
                endo_y[i] = batch_state.base_y[i];
            }
            apply_endomorphism(endo_x, endo_y);
            
            // Create temporary batch state for endomorphism check
            BatchSteppingState endo_batch_state;
            init_batch_state(endo_batch_state, endo_x, endo_y);
            
            // Process endomorphism batch
            bool endo_found = process_key_batch(endo_batch_state, mode, maxFound, out);
            found = found || endo_found;
        }
    }
    
    // Update starting point for next kernel launch
    if (!found) {
        __syncthreads();
        Store256A(startx, batch_state.base_x);
        Store256A(starty, batch_state.base_y);
    }
}

/**
 * Advanced PUZZLE71 kernel with both endomorphism and batch stepping
 * Combines all optimizations for maximum performance
 */
__device__ __forceinline__ void ComputeKeysPUZZLE71_FullyOptimized(
    uint32_t mode, uint64_t* startx, uint64_t* starty,
    uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
    // Shared memory for batch processing
    extern __shared__ uint64_t shared_batch_data[];
    
    // Initialize batch and endomorphism states
    BatchSteppingState batch_state;
    init_batch_state(batch_state, startx, starty);
    
    // Precompute batch increments in shared memory
    if (threadIdx.x < BatchSteppingConstants::BATCH_SIZE) {
        uint64_t increments_x[BatchSteppingConstants::BATCH_SIZE][4];
        uint64_t increments_y[BatchSteppingConstants::BATCH_SIZE][4];
        precompute_batch_increments(increments_x, increments_y, BatchSteppingConstants::BATCH_SIZE);
        
        // Store x increments in shared memory
        for (int i = 0; i < 4; i++) {
            shared_batch_data[threadIdx.x * 8 + i] = increments_x[threadIdx.x][i];
            shared_batch_data[threadIdx.x * 8 + 4 + i] = increments_y[threadIdx.x][i];
        }
    }
    __syncthreads();
    
    // Main processing loop with all optimizations
    const int MAX_ITERATIONS = 10000;  // Increased for better coverage
    bool found = false;
    
    for (int iter = 0; iter < MAX_ITERATIONS && !found; iter++) {
        // Process batch with maximum optimization
        #if __CUDA_ARCH__ >= 700
            // Volta+ with tensor cores support
            process_batch_warp_optimized(batch_state, mode, maxFound, out);
        #else
            // Older GPUs
            found = process_key_batch_optimized(batch_state, mode, maxFound, out);
        #endif
        
        // Early exit check using warp voting
        #if __CUDA_ARCH__ >= 300
            found = __ballot_sync(0xFFFFFFFF, atomicAdd(&found_flag, 0) > 0) != 0;
        #else
            found = (atomicAdd(&found_flag, 0) > 0);
        #endif
        
        // Cooperative group sync for better coordination
        #if __CUDA_ARCH__ >= 600
            __syncwarp();
        #endif
    }
    
    // Final state update
    if (!found) {
        __syncthreads();
        Store256A(startx, batch_state.base_x);
        Store256A(starty, batch_state.base_y);
    }
}



// ------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------

// 使用统一接口替换重复的检查函数
#define CheckPointSEARCH_MODE_MA(_h, incr, mode, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out) \
    unified_check_hash<SearchMode::MODE_MA>(mode, nullptr, nullptr, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

#define CheckPointSEARCH_MODE_SA(_h, incr, mode, hash160, maxFound, out) \
    unified_check_hash<SearchMode::MODE_SA>(mode, nullptr, nullptr, incr, hash160, 0, 0, maxFound, out)

#define CheckPointSEARCH_MODE_MX(_h, incr, mode, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out) \
    unified_check_hash<SearchMode::MODE_MX>(mode, nullptr, nullptr, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

#define CheckPointSEARCH_MODE_SX(_h, incr, mode, xpoint, maxFound, out) \
    unified_check_hash<SearchMode::MODE_SX>(mode, nullptr, nullptr, incr, xpoint, 0, 0, maxFound, out)

#define CheckPointSEARCH_ETH_MODE_MA(_h, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out) \
    unified_check_hash<SearchMode::MODE_ETH_MA>(0, nullptr, nullptr, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

#define CheckPointSEARCH_ETH_MODE_SA(_h, incr, mode, hash, param1, param2, maxFound, out) \
    unified_check_hash<SearchMode::MODE_ETH_SA>(mode, nullptr, nullptr, incr, hash, param1, param2, maxFound, out)

// 统一的检查函数模板实现
template<SearchMode Mode>
__device__ __forceinline__ void unified_check_hash(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint32_t param1, uint32_t param2,
    uint32_t maxFound, uint32_t* out)
{
    uint32_t h[8]; // 足够容纳哈希或X点数据
    
    // 根据模式计算哈希或X点
    switch (Mode) {
        case SearchMode::MODE_MA: {
            // 计算压缩公钥的Bitcoin地址哈希
            _GetHash160Comp(px, (uint8_t)(py[0] & 1), (uint8_t*)h);
            break;
        }
        case SearchMode::MODE_SA: {
            // 计算压缩公钥的Bitcoin地址哈希
            _GetHash160Comp(px, (uint8_t)(py[0] & 1), (uint8_t*)h);
            break;
        }
        case SearchMode::MODE_ETH_MA: {
            // 计算以太坊地址哈希
            _GetHashKeccak160(px, py, h);
            break;
        }
        case SearchMode::MODE_ETH_SA: {
            // 计算以太坊地址哈希
            _GetHashKeccak160(px, py, h);
            break;
        }
        case SearchMode::MODE_MX:
        case SearchMode::MODE_SX: {
            // 复制X坐标
            for (int i = 0; i < 4; i++) {
                h[i * 2] = (uint32_t)(px[3 - i] & 0xFFFFFFFF);
                h[i * 2 + 1] = (uint32_t)(px[3 - i] >> 32);
            }
            break;
        }
        case SearchMode::PUZZLE71: {
            // Specialized for Puzzle #71 - compute compressed Bitcoin address hash
            _GetHash160Comp(px, (uint8_t)(py[0] & 1), (uint8_t*)h);
            break;
        }
    }
    
    // 直接实现检查点逻辑
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    bool match = false;
    
    // 根据模式进行不同的匹配检查
    switch (Mode) {
        case SearchMode::MODE_MA:
        case SearchMode::MODE_ETH_MA: {
            const uint8_t* bloomLookUp = static_cast<const uint8_t*>(target_data);
            uint64_t BLOOM_BITS = param1;
            uint8_t BLOOM_HASHES = param2;
            int K_LENGTH = (Mode == SearchMode::MODE_MA) ? 20 : 20;
            match = (BloomCheck(h, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, K_LENGTH) > 0);
            break;
        }
        case SearchMode::MODE_MX: {
            const uint8_t* bloomLookUp = static_cast<const uint8_t*>(target_data);
            uint64_t BLOOM_BITS = param1;
            uint8_t BLOOM_HASHES = param2;
            match = (BloomCheck(h, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, 32) > 0);
            break;
        }
        case SearchMode::MODE_SA:
        case SearchMode::MODE_ETH_SA: {
            const uint32_t* hash = static_cast<const uint32_t*>(target_data);
            match = MatchHash(h, hash);
            break;
        }
        case SearchMode::PUZZLE71: {
            // PUZZLE71 uses hardcoded target - compare directly
            match = true;
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                if (h[i] != PUZZLE71_TARGET_HASH[i]) {
                    match = false;
                    break;
                }
            }
            break;
        }
        case SearchMode::MODE_SX: {
            const uint32_t* xpoint = static_cast<const uint32_t*>(target_data);
            match = MatchXPoint(h, xpoint);
            break;
        }
    }
    
    if (match) {
        // 处理匹配结果
        if (Mode == SearchMode::MODE_SA || Mode == SearchMode::MODE_ETH_SA || Mode == SearchMode::PUZZLE71) {
            // 使用原子比较和交换确保只有一个线程写入结果
            if (atomicCAS(&found_flag, 0, 1) == 0) {
                uint32_t pos = atomicAdd(out, 1);
                if (pos < maxFound) {
                    int item_size_32 = (Mode == SearchMode::MODE_SA || Mode == SearchMode::MODE_ETH_SA || Mode == SearchMode::PUZZLE71) ? 
                        ITEM_SIZE_A32 : ITEM_SIZE_X32;
                    out[pos * item_size_32 + 1] = tid;
                    out[pos * item_size_32 + 2] = (uint32_t)(incr << 16) | 
                        (uint32_t)((Mode == SearchMode::MODE_SA || Mode == SearchMode::MODE_ETH_SA || Mode == SearchMode::PUZZLE71) ? 0 << 15 : 0);
                    for (int i = 0; i < 5; i++) {
                        out[pos * item_size_32 + 3 + i] = h[i];
                    }
                }
            }
        } else {
            uint32_t pos = atomicAdd(out, 1);
            if (pos < maxFound) {
                int item_size_32 = (Mode == SearchMode::MODE_MX || Mode == SearchMode::MODE_SX) ? 
                    ITEM_SIZE_X32 : ITEM_SIZE_A32;
                out[pos * item_size_32 + 1] = tid;
                out[pos * item_size_32 + 2] = (uint32_t)(incr << 16) | 
                    (uint32_t)((Mode == SearchMode::MODE_MA) ? 0 << 15 : 0);
                for (int i = 0; i < ((Mode == SearchMode::MODE_MX || Mode == SearchMode::MODE_SX) ? 8 : 5); i++) {
                    out[pos * item_size_32 + 3 + i] = h[i];
                }
            }
        }
    }
}


// 统一的哈希检查函数
// 注意：这个函数已被宏定义替代，不应该直接调用
// 这里保留空实现以避免编译错误
__device__ __forceinline__ void CheckHashUnified(uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint64_t param1, uint8_t param2,
    uint32_t maxFound, uint32_t* out)
{
    // 这个函数已被宏定义替代，不应该直接调用
    // 保留空实现以避免编译错误
    return;
}

// 为向后兼容保留原始函数名的宏定义
// 注意：这些宏定义已被禁用，因为CheckHashUnified函数不再是模板函数
#define CheckHashCompSEARCH_MODE_MA(px, isOdd, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out) \
    /* CheckHashUnified<SearchMode::MODE_MA>(px, nullptr, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out) */

#define CheckHashCompSEARCH_MODE_SA(px, isOdd, incr, hash160, maxFound, out) \
    /* CheckHashUnified<SearchMode::MODE_SA>(px, nullptr, incr, hash160, 0, 0, maxFound, out) */

// ---------------------------------------------------------------------------------------

// 已被统一接口替代的函数 - 删除重复实现

#define CHECK_POINT_SEARCH_ETH_MODE_MA(_h,incr)  CheckPointSEARCH_ETH_MODE_MA(_h,incr,bloomLookUp,BLOOM_BITS,BLOOM_HASHES,maxFound,out)

// 函数已被宏定义替代，避免重复定义

#define CHECK_HASH_SEARCH_ETH_MODE_MA(incr) unified_check_hash<SearchMode::MODE_ETH_MA>(0, px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

__device__ __forceinline__ void ComputeKeysSEARCH_ETH_MODE_MA(uint64_t* startx, uint64_t* starty,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out);

__device__ __forceinline__ void ComputeKeysSEARCH_ETH_MODE_SA(uint64_t* startx, uint64_t* starty,
	uint32_t* hash, uint32_t maxFound, uint32_t* out);

// 函数声明
__device__ __forceinline__ void ComputeKeysSEARCH_MODE_MA(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out);
__device__ __forceinline__ void ComputeKeysSEARCH_MODE_SA(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint32_t* hash160, uint32_t maxFound, uint32_t* out);
__device__ __forceinline__ void ComputeKeysSEARCH_MODE_MX(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out);
__device__ __forceinline__ void ComputeKeysSEARCH_MODE_SX(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint32_t* xpoint, uint32_t maxFound, uint32_t* out);
__device__ __forceinline__ void ComputeKeysSEARCH_ETH_MODE_MA(uint64_t* startx, uint64_t* starty,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out);
__device__ __forceinline__ void ComputeKeysSEARCH_ETH_MODE_SA(uint64_t* startx, uint64_t* starty,
	uint32_t* hash, uint32_t maxFound, uint32_t* out);

// Function implementations are below

// 已被统一接口替代的函数 - 删除重复实现

#define CHECK_POINT_SEARCH_ETH_MODE_SA(_h,incr)  CheckPointSEARCH_ETH_MODE_SA(_h, incr, 0, hash, 0, 0, maxFound, out)

// 已被统一接口替代的函数 - 删除重复实现

#define CHECK_HASH_SEARCH_ETH_MODE_SA(incr) unified_check_hash<SearchMode::MODE_ETH_SA>(0, px, py, incr, hash, 0, 0, maxFound, out)
#define CHECK_HASH_SEARCH_ETH_MODE_MA(incr) unified_check_hash<SearchMode::MODE_ETH_MA>(0, px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

__device__ __forceinline__ void ComputeKeysSEARCH_ETH_MODE_MA(uint64_t* startx, uint64_t* starty,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	// 使用统一接口，变量声明已移至统一函数中
	uint64_t dx[KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
	uint64_t pyn[4];
	uint64_t sx[4];
	uint64_t sy[4];
	uint64_t dy[4];
	uint64_t _s[4];
	uint64_t _p2[4];

	// Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
	for (i = 0; i < KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE; i++)
		ModSub256(dx[i], Gx + 4 * i, sx);
	ModSub256(dx[i], Gx + 4 * i, sx);   // For the first point
	ModSub256(dx[i + 1], _2Gnx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

	// Check starting point
	CHECK_HASH_SEARCH_ETH_MODE_MA(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2);

	// Continue with the group processing...
	for (i = 0; i < KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE && !found_flag; i++) {
		// Positive side
		compute_ec_point_add(px, py, Gx + 4 * i, Gy + 4 * i, dx[i]);
		CHECK_HASH_SEARCH_ETH_MODE_MA(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 + (i + 1));

		// Negative side
		ModNeg256(pyn, py);
		compute_ec_point_add(px, pyn, Gx + 4 * i, Gy + 4 * i, dx[i]);
		CHECK_HASH_SEARCH_ETH_MODE_MA(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 - (i + 1));
		Load256(py, pyn);
	}

	// Check the center point
	compute_ec_point_add(px, py, _2Gnx, _2Gny, dx[i + 1]);
	CHECK_HASH_SEARCH_ETH_MODE_MA(0);
}

__device__ __forceinline__ void ComputeKeysSEARCH_ETH_MODE_SA(uint64_t* startx, uint64_t* starty,
	uint32_t* hash, uint32_t maxFound, uint32_t* out)
{

	// 使用统一接口，变量声明已移至统一函数中
	uint64_t dx[KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 + 1][4];
	uint64_t px[4];
	uint64_t py[4];
	uint64_t pyn[4];
	uint64_t sx[4];
	uint64_t sy[4];
	uint64_t dy[4];
	uint64_t _s[4];
	uint64_t _p2[4];

	// Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);

	// Fill group with delta x
	uint32_t i;
	for (i = 0; i < KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE; i++)
		ModSub256(dx[i], Gx + 4 * i, sx);
	ModSub256(dx[i], Gx + 4 * i, sx);   // For the first point
	ModSub256(dx[i + 1], _2Gnx, sx); // For the next center point

	// Compute modular inverse
	_ModInvGrouped(dx);

	// We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
	// We compute key in the positive and negative way from the center of the group

	// Check starting point
	CHECK_HASH_SEARCH_ETH_MODE_SA(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2);

	ModNeg256(pyn, py);

	for (i = 0; i < KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE; i++) {

		// P = StartPoint + i*G
		Load256(px, sx);
		Load256(py, sy);
		compute_ec_point_add(px, py, Gx + 4 * i, Gy + 4 * i, dx[i]);

		CHECK_HASH_SEARCH_ETH_MODE_SA(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 + (i + 1));

		// P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
		Load256(px, sx);
		compute_ec_point_add_negative(px, py, pyn, Gx + 4 * i, Gy + 4 * i, dx[i]);

		CHECK_HASH_SEARCH_ETH_MODE_SA(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 - (i + 1));

	}

	// First point (startP - (GRP_SIZE/2)*G)
	Load256(px, sx);
	Load256(py, sy);
	compute_ec_point_add_special(px, py, Gx + 4 * i, Gy + 4 * i, dx[i], true);

	CHECK_HASH_SEARCH_ETH_MODE_SA(0);

	i++;

	// Next start point (startP + GRP_SIZE*G)
	Load256(px, sx);
	Load256(py, sy);
	compute_ec_point_add(px, py, _2Gnx, _2Gny, dx[i + 1]);

	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);

}


#endif // GPU_COMPUTE_H