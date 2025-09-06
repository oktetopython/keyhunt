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

// Search mode enumeration for unified GPU kernel interface
enum class SearchMode {
    MODE_MA = 1,      // Multiple addresses
    MODE_SA = 2,      // Single address
    MODE_MX = 3,      // Multiple X-points
    MODE_SX = 4,      // Single X-point
    MODE_ETH_MA = 5,  // Ethereum multiple addresses
    MODE_ETH_SA = 6   // Ethereum single address
};

// Include hash function headers
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Constants.h"
#include "GPUCompute_Unified.h"

__device__ uint64_t* _2Gnx = NULL;
__device__ uint64_t* _2Gny = NULL;

__device__ uint64_t* Gx = NULL;
__device__ uint64_t* Gy = NULL;

// GPU thread synchronization for preventing duplicate results
__device__ int found_flag = 0;

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        } \
    } while(0)

// Function to reset the found flag before each kernel launch
__global__ void reset_found_flag() {
    CUDA_CHECK(cudaGetLastError());
    found_flag = 0;
    CUDA_CHECK(cudaGetLastError());
}

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

__device__ int Test_Bit_Set_Bit(const uint8_t* buf, uint32_t bit)
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

__device__ uint32_t MurMurHash2(const void* key, int len, uint32_t seed)
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

__device__ int BloomCheck(const uint32_t* hash, const uint8_t* inputBloomLookUp, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t K_LENGTH)
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

__device__ __noinline__ bool MatchHash(const uint32_t* _h, const uint32_t* hash)
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

__device__ __noinline__ bool MatchXPoint(const uint32_t* _h, const uint32_t* xpoint)
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

#define CHECK_HASH_SEARCH_MODE_MA(incr) unified_check_hash<SearchMode::MODE_MA>(mode, px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

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
__device__ void ComputeKeysUnified(uint32_t mode, uint64_t* startx, uint64_t* starty,
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

// Legacy function wrappers for backward compatibility
__device__ void ComputeKeysSEARCH_MODE_MA(uint32_t mode, uint64_t* startx, uint64_t* starty,
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

#define CHECK_HASH_SEARCH_MODE_SA(incr) CheckHashSEARCH_MODE_SA(mode, px, py, incr, hash160, maxFound, out)

__device__ void ComputeKeysSEARCH_MODE_SA(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint32_t* hash160, uint32_t maxFound, uint32_t* out)
{
	ComputeKeysUnified<SearchMode::MODE_SA>(mode, startx, starty, hash160, 0, 0, maxFound, out);
}


// -----------------------------------------------------------------------------------------

#define CHECK_PUB_SEARCH_MODE_MX(incr) CheckPubSEARCH_MODE_MX(mode, px, py, incr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out)

__device__ void ComputeKeysSEARCH_MODE_MX(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint32_t maxFound, uint32_t* out)
{
	ComputeKeysUnified<SearchMode::MODE_MX>(mode, startx, starty, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, out);
}



// -----------------------------------------------------------------------------------------

#define CHECK_PUB_SEARCH_MODE_SX(incr) CheckPubSEARCH_MODE_SX(mode, px, py, incr, xpoint, maxFound, out)

__device__ void ComputeKeysSEARCH_MODE_SX(uint32_t mode, uint64_t* startx, uint64_t* starty,
	uint32_t* xpoint, uint32_t maxFound, uint32_t* out)
{
	ComputeKeysUnified<SearchMode::MODE_SX>(mode, startx, starty, xpoint, 0, 0, maxFound, out);
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

// 统一的检查函数模板
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
        case SearchMode::MODE_SX: {
            const uint32_t* xpoint = static_cast<const uint32_t*>(target_data);
            match = MatchXPoint(h, xpoint);
            break;
        }
    }
    
    if (match) {
        // 处理匹配结果
        if (Mode == SearchMode::MODE_SA || Mode == SearchMode::MODE_ETH_SA) {
            // 使用原子比较和交换确保只有一个线程写入结果
            if (atomicCAS(&found_flag, 0, 1) == 0) {
                uint32_t pos = atomicAdd(out, 1);
                if (pos < maxFound) {
                    int item_size_32 = (Mode == SearchMode::MODE_SA || Mode == SearchMode::MODE_ETH_SA) ? 
                        ITEM_SIZE_A32 : ITEM_SIZE_X32;
                    out[pos * item_size_32 + 1] = tid;
                    out[pos * item_size_32 + 2] = (uint32_t)(incr << 16) | 
                        (uint32_t)((Mode == SearchMode::MODE_SA || Mode == SearchMode::MODE_ETH_SA) ? 0 << 15 : 0);
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

__device__ void ComputeKeysSEARCH_ETH_MODE_MA(uint64_t* startx, uint64_t* starty,
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

	ModNeg256(pyn, py);

	for (i = 0; i < KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE; i++) {

		// P = StartPoint + i*G
		Load256(px, sx);
		Load256(py, sy);
		compute_ec_point_add(px, py, Gx + 4 * i, Gy + 4 * i, dx[i]);

		CHECK_HASH_SEARCH_ETH_MODE_MA(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 + (i + 1));

		// P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
		Load256(px, sx);
		compute_ec_point_add_negative(px, py, pyn, Gx + 4 * i, Gy + 4 * i, dx[i]);

		CHECK_HASH_SEARCH_ETH_MODE_MA(KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE / 2 - (i + 1));

	}

	// First point (startP - (GRP_SIZE/2)*G)
	Load256(px, sx);
	Load256(py, sy);
	compute_ec_point_add_special(px, py, Gx + 4 * i, Gy + 4 * i, dx[i], true);

	CHECK_HASH_SEARCH_ETH_MODE_MA(0);

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



// 已被统一接口替代的函数 - 删除重复实现

#define CHECK_POINT_SEARCH_ETH_MODE_SA(_h,incr)  CheckPointSEARCH_ETH_MODE_SA(_h, incr, 0, hash, 0, 0, maxFound, out)

// 已被统一接口替代的函数 - 删除重复实现

#define CHECK_HASH_SEARCH_ETH_MODE_SA(incr) unified_check_hash<SearchMode::MODE_ETH_SA>(0, px, py, incr, hash, 0, 0, maxFound, out)

__device__ void ComputeKeysSEARCH_ETH_MODE_SA(uint64_t* startx, uint64_t* starty,
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