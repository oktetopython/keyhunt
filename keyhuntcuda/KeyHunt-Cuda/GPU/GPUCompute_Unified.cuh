/*
 * This file is part of the KeyHunt-Cuda distribution (https://github.com/your-repo/keyhunt-cuda).
 * Copyright (c) 2025 Your Name.
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

#ifndef GPU_COMPUTE_UNIFIED_CUH
#define GPU_COMPUTE_UNIFIED_CUH

#include "GPUCompute_Unified.h"
#include "GPUCompute.h"
#include "GPUHash.h"
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"

// Forward declarations of kernel functions
__global__ void compute_keys_mode_ma(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES,
    uint64_t* keys, uint32_t maxFound, uint32_t* found);
__global__ void compute_keys_comp_mode_ma(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
    uint32_t maxFound, uint32_t* found);
__global__ void compute_keys_mode_sa(uint32_t mode, uint32_t* hash160, uint64_t* keys, uint32_t maxFound, uint32_t* found);
__global__ void compute_keys_comp_mode_sa(uint32_t mode, uint32_t* hash160, uint64_t* keys, uint32_t maxFound, uint32_t* found);
__global__ void compute_keys_comp_mode_mx(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
    uint32_t maxFound, uint32_t* found);
__global__ void compute_keys_comp_mode_sx(uint32_t mode, uint32_t* xpoint, uint64_t* keys, uint32_t maxFound, uint32_t* found);
__global__ void compute_keys_mode_eth_ma(uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
    uint32_t maxFound, uint32_t* found);
__global__ void compute_keys_mode_eth_sa(uint32_t* hash, uint64_t* keys, uint32_t maxFound, uint32_t* found);
// PUZZLE71 kernels
__global__ void compute_keys_puzzle71(uint32_t mode, uint32_t* hash, uint64_t* keys, uint32_t maxFound, uint32_t* found);
__global__ void compute_keys_comp_puzzle71(uint32_t mode, uint32_t* hash, uint64_t* keys, uint32_t maxFound, uint32_t* found);

// Forward declaration of unified hash checking function template
// (Implementation is in GPUCompute.h)
template<SearchMode Mode>
__device__ __forceinline__ void unified_check_hash(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint32_t param1, uint32_t param2,
    uint32_t maxFound, uint32_t* out);

// Implementation of unified kernel launch function template
template<SearchMode Mode>
__host__ void launch_unified_kernel(
    uint32_t mode,
    const void* target_data,
    uint32_t param1,
    uint32_t param2,
    uint64_t* keys,
    uint32_t maxFound,
    uint32_t* found,
    uint32_t blocks,
    uint32_t threads_per_block,
    CompressionMode comp_mode,
    CoinType coin_type)
{
    // Reset found count
    cudaMemset(found, 0, 4);
    
    // Launch appropriate kernel based on mode
    switch (Mode) {
        case SearchMode::MODE_MA:
            if (coin_type == CoinType::BITCOIN) {
                if (comp_mode == CompressionMode::COMPRESSED) {
                    compute_keys_comp_mode_ma<<<blocks, threads_per_block>>>(
                        mode, (uint8_t*)target_data, param1, param2, keys, maxFound, found);
                } else {
                    compute_keys_mode_ma<<<blocks, threads_per_block>>>(
                        mode, (uint8_t*)target_data, param1, param2, keys, maxFound, found);
                }
            } else {
                compute_keys_mode_eth_ma<<<blocks, threads_per_block>>>(
                    (uint8_t*)target_data, param1, param2, keys, maxFound, found);
            }
            break;
            
        case SearchMode::MODE_SA:
            if (coin_type == CoinType::BITCOIN) {
                if (comp_mode == CompressionMode::COMPRESSED) {
                    compute_keys_comp_mode_sa<<<blocks, threads_per_block>>>(
                        mode, (uint32_t*)target_data, keys, maxFound, found);
                } else {
                    compute_keys_mode_sa<<<blocks, threads_per_block>>>(
                        mode, (uint32_t*)target_data, keys, maxFound, found);
                }
            } else {
                compute_keys_mode_eth_sa<<<blocks, threads_per_block>>>(
                    (uint32_t*)target_data, keys, maxFound, found);
            }
            break;
            
        case SearchMode::MODE_MX:
            if (comp_mode == CompressionMode::COMPRESSED) {
                compute_keys_comp_mode_mx<<<blocks, threads_per_block>>>(
                    mode, (uint8_t*)target_data, param1, param2, keys, maxFound, found);
            }
            break;
            
        case SearchMode::MODE_SX:
            if (comp_mode == CompressionMode::COMPRESSED) {
                compute_keys_comp_mode_sx<<<blocks, threads_per_block>>>(
                    mode, (uint32_t*)target_data, keys, maxFound, found);
            }
            break;
            
        case SearchMode::MODE_ETH_MA:
            compute_keys_mode_eth_ma<<<blocks, threads_per_block>>>(
                (uint8_t*)target_data, param1, param2, keys, maxFound, found);
            break;
            
        case SearchMode::MODE_ETH_SA:
            compute_keys_mode_eth_sa<<<blocks, threads_per_block>>>(
                (uint32_t*)target_data, keys, maxFound, found);
            break;
    }
}

#endif // GPU_COMPUTE_UNIFIED_CUH