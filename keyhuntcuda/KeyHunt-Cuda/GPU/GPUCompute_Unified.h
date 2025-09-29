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

#ifndef GPU_COMPUTE_UNIFIED_H
#define GPU_COMPUTE_UNIFIED_H

#include "GPUMemoryOptimized.h"

#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../hash/keccak160.h"
#include "../Constants.h"
#include "SearchMode.h"
#include "GPUCompute.h"
#include "GPUMath.h"
#include "GPUHash.h"

// Global device variables are declared in GPUMemoryOptimized.h

// EC functions are defined in GPUCompute.h - no forward declarations needed

// Unified hash checking function template
template<SearchMode Mode>
__device__ __forceinline__ void unified_check_hash(
    uint32_t mode, uint64_t* px, uint64_t* py, int32_t incr,
    const void* target_data, uint32_t param1, uint32_t param2,
    uint32_t maxFound, uint32_t* out);

// Unified kernel launch function template
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
    CoinType coin_type);

#endif // GPU_COMPUTE_UNIFIED_H