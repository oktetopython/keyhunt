#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

namespace gecc {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using size_t = std::size_t;

static const u32 MAX_LIMBS = 64;

static const u32 MAX_BYTES = 64 * 8;

static const u32 WARP_SIZE = 32;
static const u32 MAX_THREADS = 1024;

static const u32 MAX_BITS = 1024;
static u32 MAX_SM_NUMS = 80;
static u32 MAX_PersistingL2CacheSize = 40;

// TODO: support blockDim > 65536
__device__ __forceinline__ static u32 block_idx() { return blockIdx.x; }

__device__ __forceinline__ static u32 block_dim() { return blockDim.x; }

} // namespace gecc

#ifndef GECC_QAPW_TEST
#ifndef GECC_QAPW_OPT_SHARED_INPUTS
#define GECC_QAPW_OPT_SHARED_INPUTS
#endif
// #ifndef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
// #define GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
// #endif
#endif
