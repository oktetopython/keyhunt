#pragma once

#include "../common.h"

namespace gecc {
namespace arith {

// A warp (32 threads) is divided into several *slots* (i.e., subwarps), each of
// which has $W$ *lanes* (i.e., threads).
//
// The number $W$ is called *slot width*.
template <u32 W> struct LayoutT {
  static const u32 WIDTH = W;

  static_assert(32 % WIDTH == 0, "Lane num must be a divisor of 32.");
  static_assert(WIDTH <= 16, "Currently we only support N = 1, 2, 4, 8, 16.");

  __device__ __forceinline__ static u32 lane_idx() {
    return threadIdx.x & MASK;
  }

  __device__ __forceinline__ static u32 slot_idx() {
    return threadIdx.x >> LOG_WIDTH;
  }

  __device__ __forceinline__ static u32 global_slot_idx() {
    return (block_idx() * block_dim() + threadIdx.x) >> LOG_WIDTH;
  }

  __device__ __forceinline__ static u32 ballot(bool b) {
    if (WIDTH == 1) {
      return b << lane_idx();
    }
    return __ballot_sync(sync_mask(), b) >> slot_start();
  }

  template <typename T>
  __device__ __forceinline__ static T shfl(T value, u32 lane_idx) {
    if (WIDTH == 1) {
      return value;
    }
    return __shfl_sync(sync_mask(), value, slot_start() + lane_idx, WIDTH);
  }

  template <typename T> __device__ __forceinline__ static T shfl_bot(T value) {
    return shfl(value, bot_lane_idx);
  }

  template <typename T>
  __device__ __forceinline__ static T shfl_up(T value, T na) {
    if (WIDTH == 1) {
      return na;
    }
    T result = __shfl_up_sync(sync_mask(), value, 1, WIDTH);
    return lane_idx() == bot_lane_idx ? na : result;
  }

  template <typename T>
  __device__ __forceinline__ static T shfl_down(T value, T na) {
    if (WIDTH == 1) {
      return na;
    }
    T result = __shfl_down_sync(sync_mask(), value, 1, WIDTH);
    return lane_idx() == top_lane_idx ? na : result;
  }

private:
  static const u32 bot_lane_idx = 0;
  static const u32 top_lane_idx = WIDTH - 1;

  static constexpr u32 log(u32 n) {
    u32 k = 0;
    while ((1U << k) < n) {
      k++;
    }
    return k;
  }

  static const u32 MASK = WIDTH - 1;

  __device__ __forceinline__ static u32 slot_start() {
    return threadIdx.x & (0x1F & ~MASK);
  }

  __device__ __forceinline__ static u32 sync_mask() {
    return ((1U << WIDTH) - 1) << slot_start();
  }

public:
  static const u32 LOG_WIDTH = log(WIDTH);
};

} // namespace arith
} // namespace gecc
