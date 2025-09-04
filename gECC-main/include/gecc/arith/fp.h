#pragma once

#include "../common.h"

#include "digit.h"

#include <array>

#define DEFINE_FP(FP_NAME, FP_TYPE, DIGIT_TYPE, DIGIT_WIDTH, LAYOUT, LIMBS)                 \
  using FP_NAME##Factory =                                                     \
      gecc::arith::FpFactory<gecc::arith::DigitT<DIGIT_TYPE, DIGIT_WIDTH>, LAYOUT, LIMBS>;  \
  __device__ __constant__ FP_NAME##Factory::Constant FP_NAME##DCONST;          \
  using FP_NAME =                                                              \
      FpT<FP_NAME##Factory, gecc::arith::constants::FP_TYPE, FP_NAME##DCONST>

#define DEFINE_SM2_FP(FP_NAME, FP_TYPE, DIGIT_TYPE, DIGIT_WIDTH, LAYOUT, LIMBS, MONT_FLAG, CURVE_FLAG)                 \
  using FP_NAME##Factory =                                                     \
      gecc::arith::FpFactory<gecc::arith::DigitT<DIGIT_TYPE, DIGIT_WIDTH>, LAYOUT, LIMBS>;  \
  __device__ __constant__ FP_NAME##Factory::Constant FP_NAME##DCONST;          \
  using FP_NAME =                                                              \
      FpT<FP_NAME##Factory, gecc::arith::constants::FP_TYPE, FP_NAME##DCONST, MONT_FLAG, CURVE_FLAG>

namespace gecc {
namespace arith {

struct FpConstant {
  u32 bits;
  u32 rexp;
  u64 pinv;
  u64 p[MAX_LIMBS];
  u64 p_minus_2[MAX_LIMBS];
  u64 pp[MAX_LIMBS];
  u64 r[MAX_LIMBS];
  u64 r2[MAX_LIMBS];
  u32 adicity2;
  u64 generator[MAX_LIMBS];
  u64 inv2[MAX_LIMBS];
};

namespace constants {
#include "fp_constants.h"
} // namespace constants

enum MONTFLAG {
  CIOS,
  SOS,
};

enum CURVEFLAG {
  DEFAULT,
  SM2,
};

template <typename D, typename L, u32 N> struct FpFactory {
  using Digit = D;
  using Layout = L;

  using Base = typename Digit::Base;

  static const u32 LIMBS = N;
  static_assert(LIMBS * sizeof(Base) <= MAX_BYTES, "");
  static_assert(LIMBS % Layout::WIDTH == 0, "");
  static const u32 LIMBS_PER_LANE = LIMBS / Layout::WIDTH;

  struct Constant {
    Base p[LIMBS];
    Base pinv;
    Base p_minus_2[LIMBS];
    Base pp[LIMBS];
    Base r[LIMBS];
    Base r2[LIMBS];
    Base one[LIMBS];
  };
};

template <typename Factory, const FpConstant &HCONST,
          typename Factory::Constant &DCONST, const MONTFLAG mont_flag = MONTFLAG::CIOS, const CURVEFLAG curve_flag = CURVEFLAG::DEFAULT>
struct FpT {
  using Fp = FpT;

  using Digit = typename Factory::Digit;
  using Base = typename Digit::Base;
  using Layout = typename Factory::Layout;

  static constexpr FpConstant CONST = HCONST;

  static const int DEGREE = 1;
  static const u32 LIMBS = Factory::LIMBS;
  static const u32 LIMBS_PER_LANE = Factory::LIMBS_PER_LANE;
  static const u32 BITS = HCONST.bits;

  static const size_t SIZE = sizeof(Base) * LIMBS;

  static const u32 REXP = (sizeof(u64) / sizeof(Base)) * HCONST.rexp;

  __host__ static void layout_dependent_copy(Base *result, const u64 *bytes) {
    for (u32 lane_idx = 0; lane_idx < Layout::WIDTH; ++lane_idx) {
      for (u32 i = 0; i < LIMBS_PER_LANE; ++i) {
        if (Digit::Digit_WIDTH != 32 && Digit::Digit_WIDTH != 64) {
          result[i * Layout::WIDTH + lane_idx] = (Base)(
            bytes[lane_idx * LIMBS_PER_LANE + i]);
        }
        else {
          result[i * Layout::WIDTH + lane_idx] = reinterpret_cast<const Base *>(
            bytes)[lane_idx * LIMBS_PER_LANE + i];
        }
      }
    }
  }

  __host__ static void initialize() {
    typename Factory::Constant c;
    layout_dependent_copy(c.p, HCONST.p);
    layout_dependent_copy(c.pp, HCONST.pp);
    layout_dependent_copy(c.r, HCONST.r);
    for (u32 i = 0; i < LIMBS; ++i) {
      if (Digit::Digit_WIDTH != 32 && Digit::Digit_WIDTH != 64) {
        c.p_minus_2[i] = (Base)((HCONST.p_minus_2)[i]);
        c.r2[i] = (Base)((HCONST.r2)[i]);
      }
      else {
        c.p_minus_2[i] = reinterpret_cast<const Base *>(HCONST.p_minus_2)[i];
        c.r2[i] = reinterpret_cast<const Base *>(HCONST.r2)[i];
      }
    }
    c.one[0] = 1;
    for (u32 i = 1; i < LIMBS; ++i) {
      c.one[i] = 0;
    }
    c.pinv = (Base)HCONST.pinv;
    CUDA_CHECK(cudaMemcpyToSymbol(DCONST, &c, sizeof(c)));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  __device__ __forceinline__ static FpT zero() {
    FpT result;
#pragma unroll
    for (u32 i = 0; i < LIMBS_PER_LANE; ++i) {
      result.digits[i] = 0;
    }
    return result;
  }

  __device__ __forceinline__ static FpT
  load_const(const typename Factory::Base *C) {
    FpT result;
    auto ptr = C + Layout::lane_idx();
#pragma unroll
    for (u32 i = 0; i < LIMBS_PER_LANE; ++i, ptr += Layout::WIDTH) {
      result.digits[i] = *ptr;
    }
    return result;
  }

  __device__ __forceinline__ static FpT mont_one() {
    return load_const(DCONST.r);
  }

  __device__ __forceinline__ static FpT pp() { return load_const(DCONST.pp); }

  __device__ __forceinline__ static FpT p() { return load_const(DCONST.p); }

  __device__ __forceinline__ bool is_zero() const {
    bool equal = true;
#pragma unroll
    for (u32 i = 0; i < LIMBS_PER_LANE; ++i) {
      equal &= digits[i] == static_cast<Base>(0);
    }
    return Layout::ballot(equal) == (1U << Layout::WIDTH) - 1U;
  }

  __device__ __forceinline__ static u32 memory_offset(const u32 k,
                                                      const u32 slot_idx,
                                                      const u32 lane_idx,
                                                      const u32 j) {
#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    return j << (k + Layout::LOG_WIDTH) | slot_idx << Layout::LOG_WIDTH |
           lane_idx;
#else
    return slot_idx * LIMBS + lane_idx * LIMBS_PER_LANE + j;
#endif
  }

  __device__ __forceinline__ static u32 memory_offset_arbitrary(const u32 count,
                                                      const u32 slot_idx,
                                                      const u32 lane_idx,
                                                      const u32 j) {
#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    return (j * count * Layout::WIDTH) + (slot_idx * Layout::WIDTH) +
           lane_idx;
#else
    return slot_idx * LIMBS + lane_idx * LIMBS_PER_LANE + j;
#endif
  }

  __device__ __forceinline__ void load(const Base *memory, const u32 k,
                                       const u32 slot_idx, const u32 lane_idx) {
#pragma unroll
    for (u32 j = 0; j < LIMBS_PER_LANE; ++j) {
      digits[j] = memory[memory_offset(k, slot_idx, lane_idx, j)];
    }
  }

  __device__ __forceinline__ void load_arbitrary(const Base *memory, const u32 k,
                                       const u32 slot_idx, const u32 lane_idx) {
#pragma unroll
    for (u32 j = 0; j < LIMBS_PER_LANE; ++j) {
      digits[j] = memory[memory_offset_arbitrary(k, slot_idx, lane_idx, j)];
    }
  }

  __device__ __forceinline__ void store(Base *memory, const u32 k,
                                        const u32 slot_idx,
                                        const u32 lane_idx) const {
#pragma unroll
    for (u32 j = 0; j < LIMBS_PER_LANE; ++j) {
      memory[memory_offset(k, slot_idx, lane_idx, j)] = digits[j];
    }
  }

  __device__ __forceinline__ void store_arbitrary(Base *memory, const u32 k,
                                        const u32 slot_idx,
                                        const u32 lane_idx) const {
#pragma unroll
    for (u32 j = 0; j < LIMBS_PER_LANE; ++j) {
      memory[memory_offset_arbitrary(k, slot_idx, lane_idx, j)] = digits[j];
    }
  }

  __device__ __forceinline__ bool operator==(const FpT &o) const {
    bool equal = true;
#pragma unroll
    for (u32 i = 0; i < LIMBS_PER_LANE; ++i) {
      equal &= digits[i] == o.digits[i];
    }
    return Layout::ballot(equal) == (1U << Layout::WIDTH) - 1U;
  }


  __device__ __forceinline__ FpT reduce_to_p() const {
    return reduce_to(load_const(DCONST.p));
  }

  __device__ __forceinline__ FpT reduce_to_pp() const {
    return reduce_to(pp());
  }

  #include "details/fp_mont_add_sub.h"
  #include "details/fp_mont_multiply.h"
  #include "details/fp_mont_inverse.h"

  __device__ __forceinline__ FpT square() const { 
    return (*this) * (*this); 
  }

  __device__ __forceinline__ FpT &inplace_to_montgomery() {
    return *this = this->mont_multiply(DCONST.r2);
  }

  __device__ __forceinline__ FpT from_montgomery() const {
    return mont_multiply(DCONST.one);
  }

  Base digits[LIMBS_PER_LANE];

private:
  __device__ __forceinline__ static bool
  propagate_carries(bool carry, Base buffer[LIMBS_PER_LANE]) {
    const u32 carries = Layout::ballot(carry) << 1;
    bool is_max = true;
#pragma unroll
    for (u32 i = 0; i < LIMBS_PER_LANE; ++i) {
      is_max &= buffer[i] == Digit::MAX;
    }
    const u32 is_maxs = Layout::ballot(is_max);
    const u32 propagated_carries = ((is_maxs + carries) ^ is_maxs) | carries;
    CC<Digit, LIMBS_PER_LANE>::add_cy(
        buffer, buffer, propagated_carries >> Layout::lane_idx() & 1);
    return propagated_carries >> Layout::WIDTH;
  }

  __device__ __forceinline__ static bool
  propagate_borrows(bool borrow, Base buffer[LIMBS_PER_LANE]) {
    const u32 borrows = Layout::ballot(borrow) << 1;
    bool is_min = true;
#pragma unroll
    for (u32 i = 0; i < LIMBS_PER_LANE; ++i) {
      is_min &= buffer[i] == Digit::MIN;
    }
    const u32 is_mins = Layout::ballot(is_min);
    const u32 propagated_borrows = (((is_mins + borrows) ^ is_mins) | borrows);
    CC<Digit, LIMBS_PER_LANE>::sub_br(
        buffer, buffer, propagated_borrows >> Layout::lane_idx() & 1);
    return propagated_borrows >> Layout::WIDTH;
  }

  __device__ __forceinline__ FpT reduce_to(const FpT &o) const {
    FpT result;
    Base borrow =
        CC<Digit, LIMBS_PER_LANE>::sub_br(result.digits, digits, o.digits);
    const u32 borrows = Layout::ballot(borrow) << 1;
    bool is_min = true;
#pragma unroll
    for (u32 i = 0; i < LIMBS_PER_LANE; ++i) {
      is_min &= result.digits[i] == Digit::MIN;
    }
    const u32 is_mins = Layout::ballot(is_min);
    const u32 propagated_borrows = (((is_mins + borrows) ^ is_mins) | borrows);
    CC<Digit, LIMBS_PER_LANE>::sub_br(result.digits, result.digits,
                                      propagated_borrows >> Layout::lane_idx() &
                                          1);
    bool is_underflow = propagated_borrows >> Layout::WIDTH;
#pragma unroll
    for (u32 i = 0; i < LIMBS_PER_LANE; ++i) {
      result.digits[i] = is_underflow ? digits[i] : result.digits[i];
    }
    return result;
  }
};

} // namespace arith
} // namespace gecc
