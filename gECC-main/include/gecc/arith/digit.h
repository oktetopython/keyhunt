#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace gecc {
namespace arith {

using u32 = uint32_t;
using u64 = uint64_t;

namespace details {

template <typename B, const u32 B_WIDTH> struct DigitBase {
  using Base = B;

  static_assert(std::is_same<Base, u32>::value ||
                    std::is_same<Base, u64>::value,
                "Base must be either u32 or u64.");

  // attention please, BYTES and BITS not correct for u52.
  static const u32 BYTES = sizeof(Base);
  static const u32 BITS = 8 * BYTES;

  // Use Digit_WIDTH when you need to read the actual storage bits
  static const u32 Digit_WIDTH = B_WIDTH;

  static const Base MIN = 0;
  // static const Base MAX = ((u64)0xFFFFFFFFFFFFFFFF) >> (64 - Digit_WIDTH);
  static const Base MAX = std::is_same<Base, u64>::value ? (((u64)0xFFFFFFFFFFFFFFFF) >> (64 - Digit_WIDTH)) : (((u32)0xFFFFFFFF) >> (32 - Digit_WIDTH));
};

} // namespace details

template <typename B, const u32 B_WIDTH> struct DigitT;

template <> struct DigitT<u32, 32> : public details::DigitBase<u32, 32> {
  __device__ __forceinline__ static void add(Base &d, Base a, Base b) {
    asm("add.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  }

  __device__ __forceinline__ static Base add_cy(Base &c, Base a, Base b) {
    Base cy;
    asm("add.cc.u32 %0, %2, %3;\n\t"
        "addc.u32 %1, 0, 0;"
        : "=r"(c), "=r"(cy)
        : "r"(a), "r"(b));
    return cy;
  }

  __device__ __forceinline__ static void add_cy(Base &cy, Base &c, Base a,
                                                Base b) {
    asm("add.cc.u32 %0, %2, %3;\n\t"
        "addc.u32 %1, %4, 0;"
        : "=r"(c), "=r"(cy)
        : "r"(a), "r"(b), "r"(cy));
  }

  __device__ __forceinline__ static void sub(Base &d, Base a, Base b) {
    asm("sub.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  }

  __device__ __forceinline__ static Base mul_lo(Base a, Base b) {
    Base lo;
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(a), "r"(b));
    return lo;
  }

  __device__ __forceinline__ static Base mul_hi(Base a, Base b) {
    Base hi;
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(a), "r"(b));
    return hi;
  }

  __device__ __forceinline__ static void mad_wide(Base &lo, Base &hi, Base a,
                                                  Base b, Base c) {
    // TODO: Improve
    asm("mad.lo.cc.u32 %0, %2, %3, %4;\n\t"
        "madc.hi.u32 %1, %2, %3, 0;"
        : "=r"(lo), "=r"(hi)
        : "r"(a), "r"(b), "r"(c));
  }
};

template <> struct DigitT<u64, 64> : public details::DigitBase<u64, 64> {
  __device__ __forceinline__ static void add(Base &d, Base a, Base b) {
    asm("add.u64 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  }

  __device__ __forceinline__ static Base add_cy(Base &c, Base a, Base b) {
    Base cy;
    asm("add.cc.u64 %0, %2, %3;\n\t"
        "addc.u64 %1, 0, 0;"
        : "=l"(c), "=l"(cy)
        : "l"(a), "l"(b));
    return cy;
  }

  __device__ __forceinline__ static void add_cy(Base &cy, Base &c, Base a,
                                                Base b) {
    asm("add.cc.u64 %0, %2, %3;\n\t"
        "addc.u64 %1, %4, 0;"
        : "=l"(c), "=l"(cy)
        : "l"(a), "l"(b), "l"(cy));
  }

  __device__ __forceinline__ static void sub(Base &d, Base a, Base b) {
    asm("sub.u64 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  }

  __device__ __forceinline__ static Base mul_lo(Base a, Base b) {
    Base lo;
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
    return lo;
  }

  __device__ __forceinline__ static Base mul_hi(Base a, Base b) {
    Base hi;
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
    return hi;
  }

  __device__ __forceinline__ static void mad_wide(Base &lo, Base &hi, Base a,
                                                  Base b, Base c) {
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
        "madc.hi.u64 %1, %2, %3, 0;"
        : "=l"(lo), "=l"(hi)
        : "l"(a), "l"(b), "l"(c));
  }
};

template <typename Digit, int LIMBS_PER_LANE> struct CC {};

#include "fp_ops_cc_details.h"

} // namespace arith
} // namespace gecc
