#pragma once

namespace gecc {
namespace arith {

// http://wwwhomes.uni-bielefeld.de/achim/addition_chain.html

template <typename BF, int a> struct MultiplicativeChain {};

template <typename BF> struct MultiplicativeChain<BF, 0> {
  __device__ __forceinline__ static BF multiply(BF x) {
    const BF x_ =x;
    return x_;
  }
};

template <typename BF> struct MultiplicativeChain<BF, 2> {
  __device__ __forceinline__ static BF multiply(BF x) {
#if (!defined SPPARK_32)
    return (x + x).reduce_to_pp();
#else
    return (x + x);
#endif
  }
};

template <typename BF> struct MultiplicativeChain<BF, -3> {
  __device__ __forceinline__ static BF multiply(BF x) {
#if (!defined SPPARK_32)
    const BF x2 = (x + x).reduce_to_pp();
    const BF x3 = (x2 + x).reduce_to_pp();
    return (BF::pp() - x3).reduce_to_pp();
#else
    const BF x2 = (x + x);
    const BF x3 = (x2 + x);
    return (BF::p() - x3);
#endif
  }
};

template <typename BF> struct MultiplicativeChain<BF, 13> {
  __device__ __forceinline__ static BF multiply(BF x) {
#if (!defined SPPARK_32)
    const BF x2 = (x + x).reduce_to_pp();
    const BF x4 = (x2 + x2).reduce_to_pp();
    const BF x8 = (x4 + x4).reduce_to_pp();
    const BF x9 = (x8 + x).reduce_to_pp();
    return (x9 + x4).reduce_to_pp();
#else
    const BF x2 = (x + x);
    const BF x4 = (x2 + x2);
    const BF x8 = (x4 + x4);
    const BF x9 = (x8 + x);
    return (x9 + x4);
#endif
  }
};

template <typename BF> struct MultiplicativeChain<BF, 26> {
  __device__ __forceinline__ static BF multiply(BF x) {
#if (!defined SPPARK_32)
    const BF x13 = MultiplicativeChain<BF, 13>::multiply(x);
    return (x13 + x13).reduce_to_pp();
#else
    const BF x13 = MultiplicativeChain<BF, 13>::multiply(x);
    return (x13 + x13);
#endif
  }
};

} // namespace arith
} // namespace gecc
