#pragma once

#include "../common.h"

// #include "config.h"

#include "multiplicative_chain.h"

#include <cstdio>

#define DEFINE_EC(EC_NAME, EC_TYPE, BASE_FIELD, CURVE_TYPE, DBL_FLAG)                           \
  using EC_NAME##_##CURVE_TYPE##_Factory =                                                      \
      gecc::arith::ECFactory<BASE_FIELD>;                                                       \
  __device__ __constant__ EC_NAME##_##CURVE_TYPE##_Factory EC_NAME##_##CURVE_TYPE##_DCONST;     \
  using EC_NAME##_##EC_TYPE =                                                                   \
      gecc::arith::ECPointJacobian<BASE_FIELD,                                                  \
                                   gecc::arith::constants::EC_TYPE,                             \
                                   EC_NAME##_##CURVE_TYPE##_DCONST,                             \
                                   DBL_FLAG>;                                                 \

namespace gecc {
namespace arith {

struct ECConstant {
  int a;
  u64 a_mont[MAX_LIMBS];
};

namespace constants {
#include "ec_constants.h"
} // namespace constants

template <typename BF> struct ECFactory {
  typename BF::Base a_mont[BF::LIMBS];
};

// DBL_FLAG for a != 0 a == 0 a == -3
template <typename BF, const ECConstant &HCONST, ECFactory<BF> &ECDCONST, const int DBL_FLAG> struct ECPointJacobian {
  using BaseField = BF;
  // using ECPoint = ECPoint<Factory, DCONST>;

  using Base = typename BaseField::Base;
  using Layout = typename BaseField::Layout;

  struct Affine {
    using Base = typename BaseField::Base;
    using Layout = typename BaseField::Layout;

    static const size_t LIMBS = 2 * BaseField::LIMBS;
    static const size_t SIZE = 2 * BaseField::SIZE;

    __device__ __forceinline__ void load(const typename BaseField::Base *memory,
                                         const u32 k, const u32 slot_idx,
                                         const u32 lane_idx) {
      const u32 offset = BaseField::LIMBS << k;
      x.load(memory, k, slot_idx, lane_idx);
      y.load(memory + offset, k, slot_idx, lane_idx);
    }

    __device__ __forceinline__ void load_arbitrary(typename BaseField::Base *memory,
                                                    const u32 count, const u32 slot_idx,
                                                    const u32 lane_idx) {
      const u32 offset = BaseField::LIMBS * count;
      x.load_arbitrary(memory, count, slot_idx, lane_idx);
      y.load_arbitrary(memory + offset, count, slot_idx, lane_idx);
    }
    
    __device__ __forceinline__ void store(typename BaseField::Base *memory,
                                          const u32 k, const u32 slot_idx,
                                          const u32 lane_idx) const {
      const u32 offset = BaseField::LIMBS << k;
      x.store(memory, k, slot_idx, lane_idx);
      y.store(memory + offset, k, slot_idx, lane_idx);
    }

    __device__ __forceinline__ void store_arbitrary(typename BaseField::Base *memory,
                                                    const u32 count, const u32 slot_idx,
                                                    const u32 lane_idx) {
      const u32 offset = BaseField::LIMBS * count;
      x.store_arbitrary(memory, count, slot_idx, lane_idx);
      y.store_arbitrary(memory + offset, count, slot_idx, lane_idx);
    }

    static __device__ __forceinline__ Affine zero() {
      Affine result;
      result.x = BaseField::mont_one();
      result.y = BaseField::zero();
      return result;
    }

    __device__ __forceinline__ bool is_zero() const { return y.is_zero(); }

    __device__ __forceinline__ ECPointJacobian to_nonzero_jacobian() const {
      ECPointJacobian result;
#ifdef XYZZ
      result.x = x;
      result.y = y;
      result.zz = BaseField::mont_one();
      result.zzz = BaseField::mont_one();
#else
      result.x = x;
      result.y = y;
      result.z = BaseField::mont_one();
#endif
      return result;
    }

    __device__ __forceinline__ ECPointJacobian to_jacobian() const {
      return is_zero() ? ECPointJacobian::zero() : to_nonzero_jacobian();
    }

// #ifndef XYZZ
    /*
        http://www.hyperelliptic.org/EFD/g1p/auto-shortw.html
        P(x1, y1) != Q(x2, y2)
        x3 = (y2-y1)2/(x2-x1)2-x1-x2
        y3 = (2*x1+x2)*(y2-y1)/(x2-x1)-(y2-y1)3/(x2-x1)3-y1
    */
    __device__ __forceinline__ Affine affine_add(const Affine &o) const {
#if (!defined SPPARK_32)
#include "affine_ops_add_details.h"
#else
#include "affine_ops_add_details_with_reduce.h"
#endif
    }

    /*
        http://www.hyperelliptic.org/EFD/g1p/auto-shortw.html
        P(x1, y1) == Q(x2, y2)
        m = (3 * x^2 + a)/(2*y1)
        x3 = m^2 - x1 - x2
        y3 = m * (x1 - x3) - y1
    */
    __device__ __forceinline__ Affine affine_dbl() const {
#if (!defined SPPARK_32)
#include "affine_ops_dbl_details.h"
#else
#include "affine_ops_dbl_details_with_reduce.h"
#endif
    }

    __device__ __forceinline__ void get_diff_in_place(const Affine &p1, const Affine &p2) {
      // p2-p1
      if ((p1.is_zero() || p2.is_zero())) {
        // maybe cause error, assign a special point (1,1) to the result
        this->x = BaseField::mont_one();
        this->y = BaseField::mont_one();
        return;
      }
      else if (p1.x == p2.x) {
        // 2 * y1
        this->x = p1.y + p1.y;
        // 3 * x1^2
        this->y = p1.x.square();
        BaseField gecc_optmp_1 = (this->y + this->y);
        this->y = (gecc_optmp_1 + this->y);
        BaseField a_mont = BaseField::load_const(ECDCONST.a_mont);
        this->y = (this->y + a_mont);
      }
      else {
        // with sppark library, needn't reduce
        #if (!defined SPPARK_32)
          this->x = (p2.x + BaseField::pp() - p1.x).reduce_to_pp();
          this->y = (p2.y + BaseField::pp() - p1.y).reduce_to_pp();
        #else
          this->x = p2.x - p1.x;
          this->y = p2.y - p1.y;
        #endif
      }
    }

    __device__ __forceinline__ Affine affine_add_without_inverse(const Affine &o, const BaseField inv) {
      BaseField M, MM, X3, XD, Y3, YD, t1, t2, t3;
      YD = o.y - y;
      XD = o.x - x;
      M = inv;
      M = YD * M;
      MM = M.square();
      t1 = MM - x;
      X3 = t1 - o.x;
      t2 = x - X3;
      t3 = M * t2;
      Y3 = t3 - y;
      Affine result;
      result.x = X3;
      result.y = Y3;
      return result;
    }

    __device__ __forceinline__ Affine operator+(const Affine &o) const {
      Affine result;
      if (o.is_zero()) {
        result = *this;
      } else if (is_zero()) {
        result = o;
#if (!defined SPPARK_32)
      } else if ((this->x).reduce_to_p() == o.x.reduce_to_p()){ 
        if ((this->y).reduce_to_p() == o.y.reduce_to_p()) {
          result = affine_dbl();
        }
        else if ((this->y + o.y).reduce_to_p() == BaseField::zero()) { // this->y == -o.y
          result = Affine::zero();
        }
#else
      } else if ((this->x) == o.x) { 
        if ((this->y) == o.y) {
          result = affine_dbl();
        }
        else if ((this->y + o.y) == BaseField::zero()) { // this->y == -o.y
          result = Affine::zero();
        }
#endif
        else {
          printf(" x1 == x2, but either y1 == y2 or y1 == - y2");
        }
      } else {
        result = affine_add(o);
      }
      // __syncwarp();
      return result;
    }
// #endif

    __device__ __forceinline__ bool operator==(const Affine &o) const {
      if (is_zero() && o.is_zero()) {
        return true;
      }
      return (x == o.x) && (y == o.y);
    }
    BaseField x;
    BaseField y;
  };

  static const size_t LIMBS = 3 * BaseField::LIMBS;
  static const size_t SIZE = 3 * BaseField::SIZE;

#ifdef XYZZ
  static const size_t XYZZ_LIMBS = 5 * BaseField::LIMBS;
  static const size_t XYZZ_SIZE = 5 * BaseField::SIZE;
#endif

  __host__ static void initialize() {
    BaseField::initialize();
    ECFactory<BF> c;
    BaseField::layout_dependent_copy(c.a_mont, HCONST.a_mont);
    CUDA_CHECK(cudaMemcpyToSymbol(ECDCONST, &c, sizeof(c)));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // __device__ __forceinline__ ECPointJacobian(const ECPoint &o) {
  //   x = o.x;
  //   y = o.y;
  //   z = o.is_zero() ? BaseField::zero() : BaseField::mont_one();
  // }

  __device__ __forceinline__ void load(const typename BaseField::Base *memory,
                                       const u32 k, const u32 slot_idx,
                                       const u32 lane_idx) {
    const u32 offset = BaseField::LIMBS << k;
#ifdef XYZZ
    x.load(memory, k, slot_idx, lane_idx);
    y.load(memory + offset, k, slot_idx, lane_idx);
    BaseField z;
    z.load(memory + 2 * offset, k, slot_idx, lane_idx);
    zz = (z * z).reduce_to_p();
    zzz = (z * zz).reduce_to_p();
#else
    x.load(memory, k, slot_idx, lane_idx);
    y.load(memory + offset, k, slot_idx, lane_idx);
    z.load(memory + 2 * offset, k, slot_idx, lane_idx);
#endif
  }


  __device__ __forceinline__ void load_xyzz(const typename BaseField::Base *memory,
                                       const u32 k, const u32 slot_idx,
                                       const u32 lane_idx) {
#ifdef XYZZ
    const u32 offset = BaseField::LIMBS << k;
    x.load(memory, k, slot_idx, lane_idx);
    y.load(memory + offset, k, slot_idx, lane_idx);
    // z.load(memory + 2 * offset, k, slot_idx, lane_idx);
    zz.load(memory + 3 * offset, k, slot_idx, lane_idx);
    zzz.load(memory + 4 * offset, k, slot_idx, lane_idx);
#else
    load(memory, k, slot_idx, lane_idx);
#endif
  }

  __device__ __forceinline__ void store(typename BaseField::Base *memory,
                                        const u32 k, const u32 slot_idx,
                                        const u32 lane_idx) const {
    const u32 offset = BaseField::LIMBS << k;
#ifdef XYZZ
    if (!is_zero()) {
      ECPointJacobian result;
      const BaseField inv_z2 = zz.inverse();
      const BaseField inv_z3 = zzz.inverse();
      BaseField z_one = BaseField::mont_one();
      result.x = (x * inv_z2).reduce_to_p();
      result.y = (y * inv_z3).reduce_to_p();
      result.x.store(memory, k, slot_idx, lane_idx);
      result.y.store(memory + offset, k, slot_idx, lane_idx);
      z_one.store(memory + 2 * offset, k, slot_idx, lane_idx);
    }
    else {
      BaseField z_zero = BaseField::zero();
      x.store(memory, k, slot_idx, lane_idx);
      y.store(memory + offset, k, slot_idx, lane_idx);
      z_zero.store(memory + 2 * offset, k, slot_idx, lane_idx);
    }
#else
    x.store(memory, k, slot_idx, lane_idx);
    y.store(memory + offset, k, slot_idx, lane_idx);
    z.store(memory + 2 * offset, k, slot_idx, lane_idx);
#endif
  }

#ifdef XYZZ
  __device__ __forceinline__ void store_xyzz(typename BaseField::Base *memory,
                                        const u32 k, const u32 slot_idx,
                                        const u32 lane_idx) const {
    const u32 offset = BaseField::LIMBS << k;
    x.store(memory, k, slot_idx, lane_idx);
    y.store(memory + offset, k, slot_idx, lane_idx);
    // z.store(memory + 2 * offset, k, slot_idx, lane_idx);
    zz.store(memory + 3 * offset, k, slot_idx, lane_idx);
    zzz.store(memory + 4 * offset, k, slot_idx, lane_idx);
  }
#endif

  __device__ __forceinline__ bool operator==(const ECPointJacobian &o) const {
    if (is_zero() && o.is_zero()) {
      return true;
    } else if ((is_zero() && !o.is_zero()) || (!is_zero() && o.is_zero())) {
      return false;
    }
#ifdef XYZZ
#if (!defined SPPARK_32)
    const BaseField x1z2z2 = (x * o.zz).reduce_to_p();
    const BaseField x2z1z1 = (o.x * zz).reduce_to_p();
    const BaseField y1z2z2z2 = (y * o.zzz).reduce_to_p();
    const BaseField y2z1z1z1 = (o.y * zzz).reduce_to_p();
#else
    const BaseField x1z2z2 = (x * o.zz);
    const BaseField x2z1z1 = (o.x * zz);
    const BaseField y1z2z2z2 = (y * o.zzz);
    const BaseField y2z1z1z1 = (o.y * zzz);
#endif
#else
#if (!defined SPPARK_32)
    const BaseField z1z1 = z.square().reduce_to_p();
    const BaseField z2z2 = o.z.square().reduce_to_p();
    const BaseField x1z2z2 = (x * z2z2).reduce_to_p();
    const BaseField x2z1z1 = (o.x * z1z1).reduce_to_p();
    const BaseField z1z1z1 = (z1z1 * z).reduce_to_p();
    const BaseField z2z2z2 = (z2z2 * o.z).reduce_to_p();
    const BaseField y1z2z2z2 = (y * z2z2z2).reduce_to_p();
    const BaseField y2z1z1z1 = (o.y * z1z1z1).reduce_to_p();
#else
    const BaseField z1z1 = z.square();
    const BaseField z2z2 = o.z.square();
    const BaseField x1z2z2 = (x * z2z2);
    const BaseField x2z1z1 = (o.x * z1z1);
    const BaseField z1z1z1 = (z1z1 * z);
    const BaseField z2z2z2 = (z2z2 * o.z);
    const BaseField y1z2z2z2 = (y * z2z2z2);
    const BaseField y2z1z1z1 = (o.y * z1z1z1);
#endif
#endif
    return (x1z2z2 == x2z1z1) && (y1z2z2z2 == y2z1z1z1);
  }

  __device__ __forceinline__ bool operator==(const Affine &o) const {
#ifdef XYZZ
#if (!defined SPPARK_32)
    const BaseField x2z1z1 = (o.x * zz).reduce_to_p();
    const BaseField y2z1z1z1 = (o.y * zzz).reduce_to_p();
#else
    const BaseField x2z1z1 = (o.x * zz);
    const BaseField y2z1z1z1 = (o.y * zzz);
#endif
#else
#if (!defined SPPARK_32)
    const BaseField z1z1 = z.square();
    const BaseField x2z1z1 = (o.x * z1z1).reduce_to_p();
    const BaseField z1z1z1 = z1z1 * z;
    const BaseField y2z1z1z1 = (o.y * z1z1z1).reduce_to_p();
#else
    const BaseField z1z1 = z.square();
    const BaseField x2z1z1 = (o.x * z1z1);
    const BaseField z1z1z1 = z1z1 * z;
    const BaseField y2z1z1z1 = (o.y * z1z1z1);
#endif
#endif
    return (x == x2z1z1) && (y == y2z1z1z1);
  }

#ifndef XYZZ
  __device__ __forceinline__ ECPointJacobian
  add(const ECPointJacobian &o) const {
#if (!defined SPPARK_32)
#include "ec_ops_add_details.h"
#else
#include "ec_ops_add_details_with_reduce.h"
#endif
  }

  __device__ __forceinline__ ECPointJacobian mixed_add(const Affine &o) const {
#if (!defined SPPARK_32)
#include "ec_ops_mixed_add_details.h"
#else
#include "ec_ops_mixed_add_details_with_reduce.h"
#endif
  }
#endif

  __device__ __forceinline__ ECPointJacobian
  operator+(const ECPointJacobian &o) const {
    ECPointJacobian result;
    if (o.is_zero()) {
      result = *this;
    } else if (is_zero()) {
      result = o;
    } else {
#ifndef XYZZ
      result = *this == o ? dbl() : add(o);
#else
      /*
      * http://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
      * http://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
      * with twist to handle either input at infinity. Addition costs 12M+2S,
      * while conditional doubling - 4M+6M+3S.
      */
      BaseField U, S, P, R;
      U = (x * o.zz).reduce_to_p();          /* U1 = X1*ZZ2 */
      S = (y * o.zzz).reduce_to_p();         /* S1 = Y1*ZZZ2 */
      P = (o.x * zz).reduce_to_p();          /* U2 = X2*ZZ1 */
      R = (o.y * zzz).reduce_to_p();         /* S2 = Y2*ZZZ1 */
      P = P - U;                     /* P = U2-U1 */
      R = R - S;                     /* R = S2-S1 */

      if (!P.is_zero()) {         /* X1!=X2 */
        BaseField PP;             /* add |p1| and |p2| */

        PP = P * P;               /* PP = P^2 */
#define PPP P
        PPP = P * PP;           /* PPP = P*PP */
        result.zz = zz * PP;           /* ZZ3 = ZZ1*ZZ2*PP */
        result.zzz = zzz * PPP;         /* ZZZ3 = ZZZ1*ZZZ2*PPP */
#define Q PP
        Q = U * PP;             /* Q = U1*PP */
        result.x = R * R;            /* R^2 */
        result.x = result.x - PPP;           /* R^2-PPP */
        result.x = result.x - Q;
        result.x = result.x - Q;             /* X3 = R^2-PPP-2*Q */
        Q = Q - result.x;
        Q = Q * R;                 /* R*(Q-X3) */
        result.y = S * PPP;        /* S1*PPP */
        result.y = Q - result.y;      /* Y3 = R*(Q-X3)-S1*PPP */
        result.zz = result.zz * o.zz;        /* ZZ1*ZZ2 */
        result.zzz = result.zzz * o.zzz;      /* ZZZ1*ZZZ2 */
        result.x = result.x.reduce_to_p();
        result.y = result.y.reduce_to_p();
        result.zz = result.zz.reduce_to_p();
        result.zzz = result.zzz.reduce_to_p();
#undef PPP
#undef Q
      } else if (R.is_zero()) {   /* X1==X2 && Y1==Y2 */
        BaseField M;              /* double |p1| */

        U = y + y;      /* U = 2*Y1 */
#define V P
#define W R
        V = U * U;                /* V = U^2 */
        W = U * V;              /* W = U*V */
        S = x * V;          /* S = X1*V */
        M = x * x;
        M = M + M + M;          /* M = 3*X1^2[+a*ZZ1^2] */
        result.x = M * M;
        result.x = result.x - S;
        result.x = result.x - S;             /* X3 = M^2-2*S */
        result.y = y * W;             /* W*Y1 */
        S = S - result.x;
        S = S * M;                 /* M*(S-X3) */
        result.y = S - result.y;      /* Y3 = M*(S-X3)-W*Y1 */
        result.zz = zz * V;            /* ZZ3 = V*ZZ1 */
        result.zzz = zzz * W;           /* ZZZ3 = W*ZZZ1 */
        result.x = result.x.reduce_to_p();
        result.y = result.y.reduce_to_p();
        result.zz = result.zz.reduce_to_p();
        result.zzz = result.zzz.reduce_to_p();
#undef V
#undef W
      } else {                    /* X1==X2 && Y1==-Y2 */\
        result = zero();              /* set |p3| to infinity */\
      }
#endif
    }
    return result;
  }

  __device__ __forceinline__ ECPointJacobian operator+(const Affine &o) const {
    ECPointJacobian result;
    if (o.is_zero()) {
      result = *this;
    } else if (is_zero()) {
      result = o.to_nonzero_jacobian();
    } else {
#ifndef XYZZ
      result = *this == o ? dbl() : mixed_add(o);
#else
      /*
      * http://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
      * http://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-mdbl-2008-s-1
      * with twists to handle even subtractions and either input at infinity.
      * Addition costs 8M+2S, while conditional doubling - 2M+4M+3S.
      */
      BaseField P, R;

      R = (o.y * zzz).reduce_to_p();              /* S2 = Y2*ZZZ1 */
      R = R - y;                  /* R = S2-Y1 */
      P = (o.x * zz).reduce_to_p();               /* U2 = X2*ZZ1 */
      P = P - x;                  /* P = U2-X1 */

      if (!P.is_zero()) {         /* X1!=X2 */
        BaseField PP;             /* add |p2| to |p1| */

        PP = P * P;               /* PP = P^2 */
#define PPP P
        PPP = P * PP;             /* PPP = P*PP */
        result.zz = zz * PP;           /* ZZ3 = ZZ1*PP */
        result.zzz = zzz * PPP;         /* ZZZ3 = ZZZ1*PPP */
#define Q PP
        Q = PP * x;         /* Q = X1*PP */
        result.x = R * R;            /* R^2 */
        result.x = result.x - PPP;           /* R^2-PPP */
        result.x = result.x - Q;
        result.x = result.x - Q;             /* X3 = R^2-PPP-2*Q */
        Q = Q - result.x;
        Q = Q * R;                 /* R*(Q-X3) */
        result.y = y * PPP;           /* Y1*PPP */
        result.y = Q - result.y;      /* Y3 = R*(Q-X3)-Y1*PPP */
        result.x = result.x.reduce_to_p();
        result.y = result.y.reduce_to_p();
        result.zz = result.zz.reduce_to_p();
        result.zzz = result.zzz.reduce_to_p();
#undef Q
#undef PPP
      }
      else if (R.is_zero()) {     /* X1==X2 && Y1==Y2 */
        BaseField M;              /* double |p2| */

#define U P
        U = o.y + o.y;            /* U = 2*Y1 */
        result.zz = U * U;        /* [ZZ3 =] V = U^2 */
        result.zzz = result.zz * U;     /* [ZZZ3 =] W = U*V */
#define S R
        S = o.x * result.zz;      /* S = X1*V */
        M = o.x * o.x;
        M = M + M + M;            /* M = 3*X1^2[+a] */
        result.x = M * M;
        result.x = result.x - S;
        result.x = result.x - S;  /* X3 = M^2-2*S */
        result.y = result.zzz * o.y;    /* W*Y1 */
        S = S - result.x;
        S = S * M;                /* M*(S-X3) */
        result.y = S - result.y;  /* Y3 = M*(S-X3)-W*Y1 */
        result.x = result.x.reduce_to_p();
        result.y = result.y.reduce_to_p();
        result.zz = result.zz.reduce_to_p();
        result.zzz = result.zzz.reduce_to_p();
#undef S
#undef U
      } else {                    /* X1==X2 && Y1==-Y2 */
        result = zero();              /* set |p3| to infinity */
      }
#endif
    }
    return result;
  }

  // NOTE: very costly
  __device__ __forceinline__ Affine to_affine() const {
    Affine result;
    if (is_zero()) {
      result = Affine::zero();
    } else {
#ifdef XYZZ
      const BaseField inv_z2 = zz.inverse();
      const BaseField inv_z3 = zzz.inverse();
#else
      const BaseField inv_z = z.inverse();
      const BaseField inv_z2 = inv_z.square();
      const BaseField inv_z3 = inv_z2 * inv_z;
#endif
      result.x = (x * inv_z2).reduce_to_p();
      result.y = (y * inv_z3).reduce_to_p();
    }
    // __syncwarp();
    return result;
  }

  // NOTE: very costly
  __device__ __forceinline__ Affine get_affine_x() const {
    Affine result;
    if (is_zero()) {
      result = Affine::zero();
    } else {
#ifdef XYZZ
      const BaseField inv_z2 = zz.inverse();
#else
      const BaseField inv_z = z.inverse();
      const BaseField inv_z2 = inv_z.square();
#endif
      result.x = (x * inv_z2).reduce_to_p();
    }
    // __syncwarp();
    return result;
  }

  __device__ __forceinline__ static ECPointJacobian zero() {
    ECPointJacobian result;
#ifdef XYZZ
    result.x = BaseField::mont_one();
    result.y = BaseField::mont_one();
    result.zz = BaseField::zero();
    result.zzz = BaseField::zero();
#else
    result.x = BaseField::mont_one();
    result.y = BaseField::mont_one();
    result.z = BaseField::zero();
#endif
    return result;
  }

  __device__ __forceinline__ static void set_zero(ECPointJacobian &o) {
    // TODO: not support for BLS12_377 curve 
#ifdef XYZZ
    o.x = BaseField::mont_one();
    o.y = BaseField::mont_one();
    o.zz = BaseField::zero();
    o.zzz = BaseField::zero();
#else
    o.x = BaseField::mont_one();
    o.y = BaseField::mont_one();
    o.z = BaseField::zero();
#endif
  }

  __device__ __forceinline__ bool is_zero() const { 
#ifdef XYZZ
    return zz.is_zero(); 
#else
    return z.is_zero(); 
#endif
  }

  __device__ __forceinline__ ECPointJacobian negate() const {
    ECPointJacobian result;
#ifndef XYZZ
    result.x = x;
    result.y = (BaseField::pp() - y).reduce_to_p();
    result.z = z;
#endif
#ifdef XYZZ
    result.x = x;
    result.y = (BaseField::pp() - y).reduce_to_p();
    result.zz = zz;
    result.zzz = zzz;
#endif
    return result;
  }

#ifndef XYZZ
  __device__ __forceinline__ ECPointJacobian dbl() const {
#if (!defined SPPARK_32)
    if(DBL_FLAG == 0) {
#include "ec_ops_dbl_details.h"
    }
    else if(DBL_FLAG == 1){
#include "ec_ops_dbl_1_details.h"
    }
    else if (DBL_FLAG == 2) {
#include "ec_ops_dbl_2_details.h"
    }
#else
    if(DBL_FLAG == 0) {
#include "ec_ops_dbl_details_with_reduce.h"
    }
    else if(DBL_FLAG == 1){
#include "ec_ops_dbl_1_details_with_reduce.h"
    }
    else if (DBL_FLAG == 2) {
#include "ec_ops_dbl_2_details_with_reduce.h"
    }
#endif
  }
#endif

#ifndef XYZZ
  __device__ __forceinline__ ECPointJacobian mul_2exp(u32 exp) const {
    ECPointJacobian result = *this;
    if (is_zero()) {
      return result;
    }
    for (u32 i = 0; i < exp; i++) {
      result = result.dbl();
    }
    return result;
  }
#endif

#ifndef XYZZ
  BaseField x;
  BaseField y;
  BaseField z;
#else
  BaseField x;
  BaseField y;
  BaseField zz;
  BaseField zzz;
#endif
};

} // namespace arith
} // namespace gecc
