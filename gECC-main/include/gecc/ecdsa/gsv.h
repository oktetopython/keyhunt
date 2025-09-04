#pragma once

#include <iostream>
#include "../common.h"
#include "../support.h"

using namespace gecc;

#define DEFINE_ECDSA(ECDSA_Slover_NAME, EC_FIELD, BASE_FIELD, BASE_ORDER)                                    \
  using ECDSA_Slover_NAME##_ECDSAMultable =                                                      \
    gecc::ecdsa::ECDSAMultable<EC_FIELD>;                                                        \
  __device__ __constant__ ECDSA_Slover_NAME##_ECDSAMultable ECDSA_Slover_NAME##MultableConst;      \
  using ECDSA_Slover_NAME = gecc::ecdsa::ECDSASolver<BASE_FIELD, BASE_ORDER, EC_FIELD, gecc::ecdsa::constants::EC_FIELD, ECDSA_Slover_NAME##MultableConst>;

namespace gecc {
namespace ecdsa {
using namespace arith;

struct ECDSAConstant {
  u32 K;
  u64 SIG_AFF[MAX_BITS][2][1][MAX_LIMBS];
};

namespace constants {
#include "ecdsa_constants.h"
} // namespace constants

template <typename EC> struct ECDSAMultable {
  typename EC::Affine d_mul_table[EC::BaseField::BITS];
};

template <typename Order>
__global__ void kernel_inverse_naive(u32 count, typename Order::Base *sign_priv_key, typename Order::Base *sign_s) {
  u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
  Order priv_key, s;

  for (int i = 0; instance + i * gridDim.x * blockDim.x < count; i++) {
    priv_key.load_arbitrary(sign_priv_key, count, instance + i * gridDim.x * blockDim.x, 0);
    priv_key.inplace_to_montgomery();
    s = (priv_key + Order::mont_one());
    s = s.inverse();
    priv_key.store_arbitrary(sign_priv_key, count, instance + i * gridDim.x * blockDim.x, 0);
    s.store_arbitrary(sign_s, count, instance + i * gridDim.x * blockDim.x, 0);
  }
}

// no global memory
template <typename Order>
__global__ void kernel_inverse(u32 count, typename Order::Base *sign_priv_key, typename Order::Base *sign_s) {
  u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
  u32 slot_idx = Order::Layout::slot_idx();
  Order priv_key, acc_now, acc_inv;
  extern __shared__ Order inv_chain[];
  acc_now = Order::mont_one();

  priv_key.load_arbitrary(sign_priv_key, count, instance, 0);
  priv_key.inplace_to_montgomery();
  priv_key.store_arbitrary(sign_priv_key, count, instance, 0);
  inv_chain[slot_idx] = (priv_key + Order::mont_one());
  __syncthreads();
  
  u32 inv_out_offset = blockDim.x;
  u32 inv_input_offset = 0;
  for(u32 j = blockDim.x / 2; j > 0; j /= 2) {
    for(u32 t = slot_idx; t < j; t += j) {
      inv_chain[(inv_out_offset + t)] = 
            inv_chain[(inv_input_offset + t)] * 
            inv_chain[(inv_input_offset + t + j)];
    }
    inv_input_offset += j*2;
    inv_out_offset += j;
    __syncthreads();
  }

  if(slot_idx == 0) {
    inv_chain[inv_out_offset - 1] = inv_chain[inv_out_offset - 1].inverse();
  }
  __syncthreads();

  for(u32 j = 1; j < blockDim.x; j *= 2) {
    inv_out_offset -= j;
    inv_input_offset -= j*2;
    for(u32 t = slot_idx; t < j; t += j) {
      Order a = inv_chain[(inv_input_offset + t)];
      inv_chain[(inv_input_offset + t)] = 
          inv_chain[(inv_input_offset + t + j)] * 
          inv_chain[(inv_out_offset + t)];
      inv_chain[(inv_input_offset + t + j)] = 
          a * inv_chain[(inv_out_offset + t)];
    }
    __syncthreads();
  }

  inv_chain[slot_idx].store_arbitrary(sign_s, count, instance, 0);
}

// with global memmory
template <typename Order>
__global__ void kernel_inverse_1(u32 count, typename Order::Base *sign_priv_key, typename Order::Base *sign_s, typename Order::Base *acc_chain) {
  u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
  u32 slot_idx = Order::Layout::slot_idx();
  int i = 0;
  Order priv_key, s, acc_now, acc_inv;
  extern __shared__ Order inv_chain[];
  acc_now = Order::mont_one();

  for (; i + instance < count; i += gridDim.x * blockDim.x) {
    acc_now.store_arbitrary(acc_chain, count, instance + i, 0);
    priv_key.load_arbitrary(sign_priv_key, count, instance + i, 0);
    priv_key.inplace_to_montgomery();
    priv_key.store_arbitrary(sign_priv_key, count, instance + i, 0);
    s = (priv_key + Order::mont_one());
    s.store_arbitrary(sign_s, count, instance + i, 0);
    acc_now = s * acc_now;
  }
  inv_chain[slot_idx] = acc_now;
  __syncthreads();
  
  u32 inv_out_offset = blockDim.x;
  u32 inv_input_offset = 0;
  for(u32 j = blockDim.x / 2; j > 0; j /= 2) {
    for(u32 t = slot_idx; t < j; t += j) {
      inv_chain[(inv_out_offset + t)] = 
            inv_chain[(inv_input_offset + t)] * 
            inv_chain[(inv_input_offset + t + j)];
    }
    inv_input_offset += j*2;
    inv_out_offset += j;
    __syncthreads();
  }

  if(slot_idx == 0) {
    inv_chain[inv_out_offset - 1] = inv_chain[inv_out_offset - 1].inverse();
  }
  __syncthreads();

  for(u32 j = 1; j < blockDim.x; j *= 2) {
    inv_out_offset -= j;
    inv_input_offset -= j*2;
    for(u32 t = slot_idx; t < j; t += j) {
      Order a = inv_chain[(inv_input_offset + t)];
      inv_chain[(inv_input_offset + t)] = 
          inv_chain[(inv_input_offset + t + j)] * 
          inv_chain[(inv_out_offset + t)];
      inv_chain[(inv_input_offset + t + j)] = 
          a * inv_chain[(inv_out_offset + t)];
    }
    __syncthreads();
  }

  for (i -= gridDim.x * blockDim.x; i >= 0; i -= gridDim.x * blockDim.x) {
    acc_now.load_arbitrary(acc_chain, count, instance + i, 0);s = inv_chain[slot_idx] * acc_now;
    acc_now.load_arbitrary(sign_s, count, instance + i, 0);
    s.store_arbitrary(sign_s, count, instance + i, 0);inv_chain[slot_idx] = inv_chain[slot_idx] * acc_now;
  }
}

// remove some global memory from the kernel_inverse_1
template <typename Order>
__global__ void kernel_inverse_2(u32 count, typename Order::Base *sign_priv_key, typename Order::Base *sign_s, typename Order::Base *acc_chain) {
  u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
  u32 slot_idx = Order::Layout::slot_idx();
  if(instance >= count) return;
  int i = 0;
  Order priv_key, s, acc_now, acc_inv;
  extern __shared__ Order inv_chain[];

  priv_key.load_arbitrary(sign_priv_key, count, instance + i, 0);
  priv_key.inplace_to_montgomery();
  priv_key.store_arbitrary(sign_priv_key, count, instance + i, 0);
  s = (priv_key + Order::mont_one());
  s.store_arbitrary(sign_s, count, instance + i, 0);
  acc_now = s;
  for (i += gridDim.x * blockDim.x; i + instance < count; i += gridDim.x * blockDim.x) {
    acc_now.store_arbitrary(acc_chain, count, instance + i, 0);
    priv_key.load_arbitrary(sign_priv_key, count, instance + i, 0);
    priv_key.inplace_to_montgomery();
    priv_key.store_arbitrary(sign_priv_key, count, instance + i, 0);
    s = (priv_key + Order::mont_one());
    s.store_arbitrary(sign_s, count, instance + i, 0);
    acc_now = s * acc_now;
  }
  inv_chain[slot_idx] = acc_now;
  __syncthreads();
  
  u32 inv_out_offset = blockDim.x;
  u32 inv_input_offset = 0;
  for(u32 j = blockDim.x / 2; j > 0; j /= 2) {
    for(u32 t = slot_idx; t < j; t += j) {
      inv_chain[(inv_out_offset + t)] = 
            inv_chain[(inv_input_offset + t)] * 
            inv_chain[(inv_input_offset + t + j)];
    }
    inv_input_offset += j*2;
    inv_out_offset += j;
    __syncthreads();
  }

  if(slot_idx == 0) {
    inv_chain[inv_out_offset - 1] = inv_chain[inv_out_offset - 1].inverse();
  }
  __syncthreads();

  for(u32 j = 1; j < blockDim.x; j *= 2) {
    inv_out_offset -= j;
    inv_input_offset -= j*2;
    for(u32 t = slot_idx; t < j; t += j) {
      Order a = inv_chain[(inv_input_offset + t)];
      inv_chain[(inv_input_offset + t)] = 
          inv_chain[(inv_input_offset + t + j)] * 
          inv_chain[(inv_out_offset + t)];
      inv_chain[(inv_input_offset + t + j)] = 
          a * inv_chain[(inv_out_offset + t)];
    }
    __syncthreads();
  }

  for (i -= gridDim.x * blockDim.x; i >= gridDim.x * blockDim.x; i -= gridDim.x * blockDim.x) {
    acc_now.load_arbitrary(acc_chain, count, instance + i, 0);s = inv_chain[slot_idx] * acc_now;
    acc_now.load_arbitrary(sign_s, count, instance + i, 0);
    s.store_arbitrary(sign_s, count, instance + i, 0);inv_chain[slot_idx] = inv_chain[slot_idx] * acc_now;
  }
  s = inv_chain[slot_idx];
  s.store_arbitrary(sign_s, count, instance + i, 0);
}

template <typename ECDSA_Solver, typename EC, typename Field>
__global__ void kernel_batch_add_test(u32 count, 
                                      typename Field::Base *sign_k,
                                      typename EC::Base *sign_point,
                                      typename EC::Base *acc_chain) {
  u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
  u32 slot_idx = EC::Layout::slot_idx();
  Field k, diff, acc_now, acc_inv, inv;
  k.load_arbitrary(sign_k, count, instance, 0);
  extern __shared__ Field inv_chain_2[];

  typename EC::Affine p0 = ECDSA_Solver::get_d_mul_table(0);
  for (u32 i = 0; i + instance < count; i += gridDim.x * blockDim.x) {
    p0.store_arbitrary(sign_point, count, instance + i, 0);
  }

  int i = 0, index = 1;
  while (index < Field::BITS) {
    if (k.digits[index/32] & (1 << (index%32))) {
      {
        typename EC::Affine p1;
        p1.x.load_arbitrary(sign_point, count, instance + 0, 0);
        typename EC::Affine p2 = ECDSA_Solver::get_d_mul_table(index);
        diff = p2.x - p1.x;
        acc_now = diff;
        for (i += gridDim.x * blockDim.x; i + instance < count; i += gridDim.x * blockDim.x) {
          acc_now.store_arbitrary(acc_chain, count, instance + i, 0);
          p1.x.load_arbitrary(sign_point, count, instance + 0, 0);
          p2 = ECDSA_Solver::get_d_mul_table(index);
          diff = p2.x - p1.x;
          acc_now = diff * acc_now;
        }
        inv_chain_2[slot_idx] = acc_now;
        __syncthreads();
      }
      
      u32 inv_out_offset = blockDim.x;
      u32 inv_input_offset = 0;
      for(u32 j = blockDim.x / 2; j > 0; j /= 2) {
        for(u32 t = slot_idx; t < j; t += j) {
          inv_chain_2[(inv_out_offset + t)] = 
                inv_chain_2[(inv_input_offset + t)] * 
                inv_chain_2[(inv_input_offset + t + j)];
        }
        inv_input_offset += j*2;
        inv_out_offset += j;
        __syncthreads();
      }

      if(slot_idx == 0) {
        inv_chain_2[inv_out_offset - 1] = inv_chain_2[inv_out_offset - 1].inverse();
      }
      __syncthreads();

      for(u32 j = 1; j < blockDim.x; j *= 2) {
        inv_out_offset -= j;
        inv_input_offset -= j*2;
        for(u32 t = slot_idx; t < j; t += j) {
          Field a = inv_chain_2[(inv_input_offset + t)];
          inv_chain_2[(inv_input_offset + t)] = 
              inv_chain_2[(inv_input_offset + t + j)] * 
              inv_chain_2[(inv_out_offset + t)];
          inv_chain_2[(inv_input_offset + t + j)] = 
              a * inv_chain_2[(inv_out_offset + t)];
        }
        __syncthreads();
      }

      {
        for (i -= gridDim.x * blockDim.x; i >= gridDim.x * blockDim.x; i -= gridDim.x * blockDim.x) {
          acc_now.load_arbitrary(acc_chain, count, instance + i, 0);
          inv = inv_chain_2[slot_idx] * acc_now;
          typename EC::Affine p1;
          p1.load_arbitrary(sign_point, count, instance + 0, 0);
          typename EC::Affine p2 = ECDSA_Solver::get_d_mul_table(index);
          diff = p2.x - p1.x;
          typename EC::Affine res = p1.affine_add_without_inverse(p2, inv);
          inv_chain_2[slot_idx] = inv_chain_2[slot_idx] * diff;
          res.store_arbitrary(sign_point, count, instance + i, 0);
        }

        inv = inv_chain_2[slot_idx];
        typename EC::Affine p1;
        p1.load_arbitrary(sign_point, count, instance + 0, 0);
        typename EC::Affine p2 = ECDSA_Solver::get_d_mul_table(index);
        typename EC::Affine res = p1.affine_add_without_inverse(p2, inv);
        res.store_arbitrary(sign_point, count, instance + i, 0);
      }
    }
    index++;
  }
}

template <typename EC, typename Field, typename Order, typename ECDSA_Solver>
__global__ void kernel_sig_sign(u32 count, 
                                typename Order::Base *sign_e,
                                typename Order::Base *sign_priv_key,
                                typename Order::Base *sign_k,
                                typename Order::Base *sign_r,
                                typename Order::Base *sign_s) {
  int32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
  Order r, s, e, priv_key, k;
  if (instance >= count) return;
  
  k.load_arbitrary(sign_k, count, instance, 0);
  Order tmp_k = k;
  k.inplace_to_montgomery();
  EC p = EC::zero();
  ECDSA_Solver::fixed_point_mult(p, tmp_k, true);
  typename EC::Affine ap = p.get_affine_x();
  ap.x = ap.x.from_montgomery();

  e.load_arbitrary(sign_e, count, instance, 0);
  r = (e + ECDSA_Solver::FieldToOrder(ap.x));
  r.inplace_to_montgomery();
  if (r.is_zero()) {
    return ;
  }

  Order o_tmp;
  o_tmp = (r + k);
  if (o_tmp == Order::p()) {
    printf("exit 2\n");
    return ;
  }

  priv_key.load_arbitrary(sign_priv_key, count, instance, 0);
  o_tmp = priv_key * r;
  o_tmp = (k - o_tmp);
  s.load_arbitrary(sign_s, count, instance, 0);
  s = o_tmp * s;

  r = r.from_montgomery();
  s = s.from_montgomery();

  r.store_arbitrary(sign_r, count, instance, 0);
  s.store_arbitrary(sign_s, count, instance, 0);
}

template <typename EC, typename Field, typename Order, typename ECDSA_Solver>
__global__ void kernel_sig_sign_for_random_input(u32 count, 
                                typename Order::Base *sign_e,
                                typename Order::Base *sign_priv_key,
                                typename Order::Base *sign_k,
                                typename Order::Base *sign_r,
                                typename Order::Base *sign_s,
                                typename Order::Base *p_const) {
  int32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
  Order r, s, e, priv_key, k;
  if (instance >= count) return;
  
  k.load_arbitrary(sign_k, count, instance, 0);
  Order tmp_k = k;
  k.inplace_to_montgomery();
  EC p = EC::zero();
  // ECDSA_Solver::fixed_point_mult(p, tmp_k, true);
  ECDSA_Solver::fixed_point_mult_for_random_test(p, p_const, tmp_k, count, instance, true);
  typename EC::Affine ap = p.get_affine_x();
  ap.x = ap.x.from_montgomery();

  e.load_arbitrary(sign_e, count, instance, 0);
  r = (e + ECDSA_Solver::FieldToOrder(ap.x));
  r.inplace_to_montgomery();
  if (r.is_zero()) {
    return ;
  }

  Order o_tmp;
  o_tmp = (r + k);
  if (o_tmp == Order::p()) {
    printf("exit 2\n");
    return ;
  }

  priv_key.load_arbitrary(sign_priv_key, count, instance, 0);
  o_tmp = priv_key * r;
  o_tmp = (k - o_tmp);
  s.load_arbitrary(sign_s, count, instance, 0);
  s = o_tmp * s;

  r = r.from_montgomery();
  s = s.from_montgomery();

  r.store_arbitrary(sign_r, count, instance, 0);
  s.store_arbitrary(sign_s, count, instance, 0);
}

template <typename EC, typename Field, typename Order, typename ECDSA_Solver>
__global__ void kernel_sig_sign_2(u32 count, 
                                typename Order::Base *sign_e,
                                typename Order::Base *sign_priv_key,
                                typename Order::Base *sign_k,
                                typename Order::Base *sign_r,
                                typename Order::Base *sign_s,
                                typename Order::Base *sign_point) {
  int32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
  Order r, s, e, priv_key, k;
  if (instance >= count) return;
  
  k.load_arbitrary(sign_k, count, instance, 0);
  Order tmp_k = k;
  k.inplace_to_montgomery();
  typename EC::Affine p;
  p.load_arbitrary(sign_point, count, instance, 0);
  p.x = p.x.from_montgomery();

  e.load_arbitrary(sign_e, count, instance, 0);
  r = (e + ECDSA_Solver::FieldToOrder(p.x));
  r.inplace_to_montgomery();
  if (r.is_zero()) {
    return ;
  }

  Order o_tmp;
  o_tmp = (r + k);
  if (o_tmp == Order::p()) {
    printf("exit 2\n");
    return ;
  }

  priv_key.load_arbitrary(sign_priv_key, count, instance, 0);
  o_tmp = priv_key * r;
  o_tmp = (k - o_tmp);
  s.load_arbitrary(sign_s, count, instance, 0);
  s = o_tmp * s;

  r = r.from_montgomery();
  s = s.from_montgomery();

  r.store_arbitrary(sign_r, count, instance, 0);
  s.store_arbitrary(sign_s, count, instance, 0);
}

template <typename EC, typename Field, typename Order, typename ECDSA_Solver>
__global__ void kernel_sig_verify(u32 count,
                                  typename Order::Base *verify_r,
                                  typename Order::Base *verify_s,
                                  typename Order::Base *verify_e,
                                  typename Order::Base *verify_naf_point,
                                  typename Order::Base *R0,
                                  u32 *results) {
  int32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
  Order r, s, e, t;
  Field key_x, key_y, t_Field;
  if (instance >= count) return;

  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    r.load_arbitrary(verify_r, count, instance, 0);
    s.load_arbitrary(verify_s, count, instance, 0);
    e.load_arbitrary(verify_e, count, instance, 0);
  #else
    r.load(verify_r + instance * Order::LIMBS, 0, 0, 0);
    s.load(verify_s + instance * Order::LIMBS, 0, 0, 0);
    e.load(verify_e + instance * Order::LIMBS, 0, 0, 0);
  #endif
  t = (s + r);
  // S * G
  EC p1 = EC::zero();
  ECDSA_Solver::fixed_point_mult(p1, s, true);
  // p1.x.store_arbitrary(verify_e, count, instance, 0);
  typename EC::Affine ap1 = p1.get_affine_x();

  // if (instance == count - 1) {
  //   // 216A761F739F7364A5EBC0FC51410662A3E7D8821338EE55205F0CB9091495C6
  //   // 0FF6EEEAF5A5B2C93F33EAA2085EF16E6E02D1B0512445BB0E6BD5D4A9819E6B
  //   ECDSA_Solver::print_bn_device(ap1.x, Field::LIMBS);
  //   ECDSA_Solver::print_bn_device(ap1.x.from_montgomery(), Field::LIMBS);
  // }

  // t * PA
  EC p2 = EC::zero();
  typename EC::Affine ap2;
  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    ap2.load_arbitrary(verify_naf_point, count, instance, 0);
  #else
    ap2.load(verify_naf_point + instance * EC::Affine::LIMBS, 0, 0, 0);
  #endif
  // ECDSA_Solver::point_mult_naf(p2, ap2, t);
  ECDSA_Solver::point_mult_naf_rapidec(p2, ap2, t);
  ap2 = p2.get_affine_x();
  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    ap2.store_arbitrary(R0, count, instance, 0);
  #else
    ap2.store(R0 + instance * EC::Affine::LIMBS, 0, 0, 0);
  #endif

  // if (instance == count - 1) {
  //   // 88DE02A5446FE72C0761F90DC524314EC45CB305087B63D917867B3A11AFB14F
  //   // 66CFE6EEDB354E8975652B5D105E6A67026AB1B2E927B1492226178884C40AB8
  //   ECDSA_Solver::print_bn_device(ap2.x, Field::LIMBS);
  //   ECDSA_Solver::print_bn_device(ap2.x.from_montgomery(), Field::LIMBS);
  // }

  ap1 = ap1 + ap2;
  // t = r - e;
  // Field t_f = ECDSA_Solver::OrderToField(t);
  // t_f = t_f.inplace_to_montgomery();
  Field ef = ECDSA_Solver::OrderToField(e);
  Field rf = ECDSA_Solver::OrderToField(r);
  ef = ef.inplace_to_montgomery();
  rf = rf.inplace_to_montgomery();
  ef = ef + ap1.x;
   
  results[instance] = (rf == ef);
}

// for random test
template <typename EC, typename Field, typename Order, typename ECDSA_Solver>
__global__ void kernel_sig_verify_for_random_inputs(u32 count,
                                  typename Order::Base *verify_r,
                                  typename Order::Base *verify_s,
                                  typename Order::Base *verify_e,
                                  typename Order::Base *verify_naf_point,
                                  typename Order::Base *R0,
                                  u32 *results) {
  int32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
  Order r, s, e, t;
  Field key_x, key_y, t_Field;
  if (instance >= count) return;

  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    r.load_arbitrary(verify_r, count, instance, 0);
    s.load_arbitrary(verify_s, count, instance, 0);
    e.load_arbitrary(verify_e, count, instance, 0);
  #else
    r.load(verify_r + instance * Order::LIMBS, 0, 0, 0);
    s.load(verify_s + instance * Order::LIMBS, 0, 0, 0);
    e.load(verify_e + instance * Order::LIMBS, 0, 0, 0);
  #endif
  t = (s + r);
  // S * G
  EC p1 = EC::zero();
  // ECDSA_Solver::fixed_point_mult(p1, s, true);
  ECDSA_Solver::fixed_point_mult_for_random_test(p1, verify_naf_point, s, count, instance, true);
  // p1.x.store_arbitrary(verify_e, count, instance, 0);
  typename EC::Affine ap1 = p1.get_affine_x();

  // if (instance == count - 1) {
  //   // 216A761F739F7364A5EBC0FC51410662A3E7D8821338EE55205F0CB9091495C6
  //   // 0FF6EEEAF5A5B2C93F33EAA2085EF16E6E02D1B0512445BB0E6BD5D4A9819E6B
  //   ECDSA_Solver::print_bn_device(ap1.x, Field::LIMBS);
  //   ECDSA_Solver::print_bn_device(ap1.x.from_montgomery(), Field::LIMBS);
  // }

  // t * PA
  EC p2 = EC::zero();
  typename EC::Affine ap2;
  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    ap2.load_arbitrary(verify_naf_point, count, instance, 0);
  #else
    ap2.load(verify_naf_point + instance * EC::Affine::LIMBS, 0, 0, 0);
  #endif
  // ECDSA_Solver::point_mult_naf(p2, ap2, t);
  ECDSA_Solver::point_mult_naf_rapidec(p2, ap2, t);
  ap2 = p2.get_affine_x();
  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    ap2.store_arbitrary(R0, count, instance, 0);
  #else
    ap2.store(R0 + instance * EC::Affine::LIMBS, 0, 0, 0);
  #endif

  // if (instance == count - 1) {
  //   // 88DE02A5446FE72C0761F90DC524314EC45CB305087B63D917867B3A11AFB14F
  //   // 66CFE6EEDB354E8975652B5D105E6A67026AB1B2E927B1492226178884C40AB8
  //   ECDSA_Solver::print_bn_device(ap2.x, Field::LIMBS);
  //   ECDSA_Solver::print_bn_device(ap2.x.from_montgomery(), Field::LIMBS);
  // }

  ap1 = ap1 + ap2;
  // t = r - e;
  // Field t_f = ECDSA_Solver::OrderToField(t);
  // t_f = t_f.inplace_to_montgomery();
  Field ef = ECDSA_Solver::OrderToField(e);
  Field rf = ECDSA_Solver::OrderToField(r);
  ef = ef.inplace_to_montgomery();
  rf = rf.inplace_to_montgomery();
  ef = ef + ap1.x;
   
  results[instance] = (rf == ef);
}


// verify breakdown 1: fixed_point_mult (batch add) + other
template <typename EC, typename Field, typename Order, typename ECDSA_Solver>
__global__ void kernel_sig_verify_1(u32 count,
                                    typename Order::Base *verify_r,
                                    typename Order::Base *verify_s,
                                    typename Order::Base *verify_e,
                                    typename Order::Base *verify_point,
                                    typename Order::Base *verify_naf_point,
                                    typename Order::Base *R0,
                                    u32 *results) {
  u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
  Order r, s, e, t;
  typename EC::Affine ap1, ap2;
  if (instance >= count) return;

  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    r.load_arbitrary(verify_r, count, instance, 0);
    s.load_arbitrary(verify_s, count, instance, 0);
    e.load_arbitrary(verify_e, count, instance, 0);
  #else
    r.load(verify_r + instance * Order::LIMBS, 0, 0, 0);
    s.load(verify_s + instance * Order::LIMBS, 0, 0, 0);
    e.load(verify_e + instance * Order::LIMBS, 0, 0, 0);
  #endif

  t = (s + r);
  // t * PA
  EC p2 = EC::zero();

  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    ap2.load_arbitrary(verify_naf_point, count, instance, 0);
  #else
    ap2.load(verify_naf_point + instance * EC::Affine::LIMBS, 0, 0, 0);
  #endif
  // ECDSA_Solver::point_mult_naf(p2, ap2, t);
  ECDSA_Solver::point_mult_naf_rapidec(p2, ap2, t);
  ap2 = p2.get_affine_x();
  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    ap2.store_arbitrary(R0, count, instance, 0);
    ap1.load_arbitrary(verify_point, count, instance, 0); //s*G
  #else
    ap2.store(R0 + instance * EC::Affine::LIMBS, 0, 0, 0);
    ap1.load(verify_point + instance * EC::Affine::LIMBS, 0, 0, 0); //s*G
  #endif

  // if (instance == count - 1) {
  //   // 88DE02A5446FE72C0761F90DC524314EC45CB305087B63D917867B3A11AFB14F
  //   // 66CFE6EEDB354E8975652B5D105E6A67026AB1B2E927B1492226178884C40AB8
  //   ECDSA_Solver::print_bn_device(ap2.x, Field::LIMBS);
  //   ECDSA_Solver::print_bn_device(ap2.x.from_montgomery(), Field::LIMBS);
  // }
  
  ap1 = ap1 + ap2;
  // t = r - e;
  // Field t_f = ECDSA_Solver::OrderToField(t);
  // t_f = t_f.inplace_to_montgomery();
  Field ef = ECDSA_Solver::OrderToField(e);
  Field rf = ECDSA_Solver::OrderToField(r);
  ef = ef.inplace_to_montgomery();
  rf = rf.inplace_to_montgomery();
  ef = ef + ap1.x;
   
  results[instance] = (rf == ef);
}

// verify breakdown 2: all point_mult (batch add) + other
template <typename EC, typename Field, typename Order, typename ECDSA_Solver>
__global__ void kernel_sig_verify_2(u32 count,
                                    typename Order::Base *verify_r,
                                    typename Order::Base *verify_s,
                                    typename Order::Base *verify_e,
                                    typename Order::Base *verify_point,
                                    typename Order::Base *verify_naf_point,
                                    u32 *results) {
  u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
  Order r, s, e, t;
  typename EC::Affine ap1, ap2;
  if (instance >= count) return;

  for(; instance < count; instance += gridDim.x * blockDim.x) {
    r.load_arbitrary(verify_r, count, instance, 0);
    s.load_arbitrary(verify_s, count, instance, 0);
    e.load_arbitrary(verify_e, count, instance, 0);

    // S * G
    ap1.load_arbitrary(verify_point, count, instance, 0);
    // if (instance == 0) {
    //   ECDSA_Solver::print_bn_device(ap1.x, Field::LIMBS);
    //   ECDSA_Solver::print_bn_device(ap1.x.from_montgomery(), Field::LIMBS);
    // }

    // t * PA
    ap2.load_arbitrary(verify_naf_point, count, instance, 0);

    // if (instance == 0) {
    //   ECDSA_Solver::print_bn_device(ap2.x, Field::LIMBS);
    //   ECDSA_Solver::print_bn_device(ap2.x.from_montgomery(), Field::LIMBS);
    // }

    // ECDSA_Solver::point_mult_naf(p2, p2, t);
    // p1 = p1 + p2;
    // t = (r - e);
    // p1.x = p1.x * p1.x;
    // t = t * p1.x;
    // results[instance] = (t == p1.x);
    ap1 = ap1+ap2;
    Field ef = ECDSA_Solver::OrderToField(e);
    Field rf = ECDSA_Solver::OrderToField(r);
    ef = ef.inplace_to_montgomery();
    rf = rf.inplace_to_montgomery();
    ef = ef + ap1.x;
    results[instance] = (rf == ef);
  }

  
}

template <typename EC, typename Field, typename Order, typename ECDSA_Solver>
__global__ void kernel_ec_pmul_daa(u32 count,
                                  typename Order::Base *scalar,
                                  typename Order::Base *points,
                                  typename Order::Base *result,
                                  bool is_unknown_points) {
  u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
  Order s;
  if (instance >= count) return;

  for(; instance < count; instance += gridDim.x * blockDim.x) {
    #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
      s.load_arbitrary(scalar, count, instance, 0);
    #else
      s.load(scalar + instance * Order::LIMBS, 0, 0, 0);
    #endif
    EC p = EC::zero();
    if(!is_unknown_points) {
      // S * G
      // ECDSA_Solver::fixed_point_mult(p, s, true);
      ECDSA_Solver::fixed_point_mult_for_random_test(p, points, s, count, instance, true);
    } else {
      typename EC::Affine ap;
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        ap.load_arbitrary(points, count, instance, 0);
      #else
        ap.load(points + instance * EC::Affine::LIMBS, 0, 0, 0);
      #endif
      ECDSA_Solver::point_mult_naf(p, ap, s);
      // ECDSA_Solver::point_mult_naf_rapidec(p, ap, s);
    }
    
    #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
      p.x.store_arbitrary(result, count, instance, 0);
      p.y.store_arbitrary(result + instance * Field::LIMBS, count, instance, 0);
    #else
      p.x.store(result + instance * EC::Affine::LIMBS, 0, 0, 0);
      p.y.store(result + instance * EC::Affine::LIMBS + Field::LIMBS, 0, 0, 0);
    #endif
  }
}

template <typename EC, typename ECDSA_Solver>
__global__ void kernel_DataParallelMTA(typename EC::Base *sign_k_inv,
                                        typename EC::Base *sign_k,
                                        typename EC::Base *acc_chain,
                                        u32 count
                                        ) {
  using Layout = typename EC::Layout;
  const u32 lane_idx = Layout::lane_idx();
  const u32 slot_idx = Layout::slot_idx();
  const u32 gbl_t = Layout::global_slot_idx();

  typename EC::BaseField inv_chain, inv_chain_inv;
  typename EC::BaseField scalar, scalar_inv;

  // compress step
  inv_chain = EC::BaseField::mont_one();
  int32_t buc_index = gbl_t;
  for (; buc_index < count; buc_index += (gridDim.x * blockDim.x / Layout::WIDTH)) {
    scalar.load_arbitrary(sign_k, count, buc_index, lane_idx);
    inv_chain = inv_chain * scalar;
    #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
      inv_chain.store_arbitrary(acc_chain, count, buc_index, lane_idx);
    #else
      inv_chain.store(acc_chain + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
    #endif
  }

  // inv_chain.load_arbitrary(sign_k, count, buc_index, lane_idx);;
  // // __syncwarp();
  // __syncthreads();

  // inverse step
  inv_chain_inv = inv_chain.inverse();
  // inv_chain_inv = inv_chain;
  // if(slot_idx % 32 == 0)
  //   inv_chain_inv = inv_chain_inv.inverse();
  // __syncwarp();
  // __syncthreads();
  // inv_chain_inv.store(acc_chain + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
  
  // uncompress step
  u32 multiple = count / (gridDim.x * blockDim.x);
  u32 end_buc_index = multiple * (gridDim.x * blockDim.x) + gbl_t;
  if (end_buc_index >= count && multiple > 0)
    end_buc_index -= (gridDim.x * blockDim.x);
  buc_index = end_buc_index;

  for (; buc_index < count && buc_index >= 0; buc_index -= (gridDim.x * blockDim.x)) {
    if (buc_index == gbl_t)
      inv_chain = EC::BaseField::mont_one();
    else {
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        inv_chain.load_arbitrary(acc_chain, count, (buc_index - (gridDim.x * blockDim.x)), lane_idx);
      #else
        inv_chain.load(acc_chain + (buc_index - (gridDim.x * blockDim.x)) * EC::BaseField::LIMBS, 0, 0, lane_idx);
      #endif
    }
    // get the inv of scalar
    scalar_inv = inv_chain_inv * inv_chain; // scalar_inv
    scalar.load_arbitrary(sign_k, count, buc_index, lane_idx);
    // get the inv of inv_chain
    inv_chain_inv = inv_chain_inv * scalar;
    scalar_inv.store_arbitrary(sign_k_inv, count, buc_index, lane_idx);
  }
}

template <typename EC, typename ECDSA_Solver>
__global__ void kernel_GASMTA(typename EC::Base *sign_k_inv,
                                        typename EC::Base *sign_k,
                                        typename EC::Base *acc_chain,
                                        u32 count
                                        ) {
  using Layout = typename EC::Layout;
  const u32 lane_idx = Layout::lane_idx();
  const u32 slot_idx = Layout::slot_idx();
  const u32 gbl_t = Layout::global_slot_idx();

  typename EC::BaseField inv_chain_tmp, inv_chain_inv;
  typename EC::BaseField scalar, scalar_inv;
  extern __shared__ typename EC::BaseField inv_chain[]; //size 256+128+64+32;
  
  // compress step
  inv_chain_tmp = EC::BaseField::mont_one();
  int32_t buc_index = gbl_t;
  for (; buc_index < count; buc_index += (gridDim.x * blockDim.x / Layout::WIDTH)) {
    scalar.load_arbitrary(sign_k, count, buc_index, lane_idx);
    inv_chain_tmp = inv_chain_tmp * scalar;
    #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
      inv_chain_tmp.store_arbitrary(acc_chain, count, buc_index, lane_idx);
    #else
      inv_chain_tmp.store(acc_chain + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
    #endif
  }
  inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] = inv_chain_tmp;
  __syncthreads();

  // inverse step
  // get inv 256T -> 32T
  u32 inv_out_offset = blockDim.x;
  u32 inv_input_offset = 0;
  for (u32 j = blockDim.x / 2; j > 0; j /= 2) {
    for(u32 t = slot_idx; t < j; t += j) {
      inv_chain[(inv_out_offset + t) * EC::Layout::WIDTH + lane_idx] = 
            inv_chain[(inv_input_offset + t) * EC::Layout::WIDTH + lane_idx] * 
            inv_chain[(inv_input_offset + t + j) * EC::Layout::WIDTH + lane_idx];
    }
    inv_input_offset += j*2;
    inv_out_offset += j;
    __syncthreads();
  }
  if(slot_idx == 0) {
    inv_chain[(inv_out_offset - 1 + slot_idx) * EC::Layout::WIDTH + lane_idx] = inv_chain[(inv_out_offset - 1 + slot_idx) * EC::Layout::WIDTH + lane_idx].inverse();
  }
  __syncthreads();

  for( u32 j = 1; j < blockDim.x; j *= 2) {
    inv_out_offset -= j;
    inv_input_offset -= j*2;
    for(u32 t = slot_idx; t < j; t += j) {
      typename EC::BaseField a = inv_chain[(inv_input_offset + t) * EC::Layout::WIDTH + lane_idx];
      inv_chain[(inv_input_offset + t) * EC::Layout::WIDTH + lane_idx] = 
          inv_chain[(inv_input_offset + t + j) * EC::Layout::WIDTH + lane_idx] * 
          inv_chain[(inv_out_offset + t) * EC::Layout::WIDTH + lane_idx];
      inv_chain[(inv_input_offset + t + j) * EC::Layout::WIDTH + lane_idx] = 
          a * inv_chain[(inv_out_offset + t) * EC::Layout::WIDTH + lane_idx];
    }
    __syncthreads();
  }

  // uncompress step
  inv_chain_inv = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx];
  u32 multiple = count / (gridDim.x * blockDim.x);
  u32 end_buc_index = multiple * (gridDim.x * blockDim.x) + gbl_t;
  if (end_buc_index >= count && multiple > 0)
    end_buc_index -= (gridDim.x * blockDim.x);
  buc_index = end_buc_index;

  for (; buc_index < count && buc_index >= 0; buc_index -= (gridDim.x * blockDim.x)) {
    inv_chain_tmp = EC::BaseField::mont_one();
    if (buc_index > gbl_t) {
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        inv_chain_tmp.load_arbitrary(acc_chain, count, (buc_index - (gridDim.x * blockDim.x)), lane_idx);
      #else
        inv_chain_tmp.load(acc_chain + (buc_index - (gridDim.x * blockDim.x)) * EC::BaseField::LIMBS, 0, 0, lane_idx);
      #endif
    }
    // get the inv of scalar
    scalar_inv = inv_chain_inv * inv_chain_tmp; // scalar_inv
    scalar.load_arbitrary(sign_k, count, buc_index, lane_idx);
    // get the inv of inv_chain
    inv_chain_inv = inv_chain_inv * scalar;
    scalar_inv.store_arbitrary(sign_k_inv, count, buc_index, lane_idx);
  }
}

template <typename BaseField, typename BaseOrder, typename EC, const ECDSAConstant &HCONST, ECDSAMultable<EC> &ECDSACONST> struct ECDSASolver {
  using Field = BaseField;
  using Order = BaseOrder;

  using Base = typename BaseField::Base;

  __device__ __forceinline__ static Order FieldToOrder(Field &input) {
    Order result;
    for (int i = 0; i < Field::LIMBS_PER_LANE; i++) {
      result.digits[i] = input.digits[i];
    }
    return result;
  } 

  __device__ __forceinline__ static Field OrderToField(Order &input) {
    Field result;
    for (int i = 0; i < Field::LIMBS_PER_LANE; i++) {
      result.digits[i] = input.digits[i];
    }
    return result;
  } 

  __device__ __forceinline__ static void shift_right(Order &k, u32 r) {
    u32 i = 0;
    for (i = 0; i < Field::LIMBS_PER_LANE - 1; i++) {
      k.digits[i] = k.digits[i] >> r | k.digits[i+1] << (Order::Digit::Digit_WIDTH-r); 
    }
    k.digits[i] = k.digits[i] >> r;
  }

   __device__ __forceinline__ static void print_bn_device(Field x, uint32_t cnt) {
    int index;
    
    for (index = cnt - 1; index >= 0; index--) {
      printf("%08X", x.digits[index]);
    }
    printf("\n");
  }

  __device__ __forceinline__ static void print_bn_device(Order x, uint32_t cnt) {
    int index;
    
    for (index = cnt - 1; index >= 0; index--) {
      printf("%08X", x.digits[index]);
    }
    printf("\n");
  } 

  __device__ __forceinline__ static void fixed_point_mult(EC &r, Order &k, bool ec_operation = true) {
    u32 index = 0;
    typename EC::Affine q;
    typename EC::Affine r_aff = EC::Affine::zero();

    while (index < Order::BITS) {
      if (k.digits[index/32] & (1 << (index%32))) {
        q = get_d_mul_table(index);
        if (ec_operation)
          r = r + q;
        else
          r_aff = r_aff + q;
      }
      index++;
      __syncthreads();
    }

    if (!ec_operation)
      r = r_aff.to_jacobian();
  }

    __device__ __forceinline__ static void fixed_point_mult_for_random_test(EC &r, typename Order::Base *p_const, Order &k, u32 count, u32 instance, bool ec_operation = true) {
    u32 index = 0;
    typename EC::Affine q;
    typename EC::Affine r_aff = EC::Affine::zero();

    while (index < Order::BITS) {
      if (k.digits[index/32] & (1 << (index%32))) {
        // q = get_d_mul_table(index);
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
          q.load_arbitrary(p_const, count, instance, 0);
        #else
          q.load(p_const + instance * EC::Affine::LIMBS, 0, 0, 0);
        #endif
        if (ec_operation)
          r = r + q;
        else
          r_aff = r_aff + q;
      }
      index++;
      __syncthreads();
    }

    if (!ec_operation)
      r = r_aff.to_jacobian();
  }
  
  __device__ __forceinline__ static void point_mult_naf(EC &r, 
                                                        typename EC::Affine &p_aff, 
                                                        Order &t) {
    Order k;
    // int8_t naf[257];
    EC q;
    q = p_aff.to_jacobian();

    k = t;

    // int bits = 0;
    u32 index = 0;

    while (index < Order::BITS) {
      if (k.digits[index/32] & (1 << (index%32))) {
        r = r + q;
      }
      q = q + q;
      ++index;
    }

  }

  __device__ __forceinline__ static void point_mult_naf_rapidec(EC &r, 
                                                        typename EC::Affine &p_aff, 
                                                        Order &t) {
    Order k;
    int8_t naf[257];
    typename EC::Affine q1_aff;
    typename EC::Affine q2_aff;

    q1_aff = p_aff;
    q2_aff = q1_aff;
    k = t;
  #ifndef XYZZ
    r.z = Field::zero();
  #else
    r.zz = Field::zero();
    r.zzz = Field::zero();
  #endif
    q2_aff.y = (Field::zero()-q1_aff.y);
    int bits = 0;

    // if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    //   print_bn_device(k, Field::LIMBS);
    //   print_bn_device(q2_aff.y, Field::LIMBS);
    // }
    while (!k.is_zero()) {
      if ((k.digits[0] & 1) == 1) {
        shift_right(k, 1);
        if ((k.digits[0] & 1) == 1) {
          naf[bits] = -1;
          k.digits[0] += 1;
        } else {
          naf[bits] = 1;
        }
      } else {
        shift_right(k, 1);
        naf[bits] = 0;
      }
      ++bits;
    }
    // __syncthreads();

    for (int i = bits - 1; i >= 0; i--) {
      r = r + r;
      if (naf[i] == 1) {
        r = r + q1_aff;
      } else if (naf[i] == -1) {
        r = r + q2_aff;
      }
      // __syncthreads();
    }

  }

  __host__ static void initialize() {
    // include Field::initialize();
    EC::initialize();
    Order::initialize();

    std::vector<typename EC::Base> h_data(EC::Affine::SIZE << HCONST.K);
    for (u32 i = 0; i < 1 << HCONST.K; i++) {
      for (u32 t = 0; t < 2; ++t) {
        for (u32 j = 0; j < EC::BaseField::LIMBS; j++) {
          h_data[i * EC::Affine::LIMBS + t * EC::BaseField::LIMBS + j] = 
            reinterpret_cast<const typename EC::Base *>(HCONST.SIG_AFF[i][t])[j];
        }
      }
    }
    cudaMemcpyToSymbol(ECDSACONST, h_data.data(), EC::Affine::SIZE << HCONST.K);
    // Update Device Info
    cudaDeviceProp device_prop{};
    int current_device{0};
    cudaGetDevice(&current_device);
    cudaGetDeviceProperties(&device_prop, current_device);
    MAX_SM_NUMS = device_prop.multiProcessorCount;
    MAX_PersistingL2CacheSize = device_prop.persistingL2CacheMaxSize;
    accessPolicyMaxWindowSize = device_prop.accessPolicyMaxWindowSize;
    printf("GPU Type: %s, SM_COUNT: %d PersistingL2CacheMaxSize %dMB, accessPolicyMaxWindowSize %dMB\n", device_prop.name, MAX_SM_NUMS, MAX_PersistingL2CacheSize>>20, accessPolicyMaxWindowSize>>20);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) {
      printf("Initilize Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
  }

  void sign_init(const u64 *e, const u64 *priv_key, const u64 *k, u32 count) {
    sign_count = count;
    cudaMallocManaged(&sign_e, Order::SIZE * count);
    cudaMallocManaged(&sign_priv_key, Order::SIZE * count);
    cudaMallocManaged(&sign_k, Order::SIZE * count);
    cudaMallocManaged(&sign_r, Order::SIZE * count);
    cudaMallocManaged(&sign_s, Order::SIZE * count);
    cudaMallocManaged(&sign_point, EC::Affine::SIZE * count);
    cudaMallocManaged(&acc_chain, EC::BaseField::SIZE * count); 
    cudaMallocManaged(&xy_diff_list, EC::Affine::SIZE * count);
    cudaMemset(sign_point, 0, EC::Affine::SIZE * count);

#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    for (u32 j = 0; j < Order::LIMBS; j++) {
      for (u32 i = 0; i < count; i++) {
        sign_e[j * count + i] = reinterpret_cast<const Base *>(e)[j];
        sign_priv_key[j * count + i] = reinterpret_cast<const Base *>(priv_key)[j];
        sign_k[j * count + i] = reinterpret_cast<const Base *>(k)[j];
      }
    }
#else
    for (u32 i = 0; i < count; i++) {
      for (u32 j = 0; j < Order::LIMBS; j++) {
        sign_e[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(e)[j];
        sign_priv_key[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(priv_key)[j];
        sign_k[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(k)[j];
      }
    }
#endif

    cudaDeviceSynchronize();
  }

  // random init for test
  void sign_random_init(const u64 e[][MAX_LIMBS], const u64 priv_key[][MAX_LIMBS], const u64 k[][MAX_LIMBS], const u64 key_x[][MAX_LIMBS], const u64 key_y[][MAX_LIMBS], u32 count) {
    sign_count = count;
    cudaMallocManaged(&sign_e, Order::SIZE * count);
    cudaMallocManaged(&sign_priv_key, Order::SIZE * count);
    cudaMallocManaged(&sign_k, Order::SIZE * count);
    cudaMallocManaged(&sign_r, Order::SIZE * count);
    cudaMallocManaged(&sign_s, Order::SIZE * count);
    cudaMallocManaged(&sign_point, EC::Affine::SIZE * count);
    cudaMallocManaged(&fake_preprocess_points, EC::Affine::SIZE * count);
    cudaMallocManaged(&acc_chain, EC::BaseField::SIZE * count); 
    cudaMallocManaged(&xy_diff_list, EC::Affine::SIZE * count);
    cudaMemset(sign_point, 0, EC::Affine::SIZE * count);
    u32 N = 1 << 10;

#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    for (u32 j = 0; j < Order::LIMBS; j++) {
      for (u32 i = 0; i < count; i++) {
        sign_e[j * count + i] = reinterpret_cast<const Base *>(e[i%N])[j];
        sign_priv_key[j * count + i] = reinterpret_cast<const Base *>(priv_key[i%N])[j];
        sign_k[j * count + i] = reinterpret_cast<const Base *>(k[i%N])[j];
        fake_preprocess_points[j * count + i] = reinterpret_cast<const Base *>(key_x)[j];
        fake_preprocess_points[count * Order::LIMBS + j * count + i] = reinterpret_cast<const Base *>(key_y)[j];
      }
    }
#else
    for (u32 i = 0; i < count; i++) {
      for (u32 j = 0; j < Order::LIMBS; j++) {
        sign_e[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(e[i%N])[j];
        sign_priv_key[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(priv_key[i%N])[j];
        sign_k[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(k[i%N])[j];
        fake_preprocess_points[i * EC::Affine::LIMBS + j] = reinterpret_cast<const Base *>(key_x)[j];
        fake_preprocess_points[i * EC::Affine::LIMBS + Field::LIMBS + j] = reinterpret_cast<const Base *>(key_y)[j];
      }
    }
#endif
    P_CONST = fake_preprocess_points;
    #ifdef PERSISTENT_L2_CACHE
      u32 needed_bytes_pers_l2_cahce_size = count * EC::BaseField::SIZE;
      cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, max(needed_bytes_pers_l2_cahce_size, min(needed_bytes_pers_l2_cahce_size, MAX_PersistingL2CacheSize)));
      cudaStreamAttrValue stream_attribute_thrashing;
      stream_attribute_thrashing.accessPolicyWindow.base_ptr =
          reinterpret_cast<void*>(acc_chain);
      stream_attribute_thrashing.accessPolicyWindow.num_bytes =
          max(needed_bytes_pers_l2_cahce_size, min(needed_bytes_pers_l2_cahce_size, MAX_PersistingL2CacheSize));
      stream_attribute_thrashing.accessPolicyWindow.hitRatio = 1.0;
      // stream_attribute_thrashing.accessPolicyWindow.hitRatio = min(0.000001, needed_bytes_pers_l2_cahce_size / (length * EC::BaseField::SIZE));
      stream_attribute_thrashing.accessPolicyWindow.hitProp =
          cudaAccessPropertyPersisting;
      stream_attribute_thrashing.accessPolicyWindow.missProp =
          cudaAccessPropertyStreaming;
      cudaStreamSetAttribute(
          0, cudaStreamAttributeAccessPolicyWindow,
          &stream_attribute_thrashing);
      printf("Set Stream persistent L2 cache For ECDSA_Sign: %dMB (needed %d MB, MAX L2 PERS: %d MB)\n", max(needed_bytes_pers_l2_cahce_size, min(needed_bytes_pers_l2_cahce_size, MAX_PersistingL2CacheSize)) / 1024 / 1024, needed_bytes_pers_l2_cahce_size /1024/1024, MAX_PersistingL2CacheSize /1024/1024);
    #endif
    cudaDeviceSynchronize();
  }

  __device__ __forceinline__ static typename EC::Affine get_d_mul_table(u32 index) {
    typename EC::Affine p;
    p = ECDSACONST.d_mul_table[index];
    return p;
  }


  void sign_exec(u32 block_num = 480, u32 max_thread_per_block = 512, bool is_batch_opt = true) {
    u32 sharedMemSize = Order::SIZE * max_thread_per_block * 2;
    // cudaFuncSetAttribute((void *)(kernel_inverse<Order>), cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    // kernel_inverse<Order><<<block_num, max_thread_per_block, sharedMemSize>>>(sign_count, sign_priv_key, sign_s);
    // kernel_inverse_1<Order><<<block_num, max_thread_per_block, sharedMemSize>>>(sign_count, sign_priv_key, sign_s, acc_chain);

    if (is_batch_opt) {
      // breakdown-2: breakdown_1+batch_add
      kernel_inverse_2<Order><<<block_num, max_thread_per_block, sharedMemSize>>>(sign_count, sign_priv_key, sign_s, acc_chain);
      u32 sharedMemSize = Field::SIZE * max_thread_per_block * 2;
      kernel_batch_add_test<ECDSASolver, EC, Field><<<block_num, max_thread_per_block, sharedMemSize>>>(sign_count, sign_k, sign_point, acc_chain);
      kernel_sig_sign_2<EC, Field, Order, ECDSASolver><<<(sign_count + 256 - 1) / 256, 256>>>(sign_count, sign_e, sign_priv_key, sign_k, sign_r, sign_s, sign_point);
    }
    else {
      #ifdef BATCH_INV
      // breakdown-1: kernel_inverse_naive + kernel_sig_sign + batch_inv
        kernel_inverse_2<Order><<<block_num, max_thread_per_block, sharedMemSize>>>(sign_count, sign_priv_key, sign_s, acc_chain);
      #else
      // baseline: kernel_inverse_naive + kernel_sig_sign
        kernel_inverse_naive<Order><<<80, 128>>>(sign_count, sign_priv_key, sign_s);
      #endif

      kernel_sig_sign<EC, Field, Order, ECDSASolver><<<(sign_count + 256 - 1) / 256, 256>>>(sign_count, sign_e, sign_priv_key, sign_k, sign_r, sign_s);
    }
  }

  void sign_exec_for_random_inputs(u32 block_num = 480, u32 max_thread_per_block = 512, bool is_batch_opt = true) {
    u32 sharedMemSize = Order::SIZE * max_thread_per_block * 2;

    if (is_batch_opt) {
      // breakdown-2: breakdown_1+batch_add
      cudaFuncSetAttribute((void *)(kernel_inverse_2<Order>), cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
      kernel_inverse_2<Order><<<block_num, max_thread_per_block, sharedMemSize>>>(sign_count, sign_priv_key, sign_s, acc_chain);
      cudaDeviceSynchronize();
      if (cudaPeekAtLastError() != cudaSuccess) {
        printf("sign_exec batch_inv Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
      }
      u32 sharedMemSize = Field::SIZE * max_thread_per_block * 2;
      cudaFuncSetAttribute((void *)(arith::fixedPMulByCombinedDAA<EC, Field>), cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
      arith::fixedPMulByCombinedDAA<EC, Field><<<block_num, max_thread_per_block, sharedMemSize>>>(sign_point, fake_preprocess_points, sign_k, acc_chain, sign_count);
      cudaDeviceSynchronize();
      if (cudaPeekAtLastError() != cudaSuccess) {
        printf("sign_exec fixed PMUL Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
      }
      kernel_sig_sign_2<EC, Field, Order, ECDSASolver><<<(sign_count + 256 - 1) / 256, 256>>>(sign_count, sign_e, sign_priv_key, sign_k, sign_r, sign_s, sign_point);
    }
    else {
      #ifdef BATCH_INV
      // breakdown-1: kernel_inverse_naive + kernel_sig_sign + batch_inv
        cudaFuncSetAttribute((void *)(kernel_inverse<Order>), cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
        kernel_inverse_2<Order><<<block_num, max_thread_per_block, sharedMemSize>>>(sign_count, sign_priv_key, sign_s, acc_chain);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
          printf("sign_exec batch_inv Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        }
      #else
      // baseline: kernel_inverse_naive + kernel_sig_sign
        kernel_inverse_naive<Order><<<80, 128>>>(sign_count, sign_priv_key, sign_s);
      #endif

      kernel_sig_sign_for_random_input<EC, Field, Order, ECDSASolver><<<(sign_count + 256 - 1) / 256, 256>>>(sign_count, sign_e, sign_priv_key, sign_k, sign_r, sign_s, P_CONST);
    }
  }

  void sign_close() {
    cudaFree(sign_e);
    cudaFree(sign_priv_key);
    cudaFree(sign_k);
    cudaFree(sign_r);
    cudaFree(sign_s);
    cudaFree(sign_point);
    cudaFree(acc_chain);
    cudaFree(xy_diff_list);
  }


  void batch_upmul_opt(u32 blc_num, u32 thd_num, u32 sharedMemSize) {
    // fusion double-and-add alg based on batch-add opt
    cudaFuncSetAttribute((void *)(arith::scalarMulByCombinedDAA<EC, Field>), cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    arith::scalarMulByCombinedDAA<EC, Field><<<blc_num, thd_num, sharedMemSize>>>(R0, R1, verify_t, acc_chain, verify_count);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) {
      printf("batch_upmul_opt:scalarMulByCombinedDAA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
  }

  void verify_init(const u64*r, const u64*s, const u64*e, const u64*key_x, const u64*key_y, u32 count) {
    verify_count = count;
    cudaMallocManaged(&verify_r, Order::SIZE * count);
    cudaMallocManaged(&verify_s, Order::SIZE * count);
    cudaMallocManaged(&verify_e, Order::SIZE * count);
    cudaMallocManaged(&verify_t, Order::SIZE * count);
    cudaMallocManaged(&verify_key_x, Order::SIZE * count);
    cudaMallocManaged(&verify_key_y, Order::SIZE * count);
    cudaMallocManaged(&verify_point, EC::Affine::SIZE * count);
    cudaMallocManaged(&R0, EC::Affine::SIZE * count);
    cudaMallocManaged(&R1, EC::Affine::SIZE * count);
    cudaMallocManaged(&verify_naf_point, EC::Affine::SIZE * count);
    cudaMallocManaged(&acc_chain, EC::BaseField::SIZE * count); 
    cudaMallocManaged(&results, sizeof(u32) * count); 
    cudaMallocManaged(&naf_vec, sizeof(int8_t) * 257 * count);
    cudaMemset(naf_vec, 0, sizeof(int8_t) * 257 * count);
    cudaMemset(verify_point, 0, EC::Affine::SIZE * count);
    cudaMemset(verify_naf_point, 0, EC::Affine::SIZE * count);
    cudaMemset(R0, 0, EC::Affine::SIZE * count);

#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    for (u32 j = 0; j < Order::LIMBS; j++) {
      for (u32 i = 0; i < count; i++) {
        verify_r[j * count + i] = reinterpret_cast<const Base *>(r)[j];
        verify_s[j * count + i] = reinterpret_cast<const Base *>(s)[j];
        verify_e[j * count + i] = reinterpret_cast<const Base *>(e)[j];
        verify_key_x[j * count + i] = reinterpret_cast<const Base *>(key_x)[j];
        verify_key_y[j * count + i] = reinterpret_cast<const Base *>(key_y)[j];
      }
    }
#else
    for (u32 i = 0; i < count; i++) {
      for (u32 j = 0; j < Order::LIMBS; j++) {
        verify_r[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(r)[j];
        verify_s[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(s)[j];
        verify_e[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(e)[j];
        verify_key_x[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(key_x)[j];
        verify_key_y[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(key_y)[j];
      }
    }
#endif

    cudaDeviceSynchronize();
    processScalarKey<EC, Field, Order><<< (verify_count + 256 - 1) / 256, 256>>>(verify_r, verify_s, verify_t, verify_key_x, verify_key_y, R1, count);
  }

  // random key for test
  void verify_random_init(const u64 r[][MAX_LIMBS], const u64 s[][MAX_LIMBS], const u64 e[][MAX_LIMBS], const u64 key_x[][MAX_LIMBS], const u64 key_y[][MAX_LIMBS], u32 count) {
    verify_count = count;
    cudaMallocManaged(&verify_r, Order::SIZE * count);
    cudaMallocManaged(&verify_s, Order::SIZE * count);
    cudaMallocManaged(&verify_e, Order::SIZE * count);
    cudaMallocManaged(&verify_t, Order::SIZE * count);
    cudaMallocManaged(&verify_key_x, Order::SIZE * count);
    cudaMallocManaged(&verify_key_y, Order::SIZE * count);
    cudaMallocManaged(&verify_point, EC::Affine::SIZE * count);
    cudaMallocManaged(&R0, EC::Affine::SIZE * count);
    cudaMallocManaged(&R1, EC::Affine::SIZE * count);
    cudaMallocManaged(&verify_naf_point, EC::Affine::SIZE * count);
    // cudaMallocManaged(&acc_chain, EC::BaseField::SIZE * count * 2); 
    cudaMallocManaged(&acc_chain, EC::BaseField::SIZE * count); 
    cudaMallocManaged(&results, sizeof(u32) * count); 
    cudaMallocManaged(&naf_vec, sizeof(int8_t) * 257 * count);
    cudaMemset(naf_vec, 0, sizeof(int8_t) * 257 * count);
    cudaMemset(verify_point, 0, EC::Affine::SIZE * count);
    cudaMemset(verify_naf_point, 0, EC::Affine::SIZE * count);
    cudaMemset(R0, 0, EC::Affine::SIZE * count);
    u32 N = 1 << 10;


#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    for (u32 j = 0; j < Order::LIMBS; j++) {
      for (u32 i = 0; i < count; i++) {
        verify_r[j * count + i] = reinterpret_cast<const Base *>(r[i%N])[j];
        verify_s[j * count + i] = reinterpret_cast<const Base *>(s[i%N])[j];
        verify_e[j * count + i] = reinterpret_cast<const Base *>(e[i%N])[j];
        verify_key_x[j * count + i] = reinterpret_cast<const Base *>(key_x[i%N])[j];
        verify_key_y[j * count + i] = reinterpret_cast<const Base *>(key_y[i%N])[j];
      }
    }
#else
    for (u32 i = 0; i < count; i++) {
      for (u32 j = 0; j < Order::LIMBS; j++) {
        verify_r[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(r[i%N])[j];
        verify_s[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(s[i%N])[j];
        verify_e[j * count + i] = reinterpret_cast<const Base *>(e[i%N])[j];
        verify_key_x[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(key_x[i%N])[j];
        verify_key_y[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(key_y[i%N])[j];
      }
    }
#endif

    cudaDeviceSynchronize();
    processScalarKey<EC, Field, Order><<< (verify_count + 256 - 1) / 256, 256>>>(verify_r, verify_s, verify_t, verify_key_x, verify_key_y, R1, verify_count);
    P_CONST = R1;
    #ifdef PERSISTENT_L2_CACHE
      u32 needed_bytes_pers_l2_cahce_size = verify_count * EC::BaseField::SIZE;
      // needed_bytes_pers_l2_cahce_size = 2 * verify_count * EC::BaseField::SIZE;
      cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, max(needed_bytes_pers_l2_cahce_size, min(needed_bytes_pers_l2_cahce_size, MAX_PersistingL2CacheSize)));
      cudaStreamAttrValue stream_attribute_thrashing;
      stream_attribute_thrashing.accessPolicyWindow.base_ptr =
          reinterpret_cast<void*>(acc_chain);
      // stream_attribute_thrashing.accessPolicyWindow.num_bytes =
      //     min(needed_bytes_pers_l2_cahce_size, MAX_PersistingL2CacheSize);
      stream_attribute_thrashing.accessPolicyWindow.num_bytes =
          max(needed_bytes_pers_l2_cahce_size, min(needed_bytes_pers_l2_cahce_size, MAX_PersistingL2CacheSize));
      stream_attribute_thrashing.accessPolicyWindow.hitRatio = 1.0;
      stream_attribute_thrashing.accessPolicyWindow.hitProp =
          cudaAccessPropertyPersisting;
      stream_attribute_thrashing.accessPolicyWindow.missProp =
          cudaAccessPropertyStreaming;
      cudaStreamSetAttribute(
          0, cudaStreamAttributeAccessPolicyWindow,
          &stream_attribute_thrashing);
      // printf("Set Stream persistent L2 cache For ECDSA_Verify: %dMB (needed %d MB, MAX L2 PERS: %d MB)\n", min(needed_bytes_pers_l2_cahce_size, MAX_PersistingL2CacheSize) / 1024 / 1024, needed_bytes_pers_l2_cahce_size /1024/1024, MAX_PersistingL2CacheSize /1024/1024);
      printf("Set Stream persistent L2 cache For ECDSA_Verify: %dMB (needed %d MB, MAX L2 PERS: %d MB)\n", max(needed_bytes_pers_l2_cahce_size, min(needed_bytes_pers_l2_cahce_size, MAX_PersistingL2CacheSize)) / 1024 / 1024, needed_bytes_pers_l2_cahce_size /1024/1024, MAX_PersistingL2CacheSize /1024/1024);
    #endif
      cudaDeviceSynchronize();
  }

  void verify_exec(u32 block_num = 480, u32 max_thread_per_block = 512, bool is_batch_opt = true) {
    if (is_batch_opt) {
      u32 sharedMemSize = Field::SIZE * max_thread_per_block * 2;
      // verify_sP, fixed_point_mult
      kernel_batch_add_test<ECDSASolver, EC, Field><<<block_num, max_thread_per_block, sharedMemSize>>>(verify_count, verify_s, verify_point, acc_chain);
      cudaDeviceSynchronize();
      if (cudaPeekAtLastError() != cudaSuccess) {
        printf("kernal batch add Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
      }
      #ifndef VERIFY_BK_1
      {
        batch_upmul_opt(block_num, max_thread_per_block, sharedMemSize);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
          printf("kernal batch add naf Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        }

        kernel_sig_verify_2<EC, Field, Order, ECDSASolver><<<block_num, max_thread_per_block>>>(verify_count, verify_r, verify_s, verify_e, verify_point, R0, results);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
          printf("kernel sig verify Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        }
      }
      #else
      {
        kernel_sig_verify_1<EC, Field, Order, ECDSASolver><<<(verify_count+256-1)/256, 256>>>(verify_count, verify_r, verify_s, verify_e, verify_point, R1, R0, results);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
          printf("kernel sig verify Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        }
      }
      #endif
    } else {
      kernel_sig_verify<EC, Field, Order, ECDSASolver><<<(verify_count+256-1)/256, 256>>>(verify_count, verify_r, verify_s, verify_e, R1, R0, results);
      cudaDeviceSynchronize();
      if (cudaPeekAtLastError() != cudaSuccess) {
        printf("kernel sig verify ori Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
      }
    }
    
  }

  void verify_exec_for_random_input(u32 block_num = 480, u32 max_thread_per_block = 512, bool is_batch_opt = true) {
    if (is_batch_opt) {
      u32 sharedMemSize = Field::SIZE * max_thread_per_block * 2;
      // verify_sP, fixed_point_mult
      cudaFuncSetAttribute((void *)(arith::fixedPMulByCombinedDAA<EC, Field>), cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
      arith::fixedPMulByCombinedDAA<EC, Field><<<block_num, max_thread_per_block, sharedMemSize>>>(verify_point, R1, verify_s, acc_chain, verify_count);
      cudaDeviceSynchronize();
      if (cudaPeekAtLastError() != cudaSuccess) {
        printf("kernal batch add Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
      }
      #ifndef VERIFY_BK_1
      {
        // verify_tP, unfixed_point_mul
        batch_upmul_opt(block_num, max_thread_per_block, sharedMemSize);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
          printf("kernal batch add naf Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        }

        kernel_sig_verify_2<EC, Field, Order, ECDSASolver><<<block_num, max_thread_per_block>>>(verify_count, verify_r, verify_s, verify_e, verify_point, R0, results);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
          printf("kernel sig verify Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        }
      }
      #else
      {
        kernel_sig_verify_1<EC, Field, Order, ECDSASolver><<<(verify_count+256-1)/256, 256>>>(verify_count, verify_r, verify_s, verify_e, verify_point, R1, R0, results);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
          printf("kernel sig verify Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        }
      }
      #endif
    } else {
      // kernel_sig_verify<EC, Field, Order, ECDSASolver><<<(verify_count+256-1)/256, 256>>>(verify_count, verify_r, verify_s, verify_e, R1, R0, results);
      kernel_sig_verify_for_random_inputs<EC, Field, Order, ECDSASolver><<<(verify_count+256-1)/256, 256>>>(verify_count, verify_r, verify_s, verify_e, R1, R0, results);
      cudaDeviceSynchronize();
      if (cudaPeekAtLastError() != cudaSuccess) {
        printf("kernel sig verify ori Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
      }
    }
    
  }

  void verify_close() {
    cudaFree(verify_key_x);
    cudaFree(verify_key_y);
    cudaFree(verify_r);
    cudaFree(verify_s);
    cudaFree(verify_t);
    cudaFree(verify_e);
    cudaFree(verify_point);
    cudaFree(verify_naf_point);
    cudaFree(acc_chain);
    cudaFree(naf_vec);
    cudaFree(R0);
    cudaFree(R1);
  }


  // batch EC PMUL Breakdown Test
  void ec_pmul_random_init(const u64 s[][MAX_LIMBS], const u64 key_x[][MAX_LIMBS], const u64 key_y[][MAX_LIMBS], u32 count) {
    verify_count = count;
    cudaMallocManaged(&verify_s, Order::SIZE * count); //25MB
    cudaMallocManaged(&verify_t, Order::SIZE * count);
    cudaMallocManaged(&R0, EC::Affine::SIZE * count);
    cudaMallocManaged(&R1, EC::Affine::SIZE * count);
    // cudaMallocManaged(&acc_chain, EC::BaseField::SIZE * count * 2); 
    cudaMallocManaged(&acc_chain, EC::BaseField::SIZE * count*2); // *2 is for upmul bk test
    cudaMallocManaged(&lambda_n, EC::BaseField::SIZE * count*2); // *2 is for upmul bk test 
    cudaMallocManaged(&lambda_den, EC::BaseField::SIZE * count*2); // *2 is for upmul bk test
    u32 N = 3972;
    verify_key_x = R0;
    verify_key_y = R0 + EC::BaseField::LIMBS * count;

#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    for (u32 j = 0; j < Order::LIMBS; j++) {
      for (u32 i = 0; i < count; i++) {
        // verify_r[j * count + i] = reinterpret_cast<const Base *>(r[i%N])[j];
        verify_t[j * count + i] = reinterpret_cast<const Base *>(s[i%N])[j];
        // verify_e[j * count + i] = reinterpret_cast<const Base *>(e[i%N])[j];
        verify_key_x[j * count + i] = reinterpret_cast<const Base *>(key_x[i%N])[j];
        verify_key_y[j * count + i] = reinterpret_cast<const Base *>(key_y[i%N])[j];
      }
    }
#else
    for (u32 i = 0; i < count; i++) {
      for (u32 j = 0; j < Order::LIMBS; j++) {
        // verify_r[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(r[i%N])[j];
        verify_t[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(s[i%N])[j];
        // verify_e[j * count + i] = reinterpret_cast<const Base *>(e[i%N])[j];
        verify_key_x[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(key_x[i%N])[j];
        verify_key_y[i * Order::LIMBS + j] = reinterpret_cast<const Base *>(key_y[i%N])[j];
      }
    }
#endif

    cudaDeviceSynchronize();
    processScalarPoint<EC, Field, Order><<< (verify_count + 256 - 1) / 256, 256>>>(verify_t, verify_key_x, verify_key_y, R1, verify_count);
    cudaMemset(R0, 0, EC::Affine::SIZE * count);

    P_CONST = R1;
    #ifdef PERSISTENT_L2_CACHE
      // u32 needed_bytes_pers_l2_cahce_size = 25<<20; // 25MB
      u32 needed_bytes_pers_l2_cahce_size = verify_count * EC::BaseField::SIZE;
      u32 setted_pers_l2_cahce_size = max(needed_bytes_pers_l2_cahce_size, min(needed_bytes_pers_l2_cahce_size, accessPolicyMaxWindowSize));
      // needed_bytes_pers_l2_cahce_size = 2 * verify_count * EC::BaseField::SIZE;
      // cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, min(needed_bytes_pers_l2_cahce_size, accessPolicyMaxWindowSize));
      cudaStreamAttrValue stream_attribute_thrashing;
      stream_attribute_thrashing.accessPolicyWindow.base_ptr =
          reinterpret_cast<void*>(acc_chain);
          // reinterpret_cast<void*>(verify_s);
      // stream_attribute_thrashing.accessPolicyWindow.num_bytes =
      //     min(needed_bytes_pers_l2_cahce_size, MAX_PersistingL2CacheSize);
      stream_attribute_thrashing.accessPolicyWindow.num_bytes = setted_pers_l2_cahce_size;
      stream_attribute_thrashing.accessPolicyWindow.hitRatio = 1.0;
      // stream_attribute_thrashing.accessPolicyWindow.hitRatio = min(setted_pers_l2_cahce_size, MAX_PersistingL2CacheSize)*1.0/needed_bytes_pers_l2_cahce_size;
      // stream_attribute_thrashing.accessPolicyWindow.hitRatio = MAX_PersistingL2CacheSize*1.0/max(needed_bytes_pers_l2_cahce_size, MAX_PersistingL2CacheSize);
      stream_attribute_thrashing.accessPolicyWindow.hitProp =
          cudaAccessPropertyPersisting;
      stream_attribute_thrashing.accessPolicyWindow.missProp =
          cudaAccessPropertyStreaming;
      cudaStreamSetAttribute(
          0, cudaStreamAttributeAccessPolicyWindow,
          &stream_attribute_thrashing);
      // printf("Set Stream persistent L2 cache For ECDSA_EC_PMUL: %dMB (needed %d MB, MAX L2 PERS: %d MB)\n", min(needed_bytes_pers_l2_cahce_size, MAX_PersistingL2CacheSize) / 1024 / 1024, needed_bytes_pers_l2_cahce_size /1024/1024, MAX_PersistingL2CacheSize /1024/1024);
      // printf("Set Stream persistent L2 cache For ECDSA_EC_PMUL: %dMB (needed %d MB, MAX L2 PERS policy window size: %d MB), hitRatio %.2f\n", setted_pers_l2_cahce_size>>20, needed_bytes_pers_l2_cahce_size >> 20, accessPolicyMaxWindowSize >> 20, stream_attribute_thrashing.accessPolicyWindow.hitRatio);
    #endif
      cudaDeviceSynchronize();
  }

  void ecdsa_ec_pmul(u32 block_num = 480, u32 max_thread_per_block = 512, bool is_unknown_points = true) {
    u32 sharedMemSize = Field::SIZE * max_thread_per_block * 2;
    if (is_unknown_points) {
      // unknown_point_mult
      #ifdef EC_UPMUL_BASE
        kernel_ec_pmul_daa<EC, Field, Order, ECDSASolver><<<block_num, max_thread_per_block>>>(verify_count, verify_t, R1, R0, true);
      #else
        #ifdef BATCH_UPMUL_NO_OPT
          int bit_index = Field::BITS-1;
          sharedMemSize = Field::SIZE * max_thread_per_block * 4;
          cudaFuncSetAttribute((void *)(arith::UPMULDoubleAndAddWithMT<EC, Field>), cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
          arith::UPMULDoubleAndAddWithMT<EC, Field><<<block_num, max_thread_per_block, sharedMemSize>>>(R0, R1, verify_t, acc_chain, lambda_n, lambda_den, bit_index, verify_count);
        #else
          batch_upmul_opt(block_num, max_thread_per_block, sharedMemSize);
        #endif
      #endif
      cudaDeviceSynchronize();
      if (cudaPeekAtLastError() != cudaSuccess) {
        printf("unknown point pmul Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
      }
    } else {
      // fixed_point_mult
      #ifdef EC_FPMUL_BASE
        kernel_ec_pmul_daa<EC, Field, Order, ECDSASolver><<<block_num, max_thread_per_block>>>(verify_count, verify_t, R1, R0, false);
      #else
        #ifdef BATCH_FPMUL_NO_OPT
          // add breakdown-1 test 
          cudaFuncSetAttribute((void *)(arith::FPMULDoubleAndAddWithMT<EC, Field>), cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
          arith::FPMULDoubleAndAddWithMT<EC, Field><<<block_num, max_thread_per_block, sharedMemSize>>>(R0, R1, verify_t, acc_chain, lambda_n, lambda_den, verify_count);
        #else
          cudaFuncSetAttribute((void *)(arith::fixedPMulByCombinedDAA<EC, Field>), cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
          arith::fixedPMulByCombinedDAA<EC, Field><<<block_num, max_thread_per_block, sharedMemSize>>>(R0, R1, verify_t, acc_chain, verify_count);
        #endif
      #endif
      cudaDeviceSynchronize();
      if (cudaPeekAtLastError() != cudaSuccess) {
        printf("fixed point mul Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
      }
    }
  }

  void batch_modinv_MTA(u32 block_num = 480, u32 max_thread_per_block = 512) {
    // auto f_start = std::chrono::high_resolution_clock::now();
  #ifdef BATCHMINVWITHDP
    kernel_DataParallelMTA<EC, ECDSASolver><<<block_num, max_thread_per_block, 0, 0>>>(verify_s, verify_t, acc_chain, verify_count);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) {
      printf("batch modinv with MTA in data parallel Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
  #endif
    // auto f_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> f_diff = f_end - f_start;
    // printf("kernel_DataParallelMTA Kernel time: %lf ms\t\n", f_diff.count()*1000);
  #ifdef BATCHMINVWITHGAS
    u32 sharedMemSize = Field::SIZE * max_thread_per_block * 2;
    kernel_GASMTA<EC, ECDSASolver><<<block_num, max_thread_per_block, sharedMemSize, 0>>>(verify_s, verify_t, acc_chain, verify_count);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) {
      printf("batch modinv with MTA in GAS model Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
  #endif
  }

  void ec_pmul_close() {
    cudaFree(verify_s);
    cudaFree(verify_t);
    cudaFree(acc_chain);
    cudaFree(R0);
    cudaFree(R1);
  }


  Base *sign_r;         // sig->r, return value
  Base *sign_s;         // sig->s, return value
  Base *results;          // verify compare value, return value, not used
  Base *verify_point;     // fixed point mult, return value
  Base *verify_naf_point; // point mult naf, return value
  Base *R0, *fake_preprocess_points;
  Base *R1;
  Base* P_CONST;

private:
  u32 sign_count;
  u32 verify_count;

  Base *sign_e;         // digest
  Base *sign_priv_key;  // private key
  Base *sign_k;         // random number, no need to fill in
  Base *sign_point;     // store points
  Base *acc_chain, *lambda_den, *lambda_n;      // inverse acc chain
  Base *xy_diff_list;   // inverse xy diff list
  
  Base *verify_r;
  Base *verify_s;
  Base *verify_e;
  Base *verify_t;
  Base *verify_key_x;
  Base *verify_key_y;
  int8_t *naf_vec;
};

} // namespace ecdsa
} // namespace gecc