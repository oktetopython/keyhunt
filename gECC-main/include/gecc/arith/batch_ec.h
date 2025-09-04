#pragma once

#include "ec.h"

namespace gecc{
namespace arith{

template <typename EC, typename Field, typename Order>
__global__ void processScalarKey (typename Order::Base *verify_r,
                            typename Order::Base *verify_s,
                            typename Order::Base *verify_t,
                            typename Field::Base *verify_key_x,
                            typename Field::Base *verify_key_y,
                            typename Field::Base *R1,
                            u32 count) {
  u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
  Order s, r, t;
  Field key_x, key_y;  
  typename EC::Affine ap;
  
  if (instance >= count) return;
  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    r.load_arbitrary(verify_r, count, instance, 0);
    s.load_arbitrary(verify_s, count, instance, 0);
    key_x.load_arbitrary(verify_key_x, count, instance, 0);
    key_y.load_arbitrary(verify_key_y, count, instance, 0);
  #else
    r.load(verify_r + instance * Order::LIMBS, 0, 0, 0);
    s.load(verify_s + instance * Order::LIMBS, 0, 0, 0);
    key_x.load(verify_key_x + instance * EC::BaseField::LIMBS, 0, 0, 0);
    key_y.load(verify_key_y + instance * EC::BaseField::LIMBS, 0, 0, 0);
  #endif
  t = r + s;

  key_x = key_x.inplace_to_montgomery();
  key_y = key_y.inplace_to_montgomery();

  ap.x = key_x;
  ap.y = key_y;

  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    t.store_arbitrary(verify_t, count, instance, 0);
    key_x.store_arbitrary(verify_key_x, count, instance, 0);
    key_y.store_arbitrary(verify_key_y, count, instance, 0);
    ap.store_arbitrary(R1, count, instance, 0);
  #else
    t.store(verify_t + instance * Order::LIMBS, 0, 0, 0);
    key_x.store(verify_key_x + instance * EC::BaseField::LIMBS, 0, 0, 0);
    key_y.store(verify_key_y + instance * EC::BaseField::LIMBS, 0, 0, 0);
    ap.store(R1 + instance * EC::Affine::LIMBS, 0, 0, 0);
  #endif
}

template <typename EC, typename Field, typename Order>
__global__ void processScalarPoint (typename Order::Base *verify_s,
                            typename Field::Base *verify_key_x,
                            typename Field::Base *verify_key_y,
                            typename Field::Base *R1,
                            u32 count) {
  u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
  Order s;
  Field key_x, key_y;  
  typename EC::Affine ap;
  
  if (instance >= count) return;
  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    // s.load_arbitrary(verify_s, count, instance, 0);
    key_x.load_arbitrary(verify_key_x, count, instance, 0);
    key_y.load_arbitrary(verify_key_y, count, instance, 0);
  #else
    // s.load(verify_s + instance * Order::LIMBS, 0, 0, 0);
    key_x.load(verify_key_x + instance * EC::BaseField::LIMBS, 0, 0, 0);
    key_y.load(verify_key_y + instance * EC::BaseField::LIMBS, 0, 0, 0);
  #endif

  key_x = key_x.inplace_to_montgomery();
  key_y = key_y.inplace_to_montgomery();

  ap.x = key_x;
  ap.y = key_y;

  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
    // key_x.store_arbitrary(verify_key_x, count, instance, 0);
    // key_y.store_arbitrary(verify_key_y, count, instance, 0);
    ap.store_arbitrary(R1, count, instance, 0);
  #else
    // key_x.store(verify_key_x + instance * EC::BaseField::LIMBS, 0, 0, 0);
    // key_y.store(verify_key_y + instance * EC::BaseField::LIMBS, 0, 0, 0);
    ap.store(R1 + instance * EC::Affine::LIMBS, 0, 0, 0);
  #endif
}


template <typename EC, typename Fr>
__global__ void scalarMulByCombinedDAA( typename EC::Base *R0,
                                              typename EC::Base *R1,
                                              typename EC::Base *verify_t,
                                              typename EC::Base *acc_chain,
                                              u32 count) {
  using Layout = typename EC::Layout;
  const u32 lane_idx = Layout::lane_idx();
  const u32 slot_idx = Layout::slot_idx();
  const u32 gbl_t = Layout::global_slot_idx();

  extern __shared__ typename EC::BaseField inv_chain[]; //size 256+128+64+32;
  typename EC::Affine p1, p2, diff, res;
  typename EC::BaseField diff_x_inv, inv_chain_tmp;
  typename EC::BaseField M, MM, X3, Y3, t1, t2, t3;
  typename Fr::Base scalar;

  int32_t buc_index = gbl_t;
  for(int bit_index = 0; bit_index < Fr::BITS; bit_index++) {
    // ForwardTraversal
    inv_chain_tmp = EC::BaseField::mont_one();
    buc_index = gbl_t;
    for(; buc_index < count; buc_index += gridDim.x * blockDim.x) {
      // for pdbl diff
      // read the y-axis of point vector
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        p2.y.load_arbitrary(R1 + count * EC::BaseField::LIMBS, count, buc_index, lane_idx);
      #else
        p2.y.load(R1 + buc_index * EC::Affine::LIMBS + EC::BaseField::LIMBS, 0, 0, lane_idx);
      #endif
      
      // calculate the double diff_x
      diff_x_inv = p2.y + p2.y;
      if(p2.y.is_zero()) // bugs: x is zero doesn't get point is zero-point
        diff_x_inv = EC::BaseField::mont_one();
        
      inv_chain_tmp = inv_chain_tmp * diff_x_inv;
      
      // for padd diff
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        p1.x.load_arbitrary(R0, count, buc_index, lane_idx);
        p2.x.load_arbitrary(R1, count, buc_index, lane_idx);
      #else
        p1.x.load(R0 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
        p2.x.load(R1 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
      #endif
      diff_x_inv = p2.x - p1.x;
      if(p1.x.is_zero() || p2.x.is_zero()) // bugs: x is zero doesn't get point is zero-point
        diff_x_inv = EC::BaseField::mont_one();
      inv_chain_tmp = inv_chain_tmp * diff_x_inv;
      
      // 通过重算来获取dbl的diff 累乘结果
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        inv_chain_tmp.store_arbitrary(acc_chain, count, buc_index, lane_idx);
      #else
        inv_chain_tmp.store(acc_chain + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
      #endif
    }
    inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] = inv_chain_tmp;
    __syncthreads();

    // get inv 256T -> 32T
    u32 inv_out_offset = blockDim.x;
    u32 inv_input_offset = 0;
    // /*
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

    // BackwardTraversal
    buc_index -= (gridDim.x*blockDim.x);
    for(; buc_index < count && buc_index >= 0; buc_index -= (gridDim.x*blockDim.x)) {
      inv_chain_tmp = EC::BaseField::mont_one();
      // for padd      
      {
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
          p2.y.load_arbitrary(R1 + count * EC::BaseField::LIMBS, count, buc_index, lane_idx);
        #else
          p2.y.load(R1 + buc_index * EC::Affine::LIMBS + EC::BaseField::LIMBS, 0, 0, lane_idx);
        #endif
        diff_x_inv = p2.y + p2.y;   
        if(buc_index > gbl_t) {
          #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
            inv_chain_tmp.load_arbitrary(acc_chain, count, (buc_index - (gridDim.x*blockDim.x)), lane_idx);
          #else
            inv_chain_tmp.load(acc_chain + (buc_index - (gridDim.x*blockDim.x)) * EC::BaseField::LIMBS, 0, 0, lane_idx);
          #endif
        }
        diff_x_inv = inv_chain_tmp * diff_x_inv;
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
          p2.x.load_arbitrary(R1, count, buc_index, lane_idx);
        #else
          p2.x.load(R1 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
        #endif
        diff_x_inv = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] * diff_x_inv; // inplace inverse
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
          p1.load_arbitrary(R0, count, buc_index, lane_idx);
        #else
          p1.load(R0 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
        #endif
        diff.get_diff_in_place(p1, p2);

        M = diff.y * diff_x_inv;
        MM = M.square();
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
          scalar = verify_t[((bit_index / Fr::Digit::BITS) * count) + buc_index];
        #else
          scalar = verify_t[buc_index * Fr::LIMBS + (bit_index / Fr::Digit::BITS)];
        #endif
        t1 = p2.x + p1.x;
        X3 = MM - t1;
        t2 = p1.x - X3;
        t3 = M * t2;
        Y3 = t3 - p1.y;
        res.x = X3.reduce_to_p();
        res.y = Y3.reduce_to_p();

        if (p1.is_zero()) 
          res = p2;
        else if (p2.is_zero())
          res = p1;
        if(scalar & (1 << (bit_index % 32))) {
          #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS  
            res.store_arbitrary(R0, count, buc_index, lane_idx);
          #else
            res.store(R0 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
          #endif
        }
        inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] * diff.x; // inplace inverse
      }
      
      {
        diff_x_inv = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] * inv_chain_tmp; // inplace inverse
        diff.get_diff_in_place(p2, p2);
        M = diff.y * diff_x_inv;
        MM = M.square();
        t1 = p2.x + p2.x;
        X3 = MM - t1;
        t2 = p2.x - X3;
        t3 = M * t2;
        Y3 = t3 - p2.y;
        res.x = X3.reduce_to_p();
        res.y = Y3.reduce_to_p();
        if(p2.is_zero()) {
          res = p2;
        }
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS 
          res.store_arbitrary(R1, count, buc_index, lane_idx);
        #else
          res.store(R1 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
        #endif
        // pdbl end
        inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] * diff.x;
      }      
    }
    __syncthreads();
  }

}

template <typename EC, typename Fr>
__global__ void fixedPMulByCombinedDAA( typename EC::Base *R0,
                                              typename EC::Base *R1,
                                              typename EC::Base *verify_t,
                                              typename EC::Base *acc_chain,
                                              u32 count) {
  using Layout = typename EC::Layout;
  const u32 lane_idx = Layout::lane_idx();
  const u32 slot_idx = Layout::slot_idx();
  const u32 gbl_t = Layout::global_slot_idx();

  extern __shared__ typename EC::BaseField inv_chain[]; //size 256+128+64+32;
  typename EC::Affine p1, p2, diff, res;
  typename EC::BaseField diff_x_inv, inv_chain_tmp;
  typename EC::BaseField M, MM, X3, Y3, t1, t2, t3;
  typename Fr::Base scalar;

  int32_t buc_index = gbl_t;
  for(int bit_index = 0; bit_index < Fr::BITS; bit_index++) {
    // ForwardTraversal
    inv_chain_tmp = EC::BaseField::mont_one();
    buc_index = gbl_t;
    for(; buc_index < count; buc_index += gridDim.x * blockDim.x) {      
      // for padd diff
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        p1.x.load_arbitrary(R0, count, buc_index, lane_idx);
        p2.x.load_arbitrary(R1, count, buc_index, lane_idx);
      #else
        p1.x.load(R0 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
        p2.x.load(R1 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
      #endif
      diff_x_inv = p2.x + p1.x;
      if(p1.x.is_zero() || p2.x.is_zero()) // bugs: x is zero doesn't get point is zero-point
        diff_x_inv = EC::BaseField::mont_one();
      inv_chain_tmp = inv_chain_tmp * diff_x_inv;
      
      // 通过重算来获取dbl的diff 累乘结果
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        inv_chain_tmp.store_arbitrary(acc_chain, count, buc_index, lane_idx);
      #else
        inv_chain_tmp.store(acc_chain + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
      #endif
    }
    inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] = inv_chain_tmp;
    __syncthreads();

    // get inv 256T -> 32T
    u32 inv_out_offset = blockDim.x;
    u32 inv_input_offset = 0;
    //
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

    // BackwardTraversal
    buc_index -= (gridDim.x*blockDim.x);
    for(; buc_index < count && buc_index >= 0; buc_index -= (gridDim.x*blockDim.x)) {
      inv_chain_tmp = EC::BaseField::mont_one();
      // for padd      
      {
        // #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        //   p2.y.load_arbitrary(R1 + count * EC::BaseField::LIMBS, count, buc_index, lane_idx);
        // #else
        //   p2.y.load(R1 + buc_index * EC::Affine::LIMBS + EC::BaseField::LIMBS, 0, 0, lane_idx);
        // #endif
  
        if(buc_index > gbl_t) {
          #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
            inv_chain_tmp.load_arbitrary(acc_chain, count, (buc_index - (gridDim.x*blockDim.x)), lane_idx);
          #else
            inv_chain_tmp.load(acc_chain + (buc_index - (gridDim.x*blockDim.x)) * EC::BaseField::LIMBS, 0, 0, lane_idx);
          #endif
        }
        diff_x_inv = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] * inv_chain_tmp; // inplace inverse
          
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
          p1.load_arbitrary(R0, count, buc_index, lane_idx);
          p2.load_arbitrary(R1, count, buc_index, lane_idx);
        #else
          p1.load(R0 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
          p2.load(R1 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
        #endif
        diff.get_diff_in_place(p1, p2);

        M = diff.y * diff_x_inv;
        MM = M.square();
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
          scalar = verify_t[((bit_index / Fr::Digit::BITS) * count) + buc_index];
        #else
          scalar = verify_t[buc_index * Fr::LIMBS + (bit_index / Fr::Digit::BITS)];
        #endif
        t1 = p2.x + p1.x;
        X3 = MM - t1;
        t2 = p1.x - X3;
        t3 = M * t2;
        Y3 = t3 - p1.y;
        res.x = X3.reduce_to_p();
        res.y = Y3.reduce_to_p();

        if (p1.is_zero()) 
          res = p2;
        else if (p2.is_zero())
          res = p1;
        if(scalar & (1 << (bit_index % 32))) {
          #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS  
            res.store_arbitrary(R0, count, buc_index, lane_idx);
          #else
            res.store(R0 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
          #endif
        }
        inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] * diff.x; // inplace inverse
      }
    }
  }

}

template <typename EC, typename Fr>
__global__ void FPMULDoubleAndAddWithMT( typename EC::Base *R0,
                                              typename EC::Base *R1,
                                              typename EC::Base *verify_t,
                                              typename EC::Base *acc_chain,
                                              typename EC::Base *lambda_n,
                                              typename EC::Base *lambda_den,
                                              u32 count) {
  using Layout = typename EC::Layout;
  const u32 lane_idx = Layout::lane_idx();
  const u32 slot_idx = Layout::slot_idx();
  const u32 gbl_t = Layout::global_slot_idx();

  extern __shared__ typename EC::BaseField inv_chain[]; //size 256+128+64+32;
  typename EC::Affine p1, p2, diff, res;
  typename EC::BaseField diff_x_inv, inv_chain_tmp;
  typename EC::BaseField M, MM, X3, Y3, t1, t2, t3;
  typename Fr::Base scalar;

  int32_t buc_index = gbl_t;
  for(int bit_index = 0; bit_index < Fr::BITS; bit_index++) {
    // ForwardTraversal
    inv_chain_tmp = EC::BaseField::mont_one();
    buc_index = gbl_t;
    for(; buc_index < count; buc_index += gridDim.x * blockDim.x) {      
      // for padd diff
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        p1.load_arbitrary(R0, count, buc_index, lane_idx);
        p2.load_arbitrary(R1, count, buc_index, lane_idx);
      #else
        p1.load(R0 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
        p2.load(R1 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
      #endif
      
      diff_x_inv = p2.x + p1.x;
      if(p1.x.is_zero() || p2.x.is_zero()) // bugs: x is zero doesn't get point is zero-point
        diff_x_inv = EC::BaseField::mont_one();
      inv_chain_tmp = inv_chain_tmp * diff_x_inv;
      
      
      // 通过重算来获取dbl的diff 累乘结果
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        inv_chain_tmp.store_arbitrary(acc_chain, count, buc_index, lane_idx);
        diff_x_inv.store_arbitrary(lambda_den, count, buc_index, lane_idx);
        diff_x_inv.store_arbitrary(lambda_n, count, buc_index, lane_idx);
      #else
        inv_chain_tmp.store(acc_chain + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
        diff_x_inv.store(lambda_den + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
        diff_x_inv.store(lambda_n + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
      #endif
    }
    inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] = inv_chain_tmp;
    __syncthreads();

    // get inv 256T -> 32T
    u32 inv_out_offset = blockDim.x;
    u32 inv_input_offset = 0;
    //
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

    // BackwardTraversal
    buc_index -= (gridDim.x*blockDim.x);
    for(; buc_index < count && buc_index >= 0; buc_index -= (gridDim.x*blockDim.x)) {
      inv_chain_tmp = EC::BaseField::mont_one();
      // for padd      
      {
        // #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        //   p2.y.load_arbitrary(R1 + count * EC::BaseField::LIMBS, count, buc_index, lane_idx);
        // #else
        //   p2.y.load(R1 + buc_index * EC::Affine::LIMBS + EC::BaseField::LIMBS, 0, 0, lane_idx);
        // #endif
  
        if(buc_index > gbl_t) {
          #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
            inv_chain_tmp.load_arbitrary(acc_chain, count, (buc_index - (gridDim.x*blockDim.x)), lane_idx);
            diff.y.load_arbitrary(lambda_n, count, (buc_index - (gridDim.x*blockDim.x)), lane_idx);
            diff.x.load_arbitrary(lambda_den, count, (buc_index - (gridDim.x*blockDim.x)), lane_idx);
          #else
            inv_chain_tmp.load(acc_chain + (buc_index - (gridDim.x*blockDim.x)) * EC::BaseField::LIMBS, 0, 0, lane_idx);
            diff.y.load(lambda_n + (buc_index - (gridDim.x*blockDim.x)) * EC::BaseField::LIMBS, 0, 0, lane_idx);
            diff.x.load(lambda_den + (buc_index - (gridDim.x*blockDim.x)) * EC::BaseField::LIMBS, 0, 0, lane_idx);
          #endif
        }
        diff_x_inv = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] * inv_chain_tmp; // inplace inverse
          
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
          p1.load_arbitrary(R0, count, buc_index, lane_idx);
          p2.load_arbitrary(R1, count, buc_index, lane_idx);
        #else
          p1.load(R0 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
          p2.load(R1 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
        #endif
        // diff.get_diff_in_place(p1, p2);

        M = diff.y * diff_x_inv;
        MM = M.square();
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
          scalar = verify_t[((bit_index / Fr::Digit::BITS) * count) + buc_index];
        #else
          scalar = verify_t[buc_index * Fr::LIMBS + (bit_index / Fr::Digit::BITS)];
        #endif
        t1 = p2.x + p1.x;
        X3 = MM - t1;
        t2 = p1.x - X3;
        t3 = M * t2;
        Y3 = t3 - p1.y;
        res.x = X3.reduce_to_p();
        res.y = Y3.reduce_to_p();

        if (p1.is_zero()) 
          res = p2;
        else if (p2.is_zero())
          res = p1;
        if(scalar & (1 << (bit_index % 32))) {
          #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS  
            res.store_arbitrary(R0, count, buc_index, lane_idx);
          #else
            res.store(R0 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
          #endif
        }
        inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] * diff.x; // inplace inverse
      }
    }
  }

}

// inv kernel fusion
template <typename EC, typename Field>
__global__ void UPMULDoubleAndAddWithMT( typename EC::Base *R0,
                                              typename EC::Base *R1,
                                              typename EC::Base *verify_t,
                                              typename EC::Base *acc_chain,
                                              typename EC::Base *lambda_n,
                                              typename EC::Base *lambda_den,
                                              int bit_index,
                                              u32 count) {
  using Layout = typename EC::Layout;
  const u32 lane_idx = Layout::lane_idx();
  const u32 slot_idx = Layout::slot_idx();
  const u32 gbl_t = Layout::global_slot_idx();

  extern __shared__ typename EC::BaseField inv_chain[]; //size 256+128+64+32;
  typename EC::BaseField diff_x, dbl_y, inv_chain_tmp_add, inv_chain_tmp_dbl;
  typename EC::Affine p1, p2, diff_add, diff_dbl, res_add, res_dbl;
  typename Field::Base scalar;

  // ForwardTraversal
  inv_chain_tmp_add = EC::BaseField::mont_one();
  inv_chain_tmp_dbl = EC::BaseField::mont_one();
  for(; bit_index >= 0; bit_index--) {
    int32_t buc_index = gbl_t;
    for(; buc_index < count; buc_index += gridDim.x * blockDim.x) {
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        scalar = verify_t[((bit_index / Field::Digit::BITS) * count) + buc_index];
      #else
        scalar = verify_t[buc_index * Field::LIMBS + (bit_index / Field::Digit::BITS)];
      #endif
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        if (scalar & (1 << (bit_index % 32))) {
          p1.load_arbitrary(R0, count, buc_index, lane_idx);
          p2.load_arbitrary(R1, count, buc_index, lane_idx);
        } else {
          p1.load_arbitrary(R1, count, buc_index, lane_idx);
          p2.load_arbitrary(R0, count, buc_index, lane_idx);
        }
      #else
        if (scalar & (1 << (bit_index % 32))) {
          p1.load(R0 + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
          p2.load(R1 + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
        } else {
          p1.load(R1 + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
          p2.load(R0 + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
        }
      #endif
      
      // if (p1.x.is_zero() || p2.x.is_zero())
      //   diff_x = EC::BaseField::mont_one();
      // else
        diff_x = p2.x - p1.x;

      if (p2.y.is_zero())
        dbl_y = EC::BaseField::mont_one();
      else
        dbl_y = p2.y + p2.y;

      inv_chain_tmp_add = inv_chain_tmp_add * diff_x;
      inv_chain_tmp_dbl = inv_chain_tmp_dbl * dbl_y;

      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        inv_chain_tmp_add.store_arbitrary(acc_chain , count, buc_index, lane_idx);
        diff_x.store_arbitrary(lambda_n , count, buc_index, lane_idx);
        diff_x.store_arbitrary(lambda_den , count, buc_index, lane_idx);
        inv_chain_tmp_dbl.store_arbitrary(acc_chain + (EC::BaseField::LIMBS * count), count, buc_index, lane_idx);
        dbl_y.store_arbitrary(lambda_n + (EC::BaseField::LIMBS * count), count, buc_index, lane_idx);
        dbl_y.store_arbitrary(lambda_den + (EC::BaseField::LIMBS * count), count, buc_index, lane_idx);
      #else
        inv_chain_tmp_add.store(acc_chain + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
        diff_x.store(lambda_n + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
        diff_x.store(lambda_den + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
        inv_chain_tmp_dbl.store(acc_chain + buc_index * EC::BaseField::LIMBS + EC::BaseField::LIMBS, 0, 0, lane_idx);
        dbl_y.store(lambda_n + buc_index * EC::BaseField::LIMBS + EC::BaseField::LIMBS, 0, 0, lane_idx);
        dbl_y.store(lambda_den + buc_index * EC::BaseField::LIMBS + EC::BaseField::LIMBS, 0, 0, lane_idx);
      #endif
    }
    inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] = inv_chain_tmp_add;
    inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx + blockDim.x] = inv_chain_tmp_dbl;
    __syncthreads();

    // get inv 256T -> 32T
    u32 inv_out_offset = blockDim.x * 2;
    u32 inv_input_offset = 0;
    for (u32 j = blockDim.x; j > 0; j /= 2) {
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

    for( u32 j = 1; j < blockDim.x * 2; j *= 2) {
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
    
    
    // BackwardTraversal
    typename EC::BaseField diff_x_inv, dbl_y_inv;
    typename EC::BaseField M, MM, X3, Y3, t1, t2, t3;
    buc_index -= (gridDim.x*blockDim.x);
    for(; buc_index < count && buc_index >= 0; buc_index -= (gridDim.x*blockDim.x)) {
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        scalar = verify_t[((bit_index / Field::Digit::BITS) * count) + buc_index];
      #else
        scalar = verify_t[buc_index * Field::LIMBS + (bit_index / Field::Digit::BITS)];
      #endif

      inv_chain_tmp_add = EC::BaseField::mont_one();
      inv_chain_tmp_dbl = EC::BaseField::mont_one();
      if(buc_index > gbl_t) {
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
          inv_chain_tmp_add.load_arbitrary(acc_chain, count, (buc_index - (gridDim.x*blockDim.x)), lane_idx);
          diff_add.x.load_arbitrary(lambda_den, count, (buc_index - (gridDim.x*blockDim.x)), lane_idx);
          diff_add.y.load_arbitrary(lambda_n, count, (buc_index - (gridDim.x*blockDim.x)), lane_idx);
          inv_chain_tmp_dbl.load_arbitrary(acc_chain + (EC::BaseField::LIMBS * count), count, (buc_index - (gridDim.x*blockDim.x)), lane_idx);
          diff_dbl.x.load_arbitrary(lambda_den + (EC::BaseField::LIMBS * count), count, (buc_index - (gridDim.x*blockDim.x)), lane_idx);
          diff_dbl.y.load_arbitrary(lambda_n + (EC::BaseField::LIMBS * count), count, (buc_index - (gridDim.x*blockDim.x)), lane_idx);
          
        #else
          inv_chain_tmp_add.load(acc_chain + (buc_index - (gridDim.x*blockDim.x)) * EC::BaseField::LIMBS, 0, 0, lane_idx);
          diff_add.x.load(lambda_den + (buc_index - (gridDim.x*blockDim.x)) * EC::BaseField::LIMBS, 0, 0, lane_idx);
          diff_add.y.load(lambda_n + (buc_index - (gridDim.x*blockDim.x)) * EC::BaseField::LIMBS, 0, 0, lane_idx);
          inv_chain_tmp_dbl.load(acc_chain + (buc_index - (gridDim.x*blockDim.x)) * EC::BaseField::LIMBS + EC::BaseField::LIMBS, 0, 0, lane_idx);
          diff_dbl.x.load(lambda_den + (buc_index - (gridDim.x*blockDim.x)) * EC::BaseField::LIMBS + EC::BaseField::LIMBS, 0, 0, lane_idx);
          diff_dbl.y.load(lambda_n + (buc_index - (gridDim.x*blockDim.x)) * EC::BaseField::LIMBS + EC::BaseField::LIMBS, 0, 0, lane_idx);
        #endif
      }
      diff_x_inv = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] * inv_chain_tmp_add; // inplace inverse
      dbl_y_inv = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx + blockDim.x] * inv_chain_tmp_dbl; // inplace inverse
    
      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        if (scalar & (1 << (bit_index % 32))) {
          p1.load_arbitrary(R0, count, buc_index, lane_idx);
          p2.load_arbitrary(R1, count, buc_index, lane_idx);
        } else {
          p1.load_arbitrary(R1, count, buc_index, lane_idx);
          p2.load_arbitrary(R0, count, buc_index, lane_idx);
        }
      #else
        if (scalar & (1 << (bit_index % 32))) {
          p1.load(R0 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
          p2.load(R1 + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
        } else {
          p1.load(R1 + buc_index * EC::Affine::LIMBS, 0, 0, lane_idx);
          p2.load(R0 + buc_index * EC::BaseField::LIMBS, 0, 0, lane_idx);
        }
      #endif

      // diff_add.get_diff_in_place(p1, p2);

      M = diff_add.y * diff_x_inv;
      MM = M.square();
      t1 = (p2.x + p1.x);
      X3 = (MM - t1);
      t2 = p1.x - X3;
      t3 = M * t2;
      Y3 = (t3 - p1.y);
      res_add.x = X3;
      res_add.y = Y3;

      // diff_dbl.get_diff_in_place(p2, p2);
      
      M = diff_dbl.y * dbl_y_inv;
      MM = M.square();
      t1 = (p2.x + p2.x);
      X3 = (MM - t1);
      t2 = p2.x - X3;
      t3 = M * t2;
      Y3 = (t3 - p2.y);
      res_dbl.x = X3;
      res_dbl.y = Y3;

      if (p1.is_zero()) {
        res_add = p2; 
      } else if (p2.is_zero()) {
        res_add = p1;
        res_dbl = EC::Affine::zero();
      }

      #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS  
        if (scalar & (1 << (bit_index % 32))) {
          res_add.store_arbitrary(R0, count, buc_index, lane_idx);
          res_dbl.store_arbitrary(R1, count, buc_index, lane_idx);
        } else {
          res_add.store_arbitrary(R1, count, buc_index, lane_idx);
          res_dbl.store_arbitrary(R0, count, buc_index, lane_idx);
        }
      #else
        if (scalar & (1 << (bit_index % 32))) {
          res_add.store(R0, count, buc_index, lane_idx);
          res_dbl.store(R1, count, buc_index, lane_idx);
        } else {
          res_add.store(R1, count, buc_index, lane_idx);
          res_dbl.store(R0, count, buc_index, lane_idx);
        }
      #endif

      inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx] * diff_add.x;
      inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx + blockDim.x] = inv_chain[slot_idx * EC::Layout::WIDTH + lane_idx + blockDim.x] * diff_dbl.x;
    }
  }
  
}

}
}