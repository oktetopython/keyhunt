#include "gecc.h"
#include "gecc/support.h"
#include "gtest/gtest.h"
#include<cmath>

using namespace gecc;
using namespace gecc::arith;
using namespace gecc::ecdsa;

#include "ecdsa_test_constants.h"

template <typename ECDSA_Verify_Solver>
void test_ecdsa_verify_correctness() {
  u32 count = 1 << 22;
  bool is_batch_opt = true;

  ECDSA_Verify_Solver solver;
  ECDSA_Verify_Solver::initialize();
  
  solver.verify_init(R, S, E, KEY_X, KEY_Y, count);
  // for warm up
  solver.verify_exec(MAX_SM_NUMS<<2, 256, is_batch_opt);
  cudaDeviceSynchronize();
  if (cudaPeekAtLastError() != cudaSuccess) {
    printf("verify_exec Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

  typename ECDSA_Verify_Solver::Base verify_point_x[] = {152344006,543100089,322498133,2749880450,1363215970,2783691004,1939829604,560625183};
  typename ECDSA_Verify_Solver::Base verify_naf_point_x[] = {296726863,394689338,142304217,3294409477,3307483470,123861261,1148184364,2296251045}; 

  // test for batch add
  if (is_batch_opt) {
    for (size_t i = 0; i < count; ++i) {
      for (size_t j = 0; j < ECDSA_Verify_Solver::Field::LIMBS; ++j) {
        // printf("j %d %lu\n", j, solver.R0[j * count + i]);
  #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        // ASSERT_EQ(solver.R1[j * count + i], verify_point_x[j])
        //   << "p[" << i << "][" << j << "]";
        ASSERT_EQ(solver.R0[j * count + i], verify_naf_point_x[j])
          << "naf_p[" << i << "][" << j << "]";
        // ASSERT_EQ(solver.R0[j * count + i], verify_naf_point_x[j])
        //   << "naf_p[" << i << "][" << j << "]";
  #else
        // ASSERT_EQ(solver.verify_point[i * ECDSA_Verify_Solver::Field::LIMBS + j], verify_point_x[j])
        //   << "p[" << i << "][" << j << "]";
        // ASSERT_EQ(solver.verify_naf_point[i * ECDSA_Verify_Solver::Field::LIMBS + j], verify_point_x[j])
        //   << "naf_p[" << i << "][" << j << "]";
        ASSERT_EQ(solver.R0[i * ECDSA_Verify_Solver::Field::LIMBS * 2 + j], verify_naf_point_x[j])
          << "naf_p[" << i << "][" << j << "]";
  #endif
      }
    }
  }
  solver.verify_close();
  cudaDeviceSynchronize();
}


template <typename ECDSA_Verify_Solver>
void test_ecdsa_verify() {
  u32 count = 1 << 22;
  // u32 count = 80 * 256;
  bool is_batch_opt = true;

  ECDSA_Verify_Solver solver;
  ECDSA_Verify_Solver::initialize();
  
  // MAX_SM_NUMS=80: i=[13,17]
  // MAX_SM_NUMS=108: i=[13,17]
  // MAX_SM_NUMS=128: i=[12,16]
  for (int i = 13; i <= 16; i++) {
    count = MAX_SM_NUMS * (1<<i); //1<<18 ~ 1<<23
    printf("--------------------------- %u (%d << %d) --------------------------\n", count, MAX_SM_NUMS, i);

    // solver.verify_init(R, S, E, KEY_X, KEY_Y, count);
    solver.verify_random_init(RANDOM_R, RANDOM_S, RANDOM_E, RANDOM_KEY_X, RANDOM_KEY_Y, count);
    // warm up
    solver.verify_exec_for_random_input(MAX_SM_NUMS<<2, 256, is_batch_opt);
    cudaDeviceSynchronize();

    double min_time = DBL_MAX;
    u32 min_blk_num = 0, min_thd_num = 0;
    for (u32 block_num = MAX_SM_NUMS; block_num <= MAX_SM_NUMS * 12; block_num += MAX_SM_NUMS) {
      for (u32 thread_num = 128; thread_num <= 256; thread_num *= 2) {
        if(block_num * thread_num > count) continue;
        double min_elapsed = support::timeit(
          3, 3, [&]() {
            solver.verify_exec_for_random_input(block_num, thread_num, is_batch_opt);
        });
        cudaDeviceSynchronize();
        if (min_time > min_elapsed) {
          min_time = min_elapsed;
          min_blk_num = block_num;
          min_thd_num = thread_num;
        }
        printf("is_batch_opt %d blc_num %u thd_num: %u time: %lf speed : %lf verifies/s\n", 
            is_batch_opt, block_num, thread_num, min_elapsed, (double)count / min_elapsed); 
      }
    }
    printf("is_batch_opt %d blc_num %u thd_num: %u time: %lf the fatested speed : %lf verifies/s\n", 
            is_batch_opt, min_blk_num, min_thd_num, min_time, (double)count / min_time);

    solver.verify_close();
  }
  
}

DEFINE_SM2_FP(Fq_SM2_1, FqSM2, u32, 32, LayoutT<1>, 8, gecc::arith::MONTFLAG::SOS, gecc::arith::CURVEFLAG::SM2);
DEFINE_FP(Fq_SM2_n, FqSM2_n, u32, 32, LayoutT<1>, 8);
DEFINE_EC(G1_1, G1SM2, Fq_SM2_1, SM2_CURVE, 2);
DEFINE_ECDSA(ECDSA_Verify_Solver, G1_1_G1SM2, Fq_SM2_1, Fq_SM2_n);
TEST(ECDSA, Correctness) { test_ecdsa_verify_correctness<ECDSA_Verify_Solver>(); }  
TEST(ECDSA, Performance) { test_ecdsa_verify<ECDSA_Verify_Solver>(); } 