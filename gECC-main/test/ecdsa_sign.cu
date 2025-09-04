#include "gecc.h"
#include "gecc/support.h"

#include "gtest/gtest.h"
#include<cmath>

using namespace gecc;
using namespace gecc::arith;
using namespace gecc::ecdsa;

#include "ecdsa_test_constants.h"


template <typename ECDSA_solver>
void test_ecdsa_sign_correctness() {
  // u32 count = 80 * 256;
  u32 count = 1<<22;
  bool is_batch_opt = true;
  ECDSA_solver solver;
  ECDSA_solver::initialize();
  solver.sign_init(E, PRIV_KEY, K, count);
  // for warm up
  solver.sign_exec(320, 256, is_batch_opt);
  cudaDeviceSynchronize();
  if (cudaPeekAtLastError() != cudaSuccess) {
    printf("sign_exec Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

  typename ECDSA_solver::Base s_data[] = {53390468, 3707438294, 544191978, 4121469643, 825547682, 1915407828, 890502664, 1407657824};
  typename ECDSA_solver::Base r_data[] = {445809716, 333416348, 3825898681, 2543947809, 3339099360, 679239748, 28424172, 550243236};
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < ECDSA_solver::Field::LIMBS; ++j) {
#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
      ASSERT_EQ(solver.sign_s[j * count + i], s_data[j])
        << "s[" << i << "][" << j << "]";
      ASSERT_EQ(solver.sign_r[j * count + i], r_data[j])
        << "r[" << i << "][" << j << "]";
#else
      ASSERT_EQ(solver.sign_s[i * ECDSA_solver::Field::LIMBS + j], s_data[j])
        << "s[" << i << "][" << j << "]";
      ASSERT_EQ(solver.sign_r[i * ECDSA_solver::Field::LIMBS + j], r_data[j])
        << "r[" << i << "][" << j << "]";
#endif
    }
  }
  solver.sign_close();
  cudaDeviceSynchronize();
}

template <typename ECDSA_solver>
void test_ecdsa_sign() {
  u32 count = 1<<22;
  bool is_batch_opt = true;
  ECDSA_solver solver;
  ECDSA_solver::initialize();

  // MAX_SM_NUMS=80: i=[12,17]
  // MAX_SM_NUMS=108: i=[13,17]
  // MAX_SM_NUMS=128: i=[12,16]
  for (int i = 13; i <= 16; i++) {
    count = MAX_SM_NUMS * (1<<i); //1<<18 ~ 1<<23
    printf("--------------------------- %u (%d << %d) --------------------------\n", count, MAX_SM_NUMS, i);

    // solver.sign_init(E, PRIV_KEY, K, count);
    solver.sign_random_init(RANDOM_E, RANDOM_PRIV_KEY, RANDOM_K, RANDOM_KEY_X, RANDOM_KEY_Y, count);
    // warm up
    solver.sign_exec_for_random_inputs(480, 256, is_batch_opt);
    cudaDeviceSynchronize();

    double min_time = DBL_MAX;
    u32 min_blk_num = 0, min_thd_num = 0;
    for (u32 block_num = MAX_SM_NUMS; block_num <= MAX_SM_NUMS * 12; block_num += MAX_SM_NUMS) {
      for (u32 thread_num = 128; thread_num <= 256; thread_num *= 2) {
        if(block_num * thread_num > count) continue;
        double min_elapsed = support::timeit(
          3, 3, [&]() {
            solver.sign_exec_for_random_inputs(block_num, thread_num, is_batch_opt);
        });
        cudaDeviceSynchronize();
        if (min_time > min_elapsed) {
          min_time = min_elapsed;
          min_blk_num = block_num;
          min_thd_num = thread_num;
        }
        printf("is_batch_opt %d blc_num %u thd_num: %u time: %lf speed : %lf signs/s\n", 
            is_batch_opt, block_num, thread_num, min_elapsed, (double)count / min_elapsed);
      }
    }
    printf("is_batch_opt %d blc_num %u thd_num: %u time: %lf the fatested speed : %lf signs/s\n", 
            is_batch_opt, min_blk_num, min_thd_num, min_time, (double)count / min_time);

    solver.sign_close();
  }
}


DEFINE_SM2_FP(Fq_SM2_1, FqSM2, u32, 32, LayoutT<1>, 8, gecc::arith::MONTFLAG::SOS, gecc::arith::CURVEFLAG::SM2);
DEFINE_FP(Fq_SM2_n, FqSM2_n, u32, 32, LayoutT<1>, 8);
DEFINE_EC(G1_1, G1SM2, Fq_SM2_1, SM2_CURVE, 2);
DEFINE_ECDSA(ECDSA_solver, G1_1_G1SM2, Fq_SM2_1, Fq_SM2_n);
TEST(ECDSA, Correctness) { test_ecdsa_sign_correctness<ECDSA_solver>(); }  
TEST(ECDSA, Performance) { test_ecdsa_sign<ECDSA_solver>(); }  