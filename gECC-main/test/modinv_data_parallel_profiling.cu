#include "gecc.h"
#include "gecc/support.h"
#include "gtest/gtest.h"
#include<cmath>

using namespace gecc;
using namespace gecc::arith;
using namespace gecc::ecdsa;

#include "ecdsa_test_constants.h"

template <typename ECDSA_EC_PMUL_Solver>
void test_modinv_in_data_parallel() {
  u32 count = 1 << 22;

  ECDSA_EC_PMUL_Solver solver;
  ECDSA_EC_PMUL_Solver::initialize();

  // MAX_SM_NUMS=80: i=[12,17]+6
  // MAX_SM_NUMS=108: i=[12,17]+6
  // MAX_SM_NUMS=128: i=[11,16]+7
  for (int i = 10; i <= 18; i++) {
    count = MAX_SM_NUMS * (1<<i); //1<<18 ~ 1<<23
    printf("--------------------------- %u (~1<< %d) --------------------------\n", count, ((int)log2(MAX_SM_NUMS)) + i);

    // solver.verify_init(R, S, E, KEY_X, KEY_Y, count);
    solver.ec_pmul_random_init(RANDOM_S, RANDOM_KEY_X, RANDOM_KEY_Y, count);
    // warm up
    solver.batch_modinv_MTA(MAX_SM_NUMS<<2, 256);
    cudaDeviceSynchronize();

    double min_time = DBL_MAX;
    u32 min_blk_num = 0, min_thd_num = 0;
    for (u32 block_num = MAX_SM_NUMS*4; block_num <= MAX_SM_NUMS*4; block_num += MAX_SM_NUMS) {
      for (u32 thread_num = 256; thread_num <= 256; thread_num *= 2) {
        if(block_num * thread_num > count) continue;
        double min_elapsed = support::timeit(
          10, 100, [&]() {
            solver.batch_modinv_MTA(block_num, thread_num);
        });
        cudaDeviceSynchronize();
        if (min_time > min_elapsed) {
          min_time = min_elapsed;
          min_blk_num = block_num;
          min_thd_num = thread_num;
        }
        printf("batch modinv(data parallel): blc_num %u thd_num: %u time: %lfms\n", 
            block_num, thread_num, min_elapsed*1000); 
      }
    }
    printf("batch modinv(data parallel): blc_num %u thd_num: %u the fastest time: %lfms\n", 
            min_blk_num, min_thd_num, min_time*1000);

    solver.ec_pmul_close();
  }
  
}

DEFINE_SM2_FP(Fq_SM2_1, FqSM2, u32, 32, LayoutT<1>, 8, gecc::arith::MONTFLAG::SOS, gecc::arith::CURVEFLAG::SM2);
DEFINE_FP(Fq_SM2_n, FqSM2_n, u32, 32, LayoutT<1>, 8);
DEFINE_EC(G1_1, G1SM2, Fq_SM2_1, SM2_CURVE, 2);
DEFINE_ECDSA(ECDSA_EC_PMUL_Solver, G1_1_G1SM2, Fq_SM2_1, Fq_SM2_n);
TEST(ECDSA_EC_PMUL, Performance) { test_modinv_in_data_parallel<ECDSA_EC_PMUL_Solver>(); } 