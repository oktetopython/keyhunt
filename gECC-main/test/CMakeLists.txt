macro(add_gecc_test TEST_NAME TEST_SOURCE)
  set(test_target "${TEST_NAME}_test")
  add_executable(${test_target} ${TEST_SOURCE})
  target_link_libraries(${test_target} PRIVATE GTest::Main stdc++)
  add_dependencies(${test_target} generated_constants)
  gtest_discover_tests(${test_target})
endmacro()

add_gecc_test(fp fp.cu)
# cuda_target_compile_options(fp_test ON)

# ----------------------------------- signature generation ---------------------------------------------
##Breakdown 1: sig_gen: rapid_ec:sig_gen + opt_modmul 
add_gecc_test(ecdsa_sign_bk1 ecdsa_sign_baseline.cu)
# cuda_target_compile_options(ecdsa_sign_bk1_test ON)
target_compile_definitions(ecdsa_sign_bk1_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_sign_bk1_test PRIVATE -DSPPARK_32)

##Breakdown 2: BK1 + batch_inv
add_gecc_test(ecdsa_sign_bk2 ecdsa_sign_baseline.cu)
# cuda_target_compile_options(ecdsa_sign_bk2_test ON)
target_compile_definitions(ecdsa_sign_bk2_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_sign_bk2_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_sign_bk2_test PRIVATE -DBATCH_INV)

##Breakdown 3: BK2 + batch_pmul
add_gecc_test(ecdsa_sign_bk3 ecdsa_sign.cu)
# cuda_target_compile_options(ecdsa_sign_bk3_test ON)
target_compile_definitions(ecdsa_sign_bk3_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_sign_bk3_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_sign_bk3_test PRIVATE -DGECC_QAPW_OPT_COLUMN_MAJORED_INPUTS)
target_compile_definitions(ecdsa_sign_bk3_test PRIVATE -DPERSISTENT_L2_CACHE)

# ----------------------------------- signature verification ---------------------------------------------
#Breakdown 1: sig_verify: rapid_ec:sig_verify + opt_modmul
add_gecc_test(ecdsa_verify_bk1 ecdsa_verify_baseline.cu)
# cuda_target_compile_options(ecdsa_verify_bk1_test ON)
target_compile_definitions(ecdsa_verify_bk1_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_verify_bk1_test PRIVATE -DSPPARK_32)

##Breakdown 2: BK1 + batch fixed-point mul (no mem opt)
add_gecc_test(ecdsa_verify_bk2 ecdsa_verify.cu)
# cuda_target_compile_options(ecdsa_verify_bk2_test ON)
target_compile_definitions(ecdsa_verify_bk2_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_verify_bk2_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_verify_bk2_test PRIVATE -DVERIFY_BK_1)

##Breakdown 3: BK2 + batch pmul optimization(kernel-fusion + mem-opt)
add_gecc_test(ecdsa_verify_bk3 ecdsa_verify.cu)
# cuda_target_compile_options(ecdsa_verify_bk3_test ON)
target_compile_definitions(ecdsa_verify_bk3_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_verify_bk3_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_verify_bk3_test PRIVATE -DGECC_QAPW_OPT_COLUMN_MAJORED_INPUTS)
target_compile_definitions(ecdsa_verify_bk3_test PRIVATE -DPERSISTENT_L2_CACHE)

# ---------------------------------- EC_UNKNOWN_PMUL ------------------------
##Breakdown-1: pmul_naf + opt_modmul
add_gecc_test(ecdsa_ec_unknown_pmul_bk1 ecdsa_ec_unknown_pmul.cu)
# cuda_target_compile_options(ecdsa_ec_unknown_pmul_bk1_test ON)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk1_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk1_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk1_test PRIVATE -DEC_UPMUL_BASE)

##Breakdown-2: batch-upmul: no optimization
add_gecc_test(ecdsa_ec_unknown_pmul_bk2 ecdsa_ec_unknown_pmul.cu)
# cuda_target_compile_options(ecdsa_ec_unknown_pmul_bk2_test ON)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk2_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk2_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk2_test PRIVATE -DBATCH_UPMUL_NO_OPT)

# ##Breakdown-3: batch-upmul: kernel-fusion optimization
add_gecc_test(ecdsa_ec_unknown_pmul_bk3 ecdsa_ec_unknown_pmul.cu)
# cuda_target_compile_options(ecdsa_ec_unknown_pmul_bk3_test ON)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk3_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk3_test PRIVATE -DSPPARK_32)

##Breakdown-3-5: batch-upmul: kernel-fusion optimization
add_gecc_test(ecdsa_ec_unknown_pmul_bk3-5 ecdsa_ec_unknown_pmul.cu)
# cuda_target_compile_options(ecdsa_ec_unknown_pmul_bk3_test ON)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk3-5_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk3-5_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk3-5_test PRIVATE -DPERSISTENT_L2_CACHE)

##Breakdown-4: batch-upmul: kernel-fusion optimization + memory management
add_gecc_test(ecdsa_ec_unknown_pmul_bk4 ecdsa_ec_unknown_pmul.cu)
# cuda_target_compile_options(ecdsa_ec_unknown_pmul_bk4_test ON)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk4_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk4_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk4_test PRIVATE -DGECC_QAPW_OPT_COLUMN_MAJORED_INPUTS)
target_compile_definitions(ecdsa_ec_unknown_pmul_bk4_test PRIVATE -DPERSISTENT_L2_CACHE)

# ---------------------------------- EC_FIXED_PMUL ------------------------
##Breakdown-1: fpmul + opt_modmul
add_gecc_test(ecdsa_ec_fixed_pmul_bk1 ecdsa_ec_fixed_pmul.cu)
# cuda_target_compile_options(ecdsa_ec_fixed_pmul_bk1_test ON)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk1_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk1_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk1_test PRIVATE -DEC_FPMUL_BASE)

##Breakdown-2: batch-fpmul: no optimization
add_gecc_test(ecdsa_ec_fixed_pmul_bk2 ecdsa_ec_fixed_pmul.cu)
# cuda_target_compile_options(ecdsa_ec_fixed_pmul_bk2_test ON)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk2_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk2_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk2_test PRIVATE -DBATCH_UPMUL_NO_OPT)

##Breakdown-3: batch-fpmul: kernel-fusion optimization
add_gecc_test(ecdsa_ec_fixed_pmul_bk3 ecdsa_ec_fixed_pmul.cu)
# # cuda_target_compile_options(ecdsa_ec_fixed_pmul_bk3_test ON)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk3_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk3_test PRIVATE -DSPPARK_32)

##Breakdown-3-5: batch-fpmul: kernel-fusion optimization + L2cache opt
add_gecc_test(ecdsa_ec_fixed_pmul_bk3-5 ecdsa_ec_fixed_pmul.cu)
# # cuda_target_compile_options(ecdsa_ec_fixed_pmul_bk3-5_test ON)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk3-5_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk3-5_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk3-5_test PRIVATE -DPERSISTENT_L2_CACHE)

##Breakdown-4: batch-fpmul: kernel-fusion optimization + memory management
add_gecc_test(ecdsa_ec_fixed_pmul_bk4 ecdsa_ec_fixed_pmul.cu)
# cuda_target_compile_options(ecdsa_ec_fixed_pmul_bk4_test ON)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk4_test PRIVATE -DOVERFLOW)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk4_test PRIVATE -DSPPARK_32)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk4_test PRIVATE -DGECC_QAPW_OPT_COLUMN_MAJORED_INPUTS)
target_compile_definitions(ecdsa_ec_fixed_pmul_bk4_test PRIVATE -DPERSISTENT_L2_CACHE)

#Profiling: batch modinv in data parallel
add_gecc_test(modinv_data_parallel_profiling modinv_data_parallel_profiling.cu)
# cuda_target_compile_options(modinv_data_parallel_profiling_test ON)
target_compile_definitions(modinv_data_parallel_profiling_test PRIVATE -DOVERFLOW)
target_compile_definitions(modinv_data_parallel_profiling_test PRIVATE -DSPPARK_32)
target_compile_definitions(modinv_data_parallel_profiling_test PRIVATE -DBATCHMINVWITHDP) ## batch modinvs with data parallel model

add_gecc_test(modinv_GAS_profiling modinv_data_parallel_profiling.cu)
# cuda_target_compile_options(modinv_data_parallel_profiling_test ON)
target_compile_definitions(modinv_GAS_profiling_test PRIVATE -DOVERFLOW)
target_compile_definitions(modinv_GAS_profiling_test PRIVATE -DSPPARK_32)
target_compile_definitions(modinv_GAS_profiling_test PRIVATE -DBATCHMINVWITHGAS) ## batch modinvs with GAS model