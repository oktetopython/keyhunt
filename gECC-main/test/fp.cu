#include <cstdint>

#include "gtest/gtest.h"

#include "gecc.h"
#include "gecc/arith/layout.h"

using namespace gecc;
using namespace arith;

#include "fp_test_constants.h"

static_assert(MAX_BYTES >= 16 * 8, "");

template <typename Field>
__global__ void TestFp(const typename Field::Base *as,
                       const typename Field::Base *bs,
                       typename Field::Base *sum, typename Field::Base *prod,
                       typename Field::Base *inv_prod) {
  const u32 offset = threadIdx.x * Field::LIMBS_PER_LANE;
  Field a, b;
#pragma unroll
  for (u32 i = 0; i < Field::LIMBS_PER_LANE; ++i) {
    a.digits[i] = as[offset + i];
    b.digits[i] = bs[offset + i];
  }
Field aa = (a + b).reduce_to_p();
#pragma unroll
  for (u32 i = 0; i < Field::LIMBS_PER_LANE; ++i) {
    sum[offset + i] = aa.digits[i];
  }
  a.inplace_to_montgomery();
  b.inplace_to_montgomery();
  Field c = (a * b).from_montgomery();
#pragma unroll
  for (u32 i = 0; i < Field::LIMBS_PER_LANE; ++i) {
    prod[offset + i] = c.digits[i];
  }
Field d = (a * a.inverse()).from_montgomery();
#pragma unroll
  for (u32 i = 0; i < Field::LIMBS_PER_LANE; ++i) {
    inv_prod[offset + i] = d.digits[i];
  }
}

template <typename Field>
__global__ void TestFp_speed(const typename Field::Base *as,
                       const typename Field::Base *bs,
                       typename Field::Base *sum, 
                       typename Field::Base *out) {
  const u32 offset = 0;
  Field a, b;
#pragma unroll
  for (u32 i = 0; i < Field::LIMBS_PER_LANE; ++i) {
    a.digits[i] = as[offset + i] + threadIdx.x;
    b.digits[i] = bs[offset + i] + threadIdx.x;
  }
  Field c = (a * b);
  for (u32 i = 0; i < 10000 * Field::Layout::WIDTH; i++) {
    c = c * a;
  }

  // column-major layout && avoid write races
  const u64 offset_out = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (u32 i = 0; i < Field::LIMBS_PER_LANE; ++i) {
    out[offset_out + i * gridDim.x * blockDim.x] = c.digits[i];
  }
}

template <typename Field> void test_fp(size_t N, 
                                        const uint64_t A[][MAX_LIMBS],  
                                        const uint64_t B[][MAX_LIMBS], 
                                        const uint64_t SUM[][MAX_LIMBS], 
                                        const uint64_t PROD[][MAX_LIMBS]) {

  using Base = typename Field::Base;
  static const size_t LIMBS = Field::LIMBS;

  Field::initialize();

  Base *a, *b, *sum, *prod, *inv_prod;
  cudaMallocManaged(&a, Field::SIZE * N);
  cudaMallocManaged(&b, Field::SIZE * N);
  cudaMallocManaged(&sum, Field::SIZE * N);
  cudaMallocManaged(&prod, Field::SIZE * N);
  cudaMallocManaged(&inv_prod, Field::SIZE * N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < LIMBS; ++j) {
      a[i * Field::LIMBS + j] = reinterpret_cast<const Base *>(A[i])[j];
      b[i * Field::LIMBS + j] = reinterpret_cast<const Base *>(B[i])[j];
    }
  }

  TestFp<Field><<<1, Field::Layout::WIDTH * N>>>(a, b, sum, prod, inv_prod);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < LIMBS; ++j) {
      ASSERT_EQ(sum[i * Field::LIMBS + j], 
              reinterpret_cast<const Base *>(SUM[i])[j])
          << "SUM[" << i << "][" << j << "]";
    }
  }

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < LIMBS; ++j) {
      ASSERT_EQ(prod[i * Field::LIMBS + j], 
            reinterpret_cast<const Base *>(PROD[i])[j])
          << "PROD[" << i << "][" << j << "]";
    }
  }

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < LIMBS; ++j) {
      ASSERT_EQ(inv_prod[i * Field::LIMBS + j], j == 0 ? 1 : 0)
          << "INV_PROD[" << i << "][" << j << "]";
    }
  }

  u32 dev = 0;
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp,dev);

  // test_speed
  cudaEvent_t   start, stop;
  float         time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  Base *out;
  cudaMallocManaged(&out, Field::SIZE * 32 * devProp.multiProcessorCount  * 256);

  // warm up
  TestFp_speed<Field><<<32 * devProp.multiProcessorCount * Field::Layout::WIDTH,256>>>(a, b, sum, out);
  TestFp_speed<Field><<<32 * devProp.multiProcessorCount * Field::Layout::WIDTH,256>>>(a, b, sum, out);
  cudaDeviceSynchronize();
  for (size_t i = 0; i < 5; i++) {
    cudaEventRecord(start, 0);
    TestFp_speed<Field><<<32 * devProp.multiProcessorCount * Field::Layout::WIDTH,256>>>(a, b, sum, out);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("time %0.3f ms\n", time);
  }
  if (cudaPeekAtLastError() != cudaSuccess) {
    printf("error, %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }
  cudaDeviceSynchronize();

  cudaFree(a);
  cudaFree(b);
  cudaFree(prod);
  cudaFree(inv_prod);
  cudaFree(out);
}

#define ADD_FqSM2_FP_TEST(FIELD, DIGIT_TYPE, DIGIT_WIDTH, LAYOUT_WIDTH, LIMBS)       \
  DEFINE_SM2_FP(FIELD, FqSM2, DIGIT_TYPE, DIGIT_WIDTH, LayoutT<LAYOUT_WIDTH>, LIMBS, gecc::arith::MONTFLAG::SOS, gecc::arith::CURVEFLAG::SM2);     \
  TEST(FqSM2256K1_FP, FIELD##Correctness) { using namespace FqSM2_fp_test; test_fp<FIELD>(N, A, B, SUM, PROD); }

#define ADD_FqSM2_n_FP_TEST(FIELD, DIGIT_TYPE, DIGIT_WIDTH, LAYOUT_WIDTH, LIMBS)       \
  DEFINE_FP(FIELD, FqSM2_n, DIGIT_TYPE, DIGIT_WIDTH, LayoutT<LAYOUT_WIDTH>, LIMBS);     \
  TEST(FqSM2_FP_n, FIELD##Correctness) { using namespace FqSM2_n_fp_test; test_fp<FIELD>(N, A, B, SUM, PROD); }


ADD_FqSM2_FP_TEST(Field_SM2, u32, 32, 1, 8)
ADD_FqSM2_n_FP_TEST(Field_SM2_n, u32, 32, 1, 8)