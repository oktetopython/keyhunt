#pragma once

#include "common.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <vector>

namespace gecc {
namespace support {

static double timeit(const std::function<void()> func) {
  auto t0 = std::chrono::high_resolution_clock::now();
  func();
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now() - t0)
             .count() /
         1000000.0;
}

static double timeit(int batch_size, int batches,
                     const std::function<void()> func) {
  double min_elapsed = std::numeric_limits<double>::max();
  for (int _ = 0; _ < batches; ++_) {
    double elapsed = timeit([&]() {
                       for (int _ = 0; _ < batch_size; ++_) {
                         func();
                       }
                       cudaDeviceSynchronize();
                     }) /
                     batch_size;
    min_elapsed = std::min(min_elapsed, elapsed);
  }
  return min_elapsed;
}

template <typename Field>
static void column_majored_aware_copy(const u32 k, typename Field::Base *d_data,
                                      const typename Field::Base *h_data) {
  std::vector<typename Field::Base> buffer(Field::LIMBS << k);
  for (u32 i = 0; i < 1 << k; ++i) {
    for (u32 j = 0; j < Field::LIMBS; ++j) {
#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
      const u32 lane_idx = j / Field::LIMBS_PER_LANE;
      const u32 jj = j % Field::LIMBS_PER_LANE;
      buffer[(jj << k | i) * Field::Layout::WIDTH + lane_idx] =
          h_data[i * Field::LIMBS + j];
#else
      buffer[i * Field::LIMBS + j] = h_data[i * Field::LIMBS + j];
#endif
    }
  }
  cudaMemcpy(d_data, buffer.data(), Field::SIZE << k, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

template <typename Field>
static typename Field::Base *
initialize_device_fp_from_const(const u32 k, const u64 c[][MAX_LIMBS]) {
  std::vector<typename Field::Base> h_data(Field::LIMBS << k);
  for (u32 i = 0; i < 1 << k; ++i) {
    for (u32 j = 0; j < Field::LIMBS; ++j) {
      h_data[i * Field::LIMBS + j] =
          reinterpret_cast<const typename Field::Base *>(c[i])[j];
    }
  }
  typename Field::Base *d_data;
  cudaMallocManaged(&d_data, Field::SIZE << k);
  column_majored_aware_copy<Field>(k, d_data, h_data.data());
  return d_data;
}

template <typename EC>
static typename EC::BaseField::Base *initialize_device_affine_ec_from_const(
    const u32 k, const u64 c[][2][EC::BaseField::DEGREE][MAX_LIMBS]) {
  using BaseFp = typename EC::BaseField::Fp;

  typename EC::Base *d_data;
  cudaMallocManaged(&d_data, EC::Affine::SIZE << k);

  std::vector<typename EC::Base> h_data(BaseFp::LIMBS << k);
  for (u32 t = 0; t < 2; ++t) {
    for (u32 tt = 0; tt < EC::BaseField::DEGREE; ++tt) {
      for (u32 i = 0; i < 1 << k; ++i) {
        for (u32 j = 0; j < BaseFp::LIMBS; ++j) {
          h_data[i * BaseFp::LIMBS + j] =
              reinterpret_cast<const typename EC::Base *>(c[i][t][tt])[j];
        }
      }
      column_majored_aware_copy<BaseFp>(
          k, d_data + (t * EC::BaseField::DEGREE + tt) * (BaseFp::LIMBS << k),
          h_data.data());
    }
  }
  return d_data;
}

template <typename EC>
static typename EC::BaseField::Base *initialize_device_jacobian_ec_from_const(
    const u32 k, const u64 c[][3][EC::BaseField::DEGREE][MAX_LIMBS]) {
  using BaseFp = typename EC::BaseField::Fp;

  typename EC::Base *d_data;
  cudaMallocManaged(&d_data, EC::SIZE << k);

  std::vector<typename EC::Base> h_data(BaseFp::LIMBS << k);
  for (u32 t = 0; t < 3; ++t) {
    for (u32 tt = 0; tt < EC::BaseField::DEGREE; ++tt) {
      for (u32 i = 0; i < 1 << k; ++i) {
        for (u32 j = 0; j < BaseFp::LIMBS; ++j) {
          h_data[i * BaseFp::LIMBS + j] =
              reinterpret_cast<const typename EC::Base *>(c[i][t][tt])[j];
        }
      }
      column_majored_aware_copy<BaseFp>(
          k, d_data + (t * EC::BaseField::DEGREE + tt) * (BaseFp::LIMBS << k),
          h_data.data());
    }
  }
  return d_data;
}

template <typename Field>
static void to_row_majored_if_needed(const u32 k, typename Field::Base *data) {
#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
  std::vector<typename Field::Base> h_data(Field::LIMBS << k);
  for (u32 i = 0; i < (1 << k); ++i) {
    for (u32 j = 0; j < Field::LIMBS; ++j) {
      const u32 lane_idx = j / Field::LIMBS_PER_LANE;
      const u32 jj = j % Field::LIMBS_PER_LANE;
      h_data[i * Field::LIMBS + j] =
          data[((jj << k) | i) * Field::Layout::WIDTH + lane_idx];
    }
  }
  cudaMemcpy(data, h_data.data(), Field::SIZE << k, cudaMemcpyHostToDevice);
#endif
}

} // namespace support
} // namespace gecc
