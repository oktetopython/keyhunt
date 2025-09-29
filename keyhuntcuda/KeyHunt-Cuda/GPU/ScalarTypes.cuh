#ifndef GPU_SCALAR_TYPES_CUH
#define GPU_SCALAR_TYPES_CUH

#include <stdint.h>

struct __align__(32) Scalar256 {
    uint64_t limbs[4];
};

struct __align__(32) Uint320 {
    uint64_t limbs[5];
};

#if defined(__CUDACC__)
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

HD void load_scalar(uint64_t dest[4], const Scalar256& src) {
    dest[0] = src.limbs[0];
    dest[1] = src.limbs[1];
    dest[2] = src.limbs[2];
    dest[3] = src.limbs[3];
}

HD void store_scalar(Scalar256& dst, const uint64_t src[4]) {
    dst.limbs[0] = src[0];
    dst.limbs[1] = src[1];
    dst.limbs[2] = src[2];
    dst.limbs[3] = src[3];
}

#undef HD

#endif // GPU_SCALAR_TYPES_CUH
