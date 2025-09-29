#ifndef GPU_DEFINES_H
#define GPU_DEFINES_H

#include <cuda_runtime.h>
#include <stdint.h>

// Prefetch functions for cache optimization
// These are CUDA 11+ features, provide fallback for older versions
#if __CUDA_ARCH__ >= 800
    #define __prefetch_global_l2(ptr) __builtin_prefetch(ptr, 0, 3)
#else
    // Fallback for older architectures - no prefetch
    #define __prefetch_global_l2(ptr) ((void)0)
#endif

// EC point addition function is defined in BatchStepping.h

// Missing function definitions for compilation
// _addPoint is now defined in ECPointOps.cuh

__device__ void ModAdd256(uint64_t* result, const uint64_t* a, const uint64_t* b);

__device__ void ModInvGrouped(uint64_t* dx, int size);

__device__ inline void ModAdd256(uint64_t* result, const uint64_t* a, const uint64_t* b) {
    // Placeholder 256-bit modular addition
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a[i] + b[i] + carry;
        carry = (sum < a[i]) ? 1 : 0;
        result[i] = sum;
    }
}

__device__ inline void ModInvGrouped(uint64_t* dx, int size) {
    // Placeholder batch modular inversion
    for (int i = 0; i < size; i++) {
        // Placeholder: set to 1 (not mathematically correct)
        for (int j = 0; j < 4; j++) {
            dx[i * 4 + j] = (j == 0) ? 1 : 0;
        }
    }
}

#endif // GPU_DEFINES_H
