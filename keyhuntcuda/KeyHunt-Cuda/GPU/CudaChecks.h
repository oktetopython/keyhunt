#ifndef GPU_CUDA_CHECKS_H
#define GPU_CUDA_CHECKS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void cudaCheckImpl(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d -> %s\n", file, line, cudaGetErrorString(err));
        std::abort();
    }
}

#define CUDA_CHECK(call) cudaCheckImpl((call), __FILE__, __LINE__)

#endif // GPU_CUDA_CHECKS_H
