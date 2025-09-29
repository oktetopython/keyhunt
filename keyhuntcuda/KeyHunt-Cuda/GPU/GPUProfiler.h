#ifndef GPUPROFILER_H
#define GPUPROFILER_H

#include "GPUMath.h"

// Device-side profiling for _ModInvGrouped and point operations
// Provides cycle-level timing to quantify bottleneck components

#ifdef KEYHUNT_PROFILE_INTERNAL

// Shared memory for collecting timing data from all threads
__shared__ uint64_t shared_modinv_cycles[32];
__shared__ uint64_t shared_pointops_cycles[32];
__shared__ uint32_t shared_sample_count;

// Device-side profiler for modular inverse operations
__device__ __forceinline__ void profile_modinv_start(uint64_t* start_cycles)
{
    *start_cycles = clock64();
}

__device__ __forceinline__ void profile_modinv_end(uint64_t start_cycles)
{
    uint64_t end_cycles = clock64();
    uint64_t elapsed = end_cycles - start_cycles;
    
    int tid = threadIdx.x;
    if (tid < 32) {
        shared_modinv_cycles[tid] = elapsed;
    }
}

// Device-side profiler for point operations
__device__ __forceinline__ void profile_pointops_start(uint64_t* start_cycles)
{
    *start_cycles = clock64();
}

__device__ __forceinline__ void profile_pointops_end(uint64_t start_cycles)
{
    uint64_t end_cycles = clock64();
    uint64_t elapsed = end_cycles - start_cycles;
    
    int tid = threadIdx.x;
    if (tid < 32) {
        shared_pointops_cycles[tid] = elapsed;
    }
}

// Report timing statistics (called by thread 0 only)
__device__ void report_timing_stats()
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Calculate average cycles for modular inverse
        uint64_t total_modinv = 0;
        uint64_t total_pointops = 0;
        
        for (int i = 0; i < 32; i++) {
            total_modinv += shared_modinv_cycles[i];
            total_pointops += shared_pointops_cycles[i];
        }
        
        uint64_t avg_modinv = total_modinv / 32;
        uint64_t avg_pointops = total_pointops / 32;
        
        printf("[DEVICE_PROFILE] ModInv avg cycles: %lu, PointOps avg cycles: %lu\n", 
               avg_modinv, avg_pointops);
        printf("[DEVICE_PROFILE] ModInv percentage: %.1f%%\n", 
               (float)avg_modinv / (avg_modinv + avg_pointops) * 100.0f);
    }
}

// Instrumented version of _ModInvGrouped with cycle counting
__device__ __forceinline__ void _ModInvGrouped_Profiled(uint64_t r[GRP_SIZE / 2 + 1][4])
{
    uint64_t start_cycles;
    profile_modinv_start(&start_cycles);
    
    // Original _ModInvGrouped implementation
    uint64_t subp[GRP_SIZE / 2 + 1][4];
    uint64_t newValue[4];
    uint64_t inverse[5];

    Load256(subp[0], r[0]);
    for (uint32_t i = 1; i < (GRP_SIZE / 2 + 1); i++) {
        _ModMult(subp[i], subp[i - 1], r[i]);
    }

    // We need 320bit signed int for ModInv
    Load256(inverse, subp[(GRP_SIZE / 2 + 1) - 1]);
    inverse[4] = 0;
    _ModInv(inverse);

    for (uint32_t i = (GRP_SIZE / 2 + 1) - 1; i > 0; i--) {
        _ModMult(newValue, subp[i - 1], inverse);
        _ModMult(inverse, r[i]);
        Load256(r[i], newValue);
    }

    Load256(r[0], inverse);
    
    profile_modinv_end(start_cycles);
}

// Instrumented point addition with cycle counting
__device__ __forceinline__ void compute_ec_point_add_profiled(uint64_t* px, uint64_t* py, 
    uint64_t* gx, uint64_t* gy, uint64_t* dx)
{
    uint64_t start_cycles;
    profile_pointops_start(&start_cycles);
    
    // Original point addition implementation
    compute_ec_point_add(px, py, gx, gy, dx);
    
    profile_pointops_end(start_cycles);
}

// Instrumented negative point addition with cycle counting
__device__ __forceinline__ void compute_ec_point_add_negative_profiled(uint64_t* px, uint64_t* py,
    uint64_t* pyn, uint64_t* gx, uint64_t* gy, uint64_t* dx)
{
    uint64_t start_cycles;
    profile_pointops_start(&start_cycles);

    // Original negative point addition implementation
    compute_ec_point_add_negative(px, py, pyn, gx, gy, dx);

    profile_pointops_end(start_cycles);
}

#else

// No-op versions when profiling is disabled
#define profile_modinv_start(x)
#define profile_modinv_end(x)
#define profile_pointops_start(x)
#define profile_pointops_end(x)
#define report_timing_stats()
#define _ModInvGrouped_Profiled _ModInvGrouped
#define compute_ec_point_add_profiled compute_ec_point_add
#define compute_ec_point_add_negative_profiled compute_ec_point_add_negative

#endif // KEYHUNT_PROFILE_INTERNAL

#endif // GPUPROFILER_H
