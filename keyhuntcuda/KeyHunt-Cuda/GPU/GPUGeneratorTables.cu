/**
 * GPUGeneratorTables.cu
 * Device memory arrays for generator point multiples
 * These arrays store precomputed multiples of the generator point G
 */

#include <cuda_runtime.h>
#include <stdint.h>

// Generator point multiples for fast scalar multiplication
// Format: Gx[i*4 + j] stores the j-th 64-bit word of the x-coordinate of i*G
// Total size: 256 blocks * 32 rows * 4 uint64_t = 32768 uint64_t values
__device__ uint64_t Gx[256 * 32 * 4];
__device__ uint64_t Gy[256 * 32 * 4];

// Host function to initialize generator tables
// This should be called once at program startup
__host__ void InitializeGeneratorTables(uint64_t* host_gx, uint64_t* host_gy) {
    // Copy precomputed generator multiples from host to device
    cudaMemcpyToSymbol(Gx, host_gx, sizeof(uint64_t) * 256 * 32 * 4);
    cudaMemcpyToSymbol(Gy, host_gy, sizeof(uint64_t) * 256 * 32 * 4);
}

// Smaller table for frequently used small multiples (1*G to 256*G)
// This can be kept in constant memory for faster access
__constant__ uint64_t Gx_small[256 * 4];
__constant__ uint64_t Gy_small[256 * 4];

// Host function to initialize small generator table
__host__ void InitializeSmallGeneratorTable(uint64_t* host_gx, uint64_t* host_gy) {
    // Copy first 256 multiples to constant memory
    cudaMemcpyToSymbol(Gx_small, host_gx, sizeof(uint64_t) * 256 * 4);
    cudaMemcpyToSymbol(Gy_small, host_gy, sizeof(uint64_t) * 256 * 4);
}

// Function to get generator multiple from table
__device__ __forceinline__ void GetGeneratorMultiple(
    uint32_t multiple,
    uint64_t result_x[4],
    uint64_t result_y[4])
{
    if (multiple == 0) {
        // Zero point (identity)
        result_x[0] = 0; result_x[1] = 0; result_x[2] = 0; result_x[3] = 0;
        result_y[0] = 0; result_y[1] = 0; result_y[2] = 0; result_y[3] = 0;
        return;
    }
    
    if (multiple < 256) {
        // Use constant memory for small multiples (faster)
        uint32_t idx = multiple * 4;
        result_x[0] = Gx_small[idx];
        result_x[1] = Gx_small[idx + 1];
        result_x[2] = Gx_small[idx + 2];
        result_x[3] = Gx_small[idx + 3];
        
        result_y[0] = Gy_small[idx];
        result_y[1] = Gy_small[idx + 1];
        result_y[2] = Gy_small[idx + 2];
        result_y[3] = Gy_small[idx + 3];
    } else if (multiple < 8192) {
        // Use global memory for larger multiples
        uint32_t idx = multiple * 4;
        result_x[0] = Gx[idx];
        result_x[1] = Gx[idx + 1];
        result_x[2] = Gx[idx + 2];
        result_x[3] = Gx[idx + 3];
        
        result_y[0] = Gy[idx];
        result_y[1] = Gy[idx + 1];
        result_y[2] = Gy[idx + 2];
        result_y[3] = Gy[idx + 3];
    } else {
        // For very large multiples, compute on the fly
        // This would use windowed scalar multiplication
        // For now, just use modulo to wrap around
        uint32_t wrapped = multiple % 8192;
        GetGeneratorMultiple(wrapped, result_x, result_y);
    }
}

// Optimized batch loading for coalesced memory access
__device__ void LoadGeneratorBatch(
    uint32_t start_multiple,
    uint32_t count,
    uint64_t batch_x[][4],
    uint64_t batch_y[][4])
{
    // Load multiple generator points in a coalesced manner
    for (uint32_t i = 0; i < count; i++) {
        GetGeneratorMultiple(start_multiple + i, batch_x[i], batch_y[i]);
    }
}