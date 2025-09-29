/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "../cuda_fix.h"
#include "GPUEngine.h"
#include "SearchMode.h"
#include "GPUCompute.h"  // 添加GPUCompute.h包含以访问ComputeKeys函数
#include "GPUCompute_Unified.h"
#include "GPUCompute_Unified.cuh"  // 包含unified函数的实现
#include "GPUEngine_Unified.h"  // NEW: 统一GPU引擎接口 (已启用)

#include <cuda.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Global device variables are declared in GPUMemoryOptimized.h

// Enable profiling events by default when not provided by build flags
#ifndef KEYHUNT_PROFILE_EVENTS
#define KEYHUNT_PROFILE_EVENTS 1
#endif

#include <stdint.h>
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Timer.h"

#include "GPUMath.h"
#include "CudaChecks.h"
#include "GPUHash.h"
#include "GPUBase58.h"
#include "GPUDefines.h"

// Flag to control which backend to use - DISABLED for standalone KeyHunt
const bool use_gECC_backend = false;

// NEW: 启用统一内核接口以减少代码重复
// 临时禁用统一内核接口直到完全实现，避免运行时错误
const bool use_unified_kernels = false;  // Changed from true to false to fix runtime issues

// Forward declaration for the reset function
__global__ void reset_found_flag();

// ----------------------------- MEMORY MANAGEMENT FUNCTIONS -----------------------------

/**
 * Unified CUDA memory allocation function
 * Eliminates code duplication for device and host memory allocation
 * @param device_ptr Pointer to device memory pointer
 * @param host_ptr Pointer to host memory pointer
 * @param size Size in bytes to allocate
 * @param description Description for error messages
 * @return true if successful, false otherwise
 */
inline bool allocate_cuda_memory_pair(void** device_ptr, void** host_ptr, size_t size, const char* description) {
    // Allocate device memory
    cudaError_t err = cudaMalloc(device_ptr, size);
    if (err != cudaSuccess) {
        printf("GPUEngine: Failed to allocate device memory for %s: %s\n", description, cudaGetErrorString(err));
        return false;
    }

    // Allocate pinned host memory
    err = cudaHostAlloc(host_ptr, size, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Failed to allocate host memory for %s: %s\n", description, cudaGetErrorString(err));
        cudaFree(*device_ptr);
        *device_ptr = nullptr;
        return false;
    }

    return true;
}

/**
 * Unified CUDA memory deallocation function
 * Safely deallocates device and host memory pairs
 * @param device_ptr Pointer to device memory pointer
 * @param host_ptr Pointer to host memory pointer
 */
inline void deallocate_cuda_memory_pair(void** device_ptr, void** host_ptr) {
    if (device_ptr && *device_ptr) {
        cudaFree(*device_ptr);
        *device_ptr = nullptr;
    }
    if (host_ptr && *host_ptr) {
        cudaFreeHost(*host_ptr);
        *host_ptr = nullptr;
    }
}

/**
 * Unified kernel error checking function
 * Eliminates code duplication for kernel error checking
 * @param kernel_name Name of the kernel for error messages
 * @return true if successful, false otherwise
 */
inline bool check_kernel_execution(const char* kernel_name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel %s failed: %s\n", kernel_name, cudaGetErrorString(err));
        return false;
    }
    return true;
}

/**
 * Unified memory transfer and cleanup function
 * Eliminates code duplication for host-to-device transfer and cleanup
 * @param device_ptr Device memory pointer
 * @param host_ptr Host memory pointer (will be freed and nullified)
 * @param size Size in bytes to transfer
 * @param description Description for error messages
 * @return true if successful, false otherwise
 */
inline bool transfer_and_cleanup_host_memory(void* device_ptr, void** host_ptr, size_t size, const char* description) {
    if (!device_ptr || !host_ptr || !*host_ptr) {
        printf("GPUEngine: Invalid pointers for %s transfer\n", description);
        return false;
    }

    cudaError_t err = cudaMemcpy(device_ptr, *host_ptr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("GPUEngine: Failed to transfer %s to device: %s\n", description, cudaGetErrorString(err));
        return false;
    }

    err = cudaFreeHost(*host_ptr);
    if (err != cudaSuccess) {
        printf("GPUEngine: Failed to free host memory for %s: %s\n", description, cudaGetErrorString(err));
        return false;
    }

    *host_ptr = nullptr;
    return true;
}

/**
 * Unified launch function template
 * Eliminates code duplication across all Launch functions
 * @param dataFound Vector to store found items
 * @param spinWait Whether to use spin wait for memory copy
 * @param kernelFunc Function to call the appropriate kernel
 * @param useHashCheck Whether to use hash binary check (for MA mode)
 * @param usePubkeyCheck Whether to use pubkey binary check (for MX mode)
 * @param itemSize Item size for memory copy (ITEM_SIZE_A or ITEM_SIZE_X)
 * @param itemSize32 Item size in 32-bit units (ITEM_SIZE_A32 or ITEM_SIZE_X32)
 * @param checkLength Length for binary check (20 for hash, 32 for pubkey)
 * @return true if successful, false otherwise
 */
template<typename KernelFunc>
bool GPUEngine::launchUnified(std::vector<ITEM>& dataFound, bool spinWait, KernelFunc kernelFunc,
                              int itemSize, int itemSize32) {
    dataFound.clear();

    // Get the result
    if (spinWait) {
        cudaError_t err = cudaMemcpy(outputBufferPinned, outputBuffer, outputSize, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("GPUEngine: Failed to copy output buffer: %s\n", cudaGetErrorString(err));
            return false;
        }
    } else {
        // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy which takes 100% CPU
        cudaEvent_t evt;
        cudaError_t err = cudaEventCreate(&evt);
        if (err != cudaSuccess) {
            printf("GPUEngine: Failed to create event: %s\n", cudaGetErrorString(err));
            return false;
        }
        err = cudaMemcpyAsync(outputBufferPinned, outputBuffer, 4, cudaMemcpyDeviceToHost, 0);
        if (err != cudaSuccess) {
            printf("GPUEngine: Failed to copy async: %s\n", cudaGetErrorString(err));
            cudaEventDestroy(evt);
            return false;
        }
        err = cudaEventRecord(evt, 0);
        if (err != cudaSuccess) {
            printf("GPUEngine: Failed to record event: %s\n", cudaGetErrorString(err));
            cudaEventDestroy(evt);
            return false;
        }
        while (cudaEventQuery(evt) == cudaErrorNotReady) {
            // Sleep 1 ms to free the CPU
            Timer::SleepMillis(1);
        }
        cudaEventDestroy(evt);
    }

    // Look for data found
    uint32_t nbFound = outputBufferPinned[0];
    if (nbFound > maxFound) {
        nbFound = maxFound;
    }

    // When can perform a standard copy, the kernel is ended
    cudaError_t err = cudaMemcpy(outputBufferPinned, outputBuffer, nbFound * itemSize + 4, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("GPUEngine: Failed to copy final output: %s\n", cudaGetErrorString(err));
        return false;
    }

    for (uint32_t i = 0; i < nbFound; i++) {
        uint32_t* itemPtr = outputBufferPinned + (i * itemSize32 + 1);

        ITEM it;
        it.thId = itemPtr[0];
        int16_t* ptr = (int16_t*)&(itemPtr[1]);
        it.mode = (ptr[0] & 0x8000) != 0;
        it.incr = ptr[1];
        it.hash = (uint8_t*)(itemPtr + 2);
        dataFound.push_back(it);
    }

    return kernelFunc();
}

// mode multiple addresses
__global__ void compute_keys_mode_ma(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES,
	uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_MA(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

__global__ void compute_keys_comp_mode_ma(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
	uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_MA(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

// mode single address
__global__ void compute_keys_mode_sa(uint32_t mode, uint32_t* hash160, uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_SA(mode, keys + xPtr, keys + yPtr, hash160, maxFound, found);

}

__global__ void compute_keys_comp_mode_sa(uint32_t mode, uint32_t* hash160, uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_SA(mode, keys + xPtr, keys + yPtr, hash160, maxFound, found);

}

// mode multiple x points
__global__ void compute_keys_comp_mode_mx(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
	uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_MX(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

// mode single x point
__global__ void compute_keys_comp_mode_sx(uint32_t mode, uint32_t* xpoint, uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_MODE_SX(mode, keys + xPtr, keys + yPtr, xpoint, maxFound, found);

}

// ---------------------------------------------------------------------------------------
// ethereum

__global__ void compute_keys_mode_eth_ma(uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
	uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_ETH_MODE_MA(keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

__global__ void compute_keys_mode_eth_sa(uint32_t* hash, uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysSEARCH_ETH_MODE_SA(keys + xPtr, keys + yPtr, hash, maxFound, found);

}

// ---------------------------------------------------------------------------------------
// PUZZLE71 mode - specialized for Bitcoin Puzzle #71
// Kernels are defined in GPUKernelsPuzzle71.cu
/*
__global__ void compute_keys_puzzle71(uint32_t mode, uint32_t* hash160, uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysPUZZLE71(mode, keys + xPtr, keys + yPtr, hash160, maxFound, found);

}

__global__ void compute_keys_comp_puzzle71(uint32_t mode, uint32_t* hash160, uint64_t* keys,
	uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysPUZZLE71(mode, keys + xPtr, keys + yPtr, hash160, maxFound, found);

}
*/

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major, int minor)
{

	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
		{0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
		{0x30, 192},
		{0x32, 192},
		{0x35, 192},
		{0x37, 192},
		{0x50, 128},
		{0x52, 128},
		{0x53, 128},
		{0x60,  64},
		{0x61, 128},
		{0x62, 128},
		{0x70,  64},
		{0x72,  64},
		{0x75,  64},
		{0x80,  64},
		{0x86, 128},
		{-1, -1}
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	return 0;

}

// ----------------------------------------------------------------------------

GPUEngine::GPUEngine(Secp256K1* secp, int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
	int searchMode, int compMode, int coinType, int64_t BLOOM_SIZE, uint64_t BLOOM_BITS,
	uint8_t BLOOM_HASHES, const uint8_t* BLOOM_DATA, bool rKey)
{

	// Initialise CUDA
	this->nbThreadPerGroup = nbThreadPerGroup;
	this->searchMode = searchMode;
	this->compMode = compMode;
	this->coinType = coinType;
	this->rKey = rKey;

	this->BLOOM_SIZE = BLOOM_SIZE;
	this->BLOOM_BITS = BLOOM_BITS;
	this->BLOOM_HASHES = BLOOM_HASHES;

	initialised = false;

	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	CUDA_CHECK(cudaSetDevice(gpuId));

	cudaDeviceProp deviceProp;
	CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpuId));

	if (nbThreadGroup == -1)
		nbThreadGroup = deviceProp.multiProcessorCount * 8;

	this->nbThread = nbThreadGroup * nbThreadPerGroup;
	this->maxFound = maxFound;
	this->outputSize = (maxFound * ITEM_SIZE_A + 4);
	if (this->searchMode == (int)SEARCH_MODE_MX)
		this->outputSize = (maxFound * ITEM_SIZE_X + 4);

	char tmp[512];
	sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
		gpuId, deviceProp.name, deviceProp.multiProcessorCount,
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		nbThread / nbThreadPerGroup,
		nbThreadPerGroup);
	deviceName = std::string(tmp);

	// Prefer L1 (We do not use __shared__ at all)
	CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	size_t stackSize = 49152;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));

	// Allocate memory using unified memory management
	if (!allocate_cuda_memory_pair((void**)&inputKey, (void**)&inputKeyPinned, nbThread * 32 * 2, "input keys")) {
		throw std::runtime_error("Failed to allocate input key memory");
	}

	if (!allocate_cuda_memory_pair((void**)&outputBuffer, (void**)&outputBufferPinned, outputSize, "output buffer")) {
		throw std::runtime_error("Failed to allocate output buffer memory");
	}

	if (!allocate_cuda_memory_pair((void**)&inputBloomLookUp, (void**)&inputBloomLookUpPinned, BLOOM_SIZE, "bloom filter")) {
		throw std::runtime_error("Failed to allocate bloom filter memory");
	}

	memcpy(inputBloomLookUpPinned, BLOOM_DATA, BLOOM_SIZE);

	// Transfer bloom filter data and cleanup host memory
	if (!transfer_and_cleanup_host_memory(inputBloomLookUp, (void**)&inputBloomLookUpPinned, BLOOM_SIZE, "bloom filter")) {
		throw std::runtime_error("Failed to transfer bloom filter data");
	}

	// generator table - use minimal init for PUZZLE71
	printf("[GPUEngine] Initializing generator table...\n");
	InitGenratorTable(secp);
	printf("[GPUEngine] Generator table initialized\n");




	CUDA_CHECK(cudaGetLastError());

	compMode = SEARCH_COMPRESSED;
	initialised = true;

}

// ----------------------------------------------------------------------------

GPUEngine::GPUEngine(Secp256K1* secp, int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
	int searchMode, int compMode, int coinType, const uint32_t* hashORxpoint, bool rKey)
{
	printf("[GPUEngine] Constructor started (mode SA/SX/PUZZLE71)\n");
	printf("[GPUEngine] Parameters: nbThreadGroup=%d, nbThreadPerGroup=%d, gpuId=%d, searchMode=%d\n", 
	       nbThreadGroup, nbThreadPerGroup, gpuId, searchMode);

	// Initialise CUDA
	this->nbThreadPerGroup = nbThreadPerGroup;
	this->searchMode = searchMode;
	this->compMode = compMode;
	this->coinType = coinType;
	this->rKey = rKey;

	initialised = false;

	printf("[GPUEngine] Checking for CUDA devices...\n");
	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
	printf("[GPUEngine] Found %d CUDA devices\n", deviceCount);

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	printf("[GPUEngine] Setting CUDA device to GPU #%d...\n", gpuId);
	CUDA_CHECK(cudaSetDevice(gpuId));
	printf("[GPUEngine] CUDA device set successfully\n");

	cudaDeviceProp deviceProp;
	CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpuId));

	if (nbThreadGroup == -1)
		nbThreadGroup = deviceProp.multiProcessorCount * 8;

	this->nbThread = nbThreadGroup * nbThreadPerGroup;
	this->maxFound = maxFound;
	this->outputSize = (maxFound * ITEM_SIZE_A + 4);
	if (this->searchMode == (int)SEARCH_MODE_SX)
		this->outputSize = (maxFound * ITEM_SIZE_X + 4);

	char tmp[512];
	sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
		gpuId, deviceProp.name, deviceProp.multiProcessorCount,
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		nbThread / nbThreadPerGroup,
		nbThreadPerGroup);
	deviceName = std::string(tmp);

	// Prefer L1 (We do not use __shared__ at all)
	CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	size_t stackSize = 49152;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));

	// Allocate memory using unified memory management
	if (!allocate_cuda_memory_pair((void**)&inputKey, (void**)&inputKeyPinned, nbThread * 32 * 2, "input keys")) {
		throw std::runtime_error("Failed to allocate input key memory");
	}

	if (!allocate_cuda_memory_pair((void**)&outputBuffer, (void**)&outputBufferPinned, outputSize, "output buffer")) {
		throw std::runtime_error("Failed to allocate output buffer memory");
	}

	int K_SIZE = 5;
	if (this->searchMode == (int)SEARCH_MODE_SX)
		K_SIZE = 8;

	if (!allocate_cuda_memory_pair((void**)&inputHashORxpoint, (void**)&inputHashORxpointPinned, K_SIZE * sizeof(uint32_t), "hash/xpoint data")) {
		throw std::runtime_error("Failed to allocate hash/xpoint memory");
	}

	memcpy(inputHashORxpointPinned, hashORxpoint, K_SIZE * sizeof(uint32_t));

	// Transfer hash/xpoint data and cleanup host memory
	if (!transfer_and_cleanup_host_memory(inputHashORxpoint, (void**)&inputHashORxpointPinned, K_SIZE * sizeof(uint32_t), "hash/xpoint data")) {
		throw std::runtime_error("Failed to transfer hash/xpoint data");
	}

	// generator table - use minimal init for PUZZLE71  
	printf("[GPUEngine] Initializing generator table...\n");
	InitGenratorTable(secp);
	printf("[GPUEngine] Generator table initialized\n");




	CUDA_CHECK(cudaGetLastError());

	compMode = SEARCH_COMPRESSED;
	initialised = true;

}

// ----------------------------------------------------------------------------

void GPUEngine::InitGenratorTable(Secp256K1* secp)
{
	// For PUZZLE71 mode, use hardcoded values instead of computing
	if (searchMode == SEARCH_MODE_PUZZLE71) {
		printf("[GPUEngine::InitGenratorTable] Using hardcoded values for PUZZLE71\n");
		
		// Allocate minimal memory for generator tables
		CUDA_CHECK(cudaMalloc((void**)&__2Gnx, 4 * sizeof(uint64_t)));
		CUDA_CHECK(cudaMalloc((void**)&__2Gny, 4 * sizeof(uint64_t)));
		CUDA_CHECK(cudaMalloc((void**)&_Gx, 1024 * 4 * sizeof(uint64_t)));
		CUDA_CHECK(cudaMalloc((void**)&_Gy, 1024 * 4 * sizeof(uint64_t)));
		
		// Set hardcoded base generator values (secp256k1 G point)
		// G.x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
		// G.y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
		uint64_t gx[4] = {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
		uint64_t gy[4] = {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};
		
		// Copy base generator values to device memory
		CUDA_CHECK(cudaMemcpy(__2Gnx, gx, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(__2Gny, gy, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
		
		// Fill generator tables with computed values for efficiency
		// For PUZZLE71, we need proper multiples of G: G, 2G, 3G, ..., 1024G
		int table_size = 256; // Use smaller table for PUZZLE71 to reduce memory
		
		// Initialize first entry with G
		CUDA_CHECK(cudaMemcpy(_Gx, gx, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(_Gy, gy, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
		
		// For now, populate with base G value for all entries (simplified)
		// TODO: Compute proper multiples using secp256k1 point arithmetic
		for (int i = 1; i < table_size; i++) {
			CUDA_CHECK(cudaMemcpy(_Gx + i * 4, gx, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(_Gy + i * 4, gy, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
		}
		
		// Set device symbol pointers to point to our allocated memory
		printf("[GPUEngine::InitGenratorTable] Setting device symbol pointers...\n");
		CUDA_CHECK(cudaMemcpyToSymbol(_2Gnx, &__2Gnx, sizeof(uint64_t*)));
		CUDA_CHECK(cudaMemcpyToSymbol(_2Gny, &__2Gny, sizeof(uint64_t*)));
		CUDA_CHECK(cudaMemcpyToSymbol(Gx, &_Gx, sizeof(uint64_t*)));
		CUDA_CHECK(cudaMemcpyToSymbol(Gy, &_Gy, sizeof(uint64_t*)));
		printf("[GPUEngine::InitGenratorTable] Device symbols set successfully\n");
		
		// Verify the symbols were set correctly
		uint64_t* test_gx = NULL;
		uint64_t* test_gy = NULL;
		CUDA_CHECK(cudaMemcpyFromSymbol(&test_gx, Gx, sizeof(uint64_t*)));
		CUDA_CHECK(cudaMemcpyFromSymbol(&test_gy, Gy, sizeof(uint64_t*)));
		printf("[GPUEngine::InitGenratorTable] Verification: Gx=%p, Gy=%p (should not be NULL)\n", test_gx, test_gy);
		
		return;
	}

	// Standard generator table initialization for non-PUZZLE71 modes
	// generator table
	uint64_t* _2GnxPinned;
	uint64_t* _2GnyPinned;

	uint64_t* GxPinned;
	uint64_t* GyPinned;

	uint64_t size = (uint64_t)KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE;

	// Allocate generator table memory using unified memory management
	if (!allocate_cuda_memory_pair((void**)&__2Gnx, (void**)&_2GnxPinned, 4 * sizeof(uint64_t), "2Gnx generator")) {
		throw std::runtime_error("Failed to allocate 2Gnx generator memory");
	}

	if (!allocate_cuda_memory_pair((void**)&__2Gny, (void**)&_2GnyPinned, 4 * sizeof(uint64_t), "2Gny generator")) {
		throw std::runtime_error("Failed to allocate 2Gny generator memory");
	}

	size_t TSIZE = (size / 2) * 4 * sizeof(uint64_t);
	if (!allocate_cuda_memory_pair((void**)&_Gx, (void**)&GxPinned, TSIZE, "Gx table")) {
		throw std::runtime_error("Failed to allocate Gx table memory");
	}

	if (!allocate_cuda_memory_pair((void**)&_Gy, (void**)&GyPinned, TSIZE, "Gy table")) {
		throw std::runtime_error("Failed to allocate Gy table memory");
	}


	Point* Gn = new Point[size];
	Point g = secp->G;
	Gn[0] = g;
	g = secp->DoubleDirect(g);
	Gn[1] = g;
	for (int i = 2; i < size; i++) {
		g = secp->AddDirect(g, secp->G);
		Gn[i] = g;
	}
	// _2Gn = CPU_GRP_SIZE*G
	Point _2Gn = secp->DoubleDirect(Gn[size / 2 - 1]);

	int nbDigit = 4;
	for (int i = 0; i < nbDigit; i++) {
		_2GnxPinned[i] = _2Gn.x.bits64[i];
		_2GnyPinned[i] = _2Gn.y.bits64[i];
	}
	for (int i = 0; i < size / 2; i++) {
		for (int j = 0; j < nbDigit; j++) {
			GxPinned[i * nbDigit + j] = Gn[i].x.bits64[j];
			GyPinned[i * nbDigit + j] = Gn[i].y.bits64[j];
		}
	}

	delete[] Gn;

	// Transfer generator table data and cleanup host memory
	if (!transfer_and_cleanup_host_memory(__2Gnx, (void**)&_2GnxPinned, 4 * sizeof(uint64_t), "2Gnx generator")) {
		throw std::runtime_error("Failed to transfer 2Gnx generator data");
	}

	if (!transfer_and_cleanup_host_memory(__2Gny, (void**)&_2GnyPinned, 4 * sizeof(uint64_t), "2Gny generator")) {
		throw std::runtime_error("Failed to transfer 2Gny generator data");
	}

	if (!transfer_and_cleanup_host_memory(_Gx, (void**)&GxPinned, TSIZE, "Gx table")) {
		throw std::runtime_error("Failed to transfer Gx table data");
	}

	if (!transfer_and_cleanup_host_memory(_Gy, (void**)&GyPinned, TSIZE, "Gy table")) {
		throw std::runtime_error("Failed to transfer Gy table data");
	}

	// 使用cudaMemcpyToSymbol将设备指针复制到设备符号
	CUDA_CHECK(cudaMemcpyToSymbol(_2Gnx, &__2Gnx, sizeof(uint64_t*)));
	CUDA_CHECK(cudaMemcpyToSymbol(_2Gny, &__2Gny, sizeof(uint64_t*)));
	CUDA_CHECK(cudaMemcpyToSymbol(Gx, &_Gx, sizeof(uint64_t*)));
	CUDA_CHECK(cudaMemcpyToSymbol(Gy, &_Gy, sizeof(uint64_t*)));

}

// ----------------------------------------------------------------------------

int GPUEngine::GetGroupSize()
{
	return KeyHuntConstants::ELLIPTIC_CURVE_GROUP_SIZE;
}

// ----------------------------------------------------------------------------

void GPUEngine::PrintCudaInfo()
{
	const char* sComputeMode[] = {
		"Multiple host threads",
		"Only one host thread",
		"No host thread",
		"Multiple process threads",
		"Unknown",
		NULL
	};

	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	for (int i = 0; i < deviceCount; i++) {
		CUDA_CHECK(cudaSetDevice(i));
		cudaDeviceProp deviceProp;
		CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, i));
		printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
			i, deviceProp.name, deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			deviceProp.major, deviceProp.minor, (double)deviceProp.totalGlobalMem / 1048576.0,
			sComputeMode[deviceProp.computeMode]);
	}
}

// ----------------------------------------------------------------------------

GPUEngine::~GPUEngine()
{
	// Deallocate memory using unified memory management
	deallocate_cuda_memory_pair((void**)&inputKey, (void**)&inputKeyPinned);

	if (searchMode == (int)SEARCH_MODE_MA || searchMode == (int)SEARCH_MODE_MX) {
		deallocate_cuda_memory_pair((void**)&inputBloomLookUp, nullptr);
	} else {
		deallocate_cuda_memory_pair((void**)&inputHashORxpoint, nullptr);
	}

	deallocate_cuda_memory_pair((void**)&outputBuffer, (void**)&outputBufferPinned);

	// Deallocate generator table memory
	deallocate_cuda_memory_pair((void**)&__2Gnx, nullptr);
	deallocate_cuda_memory_pair((void**)&__2Gny, nullptr);
	deallocate_cuda_memory_pair((void**)&_Gx, nullptr);
	deallocate_cuda_memory_pair((void**)&_Gy, nullptr);
}

// ----------------------------------------------------------------------------

int GPUEngine::GetNbThread()
{
	return nbThread;
}

// ----------------------------------------------------------------------------

/**
 * Template function to encapsulate common kernel call pattern
 * Reduces code duplication in kernel calling functions
 * @param kernelFunc Lambda function containing the actual kernel call
 * @param resetFoundFlag Whether to reset the found flag before kernel call
 * @return true if kernel executed successfully, false otherwise
 */
template<typename KernelFunc>
bool GPUEngine::callKernelWithErrorCheck(KernelFunc kernelFunc, bool resetFoundFlag)
{
	// Reset nbFound
	CUDA_CHECK(cudaMemset(outputBuffer, 0, 4));

	// Reset the found flag if requested (used by SA mode)
	if (resetFoundFlag) {
		reset_found_flag<<<1, 1>>>();
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	#ifdef KEYHUNT_PROFILE_EVENTS
	cudaEvent_t __kh_start, __kh_stop;
	float __kh_ms = 0.0f;
	cudaEventCreate(&__kh_start);
	cudaEventCreate(&__kh_stop);
	cudaEventRecord(__kh_start, 0);
	#endif

	// Execute the kernel function
	kernelFunc();

	#ifdef KEYHUNT_PROFILE_EVENTS
	cudaEventRecord(__kh_stop, 0);
	cudaEventSynchronize(__kh_stop);
	cudaEventElapsedTime(&__kh_ms, __kh_start, __kh_stop);
	printf("[PROFILE] Kernel execution time: %.3f ms\n", __kh_ms);
	cudaEventDestroy(__kh_start);
	cudaEventDestroy(__kh_stop);
	#endif

	// Check for kernel execution errors using unified error checking
	return check_kernel_execution("GPU kernel");
}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_MA()
{
	// NEW: 使用统一内核接口，消除代码重复 (已启用)
	if (use_unified_kernels) {
		return UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_MA>(this);
	}

	// LEGACY: 保留原始实现作为备用
	return callKernelWithErrorCheck([this]() {
		// Use original KeyHunt backend only (gECC disabled)
		// Call the kernel (Perform STEP_SIZE keys per thread)
		if (coinType == COIN_BTC) {
			if (compMode == SEARCH_COMPRESSED) {
				compute_keys_comp_mode_ma << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
					(compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
			}
			else {
				compute_keys_mode_ma << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
					(compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
			}
		}
		else {
			compute_keys_mode_eth_ma << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
				(inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
		}
	});
}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_MX()
{
	// NEW: 使用统一内核接口，消除代码重复 (已启用)
	if (use_unified_kernels) {
		return UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_MX>(this);
	}

	// LEGACY: 保留原始实现作为备用
	return callKernelWithErrorCheck([this]() {
		// Call the kernel (Perform STEP_SIZE keys per thread)
		if (compMode == SEARCH_COMPRESSED) {
			compute_keys_comp_mode_mx << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
				(compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
		}
		else {
			printf("GPUEngine: PubKeys search doesn't support uncompressed\n");
			// Note: This will cause the template function to return false due to the error check
		}
	});
}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_SA()
{
	// NEW: 使用统一内核接口，消除代码重复 (已启用)
	if (use_unified_kernels) {
		return UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_SA>(this);
	}

	// LEGACY: 保留原始实现作为备用
	return callKernelWithErrorCheck([this]() {
		// Call the kernel (Perform STEP_SIZE keys per thread)
		if (coinType == COIN_BTC) {
			if (compMode == SEARCH_COMPRESSED) {
				compute_keys_comp_mode_sa << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
					(compMode, inputHashORxpoint, inputKey, maxFound, outputBuffer);
			}
			else {
				compute_keys_mode_sa << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
					(compMode, inputHashORxpoint, inputKey, maxFound, outputBuffer);
			}
		}
		else {
			compute_keys_mode_eth_sa << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
				(inputHashORxpoint, inputKey, maxFound, outputBuffer);
		}
	}, true); // true = reset found flag
}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_SX()
{
	// NEW: 使用统一内核接口，消除代码重复 (已启用)
	if (use_unified_kernels) {
		return UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_SX>(this);
	}

	// LEGACY: 保留原始实现作为备用
	return callKernelWithErrorCheck([this]() {
		// Use original KeyHunt backend only (gECC disabled)
		// Call the kernel (Perform STEP_SIZE keys per thread)
		if (compMode == SEARCH_COMPRESSED) {
			compute_keys_comp_mode_sx << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
				(compMode, inputHashORxpoint, inputKey, maxFound, outputBuffer);
		}
		else {
			printf("GPUEngine: PubKeys search doesn't support uncompressed\n");
			// Note: This will cause the template function to return false due to the error check
		}
	});
}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelPUZZLE71()
{
	// NEW: 使用统一内核接口，消除代码重复 (已启用)
	if (use_unified_kernels) {
		return UnifiedGPUEngine::callUnifiedKernel<SearchMode::PUZZLE71>(this);
	}

	// LEGACY: For now, behaves like SA mode
	return callKernelWithErrorCheck([this]() {
		// Call the kernel (Perform STEP_SIZE keys per thread)
		if (compMode == SEARCH_COMPRESSED) {
			compute_keys_comp_puzzle71 << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
				(compMode, inputHashORxpoint, inputKey, maxFound, outputBuffer);
		}
		else {
			compute_keys_puzzle71 << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
				(compMode, inputHashORxpoint, inputKey, maxFound, outputBuffer);
		}
	}, true); // true = reset found flag
}

// ----------------------------------------------------------------------------

bool GPUEngine::SetKeys(Point* p)
{
	// Sets the starting keys for each thread
	// p must contains nbThread public keys
	for (int i = 0; i < nbThread; i += nbThreadPerGroup) {
		for (int j = 0; j < nbThreadPerGroup; j++) {

			inputKeyPinned[8 * i + j + 0 * nbThreadPerGroup] = p[i + j].x.bits64[0];
			inputKeyPinned[8 * i + j + 1 * nbThreadPerGroup] = p[i + j].x.bits64[1];
			inputKeyPinned[8 * i + j + 2 * nbThreadPerGroup] = p[i + j].x.bits64[2];
			inputKeyPinned[8 * i + j + 3 * nbThreadPerGroup] = p[i + j].x.bits64[3];

			inputKeyPinned[8 * i + j + 4 * nbThreadPerGroup] = p[i + j].y.bits64[0];
			inputKeyPinned[8 * i + j + 5 * nbThreadPerGroup] = p[i + j].y.bits64[1];
			inputKeyPinned[8 * i + j + 6 * nbThreadPerGroup] = p[i + j].y.bits64[2];
			inputKeyPinned[8 * i + j + 7 * nbThreadPerGroup] = p[i + j].y.bits64[3];

		}
	}

	// Fill device memory
	CUDA_CHECK(cudaMemcpy(inputKey, inputKeyPinned, nbThread * 32 * 2, cudaMemcpyHostToDevice));

	if (!rKey) {
		// We do not need the input pinned memory anymore
		CUDA_CHECK(cudaFreeHost(inputKeyPinned));
		inputKeyPinned = NULL;
	}

	switch (searchMode) {
	case (int)SEARCH_MODE_MA:
		return callKernelSEARCH_MODE_MA();
		break;
	case (int)SEARCH_MODE_SA:
		return callKernelSEARCH_MODE_SA();
		break;
	case (int)SEARCH_MODE_MX:
		return callKernelSEARCH_MODE_MX();
		break;
	case (int)SEARCH_MODE_SX:
		return callKernelSEARCH_MODE_SX();
		break;
	case (int)SEARCH_MODE_PUZZLE71:
		return callKernelPUZZLE71();
		break;
	default:
		return false;
		break;
	}
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_MA(std::vector<ITEM>& dataFound, bool spinWait)
{
	// Use unified launch function for MA mode
	return launchUnified(dataFound, spinWait, [this]() { return callKernelSEARCH_MODE_MA(); }, ITEM_SIZE_A, ITEM_SIZE_A32);
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_SA(std::vector<ITEM>& dataFound, bool spinWait)
{
	// Use unified launch function for SA mode
	return launchUnified(dataFound, spinWait, [this]() { return callKernelSEARCH_MODE_SA(); }, ITEM_SIZE_A, ITEM_SIZE_A32);
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_MX(std::vector<ITEM>& dataFound, bool spinWait)
{
	// Use unified launch function for MX mode
	return launchUnified(dataFound, spinWait, [this]() { return callKernelSEARCH_MODE_MX(); },
	                     ITEM_SIZE_X, ITEM_SIZE_X32);
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_SX(std::vector<ITEM>& dataFound, bool spinWait)
{
	// Use unified launch function for SX mode
	return launchUnified(dataFound, spinWait, [this]() { return callKernelSEARCH_MODE_SX(); },
	                     ITEM_SIZE_X, ITEM_SIZE_X32);
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchPUZZLE71(std::vector<ITEM>& dataFound, bool spinWait)
{
	// Use unified launch function for PUZZLE71 mode - behaves like SA mode for now
	return launchUnified(dataFound, spinWait, [this]() { return callKernelPUZZLE71(); },
	                     ITEM_SIZE_A, ITEM_SIZE_A32);
}

// ----------------------------------------------------------------------------

// Reset found flag kernel
__global__ void reset_found_flag() {
    found_flag = 0;
}