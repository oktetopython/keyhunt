/*
 * This file is part of the KeyHunt-Cuda distribution (https://github.com/your-repo/keyhunt-cuda).
 * Copyright (c) 2025 Your Name.
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
#define UNIFIED_ENGINE_COMPILATION
#include "GPUEngine_Unified.h"
#include "GPUCompute_Unified.h"
#include "GPUCompute_Unified.cuh"
#include "../Constants.h"

// Implementation of template specializations for each search mode
template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_MA>(GPUEngine* engine) {
    // Reset nbFound
    cudaMemset(engine->getOutputBuffer(), 0, 4);
    
    // Call the kernel (Perform STEP_SIZE keys per thread)
    if (engine->getCoinType() == COIN_BTC) {
        if (engine->getCompMode() == SEARCH_COMPRESSED) {
            compute_keys_comp_mode_ma<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
                engine->getCompMode(), 
                engine->getInputBloomLookUp(), 
                engine->getBloomBits(), 
                engine->getBloomHashes(), 
                engine->getInputKey(), 
                engine->getMaxFound(), 
                engine->getOutputBuffer());
        } else {
            compute_keys_mode_ma<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
                engine->getCompMode(), 
                engine->getInputBloomLookUp(), 
                engine->getBloomBits(), 
                engine->getBloomHashes(), 
                engine->getInputKey(), 
                engine->getMaxFound(), 
                engine->getOutputBuffer());
        }
    } else {
        compute_keys_mode_eth_ma<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
            engine->getInputBloomLookUp(), 
            engine->getBloomBits(), 
            engine->getBloomHashes(), 
            engine->getInputKey(), 
            engine->getMaxFound(), 
            engine->getOutputBuffer());
    }
    
    // Check for kernel execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel MODE_MA failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_SA>(GPUEngine* engine) {
    // Reset nbFound
    cudaMemset(engine->getOutputBuffer(), 0, 4);
    
    // Reset the found flag (used by SA mode)
    reset_found_flag<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // Call the kernel (Perform STEP_SIZE keys per thread)
    if (engine->getCoinType() == COIN_BTC) {
        if (engine->getCompMode() == SEARCH_COMPRESSED) {
            compute_keys_comp_mode_sa<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
                engine->getCompMode(), 
                engine->getInputHashORxpoint(), 
                engine->getInputKey(), 
                engine->getMaxFound(), 
                engine->getOutputBuffer());
        } else {
            compute_keys_mode_sa<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
                engine->getCompMode(), 
                engine->getInputHashORxpoint(), 
                engine->getInputKey(), 
                engine->getMaxFound(), 
                engine->getOutputBuffer());
        }
    } else {
        compute_keys_mode_eth_sa<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
            engine->getInputHashORxpoint(), 
            engine->getInputKey(), 
            engine->getMaxFound(), 
            engine->getOutputBuffer());
    }
    
    // Check for kernel execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel MODE_SA failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_MX>(GPUEngine* engine) {
    // Reset nbFound
    cudaMemset(engine->getOutputBuffer(), 0, 4);
    
    // Call the kernel (Perform STEP_SIZE keys per thread)
    if (engine->getCompMode() == SEARCH_COMPRESSED) {
        compute_keys_comp_mode_mx<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
            engine->getCompMode(), 
            engine->getInputBloomLookUp(), 
            engine->getBloomBits(), 
            engine->getBloomHashes(), 
            engine->getInputKey(), 
            engine->getMaxFound(), 
            engine->getOutputBuffer());
    } else {
        printf("GPUEngine: PubKeys search doesn't support uncompressed\n");
        return false;
    }
    
    // Check for kernel execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel MODE_MX failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_SX>(GPUEngine* engine) {
    // Reset nbFound
    cudaMemset(engine->getOutputBuffer(), 0, 4);
    
    // Call the kernel (Perform STEP_SIZE keys per thread)
    if (engine->getCompMode() == SEARCH_COMPRESSED) {
        compute_keys_comp_mode_sx<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
            engine->getCompMode(), 
            engine->getInputHashORxpoint(), 
            engine->getInputKey(), 
            engine->getMaxFound(), 
            engine->getOutputBuffer());
    } else {
        printf("GPUEngine: PubKeys search doesn't support uncompressed\n");
        return false;
    }
    
    // Check for kernel execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel MODE_SX failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_ETH_MA>(GPUEngine* engine) {
    // Reset nbFound
    cudaMemset(engine->getOutputBuffer(), 0, 4);
    
    // Call the kernel (Perform STEP_SIZE keys per thread)
    compute_keys_mode_eth_ma<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
        engine->getInputBloomLookUp(), 
        engine->getBloomBits(), 
        engine->getBloomHashes(), 
        engine->getInputKey(), 
        engine->getMaxFound(), 
        engine->getOutputBuffer());
    
    // Check for kernel execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel MODE_ETH_MA failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_ETH_SA>(GPUEngine* engine) {
    // Reset nbFound
    cudaMemset(engine->getOutputBuffer(), 0, 4);
    
    // Reset the found flag (used by SA mode)
    reset_found_flag<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // Call the kernel (Perform STEP_SIZE keys per thread)
    compute_keys_mode_eth_sa<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
        engine->getInputHashORxpoint(), 
        engine->getInputKey(), 
        engine->getMaxFound(), 
        engine->getOutputBuffer());
    
    // Check for kernel execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel MODE_ETH_SA failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::PUZZLE71>(GPUEngine* engine) {
    // Reset nbFound
    cudaMemset(engine->getOutputBuffer(), 0, 4);
    
    // Reset the found flag (used by PUZZLE71 mode)
    reset_found_flag<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // Call the PUZZLE71 kernel with compressed mode (Bitcoin Puzzle #71)
    if (engine->getCompMode() == SEARCH_COMPRESSED) {
        compute_keys_comp_puzzle71<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
            engine->getCompMode(), 
            engine->getInputHashORxpoint(),  // PUZZLE71 uses hardcoded target
            engine->getInputKey(), 
            engine->getMaxFound(), 
            engine->getOutputBuffer());
    } else {
        // PUZZLE71 mode - use main kernel for uncompressed (though uncommon)
        compute_keys_puzzle71<<<engine->getNbThread() / engine->getNbThreadPerGroup(), engine->getNbThreadPerGroup()>>>(
            engine->getCompMode(), 
            engine->getInputHashORxpoint(),
            engine->getInputKey(), 
            engine->getMaxFound(), 
            engine->getOutputBuffer());
    }
    
    // Check for kernel execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel PUZZLE71 failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}
