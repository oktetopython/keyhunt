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

#ifndef GPU_ENGINE_UNIFIED_H
#define GPU_ENGINE_UNIFIED_H

#include "GPUEngine.h"
#include "SearchMode.h"

// Convenience macros for calling unified kernels
#define CALL_UNIFIED_KERNEL_MA(engine) UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_MA>(engine)
#define CALL_UNIFIED_KERNEL_SA(engine) UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_SA>(engine)
#define CALL_UNIFIED_KERNEL_MX(engine) UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_MX>(engine)
#define CALL_UNIFIED_KERNEL_SX(engine) UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_SX>(engine)
#define CALL_UNIFIED_KERNEL_ETH_MA(engine) UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_ETH_MA>(engine)
#define CALL_UNIFIED_KERNEL_ETH_SA(engine) UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_ETH_SA>(engine)
#define CALL_UNIFIED_KERNEL_PUZZLE71(engine) UnifiedGPUEngine::callUnifiedKernel<SearchMode::PUZZLE71>(engine)

// Unified GPU engine class
class UnifiedGPUEngine {
public:
    // Template function to call unified kernel
    template<SearchMode Mode>
    static bool callUnifiedKernel(GPUEngine* engine);
};

// Template specializations for each search mode
template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_MA>(GPUEngine* engine);

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_SA>(GPUEngine* engine);

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_MX>(GPUEngine* engine);

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_SX>(GPUEngine* engine);

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_ETH_MA>(GPUEngine* engine);

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::MODE_ETH_SA>(GPUEngine* engine);

template<>
bool UnifiedGPUEngine::callUnifiedKernel<SearchMode::PUZZLE71>(GPUEngine* engine);

#endif // GPU_ENGINE_UNIFIED_H
