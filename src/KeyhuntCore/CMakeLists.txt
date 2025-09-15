# KeyhuntCore Module CMakeLists.txt
# Modular architecture for GPU-accelerated Bitcoin private key search

# KeyhuntCore library target
set(KEYHUNT_CORE_SOURCES
    # Core configuration
    config.h
    
    # Main entry point
    main.cpp
    
    # Models (T026-T031) - Core data structures ✅ COMPLETED
    models/PrivateKeyRange.cpp
    models/TargetAddress.cpp
    models/CheckpointData.cpp
    models/ExperimentalResults.cpp
    models/GPUConfiguration.cpp
    models/ValidationReport.cpp
    
    # ECC Module (T032-T039) - Elliptic Curve Cryptography
    ecc/secp256k1_cpu_enhanced.cpp
    ecc/ecc_validation.cpp
    ecc/cpu_gpu_validator.cpp
    ecc/secp256k1_unified.cpp
    ecc/secp256k1_memory_manager.cpp
    ecc/secp256k1_math.cpp
    ecc/secp256k1_point.cpp
    crypto/gpu_random.cpp
    validation/ecc_validation_framework.cpp
    ecc/secp256k1_math.cu
    ecc/secp256k1_point.cu
    ecc/gpu_memory_manager.cu
    crypto/gpu_random.cu
    
    # Scanning Framework (T040-T043) - Private key range scanning with BitCrack optimizations
    scan/bitcrack_analysis.cpp
    scan/private_key_scanner.cpp
    scan/private_key_scanner_kernels.cu
    scan/checkpoint_manager.cpp
    
    # GPU Coordination (T043) - Multi-GPU coordination and load balancing
    gpu/multi_gpu_coordinator.cpp
    
    # Address Generation and Comparison (T044) - Bitcoin address pipeline with GPU optimization
    compare/bitcoin_address_generator.cpp
    compare/bloom_filter.cpp
    compare/bitcoin_address_kernels.cu
    
    # Single Target Comparison (T049) - GPU-optimized single target address comparison
    compare/single_compare.cpp
    compare/single_compare.cu
    compare/single_compare_kernels.cu
    
    # Base58 Encoding (T051) - GPU-accelerated Base58 encoding for address generation
    compare/base58.cpp
    compare/base58.cu
    
    # GPU Kernel Optimization (T052) - Comprehensive GPU kernel resource optimization
    optimization/gpu_kernel_optimizer.cu
    
    # Integrated Scanning Engine (T045) - Unified orchestration of all KeyhuntCUDA components
    engine/integrated_scanning_engine.cpp
    engine/scanning_engine_factory.cpp
    
    # Performance Optimization (T046) - Adaptive parameter tuning and bottleneck analysis
    optimization/performance_optimizer.cpp
    
    # Advanced Memory Management (T047) - GPU memory pools and smart allocation strategies
    memory/advanced_memory_manager.cpp
    
    # Logging and Monitoring Infrastructure (T048) - Comprehensive logging and monitoring system
    logging/logging_system.cpp
    monitoring/monitoring_system.cpp
    analytics/analytics_engine.cpp
    
    # Note: Additional sources will be added as they are implemented in subsequent tasks
    # Models (T027-T031): models/TargetAddress.cpp, models/CheckpointData.cpp, etc.
    # ECC module (T032-T039): ecc/secp256k1.cu, ecc/secp256k1_math.cu, etc.
    # Scanning framework (T044-T045): scan/performance_monitor.cpp, scan/result_aggregator.cpp, etc.
    # Address comparison (T046-T051): compare/address_gen.cu, compare/hash.cu, etc.
    # GPU coordination (T057-T060): gpu/coordinator.cpp, gpu/load_balancer.cu, etc.
    # Utilities (T063-T064): utils/logger.cpp, utils/monitor.cpp, etc.
    # CLI interface (T076): cli/keyhunt_cli.cpp, cli/command_parser.cpp
)

# Headers for public interface
set(KEYHUNT_CORE_HEADERS
    config.h
    # Note: Additional headers will be added as they are implemented in subsequent tasks
)

# Create KeyhuntCore library
add_library(KeyhuntCore STATIC ${KEYHUNT_CORE_SOURCES})

# Set properties for CUDA compilation
set_target_properties(KeyhuntCore PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Target compile features
target_compile_features(KeyhuntCore 
    PUBLIC 
        cxx_std_17
    PRIVATE
        cuda_std_17
)

# Include directories
target_include_directories(KeyhuntCore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(KeyhuntCore
    PUBLIC
        CUDA::cudart
        CUDA::curand
        secp256k1
        xxhash
    PRIVATE
        CUDA::cufft
        CUDA::cublas
)

# Link NCCL if available
if(HAVE_NCCL)
    target_link_libraries(KeyhuntCore PRIVATE nccl)
    target_compile_definitions(KeyhuntCore PRIVATE HAVE_NCCL=1)
endif()

# CUDA compilation flags
target_compile_options(KeyhuntCore 
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --extended-lambda
            --default-stream per-thread
        >
)

# Aggressive optimizations for CUDA
if(ENABLE_AGGRESSIVE_OPTIMIZATIONS)
    target_compile_options(KeyhuntCore 
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --maxrregcount=64
                --use_fast_math
                -Xptxas=-v
                -Xptxas=-O3
            >
    )
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(KeyhuntCore 
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:
                -march=native
                -mtune=native
                -ffast-math
                -funroll-loops
            >
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(KeyhuntCore 
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:
                -march=native
                -mtune=native
                -ffast-math
                -funroll-loops
            >
    )
endif()

# Preprocessor definitions
target_compile_definitions(KeyhuntCore
    PUBLIC
        KEYHUNT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        KEYHUNT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        KEYHUNT_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    PRIVATE
        $<$<CONFIG:Debug>:KEYHUNT_DEBUG=1>
        $<$<CONFIG:Release>:KEYHUNT_RELEASE=1>
)

# Create main executable
add_executable(keyhunt
    main.cpp
)

# Link main executable with core library
target_link_libraries(keyhunt PRIVATE KeyhuntCore)

# Set executable properties
set_target_properties(keyhunt PROPERTIES
    OUTPUT_NAME "keyhunt"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Module-specific subdirectories for detailed implementations
# These will be created as the project develops

# ECC module
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ecc/CMakeLists.txt)
    add_subdirectory(ecc)
endif()

# Scanning module  
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scan/CMakeLists.txt)
    add_subdirectory(scan)
endif()

# Comparison module
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/compare/CMakeLists.txt)
    add_subdirectory(compare)
endif()

# GPU coordination module
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/CMakeLists.txt)
    add_subdirectory(gpu)
endif()

# Utilities module
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/utils/CMakeLists.txt)
    add_subdirectory(utils)
endif()

# CLI module
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cli/CMakeLists.txt)
    add_subdirectory(cli)
endif()

# Installation rules
install(TARGETS KeyhuntCore keyhunt
    EXPORT KeyhuntCoreTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(FILES ${KEYHUNT_CORE_HEADERS}
    DESTINATION include/keyhunt
    COMPONENT development
)

# Create package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/KeyhuntCoreConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT KeyhuntCoreTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/KeyhuntCoreTargets.cmake"
    NAMESPACE KeyhuntCore::
)

# Status message
message(STATUS "KeyhuntCore module configured with:")
message(STATUS "  Sources: ${KEYHUNT_CORE_SOURCES}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Aggressive optimizations: ${ENABLE_AGGRESSIVE_OPTIMIZATIONS}")