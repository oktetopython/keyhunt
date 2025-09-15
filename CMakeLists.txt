cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project configuration
project(KeyhuntCUDA 
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated Bitcoin private key search system for Bitcoin puzzle challenges"
    LANGUAGES CXX CUDA
)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Configuration  
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
enable_language(CUDA)

# Minimum CUDA version check
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
    message(FATAL_ERROR "CUDA 11.0 or higher is required. Found: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# CUDA Architecture Targets
# Turing (7.5) and newer architectures for optimal performance
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures to build for")

# Build configuration
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")

# Position Independent Code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find required packages
find_package(CUDAToolkit 11.0 REQUIRED)

# Find libsecp256k1 for CPU validation reference
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(SECP256K1 libsecp256k1)
endif()

if(NOT SECP256K1_FOUND)
    find_library(SECP256K1_LIBRARY secp256k1)
    find_path(SECP256K1_INCLUDE_DIR secp256k1.h)
    
    if(SECP256K1_LIBRARY AND SECP256K1_INCLUDE_DIR)
        set(SECP256K1_FOUND TRUE)
        set(SECP256K1_LIBRARIES ${SECP256K1_LIBRARY})
        set(SECP256K1_INCLUDE_DIRS ${SECP256K1_INCLUDE_DIR})
    endif()
endif()

if(SECP256K1_FOUND)
    message(STATUS "Found libsecp256k1: ${SECP256K1_LIBRARIES}")
    message(STATUS "  Include directories: ${SECP256K1_INCLUDE_DIRS}")
    
    # Create imported target for secp256k1
    if(NOT TARGET secp256k1)
        add_library(secp256k1 UNKNOWN IMPORTED)
        if(SECP256K1_LINK_LIBRARIES)
            set_target_properties(secp256k1 PROPERTIES
                IMPORTED_LOCATION ${SECP256K1_LINK_LIBRARIES}
                INTERFACE_INCLUDE_DIRECTORIES "${SECP256K1_INCLUDE_DIRS}"
            )
        else()
            set_target_properties(secp256k1 PROPERTIES
                INTERFACE_LINK_LIBRARIES "${SECP256K1_LIBRARIES}"
                INTERFACE_INCLUDE_DIRECTORIES "${SECP256K1_INCLUDE_DIRS}"
            )
        endif()
    endif()
    
    # Enable scientific validation with CPU reference
    add_compile_definitions(
        SECP256K1_CPU_VALIDATION=1
        PRECISION_THRESHOLD=1e-10
        CPU_REFERENCE_AVAILABLE=1
    )
    
    # Test secp256k1 functionality during configuration
    message(STATUS "Configuring libsecp256k1 for scientific validation...")
    set(SECP256K1_VALIDATION_ENABLED ON CACHE BOOL "Enable CPU/GPU consistency validation")
    
else()
    message(WARNING "libsecp256k1-dev not found. CPU validation reference will be disabled.")
    message(STATUS "  Install with: sudo apt-get install libsecp256k1-dev")
    message(STATUS "  This is REQUIRED for scientific validation and precision testing.")
    
    # Create a dummy target to prevent build errors
    if(NOT TARGET secp256k1)
        add_library(secp256k1 INTERFACE)
    endif()
    
    # Disable validation features
    add_compile_definitions(
        NO_SECP256K1_VALIDATION=1
        CPU_REFERENCE_AVAILABLE=0
    )
    set(SECP256K1_VALIDATION_ENABLED OFF CACHE BOOL "CPU validation disabled - no libsecp256k1")
    
    # Add to installation requirements
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(FATAL_ERROR "libsecp256k1-dev is REQUIRED for Release builds with scientific validation")
    endif()
endif()

# NCCL for multi-GPU communication and coordination
# Try multiple approaches to find NCCL
find_library(NCCL_LIBRARY nccl 
    PATHS
        /usr/local/nccl/lib
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDAToolkit_LIBRARY_DIR}
    DOC "NCCL library for multi-GPU communication"
)

find_path(NCCL_INCLUDE_DIR nccl.h
    PATHS
        /usr/local/nccl/include
        /usr/local/include
        /usr/include
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    DOC "NCCL include directory"
)

if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
    message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
    message(STATUS "  NCCL include: ${NCCL_INCLUDE_DIR}")
    
    # Create imported target for NCCL
    add_library(nccl UNKNOWN IMPORTED)
    set_target_properties(nccl PROPERTIES
        IMPORTED_LOCATION ${NCCL_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${NCCL_INCLUDE_DIR}
    )
    
    # Enable multi-GPU features
    set(HAVE_NCCL ON)
    add_compile_definitions(
        NCCL_AVAILABLE=1
        MULTI_GPU_SUPPORT=1
    )
    
    # Multi-GPU configuration options
    set(MAX_GPU_DEVICES 8 CACHE STRING "Maximum number of GPUs to support")
    set(NCCL_ENABLED ON CACHE BOOL "Enable NCCL-based multi-GPU coordination")
    
    message(STATUS "Multi-GPU support enabled with NCCL")
    message(STATUS "  Maximum GPU devices: ${MAX_GPU_DEVICES}")
    
else()
    message(WARNING "NCCL not found. Multi-GPU support will be limited.")
    message(STATUS "  Install NCCL for optimal multi-GPU performance:")
    message(STATUS "    - Ubuntu: sudo apt-get install libnccl2 libnccl-dev")
    message(STATUS "    - From source: https://github.com/NVIDIA/nccl")
    
    # Fallback: basic multi-GPU without NCCL
    set(HAVE_NCCL OFF)
    add_compile_definitions(
        NCCL_AVAILABLE=0
        MULTI_GPU_SUPPORT=0
    )
    set(NCCL_ENABLED OFF CACHE BOOL "NCCL not available")
    
    # Create dummy target to prevent build errors
    add_library(nccl INTERFACE)
    
    # Limited multi-GPU without NCCL
    message(STATUS "Basic multi-GPU coordination will be used (CUDA streams only)")
endif()

# Configuration header
configure_file(
    ${CMAKE_SOURCE_DIR}/src/KeyhuntCore/config.h.in
    ${CMAKE_BINARY_DIR}/include/keyhunt/config.h
    @ONLY
)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src/KeyhuntCore
    ${CMAKE_BINARY_DIR}/include
    ${SECP256K1_INCLUDE_DIR}
)

# Enable testing
enable_testing()

# Build options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build benchmark suite" OFF)
option(ENABLE_AGGRESSIVE_OPTIMIZATIONS "Enable aggressive GPU optimizations" ON)

# Scientific validation requirements
add_compile_definitions(
    SCIENTIFIC_VALIDATION_ENABLED=1
    PRECISION_THRESHOLD=1e-10
)

# Platform-specific optimizations
if(ENABLE_AGGRESSIVE_OPTIMIZATIONS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=64 --use_fast_math")
    add_compile_definitions(AGGRESSIVE_OPTIMIZATIONS=1)
endif()

# Add subdirectories
add_subdirectory(src/KeyhuntCore)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Install configuration
install(TARGETS keyhunt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Package configuration  
set(CPACK_PACKAGE_NAME "KeyhuntCUDA")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VENDOR "Keyhunt Research Team")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

include(CPack)

# Status summary
message(STATUS "KeyhuntCUDA Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  libsecp256k1: ${SECP256K1_LIBRARY}")
message(STATUS "  NCCL support: ${HAVE_NCCL}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Aggressive optimizations: ${ENABLE_AGGRESSIVE_OPTIMIZATIONS}")

# Display GPU capability check
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA toolkit root: ${CUDAToolkit_ROOT}")
endif()