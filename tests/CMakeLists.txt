# Test configuration for Keyhunt-CUDA
# Scientific validation and TDD framework

# Test framework configuration
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    # Use local GoogleTest from specs directory
    set(GOOGLETEST_ROOT "${CMAKE_SOURCE_DIR}/specs/001-keyhunt-cuda-puzzle/src/googleTest")
    
    if(EXISTS ${GOOGLETEST_ROOT}/CMakeLists.txt)
        message(STATUS "Using local GoogleTest from: ${GOOGLETEST_ROOT}")
        
        # Add GoogleTest subdirectory
        add_subdirectory(${GOOGLETEST_ROOT} ${CMAKE_BINARY_DIR}/googletest)
        
        # Create aliases for consistency
        if(NOT TARGET GTest::gtest)
            add_library(GTest::gtest ALIAS gtest)
            add_library(GTest::gtest_main ALIAS gtest_main)
            add_library(GTest::gmock ALIAS gmock)
            add_library(GTest::gmock_main ALIAS gmock_main)
        endif()
    else()
        message(WARNING "Local GoogleTest not found at ${GOOGLETEST_ROOT}")
        message(STATUS "Trying to download GoogleTest from internet...")
        
        # Fallback: Download GoogleTest if local copy not available
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/v1.14.0.zip
            DOWNLOAD_EXTRACT_TIMESTAMP true
        )
        
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        
        # Create aliases for consistency
        add_library(GTest::gtest ALIAS gtest)
        add_library(GTest::gtest_main ALIAS gtest_main)
        add_library(GTest::gmock ALIAS gmock)
        add_library(GTest::gmock_main ALIAS gmock_main)
    endif()
endif()

# Include test directories
include_directories(${CMAKE_SOURCE_DIR}/src/KeyhuntCore)

# Test categories based on TDD approach
set(TEST_CATEGORIES
    contract
    validation  
    integration
    unit
)

# Contract tests (T007-T015) - API endpoint validation
add_subdirectory(contract)

# Scientific validation tests (T016-T020) - CPU/GPU consistency
add_subdirectory(validation)

# Integration tests (T021-T025) - User story validation
add_subdirectory(integration)

# Unit tests (T085-T086) - Component testing
add_subdirectory(unit)

# Custom test targets
add_custom_target(test_contract
    COMMAND ${CMAKE_CTEST_COMMAND} -L contract -V
    DEPENDS contract_tests
    COMMENT "Running contract tests"
)

add_custom_target(test_validation
    COMMAND ${CMAKE_CTEST_COMMAND} -L validation -V  
    DEPENDS validation_tests
    COMMENT "Running scientific validation tests"
)

add_custom_target(test_integration
    COMMAND ${CMAKE_CTEST_COMMAND} -L integration -V
    DEPENDS integration_tests
    COMMENT "Running integration tests"
)

add_custom_target(test_unit
    COMMAND ${CMAKE_CTEST_COMMAND} -L unit -V
    DEPENDS unit_tests
    COMMENT "Running unit tests"
)

# Scientific validation target
add_custom_target(validate_scientific
    COMMAND ${CMAKE_CTEST_COMMAND} -L "validation|contract" -V
    DEPENDS validation_tests contract_tests
    COMMENT "Running scientific validation with contract verification"
)

# Performance testing target
add_custom_target(test_performance
    COMMAND ${CMAKE_CTEST_COMMAND} -L performance -V
    COMMENT "Running performance validation tests"
)

# Scientific validation with precision requirements (<1e-10)
add_custom_target(validate_precision
    COMMAND ${CMAKE_CTEST_COMMAND} -L "precision|ecc" -V
    COMMENT "Validating computational precision requirements"
)

# Multi-GPU validation target
add_custom_target(validate_multi_gpu
    COMMAND ${CMAKE_CTEST_COMMAND} -L "multi_gpu" -V
    COMMENT "Validating multi-GPU coordination and scaling"
)

# Comprehensive scientific validation suite
add_custom_target(validate_all
    COMMAND ${CMAKE_CTEST_COMMAND} -L "validation|contract|precision|multi_gpu" -V
    DEPENDS validation_tests contract_tests
    COMMENT "Complete scientific validation suite"
)

# TDD workflow target - ensure tests are written and failing first
add_custom_target(verify_tdd
    COMMAND echo "Verifying TDD compliance: All tests must fail before implementation"
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMENT "TDD Verification: Tests should fail before implementation"
)

# Test configuration summary
message(STATUS "Test Configuration:")
message(STATUS "  Google Test: ${GTest_FOUND}")
message(STATUS "  Test categories: ${TEST_CATEGORIES}")
message(STATUS "  Scientific validation: Enabled")
message(STATUS "  TDD compliance: Required")

# Quick setup test to verify GoogleTest integration
add_executable(setup_test setup_test.cpp)
target_link_libraries(setup_test 
    PRIVATE 
        GTest::gtest 
        GTest::gtest_main
        KeyhuntCore
)
target_include_directories(setup_test PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/KeyhuntCore
    ${CMAKE_BINARY_DIR}/include
)

# Add to test suite
add_test(NAME SetupTest COMMAND setup_test)
set_tests_properties(SetupTest PROPERTIES
    LABELS "setup;quick"
    TIMEOUT 30
)