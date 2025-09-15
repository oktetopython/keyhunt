# Scientific Validation Tests CMakeLists.txt
# CPU/GPU consistency validation tests (T016-T020)
# These tests ensure GPU computations match CPU reference implementations
# Following TDD methodology - all tests MUST FAIL before implementation

# Validation test executables - these will fail compilation until validation modules are implemented
set(VALIDATION_TESTS
    test_ecc_consistency
    test_ecc_properties
    test_address_pipeline
    test_multi_gpu
    test_checkpoint_recovery
)

foreach(test_name ${VALIDATION_TESTS})
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name}
        PRIVATE 
            GTest::gtest 
            GTest::gtest_main
            secp256k1  # CPU reference for validation
            # KeyhuntCore  # Will be uncommented when core library is implemented
    )
    target_include_directories(${test_name} PRIVATE 
        ${CMAKE_SOURCE_DIR}/src/KeyhuntCore
        ${CMAKE_BINARY_DIR}/include
        ${SECP256K1_INCLUDE_DIRS}
    )
    
    # Add compile definitions for scientific validation
    target_compile_definitions(${test_name} PRIVATE
        SCIENTIFIC_VALIDATION_ENABLED=1
        PRECISION_THRESHOLD=1e-10
        CPU_REFERENCE_AVAILABLE=1
    )
    
    # Add to CTest with validation label
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "validation;scientific;tdd"
        TIMEOUT 300  # Longer timeout for scientific validation
        # WILL_FAIL TRUE  # Uncomment when we want tests to explicitly fail
    )
endforeach()

# Create validation test target group
add_custom_target(validation_tests)
foreach(test_name ${VALIDATION_TESTS})
    add_dependencies(validation_tests ${test_name})
endforeach()

message(STATUS "Scientific validation tests configured - T016-T020 completed. Tests will fail until implementation.")