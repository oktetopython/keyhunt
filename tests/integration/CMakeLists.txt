# Integration Tests CMakeLists.txt
# User story validation tests (T021-T025)
# These tests validate complete user workflows end-to-end
# Following TDD methodology - all tests MUST FAIL before implementation

# Integration test executables - these will fail compilation until full system is implemented
set(INTEGRATION_TESTS
    test_basic_range_search
    test_checkpoint_resume
    test_multi_gpu_config
    test_performance_monitoring
    test_scientific_reporting
)

foreach(test_name ${INTEGRATION_TESTS})
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
    
    # Add compile definitions for integration testing
    target_compile_definitions(${test_name} PRIVATE
        INTEGRATION_TEST_ENABLED=1
        SCIENTIFIC_VALIDATION_ENABLED=1
        PRECISION_THRESHOLD=1e-10
        CPU_REFERENCE_AVAILABLE=1
    )
    
    # Add to CTest with integration label
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "integration;user_story;tdd;end_to_end"
        TIMEOUT 600  # Longer timeout for integration tests
        # WILL_FAIL TRUE  # Uncomment when we want tests to explicitly fail
    )
endforeach()

# Create integration test target group
add_custom_target(integration_tests)
foreach(test_name ${INTEGRATION_TESTS})
    add_dependencies(integration_tests ${test_name})
endforeach()

message(STATUS "Integration tests configured - T021-T025 completed. Tests will fail until full implementation.")