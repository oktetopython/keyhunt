# Contract Tests CMakeLists.txt
# API endpoint contract validation tests (T007-T015)
# These tests validate API endpoints against OpenAPI specifications
# Following TDD methodology - all tests MUST FAIL before implementation

# Contract test executables - these will fail compilation until API controllers are implemented
set(CONTRACT_TESTS
    test_scan_configure
    test_scan_start
    test_scan_status
    test_scan_pause
    test_scan_resume
    test_targets_configure
    test_validation_run
    test_results_matches
    test_experimental_report
)

foreach(test_name ${CONTRACT_TESTS})
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name}
        PRIVATE 
            GTest::gtest 
            GTest::gtest_main
            # KeyhuntCore  # Will be uncommented when core library is implemented
    )
    target_include_directories(${test_name} PRIVATE 
        ${CMAKE_SOURCE_DIR}/src/KeyhuntCore
        ${CMAKE_BINARY_DIR}/include
    )
    
    # Add to CTest with contract label
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "contract;tdd"
        TIMEOUT 30
        # WILL_FAIL TRUE  # Uncomment when we want tests to explicitly fail
    )
endforeach()

# Create contract test target group
add_custom_target(contract_tests)
foreach(test_name ${CONTRACT_TESTS})
    add_dependencies(contract_tests ${test_name})
endforeach()

message(STATUS "Contract tests configured - T007-T015 completed. Tests will fail until API implementation.")