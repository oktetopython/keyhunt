# Unit Tests CMakeLists.txt
# Component-level unit tests (T085-T086)

# PrivateKeyRange unit tests (T026)
add_executable(test_privatekey_range_unit test_privatekey_range_unit.cpp)
target_link_libraries(test_privatekey_range_unit 
    PRIVATE 
        GTest::gtest 
        GTest::gtest_main
        KeyhuntCore
)
target_include_directories(test_privatekey_range_unit PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/KeyhuntCore
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

# Add to test suite
add_test(NAME PrivateKeyRangeUnit COMMAND test_privatekey_range_unit)
set_tests_properties(PrivateKeyRangeUnit PROPERTIES
    LABELS "unit;models;T026"
    TIMEOUT 60
)

# Unit test target collection
set(UNIT_TEST_TARGETS
    test_privatekey_range_unit
)

# Custom target for all unit tests
add_custom_target(unit_tests 
    DEPENDS ${UNIT_TEST_TARGETS}
    COMMENT "Building all unit tests"
)

message(STATUS "Unit tests configured:")
message(STATUS "  - PrivateKeyRange unit tests (T026)")
message(STATUS "  - Additional unit tests pending (T085-T086)")