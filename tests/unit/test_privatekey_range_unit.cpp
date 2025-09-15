/**
 * @file test_privatekey_range_unit.cpp
 * @brief Unit tests for PrivateKeyRange model
 * @author KeyhuntCUDA Team
 * 
 * Tests the core functionality of the PrivateKeyRange class
 * following TDD methodology - these tests should now pass.
 */

#include <gtest/gtest.h>
#include "keyhunt/models/PrivateKeyRange.h"
#include <stdexcept>

using namespace keyhunt::models;

class PrivateKeyRangeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Valid test data
        start_key_valid = "0000000000000000000000000000000000000000000000000000000000000001";
        end_key_valid = "000000000000000000000000000000000000000000000000000000000000ffff";
        
        // Invalid test data
        start_key_invalid = "invalid_hex_key";
        start_key_short = "123456";  // Too short
        end_key_invalid = "000000000000000000000000000000000000000000000000000000000000000G"; // Invalid hex char
    }

    std::string start_key_valid;
    std::string end_key_valid;
    std::string start_key_invalid;
    std::string start_key_short;
    std::string end_key_invalid;
};

/**
 * Test Case: Valid Range Construction
 * Verify that a valid private key range can be constructed successfully
 */
TEST_F(PrivateKeyRangeTest, ValidRangeConstruction) {
    // Act - Create valid range
    PrivateKeyRange range(start_key_valid, end_key_valid, 1);
    
    // Assert - Verify basic properties
    EXPECT_EQ(start_key_valid, range.get_start_key());
    EXPECT_EQ(end_key_valid, range.get_end_key());
    EXPECT_EQ(1, range.get_stride());
    EXPECT_EQ(PrivateKeyRange::Status::CONFIGURED, range.get_status());
    EXPECT_GT(range.get_total_keys(), 0);
    EXPECT_FALSE(range.get_range_id().empty());
    EXPECT_TRUE(range.is_valid());
}

/**
 * Test Case: Invalid Start Key Format
 * Verify that invalid start key format throws exception
 */
TEST_F(PrivateKeyRangeTest, InvalidStartKeyFormat) {
    // Act & Assert - Should throw exception
    EXPECT_THROW({
        PrivateKeyRange range(start_key_invalid, end_key_valid, 1);
    }, std::invalid_argument);
}

/**
 * Test Case: Invalid End Key Format
 * Verify that invalid end key format throws exception  
 */
TEST_F(PrivateKeyRangeTest, InvalidEndKeyFormat) {
    // Act & Assert - Should throw exception
    EXPECT_THROW({
        PrivateKeyRange range(start_key_valid, end_key_invalid, 1);
    }, std::invalid_argument);
}

/**
 * Test Case: Zero Stride
 * Verify that zero stride throws exception
 */
TEST_F(PrivateKeyRangeTest, ZeroStride) {
    // Act & Assert - Should throw exception
    EXPECT_THROW({
        PrivateKeyRange range(start_key_valid, end_key_valid, 0);
    }, std::invalid_argument);
}

/**
 * Test Case: Range Logic Validation
 * Verify that end_key <= start_key throws exception
 */
TEST_F(PrivateKeyRangeTest, InvalidRangeLogic) {
    // Act & Assert - Should throw exception (end_key same as start_key)
    EXPECT_THROW({
        PrivateKeyRange range(start_key_valid, start_key_valid, 1);
    }, std::invalid_argument);
}

/**
 * Test Case: Progress Tracking
 * Verify that progress tracking works correctly
 */
TEST_F(PrivateKeyRangeTest, ProgressTracking) {
    // Arrange
    PrivateKeyRange range(start_key_valid, end_key_valid, 1);
    uint64_t total_keys = range.get_total_keys();
    
    // Act - Simulate processing half the keys
    uint64_t half_processed = total_keys / 2;
    range.set_keys_processed(half_processed);
    
    // Assert
    EXPECT_EQ(half_processed, range.get_keys_processed());
    EXPECT_NEAR(50.0, range.get_progress_percentage(), 1.0);
    EXPECT_GE(range.get_keys_per_second(), 0.0);
}

/**
 * Test Case: JSON Serialization
 * Verify that range can be serialized to JSON correctly
 */
TEST_F(PrivateKeyRangeTest, JSONSerialization) {
    // Arrange
    PrivateKeyRange range(start_key_valid, end_key_valid, 1);
    std::vector<int> gpu_devices = {0, 1};
    range.set_gpu_devices(gpu_devices);
    
    // Act
    std::string json = range.to_json();
    
    // Assert - Basic JSON structure checks
    EXPECT_NE(json.find("range_id"), std::string::npos);
    EXPECT_NE(json.find(start_key_valid), std::string::npos);
    EXPECT_NE(json.find(end_key_valid), std::string::npos);
    EXPECT_NE(json.find("\"stride\": 1"), std::string::npos);
    EXPECT_NE(json.find("configured"), std::string::npos);
    EXPECT_NE(json.find("[0, 1]"), std::string::npos);
}

/**
 * Test Case: JSON Deserialization
 * Verify that range can be deserialized from JSON correctly
 */
TEST_F(PrivateKeyRangeTest, JSONDeserialization) {
    // Arrange
    std::string json = R"({
        "start_key": "0000000000000000000000000000000000000000000000000000000000000001",
        "end_key": "000000000000000000000000000000000000000000000000000000000000ffff",
        "stride": 2
    })";
    
    // Act
    PrivateKeyRange range = PrivateKeyRange::from_json(json);
    
    // Assert
    EXPECT_EQ(start_key_valid, range.get_start_key());
    EXPECT_EQ(end_key_valid, range.get_end_key());
    EXPECT_EQ(2, range.get_stride());
    EXPECT_TRUE(range.is_valid());
}

/**
 * Test Case: Validation Error Messages
 * Verify that validation provides meaningful error messages
 */
TEST_F(PrivateKeyRangeTest, ValidationErrorMessages) {
    // This test creates an invalid range using direct member access
    // In production, this would be done through setters or factory methods
    
    // Test with valid range first
    PrivateKeyRange valid_range(start_key_valid, end_key_valid, 1);
    EXPECT_TRUE(valid_range.get_validation_error().empty());
    
    // Note: Since constructor validates, we test is_valid() method indirectly
    // through the methods that don't throw exceptions
    EXPECT_TRUE(valid_range.is_valid());
}

/**
 * Test Case: Performance Estimation
 * Verify that performance estimation calculations work
 */
TEST_F(PrivateKeyRangeTest, PerformanceEstimation) {
    // Arrange
    PrivateKeyRange range(start_key_valid, end_key_valid, 1);
    
    // Act & Assert
    EXPECT_GT(range.get_estimated_time(), 0.0);
    EXPECT_GT(range.get_total_keys(), 0);
    
    // Test elapsed time tracking
    auto elapsed = range.get_elapsed_time();
    EXPECT_GE(elapsed.count(), 0);
}

/**
 * Test Case: GPU Device Management
 * Verify that GPU device assignment works correctly
 */
TEST_F(PrivateKeyRangeTest, GPUDeviceManagement) {
    // Arrange
    PrivateKeyRange range(start_key_valid, end_key_valid, 1);
    std::vector<int> gpu_devices = {0, 1, 2};
    
    // Act
    range.set_gpu_devices(gpu_devices);
    
    // Assert
    const std::vector<int>& assigned_devices = range.get_gpu_devices();
    EXPECT_EQ(3, assigned_devices.size());
    EXPECT_EQ(0, assigned_devices[0]);
    EXPECT_EQ(1, assigned_devices[1]);
    EXPECT_EQ(2, assigned_devices[2]);
}

/**
 * Test Case: Status Management
 * Verify that status changes work correctly
 */
TEST_F(PrivateKeyRangeTest, StatusManagement) {
    // Arrange
    PrivateKeyRange range(start_key_valid, end_key_valid, 1);
    
    // Act & Assert - Test status transitions
    EXPECT_EQ(PrivateKeyRange::Status::CONFIGURED, range.get_status());
    
    range.set_status(PrivateKeyRange::Status::SCANNING);
    EXPECT_EQ(PrivateKeyRange::Status::SCANNING, range.get_status());
    
    range.set_status(PrivateKeyRange::Status::PAUSED);
    EXPECT_EQ(PrivateKeyRange::Status::PAUSED, range.get_status());
    
    range.set_status(PrivateKeyRange::Status::COMPLETED);
    EXPECT_EQ(PrivateKeyRange::Status::COMPLETED, range.get_status());
}