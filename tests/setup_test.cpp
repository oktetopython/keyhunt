// Quick test to verify GoogleTest integration
// This will be replaced by proper tests in subsequent tasks

#include <gtest/gtest.h>

// Simple test to verify GoogleTest is working
TEST(KeyhuntSetup, GoogleTestIntegration) {
    EXPECT_EQ(1 + 1, 2);
    EXPECT_TRUE(true);
}

// Test version information
TEST(KeyhuntSetup, VersionConstants) {
    EXPECT_GT(KEYHUNT_VERSION_MAJOR, -1);
    EXPECT_GE(KEYHUNT_VERSION_MINOR, 0);
    EXPECT_GE(KEYHUNT_VERSION_PATCH, 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}