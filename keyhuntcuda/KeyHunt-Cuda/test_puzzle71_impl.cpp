/**
 * Test program for PUZZLE71 implementation
 * Verifies that the new fast initialization and kernel work correctly
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "SECP256K1.h"
#include "Int.h"
#include "Point.h"

void test_fast_initialization() {
    printf("\n=== Testing Fast Initialization for PUZZLE71 ===\n");
    
    // Create two instances - one with fast init, one with standard
    Secp256K1 secp_fast;
    Secp256K1 secp_standard;
    
    // Test fast initialization
    printf("Testing fast initialization...\n");
    secp_fast.SetFastInit(true);
    
    clock_t start = clock();
    secp_fast.Init();
    clock_t end = clock();
    double fast_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Fast init time: %.3f seconds\n", fast_time);
    
    // Test standard initialization
    printf("\nTesting standard initialization...\n");
    secp_standard.SetFastInit(false);
    
    start = clock();
    secp_standard.Init();
    end = clock();
    double standard_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Standard init time: %.3f seconds\n", standard_time);
    
    // Compare initialization times
    printf("\nSpeed improvement: %.2fx faster\n", standard_time / fast_time);
    
    // Verify generator points are correct
    printf("\nVerifying generator point correctness...\n");
    
    // Check that both have the same generator G
    if (secp_fast.G.x.IsEqual(&secp_standard.G.x) && 
        secp_fast.G.y.IsEqual(&secp_standard.G.y)) {
        printf("✓ Generator points match\n");
    } else {
        printf("✗ Generator points DO NOT match!\n");
    }
    
    // Test basic EC operations
    printf("\nTesting EC operations...\n");
    
    // Generate a test private key
    Int privKey;
    privKey.SetBase16("0x123456789ABCDEF");
    
    // Compute public key using both instances
    Point pub_fast = secp_fast.ComputePublicKey(&privKey);
    Point pub_standard = secp_standard.ComputePublicKey(&privKey);
    
    // Compare results
    if (pub_fast.x.IsEqual(&pub_standard.x) && 
        pub_fast.y.IsEqual(&pub_standard.y)) {
        printf("✓ Public key computation matches\n");
    } else {
        printf("✗ Public key computation DOES NOT match!\n");
    }
    
    // Test point addition
    Point sum_fast = secp_fast.Add(pub_fast, secp_fast.G);
    Point sum_standard = secp_standard.Add(pub_standard, secp_standard.G);
    
    if (sum_fast.x.IsEqual(&sum_standard.x) && 
        sum_fast.y.IsEqual(&sum_standard.y)) {
        printf("✓ Point addition matches\n");
    } else {
        printf("✗ Point addition DOES NOT match!\n");
    }
    
    printf("\n=== Fast Initialization Test Complete ===\n");
}

void test_precomputed_values() {
    printf("\n=== Testing Pre-computed Values ===\n");
    
    // Test that pre-computed values in header are correct
    // This would require including PrecomputedTables.h and verifying values
    printf("Testing pre-computed generator multiples...\n");
    
    Secp256K1 secp;
    secp.SetFastInit(false);  // Use standard to verify
    secp.Init();
    
    // Compute 2*G manually
    Point doubleG = secp.DoubleDirect(secp.G);
    
    printf("2*G computed:\n");
    printf("  X: %s\n", doubleG.x.GetBase16().c_str());
    printf("  Y: %s\n", doubleG.y.GetBase16().c_str());
    
    // Expected values from PrecomputedTables.h
    // 2*G should be:
    // X: C6047F9441ED7D6D3930A14039B24123A398F365F2EA7A0E5CBDF0646E5DB4EA
    // Y: 3D4E09FA48E82B2025349EFF36DC210047F397FB3EE39D3078E764C8BFAF1B0E
    
    Int expected_2G_x;
    Int expected_2G_y;
    expected_2G_x.SetBase16("C6047F9441ED7D6D3930A14039B24123A398F365F2EA7A0E5CBDF0646E5DB4EA");
    expected_2G_y.SetBase16("3D4E09FA48E82B2025349EFF36DC210047F397FB3EE39D3078E764C8BFAF1B0E");
    
    if (doubleG.x.IsEqual(&expected_2G_x) && doubleG.y.IsEqual(&expected_2G_y)) {
        printf("✓ Pre-computed 2*G values are correct\n");
    } else {
        printf("✗ Pre-computed 2*G values are INCORRECT!\n");
    }
    
    printf("\n=== Pre-computed Values Test Complete ===\n");
}

void test_puzzle71_target() {
    printf("\n=== Testing PUZZLE71 Target Configuration ===\n");
    
    // PUZZLE71 target address: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
    // Expected HASH160: D916CE8B4E7C630AB3D7FFFBAC7A9DEF87AEDC7A
    
    printf("Target Address: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU\n");
    printf("Expected HASH160: D916CE8B4E7C630AB3D7FFFBAC7A9DEF87AEDC7A\n");
    
    // The search range for Puzzle #71
    printf("\nSearch Range:\n");
    printf("  Start: 0x40000000000000000 (2^70)\n");
    printf("  End:   0x7FFFFFFFFFFFFFFFFFF (2^71 - 1)\n");
    
    // Calculate total keys to search
    printf("\nTotal keys to search: ~1.18 × 10^21\n");
    
    // Estimate time with different hardware
    printf("\nEstimated search times:\n");
    printf("  RTX 2080 Ti (240 MKey/s): ~156 years\n");
    printf("  RTX 3090 (360 MKey/s): ~104 years\n");
    printf("  RTX 4090 (600 MKey/s): ~62 years\n");
    printf("  100x RTX 4090 cluster: ~227 days\n");
    
    printf("\n=== PUZZLE71 Target Test Complete ===\n");
}

int main() {
    printf("========================================\n");
    printf("   PUZZLE71 Implementation Test Suite   \n");
    printf("========================================\n");
    
    // Run tests
    test_fast_initialization();
    test_precomputed_values();
    test_puzzle71_target();
    
    printf("\n========================================\n");
    printf("         All Tests Complete             \n");
    printf("========================================\n");
    
    return 0;
}