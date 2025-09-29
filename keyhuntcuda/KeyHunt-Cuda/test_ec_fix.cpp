#include <iostream>
#include <chrono>
#include "SECP256k1.h"

int main() {
    printf("Testing fixed elliptic curve implementation...\n");
    
    // Initialize secp256k1
    Secp256K1 secp;
    secp.Init();
    
    // Test 1: Verify generator point G is correct
    Point G = secp.G;
    printf("Generator point G:\n");
    printf("  x: %s\n", G.x.GetBase16().c_str());
    printf("  y: %s\n", G.y.GetBase16().c_str());
    
    // Test 2: Test ComputePublicKey with small private keys
    printf("\n=== Testing ComputePublicKey ===\n");
    
    Int k1;
    k1.SetInt32(1);
    Point pub1 = secp.ComputePublicKey(&k1);
    printf("Private key 1: %s\n", k1.GetBase16().c_str());
    printf("Public key 1:\n");
    printf("  x: %s\n", pub1.x.GetBase16().c_str());
    printf("  y: %s\n", pub1.y.GetBase16().c_str());
    
    // Should be same as generator G
    if (pub1.x.IsEqual(&G.x) && pub1.y.IsEqual(&G.y)) {
        printf("✓ PASS: 1*G = G (correct)\n");
    } else {
        printf("✗ FAIL: 1*G != G (incorrect)\n");
    }
    
    Int k2;
    k2.SetInt32(2);
    Point pub2 = secp.ComputePublicKey(&k2);
    printf("\nPrivate key 2: %s\n", k2.GetBase16().c_str());
    printf("Public key 2:\n");
    printf("  x: %s\n", pub2.x.GetBase16().c_str());
    printf("  y: %s\n", pub2.y.GetBase16().c_str());
    
    // Test 3: Test that 2*G = G + G
    Point doubled_g = secp.DoubleDirect(G);
    printf("\nDouble(G):\n");
    printf("  x: %s\n", doubled_g.x.GetBase16().c_str());
    printf("  y: %s\n", doubled_g.y.GetBase16().c_str());
    
    if (pub2.x.IsEqual(&doubled_g.x) && pub2.y.IsEqual(&doubled_g.y)) {
        printf("✓ PASS: 2*G = Double(G) (correct)\n");
    } else {
        printf("✗ FAIL: 2*G != Double(G) (incorrect)\n");
    }
    
    // Test 4: Test a larger private key
    Int k_large;
    k_large.SetBase16("123456789ABCDEF");
    Point pub_large = secp.ComputePublicKey(&k_large);
    printf("\nLarge private key: %s\n", k_large.GetBase16().c_str());
    printf("Large public key:\n");
    printf("  x: %s\n", pub_large.x.GetBase16().c_str());
    printf("  y: %s\n", pub_large.y.GetBase16().c_str());
    
    // Test 5: Performance test
    printf("\n=== Performance Test ===\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    int num_tests = 100;
    for (int i = 1; i <= num_tests; i++) {
        Int ki;
        ki.SetInt32(i);
        Point pubi = secp.ComputePublicKey(&ki);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("Computed %d public keys in %ld microseconds\n", num_tests, duration.count());
    printf("Average: %.2f microseconds per key\n", (double)duration.count() / num_tests);
    
    printf("\n=== Test Complete ===\n");
    return 0;
}