#include <stdio.h>
#include "SECP256k1.h"

// Simple test without complex dependencies
int main() {
    printf("Testing basic elliptic curve functions...\n");
    
    try {
        // Initialize secp256k1
        Secp256K1 secp;
        secp.SetFastInit(true);  // Use fast mode to avoid hanging
        secp.Init();
        
        printf("Secp256k1 initialized successfully\n");
        
        // Test generator point
        Point G = secp.G;
        printf("Generator point G loaded\n");
        printf("G.x: %s\n", G.x.GetBase16().c_str());
        printf("G.y: %s\n", G.y.GetBase16().c_str());
        
        // Test ComputePublicKey with key = 1 (should return G)
        Int k1;
        k1.SetInt32(1);
        
        printf("\nTesting ComputePublicKey with k=1...\n");
        Point pub1 = secp.ComputePublicKey(&k1);
        
        printf("Public key for k=1:\n");
        printf("pub1.x: %s\n", pub1.x.GetBase16().c_str());
        printf("pub1.y: %s\n", pub1.y.GetBase16().c_str());
        
        // Check if pub1 == G
        bool x_equal = pub1.x.IsEqual(&G.x);
        bool y_equal = pub1.y.IsEqual(&G.y);
        
        if (x_equal && y_equal) {
            printf("✓ SUCCESS: 1*G = G (as expected)\n");
        } else {
            printf("✗ FAILURE: 1*G != G\n");
            printf("  x_equal: %s, y_equal: %s\n", 
                   x_equal ? "true" : "false", 
                   y_equal ? "true" : "false");
        }
        
        // Test with k=2
        Int k2;
        k2.SetInt32(2);
        
        printf("\nTesting ComputePublicKey with k=2...\n");
        Point pub2 = secp.ComputePublicKey(&k2);
        
        printf("Public key for k=2:\n");
        printf("pub2.x: %s\n", pub2.x.GetBase16().c_str());
        printf("pub2.y: %s\n", pub2.y.GetBase16().c_str());
        
        // Test DoubleDirect on G 
        printf("\nTesting DoubleDirect(G)...\n");
        Point doubled_G = secp.DoubleDirect(G);
        printf("Double(G):\n");
        printf("doubled_G.x: %s\n", doubled_G.x.GetBase16().c_str());
        printf("doubled_G.y: %s\n", doubled_G.y.GetBase16().c_str());
        
        // Check if pub2 == Double(G)
        bool x_equal_2 = pub2.x.IsEqual(&doubled_G.x);
        bool y_equal_2 = pub2.y.IsEqual(&doubled_G.y);
        
        if (x_equal_2 && y_equal_2) {
            printf("✓ SUCCESS: 2*G = Double(G) (as expected)\n");
        } else {
            printf("✗ FAILURE: 2*G != Double(G)\n");
            printf("  x_equal: %s, y_equal: %s\n", 
                   x_equal_2 ? "true" : "false", 
                   y_equal_2 ? "true" : "false");
        }
        
        printf("\n=== Basic EC Test Complete ===\n");
        
    } catch (const std::exception& e) {
        printf("ERROR: %s\n", e.what());
        return 1;
    } catch (...) {
        printf("ERROR: Unknown exception occurred\n");
        return 1;
    }
    
    return 0;
}