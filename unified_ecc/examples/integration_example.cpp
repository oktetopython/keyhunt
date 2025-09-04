/**
 * Unified ECC Integration Example
 *
 * This example demonstrates how to use the unified ECC interface
 * to seamlessly switch between different ECC implementations.
 */

#include "../include/ecc_interface.h"
#include "../include/config_manager.h"
#include <iostream>
#include <chrono>

using namespace UnifiedECC;

void demonstrate_basic_usage() {
    std::cout << "=== Basic ECC Usage Example ===" << std::endl;

    // Configure for optimal performance
    configure_for_performance();

    // Create ECC instance using configuration
    auto ecc = ECCFactory::create(ECCConfig{});

    if (!ecc) {
        std::cout << "Failed to create ECC instance" << std::endl;
        return;
    }

    // Initialize with secp256k1 curve
    bool initialized = ecc->initialize(Curves::secp256k1);
    if (!initialized) {
        std::cout << "Failed to initialize ECC" << std::endl;
        return;
    }

    std::cout << "ECC initialized: " << ecc->get_implementation_name() << std::endl;
    std::cout << "Supports GPU: " << (ecc->supports_gpu() ? "Yes" : "No") << std::endl;

    // Get generator point
    ECPoint G = ecc->get_generator();
    std::cout << "Generator point: " << G.to_string() << std::endl;

    // Perform scalar multiplication
    UInt256 scalar(42, 0, 0, 0); // scalar = 42
    ECPoint result = ecc->scalar_mul(scalar, G);
    std::cout << "42 * G = " << result.to_string() << std::endl;

    // Verify point is on curve
    bool on_curve = ecc->is_on_curve(result);
    std::cout << "Point is on curve: " << (on_curve ? "Yes" : "No") << std::endl;
}

void demonstrate_batch_operations() {
    std::cout << "\n=== Batch Operations Example ===" << std::endl;

    // Configure for batch processing
    ConfigBuilder()
        .enable_batch_ops(true)
        .batch_size(1000)
        .build();

    auto ecc = ECCFactory::create(ECCConfig{});
    ecc->initialize(Curves::secp256k1);

    // Create batch of scalars and points
    std::vector<UInt256> scalars;
    std::vector<ECPoint> points;

    ECPoint G = ecc->get_generator();
    for (int i = 1; i <= 10; ++i) {
        scalars.emplace_back(i, 0, 0, 0);
        points.push_back(G);
    }

    // Perform batch scalar multiplication
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<ECPoint> results = ecc->batch_scalar_mul(scalars, points);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Batch scalar multiplication completed in " << duration.count() << " microseconds" << std::endl;
    std::cout << "Results:" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "  " << (i + 1) << " * G = " << results[i].to_string() << std::endl;
    }
}

void demonstrate_modular_arithmetic() {
    std::cout << "\n=== Modular Arithmetic Example ===" << std::endl;

    auto ecc = ECCFactory::create(ECCConfig{});
    ecc->initialize(Curves::secp256k1);

    UInt256 a(12345, 0, 0, 0);
    UInt256 b(67890, 0, 0, 0);

    std::cout << "a = " << a.to_hex() << std::endl;
    std::cout << "b = " << b.to_hex() << std::endl;

    // Modular addition
    UInt256 sum = ecc->mod_add(a, b);
    std::cout << "a + b mod p = " << sum.to_hex() << std::endl;

    // Modular multiplication
    UInt256 product = ecc->mod_mul(a, b);
    std::cout << "a * b mod p = " << product.to_hex() << std::endl;

    // Modular inverse
    UInt256 inv_a = ecc->mod_inv(a);
    std::cout << "a^(-1) mod p = " << inv_a.to_hex() << std::endl;

    // Verify inverse: a * a^(-1) should equal 1 mod p
    UInt256 verify = ecc->mod_mul(a, inv_a);
    std::cout << "Verification (should be 1): " << verify.to_hex() << std::endl;
}

void demonstrate_configuration_management() {
    std::cout << "\n=== Configuration Management Example ===" << std::endl;

    // Show different configuration presets
    std::cout << "Performance configuration:" << std::endl;
    configure_for_performance();
    ConfigManager::instance().print_config();

    std::cout << "\nSecurity configuration:" << std::endl;
    configure_for_security();
    ConfigManager::instance().print_config();

    std::cout << "\nDevelopment configuration:" << std::endl;
    configure_for_development();
    ConfigManager::instance().print_config();
}

void demonstrate_dynamic_backend_switching() {
    std::cout << "\n=== Dynamic Backend Switching Example ===" << std::endl;

    // Test different backends
    std::vector<std::string> backends = {"cpu_optimized", "cpu_generic"};

    for (const auto& backend_name : backends) {
        std::cout << "Testing backend: " << backend_name << std::endl;

        ECCConfig config;
        config.backend = ECCBackend::CPU_OPTIMIZED; // For now, only CPU_OPTIMIZED is implemented

        auto ecc = ECCFactory::create(config);
        if (ecc) {
            ecc->initialize(Curves::secp256k1);
            std::cout << "  Implementation: " << ecc->get_implementation_name() << std::endl;
            std::cout << "  Performance score: " << ecc->get_performance_score() << std::endl;
        } else {
            std::cout << "  Backend not available" << std::endl;
        }
        std::cout << std::endl;
    }
}

void demonstrate_error_handling() {
    std::cout << "\n=== Error Handling Example ===" << std::endl;

    try {
        // Test with invalid curve
        ECCConfig config;
        auto ecc = ECCFactory::create(config);

        ECCParams invalid_params;
        invalid_params.name = "invalid_curve";

        bool result = ecc->initialize(invalid_params);
        if (!result) {
            std::cout << "Expected failure with invalid curve parameters" << std::endl;
        }

        // Test point operations with infinity
        ECPoint inf_point(true); // infinity point
        ECPoint G = ecc->get_generator();

        ECPoint result_point = ecc->point_add(inf_point, G);
        std::cout << "Infinity + G = " << result_point.to_string() << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Unified ECC Interface Integration Example" << std::endl;
    std::cout << "=========================================" << std::endl;

    try {
        demonstrate_basic_usage();
        demonstrate_batch_operations();
        demonstrate_modular_arithmetic();
        demonstrate_configuration_management();
        demonstrate_dynamic_backend_switching();
        demonstrate_error_handling();

        std::cout << "\n🎉 All integration examples completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ Integration example failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Integration example failed with unknown error" << std::endl;
        return 1;
    }
}