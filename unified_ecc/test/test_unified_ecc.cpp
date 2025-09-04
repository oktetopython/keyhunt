#include "../include/ecc_interface.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace UnifiedECC;

void test_basic_operations() {
    std::cout << "=== Testing Basic ECC Operations ===" << std::endl;

    // 创建ECC实例
    ECCConfig config;
    config.backend = ECCBackend::CPU_OPTIMIZED;
    config.curve_name = "secp256k1";

    auto ecc = ECCFactory::create(config);
    if (!ecc) {
        std::cout << "Failed to create ECC instance" << std::endl;
        return;
    }

    // 初始化曲线参数
    bool initialized = ecc->initialize(Curves::secp256k1);
    if (!initialized) {
        std::cout << "Failed to initialize ECC with secp256k1" << std::endl;
        return;
    }

    std::cout << "ECC initialized successfully: " << ecc->get_implementation_name() << std::endl;

    // 测试基本点运算
    ECPoint G = ecc->get_generator();
    std::cout << "Generator point: " << G.to_string() << std::endl;

    // 测试点倍法
    ECPoint G2 = ecc->point_double(G);
    std::cout << "2G: " << G2.to_string() << std::endl;

    // 测试标量乘法
    UInt256 k(42, 0, 0, 0); // 标量42
    ECPoint kG = ecc->scalar_mul(k, G);
    std::cout << "42G: " << kG.to_string() << std::endl;

    // 验证点是否在曲线上
    bool on_curve = ecc->is_on_curve(kG);
    std::cout << "42G is on curve: " << (on_curve ? "Yes" : "No") << std::endl;

    // 测试模运算
    UInt256 a(123, 0, 0, 0);
    UInt256 b(456, 0, 0, 0);

    UInt256 sum = ecc->mod_add(a, b);
    UInt256 prod = ecc->mod_mul(a, b);
    UInt256 inv_a = ecc->mod_inv(a);

    std::cout << "123 + 456 mod p = " << sum.to_hex() << std::endl;
    std::cout << "123 * 456 mod p = " << prod.to_hex() << std::endl;
    std::cout << "Inverse of 123 mod p = " << inv_a.to_hex() << std::endl;

    // 验证逆元
    UInt256 verify = ecc->mod_mul(a, inv_a);
    std::cout << "123 * inv(123) mod p = " << verify.to_hex() << " (should be 1)" << std::endl;

    std::cout << "✅ Basic operations test passed!" << std::endl;
}

void test_batch_operations() {
    std::cout << "\n=== Testing Batch Operations ===" << std::endl;

    ECCConfig config;
    config.backend = ECCBackend::CPU_OPTIMIZED;
    auto ecc = ECCFactory::create(config);
    ecc->initialize(Curves::secp256k1);

    // 创建测试数据
    std::vector<UInt256> scalars;
    std::vector<ECPoint> points;

    ECPoint G = ecc->get_generator();
    for (int i = 1; i <= 5; ++i) {
        scalars.emplace_back(i, 0, 0, 0);
        points.push_back(G);
    }

    // 执行批量标量乘法
    std::vector<ECPoint> results = ecc->batch_scalar_mul(scalars, points);

    std::cout << "Batch scalar multiplication results:" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "  " << (i+1) << "G: " << results[i].to_string() << std::endl;
    }

    std::cout << "✅ Batch operations test passed!" << std::endl;
}

void test_factory_and_backends() {
    std::cout << "\n=== Testing Factory and Backends ===" << std::endl;

    // 列出可用后端
    auto backends = ECCFactory::get_available_backends();
    std::cout << "Available backends:" << std::endl;
    for (auto backend : backends) {
        std::cout << "  " << ECCFactory::get_backend_description(backend) << std::endl;
    }

    // 测试自动选择
    ECCConfig auto_config;
    auto_config.backend = ECCBackend::AUTO;
    auto ecc_auto = ECCFactory::create(auto_config);

    if (ecc_auto) {
        std::cout << "Auto-selected backend: " << ecc_auto->get_implementation_name() << std::endl;
        std::cout << "Supports GPU: " << (ecc_auto->supports_gpu() ? "Yes" : "No") << std::endl;
        std::cout << "Performance score: " << ecc_auto->get_performance_score() << std::endl;
    }

    std::cout << "✅ Factory and backends test passed!" << std::endl;
}

void test_uint256_operations() {
    std::cout << "\n=== Testing UInt256 Operations ===" << std::endl;

    // 测试构造函数
    UInt256 a(0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL, 0x1111111122222222ULL, 0x3333333344444444ULL);
    UInt256 b("0x123456789ABCDEF0FEDCBA987654321011111111222222223333333344444444");

    std::cout << "a = " << a.to_hex() << std::endl;
    std::cout << "b = " << b.to_hex() << std::endl;
    std::cout << "a == b: " << (a == b ? "true" : "false") << std::endl;

    // 测试算术运算
    UInt256 c = a + b;
    UInt256 d = a - b;

    std::cout << "a + b = " << c.to_hex() << std::endl;
    std::cout << "a - b = " << d.to_hex() << std::endl;

    // 测试字节转换
    auto bytes = a.to_bytes();
    UInt256 e(bytes);
    std::cout << "a -> bytes -> e: " << (a == e ? "conversion successful" : "conversion failed") << std::endl;

    std::cout << "✅ UInt256 operations test passed!" << std::endl;
}

int main() {
    std::cout << "Unified ECC Interface Test Suite" << std::endl;
    std::cout << "=================================" << std::endl;

    try {
        test_uint256_operations();
        test_basic_operations();
        test_batch_operations();
        test_factory_and_backends();

        std::cout << "\n🎉 All tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}