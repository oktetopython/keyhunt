#include "../include/math_common.h"
#include <iostream>
#include <cassert>

using namespace UnifiedECC::Math;

void test_bit_utils() {
    std::cout << "=== Testing BitUtils ===" << std::endl;

    uint64_t limbs[2] = {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL};

    // 测试位获取
    assert(BitUtils::get_bit(limbs, 0) == false); // 最低位是0
    assert(BitUtils::get_bit(limbs, 4) == true);  // 第4位是1

    // 测试位设置
    BitUtils::set_bit(limbs, 0, true);
    assert(BitUtils::get_bit(limbs, 0) == true);

    BitUtils::set_bit(limbs, 0, false);
    assert(BitUtils::get_bit(limbs, 0) == false);

    // 测试位长度
    int bit_len = BitUtils::bit_length(limbs, 2);
    std::cout << "Bit length: " << bit_len << std::endl;

    // 测试汉明重量
    int weight = BitUtils::hamming_weight(limbs, 2);
    std::cout << "Hamming weight: " << weight << std::endl;

    std::cout << "✅ BitUtils test passed!" << std::endl;
}

void test_bigint_utils() {
    std::cout << "\n=== Testing BigIntUtils ===" << std::endl;

    uint64_t a[2] = {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL};
    uint64_t b[2] = {0x1111111122222222ULL, 0x3333333344444444ULL};
    uint64_t result[2];

    // 测试加法
    BigIntUtils::add(result, a, b, 2);
    std::cout << "Addition result: " << std::hex << result[1] << result[0] << std::dec << std::endl;

    // 测试比较
    int cmp = BigIntUtils::compare(a, b, 2);
    std::cout << "Comparison result: " << cmp << std::endl;

    std::cout << "✅ BigIntUtils test passed!" << std::endl;
}

void test_number_theory() {
    std::cout << "\n=== Testing NumberTheory ===" << std::endl;

    // 测试GCD
    uint64_t gcd_result = NumberTheory::gcd(48, 18);
    assert(gcd_result == 6);
    std::cout << "GCD(48, 18) = " << gcd_result << std::endl;

    // 测试模逆
    uint64_t inv_result = NumberTheory::mod_inverse(7, 13);
    assert(inv_result == 2); // 7 * 2 = 14 ≡ 1 mod 13
    std::cout << "Modular inverse of 7 mod 13 = " << inv_result << std::endl;

    // 测试模幂
    uint64_t pow_result = NumberTheory::mod_pow(2, 10, 1000);
    assert(pow_result == 24); // 2^10 = 1024 ≡ 24 mod 1000
    std::cout << "2^10 mod 1000 = " << pow_result << std::endl;

    std::cout << "✅ NumberTheory test passed!" << std::endl;
}

void test_performance_monitor() {
    std::cout << "\n=== Testing PerformanceMonitor ===" << std::endl;

    auto timing = PerformanceMonitor::time_operation_simple(1000);
    std::cout << "Timing result: " << timing.elapsed_seconds << "s, "
              << timing.operations_per_second << " ops/s" << std::endl;

    std::string formatted = PerformanceMonitor::format_performance(1500000.0);
    std::cout << "Formatted performance: " << formatted << std::endl;

    std::cout << "✅ PerformanceMonitor test passed!" << std::endl;
}

void test_memory_utils() {
    std::cout << "\n=== Testing MemoryUtils ===" << std::endl;

    // 测试安全内存分配
    int* ptr = MemoryUtils::safe_malloc<int>(10);
    for (int i = 0; i < 10; ++i) {
        ptr[i] = i * 2;
    }

    // 验证数据
    for (int i = 0; i < 10; ++i) {
        assert(ptr[i] == i * 2);
    }

    // 测试安全释放
    MemoryUtils::safe_free(ptr);
    assert(ptr == nullptr);

    std::cout << "✅ MemoryUtils test passed!" << std::endl;
}

int main() {
    std::cout << "Unified ECC Math Common Library Test Suite" << std::endl;
    std::cout << "==========================================" << std::endl;

    try {
        test_bit_utils();
        test_bigint_utils();
        test_number_theory();
        test_performance_monitor();
        test_memory_utils();

        std::cout << "\n🎉 All math common tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}