#include "../include/uint256.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>

void test_simple_mod_mul() {
    std::cout << "=== 简单模乘法测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    std::cout << "secp256k1素数p = " << p.to_hex().substr(0, 32) << "..." << std::endl;
    std::cout << std::endl;
    
    // 测试简单的模乘法
    UInt256 a(2, 0, 0, 0);
    UInt256 b(3, 0, 0, 0);
    UInt256 expected(6, 0, 0, 0);
    
    UInt256 result = ModOp::mul(a, b);
    
    std::cout << "a = " << a.to_hex() << std::endl;
    std::cout << "b = " << b.to_hex() << std::endl;
    std::cout << "a * b mod p = " << result.to_hex() << std::endl;
    std::cout << "期望 = " << expected.to_hex() << std::endl;
    
    bool test1 = (result == expected);
    std::cout << "2 * 3 mod p == 6: " << (test1 ? "✅ 通过" : "❌ 失败") << std::endl;
    std::cout << std::endl;
    
    // 测试与1的乘法
    UInt256 one(1, 0, 0, 0);
    UInt256 result_one = ModOp::mul(a, one);
    
    std::cout << "a * 1 mod p = " << result_one.to_hex() << std::endl;
    bool test2 = (result_one == a);
    std::cout << "2 * 1 mod p == 2: " << (test2 ? "✅ 通过" : "❌ 失败") << std::endl;
    std::cout << std::endl;
    
    // 测试与0的乘法
    UInt256 zero(0, 0, 0, 0);
    UInt256 result_zero = ModOp::mul(a, zero);
    
    std::cout << "a * 0 mod p = " << result_zero.to_hex() << std::endl;
    bool test3 = result_zero.is_zero();
    std::cout << "2 * 0 mod p == 0: " << (test3 ? "✅ 通过" : "❌ 失败") << std::endl;
    std::cout << std::endl;
    
    // 测试大数乘法
    UInt256 large1(1000000, 0, 0, 0);
    UInt256 large2(2000000, 0, 0, 0);
    UInt256 expected_large(2000000000000ULL, 0, 0, 0);
    
    UInt256 result_large = ModOp::mul(large1, large2);
    
    std::cout << "large1 = " << large1.to_hex() << std::endl;
    std::cout << "large2 = " << large2.to_hex() << std::endl;
    std::cout << "large1 * large2 mod p = " << result_large.to_hex() << std::endl;
    std::cout << "期望 = " << expected_large.to_hex() << std::endl;
    
    bool test4 = (result_large == expected_large);
    std::cout << "1000000 * 2000000 mod p == 2000000000000: " << (test4 ? "✅ 通过" : "❌ 失败") << std::endl;
    
    if (test1 && test2 && test3 && test4) {
        std::cout << "\n✅ 所有模乘法测试通过！" << std::endl;
    } else {
        std::cout << "\n❌ 模乘法测试有失败项" << std::endl;
    }
}

void test_known_mod_inverse() {
    std::cout << "\n=== 已知模逆测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    // 对于素数p，我们知道2的模逆应该是 (p+1)/2
    UInt256 two(2, 0, 0, 0);
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    UInt256 expected_inv = p_plus_1 / UInt256(2, 0, 0, 0);
    
    std::cout << "p = " << p.to_hex().substr(0, 32) << "..." << std::endl;
    std::cout << "2的理论模逆 = (p+1)/2 = " << expected_inv.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    
    // 验证：2 * ((p+1)/2) mod p 应该等于 1
    UInt256 verification = ModOp::mul(two, expected_inv);
    std::cout << "验证：2 * ((p+1)/2) mod p = " << verification.to_hex() << std::endl;
    
    bool correct = (verification == UInt256(1, 0, 0, 0));
    std::cout << "验证结果: " << (correct ? "✅ 通过" : "❌ 失败") << std::endl;
    
    if (!correct) {
        std::cout << "期望: 0000000000000000000000000000000000000000000000000000000000000001" << std::endl;
        std::cout << "实际: " << verification.to_hex() << std::endl;
        
        // 详细分析
        std::cout << "\n详细分析:" << std::endl;
        std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
        std::cout << "(p+1)/2 = " << expected_inv.to_hex() << std::endl;
        
        // 验证除法是否正确
        UInt256 double_inv = expected_inv * UInt256(2, 0, 0, 0);
        std::cout << "验证除法：2 * ((p+1)/2) = " << double_inv.to_hex() << std::endl;
        std::cout << "应该等于p+1: " << (double_inv == p_plus_1 ? "是" : "否") << std::endl;
    }
}

int main() {
    std::cout << "轻量级ECC库 - 模乘法测试" << std::endl;
    std::cout << "========================" << std::endl;
    
    try {
        test_simple_mod_mul();
        test_known_mod_inverse();
        
        std::cout << "\n🎉 模乘法测试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
