#include "../include/uint256.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>
#include <cassert>

void test_basic_mod_operations() {
    std::cout << "=== 基础模运算测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    std::cout << "secp256k1素数p = " << p.to_hex().substr(0, 32) << "..." << std::endl;
    std::cout << std::endl;
    
    // 测试简单的模乘法
    std::cout << "=== 测试模乘法 ===" << std::endl;
    
    UInt256 a(2, 0, 0, 0);
    UInt256 b(3, 0, 0, 0);
    UInt256 expected(6, 0, 0, 0);
    
    UInt256 result = ModOp::mul(a, b);
    
    std::cout << "a = " << a.to_hex() << std::endl;
    std::cout << "b = " << b.to_hex() << std::endl;
    std::cout << "a * b mod p = " << result.to_hex() << std::endl;
    std::cout << "期望 = " << expected.to_hex() << std::endl;
    
    bool mul_correct = (result == expected);
    std::cout << "2 * 3 mod p == 6: " << (mul_correct ? "✅ 通过" : "❌ 失败") << std::endl;
    std::cout << std::endl;
    
    // 测试模幂运算
    std::cout << "=== 测试模幂运算 ===" << std::endl;
    
    UInt256 base(2, 0, 0, 0);
    UInt256 exp(3, 0, 0, 0);
    UInt256 expected_pow(8, 0, 0, 0);
    
    UInt256 pow_result = ModOp::pow_mod_simple(base, exp);
    
    std::cout << "base = " << base.to_hex() << std::endl;
    std::cout << "exp = " << exp.to_hex() << std::endl;
    std::cout << "base^exp mod p = " << pow_result.to_hex() << std::endl;
    std::cout << "期望 = " << expected_pow.to_hex() << std::endl;
    
    bool pow_correct = (pow_result == expected_pow);
    std::cout << "2^3 mod p == 8: " << (pow_correct ? "✅ 通过" : "❌ 失败") << std::endl;
    std::cout << std::endl;
    
    // 测试简单的模逆
    std::cout << "=== 测试简单模逆 ===" << std::endl;
    
    // 对于a=2，我们知道2的模逆应该是 (p+1)/2
    UInt256 a_inv_expected = (p + UInt256(1, 0, 0, 0)) / UInt256(2, 0, 0, 0);
    
    std::cout << "a = 2" << std::endl;
    std::cout << "理论上2的模逆 = (p+1)/2 = " << a_inv_expected.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    
    // 验证：2 * ((p+1)/2) mod p 应该等于 1
    UInt256 check = ModOp::mul(UInt256(2, 0, 0, 0), a_inv_expected);
    std::cout << "验证：2 * ((p+1)/2) mod p = " << check.to_hex() << std::endl;
    
    bool inv_check = (check == UInt256(1, 0, 0, 0));
    std::cout << "验证结果: " << (inv_check ? "✅ 通过" : "❌ 失败") << std::endl;
    
    if (inv_check) {
        std::cout << "✅ 基础模运算工作正常" << std::endl;
    } else {
        std::cout << "❌ 基础模运算有问题" << std::endl;
    }
}

void test_uint256_operations() {
    std::cout << "\n=== UInt256基础运算测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    UInt256 a(2, 0, 0, 0);
    UInt256 b(3, 0, 0, 0);
    
    std::cout << "a = " << a.to_hex() << std::endl;
    std::cout << "b = " << b.to_hex() << std::endl;
    
    UInt256 sum = a + b;
    UInt256 product = a * b;
    
    std::cout << "a + b = " << sum.to_hex() << std::endl;
    std::cout << "a * b = " << product.to_hex() << std::endl;
    
    bool add_correct = (sum == UInt256(5, 0, 0, 0));
    bool mul_correct = (product == UInt256(6, 0, 0, 0));
    
    std::cout << "2 + 3 == 5: " << (add_correct ? "✅ 通过" : "❌ 失败") << std::endl;
    std::cout << "2 * 3 == 6: " << (mul_correct ? "✅ 通过" : "❌ 失败") << std::endl;
    
    // 测试位操作
    std::cout << "\n=== 位操作测试 ===" << std::endl;
    
    UInt256 test_val(5, 0, 0, 0);  // 二进制: 101
    
    std::cout << "test_val = 5 (二进制: 101)" << std::endl;
    std::cout << "bit 0: " << test_val.get_bit(0) << " (期望: 1)" << std::endl;
    std::cout << "bit 1: " << test_val.get_bit(1) << " (期望: 0)" << std::endl;
    std::cout << "bit 2: " << test_val.get_bit(2) << " (期望: 1)" << std::endl;
    
    bool bit0_correct = (test_val.get_bit(0) == 1);
    bool bit1_correct = (test_val.get_bit(1) == 0);
    bool bit2_correct = (test_val.get_bit(2) == 1);
    
    std::cout << "位操作测试: " << (bit0_correct && bit1_correct && bit2_correct ? "✅ 通过" : "❌ 失败") << std::endl;
}

int main() {
    std::cout << "轻量级ECC库 - 基础模运算测试" << std::endl;
    std::cout << "==============================" << std::endl;
    
    try {
        test_uint256_operations();
        test_basic_mod_operations();
        
        std::cout << "\n🎉 基础模运算测试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
