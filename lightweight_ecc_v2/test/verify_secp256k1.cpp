#include "../include/uint256.h"
#include "../include/secp256k1.h"
#include <iostream>

void verify_secp256k1_constants() {
    std::cout << "=== 验证secp256k1常数 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    auto params = Secp256k1::get_params();
    
    std::cout << "secp256k1素数p = " << params.p.to_hex() << std::endl;
    
    // 标准的secp256k1素数应该是：
    // p = 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1
    // p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    
    UInt256 expected_p = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    std::cout << "标准secp256k1素数 = " << expected_p.to_hex() << std::endl;
    
    bool p_correct = (params.p == expected_p);
    std::cout << "素数p正确: " << (p_correct ? "✅ 是" : "❌ 否") << std::endl;
    
    if (!p_correct) {
        std::cout << "❌ 素数p不正确！这可能是问题的根源。" << std::endl;
        return;
    }
    
    // 验证基本模运算
    std::cout << "\n=== 验证基本模运算 ===" << std::endl;
    
    // 测试 p mod p = 0
    UInt256 p_mod_p = params.p % params.p;
    std::cout << "p mod p = " << p_mod_p.to_hex() << std::endl;
    bool test1 = p_mod_p.is_zero();
    std::cout << "p mod p == 0: " << (test1 ? "✅ 通过" : "❌ 失败") << std::endl;
    
    // 测试 (p+1) mod p = 1
    UInt256 p_plus_1 = params.p + UInt256(1, 0, 0, 0);
    UInt256 p_plus_1_mod_p = p_plus_1 % params.p;
    std::cout << "(p+1) mod p = " << p_plus_1_mod_p.to_hex() << std::endl;
    bool test2 = (p_plus_1_mod_p == UInt256(1, 0, 0, 0));
    std::cout << "(p+1) mod p == 1: " << (test2 ? "✅ 通过" : "❌ 失败") << std::endl;
    
    // 测试 (p-1) mod p = p-1
    UInt256 p_minus_1 = params.p - UInt256(1, 0, 0, 0);
    UInt256 p_minus_1_mod_p = p_minus_1 % params.p;
    std::cout << "(p-1) mod p = " << p_minus_1_mod_p.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    bool test3 = (p_minus_1_mod_p == p_minus_1);
    std::cout << "(p-1) mod p == p-1: " << (test3 ? "✅ 通过" : "❌ 失败") << std::endl;
    
    if (test1 && test2 && test3) {
        std::cout << "\n✅ 所有基本模运算测试通过！" << std::endl;
    } else {
        std::cout << "\n❌ 基本模运算测试失败，需要进一步调试。" << std::endl;
        
        // 详细调试
        std::cout << "\n=== 详细调试信息 ===" << std::endl;
        
        if (!test1) {
            std::cout << "p mod p 失败分析：" << std::endl;
            std::cout << "  p = " << params.p.to_hex() << std::endl;
            std::cout << "  p mod p = " << p_mod_p.to_hex() << std::endl;
            std::cout << "  期望 = 0000000000000000000000000000000000000000000000000000000000000000" << std::endl;
        }
        
        if (!test2) {
            std::cout << "(p+1) mod p 失败分析：" << std::endl;
            std::cout << "  p+1 = " << p_plus_1.to_hex() << std::endl;
            std::cout << "  (p+1) mod p = " << p_plus_1_mod_p.to_hex() << std::endl;
            std::cout << "  期望 = 0000000000000000000000000000000000000000000000000000000000000001" << std::endl;
        }
    }
}

void test_simple_division() {
    std::cout << "\n=== 简单除法测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 测试简单的除法运算
    UInt256 a(10, 0, 0, 0);
    UInt256 b(3, 0, 0, 0);
    
    UInt256 quotient = a / b;
    UInt256 remainder = a % b;
    
    std::cout << "10 / 3 = " << quotient.to_hex() << std::endl;
    std::cout << "10 % 3 = " << remainder.to_hex() << std::endl;
    
    bool div_correct = (quotient == UInt256(3, 0, 0, 0));
    bool mod_correct = (remainder == UInt256(1, 0, 0, 0));
    
    std::cout << "10 / 3 == 3: " << (div_correct ? "✅ 通过" : "❌ 失败") << std::endl;
    std::cout << "10 % 3 == 1: " << (mod_correct ? "✅ 通过" : "❌ 失败") << std::endl;
    
    // 验证：quotient * divisor + remainder == dividend
    UInt256 verification = quotient * b + remainder;
    bool verify_correct = (verification == a);
    std::cout << "验证：3 * 3 + 1 == 10: " << (verify_correct ? "✅ 通过" : "❌ 失败") << std::endl;
    
    if (div_correct && mod_correct && verify_correct) {
        std::cout << "✅ 简单除法测试通过！" << std::endl;
    } else {
        std::cout << "❌ 简单除法测试失败！" << std::endl;
    }
}

int main() {
    std::cout << "轻量级ECC库 - secp256k1常数验证" << std::endl;
    std::cout << "================================" << std::endl;
    
    try {
        test_simple_division();
        verify_secp256k1_constants();
        
        std::cout << "\n🎉 secp256k1常数验证完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 验证失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
