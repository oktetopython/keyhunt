#include "../include/uint256.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>

void debug_mod_inverse_calculation() {
    std::cout << "=== 调试模逆计算 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    std::cout << "secp256k1素数p = " << p.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 验证p是奇数
    std::cout << "p是奇数: " << (p.is_odd() ? "是" : "否") << std::endl;
    
    // 计算p+1
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    
    // 验证p+1是偶数
    std::cout << "p+1是偶数: " << (p_plus_1.is_even() ? "是" : "否") << std::endl;
    
    // 计算(p+1)/2
    UInt256 half_p_plus_1 = p_plus_1 / UInt256(2, 0, 0, 0);
    std::cout << "(p+1)/2 = " << half_p_plus_1.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 验证：2 * ((p+1)/2) 应该等于 p+1
    UInt256 two_times_half = UInt256(2, 0, 0, 0) * half_p_plus_1;
    std::cout << "验证：2 * ((p+1)/2) = " << two_times_half.to_hex() << std::endl;
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    std::cout << "2 * ((p+1)/2) == p+1: " << (two_times_half == p_plus_1 ? "是" : "否") << std::endl;
    std::cout << std::endl;
    
    // 现在验证模运算：2 * ((p+1)/2) mod p 应该等于 1
    UInt256 mod_result = ModOp::mul(UInt256(2, 0, 0, 0), half_p_plus_1);
    std::cout << "2 * ((p+1)/2) mod p = " << mod_result.to_hex() << std::endl;
    
    // 手动计算：(p+1) mod p = 1
    UInt256 manual_mod = p_plus_1 % p;
    std::cout << "(p+1) mod p = " << manual_mod.to_hex() << std::endl;
    
    bool correct = (mod_result == UInt256(1, 0, 0, 0));
    std::cout << "结果正确: " << (correct ? "是" : "否") << std::endl;
    
    if (!correct) {
        std::cout << "\n=== 详细分析 ===" << std::endl;
        
        // 分析差异
        if (mod_result == manual_mod) {
            std::cout << "模运算结果与手动计算一致，说明模运算正确" << std::endl;
            std::cout << "问题可能在于理论分析：(p+1)/2 可能不是2的模逆" << std::endl;
        } else {
            std::cout << "模运算结果与手动计算不一致，说明模运算有问题" << std::endl;
        }
        
        // 尝试暴力搜索2的真正模逆
        std::cout << "\n=== 暴力搜索2的模逆 ===" << std::endl;
        
        bool found = false;
        for (uint64_t i = 1; i <= 1000000 && !found; i++) {
            UInt256 candidate(i, 0, 0, 0);
            UInt256 product = ModOp::mul(UInt256(2, 0, 0, 0), candidate);
            if (product == UInt256(1, 0, 0, 0)) {
                std::cout << "找到2的模逆: " << candidate.to_hex() << std::endl;
                std::cout << "验证：2 * " << i << " mod p = " << product.to_hex() << std::endl;
                found = true;
            }
        }
        
        if (!found) {
            std::cout << "在前100万个数中未找到2的模逆" << std::endl;
        }
    }
}

void test_simple_mod_operations() {
    std::cout << "\n=== 简单模运算测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    // 测试 (p-1) mod p = p-1
    UInt256 p_minus_1 = p - UInt256(1, 0, 0, 0);
    UInt256 result1 = p_minus_1 % p;
    std::cout << "(p-1) mod p = " << result1.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "期望 = " << p_minus_1.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "(p-1) mod p == p-1: " << (result1 == p_minus_1 ? "✅ 通过" : "❌ 失败") << std::endl;
    
    // 测试 p mod p = 0
    UInt256 result2 = p % p;
    std::cout << "p mod p = " << result2.to_hex() << std::endl;
    std::cout << "p mod p == 0: " << (result2.is_zero() ? "✅ 通过" : "❌ 失败") << std::endl;
    
    // 测试 (p+1) mod p = 1
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    UInt256 result3 = p_plus_1 % p;
    std::cout << "(p+1) mod p = " << result3.to_hex() << std::endl;
    std::cout << "(p+1) mod p == 1: " << (result3 == UInt256(1, 0, 0, 0) ? "✅ 通过" : "❌ 失败") << std::endl;
}

int main() {
    std::cout << "轻量级ECC库 - 模逆计算调试" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        test_simple_mod_operations();
        debug_mod_inverse_calculation();
        
        std::cout << "\n🎉 模逆计算调试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 调试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
