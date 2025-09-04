#include "../include/uint256.h"
#include <iostream>

void debug_division_algorithm() {
    std::cout << "=== 调试UInt256除法算法 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 测试简单除法
    std::cout << "=== 简单除法测试 ===" << std::endl;
    UInt256 a(10, 0, 0, 0);
    UInt256 b(3, 0, 0, 0);
    
    UInt256 quotient = a / b;
    UInt256 remainder = a % b;
    
    std::cout << "10 / 3 = " << quotient.to_hex() << std::endl;
    std::cout << "10 % 3 = " << remainder.to_hex() << std::endl;
    
    bool simple_correct = (quotient == UInt256(3, 0, 0, 0)) && (remainder == UInt256(1, 0, 0, 0));
    std::cout << "简单除法正确: " << (simple_correct ? "是" : "否") << std::endl;
    std::cout << std::endl;
    
    // 测试问题案例：(p+1) mod p
    std::cout << "=== 问题案例：(p+1) mod p ===" << std::endl;
    
    UInt256 p = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    
    std::cout << "p = " << p.to_hex() << std::endl;
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    
    // 手动分析：p+1 应该比 p 大 1
    bool p_plus_1_greater = (p_plus_1 > p);
    std::cout << "p+1 > p: " << (p_plus_1_greater ? "是" : "否") << std::endl;
    
    // 计算 (p+1) / p 和 (p+1) % p
    UInt256 div_result = p_plus_1 / p;
    UInt256 mod_result = p_plus_1 % p;
    
    std::cout << "(p+1) / p = " << div_result.to_hex() << std::endl;
    std::cout << "(p+1) % p = " << mod_result.to_hex() << std::endl;
    
    // 理论上：(p+1) / p = 1, (p+1) % p = 1
    bool div_correct = (div_result == UInt256(1, 0, 0, 0));
    bool mod_correct = (mod_result == UInt256(1, 0, 0, 0));
    
    std::cout << "除法结果正确: " << (div_correct ? "是" : "否") << std::endl;
    std::cout << "模运算结果正确: " << (mod_correct ? "是" : "否") << std::endl;
    
    // 验证：quotient * divisor + remainder == dividend
    UInt256 verification = div_result * p + mod_result;
    bool verify_correct = (verification == p_plus_1);
    std::cout << "验证：quotient * p + remainder == p+1: " << (verify_correct ? "是" : "否") << std::endl;
    
    if (!verify_correct) {
        std::cout << "验证失败！" << std::endl;
        std::cout << "quotient * p + remainder = " << verification.to_hex() << std::endl;
        std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    }
    
    std::cout << std::endl;
    
    // 测试更简单的案例
    std::cout << "=== 更简单的测试案例 ===" << std::endl;
    
    UInt256 test_dividend(0xFFFFFFFFFFFFFFFFULL, 0, 0, 0);  // 64位全1
    UInt256 test_divisor(0xFFFFFFFFFFFFFFFEULL, 0, 0, 0);   // 64位全1减1
    
    std::cout << "dividend = " << test_dividend.to_hex() << std::endl;
    std::cout << "divisor = " << test_divisor.to_hex() << std::endl;
    
    UInt256 test_div = test_dividend / test_divisor;
    UInt256 test_mod = test_dividend % test_divisor;
    
    std::cout << "dividend / divisor = " << test_div.to_hex() << std::endl;
    std::cout << "dividend % divisor = " << test_mod.to_hex() << std::endl;
    
    // 理论上：0xFFFFFFFFFFFFFFFF / 0xFFFFFFFFFFFFFFFE = 1, 余数 = 1
    bool test_div_correct = (test_div == UInt256(1, 0, 0, 0));
    bool test_mod_correct = (test_mod == UInt256(1, 0, 0, 0));
    
    std::cout << "测试除法正确: " << (test_div_correct ? "是" : "否") << std::endl;
    std::cout << "测试模运算正确: " << (test_mod_correct ? "是" : "否") << std::endl;
    
    // 验证
    UInt256 test_verification = test_div * test_divisor + test_mod;
    bool test_verify_correct = (test_verification == test_dividend);
    std::cout << "测试验证正确: " << (test_verify_correct ? "是" : "否") << std::endl;
    
    std::cout << std::endl;
    
    // 总结
    std::cout << "=== 调试总结 ===" << std::endl;
    std::cout << "简单除法 (10/3): " << (simple_correct ? "正确" : "错误") << std::endl;
    std::cout << "问题案例 ((p+1)/p): " << (div_correct && mod_correct ? "正确" : "错误") << std::endl;
    std::cout << "测试案例: " << (test_div_correct && test_mod_correct ? "正确" : "错误") << std::endl;
    
    if (!div_correct || !mod_correct) {
        std::cout << "\n❌ 发现问题：UInt256除法算法在处理大数时有错误！" << std::endl;
        std::cout << "这解释了为什么模运算失败。" << std::endl;
    } else {
        std::cout << "\n✅ 除法算法看起来正确，问题可能在别处。" << std::endl;
    }
}

int main() {
    std::cout << "轻量级ECC库 - UInt256除法算法调试" << std::endl;
    std::cout << "===================================" << std::endl;
    
    try {
        debug_division_algorithm();
        
        std::cout << "\n🔍 除法算法调试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 调试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
