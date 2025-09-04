#include "../include/uint256.h"
#include <iostream>
#include <cassert>
#include <vector>

void debug_mod_operator() {
    std::cout << "=== %运算符专项调试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 测试数据
    UInt256 p = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    
    std::cout << "p = " << p.to_hex() << std::endl;
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    
    // 测试1：直接使用%运算符
    std::cout << "\n=== 测试1：直接使用%运算符 ===" << std::endl;
    UInt256 direct_mod = p_plus_1 % p;
    std::cout << "(p+1) % p = " << direct_mod.to_hex() << std::endl;
    std::cout << "Expected: 0000000000000000000000000000000000000000000000000000000000000001" << std::endl;
    
    // 测试2：使用div_limbs函数
    std::cout << "\n=== 测试2：使用div_limbs函数 ===" << std::endl;
    UInt256 quotient, remainder;
    UInt256::div_limbs(p_plus_1.limbs, p.limbs, quotient.limbs, remainder.limbs);
    std::cout << "div_limbs quotient = " << quotient.to_hex() << std::endl;
    std::cout << "div_limbs remainder = " << remainder.to_hex() << std::endl;
    std::cout << "Expected quotient: 0000000000000000000000000000000000000000000000000000000000000001" << std::endl;
    std::cout << "Expected remainder: 0000000000000000000000000000000000000000000000000000000000000001" << std::endl;
    
    // 测试3：验证%运算符是否正确调用div_limbs
    std::cout << "\n=== 测试3：验证%运算符实现 ===" << std::endl;
    std::cout << "检查%运算符是否正确调用div_limbs..." << std::endl;
    
    // 手动实现%运算符逻辑进行对比
    UInt256 manual_quotient, manual_remainder;
    UInt256::div_limbs(p_plus_1.limbs, p.limbs, manual_quotient.limbs, manual_remainder.limbs);
    
    std::cout << "Manual remainder = " << manual_remainder.to_hex() << std::endl;
    std::cout << "Operator % remainder = " << direct_mod.to_hex() << std::endl;
    std::cout << "Match: " << (manual_remainder == direct_mod ? "YES" : "NO") << std::endl;
    
    // 测试4：检查UInt256::operator%的实现
    std::cout << "\n=== 测试4：检查operator%实现 ===" << std::endl;
    std::cout << "当前UInt256::operator%实现：" << std::endl;
    std::cout << "请检查src/uint256.cpp中的operator%函数是否正确调用div_limbs" << std::endl;
    
    // 测试5：验证除法运算符
    std::cout << "\n=== 测试5：验证除法运算符 ===" << std::endl;
    UInt256 direct_div = p_plus_1 / p;
    std::cout << "(p+1) / p = " << direct_div.to_hex() << std::endl;
    std::cout << "Expected: 0000000000000000000000000000000000000000000000000000000000000001" << std::endl;
    std::cout << "Division correct: " << (direct_div == UInt256(1, 0, 0, 0) ? "YES" : "NO") << std::endl;
    
    // 测试6：验证除法和模运算的一致性
    std::cout << "\n=== 测试6：验证除法和模运算的一致性 ===" << std::endl;
    UInt256 verification = direct_div * p + direct_mod;
    std::cout << "验证：(quotient * divisor + remainder) = " << verification.to_hex() << std::endl;
    std::cout << "原始被除数：" << p_plus_1.to_hex() << std::endl;
    std::cout << "一致性检查: " << (verification == p_plus_1 ? "PASS" : "FAIL") << std::endl;
}

void test_div_limbs_directly() {
    std::cout << "\n=== 直接测试div_limbs函数 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 测试用例1：简单除法
    UInt256 dividend(10, 0, 0, 0);
    UInt256 divisor(3, 0, 0, 0);
    UInt256 quotient, remainder;
    
    UInt256::div_limbs(dividend.limbs, divisor.limbs, quotient.limbs, remainder.limbs);
    
    std::cout << "10 / 3 = " << quotient.to_hex() << " (expected: 3)" << std::endl;
    std::cout << "10 % 3 = " << remainder.to_hex() << " (expected: 1)" << std::endl;
    
    bool simple_correct = (quotient == UInt256(3, 0, 0, 0)) && (remainder == UInt256(1, 0, 0, 0));
    std::cout << "简单除法正确: " << (simple_correct ? "YES" : "NO") << std::endl;
    
    // 测试用例2：大数除法
    UInt256 big_dividend = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC30");
    UInt256 big_divisor = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    
    UInt256::div_limbs(big_dividend.limbs, big_divisor.limbs, quotient.limbs, remainder.limbs);
    
    std::cout << "\n大数除法测试：" << std::endl;
    std::cout << "(p+1) / p = " << quotient.to_hex() << " (expected: 1)" << std::endl;
    std::cout << "(p+1) % p = " << remainder.to_hex() << " (expected: 1)" << std::endl;
    
    bool big_div_correct = (quotient == UInt256(1, 0, 0, 0));
    bool big_mod_correct = (remainder == UInt256(1, 0, 0, 0));
    std::cout << "大数除法正确: " << (big_div_correct ? "YES" : "NO") << std::endl;
    std::cout << "大数模运算正确: " << (big_mod_correct ? "YES" : "NO") << std::endl;
    
    // 验证大数除法
    UInt256 big_verification = quotient * big_divisor + remainder;
    std::cout << "大数验证: " << (big_verification == big_dividend ? "PASS" : "FAIL") << std::endl;
}

void test_operator_consistency() {
    std::cout << "\n=== 运算符一致性测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 测试多个案例的一致性
    std::vector<std::pair<UInt256, UInt256>> test_cases = {
        {UInt256(10, 0, 0, 0), UInt256(3, 0, 0, 0)},
        {UInt256(100, 0, 0, 0), UInt256(7, 0, 0, 0)},
        {UInt256(0xFFFFFFFFFFFFFFFFULL, 0, 0, 0), UInt256(0xFFFFFFFFFFFFFFFEULL, 0, 0, 0)}
    };
    
    for (size_t i = 0; i < test_cases.size(); i++) {
        UInt256 dividend = test_cases[i].first;
        UInt256 divisor = test_cases[i].second;
        
        std::cout << "\n测试案例 " << (i+1) << ":" << std::endl;
        std::cout << "被除数: " << dividend.to_hex().substr(48, 16) << " (低64位)" << std::endl;
        std::cout << "除数: " << divisor.to_hex().substr(48, 16) << " (低64位)" << std::endl;
        
        // 使用运算符
        UInt256 op_quotient = dividend / divisor;
        UInt256 op_remainder = dividend % divisor;
        
        // 使用div_limbs
        UInt256 dl_quotient, dl_remainder;
        UInt256::div_limbs(dividend.limbs, divisor.limbs, dl_quotient.limbs, dl_remainder.limbs);
        
        std::cout << "运算符商: " << op_quotient.to_hex().substr(48, 16) << " (低64位)" << std::endl;
        std::cout << "div_limbs商: " << dl_quotient.to_hex().substr(48, 16) << " (低64位)" << std::endl;
        std::cout << "运算符余数: " << op_remainder.to_hex().substr(48, 16) << " (低64位)" << std::endl;
        std::cout << "div_limbs余数: " << dl_remainder.to_hex().substr(48, 16) << " (低64位)" << std::endl;
        
        bool quotient_match = (op_quotient == dl_quotient);
        bool remainder_match = (op_remainder == dl_remainder);
        
        std::cout << "商一致: " << (quotient_match ? "YES" : "NO") << std::endl;
        std::cout << "余数一致: " << (remainder_match ? "YES" : "NO") << std::endl;
        
        // 验证
        UInt256 op_verification = op_quotient * divisor + op_remainder;
        UInt256 dl_verification = dl_quotient * divisor + dl_remainder;
        
        std::cout << "运算符验证: " << (op_verification == dividend ? "PASS" : "FAIL") << std::endl;
        std::cout << "div_limbs验证: " << (dl_verification == dividend ? "PASS" : "FAIL") << std::endl;
    }
}

int main() {
    std::cout << "UInt256 %运算符调试工具" << std::endl;
    std::cout << "======================" << std::endl;
    
    try {
        debug_mod_operator();
        test_div_limbs_directly();
        test_operator_consistency();
        
        std::cout << "\n=== 调试总结 ===" << std::endl;
        std::cout << "请检查以上输出，找出%运算符的问题所在" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "调试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
