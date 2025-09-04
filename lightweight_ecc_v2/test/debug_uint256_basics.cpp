#include "../include/uint256.h"
#include <iostream>

void debug_uint256_basics() {
    std::cout << "=== 调试UInt256基础操作 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 测试UInt256的构造和比较
    UInt256 p = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    
    std::cout << "p = " << p.to_hex() << std::endl;
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    
    // 检查p+1是否真的比p大1
    UInt256 diff = p_plus_1 - p;
    std::cout << "(p+1) - p = " << diff.to_hex() << std::endl;
    std::cout << "差值是否为1: " << (diff == UInt256(1, 0, 0, 0) ? "是" : "否") << std::endl;
    
    // 检查比较操作
    std::cout << "p+1 > p: " << (p_plus_1 > p ? "是" : "否") << std::endl;
    std::cout << "p+1 >= p: " << (p_plus_1 >= p ? "是" : "否") << std::endl;
    std::cout << "p+1 == p: " << (p_plus_1 == p ? "是" : "否") << std::endl;
    
    // 检查p的内部表示
    std::cout << "\np的内部表示 (limbs):" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "limbs[" << i << "] = 0x" << std::hex << p.limbs[i] << std::dec << std::endl;
    }
    
    std::cout << "\np+1的内部表示 (limbs):" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "limbs[" << i << "] = 0x" << std::hex << p_plus_1.limbs[i] << std::dec << std::endl;
    }
    
    // 测试除法的边界条件
    std::cout << "\n=== 测试除法边界条件 ===" << std::endl;
    
    // 测试 p / p
    UInt256 p_div_p = p / p;
    UInt256 p_mod_p = p % p;
    
    std::cout << "p / p = " << p_div_p.to_hex() << std::endl;
    std::cout << "p % p = " << p_mod_p.to_hex() << std::endl;
    std::cout << "p / p == 1: " << (p_div_p == UInt256(1, 0, 0, 0) ? "是" : "否") << std::endl;
    std::cout << "p % p == 0: " << (p_mod_p.is_zero() ? "是" : "否") << std::endl;
    
    // 测试 (p-1) / p
    UInt256 p_minus_1 = p - UInt256(1, 0, 0, 0);
    UInt256 p_minus_1_div_p = p_minus_1 / p;
    UInt256 p_minus_1_mod_p = p_minus_1 % p;
    
    std::cout << "\n(p-1) / p = " << p_minus_1_div_p.to_hex() << std::endl;
    std::cout << "(p-1) % p = " << p_minus_1_mod_p.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "(p-1) / p == 0: " << (p_minus_1_div_p.is_zero() ? "是" : "否") << std::endl;
    std::cout << "(p-1) % p == p-1: " << (p_minus_1_mod_p == p_minus_1 ? "是" : "否") << std::endl;
    
    // 分析问题：为什么(p+1) / p = 0？
    std::cout << "\n=== 分析(p+1) / p问题 ===" << std::endl;
    
    // 手动检查：p+1是否真的大于p？
    bool manual_greater = false;
    for (int i = 3; i >= 0; i--) {
        if (p_plus_1.limbs[i] > p.limbs[i]) {
            manual_greater = true;
            break;
        } else if (p_plus_1.limbs[i] < p.limbs[i]) {
            manual_greater = false;
            break;
        }
    }
    
    std::cout << "手动比较 p+1 > p: " << (manual_greater ? "是" : "否") << std::endl;
    
    // 检查UInt256的>=操作符
    bool ge_result = (p_plus_1 >= p);
    std::cout << ">=操作符结果: " << (ge_result ? "是" : "否") << std::endl;
    
    if (manual_greater && !ge_result) {
        std::cout << "❌ 发现问题：UInt256的比较操作符有bug！" << std::endl;
    } else if (!manual_greater) {
        std::cout << "❌ 发现问题：UInt256的加法操作有bug！" << std::endl;
    } else {
        std::cout << "✅ 比较和加法操作看起来正确。" << std::endl;
    }
    
    // 测试更简单的大数除法
    std::cout << "\n=== 测试简单大数除法 ===" << std::endl;
    
    UInt256 big_num(0, 0, 0, 1);  // 2^192
    UInt256 big_div = big_num / UInt256(2, 0, 0, 0);
    UInt256 big_mod = big_num % UInt256(2, 0, 0, 0);
    
    std::cout << "2^192 / 2 = " << big_div.to_hex() << std::endl;
    std::cout << "2^192 % 2 = " << big_mod.to_hex() << std::endl;
    
    // 验证：big_div * 2 + big_mod 应该等于 big_num
    UInt256 big_verify = big_div * UInt256(2, 0, 0, 0) + big_mod;
    std::cout << "验证正确: " << (big_verify == big_num ? "是" : "否") << std::endl;
}

int main() {
    std::cout << "轻量级ECC库 - UInt256基础操作调试" << std::endl;
    std::cout << "==================================" << std::endl;
    
    try {
        debug_uint256_basics();
        
        std::cout << "\n🔍 UInt256基础操作调试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 调试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
