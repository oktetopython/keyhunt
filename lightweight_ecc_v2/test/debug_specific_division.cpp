#include "../include/uint256.h"
#include <iostream>

void debug_specific_division_case() {
    std::cout << "=== 调试特定除法案例 ===" << std::endl;
    
    using namespace LightweightECC;
    
    UInt256 p = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    
    std::cout << "p = " << p.to_hex() << std::endl;
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 手动模拟长除法过程
    std::cout << "=== 手动模拟长除法过程 ===" << std::endl;
    
    // 检查最高位
    std::cout << "p+1的最高位非零limb: ";
    for (int i = 3; i >= 0; i--) {
        if (p_plus_1.limbs[i] != 0) {
            std::cout << "limbs[" << i << "] = 0x" << std::hex << p_plus_1.limbs[i] << std::dec << std::endl;
            break;
        }
    }
    
    std::cout << "p的最高位非零limb: ";
    for (int i = 3; i >= 0; i--) {
        if (p.limbs[i] != 0) {
            std::cout << "limbs[" << i << "] = 0x" << std::hex << p.limbs[i] << std::dec << std::endl;
            break;
        }
    }
    
    // 检查位长度
    int p_plus_1_bits = 0;
    int p_bits = 0;
    
    for (int i = 255; i >= 0; i--) {
        if (p_plus_1.get_bit(i) && p_plus_1_bits == 0) {
            p_plus_1_bits = i + 1;
        }
        if (p.get_bit(i) && p_bits == 0) {
            p_bits = i + 1;
        }
        if (p_plus_1_bits > 0 && p_bits > 0) break;
    }
    
    std::cout << "p+1的位长度: " << p_plus_1_bits << std::endl;
    std::cout << "p的位长度: " << p_bits << std::endl;
    
    // 检查最高几位
    std::cout << "\n最高8位比较:" << std::endl;
    for (int i = 255; i >= 248; i--) {
        std::cout << "bit[" << i << "]: p+1=" << p_plus_1.get_bit(i) << ", p=" << p.get_bit(i) << std::endl;
    }
    
    // 手动计算应该的结果
    std::cout << "\n=== 手动计算验证 ===" << std::endl;
    
    // p+1 应该等于 1*p + 1
    UInt256 manual_quotient(1, 0, 0, 0);
    UInt256 manual_remainder(1, 0, 0, 0);
    UInt256 manual_result = manual_quotient * p + manual_remainder;
    
    std::cout << "手动计算: 1*p + 1 = " << manual_result.to_hex() << std::endl;
    std::cout << "实际p+1 = " << p_plus_1.to_hex() << std::endl;
    std::cout << "手动计算正确: " << (manual_result == p_plus_1 ? "是" : "否") << std::endl;
    
    // 测试我们的除法算法
    std::cout << "\n=== 测试我们的除法算法 ===" << std::endl;
    
    UInt256 our_quotient = p_plus_1 / p;
    UInt256 our_remainder = p_plus_1 % p;
    UInt256 our_result = our_quotient * p + our_remainder;
    
    std::cout << "我们的商: " << our_quotient.to_hex() << std::endl;
    std::cout << "我们的余数: " << our_remainder.to_hex() << std::endl;
    std::cout << "我们的验证: " << our_result.to_hex() << std::endl;
    std::cout << "我们的算法正确: " << (our_result == p_plus_1 ? "是" : "否") << std::endl;
    
    // 分析差异
    if (our_result != p_plus_1) {
        std::cout << "\n=== 差异分析 ===" << std::endl;
        std::cout << "期望结果: " << p_plus_1.to_hex() << std::endl;
        std::cout << "实际结果: " << our_result.to_hex() << std::endl;
        
        if (our_result < p_plus_1) {
            UInt256 diff = p_plus_1 - our_result;
            std::cout << "差值: " << diff.to_hex() << std::endl;
        } else {
            UInt256 diff = our_result - p_plus_1;
            std::cout << "差值: " << diff.to_hex() << std::endl;
        }
    }
    
    // 测试更简单的相似案例
    std::cout << "\n=== 测试相似的简单案例 ===" << std::endl;
    
    UInt256 simple_divisor(0xFFFFFFFFFFFFFFFEULL, 0, 0, 0);
    UInt256 simple_dividend = simple_divisor + UInt256(1, 0, 0, 0);
    
    UInt256 simple_q = simple_dividend / simple_divisor;
    UInt256 simple_r = simple_dividend % simple_divisor;
    
    std::cout << "简单案例: (0xFFFFFFFFFFFFFFFE + 1) / 0xFFFFFFFFFFFFFFFE" << std::endl;
    std::cout << "商: " << simple_q.to_hex() << std::endl;
    std::cout << "余数: " << simple_r.to_hex() << std::endl;
    std::cout << "正确: " << (simple_q == UInt256(1, 0, 0, 0) && simple_r == UInt256(1, 0, 0, 0) ? "是" : "否") << std::endl;
}

int main() {
    std::cout << "轻量级ECC库 - 特定除法案例调试" << std::endl;
    std::cout << "===============================" << std::endl;
    
    try {
        debug_specific_division_case();
        
        std::cout << "\n🔍 特定除法案例调试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 调试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
