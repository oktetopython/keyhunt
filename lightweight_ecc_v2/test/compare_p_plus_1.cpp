#include "../include/uint256.h"
#include "../include/secp256k1.h"
#include "../include/mod_op.h"
#include <iostream>

void compare_p_plus_1_calculations() {
    std::cout << "=== 比较p+1计算方法 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 方法1：直接从hex创建p，然后加1
    UInt256 p1 = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    UInt256 p1_plus_1 = p1 + UInt256(1, 0, 0, 0);
    
    std::cout << "方法1 - 直接hex创建p:" << std::endl;
    std::cout << "p1 = " << p1.to_hex() << std::endl;
    std::cout << "p1+1 = " << p1_plus_1.to_hex() << std::endl;
    std::cout << "(p1+1) % p1 = " << (p1_plus_1 % p1).to_hex() << std::endl;
    
    // 方法2：通过Secp256k1获取p，然后加1
    Secp256k1::init();
    UInt256 p2 = Secp256k1::get_params().p;
    UInt256 p2_plus_1 = p2 + UInt256(1, 0, 0, 0);
    
    std::cout << "\n方法2 - Secp256k1获取p:" << std::endl;
    std::cout << "p2 = " << p2.to_hex() << std::endl;
    std::cout << "p2+1 = " << p2_plus_1.to_hex() << std::endl;
    std::cout << "(p2+1) % p2 = " << (p2_plus_1 % p2).to_hex() << std::endl;
    
    // 比较两种方法
    std::cout << "\n=== 比较结果 ===" << std::endl;
    std::cout << "p1 == p2: " << (p1 == p2 ? "YES" : "NO") << std::endl;
    std::cout << "p1+1 == p2+1: " << (p1_plus_1 == p2_plus_1 ? "YES" : "NO") << std::endl;
    
    UInt256 result1 = p1_plus_1 % p1;
    UInt256 result2 = p2_plus_1 % p2;
    std::cout << "结果1 == 结果2: " << (result1 == result2 ? "YES" : "NO") << std::endl;
    
    UInt256 expected(1, 0, 0, 0);
    std::cout << "结果1 == 1: " << (result1 == expected ? "YES" : "NO") << std::endl;
    std::cout << "结果2 == 1: " << (result2 == expected ? "YES" : "NO") << std::endl;
    
    // 详细分析差异
    if (result1 != result2) {
        std::cout << "\n=== 差异分析 ===" << std::endl;
        std::cout << "结果1: " << result1.to_hex() << std::endl;
        std::cout << "结果2: " << result2.to_hex() << std::endl;
        
        if (result1 > result2) {
            UInt256 diff = result1 - result2;
            std::cout << "差值: " << diff.to_hex() << std::endl;
        } else {
            UInt256 diff = result2 - result1;
            std::cout << "差值: " << diff.to_hex() << std::endl;
        }
    }
    
    // 测试ModOp是否影响结果
    std::cout << "\n=== 测试ModOp影响 ===" << std::endl;
    
    // 不使用ModOp，直接计算
    UInt256 direct_result = p2_plus_1 % p2;
    std::cout << "直接计算: " << direct_result.to_hex() << std::endl;
    
    // 使用ModOp
    ModOp::init(p2);
    UInt256 modop_result = ModOp::mul(UInt256(1, 0, 0, 0), UInt256(1, 0, 0, 0));
    std::cout << "ModOp 1*1: " << modop_result.to_hex() << std::endl;
    
    // 检查ModOp是否改变了全局状态
    UInt256 after_modop_result = p2_plus_1 % p2;
    std::cout << "ModOp后计算: " << after_modop_result.to_hex() << std::endl;
    
    std::cout << "ModOp前后一致: " << (direct_result == after_modop_result ? "YES" : "NO") << std::endl;
}

void test_step_by_step_calculation() {
    std::cout << "\n=== 逐步计算测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    
    std::cout << "p = " << p.to_hex() << std::endl;
    
    // 逐步计算p+1
    UInt256 one(1, 0, 0, 0);
    std::cout << "1 = " << one.to_hex() << std::endl;
    
    UInt256 p_plus_1 = p + one;
    std::cout << "p + 1 = " << p_plus_1.to_hex() << std::endl;
    
    // 检查加法是否正确
    UInt256 diff = p_plus_1 - p;
    std::cout << "(p+1) - p = " << diff.to_hex() << std::endl;
    std::cout << "差值是否为1: " << (diff == one ? "YES" : "NO") << std::endl;
    
    // 逐步计算模运算
    std::cout << "\n逐步模运算:" << std::endl;
    std::cout << "被除数: " << p_plus_1.to_hex() << std::endl;
    std::cout << "除数: " << p.to_hex() << std::endl;
    
    UInt256 quotient = p_plus_1 / p;
    UInt256 remainder = p_plus_1 % p;
    
    std::cout << "商: " << quotient.to_hex() << std::endl;
    std::cout << "余数: " << remainder.to_hex() << std::endl;
    
    // 验证
    UInt256 verification = quotient * p + remainder;
    std::cout << "验证: " << verification.to_hex() << std::endl;
    std::cout << "验证正确: " << (verification == p_plus_1 ? "YES" : "NO") << std::endl;
    
    // 检查余数是否为1
    std::cout << "余数是否为1: " << (remainder == one ? "YES" : "NO") << std::endl;
    
    if (remainder != one) {
        std::cout << "\n余数错误分析:" << std::endl;
        std::cout << "期望: " << one.to_hex() << std::endl;
        std::cout << "实际: " << remainder.to_hex() << std::endl;
        
        // 检查余数的内部表示
        std::cout << "余数limbs: ";
        for (int i = 0; i < 4; i++) {
            std::cout << "0x" << std::hex << remainder.limbs[i] << std::dec << " ";
        }
        std::cout << std::endl;
        
        std::cout << "期望limbs: ";
        for (int i = 0; i < 4; i++) {
            std::cout << "0x" << std::hex << one.limbs[i] << std::dec << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "p+1计算方法比较工具" << std::endl;
    std::cout << "==================" << std::endl;
    
    try {
        compare_p_plus_1_calculations();
        test_step_by_step_calculation();
        
        std::cout << "\n=== 比较完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "比较失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
