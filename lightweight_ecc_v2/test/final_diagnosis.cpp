#include "../include/uint256.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>

void diagnose_core_problems() {
    std::cout << "=== 最终诊断：找出真正的问题 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    std::cout << "secp256k1素数p = " << p.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 问题1：为什么(p+1) mod p != 1？
    std::cout << "=== 问题1：(p+1) mod p 分析 ===" << std::endl;
    
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    UInt256 result = p_plus_1 % p;
    
    std::cout << "p = " << p.to_hex() << std::endl;
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    std::cout << "(p+1) mod p = " << result.to_hex() << std::endl;
    
    // 手动验证：(p+1) - p 应该等于 1
    UInt256 manual_result = p_plus_1 - p;
    std::cout << "手动计算：(p+1) - p = " << manual_result.to_hex() << std::endl;
    
    bool manual_correct = (manual_result == UInt256(1, 0, 0, 0));
    std::cout << "手动计算正确: " << (manual_correct ? "是" : "否") << std::endl;
    
    // 分析差异
    if (result != UInt256(1, 0, 0, 0)) {
        std::cout << "问题：我们的模运算实现有错误！" << std::endl;
        std::cout << "期望: 0000000000000000000000000000000000000000000000000000000000000001" << std::endl;
        std::cout << "实际: " << result.to_hex() << std::endl;
    }
    
    std::cout << std::endl;
    
    // 问题2：为什么模乘法验证失败？
    std::cout << "=== 问题2：模乘法验证失败分析 ===" << std::endl;
    
    UInt256 a(2, 0, 0, 0);
    UInt256 b(3, 0, 0, 0);
    UInt256 expected(6, 0, 0, 0);
    
    UInt256 mul_result = ModOp::mul(a, b);
    
    std::cout << "简单测试：2 * 3 mod p" << std::endl;
    std::cout << "结果: " << mul_result.to_hex() << std::endl;
    std::cout << "期望: " << expected.to_hex() << std::endl;
    std::cout << "正确: " << (mul_result == expected ? "是" : "否") << std::endl;
    
    // 测试大数乘法
    UInt256 large_a = p / UInt256(2, 0, 0, 0);  // p/2
    UInt256 large_b(2, 0, 0, 0);
    UInt256 large_result = ModOp::mul(large_a, large_b);
    
    std::cout << "\n大数测试：(p/2) * 2 mod p" << std::endl;
    std::cout << "p/2 = " << large_a.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "(p/2) * 2 mod p = " << large_result.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    
    // 理论上 (p/2) * 2 应该接近 p，但要小于 p
    std::cout << "结果是否 < p: " << (large_result < p ? "是" : "否") << std::endl;
    
    std::cout << std::endl;
    
    // 问题3：模逆运算为什么失败？
    std::cout << "=== 问题3：模逆运算失败的根本原因 ===" << std::endl;
    
    // 测试最简单的情况：1的模逆应该是1
    UInt256 one(1, 0, 0, 0);
    UInt256 one_mul_one = ModOp::mul(one, one);
    
    std::cout << "最简单测试：1 * 1 mod p" << std::endl;
    std::cout << "结果: " << one_mul_one.to_hex() << std::endl;
    std::cout << "期望: " << one.to_hex() << std::endl;
    std::cout << "正确: " << (one_mul_one == one ? "是" : "否") << std::endl;
    
    if (one_mul_one != one) {
        std::cout << "❌ 连最基本的 1*1 mod p 都错误！模乘法实现有根本问题。" << std::endl;
    }
    
    std::cout << std::endl;
    
    // 总结
    std::cout << "=== 诊断总结 ===" << std::endl;
    std::cout << "1. 模运算基础问题: " << (result != UInt256(1, 0, 0, 0) ? "存在" : "不存在") << std::endl;
    std::cout << "2. 模乘法基础问题: " << (one_mul_one != one ? "存在" : "不存在") << std::endl;
    std::cout << "3. 大数运算问题: " << (large_result >= p ? "存在" : "可能存在") << std::endl;
    
    if (result != UInt256(1, 0, 0, 0) || one_mul_one != one) {
        std::cout << "\n❌ 结论：我们的基础模运算实现有根本性错误！" << std::endl;
        std::cout << "这解释了为什么模逆运算和椭圆曲线运算都失败。" << std::endl;
        std::cout << "需要重新审视和修复基础的模运算算法。" << std::endl;
    } else {
        std::cout << "\n✅ 基础模运算正确，问题可能在更高层的算法中。" << std::endl;
    }
}

int main() {
    std::cout << "轻量级ECC库 - 最终问题诊断" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        diagnose_core_problems();
        
        std::cout << "\n🔍 最终诊断完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 诊断失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
