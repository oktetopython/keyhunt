#include "../include/uint256.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>

void debug_specific_large_mod_mul() {
    std::cout << "=== 调试特定大数模乘法 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    std::cout << "p = " << p.to_hex() << std::endl;
    
    // 计算(p+1)/2
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    UInt256 inv_2 = p_plus_1 / UInt256(2, 0, 0, 0);
    
    std::cout << "(p+1)/2 = " << inv_2.to_hex() << std::endl;
    
    // 测试：2 * ((p+1)/2) mod p 应该等于 1
    UInt256 two(2, 0, 0, 0);
    
    std::cout << "\n=== 逐步调试模乘法 ===" << std::endl;
    
    // 检查输入
    std::cout << "a = 2 = " << two.to_hex() << std::endl;
    std::cout << "b = (p+1)/2 = " << inv_2.to_hex() << std::endl;
    
    // 检查输入是否在模数范围内
    std::cout << "a < p: " << (two < p ? "YES" : "NO") << std::endl;
    std::cout << "b < p: " << (inv_2 < p ? "YES" : "NO") << std::endl;
    
    // 使用我们的模乘法
    UInt256 mod_result = ModOp::mul(two, inv_2);
    std::cout << "ModOp::mul(2, (p+1)/2) = " << mod_result.to_hex() << std::endl;
    
    // 使用直接乘法然后取模
    UInt256 direct_mul = two * inv_2;
    std::cout << "直接乘法 2 * (p+1)/2 = " << direct_mul.to_hex() << std::endl;
    
    UInt256 direct_mod = direct_mul % p;
    std::cout << "直接乘法取模 = " << direct_mod.to_hex() << std::endl;
    
    // 比较结果
    std::cout << "\n=== 结果比较 ===" << std::endl;
    std::cout << "ModOp结果: " << mod_result.to_hex() << std::endl;
    std::cout << "直接计算结果: " << direct_mod.to_hex() << std::endl;
    std::cout << "结果一致: " << (mod_result == direct_mod ? "YES" : "NO") << std::endl;
    
    UInt256 expected(1, 0, 0, 0);
    std::cout << "ModOp结果 == 1: " << (mod_result == expected ? "YES" : "NO") << std::endl;
    std::cout << "直接计算 == 1: " << (direct_mod == expected ? "YES" : "NO") << std::endl;
    
    // 分析错误
    if (mod_result != expected) {
        std::cout << "\n=== 错误分析 ===" << std::endl;
        std::cout << "期望: " << expected.to_hex() << std::endl;
        std::cout << "实际: " << mod_result.to_hex() << std::endl;
        
        if (mod_result > expected) {
            UInt256 diff = mod_result - expected;
            std::cout << "差值: " << diff.to_hex() << std::endl;
            
            // 检查差值是否是p的倍数
            if (diff >= p) {
                UInt256 diff_div_p = diff / p;
                UInt256 diff_mod_p = diff % p;
                std::cout << "差值 / p = " << diff_div_p.to_hex() << std::endl;
                std::cout << "差值 % p = " << diff_mod_p.to_hex() << std::endl;
                
                if (diff_mod_p.is_zero()) {
                    std::cout << "发现：差值是p的倍数！这说明我们的模运算没有正确执行。" << std::endl;
                }
            }
        }
    }
}

void test_modop_mul_step_by_step() {
    std::cout << "\n=== 逐步测试ModOp::mul算法 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    UInt256 a(2, 0, 0, 0);
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    UInt256 b = p_plus_1 / UInt256(2, 0, 0, 0);
    
    std::cout << "测试 ModOp::mul(2, (p+1)/2)" << std::endl;
    std::cout << "a = " << a.to_hex() << std::endl;
    std::cout << "b = " << b.to_hex() << std::endl;
    
    // 手动实现ModOp::mul的逻辑
    UInt256 result(0, 0, 0, 0);
    UInt256 temp = a;
    
    std::cout << "\n逐步执行二进制乘法:" << std::endl;
    
    int step = 0;
    for (int i = 0; i < 256 && step < 10; i++) {  // 只显示前10步
        if (b.get_bit(i)) {
            std::cout << "步骤 " << step << " (bit " << i << "): ";
            std::cout << "result += temp" << std::endl;
            std::cout << "  result = " << result.to_hex().substr(48, 16) << " (低64位)" << std::endl;
            std::cout << "  temp = " << temp.to_hex().substr(48, 16) << " (低64位)" << std::endl;
            
            UInt256 old_result = result;
            result = ModOp::safe_add_mod(result, temp);
            
            std::cout << "  new_result = " << result.to_hex().substr(48, 16) << " (低64位)" << std::endl;
            
            // 检查是否有异常大的跳跃
            if (result < old_result) {
                std::cout << "  ⚠️ 检测到结果减小，可能发生了模运算" << std::endl;
            }
            
            step++;
        }
        
        // temp = temp * 2 mod p
        temp = ModOp::safe_add_mod(temp, temp);
        
        // 检查temp是否变得过大
        if (temp >= p) {
            std::cout << "  ⚠️ temp >= p，这不应该发生" << std::endl;
        }
    }
    
    std::cout << "\n最终结果: " << result.to_hex() << std::endl;
    std::cout << "期望结果: " << UInt256(1, 0, 0, 0).to_hex() << std::endl;
}

void test_safe_add_mod_directly() {
    std::cout << "\n=== 直接测试safe_add_mod ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    // 测试接近p的大数加法
    UInt256 large1 = p - UInt256(10, 0, 0, 0);  // p - 10
    UInt256 large2(20, 0, 0, 0);  // 20
    
    std::cout << "测试 (p-10) + 20 mod p" << std::endl;
    std::cout << "a = p - 10 = " << large1.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "b = 20 = " << large2.to_hex() << std::endl;
    
    UInt256 safe_result = ModOp::safe_add_mod(large1, large2);
    std::cout << "safe_add_mod结果 = " << safe_result.to_hex() << std::endl;
    
    // 手动计算期望结果
    UInt256 manual_sum = large1 + large2;
    UInt256 manual_mod = manual_sum % p;
    std::cout << "手动计算结果 = " << manual_mod.to_hex() << std::endl;
    
    std::cout << "结果一致: " << (safe_result == manual_mod ? "YES" : "NO") << std::endl;
    
    // 期望结果应该是 10
    UInt256 expected(10, 0, 0, 0);
    std::cout << "结果 == 10: " << (safe_result == expected ? "YES" : "NO") << std::endl;
}

int main() {
    std::cout << "大数模乘法调试工具" << std::endl;
    std::cout << "==================" << std::endl;
    
    try {
        debug_specific_large_mod_mul();
        test_modop_mul_step_by_step();
        test_safe_add_mod_directly();
        
        std::cout << "\n=== 大数模乘法调试完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "调试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
