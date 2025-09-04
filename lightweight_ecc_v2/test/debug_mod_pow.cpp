#include "../include/uint256.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>
#include <chrono>

void test_simple_mod_pow() {
    std::cout << "=== 简单模幂运算测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    // 测试简单的模幂运算：2^3 mod p = 8
    UInt256 base(2, 0, 0, 0);
    UInt256 exp(3, 0, 0, 0);
    UInt256 expected(8, 0, 0, 0);
    
    std::cout << "测试：2^3 mod p" << std::endl;
    std::cout << "base = " << base.to_hex() << std::endl;
    std::cout << "exp = " << exp.to_hex() << std::endl;
    
    UInt256 result = ModOp::pow_mod_optimized_fermat(base, exp);
    std::cout << "结果 = " << result.to_hex() << std::endl;
    std::cout << "期望 = " << expected.to_hex() << std::endl;
    std::cout << "正确: " << (result == expected ? "YES" : "NO") << std::endl;
    
    // 手动验证：2^3 = 2 * 2 * 2
    UInt256 manual = ModOp::mul(base, base);  // 2 * 2 = 4
    manual = ModOp::mul(manual, base);        // 4 * 2 = 8
    std::cout << "手动计算 = " << manual.to_hex() << std::endl;
    std::cout << "手动计算正确: " << (manual == expected ? "YES" : "NO") << std::endl;
}

void test_fermat_little_theorem() {
    std::cout << "\n=== 费马小定理测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    // 费马小定理：a^(p-1) ≡ 1 mod p （对于素数p和gcd(a,p)=1）
    UInt256 a(2, 0, 0, 0);
    UInt256 p_minus_1 = p - UInt256(1, 0, 0, 0);
    
    std::cout << "测试费马小定理：2^(p-1) mod p 应该等于 1" << std::endl;
    std::cout << "a = " << a.to_hex() << std::endl;
    std::cout << "p-1 = " << p_minus_1.to_hex().substr(0, 32) << "..." << std::endl;
    
    std::cout << "开始计算 2^(p-1) mod p..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    UInt256 result = ModOp::pow_mod_optimized_fermat(a, p_minus_1);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "结果 = " << result.to_hex() << std::endl;
    std::cout << "计算时间: " << duration.count() << " 毫秒" << std::endl;
    
    UInt256 expected(1, 0, 0, 0);
    std::cout << "费马小定理验证: " << (result == expected ? "✅ 通过" : "❌ 失败") << std::endl;
    
    if (result != expected) {
        std::cout << "期望: " << expected.to_hex() << std::endl;
        std::cout << "实际: " << result.to_hex() << std::endl;
    }
}

void test_modular_inverse_calculation() {
    std::cout << "\n=== 模逆计算测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    // 测试2的模逆：2^(-1) ≡ 2^(p-2) mod p
    UInt256 a(2, 0, 0, 0);
    UInt256 p_minus_2 = p - UInt256(2, 0, 0, 0);
    
    std::cout << "测试2的模逆：2^(p-2) mod p" << std::endl;
    std::cout << "a = " << a.to_hex() << std::endl;
    std::cout << "p-2 = " << p_minus_2.to_hex().substr(0, 32) << "..." << std::endl;
    
    std::cout << "开始计算 2^(p-2) mod p..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    UInt256 inv_result = ModOp::pow_mod_optimized_fermat(a, p_minus_2);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "2的模逆 = " << inv_result.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "计算时间: " << duration.count() << " 毫秒" << std::endl;
    
    // 验证：2 * inv(2) mod p 应该等于 1
    std::cout << "\n验证：2 * inv(2) mod p" << std::endl;
    UInt256 verification = ModOp::mul(a, inv_result);
    std::cout << "2 * inv(2) = " << verification.to_hex() << std::endl;
    
    UInt256 expected(1, 0, 0, 0);
    std::cout << "验证结果: " << (verification == expected ? "✅ 通过" : "❌ 失败") << std::endl;
    
    if (verification != expected) {
        std::cout << "期望: " << expected.to_hex() << std::endl;
        std::cout << "实际: " << verification.to_hex() << std::endl;
        
        // 检查是否是模运算问题
        UInt256 direct_mul = a * inv_result;
        UInt256 direct_mod = direct_mul % p;
        std::cout << "直接计算 (2 * inv(2)) % p = " << direct_mod.to_hex() << std::endl;
        std::cout << "直接计算正确: " << (direct_mod == expected ? "YES" : "NO") << std::endl;
    }
    
    // 与已知正确值比较
    UInt256 known_inv = (p + UInt256(1, 0, 0, 0)) / UInt256(2, 0, 0, 0);
    std::cout << "\n与已知正确值比较:" << std::endl;
    std::cout << "已知2的模逆 = " << known_inv.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "计算的模逆 = " << inv_result.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "模逆值一致: " << (inv_result == known_inv ? "YES" : "NO") << std::endl;
}

void debug_mod_pow_algorithm() {
    std::cout << "\n=== 调试模幂算法 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    // 测试一个小的指数
    UInt256 base(2, 0, 0, 0);
    UInt256 exp(10, 0, 0, 0);  // 2^10 = 1024
    
    std::cout << "调试：2^10 mod p" << std::endl;
    std::cout << "base = " << base.to_hex() << std::endl;
    std::cout << "exp = " << exp.to_hex() << std::endl;
    
    // 手动实现快速幂算法
    UInt256 result(1, 0, 0, 0);
    UInt256 current = base;
    UInt256 e = exp;
    
    std::cout << "\n逐步执行快速幂:" << std::endl;
    int step = 0;
    while (!e.is_zero() && step < 10) {
        std::cout << "步骤 " << step << ":" << std::endl;
        std::cout << "  e = " << e.to_hex() << std::endl;
        std::cout << "  current = " << current.to_hex().substr(48, 16) << " (低64位)" << std::endl;
        std::cout << "  result = " << result.to_hex().substr(48, 16) << " (低64位)" << std::endl;
        
        if (e.is_odd()) {
            std::cout << "  e是奇数，result *= current" << std::endl;
            result = ModOp::mul(result, current);
            std::cout << "  new result = " << result.to_hex().substr(48, 16) << " (低64位)" << std::endl;
        }
        
        current = ModOp::mul(current, current);
        e = e >> 1;
        step++;
    }
    
    std::cout << "\n最终结果: " << result.to_hex() << std::endl;
    
    // 验证：2^10 = 1024
    UInt256 expected(1024, 0, 0, 0);
    std::cout << "期望结果: " << expected.to_hex() << std::endl;
    std::cout << "结果正确: " << (result == expected ? "YES" : "NO") << std::endl;
    
    // 使用我们的函数
    UInt256 func_result = ModOp::pow_mod_optimized_fermat(base, exp);
    std::cout << "函数结果: " << func_result.to_hex() << std::endl;
    std::cout << "函数正确: " << (func_result == expected ? "YES" : "NO") << std::endl;
}

int main() {
    std::cout << "模幂运算调试工具" << std::endl;
    std::cout << "================" << std::endl;
    
    try {
        test_simple_mod_pow();
        debug_mod_pow_algorithm();
        test_modular_inverse_calculation();
        // test_fermat_little_theorem();  // 这个测试时间很长，先注释掉
        
        std::cout << "\n=== 模幂运算调试完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "调试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
