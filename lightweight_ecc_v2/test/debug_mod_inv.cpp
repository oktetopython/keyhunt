#include "../include/uint256.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>

void debug_mod_inv() {
    std::cout << "=== 调试模逆运算 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 初始化secp256k1
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    std::cout << "secp256k1素数p = " << p.to_hex() << std::endl;
    
    // 测试简单的模逆
    UInt256 a(12345, 0, 0, 0);
    std::cout << "a = " << a.to_hex() << std::endl;
    
    // 测试模幂运算
    UInt256 p_minus_2 = p - UInt256(2, 0, 0, 0);
    std::cout << "p-2 = " << p_minus_2.to_hex().substr(0, 32) << "..." << std::endl;
    
    // 手动测试小的模幂运算
    UInt256 small_mod(97, 0, 0, 0);  // 使用小素数97
    UInt256 small_a(12, 0, 0, 0);
    UInt256 small_exp(95, 0, 0, 0);  // 97-2 = 95
    
    std::cout << "\n使用小素数97测试:" << std::endl;
    std::cout << "a = " << small_a.limbs[0] << std::endl;
    std::cout << "mod = " << small_mod.limbs[0] << std::endl;
    std::cout << "exp = " << small_exp.limbs[0] << std::endl;
    
    // 手动计算 12^95 mod 97
    UInt256 result(1, 0, 0, 0);
    UInt256 base = small_a;
    UInt256 exp = small_exp;
    
    std::cout << "手动计算 12^95 mod 97:" << std::endl;
    while (!exp.is_zero()) {
        if (exp.is_odd()) {
            result = (result * base) % small_mod;
            std::cout << "result = " << result.limbs[0] << std::endl;
        }
        base = (base * base) % small_mod;
        exp = exp >> 1;
        if (!exp.is_zero()) {
            std::cout << "base = " << base.limbs[0] << ", exp = " << exp.limbs[0] << std::endl;
        }
    }
    
    std::cout << "最终结果: " << result.limbs[0] << std::endl;
    
    // 验证: 12 * result mod 97 应该等于 1
    UInt256 check = (small_a * result) % small_mod;
    std::cout << "验证: 12 * " << result.limbs[0] << " mod 97 = " << check.limbs[0] << std::endl;
    
    // 测试我们的mod_pow函数
    std::cout << "\n测试mod_pow函数:" << std::endl;
    UInt256 mod_pow_result = ModOp::mod_pow(small_a, small_exp, small_mod);
    std::cout << "mod_pow(12, 95, 97) = " << mod_pow_result.limbs[0] << std::endl;
}

int main() {
    try {
        debug_mod_inv();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
