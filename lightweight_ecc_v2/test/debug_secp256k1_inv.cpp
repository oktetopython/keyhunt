#include "../include/uint256.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>

void debug_secp256k1_inv() {
    std::cout << "=== 调试secp256k1模逆运算 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 初始化secp256k1
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    std::cout << "secp256k1素数p = " << p.to_hex() << std::endl;
    
    // 测试简单的数值
    UInt256 a(12, 0, 0, 0);
    std::cout << "a = " << a.to_hex() << std::endl;
    
    // 计算p-2
    UInt256 two(2, 0, 0, 0);
    UInt256 p_minus_2 = p - two;
    std::cout << "p-2 = " << p_minus_2.to_hex() << std::endl;
    
    // 检查p-2是否正确
    UInt256 check_p = p_minus_2 + two;
    std::cout << "验证: (p-2) + 2 = " << check_p.to_hex() << std::endl;
    std::cout << "p == (p-2) + 2: " << (p == check_p) << std::endl;
    
    // 测试模幂运算
    std::cout << "\n测试模幂运算:" << std::endl;
    UInt256 result = ModOp::mod_pow(a, p_minus_2, p);
    std::cout << "mod_pow(12, p-2, p) = " << result.to_hex() << std::endl;
    
    // 验证结果
    UInt256 verification = ModOp::mul(a, result);
    std::cout << "12 * result mod p = " << verification.to_hex() << std::endl;
    
    // 检查是否等于1
    UInt256 one(1, 0, 0, 0);
    std::cout << "result == 1: " << (verification == one) << std::endl;
    
    // 测试ModOp::inv函数
    std::cout << "\n测试ModOp::inv函数:" << std::endl;
    try {
        UInt256 inv_result = ModOp::inv(a);
        std::cout << "ModOp::inv(12) = " << inv_result.to_hex() << std::endl;
        
        UInt256 inv_verification = ModOp::mul(a, inv_result);
        std::cout << "12 * inv(12) mod p = " << inv_verification.to_hex() << std::endl;
        std::cout << "inv result == 1: " << (inv_verification == one) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ModOp::inv异常: " << e.what() << std::endl;
    }
}

int main() {
    try {
        debug_secp256k1_inv();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
