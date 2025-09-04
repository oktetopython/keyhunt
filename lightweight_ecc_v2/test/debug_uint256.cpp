#include "../include/uint256.h"
#include <iostream>

void debug_uint256_basic() {
    std::cout << "=== 调试UInt256基本功能 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 测试构造函数
    UInt256 a(12345, 0, 0, 0);
    UInt256 b(67890, 0, 0, 0);
    UInt256 zero_test(0, 0, 0, 0);
    UInt256 one_test(1, 0, 0, 0);
    
    std::cout << "构造函数测试:" << std::endl;
    std::cout << "a.limbs[0] = " << a.limbs[0] << std::endl;
    std::cout << "a.limbs[1] = " << a.limbs[1] << std::endl;
    std::cout << "a.limbs[2] = " << a.limbs[2] << std::endl;
    std::cout << "a.limbs[3] = " << a.limbs[3] << std::endl;
    
    std::cout << "b.limbs[0] = " << b.limbs[0] << std::endl;
    std::cout << "b.limbs[1] = " << b.limbs[1] << std::endl;
    std::cout << "b.limbs[2] = " << b.limbs[2] << std::endl;
    std::cout << "b.limbs[3] = " << b.limbs[3] << std::endl;
    
    // 测试to_hex函数
    std::cout << "\nto_hex测试:" << std::endl;
    std::cout << "a.to_hex() = " << a.to_hex() << std::endl;
    std::cout << "b.to_hex() = " << b.to_hex() << std::endl;
    std::cout << "zero.to_hex() = " << zero_test.to_hex() << std::endl;
    std::cout << "one.to_hex() = " << one_test.to_hex() << std::endl;
    
    // 测试基本运算
    std::cout << "\n基本运算测试:" << std::endl;
    UInt256 sum = a + b;
    std::cout << "a + b = " << sum.to_hex() << std::endl;
    std::cout << "sum.limbs[0] = " << sum.limbs[0] << std::endl;
    
    UInt256 diff = b - a;
    std::cout << "b - a = " << diff.to_hex() << std::endl;
    std::cout << "diff.limbs[0] = " << diff.limbs[0] << std::endl;
    
    // 测试比较
    std::cout << "\n比较测试:" << std::endl;
    std::cout << "a == b: " << (a == b) << std::endl;
    std::cout << "a != b: " << (a != b) << std::endl;
    std::cout << "a < b: " << (a < b) << std::endl;
    std::cout << "a > b: " << (a > b) << std::endl;
    
    // 测试is_zero
    std::cout << "\nis_zero测试:" << std::endl;
    std::cout << "a.is_zero(): " << a.is_zero() << std::endl;
    std::cout << "zero_test.is_zero(): " << zero_test.is_zero() << std::endl;
    
    // 测试静态常量
    std::cout << "\n静态常量测试:" << std::endl;
    std::cout << "UInt256::get_zero().to_hex() = " << UInt256::get_zero().to_hex() << std::endl;
    std::cout << "UInt256::get_one().to_hex() = " << UInt256::get_one().to_hex() << std::endl;
}

int main() {
    try {
        debug_uint256_basic();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
