#include "../include/uint256.h"
#include "../include/point.h"
#include "../include/secp256k1.h"
#include <iostream>

void debug_point_operations() {
    std::cout << "=== 调试椭圆曲线点运算 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 初始化
    Secp256k1::init();
    Point G = Secp256k1::get_generator();
    
    std::cout << "生成元 G:" << std::endl;
    std::cout << "  x = " << G.x.to_hex().substr(0, 16) << "..." << std::endl;
    std::cout << "  y = " << G.y.to_hex().substr(0, 16) << "..." << std::endl;
    std::cout << std::endl;
    
    // 测试点倍法
    std::cout << "测试点倍法:" << std::endl;
    Point G2_double = Secp256k1::point_double(G);
    Point G2_scalar = Secp256k1::scalar_mul(UInt256(2, 0, 0, 0));
    
    std::cout << "G2 (point_double):" << std::endl;
    std::cout << "  x = " << G2_double.x.to_hex().substr(0, 16) << "..." << std::endl;
    std::cout << "  y = " << G2_double.y.to_hex().substr(0, 16) << "..." << std::endl;
    
    std::cout << "G2 (scalar_mul(2)):" << std::endl;
    std::cout << "  x = " << G2_scalar.x.to_hex().substr(0, 16) << "..." << std::endl;
    std::cout << "  y = " << G2_scalar.y.to_hex().substr(0, 16) << "..." << std::endl;
    
    bool double_equal = (G2_double == G2_scalar);
    std::cout << "point_double(G) == scalar_mul(2): " << (double_equal ? "是" : "否") << std::endl;
    std::cout << std::endl;
    
    // 使用point_double的结果继续测试
    Point G2 = G2_double;
    
    // 测试3G
    std::cout << "测试3G:" << std::endl;
    Point G3_scalar = Secp256k1::scalar_mul(UInt256(3, 0, 0, 0));
    Point G3_add = Secp256k1::point_add(G, G2);
    
    std::cout << "G3 (scalar_mul(3)):" << std::endl;
    std::cout << "  x = " << G3_scalar.x.to_hex().substr(0, 16) << "..." << std::endl;
    std::cout << "  y = " << G3_scalar.y.to_hex().substr(0, 16) << "..." << std::endl;
    
    std::cout << "G3 (G + G2):" << std::endl;
    std::cout << "  x = " << G3_add.x.to_hex().substr(0, 16) << "..." << std::endl;
    std::cout << "  y = " << G3_add.y.to_hex().substr(0, 16) << "..." << std::endl;
    
    bool add_equal = (G3_scalar == G3_add);
    std::cout << "scalar_mul(3) == G + G2: " << (add_equal ? "是" : "否") << std::endl;
    std::cout << std::endl;
    
    // 测试交换律
    std::cout << "测试交换律:" << std::endl;
    Point G_plus_G2 = Secp256k1::point_add(G, G2);
    Point G2_plus_G = Secp256k1::point_add(G2, G);
    
    std::cout << "G + G2:" << std::endl;
    std::cout << "  x = " << G_plus_G2.x.to_hex().substr(0, 16) << "..." << std::endl;
    std::cout << "  y = " << G_plus_G2.y.to_hex().substr(0, 16) << "..." << std::endl;
    
    std::cout << "G2 + G:" << std::endl;
    std::cout << "  x = " << G2_plus_G.x.to_hex().substr(0, 16) << "..." << std::endl;
    std::cout << "  y = " << G2_plus_G.y.to_hex().substr(0, 16) << "..." << std::endl;
    
    bool commutative = (G_plus_G2 == G2_plus_G);
    std::cout << "G + G2 == G2 + G: " << (commutative ? "是" : "否") << std::endl;
    std::cout << std::endl;
    
    // 详细比较
    if (!commutative) {
        std::cout << "详细比较 G + G2 vs G2 + G:" << std::endl;
        std::cout << "X坐标相等: " << (G_plus_G2.x == G2_plus_G.x ? "是" : "否") << std::endl;
        std::cout << "Y坐标相等: " << (G_plus_G2.y == G2_plus_G.y ? "是" : "否") << std::endl;
        std::cout << "无穷远点状态: " << G_plus_G2.infinity << " vs " << G2_plus_G.infinity << std::endl;
    }
    
    // 检查是否在曲线上
    std::cout << "验证点是否在曲线上:" << std::endl;
    std::cout << "G 在曲线上: " << (Secp256k1::is_on_curve(G) ? "是" : "否") << std::endl;
    std::cout << "G2 在曲线上: " << (Secp256k1::is_on_curve(G2) ? "是" : "否") << std::endl;
    std::cout << "G + G2 在曲线上: " << (Secp256k1::is_on_curve(G_plus_G2) ? "是" : "否") << std::endl;
    std::cout << "G2 + G 在曲线上: " << (Secp256k1::is_on_curve(G2_plus_G) ? "是" : "否") << std::endl;
}

int main() {
    try {
        debug_point_operations();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
