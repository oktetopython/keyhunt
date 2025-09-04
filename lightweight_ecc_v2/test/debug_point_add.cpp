#include "../include/uint256.h"
#include "../include/point.h"
#include "../include/secp256k1.h"
#include <iostream>

void debug_point_addition() {
    std::cout << "=== 调试椭圆曲线点加法交换律 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 初始化
    Secp256k1::init();
    Point G = Secp256k1::get_generator();
    
    std::cout << "生成元 G:" << std::endl;
    std::cout << "  x = " << G.x.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "  y = " << G.y.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "  infinity = " << G.infinity << std::endl;
    std::cout << std::endl;
    
    // 计算G2
    Point G2 = Secp256k1::point_double(G);
    std::cout << "G2 = point_double(G):" << std::endl;
    std::cout << "  x = " << G2.x.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "  y = " << G2.y.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "  infinity = " << G2.infinity << std::endl;
    std::cout << std::endl;
    
    // 计算G + G2
    std::cout << "计算 G + G2:" << std::endl;
    Point G_plus_G2 = Secp256k1::point_add(G, G2);
    std::cout << "G + G2:" << std::endl;
    std::cout << "  x = " << G_plus_G2.x.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "  y = " << G_plus_G2.y.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "  infinity = " << G_plus_G2.infinity << std::endl;
    std::cout << std::endl;
    
    // 计算G2 + G
    std::cout << "计算 G2 + G:" << std::endl;
    Point G2_plus_G = Secp256k1::point_add(G2, G);
    std::cout << "G2 + G:" << std::endl;
    std::cout << "  x = " << G2_plus_G.x.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "  y = " << G2_plus_G.y.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "  infinity = " << G2_plus_G.infinity << std::endl;
    std::cout << std::endl;
    
    // 详细比较
    std::cout << "详细比较:" << std::endl;
    std::cout << "X坐标相等: " << (G_plus_G2.x == G2_plus_G.x ? "是" : "否") << std::endl;
    std::cout << "Y坐标相等: " << (G_plus_G2.y == G2_plus_G.y ? "是" : "否") << std::endl;
    std::cout << "无穷远点状态相等: " << (G_plus_G2.infinity == G2_plus_G.infinity ? "是" : "否") << std::endl;
    
    if (G_plus_G2.x != G2_plus_G.x) {
        std::cout << "X坐标差异:" << std::endl;
        std::cout << "  G + G2.x = " << G_plus_G2.x.to_hex() << std::endl;
        std::cout << "  G2 + G.x = " << G2_plus_G.x.to_hex() << std::endl;
    }
    
    if (G_plus_G2.y != G2_plus_G.y) {
        std::cout << "Y坐标差异:" << std::endl;
        std::cout << "  G + G2.y = " << G_plus_G2.y.to_hex() << std::endl;
        std::cout << "  G2 + G.y = " << G2_plus_G.y.to_hex() << std::endl;
    }
    
    // 验证3G
    std::cout << "\n验证3G:" << std::endl;
    Point G3_scalar = Secp256k1::scalar_mul(UInt256(3, 0, 0, 0));
    std::cout << "3G (scalar_mul):" << std::endl;
    std::cout << "  x = " << G3_scalar.x.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "  y = " << G3_scalar.y.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "  infinity = " << G3_scalar.infinity << std::endl;
    
    std::cout << "G + G2 == 3G: " << (G_plus_G2 == G3_scalar ? "是" : "否") << std::endl;
    std::cout << "G2 + G == 3G: " << (G2_plus_G == G3_scalar ? "是" : "否") << std::endl;
}

int main() {
    try {
        debug_point_addition();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
