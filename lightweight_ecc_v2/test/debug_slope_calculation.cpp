#include "../include/uint256.h"
#include "../include/point.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>

void debug_slope_calculation() {
    std::cout << "=== 调试斜率计算 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 初始化
    Secp256k1::init();
    Point G = Secp256k1::get_generator();
    Point G2 = Secp256k1::scalar_mul(UInt256(2, 0, 0, 0));
    
    std::cout << "G.x = " << G.x.to_hex() << std::endl;
    std::cout << "G.y = " << G.y.to_hex() << std::endl;
    std::cout << "G2.x = " << G2.x.to_hex() << std::endl;
    std::cout << "G2.y = " << G2.y.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 计算 G + G2 的斜率
    std::cout << "计算 G + G2 的斜率:" << std::endl;
    UInt256 x1 = G.x;
    UInt256 y1 = G.y;
    UInt256 x2 = G2.x;
    UInt256 y2 = G2.y;
    
    UInt256 dx_1 = ModOp::sub(x2, x1);
    UInt256 dy_1 = ModOp::sub(y2, y1);
    
    std::cout << "dx = x2 - x1 = " << dx_1.to_hex() << std::endl;
    std::cout << "dy = y2 - y1 = " << dy_1.to_hex() << std::endl;
    
    UInt256 dx_inv_1 = ModOp::inv(dx_1);
    UInt256 lambda_1 = ModOp::mul(dy_1, dx_inv_1);
    
    std::cout << "dx_inv = " << dx_inv_1.to_hex() << std::endl;
    std::cout << "lambda = dy * dx_inv = " << lambda_1.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 计算 G2 + G 的斜率
    std::cout << "计算 G2 + G 的斜率:" << std::endl;
    UInt256 x1_rev = G2.x;
    UInt256 y1_rev = G2.y;
    UInt256 x2_rev = G.x;
    UInt256 y2_rev = G.y;
    
    UInt256 dx_2 = ModOp::sub(x2_rev, x1_rev);
    UInt256 dy_2 = ModOp::sub(y2_rev, y1_rev);
    
    std::cout << "dx = x2 - x1 = " << dx_2.to_hex() << std::endl;
    std::cout << "dy = y2 - y1 = " << dy_2.to_hex() << std::endl;
    
    UInt256 dx_inv_2 = ModOp::inv(dx_2);
    UInt256 lambda_2 = ModOp::mul(dy_2, dx_inv_2);
    
    std::cout << "dx_inv = " << dx_inv_2.to_hex() << std::endl;
    std::cout << "lambda = dy * dx_inv = " << lambda_2.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 比较斜率
    std::cout << "斜率比较:" << std::endl;
    std::cout << "lambda_1 == lambda_2: " << (lambda_1 == lambda_2 ? "是" : "否") << std::endl;
    
    // 检查 dx 和 dy 的关系
    std::cout << "\n检查 dx 和 dy 的关系:" << std::endl;
    std::cout << "dx_1 = " << dx_1.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "dx_2 = " << dx_2.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    
    // dx_2 应该等于 -dx_1 (mod p)
    UInt256 neg_dx_1 = ModOp::sub(UInt256(0, 0, 0, 0), dx_1);
    std::cout << "-dx_1 = " << neg_dx_1.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "dx_2 == -dx_1: " << (dx_2 == neg_dx_1 ? "是" : "否") << std::endl;
    
    std::cout << "dy_1 = " << dy_1.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "dy_2 = " << dy_2.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    
    // dy_2 应该等于 -dy_1 (mod p)
    UInt256 neg_dy_1 = ModOp::sub(UInt256(0, 0, 0, 0), dy_1);
    std::cout << "-dy_1 = " << neg_dy_1.to_hex().substr(48, 16) << " (低64位)" << std::endl;
    std::cout << "dy_2 == -dy_1: " << (dy_2 == neg_dy_1 ? "是" : "否") << std::endl;
    
    // 如果 dx_2 = -dx_1 且 dy_2 = -dy_1，那么 lambda_2 = (-dy_1)/(-dx_1) = dy_1/dx_1 = lambda_1
    if (dx_2 == neg_dx_1 && dy_2 == neg_dy_1) {
        std::cout << "\n✅ dx和dy的符号关系正确，斜率应该相等" << std::endl;
    } else {
        std::cout << "\n❌ dx和dy的符号关系不正确" << std::endl;
    }
}

int main() {
    try {
        debug_slope_calculation();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
