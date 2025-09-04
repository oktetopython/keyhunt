#include "../include/uint256.h"
#include "../include/point.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>

void debug_detailed_addition() {
    std::cout << "=== 详细调试椭圆曲线点加法 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 初始化
    Secp256k1::init();
    Point G = Secp256k1::get_generator();
    Point G2 = Secp256k1::scalar_mul(UInt256(2, 0, 0, 0));
    
    std::cout << "输入点:" << std::endl;
    std::cout << "G.x  = " << G.x.to_hex() << std::endl;
    std::cout << "G.y  = " << G.y.to_hex() << std::endl;
    std::cout << "G2.x = " << G2.x.to_hex() << std::endl;
    std::cout << "G2.y = " << G2.y.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 手动计算 G + G2
    std::cout << "=== 手动计算 G + G2 ===" << std::endl;
    UInt256 x1 = G.x;
    UInt256 y1 = G.y;
    UInt256 x2 = G2.x;
    UInt256 y2 = G2.y;
    
    UInt256 dx = ModOp::sub(x2, x1);
    UInt256 dy = ModOp::sub(y2, y1);
    
    std::cout << "dx = x2 - x1 = " << dx.to_hex() << std::endl;
    std::cout << "dy = y2 - y1 = " << dy.to_hex() << std::endl;
    
    UInt256 dx_inv = ModOp::inv(dx);
    UInt256 lambda = ModOp::mul(dy, dx_inv);
    
    std::cout << "dx_inv = " << dx_inv.to_hex() << std::endl;
    std::cout << "lambda = " << lambda.to_hex() << std::endl;
    
    UInt256 lambda_sq = ModOp::mul(lambda, lambda);
    UInt256 x3_1 = ModOp::sub(lambda_sq, x1);
    x3_1 = ModOp::sub(x3_1, x2);
    
    UInt256 x1_minus_x3_1 = ModOp::sub(x1, x3_1);
    UInt256 y3_1 = ModOp::sub(ModOp::mul(lambda, x1_minus_x3_1), y1);
    
    std::cout << "x3 = " << x3_1.to_hex() << std::endl;
    std::cout << "y3 = " << y3_1.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 手动计算 G2 + G
    std::cout << "=== 手动计算 G2 + G ===" << std::endl;
    UInt256 x1_rev = G2.x;
    UInt256 y1_rev = G2.y;
    UInt256 x2_rev = G.x;
    UInt256 y2_rev = G.y;
    
    UInt256 dx_rev = ModOp::sub(x2_rev, x1_rev);
    UInt256 dy_rev = ModOp::sub(y2_rev, y1_rev);
    
    std::cout << "dx = x2 - x1 = " << dx_rev.to_hex() << std::endl;
    std::cout << "dy = y2 - y1 = " << dy_rev.to_hex() << std::endl;
    
    UInt256 dx_inv_rev = ModOp::inv(dx_rev);
    UInt256 lambda_rev = ModOp::mul(dy_rev, dx_inv_rev);
    
    std::cout << "dx_inv = " << dx_inv_rev.to_hex() << std::endl;
    std::cout << "lambda = " << lambda_rev.to_hex() << std::endl;
    
    UInt256 lambda_sq_rev = ModOp::mul(lambda_rev, lambda_rev);
    UInt256 x3_2 = ModOp::sub(lambda_sq_rev, x1_rev);
    x3_2 = ModOp::sub(x3_2, x2_rev);
    
    UInt256 x1_minus_x3_2 = ModOp::sub(x1_rev, x3_2);
    UInt256 y3_2 = ModOp::sub(ModOp::mul(lambda_rev, x1_minus_x3_2), y1_rev);
    
    std::cout << "x3 = " << x3_2.to_hex() << std::endl;
    std::cout << "y3 = " << y3_2.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 比较结果
    std::cout << "=== 结果比较 ===" << std::endl;
    std::cout << "x3_1 == x3_2: " << (x3_1 == x3_2 ? "是" : "否") << std::endl;
    std::cout << "y3_1 == y3_2: " << (y3_1 == y3_2 ? "是" : "否") << std::endl;
    
    // 检查斜率关系
    std::cout << "\n=== 斜率关系检查 ===" << std::endl;
    std::cout << "lambda_1 == lambda_2: " << (lambda == lambda_rev ? "是" : "否") << std::endl;
    
    // 检查dx和dy的关系
    UInt256 neg_dx = ModOp::sub(UInt256(0, 0, 0, 0), dx);
    UInt256 neg_dy = ModOp::sub(UInt256(0, 0, 0, 0), dy);
    
    std::cout << "dx_rev == -dx: " << (dx_rev == neg_dx ? "是" : "否") << std::endl;
    std::cout << "dy_rev == -dy: " << (dy_rev == neg_dy ? "是" : "否") << std::endl;
    
    // 验证模逆运算
    std::cout << "\n=== 模逆验证 ===" << std::endl;
    UInt256 check1 = ModOp::mul(dx, dx_inv);
    UInt256 check2 = ModOp::mul(dx_rev, dx_inv_rev);
    
    std::cout << "dx * dx_inv = " << check1.to_hex() << std::endl;
    std::cout << "dx_rev * dx_inv_rev = " << check2.to_hex() << std::endl;
    std::cout << "check1 == 1: " << (check1 == UInt256(1, 0, 0, 0) ? "是" : "否") << std::endl;
    std::cout << "check2 == 1: " << (check2 == UInt256(1, 0, 0, 0) ? "是" : "否") << std::endl;
}

int main() {
    try {
        debug_detailed_addition();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
