#include "ec_op.h"
#include "mod_op.h"
#include <stdexcept>
#include <algorithm>

namespace LightweightECC {

// 静态成员变量定义
CurveParams ECOp::curve;
bool ECOp::initialized = false;

void ECOp::init(const CurveParams& params) {
    curve = params;
    ModOp::init(params.p);
    initialized = true;
}

bool ECOp::is_initialized() {
    return initialized;
}

const CurveParams& ECOp::get_params() {
    if (!initialized) {
        throw std::runtime_error("ECOp not initialized");
    }
    return curve;
}

Point ECOp::get_generator() {
    if (!initialized) {
        throw std::runtime_error("ECOp not initialized");
    }
    return curve.G;
}

// 椭圆曲线点加法
Point ECOp::point_add(const Point& P, const Point& Q) {
    if (!initialized) {
        throw std::runtime_error("ECOp not initialized");
    }
    
    // 边界条件处理
    if (P.is_infinity()) return Q;
    if (Q.is_infinity()) return P;
    
    // 检查是否为相同点
    if (points_equal(P, Q)) {
        return point_double(P);
    }
    
    // 检查是否为相反点 (x相同，y相反)
    if (P.x == Q.x) {
        UInt256 neg_py = ModOp::sub(curve.p, P.y);
        if (Q.y == neg_py) {
            return Point();  // 返回无穷远点
        }
    }
    
    return point_add_affine(P, Q);
}

// 椭圆曲线点倍法
Point ECOp::point_double(const Point& P) {
    if (!initialized) {
        throw std::runtime_error("ECOp not initialized");
    }
    
    if (P.is_infinity()) {
        return Point();  // 无穷远点的倍点还是无穷远点
    }
    
    // 检查 y = 0 的特殊情况
    if (P.y.is_zero()) {
        return Point();  // y=0 时倍点为无穷远点
    }
    
    return point_double_affine(P);
}

// 修复：标量乘法
Point ECOp::scalar_mul(const UInt256& k, const Point& P) {
    if (!initialized) {
        throw std::runtime_error("ECOp not initialized");
    }

    // 边界条件
    if (k.is_zero() || P.is_infinity()) {
        return Point();
    }

    // 修复：使用简单但正确的二进制方法
    return scalar_mul_binary(k, P);
}

// 修复：对称的仿射坐标点加法实现
Point ECOp::point_add_affine(const Point& P, const Point& Q) {
    // 修复：确保函数对P和Q完全对称处理

    // 修复：检查边界条件（虽然调用者已检查，但为了安全）
    if (P.is_infinity()) return Q;
    if (Q.is_infinity()) return P;

    // 修复：检查是否为相同点（调用者已处理，但为了安全）
    if (P.x == Q.x) {
        if (P.y == Q.y) {
            return point_double_affine(P);
        } else {
            return Point();  // 相反点，返回无穷远点
        }
    }

    // 修复：对称计算斜率 λ = (y2 - y1) / (x2 - x1)
    UInt256 x1 = P.x;
    UInt256 y1 = P.y;
    UInt256 x2 = Q.x;
    UInt256 y2 = Q.y;

    // 修复：计算 dx = x2 - x1
    UInt256 dx = ModOp::sub(x2, x1);

    // 修复：计算 dy = y2 - y1
    UInt256 dy = ModOp::sub(y2, y1);

    // 修复：计算斜率 λ = dy / dx
    UInt256 dx_inv = ModOp::inv(dx);
    UInt256 lambda = ModOp::mul(dy, dx_inv);

    // 修复：计算 x3 = λ² - x1 - x2
    UInt256 lambda_sq = ModOp::mul(lambda, lambda);
    UInt256 x3 = ModOp::sub(lambda_sq, x1);
    x3 = ModOp::sub(x3, x2);

    // 修复：计算 y3 = λ(x1 - x3) - y1
    UInt256 x1_minus_x3 = ModOp::sub(x1, x3);
    UInt256 lambda_times_diff = ModOp::mul(lambda, x1_minus_x3);
    UInt256 y3 = ModOp::sub(lambda_times_diff, y1);

    return Point(x3, y3);
}

// 修复：仿射坐标点倍法实现
Point ECOp::point_double_affine(const Point& P) {
    // 修复：确保P不是无穷远点且y≠0（调用者已检查）

    // 修复：计算斜率 λ = (3x^2 + a) / (2y)
    UInt256 x_squared = ModOp::mul(P.x, P.x);
    UInt256 three_x_squared = ModOp::add(x_squared, x_squared);
    three_x_squared = ModOp::add(three_x_squared, x_squared);
    UInt256 numerator = ModOp::add(three_x_squared, curve.a);

    UInt256 two_y = ModOp::add(P.y, P.y);
    UInt256 two_y_inv = ModOp::inv(two_y);
    UInt256 lambda = ModOp::mul(numerator, two_y_inv);

    // 修复：计算x3 = λ^2 - 2x
    UInt256 lambda_sq = ModOp::mul(lambda, lambda);
    UInt256 two_x = ModOp::add(P.x, P.x);
    UInt256 x3 = ModOp::sub(lambda_sq, two_x);

    // 修复：计算y3 = λ(x1 - x3) - y1
    UInt256 x1_minus_x3 = ModOp::sub(P.x, x3);
    UInt256 lambda_times_diff = ModOp::mul(lambda, x1_minus_x3);
    UInt256 y3 = ModOp::sub(lambda_times_diff, P.y);

    return Point(x3, y3);
}

// 修复：实现正确的二进制标量乘法
Point ECOp::scalar_mul_binary(const UInt256& k, const Point& P) {
    Point result = Point();  // 无穷远点
    Point current = P;

    // 修复：从最低位开始处理
    for (int i = 0; i < 256; i++) {
        if (k.get_bit(i)) {
            result = point_add(result, current);
        }
        current = point_double(current);
    }

    return result;
}

// 窗口方法标量乘法
Point ECOp::scalar_mul_window(const UInt256& k, const Point& P, int window_size) {
    if (window_size <= 0 || window_size > 8) {
        window_size = 4;  // 默认窗口大小
    }
    
    // 预计算窗口表
    int table_size = 1 << window_size;
    std::vector<Point> window_table(table_size);
    window_table[0] = Point();  // 无穷远点
    window_table[1] = P;
    
    for (int i = 2; i < table_size; i++) {
        window_table[i] = point_add(window_table[i-1], P);
    }
    
    // 主循环
    Point result = Point();
    bool first = true;
    
    for (int i = 255; i >= 0; i -= window_size) {
        if (!first) {
            for (int j = 0; j < window_size; j++) {
                result = point_double(result);
            }
        } else {
            first = false;
        }
        
        int window_value = 0;
        for (int j = 0; j < window_size && (i - j) >= 0; j++) {
            if (k.get_bit(i - j)) {
                window_value |= (1 << (window_size - 1 - j));
            }
        }
        
        if (window_value != 0) {
            result = point_add(result, window_table[window_value]);
        }
    }
    
    return result;
}

// 批量标量乘法
std::vector<Point> ECOp::batch_scalar_mul(const std::vector<UInt256>& scalars, const Point& P) {
    std::vector<Point> results;
    results.reserve(scalars.size());
    
    for (const auto& scalar : scalars) {
        results.push_back(scalar_mul(scalar, P));
    }
    
    return results;
}

// 验证点是否在曲线上
bool ECOp::is_on_curve(const Point& P) {
    if (!initialized) {
        return false;
    }
    
    if (P.is_infinity()) {
        return true;
    }
    
    // 检查 y^2 ≡ x^3 + ax + b (mod p)
    UInt256 y_squared = ModOp::mul(P.y, P.y);
    UInt256 x_squared = ModOp::mul(P.x, P.x);
    UInt256 x_cubed = ModOp::mul(x_squared, P.x);
    UInt256 ax = ModOp::mul(curve.a, P.x);
    UInt256 right_side = ModOp::add(ModOp::add(x_cubed, ax), curve.b);
    
    return y_squared == right_side;
}

// 验证点是否有效
bool ECOp::is_valid_point(const Point& P) {
    return is_on_curve(P);
}

// 辅助函数：检查两点是否相等
bool ECOp::points_equal(const Point& P, const Point& Q) {
    if (P.is_infinity() && Q.is_infinity()) {
        return true;
    }
    if (P.is_infinity() || Q.is_infinity()) {
        return false;
    }
    return (P.x == Q.x) && (P.y == Q.y);
}

// 辅助函数：点的负值
Point ECOp::point_negate(const Point& P) {
    if (P.is_infinity()) {
        return Point();
    }
    
    UInt256 neg_y = ModOp::sub(curve.p, P.y);
    return Point(P.x, neg_y);
}

} // namespace LightweightECC
