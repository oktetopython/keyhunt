#include "point.h"
#include <sstream>

namespace LightweightECC {

// 修复默认构造函数
Point::Point() {
    init_infinity();
}

// 添加拷贝构造函数
Point::Point(const Point& other)
    : x(other.x), y(other.y), infinity(other.infinity) {
}

// 添加赋值运算符
Point& Point::operator=(const Point& other) {
    if (this != &other) {
        x = other.x;
        y = other.y;
        infinity = other.infinity;
    }
    return *this;
}

Point::Point(const UInt256& x, const UInt256& y) : x(x), y(y), infinity(false) {}

Point::Point(const UInt256& x, const UInt256& y, bool infinity)
    : x(x), y(y), infinity(infinity) {}

bool Point::operator==(const Point& other) const {
    if (infinity != other.infinity) return false;
    if (infinity) return true;
    return x == other.x && y == other.y;
}

bool Point::operator!=(const Point& other) const {
    return !(*this == other);
}

bool Point::is_infinity() const {
    return infinity;
}

bool Point::is_on_curve(const UInt256& a, const UInt256& b, const UInt256& p) const {
    if (infinity) return true;

    // 修复：正确计算 y^2 mod p 和 x^3 + ax + b mod p
    // 注意：这里需要使用模运算，但Point类不应该依赖ModOp
    // 暂时使用基本模运算
    UInt256 y_squared = (y * y) % p;

    UInt256 x_squared = (x * x) % p;
    UInt256 x_cubed = (x_squared * x) % p;
    UInt256 ax = (a * x) % p;
    UInt256 right_side = (x_cubed + ax + b) % p;

    return y_squared == right_side;
}

std::string Point::to_string() const {
    if (infinity) {
        return "Point(INFINITY)";
    }
    
    std::stringstream ss;
    ss << "Point(" << x.to_hex() << ", " << y.to_hex() << ")";
    return ss.str();
}

void Point::normalize() {
    // 点不需要标准化
}

// 添加内部初始化函数
void Point::init_infinity() {
    // 直接初始化，避免依赖UInt256::zero
    x = UInt256(0, 0, 0, 0);
    y = UInt256(0, 0, 0, 0);
    infinity = true;
}

} // namespace LightweightECC
