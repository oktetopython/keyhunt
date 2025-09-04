#ifndef LIGHTWEIGHT_ECC_POINT_H
#define LIGHTWEIGHT_ECC_POINT_H

#include "uint256.h"

namespace LightweightECC {

// 椭圆曲线点 - 基于gECC的ECPointJacobian思想
class Point {
public:
    // 构造函数
    Point();
    Point(const UInt256& x, const UInt256& y);
    Point(const UInt256& x, const UInt256& y, bool infinity);

    // 添加拷贝构造函数
    Point(const Point& other);

    // 添加赋值运算符
    Point& operator=(const Point& other);
    
    // 比较运算
    bool operator==(const Point& other) const;
    bool operator!=(const Point& other) const;
    
    // 检查函数
    bool is_infinity() const;
    bool is_on_curve(const UInt256& a, const UInt256& b, const UInt256& p) const;
    
    // 转换函数
    std::string to_string() const;
    
    // 坐标访问
    UInt256 x;
    UInt256 y;
    bool infinity;
    
private:
    // 内部辅助函数
    void normalize();
    void init_infinity();
};

} // namespace LightweightECC

#endif // LIGHTWEIGHT_ECC_POINT_H
