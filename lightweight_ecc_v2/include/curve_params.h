#ifndef LIGHTWEIGHT_ECC_CURVE_PARAMS_H
#define LIGHTWEIGHT_ECC_CURVE_PARAMS_H

#include "uint256.h"
#include "point.h"

namespace LightweightECC {

// 椭圆曲线参数 - 基于gECC的参数设计
struct CurveParams {
    UInt256 p;      // 素数p
    UInt256 a;      // 曲线参数a
    UInt256 b;      // 曲线参数b
    UInt256 n;      // 阶n
    Point G;        // 生成元G
    UInt256 h;      // 余因子h
    
    // 构造函数
    CurveParams();
    CurveParams(const UInt256& p, const UInt256& a, const UInt256& b,
               const UInt256& n, const Point& G, const UInt256& h);
    
    // 验证参数
    bool is_valid() const;
    
    // 字符串表示
    std::string to_string() const;
};

// secp256k1曲线参数
CurveParams get_secp256k1_params();

// SM2曲线参数（如果需要）
CurveParams get_sm2_params();

} // namespace LightweightECC

#endif // LIGHTWEIGHT_ECC_CURVE_PARAMS_H
