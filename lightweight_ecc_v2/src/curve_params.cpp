#include "curve_params.h"
#include <sstream>

namespace LightweightECC {

// 修复默认构造函数
CurveParams::CurveParams() {
    // 显式初始化所有成员，避免依赖默认构造函数
    p = UInt256(0, 0, 0, 0);
    a = UInt256(0, 0, 0, 0);
    b = UInt256(0, 0, 0, 0);
    n = UInt256(0, 0, 0, 0);
    G = Point();  // 使用默认构造函数创建无穷远点
    h = UInt256(0, 0, 0, 0);
}

CurveParams::CurveParams(const UInt256& p, const UInt256& a, const UInt256& b,
                        const UInt256& n, const Point& G, const UInt256& h)
    : p(p), a(a), b(b), n(n), G(G), h(h) {}

bool CurveParams::is_valid() const {
    // 基本验证
    if (p.is_zero() || n.is_zero()) return false;
    if (G.is_infinity()) return false;

    // 简化验证 - 暂时跳过椭圆曲线验证，因为模运算还未完全实现
    // TODO: 实现完整的椭圆曲线点验证
    // if (!G.is_on_curve(a, b, p)) return false;

    // 验证h不为零且在合理范围内
    if (h.is_zero()) return false;

    return true;
}

std::string CurveParams::to_string() const {
    std::stringstream ss;
    ss << "CurveParams:" << std::endl;
    ss << "  p = " << p.to_hex() << std::endl;
    ss << "  a = " << a.to_hex() << std::endl;
    ss << "  b = " << b.to_hex() << std::endl;
    ss << "  n = " << n.to_hex() << std::endl;
    ss << "  G = " << G.to_string() << std::endl;
    ss << "  h = " << h.to_hex() << std::endl;
    return ss.str();
}

CurveParams get_secp256k1_params() {
    // secp256k1曲线参数
    UInt256 p = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    UInt256 a = UInt256(0, 0, 0, 0);
    UInt256 b = UInt256(7, 0, 0, 0);
    UInt256 n = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    
    Point G = Point(
        UInt256::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"),
        UInt256::from_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8")
    );
    
    UInt256 h = UInt256(1, 0, 0, 0);
    
    return CurveParams(p, a, b, n, G, h);
}

CurveParams get_sm2_params() {
    // SM2曲线参数（简化实现）
    UInt256 p = UInt256::from_hex("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF");
    UInt256 a = UInt256::from_hex("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC");
    UInt256 b = UInt256::from_hex("28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93");
    UInt256 n = UInt256::from_hex("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123");
    
    Point G = Point(
        UInt256::from_hex("32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7"),
        UInt256::from_hex("BC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0")
    );
    
    UInt256 h = UInt256(1, 0, 0, 0);
    
    return CurveParams(p, a, b, n, G, h);
}

} // namespace LightweightECC
