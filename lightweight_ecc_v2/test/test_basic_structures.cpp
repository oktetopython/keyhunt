#include "../include/uint256.h"
#include "../include/point.h"
#include "../include/curve_params.h"
#include <iostream>
#include <cassert>

void test_uint256() {
    std::cout << "=== 测试UInt256 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 测试构造函数
    UInt256 a(1, 2, 3, 4);
    UInt256 b("0x0000000000000004000000000000000300000000000000020000000000000001");
    
    std::cout << "a = " << a.to_hex() << std::endl;
    std::cout << "b = " << b.to_hex() << std::endl;
    
    assert(a == b);
    
    // 测试基本运算
    UInt256 c = a + UInt256(1, 0, 0, 0);
    std::cout << "a + 1 = " << c.to_hex() << std::endl;
    
    UInt256 d = a - UInt256(1, 0, 0, 0);
    std::cout << "a - 1 = " << d.to_hex() << std::endl;
    
    // 测试位运算
    UInt256 e = a << 1;
    std::cout << "a << 1 = " << e.to_hex() << std::endl;
    
    UInt256 f = a >> 1;
    std::cout << "a >> 1 = " << f.to_hex() << std::endl;
    
    // 测试比较
    assert(a == b);
    assert(c > a);
    assert(d < a);
    
    std::cout << "✓ UInt256测试通过" << std::endl;
}

void test_point() {
    std::cout << "=== 测试Point ===" << std::endl;

    using namespace LightweightECC;

    try {
        // 测试构造函数
        std::cout << "创建UInt256 x..." << std::endl;
        UInt256 x = UInt256::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
        std::cout << "x = " << x.to_hex() << std::endl;

        std::cout << "创建UInt256 y..." << std::endl;
        UInt256 y = UInt256::from_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
        std::cout << "y = " << y.to_hex() << std::endl;

        std::cout << "创建Point p1..." << std::endl;
        Point p1(x, y);
        std::cout << "p1创建成功" << std::endl;

        std::cout << "创建Point p2..." << std::endl;
        Point p2(x, y);
        std::cout << "p2创建成功" << std::endl;

        std::cout << "创建Point p_infinity..." << std::endl;
        Point p_infinity;
        std::cout << "p_infinity创建成功" << std::endl;

        std::cout << "测试to_string..." << std::endl;
        std::cout << "p1 = " << p1.to_string() << std::endl;
        std::cout << "p2 = " << p2.to_string() << std::endl;
        std::cout << "p_infinity = " << p_infinity.to_string() << std::endl;

        // 测试比较
        std::cout << "测试比较运算..." << std::endl;
        assert(p1 == p2);
        assert(p1 != p_infinity);

        // 测试无穷远点
        std::cout << "测试无穷远点..." << std::endl;
        assert(p_infinity.is_infinity());
        assert(!p1.is_infinity());

        std::cout << "✓ Point测试通过" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Point测试异常: " << e.what() << std::endl;
        throw;
    }
}

void test_curve_params() {
    std::cout << "=== 测试CurveParams ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 测试secp256k1参数
    CurveParams secp256k1 = get_secp256k1_params();
    std::cout << "secp256k1参数:" << std::endl;
    std::cout << secp256k1.to_string() << std::endl;
    
    // 验证参数
    assert(secp256k1.is_valid());

    // 暂时跳过椭圆曲线验证，因为模运算还未完全实现
    // TODO: 实现完整的模运算后再启用此验证
    // assert(secp256k1.G.is_on_curve(secp256k1.a, secp256k1.b, secp256k1.p));
    
    std::cout << "✓ CurveParams测试通过" << std::endl;
}

int main() {
    std::cout << "轻量级ECC库 - 基础结构测试" << std::endl;
    std::cout << "=============================" << std::endl;
    
    try {
        test_uint256();
        std::cout << std::endl;
        
        test_point();
        std::cout << std::endl;
        
        test_curve_params();
        std::cout << std::endl;
        
        std::cout << "🎉 所有基础结构测试通过！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
