#include "../include/uint256.h"
#include "../include/point.h"
#include "../include/curve_params.h"
#include "../include/mod_op.h"
#include "../include/ec_op.h"
#include "../include/secp256k1.h"
#include <iostream>
#include <cassert>

void test_commutativity() {
    std::cout << "=== 交换律专项测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    Point G = Secp256k1::get_generator();
    
    // 计算 G2 = 2G
    UInt256 k2(2, 0, 0, 0);
    Point G2 = Secp256k1::scalar_mul(k2);
    
    std::cout << "G = (" << G.x.to_hex().substr(48, 16) << "...)" << std::endl;
    std::cout << "G2 = (" << G2.x.to_hex().substr(48, 16) << "...)" << std::endl;
    
    // 测试交换律：G + G2 == G2 + G
    Point G_plus_G2 = ECOp::point_add(G, G2);
    Point G2_plus_G = ECOp::point_add(G2, G);
    
    std::cout << "G + G2 = (" << G_plus_G2.x.to_hex().substr(48, 16) << "...)" << std::endl;
    std::cout << "G2 + G = (" << G2_plus_G.x.to_hex().substr(48, 16) << "...)" << std::endl;
    
    bool commutative = (G_plus_G2 == G2_plus_G);
    std::cout << "G + G2 == G2 + G: " << (commutative ? "✅ 通过" : "❌ 失败") << std::endl;
    
    if (!commutative) {
        std::cout << "详细对比：" << std::endl;
        std::cout << "G + G2.x = " << G_plus_G2.x.to_hex() << std::endl;
        std::cout << "G2 + G.x = " << G2_plus_G.x.to_hex() << std::endl;
        std::cout << "G + G2.y = " << G_plus_G2.y.to_hex() << std::endl;
        std::cout << "G2 + G.y = " << G2_plus_G.y.to_hex() << std::endl;
    }
    
    assert(commutative);
    
    // 验证结果应该是 3G
    UInt256 k3(3, 0, 0, 0);
    Point three_G = Secp256k1::scalar_mul(k3);
    
    bool correct_result = (G_plus_G2 == three_G);
    std::cout << "G + G2 == 3G: " << (correct_result ? "✅ 通过" : "❌ 失败") << std::endl;
    assert(correct_result);
    
    std::cout << "✅ 交换律测试通过" << std::endl;
}

void test_associativity() {
    std::cout << "\n=== 结合律专项测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    Point G = Secp256k1::get_generator();
    
    // 计算 G2 = 2G, G3 = 3G
    UInt256 k2(2, 0, 0, 0);
    UInt256 k3(3, 0, 0, 0);
    Point G2 = Secp256k1::scalar_mul(k2);
    Point G3 = Secp256k1::scalar_mul(k3);
    
    // 测试结合律：(G + G) + G = G + (G + G)
    Point G_plus_G = ECOp::point_add(G, G);
    Point G_plus_G_plus_G = ECOp::point_add(G_plus_G, G);
    
    Point G_plus_G2 = ECOp::point_add(G, G2);
    
    bool associative = (G_plus_G_plus_G == G_plus_G2);
    std::cout << "(G + G) + G == G + (G + G): " << (associative ? "✅ 通过" : "❌ 失败") << std::endl;
    assert(associative);
    
    // 验证结果应该是 3G
    bool correct_result = (G_plus_G_plus_G == G3);
    std::cout << "(G + G) + G == 3G: " << (correct_result ? "✅ 通过" : "❌ 失败") << std::endl;
    assert(correct_result);
    
    std::cout << "✅ 结合律测试通过" << std::endl;
}

void test_detailed_commutativity() {
    std::cout << "\n=== 详细交换律测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    Point G = Secp256k1::get_generator();
    
    // 测试多个点对的交换律
    for (int i = 1; i <= 5; i++) {
        UInt256 ki(i, 0, 0, 0);
        Point Gi = Secp256k1::scalar_mul(ki);
        
        UInt256 kj(i + 1, 0, 0, 0);
        Point Gj = Secp256k1::scalar_mul(kj);
        
        Point Gi_plus_Gj = ECOp::point_add(Gi, Gj);
        Point Gj_plus_Gi = ECOp::point_add(Gj, Gi);
        
        bool commutative = (Gi_plus_Gj == Gj_plus_Gi);
        std::cout << i << "G + " << (i+1) << "G == " << (i+1) << "G + " << i << "G: " 
                  << (commutative ? "✅ 通过" : "❌ 失败") << std::endl;
        
        if (!commutative) {
            std::cout << "  " << i << "G + " << (i+1) << "G.x = " << Gi_plus_Gj.x.to_hex() << std::endl;
            std::cout << "  " << (i+1) << "G + " << i << "G.x = " << Gj_plus_Gi.x.to_hex() << std::endl;
        }
        
        assert(commutative);
    }
    
    std::cout << "✅ 详细交换律测试通过" << std::endl;
}

int main() {
    std::cout << "轻量级ECC库 - 交换律专项测试" << std::endl;
    std::cout << "=============================" << std::endl;
    
    try {
        test_commutativity();
        test_associativity();
        test_detailed_commutativity();
        
        std::cout << "\n🎉 所有交换律测试通过！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 交换律测试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
