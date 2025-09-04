#include "../include/uint256.h"
#include "../include/secp256k1.h"
#include "../include/mod_op.h"
#include <iostream>

void test_without_modop() {
    std::cout << "=== 测试1：不使用ModOp ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 直接创建p，不通过Secp256k1
    UInt256 p = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    
    std::cout << "p = " << p.to_hex() << std::endl;
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    
    UInt256 result = p_plus_1 % p;
    std::cout << "(p+1) % p = " << result.to_hex() << std::endl;
    std::cout << "结果是否为1: " << (result == UInt256(1, 0, 0, 0) ? "YES" : "NO") << std::endl;
}

void test_with_secp256k1_only() {
    std::cout << "\n=== 测试2：仅使用Secp256k1 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    auto params = Secp256k1::get_params();
    
    UInt256 p_plus_1 = params.p + UInt256(1, 0, 0, 0);
    
    std::cout << "p = " << params.p.to_hex() << std::endl;
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    
    UInt256 result = p_plus_1 % params.p;
    std::cout << "(p+1) % p = " << result.to_hex() << std::endl;
    std::cout << "结果是否为1: " << (result == UInt256(1, 0, 0, 0) ? "YES" : "NO") << std::endl;
}

void test_with_modop_init() {
    std::cout << "\n=== 测试3：使用ModOp::init ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    auto params = Secp256k1::get_params();
    
    // 初始化ModOp
    ModOp::init(params.p);
    
    UInt256 p_plus_1 = params.p + UInt256(1, 0, 0, 0);
    
    std::cout << "p = " << params.p.to_hex() << std::endl;
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    
    UInt256 result = p_plus_1 % params.p;
    std::cout << "(p+1) % p = " << result.to_hex() << std::endl;
    std::cout << "结果是否为1: " << (result == UInt256(1, 0, 0, 0) ? "YES" : "NO") << std::endl;
}

void test_with_modop_operations() {
    std::cout << "\n=== 测试4：使用ModOp运算 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    auto params = Secp256k1::get_params();
    ModOp::init(params.p);
    
    // 执行一些ModOp运算
    UInt256 test_result = ModOp::mul(UInt256(2, 0, 0, 0), UInt256(3, 0, 0, 0));
    std::cout << "ModOp: 2 * 3 = " << test_result.to_hex() << std::endl;
    
    UInt256 p_plus_1 = params.p + UInt256(1, 0, 0, 0);
    
    std::cout << "p = " << params.p.to_hex() << std::endl;
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    
    UInt256 result = p_plus_1 % params.p;
    std::cout << "(p+1) % p = " << result.to_hex() << std::endl;
    std::cout << "结果是否为1: " << (result == UInt256(1, 0, 0, 0) ? "YES" : "NO") << std::endl;
}

void test_exact_verify_secp256k1_sequence() {
    std::cout << "\n=== 测试5：完全模拟verify_secp256k1序列 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 完全按照verify_secp256k1的顺序执行
    Secp256k1::init();
    auto params = Secp256k1::get_params();
    
    std::cout << "secp256k1素数p = " << params.p.to_hex() << std::endl;
    
    // 验证素数正确性
    UInt256 expected_p = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    bool p_correct = (params.p == expected_p);
    std::cout << "素数p正确: " << (p_correct ? "✅ 是" : "❌ 否") << std::endl;
    
    // 测试 p mod p = 0
    UInt256 p_mod_p = params.p % params.p;
    std::cout << "p mod p = " << p_mod_p.to_hex() << std::endl;
    bool test1 = p_mod_p.is_zero();
    std::cout << "p mod p == 0: " << (test1 ? "✅ 通过" : "❌ 失败") << std::endl;
    
    // 测试 (p+1) mod p = 1
    UInt256 p_plus_1 = params.p + UInt256(1, 0, 0, 0);
    UInt256 p_plus_1_mod_p = p_plus_1 % params.p;
    std::cout << "(p+1) mod p = " << p_plus_1_mod_p.to_hex() << std::endl;
    bool test2 = (p_plus_1_mod_p == UInt256(1, 0, 0, 0));
    std::cout << "(p+1) mod p == 1: " << (test2 ? "✅ 通过" : "❌ 失败") << std::endl;
    
    // 如果失败，详细分析
    if (!test2) {
        std::cout << "\n详细分析:" << std::endl;
        std::cout << "p = " << params.p.to_hex() << std::endl;
        std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
        std::cout << "期望余数: " << UInt256(1, 0, 0, 0).to_hex() << std::endl;
        std::cout << "实际余数: " << p_plus_1_mod_p.to_hex() << std::endl;
        
        // 检查内部表示
        std::cout << "实际余数limbs: ";
        for (int i = 0; i < 4; i++) {
            std::cout << "0x" << std::hex << p_plus_1_mod_p.limbs[i] << std::dec << " ";
        }
        std::cout << std::endl;
        
        // 手动验证
        UInt256 manual_quotient = p_plus_1 / params.p;
        UInt256 manual_verification = manual_quotient * params.p + p_plus_1_mod_p;
        std::cout << "手动验证: " << (manual_verification == p_plus_1 ? "PASS" : "FAIL") << std::endl;
    }
}

void test_memory_corruption() {
    std::cout << "\n=== 测试6：检查内存损坏 ===" << std::endl;
    
    using namespace LightweightECC;
    
    // 创建多个UInt256对象，检查是否有内存问题
    UInt256 p1 = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    UInt256 p2 = UInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    UInt256 one1(1, 0, 0, 0);
    UInt256 one2(1, 0, 0, 0);
    
    std::cout << "p1 == p2: " << (p1 == p2 ? "YES" : "NO") << std::endl;
    std::cout << "one1 == one2: " << (one1 == one2 ? "YES" : "NO") << std::endl;
    
    UInt256 p1_plus_1 = p1 + one1;
    UInt256 p2_plus_1 = p2 + one2;
    
    std::cout << "p1+1 == p2+1: " << (p1_plus_1 == p2_plus_1 ? "YES" : "NO") << std::endl;
    
    UInt256 result1 = p1_plus_1 % p1;
    UInt256 result2 = p2_plus_1 % p2;
    
    std::cout << "result1 = " << result1.to_hex() << std::endl;
    std::cout << "result2 = " << result2.to_hex() << std::endl;
    std::cout << "result1 == result2: " << (result1 == result2 ? "YES" : "NO") << std::endl;
    std::cout << "result1 == 1: " << (result1 == one1 ? "YES" : "NO") << std::endl;
    std::cout << "result2 == 1: " << (result2 == one2 ? "YES" : "NO") << std::endl;
}

int main() {
    std::cout << "状态差异调试工具" << std::endl;
    std::cout << "================" << std::endl;
    
    try {
        test_without_modop();
        test_with_secp256k1_only();
        test_with_modop_init();
        test_with_modop_operations();
        test_exact_verify_secp256k1_sequence();
        test_memory_corruption();
        
        std::cout << "\n=== 状态差异调试完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "调试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
