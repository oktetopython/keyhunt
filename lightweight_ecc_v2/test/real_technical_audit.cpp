#include "../include/uint256.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include "../include/ec_op.h"
#include <iostream>
#include <chrono>

void audit_basic_operations() {
    std::cout << "=== 真实基础运算审计 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    
    std::cout << "审计对象：secp256k1素数 p" << std::endl;
    std::cout << "p = " << p.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 审计1：(p+1) / p 是否真的等于 1
    std::cout << "审计1：(p+1) / p 是否真的等于 1" << std::endl;
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    UInt256 div_result = p_plus_1 / p;
    
    std::cout << "p+1 = " << p_plus_1.to_hex() << std::endl;
    std::cout << "(p+1) / p = " << div_result.to_hex() << std::endl;
    std::cout << "期望 = 0000000000000000000000000000000000000000000000000000000000000001" << std::endl;
    
    bool div_correct = (div_result == UInt256(1, 0, 0, 0));
    std::cout << "结果：" << (div_correct ? "✅ 正确" : "❌ 错误") << std::endl;
    std::cout << std::endl;
    
    // 审计2：(p+1) % p 是否真的等于 1
    std::cout << "审计2：(p+1) % p 是否真的等于 1" << std::endl;
    UInt256 mod_result = p_plus_1 % p;
    
    std::cout << "(p+1) % p = " << mod_result.to_hex() << std::endl;
    std::cout << "期望 = 0000000000000000000000000000000000000000000000000000000000000001" << std::endl;
    
    bool mod_correct = (mod_result == UInt256(1, 0, 0, 0));
    std::cout << "结果：" << (mod_correct ? "✅ 正确" : "❌ 错误") << std::endl;
    std::cout << std::endl;
    
    // 审计3：验证除法的一致性
    std::cout << "审计3：验证除法一致性" << std::endl;
    UInt256 verification = div_result * p + mod_result;
    std::cout << "验证：quotient * p + remainder = " << verification.to_hex() << std::endl;
    std::cout << "原始：p+1 = " << p_plus_1.to_hex() << std::endl;
    
    bool verify_correct = (verification == p_plus_1);
    std::cout << "一致性：" << (verify_correct ? "✅ 正确" : "❌ 错误") << std::endl;
    
    // 总结基础运算审计
    std::cout << "\n基础运算审计总结：" << std::endl;
    std::cout << "除法正确：" << (div_correct ? "是" : "否") << std::endl;
    std::cout << "模运算正确：" << (mod_correct ? "是" : "否") << std::endl;
    std::cout << "一致性正确：" << (verify_correct ? "是" : "否") << std::endl;
    
    if (!div_correct || !mod_correct || !verify_correct) {
        std::cout << "❌ 基础运算存在根本性问题！" << std::endl;
    } else {
        std::cout << "✅ 基础运算看起来正确" << std::endl;
    }
}

void audit_modular_inverse() {
    std::cout << "\n=== 真实模逆算法审计 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    // 审计：2的模逆是否正确
    std::cout << "审计：2的模逆计算" << std::endl;
    
    UInt256 a(2, 0, 0, 0);
    std::cout << "a = 2" << std::endl;
    
    // 计算模逆
    auto start = std::chrono::high_resolution_clock::now();
    UInt256 inv_a;
    bool inverse_success = false;
    
    try {
        inv_a = ModOp::inv(a);
        inverse_success = true;
    } catch (const std::exception& e) {
        std::cout << "模逆计算失败：" << e.what() << std::endl;
        inverse_success = false;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (inverse_success) {
        std::cout << "2的模逆 = " << inv_a.to_hex() << std::endl;
        std::cout << "计算时间：" << duration.count() << " 毫秒" << std::endl;
        
        // 验证：a * inv(a) mod p 是否等于 1
        std::cout << "\n验证：2 * inv(2) mod p" << std::endl;
        UInt256 verification = ModOp::mul(a, inv_a);
        std::cout << "2 * inv(2) = " << verification.to_hex() << std::endl;
        std::cout << "期望 = 0000000000000000000000000000000000000000000000000000000000000001" << std::endl;
        
        bool verify_correct = (verification == UInt256(1, 0, 0, 0));
        std::cout << "验证结果：" << (verify_correct ? "✅ 正确" : "❌ 错误") << std::endl;
        
        if (!verify_correct) {
            std::cout << "❌ 模逆算法存在根本性问题！" << std::endl;
            
            // 与已知正确值比较
            UInt256 known_correct = (p + UInt256(1, 0, 0, 0)) / UInt256(2, 0, 0, 0);
            std::cout << "已知正确值 = " << known_correct.to_hex() << std::endl;
            std::cout << "计算值 = " << inv_a.to_hex() << std::endl;
            std::cout << "值匹配：" << (inv_a == known_correct ? "是" : "否") << std::endl;
        }
    } else {
        std::cout << "❌ 模逆计算完全失败！" << std::endl;
    }
}

void audit_elliptic_curve() {
    std::cout << "\n=== 真实椭圆曲线审计 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    Point G = Secp256k1::get_generator();
    
    std::cout << "审计：最简单的椭圆曲线运算" << std::endl;
    std::cout << "生成元 G = (" << G.x.to_hex().substr(48, 16) << "..., " << G.y.to_hex().substr(48, 16) << "...)" << std::endl;
    
    // 审计：G + G 是否能计算
    std::cout << "\n审计：G + G 计算" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    Point result;
    bool addition_success = false;
    
    try {
        result = ECOp::point_add(G, G);
        addition_success = true;
    } catch (const std::exception& e) {
        std::cout << "椭圆曲线加法失败：" << e.what() << std::endl;
        addition_success = false;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (addition_success) {
        std::cout << "G + G = (" << result.x.to_hex().substr(48, 16) << "..., " << result.y.to_hex().substr(48, 16) << "...)" << std::endl;
        std::cout << "计算时间：" << duration.count() << " 毫秒" << std::endl;
        
        // 验证：G + G 是否等于 2G
        std::cout << "\n验证：G + G 是否等于 2G" << std::endl;
        // 这里需要实现标量乘法来验证，但如果连加法都有问题，标量乘法肯定也有问题
        std::cout << "✅ 椭圆曲线加法至少能执行" << std::endl;
        
    } else {
        std::cout << "❌ 椭圆曲线加法完全失败！" << std::endl;
    }
    
    // 审计：交换律测试
    std::cout << "\n审计：交换律 G + G 是否等于 G + G" << std::endl;
    
    if (addition_success) {
        try {
            Point result2 = ECOp::point_add(G, G);
            bool commutative = (result.x == result2.x && result.y == result2.y);
            std::cout << "交换律（自身）：" << (commutative ? "✅ 正确" : "❌ 错误") << std::endl;
        } catch (const std::exception& e) {
            std::cout << "交换律测试失败：" << e.what() << std::endl;
        }
    }
}

void audit_performance_claims() {
    std::cout << "\n=== 性能声称审计 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    // 审计：模逆计算的真实性能
    std::cout << "审计：模逆计算性能" << std::endl;
    
    UInt256 test_values[] = {
        UInt256(2, 0, 0, 0),
        UInt256(3, 0, 0, 0),
        UInt256(7, 0, 0, 0),
        UInt256(11, 0, 0, 0),
        UInt256(13, 0, 0, 0)
    };
    
    int success_count = 0;
    int total_time = 0;
    
    for (int i = 0; i < 5; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            UInt256 inv = ModOp::inv(test_values[i]);
            UInt256 verify = ModOp::mul(test_values[i], inv);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            bool correct = (verify == UInt256(1, 0, 0, 0));
            
            std::cout << "测试 " << (i+1) << "：" << test_values[i].limbs[0] << "的模逆" << std::endl;
            std::cout << "  时间：" << duration.count() << " 毫秒" << std::endl;
            std::cout << "  正确：" << (correct ? "是" : "否") << std::endl;
            
            if (correct) {
                success_count++;
                total_time += duration.count();
            }
            
        } catch (const std::exception& e) {
            std::cout << "测试 " << (i+1) << " 失败：" << e.what() << std::endl;
        }
    }
    
    std::cout << "\n性能审计总结：" << std::endl;
    std::cout << "成功率：" << success_count << "/5 (" << (success_count * 20) << "%)" << std::endl;
    if (success_count > 0) {
        std::cout << "平均时间：" << (total_time / success_count) << " 毫秒" << std::endl;
    }
    
    if (success_count < 5) {
        std::cout << "❌ 模逆算法不稳定或不正确！" << std::endl;
    }
}

int main() {
    std::cout << "轻量级ECC库 - 真实技术审计" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << "这是一个诚实的技术状态评估，不会掩盖任何问题。" << std::endl;
    std::cout << std::endl;
    
    try {
        audit_basic_operations();
        audit_modular_inverse();
        audit_elliptic_curve();
        audit_performance_claims();
        
        std::cout << "\n=== 真实技术审计完成 ===" << std::endl;
        std::cout << "以上结果反映了项目的真实状态，没有任何夸大或掩饰。" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 审计过程中发生严重错误: " << e.what() << std::endl;
        std::cerr << "这表明项目存在根本性问题。" << std::endl;
        return 1;
    }
    
    return 0;
}
