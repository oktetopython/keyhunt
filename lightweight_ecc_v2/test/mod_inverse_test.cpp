#include "../include/uint256.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <vector>

void test_mod_inverse_correctness() {
    std::cout << "=== 模逆运算专项测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    std::cout << "secp256k1素数p = " << p.to_hex().substr(0, 32) << "..." << std::endl;
    std::cout << std::endl;
    
    // 测试多个值的模逆
    std::vector<UInt256> test_values = {
        UInt256(2, 0, 0, 0),
        UInt256(3, 0, 0, 0),
        UInt256(7, 0, 0, 0),
        UInt256(12345, 0, 0, 0),
        UInt256::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
    };
    
    int passed = 0;
    int total = test_values.size();
    
    for (size_t i = 0; i < test_values.size(); i++) {
        const auto& a = test_values[i];
        
        std::cout << "测试 " << (i+1) << "/" << total << ":" << std::endl;
        std::cout << "a = " << a.to_hex().substr(48, 16) << " (低64位)" << std::endl;
        
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            UInt256 a_inv = ModOp::inv(a);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            std::cout << "a^-1 = " << a_inv.to_hex().substr(48, 16) << " (低64位)" << std::endl;
            std::cout << "计算时间: " << duration.count() << " 微秒" << std::endl;
            
            UInt256 check = ModOp::mul(a, a_inv);
            std::cout << "a * a^-1 = " << check.to_hex().substr(48, 16) << " (低64位)" << std::endl;
            
            bool correct = (check == UInt256(1, 0, 0, 0));
            std::cout << "验证: " << (correct ? "✅ 通过" : "❌ 失败") << std::endl;
            
            if (correct) {
                passed++;
            } else {
                std::cout << "期望: 0000000000000001" << std::endl;
                std::cout << "实际: " << check.to_hex() << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cout << "❌ 异常: " << e.what() << std::endl;
        }
        
        std::cout << "---" << std::endl;
    }
    
    std::cout << "模逆运算测试结果: " << passed << "/" << total << " 通过" << std::endl;
    
    if (passed == total) {
        std::cout << "✅ 模逆运算专项测试完全通过！" << std::endl;
    } else {
        std::cout << "❌ 模逆运算测试有失败项" << std::endl;
    }
}

void test_mod_inverse_performance() {
    std::cout << "\n=== 模逆运算性能测试 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    ModOp::init(Secp256k1::get_params().p);
    
    const int test_count = 100;
    std::vector<UInt256> test_values;
    
    // 生成测试数据
    for (int i = 0; i < test_count; i++) {
        test_values.push_back(UInt256(i + 2, 0, 0, 0));  // 避免0和1
    }
    
    std::cout << "测试数量: " << test_count << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int successful = 0;
    for (const auto& a : test_values) {
        try {
            UInt256 a_inv = ModOp::inv(a);
            UInt256 check = ModOp::mul(a, a_inv);
            if (check == UInt256(1, 0, 0, 0)) {
                successful++;
            }
        } catch (const std::exception& e) {
            // 忽略异常，继续测试
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    double ops_per_second = (double)successful / (duration.count() / 1000000.0);
    
    std::cout << "成功计算: " << successful << "/" << test_count << std::endl;
    std::cout << "总耗时: " << duration.count() << " 微秒" << std::endl;
    std::cout << "平均每次: " << (duration.count() / test_count) << " 微秒" << std::endl;
    std::cout << "性能: " << ops_per_second << " ops/second" << std::endl;
    
    // 性能目标：> 10,000 ops/sec
    if (ops_per_second > 10000) {
        std::cout << "✅ 性能测试通过（目标: > 10,000 ops/sec）" << std::endl;
    } else {
        std::cout << "⚠️ 性能低于目标（目标: > 10,000 ops/sec）" << std::endl;
    }
}

int main() {
    std::cout << "轻量级ECC库 - 模逆运算专项测试" << std::endl;
    std::cout << "================================" << std::endl;
    
    try {
        test_mod_inverse_correctness();
        test_mod_inverse_performance();
        
        std::cout << "\n🎉 模逆运算专项测试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
