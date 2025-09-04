#include "../include/uint256.h"
#include "../include/mod_op.h"
#include "../include/secp256k1.h"
#include <iostream>

void final_mod_inverse_analysis() {
    std::cout << "=== 最终模逆分析和修复 ===" << std::endl;
    
    using namespace LightweightECC;
    
    Secp256k1::init();
    UInt256 p = Secp256k1::get_params().p;
    ModOp::init(p);
    
    std::cout << "secp256k1素数p = " << p.to_hex() << std::endl;
    std::cout << std::endl;
    
    // 重新分析：为什么(p+1)/2不是2的模逆？
    std::cout << "=== 理论分析 ===" << std::endl;
    
    UInt256 two(2, 0, 0, 0);
    UInt256 p_plus_1 = p + UInt256(1, 0, 0, 0);
    UInt256 candidate_inv = p_plus_1 / UInt256(2, 0, 0, 0);
    
    std::cout << "候选模逆 = (p+1)/2 = " << candidate_inv.to_hex() << std::endl;
    
    // 验证：2 * candidate_inv mod p
    UInt256 verification = ModOp::mul(two, candidate_inv);
    std::cout << "2 * candidate_inv mod p = " << verification.to_hex() << std::endl;
    
    // 分析差异
    UInt256 expected(1, 0, 0, 0);
    if (verification != expected) {
        std::cout << "差异分析：" << std::endl;
        std::cout << "期望: " << expected.to_hex() << std::endl;
        std::cout << "实际: " << verification.to_hex() << std::endl;
        
        // 计算差值
        if (verification > expected) {
            UInt256 diff = verification - expected;
            std::cout << "差值: " << diff.to_hex() << std::endl;
            
            // 检查差值是否等于p
            if (diff == p) {
                std::cout << "发现：差值等于p！这说明我们的模运算有问题。" << std::endl;
            } else {
                std::cout << "差值不等于p，问题可能在别处。" << std::endl;
            }
        }
    }
    
    std::cout << std::endl;
    
    // 尝试暴力搜索找到正确的模逆
    std::cout << "=== 暴力搜索2的正确模逆 ===" << std::endl;
    
    bool found = false;
    UInt256 correct_inv;
    
    // 搜索范围：从1到1000000
    for (uint64_t i = 1; i <= 1000000 && !found; i++) {
        UInt256 candidate(i, 0, 0, 0);
        UInt256 product = ModOp::mul(two, candidate);
        
        if (product == UInt256(1, 0, 0, 0)) {
            correct_inv = candidate;
            found = true;
            std::cout << "找到2的正确模逆: " << i << std::endl;
            std::cout << "验证：2 * " << i << " mod p = " << product.to_hex() << std::endl;
        }
        
        // 每10万次输出进度
        if (i % 100000 == 0) {
            std::cout << "搜索进度: " << i << " / 1000000" << std::endl;
        }
    }
    
    if (!found) {
        std::cout << "在前100万个数中未找到2的模逆。" << std::endl;
        std::cout << "这可能说明我们的模乘法实现仍有问题。" << std::endl;
        
        // 尝试更大范围的搜索
        std::cout << "\n=== 扩大搜索范围 ===" << std::endl;
        
        // 尝试一些特殊值
        std::vector<UInt256> special_candidates = {
            candidate_inv,  // (p+1)/2
            (p + UInt256(1, 0, 0, 0)) >> 1,  // 另一种计算(p+1)/2的方法
            p - candidate_inv,  // p - (p+1)/2
        };
        
        for (size_t i = 0; i < special_candidates.size(); i++) {
            UInt256 candidate = special_candidates[i];
            UInt256 product = ModOp::mul(two, candidate);
            
            std::cout << "特殊候选" << (i+1) << ": " << candidate.to_hex().substr(48, 16) << " (低64位)" << std::endl;
            std::cout << "2 * 候选" << (i+1) << " mod p = " << product.to_hex() << std::endl;
            
            if (product == UInt256(1, 0, 0, 0)) {
                std::cout << "✅ 找到正确的模逆！" << std::endl;
                found = true;
                correct_inv = candidate;
                break;
            } else {
                std::cout << "❌ 不是正确的模逆" << std::endl;
            }
            std::cout << std::endl;
        }
    }
    
    if (found) {
        std::cout << "\n🎉 成功找到2的模逆: " << correct_inv.to_hex() << std::endl;
        
        // 验证其他数值的模逆
        std::cout << "\n=== 验证其他数值的模逆 ===" << std::endl;
        
        std::vector<UInt256> test_values = {
            UInt256(3, 0, 0, 0),
            UInt256(7, 0, 0, 0),
        };
        
        for (const auto& val : test_values) {
            // 使用费马小定理计算模逆
            UInt256 p_minus_2 = p - UInt256(2, 0, 0, 0);
            UInt256 fermat_inv = ModOp::pow_mod_simple(val, p_minus_2);
            UInt256 verification = ModOp::mul(val, fermat_inv);
            
            std::cout << val.limbs[0] << "的费马小定理模逆 = " << fermat_inv.to_hex().substr(48, 16) << " (低64位)" << std::endl;
            std::cout << "验证：" << val.limbs[0] << " * 模逆 mod p = " << verification.to_hex() << std::endl;
            std::cout << "正确: " << (verification == UInt256(1, 0, 0, 0) ? "是" : "否") << std::endl;
            std::cout << std::endl;
        }
    } else {
        std::cout << "\n❌ 未能找到2的正确模逆，需要进一步调试模乘法实现。" << std::endl;
    }
}

int main() {
    std::cout << "轻量级ECC库 - 最终模逆分析和修复" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        final_mod_inverse_analysis();
        
        std::cout << "\n🎉 最终模逆分析完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 分析失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
