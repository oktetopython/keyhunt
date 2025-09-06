#ifndef KARATSUBA_MULTIPLIER_H
#define KARATSUBA_MULTIPLIER_H

#include "Int.h"
#include <vector>

/**
 * @brief Karatsuba乘法优化器
 * @details 实现Karatsuba算法来加速大数乘法运算
 */
class KaratsubaMultiplier {
private:
    // 阈值：当位数小于这个值时使用朴素乘法
    static constexpr int KARATSUBA_THRESHOLD = 128;
    
    // 临时存储，避免重复分配
    std::vector<Int> temp_storage;

public:
    KaratsubaMultiplier() : temp_storage(16) {}
    
    /**
     * @brief 执行Karatsuba乘法
     * @param result 结果
     * @param a 乘数1
     * @param b 乘数2
     */
    void multiply(Int* result, const Int* a, const Int* b) {
        int bits = std::max(a->GetBitLength(), b->GetBitLength());
        
        // 小数字使用朴素乘法
        if (bits < KARATSUBA_THRESHOLD) {
            result->Mult(const_cast<Int*>(a), const_cast<Int*>(b));
            return;
        }
        
        // 大数字使用Karatsuba算法
        karatsuba_multiply(result, a, b, bits);
    }

private:
    /**
     * @brief Karatsuba乘法核心实现
     */
    void karatsuba_multiply(Int* result, const Int* a, const Int* b, int n) {
        // 确保有足够的临时存储
        if (temp_storage.size() < 4) {
            temp_storage.resize(4);
        }
        
        // 将输入分解为两部分
        int m = n / 2;
        
        Int& high1 = temp_storage[0];
        Int& low1 = temp_storage[1];
        Int& high2 = temp_storage[2];
        Int& low2 = temp_storage[3];
        
        // 分解数字
        split_number(a, high1, low1, m);
        split_number(b, high2, low2, m);
        
        // 计算三个乘积
        Int z0, z1, z2;
        
        // z0 = low1 * low2
        if (low1.GetBitLength() < KARATSUBA_THRESHOLD || low2.GetBitLength() < KARATSUBA_THRESHOLD) {
            z0.Mult(const_cast<Int*>(&low1), const_cast<Int*>(&low2));
        } else {
            karatsuba_multiply(&z0, &low1, &low2, m);
        }
        
        // z2 = high1 * high2
        if (high1.GetBitLength() < KARATSUBA_THRESHOLD || high2.GetBitLength() < KARATSUBA_THRESHOLD) {
            z2.Mult(const_cast<Int*>(&high1), const_cast<Int*>(&high2));
        } else {
            karatsuba_multiply(&z2, &high1, high2, n - m);
        }
        
        // (high1 + low1) * (high2 + low2)
        Int sum1, sum2;
        sum1.Add(const_cast<Int*>(&high1), const_cast<Int*>(&low1));
        sum2.Add(const_cast<Int*>(&high2), const_cast<Int*>(&low2));
        
        // z1 = (high1 + low1) * (high2 + low2) - z2 - z0
        if (sum1.GetBitLength() < KARATSUBA_THRESHOLD || sum2.GetBitLength() < KARATSUBA_THRESHOLD) {
            z1.Mult(&sum1, &sum2);
        } else {
            karatsuba_multiply(&z1, &sum1, &sum2, std::max(sum1.GetBitLength(), sum2.GetBitLength()));
        }
        
        z1.Sub(&z1, &z2);
        z1.Sub(&z1, &z0);
        
        // 组合结果：result = z2 * B^(2m) + z1 * B^m + z0
        combine_result(result, &z0, &z1, &z2, m);
    }
    
    /**
     * @brief 将大数分解为两部分
     */
    void split_number(const Int* num, Int& high, Int& low, int split_point) {
        int total_bits = num->GetBitLength();
        
        if (total_bits <= split_point) {
            high.SetInt32(0);
            low.Set(num);
            return;
        }
        
        // 低位部分
        low.Set(num);
        low.MaskByte(split_point / 8);
        
        // 高位部分
        high.Set(num);
        high.ShiftR(split_point);
    }
    
    /**
     * @brief 组合Karatsuba结果
     */
    void combine_result(Int* result, const Int* z0, const Int* z1, const Int* z2, int m) {
        result->Set(z0);  // 基础部分
        
        // 添加中间部分：z1 * B^m
        Int temp(z1);
        temp.ShiftL(m);
        result->Add(result, &temp);
        
        // 添加高位部分：z2 * B^(2m)
        temp.Set(z2);
        temp.ShiftL(2 * m);
        result->Add(result, &temp);
    }
};

/**
 * @brief 全局Karatsuba乘法器实例
 */
extern KaratsubaMultiplier g_karatsuba_multiplier;

#endif // KARATSUBA_MULTIPLIER_H