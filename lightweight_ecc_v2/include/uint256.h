#ifndef LIGHTWEIGHT_ECC_UINT256_H
#define LIGHTWEIGHT_ECC_UINT256_H

#include <cstdint>
#include <string>
#include <cstring>
#include <iostream>

namespace LightweightECC {

// 256位无符号整数 - 基于gECC的多精度设计
class UInt256 {
public:
    // 构造函数
    UInt256();
    UInt256(uint64_t a, uint64_t b, uint64_t c, uint64_t d);
    UInt256(const std::string& hex);
    UInt256(const uint8_t* bytes);

    // 添加显式拷贝构造函数
    UInt256(const UInt256& other);

    // 添加移动构造函数
    UInt256(UInt256&& other) noexcept;
    
    // 基本运算
    UInt256 operator+(const UInt256& other) const;
    UInt256 operator-(const UInt256& other) const;
    UInt256 operator*(const UInt256& other) const;
    UInt256 operator/(const UInt256& other) const;
    UInt256 operator%(const UInt256& other) const;
    
    // 位运算
    UInt256 operator>>(int shift) const;
    UInt256 operator<<(int shift) const;
    UInt256 operator|(const UInt256& other) const;
    UInt256 operator&(const UInt256& other) const;
    UInt256 operator^(const UInt256& other) const;
    
    // 比较运算
    bool operator==(const UInt256& other) const;
    bool operator!=(const UInt256& other) const;
    bool operator<(const UInt256& other) const;
    bool operator>(const UInt256& other) const;
    bool operator<=(const UInt256& other) const;
    bool operator>=(const UInt256& other) const;
    
    // 赋值运算
    UInt256& operator=(const UInt256& other);
    UInt256& operator=(UInt256&& other) noexcept;
    UInt256& operator+=(const UInt256& other);
    UInt256& operator-=(const UInt256& other);
    UInt256& operator*=(const UInt256& other);
    UInt256& operator/=(const UInt256& other);
    UInt256& operator%=(const UInt256& other);
    
    // 特殊函数
    bool is_zero() const;
    bool is_odd() const;
    bool is_even() const;
    int bit_length() const;
    int get_bit(int position) const;
    void set_bit(int position, bool value);
    
    // 转换函数
    std::string to_hex() const;
    void to_bytes(uint8_t* bytes) const;
    static UInt256 from_hex(const std::string& hex);
    static UInt256 from_bytes(const uint8_t* bytes);
    
    // 修改静态常量声明方式
    static const UInt256& get_zero();
    static const UInt256& get_one();

    // 为了向后兼容，保留旧的访问方式
    static const UInt256& zero;
    static const UInt256& one;
    
    // 调试函数
    void print() const;
    
    // 直接访问limbs（用于优化）
    uint64_t limbs[4];

    // 公有函数用于调试
    static void div_limbs(const uint64_t* a, const uint64_t* b, uint64_t* quotient, uint64_t* remainder);

private:
    // 内部辅助函数
    void normalize();
    void init_zero();
    void init_one();
    static void add_limbs(const uint64_t* a, const uint64_t* b, uint64_t* result);
    static void sub_limbs(const uint64_t* a, const uint64_t* b, uint64_t* result);
    static void mul_limbs(const uint64_t* a, const uint64_t* b, uint64_t* result);
};

} // namespace LightweightECC

#endif // LIGHTWEIGHT_ECC_UINT256_H
