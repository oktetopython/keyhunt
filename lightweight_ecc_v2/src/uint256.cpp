#include "uint256.h"
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace LightweightECC {

// 修改静态常量定义 - 使用函数静态变量解决初始化顺序问题
const UInt256& UInt256::get_zero() {
    static const UInt256 zero_instance(0, 0, 0, 0);
    return zero_instance;
}

const UInt256& UInt256::get_one() {
    static const UInt256 one_instance(1, 0, 0, 0);
    return one_instance;
}

// 为了向后兼容，添加引用
const UInt256& UInt256::zero = get_zero();
const UInt256& UInt256::one = get_one();

// 修复默认构造函数
UInt256::UInt256() {
    init_zero();
}

UInt256::UInt256(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    limbs[0] = a;
    limbs[1] = b;
    limbs[2] = c;
    limbs[3] = d;
}

// 添加显式拷贝构造函数
UInt256::UInt256(const UInt256& other) {
    for (int i = 0; i < 4; i++) {
        limbs[i] = other.limbs[i];
    }
}

// 添加移动构造函数
UInt256::UInt256(UInt256&& other) noexcept {
    for (int i = 0; i < 4; i++) {
        limbs[i] = other.limbs[i];
        other.limbs[i] = 0;  // 不是必须的，但是好习惯
    }
}

UInt256::UInt256(const std::string& hex) {
    *this = from_hex(hex);
}

UInt256::UInt256(const uint8_t* bytes) {
    *this = from_bytes(bytes);
}

UInt256 UInt256::operator+(const UInt256& other) const {
    UInt256 result;
    add_limbs(limbs, other.limbs, result.limbs);
    return result;
}

UInt256 UInt256::operator-(const UInt256& other) const {
    UInt256 result;
    sub_limbs(limbs, other.limbs, result.limbs);
    return result;
}

UInt256 UInt256::operator*(const UInt256& other) const {
    UInt256 result;
    mul_limbs(limbs, other.limbs, result.limbs);
    return result;
}

UInt256 UInt256::operator/(const UInt256& other) const {
    UInt256 quotient, remainder;
    div_limbs(limbs, other.limbs, quotient.limbs, remainder.limbs);
    return quotient;
}

UInt256 UInt256::operator%(const UInt256& other) const {
    UInt256 quotient, remainder;
    div_limbs(limbs, other.limbs, quotient.limbs, remainder.limbs);
    return remainder;
}

UInt256 UInt256::operator>>(int shift) const {
    UInt256 result = *this;

    // 修复：正确的右移操作
    while (shift >= 64) {
        // 右移64位：高位limbs移到低位limbs
        for (int i = 0; i < 3; i++) {
            result.limbs[i] = result.limbs[i+1];
        }
        result.limbs[3] = 0;  // 最高位清零
        shift -= 64;
    }

    if (shift > 0) {
        uint64_t carry = 0;
        // 从高位到低位处理
        for (int i = 3; i >= 0; i--) {
            uint64_t new_carry = result.limbs[i] << (64 - shift);
            result.limbs[i] = (result.limbs[i] >> shift) | carry;
            carry = new_carry;
        }
    }

    return result;
}

UInt256 UInt256::operator<<(int shift) const {
    UInt256 result = *this;
    
    while (shift >= 64) {
        for (int i = 0; i < 3; i++) {
            result.limbs[i] = result.limbs[i+1];
        }
        result.limbs[3] = 0;
        shift -= 64;
    }
    
    if (shift > 0) {
        uint64_t carry = 0;
        for (int i = 3; i >= 0; i--) {
            uint64_t new_carry = result.limbs[i] >> (64 - shift);
            result.limbs[i] = (result.limbs[i] << shift) | carry;
            carry = new_carry;
        }
    }
    
    return result;
}

bool UInt256::operator==(const UInt256& other) const {
    for (int i = 0; i < 4; i++) {
        if (limbs[i] != other.limbs[i]) return false;
    }
    return true;
}

bool UInt256::operator!=(const UInt256& other) const {
    return !(*this == other);
}

bool UInt256::operator<(const UInt256& other) const {
    for (int i = 3; i >= 0; i--) {
        if (limbs[i] < other.limbs[i]) return true;
        if (limbs[i] > other.limbs[i]) return false;
    }
    return false;
}

bool UInt256::operator>(const UInt256& other) const {
    return other < *this;
}

bool UInt256::operator<=(const UInt256& other) const {
    return !(other < *this);
}

bool UInt256::operator>=(const UInt256& other) const {
    return !(*this < other);
}

// 修复赋值运算符
UInt256& UInt256::operator=(const UInt256& other) {
    if (this != &other) {
        for (int i = 0; i < 4; i++) {
            limbs[i] = other.limbs[i];
        }
    }
    return *this;
}

// 添加移动赋值运算符
UInt256& UInt256::operator=(UInt256&& other) noexcept {
    if (this != &other) {
        for (int i = 0; i < 4; i++) {
            limbs[i] = other.limbs[i];
            other.limbs[i] = 0;
        }
    }
    return *this;
}

bool UInt256::is_zero() const {
    return *this == zero;
}

bool UInt256::is_odd() const {
    return (limbs[0] & 1) != 0;
}

bool UInt256::is_even() const {
    return !is_odd();
}

int UInt256::bit_length() const {
    for (int i = 3; i >= 0; i--) {
        if (limbs[i] != 0) {
            return i * 64 + (64 - __builtin_clzll(limbs[i]));
        }
    }
    return 0;
}

int UInt256::get_bit(int position) const {
    if (position < 0 || position >= 256) return 0;
    
    int limb_index = position / 64;
    int bit_index = position % 64;
    
    return (limbs[limb_index] >> bit_index) & 1;
}

void UInt256::set_bit(int position, bool value) {
    if (position < 0 || position >= 256) return;
    
    int limb_index = position / 64;
    int bit_index = position % 64;
    
    if (value) {
        limbs[limb_index] |= (1ULL << bit_index);
    } else {
        limbs[limb_index] &= ~(1ULL << bit_index);
    }
}

std::string UInt256::to_hex() const {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    
    for (int i = 3; i >= 0; i--) {
        ss << std::setw(16) << limbs[i];
    }
    
    return ss.str();
}

void UInt256::to_bytes(uint8_t* bytes) const {
    for (int i = 0; i < 4; i++) {
        uint64_t limb = limbs[i];
        for (int j = 0; j < 8; j++) {
            bytes[i * 8 + j] = (limb >> (j * 8)) & 0xFF;
        }
    }
}

// 修改from_hex函数避免依赖静态常量
UInt256 UInt256::from_hex(const std::string& hex) {
    UInt256 result;  // 使用默认构造函数而不是zero

    // 去除0x前缀
    std::string clean_hex = hex;
    if (clean_hex.substr(0, 2) == "0x") {
        clean_hex = clean_hex.substr(2);
    }

    // 填充到64字符
    while (clean_hex.length() < 64) {
        clean_hex = "0" + clean_hex;
    }

    // 解析每个limb
    for (int i = 0; i < 4; i++) {
        std::string limb_hex = clean_hex.substr(i * 16, 16);
        result.limbs[3 - i] = std::stoull(limb_hex, nullptr, 16);
    }

    return result;
}

// 修改from_bytes函数避免依赖静态常量
UInt256 UInt256::from_bytes(const uint8_t* bytes) {
    UInt256 result;  // 使用默认构造函数而不是zero

    for (int i = 0; i < 4; i++) {
        uint64_t limb = 0;
        for (int j = 0; j < 8; j++) {
            limb |= (uint64_t)bytes[i * 8 + j] << (j * 8);
        }
        result.limbs[i] = limb;
    }

    return result;
}

void UInt256::print() const {
    std::cout << "0x" << to_hex() << std::endl;
}

void UInt256::normalize() {
    // UInt256不需要标准化，总是4个limbs
}

// 添加内部初始化函数
void UInt256::init_zero() {
    for (int i = 0; i < 4; i++) {
        limbs[i] = 0;
    }
}

void UInt256::init_one() {
    limbs[0] = 1;
    for (int i = 1; i < 4; i++) {
        limbs[i] = 0;
    }
}

void UInt256::add_limbs(const uint64_t* a, const uint64_t* b, uint64_t* result) {
    uint64_t carry = 0;

    for (int i = 0; i < 4; i++) {
        result[i] = a[i] + b[i] + carry;
        carry = (result[i] < a[i]) ? 1 : 0;
    }
}

void UInt256::sub_limbs(const uint64_t* a, const uint64_t* b, uint64_t* result) {
    uint64_t borrow = 0;

    for (int i = 0; i < 4; i++) {
        result[i] = a[i] - b[i] - borrow;
        borrow = (result[i] > a[i]) ? 1 : 0;
    }
}

void UInt256::mul_limbs(const uint64_t* a, const uint64_t* b, uint64_t* result) {
    // 简化的乘法实现 - 基于gECC的思想
    uint64_t temp[8] = {0};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            __uint128_t product = (__uint128_t)a[i] * b[j];
            uint64_t low = (uint64_t)product;
            uint64_t high = (uint64_t)(product >> 64);

            temp[i + j] += low;
            if (temp[i + j] < low) temp[i + j + 1]++;

            temp[i + j + 1] += high;
        }
    }

    // 只取低256位
    for (int i = 0; i < 4; i++) {
        result[i] = temp[i];
    }
}

void UInt256::div_limbs(const uint64_t* a, const uint64_t* b, uint64_t* quotient, uint64_t* remainder) {
    // 第3天修复：实现正确的256位除法算法
    UInt256 dividend(a[0], a[1], a[2], a[3]);
    UInt256 divisor(b[0], b[1], b[2], b[3]);

    if (divisor.is_zero()) {
        throw std::runtime_error("Division by zero");
    }

    // 初始化商和余数为0
    UInt256 q(0, 0, 0, 0);  // 显式初始化
    UInt256 r(0, 0, 0, 0);  // 显式初始化

    // 如果被除数等于除数，商为1，余数为0
    if (dividend == divisor) {
        quotient[0] = 1;
        quotient[1] = 0;
        quotient[2] = 0;
        quotient[3] = 0;
        remainder[0] = 0;
        remainder[1] = 0;
        remainder[2] = 0;
        remainder[3] = 0;
        return;
    }

    // 如果被除数小于除数，商为0，余数为被除数
    if (dividend < divisor) {
        for (int i = 0; i < 4; i++) {
            quotient[i] = 0;
            remainder[i] = dividend.limbs[i];
        }
        return;
    }

    // 最终修复：避免左移溢出的256位长除法算法
    for (int i = 255; i >= 0; i--) {
        // 手动实现安全的左移，避免高位溢出
        // 将r左移1位，但要确保不丢失数据

        // 检查最高位是否会溢出
        bool will_overflow = r.get_bit(255);

        // 如果会溢出且r >= divisor，说明商的这一位应该是1
        if (will_overflow) {
            // 先减去除数，再左移
            r = r - divisor;
            q.set_bit(i, true);
        }

        // 安全左移：手动移位而不使用<<操作符
        for (int j = 255; j > 0; j--) {
            if (r.get_bit(j-1)) {
                r.set_bit(j, true);
            } else {
                r.set_bit(j, false);
            }
        }

        // 设置最低位为被除数的当前位
        if (dividend.get_bit(i)) {
            r.set_bit(0, true);
        } else {
            r.set_bit(0, false);
        }

        // 如果余数大于等于除数，进行减法并设置商的对应位
        if (!will_overflow && r >= divisor) {
            r = r - divisor;
            q.set_bit(i, true);
        } else if (!will_overflow) {
            q.set_bit(i, false);
        }
    }

    // 设置结果
    for (int i = 0; i < 4; i++) {
        quotient[i] = q.limbs[i];
        remainder[i] = r.limbs[i];
    }
}

} // namespace LightweightECC
