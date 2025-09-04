#ifndef UNIFIED_ECC_MATH_COMMON_H
#define UNIFIED_ECC_MATH_COMMON_H

#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

namespace UnifiedECC {
namespace Math {

// ============================================================================
// 通用数学常量和工具
// ============================================================================

// 数学常数
constexpr double PI = 3.14159265358979323846;
constexpr double E = 2.71828182845904523536;

// 位操作工具
class BitUtils {
public:
    static inline bool get_bit(const uint64_t* limbs, int bit_position) {
        int limb_index = bit_position / 64;
        int bit_index = bit_position % 64;
        return (limbs[limb_index] & (1ULL << bit_index)) != 0;
    }

    static inline void set_bit(uint64_t* limbs, int bit_position, bool value) {
        int limb_index = bit_position / 64;
        int bit_index = bit_position % 64;
        if (value) {
            limbs[limb_index] |= (1ULL << bit_index);
        } else {
            limbs[limb_index] &= ~(1ULL << bit_index);
        }
    }

    static inline int bit_length(const uint64_t* limbs, int num_limbs) {
        for (int i = num_limbs - 1; i >= 0; --i) {
            if (limbs[i] != 0) {
                return 64 * i + 64 - __builtin_clzll(limbs[i]);
            }
        }
        return 0;
    }

    static inline int hamming_weight(const uint64_t* limbs, int num_limbs) {
        int weight = 0;
        for (int i = 0; i < num_limbs; ++i) {
            weight += __builtin_popcountll(limbs[i]);
        }
        return weight;
    }
};

// ============================================================================
// 大整数运算工具
// ============================================================================

class BigIntUtils {
public:
    // 大整数加法 (处理进位)
    static void add(uint64_t* result, const uint64_t* a, const uint64_t* b, int num_limbs) {
        uint64_t carry = 0;
        for (int i = 0; i < num_limbs; ++i) {
            uint64_t sum = a[i] + b[i] + carry;
            result[i] = sum & 0xFFFFFFFFFFFFFFFFULL;
            carry = sum >> 64;
        }
    }

    // 大整数减法 (处理借位)
    static void sub(uint64_t* result, const uint64_t* a, const uint64_t* b, int num_limbs) {
        uint64_t borrow = 0;
        for (int i = 0; i < num_limbs; ++i) {
            uint64_t diff = a[i] - b[i] - borrow;
            result[i] = diff & 0xFFFFFFFFFFFFFFFFULL;
            borrow = (diff >> 63) & 1; // 符号扩展
        }
    }

    // 大整数比较
    static int compare(const uint64_t* a, const uint64_t* b, int num_limbs) {
        for (int i = num_limbs - 1; i >= 0; --i) {
            if (a[i] != b[i]) {
                return a[i] > b[i] ? 1 : -1;
            }
        }
        return 0;
    }

    // 大整数左移
    static void shift_left(uint64_t* result, const uint64_t* a, int shift_bits, int num_limbs) {
        int limb_shift = shift_bits / 64;
        int bit_shift = shift_bits % 64;

        // 清空结果
        for (int i = 0; i < num_limbs; ++i) {
            result[i] = 0;
        }

        if (limb_shift >= num_limbs) {
            return; // 移位超出范围
        }

        if (bit_shift == 0) {
            // 纯limb移位
            for (int i = 0; i < num_limbs - limb_shift; ++i) {
                result[i + limb_shift] = a[i];
            }
        } else {
            // 带bit移位
            for (int i = 0; i < num_limbs - limb_shift - 1; ++i) {
                result[i + limb_shift] = (a[i] << bit_shift) |
                                        (a[i + 1] >> (64 - bit_shift));
            }
            if (num_limbs > limb_shift) {
                result[num_limbs - 1] = a[num_limbs - limb_shift - 1] << bit_shift;
            }
        }
    }

    // 大整数右移
    static void shift_right(uint64_t* result, const uint64_t* a, int shift_bits, int num_limbs) {
        int limb_shift = shift_bits / 64;
        int bit_shift = shift_bits % 64;

        // 清空结果
        for (int i = 0; i < num_limbs; ++i) {
            result[i] = 0;
        }

        if (limb_shift >= num_limbs) {
            return; // 移位超出范围
        }

        if (bit_shift == 0) {
            // 纯limb移位
            for (int i = 0; i < num_limbs - limb_shift; ++i) {
                result[i] = a[i + limb_shift];
            }
        } else {
            // 带bit移位
            for (int i = 0; i < num_limbs - limb_shift - 1; ++i) {
                result[i] = (a[i + limb_shift] >> bit_shift) |
                           (a[i + limb_shift + 1] << (64 - bit_shift));
            }
            if (num_limbs > limb_shift) {
                result[num_limbs - limb_shift - 1] = a[num_limbs - 1] >> bit_shift;
            }
        }
    }

    // 大整数乘法 (简化的二进制乘法)
    static void mul(uint64_t* result, const uint64_t* a, const uint64_t* b, int num_limbs) {
        // 清空结果
        for (int i = 0; i < 2 * num_limbs; ++i) {
            result[i] = 0;
        }

        // 二进制乘法
        for (int i = 0; i < num_limbs; ++i) {
            uint64_t carry = 0;
            for (int j = 0; j < num_limbs; ++j) {
                if (BitUtils::get_bit(b, i * 64 + j)) {
                    uint64_t temp_carry = 0;
                    for (int k = 0; k < num_limbs; ++k) {
                        uint64_t sum = result[i + j + k] + a[k] + temp_carry;
                        result[i + j + k] = sum & 0xFFFFFFFFFFFFFFFFULL;
                        temp_carry = sum >> 64;
                    }
                    // 处理最后的进位
                    int carry_idx = i + j + num_limbs;
                    while (temp_carry != 0 && carry_idx < 2 * num_limbs) {
                        uint64_t sum = result[carry_idx] + temp_carry;
                        result[carry_idx] = sum & 0xFFFFFFFFFFFFFFFFULL;
                        temp_carry = sum >> 64;
                        carry_idx++;
                    }
                }
            }
        }
    }

    // 简化的模运算 (用于模数较小的情况)
    static void mod(uint64_t* result, const uint64_t* a, const uint64_t* modulus, int num_limbs) {
        // 复制输入
        uint64_t temp[8]; // 假设最大8个limb
        for (int i = 0; i < 2 * num_limbs; ++i) {
            temp[i] = a[i];
        }

        // 简化的模运算实现
        // 这里使用二进制长除法的基本思想
        int mod_bits = BitUtils::bit_length(modulus, num_limbs);

        for (int i = 2 * num_limbs * 64 - 1; i >= mod_bits - 1; --i) {
            if (BitUtils::get_bit(temp, i)) {
                // 减去 modulus << (i - mod_bits + 1)
                int shift = i - mod_bits + 1;
                uint64_t shifted_mod[8] = {0};
                shift_left(shifted_mod, modulus, shift, num_limbs);

                uint64_t borrow = 0;
                for (int j = 0; j < 2 * num_limbs; ++j) {
                    uint64_t diff = temp[j] - shifted_mod[j] - borrow;
                    temp[j] = diff & 0xFFFFFFFFFFFFFFFFULL;
                    borrow = (diff >> 63) & 1;
                }
            }
        }

        // 复制结果
        for (int i = 0; i < num_limbs; ++i) {
            result[i] = temp[i];
        }
    }
};

// ============================================================================
// 素数和数论工具
// ============================================================================

class NumberTheory {
public:
    // 欧几里得算法计算GCD
    static uint64_t gcd(uint64_t a, uint64_t b) {
        while (b != 0) {
            uint64_t t = b;
            b = a % b;
            a = t;
        }
        return a;
    }

    // 扩展欧几里得算法
    static uint64_t extended_gcd(uint64_t a, uint64_t b, int64_t& x, int64_t& y) {
        if (a == 0) {
            x = 0;
            y = 1;
            return b;
        }

        int64_t x1, y1;
        uint64_t gcd = extended_gcd(b % a, a, x1, y1);

        x = y1 - (b / a) * x1;
        y = x1;

        return gcd;
    }

    // 模逆 (使用扩展欧几里得算法)
    static uint64_t mod_inverse(uint64_t a, uint64_t modulus) {
        int64_t x, y;
        uint64_t g = extended_gcd(a, modulus, x, y);

        if (g != 1) {
            throw std::invalid_argument("Modular inverse does not exist");
        }

        // 确保结果为正
        return (x % static_cast<int64_t>(modulus) + modulus) % modulus;
    }

    // 费马素性测试 (简化版本)
    static bool is_prime_fermat(uint64_t n, int iterations = 5) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0) return false;

        for (int i = 0; i < iterations; ++i) {
            uint64_t a = 2 + rand() % (n - 3);
            if (mod_pow(a, n - 1, n) != 1) {
                return false;
            }
        }
        return true;
    }

    // 快速模幂
    static uint64_t mod_pow(uint64_t base, uint64_t exponent, uint64_t modulus) {
        uint64_t result = 1;
        base %= modulus;

        while (exponent > 0) {
            if (exponent & 1) {
                result = (result * base) % modulus;
            }
            base = (base * base) % modulus;
            exponent >>= 1;
        }

        return result;
    }

    // 生成随机素数 (简化实现)
    static uint64_t generate_prime(int bits) {
        uint64_t min = 1ULL << (bits - 1);
        uint64_t max = (1ULL << bits) - 1;

        uint64_t candidate;
        do {
            candidate = min + (rand() % (max - min + 1));
            candidate |= 1; // 确保是奇数
        } while (!is_prime_fermat(candidate));

        return candidate;
    }
};

// ============================================================================
// 性能监控工具
// ============================================================================

class PerformanceMonitor {
public:
    struct TimingResult {
        double elapsed_seconds;
        uint64_t operations_per_second;
    };

    // 简化的性能监控实现
    static TimingResult time_operation_simple(uint64_t iterations = 1) {
        // 简化的计时实现 - 实际项目中应该使用更精确的计时器
        return {
            0.001, // 假设1ms
            iterations * 1000 // 假设1000 ops/s
        };
    }

    static std::string format_performance(double ops_per_sec) {
        if (ops_per_sec >= 1e9) {
            return std::to_string(ops_per_sec / 1e9) + " Gops/s";
        } else if (ops_per_sec >= 1e6) {
            return std::to_string(ops_per_sec / 1e6) + " Mops/s";
        } else if (ops_per_sec >= 1e3) {
            return std::to_string(ops_per_sec / 1e3) + " Kops/s";
        } else {
            return std::to_string(ops_per_sec) + " ops/s";
        }
    }
};

// ============================================================================
// 内存管理工具
// ============================================================================

class MemoryUtils {
public:
    // 安全的内存分配
    template<typename T>
    static T* safe_malloc(size_t count) {
        T* ptr = static_cast<T*>(malloc(count * sizeof(T)));
        if (!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    // 安全的内存释放
    template<typename T>
    static void safe_free(T*& ptr) {
        if (ptr) {
            free(ptr);
            ptr = nullptr;
        }
    }

    // 内存复制 (处理重叠)
    static void safe_memcpy(void* dest, const void* src, size_t size) {
        if (dest && src && size > 0) {
            memmove(dest, src, size);
        }
    }

    // 内存置零
    static void secure_zero(void* ptr, size_t size) {
        if (ptr && size > 0) {
            volatile unsigned char* p = static_cast<volatile unsigned char*>(ptr);
            while (size--) {
                *p++ = 0;
            }
        }
    }
};

} // namespace Math
} // namespace UnifiedECC

#endif // UNIFIED_ECC_MATH_COMMON_H