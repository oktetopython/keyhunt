#ifndef LIGHTWEIGHT_ECC_MOD_OP_H
#define LIGHTWEIGHT_ECC_MOD_OP_H

#include "uint256.h"

namespace LightweightECC {

// 模运算类 - 基于gECC的蒙哥马利算法思想
class ModOp {
public:
    // 初始化模数
    static void init(const UInt256& modulus);
    
    // 基本模运算
    static UInt256 add(const UInt256& a, const UInt256& b);
    static UInt256 sub(const UInt256& a, const UInt256& b);
    static UInt256 mul(const UInt256& a, const UInt256& b);
    static UInt256 inv(const UInt256& a);  // 模逆
    
    // 蒙哥马利运算（优化版本）
    static UInt256 montgomery_mul(const UInt256& a, const UInt256& b);
    static UInt256 montgomery_reduce(const UInt256& t);
    static UInt256 to_montgomery(const UInt256& a);
    static UInt256 from_montgomery(const UInt256& a);
    
    // 工具函数
    static bool is_initialized();
    static const UInt256& get_modulus();

    // 模幂运算（公开接口用于调试）
    static UInt256 mod_pow(const UInt256& base, const UInt256& exp, const UInt256& mod);
    static UInt256 pow_mod(const UInt256& base, const UInt256& exponent);
    static UInt256 pow_mod_optimized(const UInt256& base, const UInt256& exponent);
    static UInt256 pow_mod_simple(const UInt256& base, const UInt256& exponent);
    static UInt256 safe_add_mod(const UInt256& a, const UInt256& b);
    static UInt256 inv_fermat(const UInt256& a);
    static UInt256 pow_mod_optimized_fermat(const UInt256& base, const UInt256& exponent);

private:
    static UInt256 modulus;
    static uint64_t modulus_inv;  // 蒙哥马利常数
    static UInt256 r_squared;     // R^2 mod p
    static bool initialized;
    
    // 内部辅助函数
    static uint64_t compute_montgomery_constant(const UInt256& p);
    static UInt256 compute_r_squared(const UInt256& p);
    static UInt256 extended_gcd(const UInt256& a, const UInt256& b, UInt256& x, UInt256& y);
    static UInt256 basic_mul(const UInt256& a, const UInt256& b);
    static UInt256 basic_mod(const UInt256& a, const UInt256& p);
};

} // namespace LightweightECC

#endif // LIGHTWEIGHT_ECC_MOD_OP_H
