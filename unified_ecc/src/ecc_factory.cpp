#include "../include/ecc_interface.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

// 条件包含各个后端的适配器
#ifdef WITH_LIGHTWEIGHT_ECC
#include "../../lightweight_ecc_v2/include/ec_op.h"
#include "../../lightweight_ecc_v2/include/mod_op.h"
#endif

#ifdef WITH_GECC
#include "../../gECC-main/include/gecc.h"
#endif

#ifdef WITH_KEYHUNT
#include "../../keyhuntcuda/KeyHunt-Cuda/Secp256K1.h"
#endif

namespace UnifiedECC {

// ============================================================================
// UInt256 实现
// ============================================================================

UInt256::UInt256() {
    limbs[0] = limbs[1] = limbs[2] = limbs[3] = 0;
}

UInt256::UInt256(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    limbs[0] = a;
    limbs[1] = b;
    limbs[2] = c;
    limbs[3] = d;
}

UInt256::UInt256(const std::string& hex) {
    // 简化的十六进制解析实现
    std::string clean_hex = hex;
    if (clean_hex.substr(0, 2) == "0x") {
        clean_hex = clean_hex.substr(2);
    }

    // 填充到64个字符
    while (clean_hex.length() < 64) {
        clean_hex = "0" + clean_hex;
    }

    // 解析4个16字符的块
    for (int i = 0; i < 4; ++i) {
        std::string chunk = clean_hex.substr(i * 16, 16);
        limbs[3 - i] = std::stoull(chunk, nullptr, 16);
    }
}

UInt256::UInt256(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != 32) {
        throw std::invalid_argument("UInt256 requires exactly 32 bytes");
    }

    for (int i = 0; i < 4; ++i) {
        limbs[i] = 0;
        for (int j = 0; j < 8; ++j) {
            limbs[i] |= static_cast<uint64_t>(bytes[i * 8 + j]) << (j * 8);
        }
    }
}

// 简化的运算符实现（实际项目中需要完整的256位运算）
UInt256 UInt256::operator+(const UInt256& other) const {
    UInt256 result = *this;
    uint64_t carry = 0;

    for (int i = 0; i < 4; ++i) {
        uint64_t sum = result.limbs[i] + other.limbs[i] + carry;
        result.limbs[i] = sum & 0xFFFFFFFFFFFFFFFFULL;
        carry = sum >> 64;
    }

    return result;
}

UInt256 UInt256::operator-(const UInt256& other) const {
    UInt256 result = *this;
    uint64_t borrow = 0;

    for (int i = 0; i < 4; ++i) {
        uint64_t diff = result.limbs[i] - other.limbs[i] - borrow;
        result.limbs[i] = diff & 0xFFFFFFFFFFFFFFFFULL;
        borrow = (diff >> 63) & 1; // 借位
    }

    return result;
}

UInt256 UInt256::operator*(const UInt256& other) const {
    // 简化的实现 - 只处理低64位
    return UInt256(limbs[0] * other.limbs[0], 0, 0, 0);
}

UInt256 UInt256::operator/(const UInt256& other) const {
    // 简化的实现
    if (other.is_zero()) throw std::invalid_argument("Division by zero");
    return UInt256(limbs[0] / other.limbs[0], 0, 0, 0);
}

UInt256 UInt256::operator%(const UInt256& other) const {
    // 简化的实现
    if (other.is_zero()) throw std::invalid_argument("Modulo by zero");
    return UInt256(limbs[0] % other.limbs[0], 0, 0, 0);
}

bool UInt256::operator==(const UInt256& other) const {
    return limbs[0] == other.limbs[0] &&
           limbs[1] == other.limbs[1] &&
           limbs[2] == other.limbs[2] &&
           limbs[3] == other.limbs[3];
}

bool UInt256::operator<(const UInt256& other) const {
    for (int i = 3; i >= 0; --i) {
        if (limbs[i] != other.limbs[i]) {
            return limbs[i] < other.limbs[i];
        }
    }
    return false;
}

bool UInt256::is_zero() const {
    return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
}

bool UInt256::is_odd() const {
    return (limbs[0] & 1) != 0;
}

std::string UInt256::to_hex() const {
    std::stringstream ss;
    ss << "0x";
    for (int i = 3; i >= 0; --i) {
        ss << std::hex << std::setfill('0') << std::setw(16) << limbs[i];
    }
    return ss.str();
}

std::vector<uint8_t> UInt256::to_bytes() const {
    std::vector<uint8_t> bytes(32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            bytes[i * 8 + j] = (limbs[i] >> (j * 8)) & 0xFF;
        }
    }
    return bytes;
}

// ============================================================================
// ECPoint 实现
// ============================================================================

ECPoint::ECPoint() : infinity(true) {}

ECPoint::ECPoint(const UInt256& x, const UInt256& y) : x(x), y(y), infinity(false) {}

ECPoint::ECPoint(bool infinity) : infinity(infinity) {}

bool ECPoint::is_infinity() const {
    return infinity;
}

std::string ECPoint::to_string() const {
    if (infinity) {
        return "Point at infinity";
    }
    return "(" + x.to_hex() + ", " + y.to_hex() + ")";
}

// ============================================================================
// 预定义曲线参数
// ============================================================================

namespace Curves {
    const ECCParams secp256k1 = {
        UInt256(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFEFFFFFC2FULL), // p
        UInt256(0, 0, 0, 0), // a
        UInt256(0, 0, 0, 7), // b
        ECPoint(UInt256(0x79BE667EF9DCBBACULL, 0x55A06295CE870B07ULL, 0x029BFCDB2DCE28D9ULL, 0x59F2815B16F81798ULL),
                UInt256(0x483ADA7726A3C465ULL, 0x5DA4FBFC0E1108A8ULL, 0xFD17B448A6855419ULL, 0x9C47D08FFB10D4B8ULL)), // G
        UInt256(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFEULL, 0xBAAEDCE6AF48A03BULL, 0xBFD25E8CD0364141ULL), // n
        "secp256k1"
    };

    const ECCParams secp256r1 = {
        UInt256(0xFFFFFFFF00000001ULL, 0x0000000000000000ULL, 0x00000000FFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL), // p
        UInt256(0xFFFFFFFF00000001ULL, 0x0000000000000000ULL, 0x00000000FFFFFFFFULL, 0xFFFFFFFCULL), // a
        UInt256(0x5AC635D8AA3A93E7ULL, 0xB3EBBD55769886BCULL, 0x651D06B0CC53B0F6ULL, 0x3BCE3C3E27D2604BULL), // b
        ECPoint(), // G (需要完整实现)
        UInt256(0xFFFFFFFF00000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xBCE6FAADA7179E84ULL, 0xF3B9CAC2FC632551ULL), // n
        "secp256r1"
    };

    const ECCParams ed25519 = {
        UInt256(0x7FFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFEDULL), // p
        UInt256(0x7FFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFECULL), // a = -1 mod p
        UInt256(0x52036CEE2B6FFE73ULL, 0x8CC740797779E898ULL, 0x00700A4D4141D8ABULL, 0x75EB4DCA135978A3ULL), // b
        ECPoint(), // G (will be set properly later)
        UInt256(0x1000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL), // n (simplified)
        "ed25519"
    };
}

// ============================================================================
// 类型转换辅助函数
// ============================================================================

#ifdef WITH_LIGHTWEIGHT_ECC
// 转换UnifiedECC::UInt256到LightweightECC::UInt256
LightweightECC::UInt256 toLightweightUInt256(const UnifiedECC::UInt256& u) {
    return LightweightECC::UInt256(u.limbs[0], u.limbs[1], u.limbs[2], u.limbs[3]);
}

// 转换LightweightECC::UInt256到UnifiedECC::UInt256
UnifiedECC::UInt256 fromLightweightUInt256(const LightweightECC::UInt256& u) {
    return UnifiedECC::UInt256(u.limbs[0], u.limbs[1], u.limbs[2], u.limbs[3]);
}

// 转换UnifiedECC::ECPoint到LightweightECC::Point
LightweightECC::Point toLightweightPoint(const UnifiedECC::ECPoint& p) {
    if (p.is_infinity()) {
        return LightweightECC::Point();
    }
    return LightweightECC::Point(toLightweightUInt256(p.x), toLightweightUInt256(p.y));
}

// 转换LightweightECC::Point到UnifiedECC::ECPoint
UnifiedECC::ECPoint fromLightweightPoint(const LightweightECC::Point& p) {
    if (p.is_infinity()) {
        return UnifiedECC::ECPoint(true);
    }
    return UnifiedECC::ECPoint(fromLightweightUInt256(p.x), fromLightweightUInt256(p.y));
}
#endif

// ============================================================================
// 轻量级ECC适配器实现
// ============================================================================

#ifdef WITH_LIGHTWEIGHT_ECC
class LightweightECCAdapter : public IECCOperations {
private:
    ECCParams params_;

public:
    bool initialize(const ECCParams& params) override {
        params_ = params;

        LightweightECC::CurveParams lw_params;
        lw_params.p = toLightweightUInt256(params.p);
        lw_params.a = toLightweightUInt256(params.a);
        lw_params.b = toLightweightUInt256(params.b);
        lw_params.G = toLightweightPoint(params.G);
        lw_params.n = toLightweightUInt256(params.n);

        try {
            LightweightECC::ECOp::init(lw_params);
            LightweightECC::ModOp::init(toLightweightUInt256(params.p));
            return true;
        } catch (...) {
            return false;
        }
    }

    ECPoint point_add(const ECPoint& P, const ECPoint& Q) override {
        LightweightECC::Point lp = LightweightECC::ECOp::point_add(
            toLightweightPoint(P), toLightweightPoint(Q));
        return fromLightweightPoint(lp);
    }

    ECPoint point_double(const ECPoint& P) override {
        LightweightECC::Point lp = LightweightECC::ECOp::point_double(toLightweightPoint(P));
        return fromLightweightPoint(lp);
    }

    ECPoint scalar_mul(const UInt256& k, const ECPoint& P) override {
        LightweightECC::Point lp = LightweightECC::ECOp::scalar_mul(
            toLightweightUInt256(k), toLightweightPoint(P));
        return fromLightweightPoint(lp);
    }

    std::vector<ECPoint> batch_scalar_mul(const std::vector<UInt256>& scalars,
                                         const std::vector<ECPoint>& points) override {
        // 简化的批量实现
        std::vector<ECPoint> results;
        for (size_t i = 0; i < scalars.size(); ++i) {
            results.push_back(scalar_mul(scalars[i], points[i]));
        }
        return results;
    }

    UInt256 mod_add(const UInt256& a, const UInt256& b) override {
        LightweightECC::UInt256 result = LightweightECC::ModOp::add(
            toLightweightUInt256(a), toLightweightUInt256(b));
        return fromLightweightUInt256(result);
    }

    UInt256 mod_sub(const UInt256& a, const UInt256& b) override {
        LightweightECC::UInt256 result = LightweightECC::ModOp::sub(
            toLightweightUInt256(a), toLightweightUInt256(b));
        return fromLightweightUInt256(result);
    }

    UInt256 mod_mul(const UInt256& a, const UInt256& b) override {
        LightweightECC::UInt256 result = LightweightECC::ModOp::mul(
            toLightweightUInt256(a), toLightweightUInt256(b));
        return fromLightweightUInt256(result);
    }

    UInt256 mod_inv(const UInt256& a) override {
        LightweightECC::UInt256 result = LightweightECC::ModOp::inv(toLightweightUInt256(a));
        return fromLightweightUInt256(result);
    }

    UInt256 mod_pow(const UInt256& base, const UInt256& exp) override {
        LightweightECC::UInt256 result = LightweightECC::ModOp::pow_mod(
            toLightweightUInt256(base), toLightweightUInt256(exp));
        return fromLightweightUInt256(result);
    }

    bool is_on_curve(const ECPoint& P) override {
        if (P.is_infinity()) return true;

        // y^2 = x^3 + ax + b
        UInt256 y2 = mod_mul(P.y, P.y);
        UInt256 x3 = mod_mul(mod_mul(P.x, P.x), P.x);
        UInt256 ax = mod_mul(params_.a, P.x);
        UInt256 right = mod_add(mod_add(x3, ax), params_.b);

        return y2 == right;
    }

    ECPoint get_generator() override {
        return params_.G;
    }

    UInt256 get_order() override {
        return params_.n;
    }

    std::string get_implementation_name() const override {
        return "LightweightECC_CPU";
    }

    bool supports_gpu() const override {
        return false;
    }

    double get_performance_score() const override {
        return 1.0; // 基准性能分数
    }
};
#endif

// ============================================================================
// 工厂实现
// ============================================================================

std::unique_ptr<IECCOperations> ECCFactory::create(const ECCConfig& config) {
    switch (config.backend) {
#ifdef WITH_LIGHTWEIGHT_ECC
        case ECCBackend::CPU_OPTIMIZED:
            return std::make_unique<LightweightECCAdapter>();
#endif
        case ECCBackend::CPU_GENERIC:
            // 返回通用CPU实现
            return nullptr; // 暂时未实现
        case ECCBackend::AUTO:
            // 自动选择最佳实现
#ifdef WITH_LIGHTWEIGHT_ECC
            return std::make_unique<LightweightECCAdapter>();
#else
            return nullptr;
#endif
        default:
            return nullptr;
    }
}

std::vector<ECCBackend> ECCFactory::get_available_backends() {
    std::vector<ECCBackend> backends;

#ifdef WITH_LIGHTWEIGHT_ECC
    backends.push_back(ECCBackend::CPU_OPTIMIZED);
#endif

    return backends;
}

std::string ECCFactory::get_backend_description(ECCBackend backend) {
    switch (backend) {
        case ECCBackend::CPU_GENERIC:
            return "Generic CPU implementation";
        case ECCBackend::CPU_OPTIMIZED:
            return "Optimized CPU implementation (LightweightECC)";
        case ECCBackend::CUDA_GECC:
            return "CUDA implementation (gECC)";
        case ECCBackend::CUDA_KEYHUNT:
            return "CUDA implementation (KeyHunt)";
        case ECCBackend::AUTO:
            return "Automatic backend selection";
        default:
            return "Unknown backend";
    }
}

} // namespace UnifiedECC