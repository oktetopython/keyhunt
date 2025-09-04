#ifndef UNIFIED_ECC_INTERFACE_H
#define UNIFIED_ECC_INTERFACE_H

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace UnifiedECC {

// ============================================================================
// 基础数据类型定义
// ============================================================================

// 256位无符号整数
class UInt256 {
public:
    UInt256();
    UInt256(uint64_t a, uint64_t b, uint64_t c, uint64_t d);
    UInt256(const std::string& hex);
    UInt256(const std::vector<uint8_t>& bytes);

    // 运算符重载
    UInt256 operator+(const UInt256& other) const;
    UInt256 operator-(const UInt256& other) const;
    UInt256 operator*(const UInt256& other) const;
    UInt256 operator/(const UInt256& other) const;
    UInt256 operator%(const UInt256& other) const;

    bool operator==(const UInt256& other) const;
    bool operator<(const UInt256& other) const;

    // 工具函数
    bool is_zero() const;
    bool is_odd() const;
    std::string to_hex() const;
    std::vector<uint8_t> to_bytes() const;

    // 内部数据访问
    uint64_t limbs[4];
};

// 椭圆曲线点
class ECPoint {
public:
    ECPoint();
    ECPoint(const UInt256& x, const UInt256& y);
    explicit ECPoint(bool infinity);

    bool is_infinity() const;
    std::string to_string() const;

    UInt256 x, y;
    bool infinity;
};

// ECC参数
struct ECCParams {
    UInt256 p;        // 素数模数
    UInt256 a;        // 曲线参数a
    UInt256 b;        // 曲线参数b
    ECPoint G;        // 基点
    UInt256 n;        // 基点阶
    std::string name; // 曲线名称
};

// ============================================================================
// 核心ECC接口定义
// ============================================================================

class IECCOperations {
public:
    virtual ~IECCOperations() = default;

    // 初始化
    virtual bool initialize(const ECCParams& params) = 0;

    // 点运算
    virtual ECPoint point_add(const ECPoint& P, const ECPoint& Q) = 0;
    virtual ECPoint point_double(const ECPoint& P) = 0;
    virtual ECPoint scalar_mul(const UInt256& k, const ECPoint& P) = 0;

    // 批量运算
    virtual std::vector<ECPoint> batch_scalar_mul(const std::vector<UInt256>& scalars,
                                                 const std::vector<ECPoint>& points) = 0;

    // 模运算
    virtual UInt256 mod_add(const UInt256& a, const UInt256& b) = 0;
    virtual UInt256 mod_sub(const UInt256& a, const UInt256& b) = 0;
    virtual UInt256 mod_mul(const UInt256& a, const UInt256& b) = 0;
    virtual UInt256 mod_inv(const UInt256& a) = 0;
    virtual UInt256 mod_pow(const UInt256& base, const UInt256& exp) = 0;

    // 实用函数
    virtual bool is_on_curve(const ECPoint& P) = 0;
    virtual ECPoint get_generator() = 0;
    virtual UInt256 get_order() = 0;

    // 性能信息
    virtual std::string get_implementation_name() const = 0;
    virtual bool supports_gpu() const = 0;
    virtual double get_performance_score() const = 0;
};

// ============================================================================
// 工厂类和配置
// ============================================================================

enum class ECCBackend {
    CPU_GENERIC,      // 通用CPU实现
    CPU_OPTIMIZED,    // 优化的CPU实现
    CUDA_GECC,        // gECC CUDA实现
    CUDA_KEYHUNT,     // KeyHunt CUDA实现
    AUTO              // 自动选择最佳实现
};

struct ECCConfig {
    ECCBackend backend = ECCBackend::AUTO;
    bool enable_gpu = true;
    bool enable_batch_ops = true;
    int gpu_device_id = 0;
    std::string curve_name = "secp256k1";
    bool enable_performance_monitoring = false;
};

class ECCFactory {
public:
    static std::unique_ptr<IECCOperations> create(const ECCConfig& config);
    static std::vector<ECCBackend> get_available_backends();
    static std::string get_backend_description(ECCBackend backend);
};

// ============================================================================
// 适配器基类
// ============================================================================

template<typename T>
class ECCAdapter : public IECCOperations {
protected:
    T* implementation_;
    ECCParams params_;

public:
    ECCAdapter(T* impl = nullptr) : implementation_(impl) {}
    virtual ~ECCAdapter() = default;

    bool initialize(const ECCParams& params) override {
        params_ = params;
        return do_initialize(params);
    }

    virtual bool do_initialize(const ECCParams& params) = 0;
};

// ============================================================================
// 预定义曲线参数
// ============================================================================

namespace Curves {
    extern const ECCParams secp256k1;
    extern const ECCParams secp256r1;
    extern const ECCParams ed25519;
}

} // namespace UnifiedECC

#endif // UNIFIED_ECC_INTERFACE_H