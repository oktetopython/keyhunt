#ifndef LIGHTWEIGHT_ECC_SECP256K1_H
#define LIGHTWEIGHT_ECC_SECP256K1_H

#include "uint256.h"
#include "point.h"
#include "curve_params.h"
#include "ec_op.h"
#include <vector>

namespace LightweightECC {

// secp256k1专用类 - 便捷接口
class Secp256k1 {
public:
    // 初始化secp256k1曲线
    static void init();
    
    // 基本椭圆曲线运算
    static Point scalar_mul(const UInt256& k);
    static Point scalar_mul(const UInt256& k, const Point& P);
    static Point point_add(const Point& P, const Point& Q);
    static Point point_double(const Point& P);
    
    // 密钥对生成
    static Point generate_public_key(const UInt256& private_key);
    static bool is_valid_private_key(const UInt256& private_key);
    
    // 批量运算
    static std::vector<Point> batch_generate_public_keys(const std::vector<UInt256>& private_keys);
    
    // 获取曲线参数
    static const CurveParams& get_params();
    static Point get_generator();
    static const UInt256& get_order();
    static const UInt256& get_prime();
    
    // 验证函数
    static bool is_on_curve(const Point& P);
    static bool is_valid_point(const Point& P);
    
    // 工具函数
    static bool is_initialized();
    
    // 性能测试
    static double benchmark_scalar_mul(int iterations = 1000);
    static void print_performance_info();
    
private:
    static bool initialized;
    static CurveParams params;
    
    // 内部辅助函数
    static void ensure_initialized();
};

} // namespace LightweightECC

#endif // LIGHTWEIGHT_ECC_SECP256K1_H
