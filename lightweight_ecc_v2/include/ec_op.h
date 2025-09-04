#ifndef LIGHTWEIGHT_ECC_EC_OP_H
#define LIGHTWEIGHT_ECC_EC_OP_H

#include "uint256.h"
#include "point.h"
#include "curve_params.h"
#include <vector>

namespace LightweightECC {

// 椭圆曲线运算类 - 基于gECC的椭圆曲线算法思想
class ECOp {
public:
    // 初始化椭圆曲线参数
    static void init(const CurveParams& params);
    
    // 基本椭圆曲线运算
    static Point point_add(const Point& P, const Point& Q);
    static Point point_double(const Point& P);
    static Point scalar_mul(const UInt256& k, const Point& P);
    
    // 批量运算（性能优化）
    static std::vector<Point> batch_scalar_mul(const std::vector<UInt256>& scalars, const Point& P);
    
    // 工具函数
    static bool is_initialized();
    static const CurveParams& get_params();
    static Point get_generator();
    
    // 验证函数
    static bool is_on_curve(const Point& P);
    static bool is_valid_point(const Point& P);
    
private:
    static CurveParams curve;
    static bool initialized;
    
    // 内部椭圆曲线运算（仿射坐标）
    static Point point_add_affine(const Point& P, const Point& Q);
    static Point point_double_affine(const Point& P);
    
    // 标量乘法算法
    static Point scalar_mul_binary(const UInt256& k, const Point& P);
    static Point scalar_mul_window(const UInt256& k, const Point& P, int window_size);
    static Point scalar_mul_sliding_window(const UInt256& k, const Point& P);
    
    // 预计算表
    static std::vector<Point> compute_precompute_table(const Point& P, int size);
    
    // 辅助函数
    static bool points_equal(const Point& P, const Point& Q);
    static Point point_negate(const Point& P);
};

} // namespace LightweightECC

#endif // LIGHTWEIGHT_ECC_EC_OP_H
