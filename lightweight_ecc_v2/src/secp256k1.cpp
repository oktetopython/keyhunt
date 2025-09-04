#include "secp256k1.h"
#include "mod_op.h"
#include <stdexcept>
#include <chrono>
#include <iostream>

namespace LightweightECC {

// 静态成员变量定义
bool Secp256k1::initialized = false;
CurveParams Secp256k1::params;

void Secp256k1::init() {
    if (initialized) {
        return;  // 已经初始化过了
    }
    
    // 获取secp256k1参数
    params = get_secp256k1_params();
    
    // 初始化椭圆曲线运算
    ECOp::init(params);
    
    initialized = true;
}

bool Secp256k1::is_initialized() {
    return initialized;
}

void Secp256k1::ensure_initialized() {
    if (!initialized) {
        throw std::runtime_error("Secp256k1 not initialized. Call Secp256k1::init() first.");
    }
}

// 标量乘法（使用生成元）
Point Secp256k1::scalar_mul(const UInt256& k) {
    ensure_initialized();
    return ECOp::scalar_mul(k, params.G);
}

// 标量乘法（指定点）
Point Secp256k1::scalar_mul(const UInt256& k, const Point& P) {
    ensure_initialized();
    return ECOp::scalar_mul(k, P);
}

// 点加法
Point Secp256k1::point_add(const Point& P, const Point& Q) {
    ensure_initialized();
    return ECOp::point_add(P, Q);
}

// 点倍法
Point Secp256k1::point_double(const Point& P) {
    ensure_initialized();
    return ECOp::point_double(P);
}

// 生成公钥
Point Secp256k1::generate_public_key(const UInt256& private_key) {
    ensure_initialized();
    
    if (!is_valid_private_key(private_key)) {
        throw std::runtime_error("Invalid private key");
    }
    
    return scalar_mul(private_key);
}

// 验证私钥有效性
bool Secp256k1::is_valid_private_key(const UInt256& private_key) {
    ensure_initialized();
    
    // 私钥必须在 [1, n-1] 范围内
    if (private_key.is_zero()) {
        return false;
    }
    
    if (private_key >= params.n) {
        return false;
    }
    
    return true;
}

// 批量生成公钥
std::vector<Point> Secp256k1::batch_generate_public_keys(const std::vector<UInt256>& private_keys) {
    ensure_initialized();
    
    std::vector<Point> public_keys;
    public_keys.reserve(private_keys.size());
    
    for (const auto& priv_key : private_keys) {
        if (!is_valid_private_key(priv_key)) {
            throw std::runtime_error("Invalid private key in batch");
        }
        public_keys.push_back(generate_public_key(priv_key));
    }
    
    return public_keys;
}

// 获取曲线参数
const CurveParams& Secp256k1::get_params() {
    ensure_initialized();
    return params;
}

// 获取生成元
Point Secp256k1::get_generator() {
    ensure_initialized();
    return params.G;
}

// 获取曲线阶
const UInt256& Secp256k1::get_order() {
    ensure_initialized();
    return params.n;
}

// 获取素数
const UInt256& Secp256k1::get_prime() {
    ensure_initialized();
    return params.p;
}

// 验证点是否在曲线上
bool Secp256k1::is_on_curve(const Point& P) {
    ensure_initialized();
    return ECOp::is_on_curve(P);
}

// 验证点是否有效
bool Secp256k1::is_valid_point(const Point& P) {
    ensure_initialized();
    return ECOp::is_valid_point(P);
}

// 性能基准测试
double Secp256k1::benchmark_scalar_mul(int iterations) {
    ensure_initialized();
    
    if (iterations <= 0) {
        iterations = 1000;
    }
    
    // 准备测试数据
    std::vector<UInt256> test_scalars;
    test_scalars.reserve(iterations);
    
    for (int i = 0; i < iterations; i++) {
        test_scalars.push_back(UInt256(i + 1, 0, 0, 0));
    }
    
    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 执行标量乘法
    for (const auto& scalar : test_scalars) {
        scalar_mul(scalar);
    }
    
    // 结束计时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // 计算每秒操作数
    double seconds = duration.count() / 1000000.0;
    double ops_per_second = iterations / seconds;
    
    return ops_per_second;
}

// 打印性能信息
void Secp256k1::print_performance_info() {
    ensure_initialized();
    
    std::cout << "=== Secp256k1 性能信息 ===" << std::endl;
    
    // 测试不同迭代次数的性能
    std::vector<int> test_iterations = {100, 500, 1000};
    
    for (int iterations : test_iterations) {
        double ops_per_sec = benchmark_scalar_mul(iterations);
        std::cout << "标量乘法性能 (" << iterations << " 次): " 
                  << ops_per_sec << " ops/sec" << std::endl;
    }
    
    // 测试单次操作时间
    auto start = std::chrono::high_resolution_clock::now();
    scalar_mul(UInt256(12345, 0, 0, 0));
    auto end = std::chrono::high_resolution_clock::now();
    auto single_op_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "单次标量乘法时间: " << single_op_time.count() << " 微秒" << std::endl;
    
    // 内存使用信息
    std::cout << "UInt256 大小: " << sizeof(UInt256) << " 字节" << std::endl;
    std::cout << "Point 大小: " << sizeof(Point) << " 字节" << std::endl;
    std::cout << "CurveParams 大小: " << sizeof(CurveParams) << " 字节" << std::endl;
    
    std::cout << "=========================" << std::endl;
}

} // namespace LightweightECC
