/**
 * @file secp256k1_unified.h
 * @brief Unified CPU/GPU interface for secp256k1 operations with consistent signatures
 * @author KeyhuntCUDA Team
 * 
 * T034: Design unified CPU/GPU interface for ECC operations with consistent function signatures
 * 
 * This header provides a unified interface that abstracts CPU/GPU implementation details,
 * allowing runtime switching between implementations while maintaining identical function
 * signatures for scientific validation and performance optimization.
 */

#pragma once

#include "secp256k1.h"
#include <functional>
#include <chrono>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>

namespace keyhunt {
namespace ecc {
namespace unified {

/**
 * @brief Execution context for operation routing
 */
enum class ExecutionContext {
    CPU,               // Force CPU execution
    GPU,               // Force GPU execution (if available)
    AUTO,              // Automatic selection based on workload
    HYBRID_VALIDATION, // Execute on both CPU and GPU for validation
    BENCHMARK          // Execute on all available backends for comparison
};

/**
 * @brief Performance metrics for operation tracking
 */
struct PerformanceMetrics {
    std::chrono::milliseconds execution_time;
    size_t operations_completed;
    double operations_per_second;
    size_t memory_used_bytes;
    ExecutionContext context_used;
    bool validation_passed;
    double precision_error;
    
    PerformanceMetrics() : execution_time(0), operations_completed(0), 
                          operations_per_second(0.0), memory_used_bytes(0),
                          context_used(ExecutionContext::AUTO), 
                          validation_passed(true), precision_error(0.0) {}
};

/**
 * @brief Operation configuration for fine-grained control
 */
struct OperationConfig {
    ExecutionContext preferred_context;
    bool enable_validation;
    double precision_threshold;
    size_t batch_threshold;
    int gpu_device_id;
    bool enable_profiling;
    
    OperationConfig() : preferred_context(ExecutionContext::AUTO),
                       enable_validation(false), precision_threshold(1e-10),
                       batch_threshold(1000), gpu_device_id(0),
                       enable_profiling(false) {}
};

/**
 * @brief Unified result wrapper with validation information
 */
template<typename T>
struct UnifiedResult {
    T result;
    PerformanceMetrics metrics;
    bool success;
    std::string error_message;
    
    UnifiedResult() : success(false) {}
    UnifiedResult(const T& r) : result(r), success(true) {}
    
    operator bool() const { return success; }
    const T& operator*() const { return result; }
    T& operator*() { return result; }
};

/**
 * @brief Function signature type definitions for unified interface
 */
namespace signatures {
    // Scalar multiplication signature
    using ScalarMultiplyFunc = std::function<Point(const BigInt256&, const Point&)>;
    
    // Point addition signature  
    using PointAddFunc = std::function<Point(const Point&, const Point&)>;
    
    // Point doubling signature
    using PointDoubleFunc = std::function<Point(const Point&)>;
    
    // Public key computation signature
    using ComputePublicKeyFunc = std::function<PublicKey(const PrivateKey&)>;
    
    // Batch scalar multiplication signature
    using BatchScalarMultiplyFunc = std::function<std::vector<Point>(const std::vector<BigInt256>&, const Point&)>;
    
    // Batch public key computation signature
    using BatchComputePublicKeyFunc = std::function<std::vector<PublicKey>(const std::vector<PrivateKey>&)>;
}

/**
 * @brief Abstract backend interface for implementation polymorphism
 */
class IBackend {
public:
    virtual ~IBackend() = default;
    
    virtual bool initialize() = 0;
    virtual void cleanup() = 0;
    virtual bool is_available() const = 0;
    virtual ExecutionContext get_context() const = 0;
    virtual std::string get_name() const = 0;
    
    // Core ECC operations with identical signatures
    virtual Point scalar_multiply(const BigInt256& scalar, const Point& point) = 0;
    virtual Point point_add(const Point& p1, const Point& p2) = 0;
    virtual Point point_double(const Point& p) = 0;
    virtual Point point_negate(const Point& p) = 0;
    
    // Key operations
    virtual PublicKey compute_public_key(const PrivateKey& private_key) = 0;
    virtual bool verify_key_pair(const PrivateKey& priv_key, const PublicKey& pub_key) = 0;
    
    // Batch operations
    virtual std::vector<Point> batch_scalar_multiply(const std::vector<BigInt256>& scalars, const Point& base_point) = 0;
    virtual std::vector<PublicKey> batch_compute_public_keys(const std::vector<PrivateKey>& private_keys) = 0;
    
    // Performance metrics
    virtual PerformanceMetrics get_last_operation_metrics() const = 0;
    virtual void reset_metrics() = 0;
    
    // Validation support
    virtual bool supports_validation() const = 0;
    virtual validation::ConsistencyValidator::ValidationResults validate_operations(size_t test_count) = 0;
};

/**
 * @brief CPU backend implementation wrapper
 */
class CPUBackend : public IBackend {
public:
    CPUBackend();
    virtual ~CPUBackend();
    
    bool initialize() override;
    void cleanup() override;
    bool is_available() const override { return initialized_; }
    ExecutionContext get_context() const override { return ExecutionContext::CPU; }
    std::string get_name() const override { return "CPU Reference"; }
    
    Point scalar_multiply(const BigInt256& scalar, const Point& point) override;
    Point point_add(const Point& p1, const Point& p2) override;
    Point point_double(const Point& p) override;
    Point point_negate(const Point& p) override;
    
    PublicKey compute_public_key(const PrivateKey& private_key) override;
    bool verify_key_pair(const PrivateKey& priv_key, const PublicKey& pub_key) override;
    
    std::vector<Point> batch_scalar_multiply(const std::vector<BigInt256>& scalars, const Point& base_point) override;
    std::vector<PublicKey> batch_compute_public_keys(const std::vector<PrivateKey>& private_keys) override;
    
    PerformanceMetrics get_last_operation_metrics() const override;
    void reset_metrics() override;
    
    bool supports_validation() const override { return true; }
    validation::ConsistencyValidator::ValidationResults validate_operations(size_t test_count) override;

private:
    bool initialized_;
    std::unique_ptr<cpu::Secp256k1> impl_;
    mutable PerformanceMetrics last_metrics_;
    mutable std::mutex metrics_mutex_;
    
    void update_metrics(std::chrono::milliseconds duration, size_t ops_count, size_t memory_used);
};

#ifdef __CUDACC__
/**
 * @brief GPU backend implementation wrapper
 */
class GPUBackend : public IBackend {
public:
    GPUBackend(int device_id = 0);
    virtual ~GPUBackend();
    
    bool initialize() override;
    void cleanup() override;
    bool is_available() const override;
    ExecutionContext get_context() const override { return ExecutionContext::GPU; }
    std::string get_name() const override;
    
    Point scalar_multiply(const BigInt256& scalar, const Point& point) override;
    Point point_add(const Point& p1, const Point& p2) override;
    Point point_double(const Point& p) override;
    Point point_negate(const Point& p) override;
    
    PublicKey compute_public_key(const PrivateKey& private_key) override;
    bool verify_key_pair(const PrivateKey& priv_key, const PublicKey& pub_key) override;
    
    std::vector<Point> batch_scalar_multiply(const std::vector<BigInt256>& scalars, const Point& base_point) override;
    std::vector<PublicKey> batch_compute_public_keys(const std::vector<PrivateKey>& private_keys) override;
    
    PerformanceMetrics get_last_operation_metrics() const override;
    void reset_metrics() override;
    
    bool supports_validation() const override { return true; }
    validation::ConsistencyValidator::ValidationResults validate_operations(size_t test_count) override;

private:
    int device_id_;
    bool initialized_;
    std::unique_ptr<gpu::Secp256k1> impl_;
    mutable PerformanceMetrics last_metrics_;
    mutable std::mutex metrics_mutex_;
    std::string device_name_;
    
    void update_metrics(std::chrono::milliseconds duration, size_t ops_count, size_t memory_used);
    bool query_device_info();
};
#endif

/**
 * @brief Unified secp256k1 interface with runtime backend selection
 */
class UnifiedSecp256k1 {
public:
    UnifiedSecp256k1();
    ~UnifiedSecp256k1();
    
    // Initialization and configuration
    bool initialize();
    void cleanup();
    bool is_initialized() const { return initialized_; }
    
    void set_default_config(const OperationConfig& config);
    OperationConfig get_default_config() const { return default_config_; }
    
    // Backend management
    std::vector<std::string> get_available_backends() const;
    bool set_preferred_backend(ExecutionContext context);
    ExecutionContext get_current_backend() const;
    
    // Core operations with unified signatures
    UnifiedResult<Point> scalar_multiply(const BigInt256& scalar, const Point& point,
                                       const OperationConfig* config = nullptr);
    
    UnifiedResult<Point> point_add(const Point& p1, const Point& p2,
                                 const OperationConfig* config = nullptr);
    
    UnifiedResult<Point> point_double(const Point& p,
                                    const OperationConfig* config = nullptr);
    
    UnifiedResult<PublicKey> compute_public_key(const PrivateKey& private_key,
                                              const OperationConfig* config = nullptr);
    
    // Batch operations
    UnifiedResult<std::vector<Point>> batch_scalar_multiply(const std::vector<BigInt256>& scalars, 
                                                          const Point& base_point,
                                                          const OperationConfig* config = nullptr);
    
    UnifiedResult<std::vector<PublicKey>> batch_compute_public_keys(const std::vector<PrivateKey>& private_keys,
                                                                  const OperationConfig* config = nullptr);
    
    // Validation and testing
    bool run_cross_validation(size_t test_count = 10000);
    bool run_performance_benchmark();
    validation::ConsistencyValidator::ValidationResults get_validation_results() const;
    
    // Performance analysis
    std::vector<PerformanceMetrics> get_performance_history() const;
    PerformanceMetrics get_aggregated_metrics() const;
    void clear_metrics_history();
    
    // Scientific validation
    struct ScientificValidationReport {
        bool overall_passed;
        size_t total_tests;
        size_t tests_passed;
        double max_precision_error;
        double average_precision_error;
        std::map<ExecutionContext, PerformanceMetrics> backend_performance;
        std::string detailed_report;
        std::chrono::system_clock::time_point timestamp;
    };
    
    ScientificValidationReport generate_scientific_report();
    bool export_validation_data(const std::string& filename) const;

private:
    bool initialized_;
    OperationConfig default_config_;
    
    // Backend instances
    std::unique_ptr<CPUBackend> cpu_backend_;
#ifdef __CUDACC__
    std::unique_ptr<GPUBackend> gpu_backend_;
#endif
    
    // Performance tracking
    mutable std::vector<PerformanceMetrics> performance_history_;
    mutable std::mutex history_mutex_;
    
    // Validation results
    mutable validation::ConsistencyValidator::ValidationResults last_validation_;
    
    // Backend selection logic
    IBackend* select_backend(const OperationConfig& config) const;
    bool validate_operation_result(const OperationConfig& config, 
                                 IBackend* primary_backend,
                                 std::function<void(IBackend*)> operation) const;
    
    void record_metrics(const PerformanceMetrics& metrics) const;
    
    // Hybrid validation execution
    template<typename ResultType>
    UnifiedResult<ResultType> execute_with_validation(
        std::function<ResultType(IBackend*)> operation,
        const OperationConfig& config) const;
};

/**
 * @brief Factory for creating optimized unified instances
 */
class UnifiedFactory {
public:
    static std::unique_ptr<UnifiedSecp256k1> create_optimized_instance();
    static std::unique_ptr<UnifiedSecp256k1> create_validation_instance();
    static std::unique_ptr<UnifiedSecp256k1> create_performance_instance();
    
    static OperationConfig get_default_cpu_config();
    static OperationConfig get_default_gpu_config();
    static OperationConfig get_validation_config();
    static OperationConfig get_performance_config();
};

/**
 * @brief Thread-safe singleton for global unified instance
 */
class GlobalUnifiedSecp256k1 {
public:
    static UnifiedSecp256k1& instance();
    static bool initialize_global();
    static void cleanup_global();

private:
    GlobalUnifiedSecp256k1() = default;
    static std::unique_ptr<UnifiedSecp256k1> instance_;
    static std::once_flag init_flag_;
    static std::mutex instance_mutex_;
};

/**
 * @brief High-level convenience functions with automatic backend selection
 */
namespace convenience {
    
    // Simple operations with automatic backend selection
    Point scalar_multiply(const BigInt256& scalar, const Point& point);
    PublicKey compute_public_key(const PrivateKey& private_key);
    std::vector<PublicKey> batch_compute_public_keys(const std::vector<PrivateKey>& private_keys);
    
    // Validation helpers
    bool validate_implementation(size_t test_count = 10000);
    void benchmark_all_backends();
    
    // Configuration helpers
    void prefer_cpu();
    void prefer_gpu();
    void enable_validation();
    void disable_validation();
}

} // namespace unified
} // namespace ecc
} // namespace keyhunt