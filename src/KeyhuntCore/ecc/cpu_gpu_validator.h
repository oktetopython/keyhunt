/**
 * @file cpu_gpu_validator.h
 * @brief Header for CPU/GPU consistency validation framework
 * @author KeyhuntCUDA Team
 * 
 * T038: Complete CPU/GPU consistency validation testing for ECC operations
 * 
 * Provides comprehensive validation infrastructure to ensure GPU implementations
 * match CPU reference with <1e-10 precision. Uses libsecp256k1 as authoritative
 * reference for all elliptic curve operations.
 */

#pragma once

#include "secp256k1.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>
#include <thread>
#include <mutex>
#include <future>
#include <random>

// Include libsecp256k1 for CPU reference
extern "C" {
#include <secp256k1.h>
#include <secp256k1_extrakeys.h>
}

namespace keyhunt {
namespace ecc {
namespace validation {

/**
 * @brief Validation test result for individual operations
 */
struct ValidationResult {
    bool passed;
    double max_error;
    double avg_error;
    size_t samples_tested;
    std::chrono::milliseconds execution_time;
    std::string error_message;
    
    ValidationResult() : passed(false), max_error(0.0), avg_error(0.0), 
                        samples_tested(0), execution_time(0) {}
};

/**
 * @brief Test categories for comprehensive validation
 */
enum class TestCategory {
    SCALAR_MULTIPLICATION,
    POINT_ADDITION,
    POINT_DOUBLING,
    MODULAR_ARITHMETIC,
    FIELD_OPERATIONS,
    EDGE_CASES,
    RANDOM_OPERATIONS
};

/**
 * @brief Precision levels for different validation requirements
 */
enum class PrecisionLevel {
    BASIC,      // 1e-6 precision, 1K samples
    STANDARD,   // 1e-8 precision, 10K samples
    SCIENTIFIC, // 1e-10 precision, 100K samples
    EXHAUSTIVE  // 1e-12 precision, 1M samples
};

/**
 * @brief Main CPU/GPU validation class
 */
class CPUGPUValidator {
public:
    CPUGPUValidator();
    ~CPUGPUValidator();
    
    // Initialization
    bool initialize(int gpu_device_id = 0);
    void cleanup();
    
    // Core validation methods
    ValidationResult validate_scalar_multiplication(size_t sample_count = 100000,
                                                   PrecisionLevel precision = PrecisionLevel::SCIENTIFIC);
    
    ValidationResult validate_point_addition(size_t sample_count = 100000,
                                            PrecisionLevel precision = PrecisionLevel::SCIENTIFIC);
    
    ValidationResult validate_point_doubling(size_t sample_count = 50000,
                                            PrecisionLevel precision = PrecisionLevel::SCIENTIFIC);
    
    ValidationResult validate_modular_arithmetic(size_t sample_count = 200000,
                                                PrecisionLevel precision = PrecisionLevel::SCIENTIFIC);
    
    ValidationResult validate_field_operations(size_t sample_count = 150000,
                                              PrecisionLevel precision = PrecisionLevel::SCIENTIFIC);
    
    ValidationResult validate_edge_cases(PrecisionLevel precision = PrecisionLevel::SCIENTIFIC);
    
    // Comprehensive validation
    struct ComprehensiveResults {
        std::vector<ValidationResult> test_results;
        std::map<TestCategory, ValidationResult> category_results;
        bool overall_passed;
        double minimum_precision_achieved;
        size_t total_samples_tested;
        std::chrono::milliseconds total_execution_time;
    };
    
    ComprehensiveResults run_comprehensive_validation(PrecisionLevel precision = PrecisionLevel::SCIENTIFIC);
    
    // Performance comparison
    struct PerformanceComparison {
        double cpu_operations_per_second;
        double gpu_operations_per_second;
        double gpu_speedup_factor;
        double precision_overhead_percent;
    };
    
    PerformanceComparison benchmark_cpu_vs_gpu(TestCategory category, size_t operation_count = 50000);
    
    // Configuration
    void set_precision_threshold(double threshold) { precision_threshold_ = threshold; }
    void set_random_seed(uint64_t seed) { rng_.seed(seed); }
    void enable_detailed_logging(bool enable) { detailed_logging_ = enable; }
    
    // Statistics
    struct ValidationStatistics {
        size_t total_tests_run;
        size_t total_tests_passed;
        double overall_pass_rate;
        double best_precision_achieved;
        double worst_precision_achieved;
        std::chrono::milliseconds total_validation_time;
    };
    
    ValidationStatistics get_statistics() const;
    void reset_statistics();
    
private:
    // Core validation infrastructure
    secp256k1_context* secp256k1_ctx_;
    std::mt19937_64 rng_;
    
    // Configuration
    double precision_threshold_;
    bool detailed_logging_;
    bool initialized_;
    int gpu_device_id_;
    
    // Statistics tracking
    mutable ValidationStatistics stats_;
    mutable std::mutex stats_mutex_;
    
    // GPU memory management
    Point* d_points_;
    BigInt256* d_scalars_;
    Point* d_results_;
    size_t allocated_point_count_;
    size_t allocated_scalar_count_;
    
    // CUDA streams for performance
    cudaStream_t validation_stream_;
    cudaStream_t memory_stream_;
    
    // Helper methods
    bool allocate_gpu_memory(size_t max_points, size_t max_scalars);
    void free_gpu_memory();
    
    // Test data generation
    std::vector<BigInt256> generate_random_scalars(size_t count);
    std::vector<Point> generate_random_points(size_t count);
    std::vector<BigInt256> generate_edge_case_scalars();
    std::vector<Point> generate_edge_case_points();
    
    // CPU reference implementations using libsecp256k1
    Point cpu_scalar_multiply(const BigInt256& scalar, const Point& point);
    Point cpu_point_add(const Point& p1, const Point& p2);
    Point cpu_point_double(const Point& point);
    BigInt256 cpu_modular_add(const BigInt256& a, const BigInt256& b);
    BigInt256 cpu_modular_multiply(const BigInt256& a, const BigInt256& b);
    
    // GPU implementations (calling existing CUDA kernels)
    std::vector<Point> gpu_scalar_multiply_batch(const std::vector<BigInt256>& scalars,
                                               const std::vector<Point>& points);
    std::vector<Point> gpu_point_add_batch(const std::vector<Point>& p1_array,
                                          const std::vector<Point>& p2_array);
    std::vector<Point> gpu_point_double_batch(const std::vector<Point>& points);
    
    // Precision calculation
    double calculate_point_error(const Point& cpu_result, const Point& gpu_result);
    double calculate_scalar_error(const BigInt256& cpu_result, const BigInt256& gpu_result);
    
    // Utility methods
    void log_validation_details(const std::string& test_name, const ValidationResult& result);
    bool is_point_valid(const Point& point);
    bool is_scalar_valid(const BigInt256& scalar);
    
    // Conversion utilities
    Point libsecp256k1_to_point(const secp256k1_pubkey& pubkey);
    secp256k1_pubkey point_to_libsecp256k1(const Point& point);
    void bigint_to_bytes(const BigInt256& bigint, uint8_t bytes[32]);
    BigInt256 bytes_to_bigint(const uint8_t bytes[32]);
};

/**
 * @brief Test suite runner for automated validation
 */
class ValidationTestRunner {
public:
    ValidationTestRunner(std::shared_ptr<CPUGPUValidator> validator);
    ~ValidationTestRunner();
    
    // Test suite execution
    struct TestSuiteResults {
        std::vector<ValidationResult> individual_results;
        CPUGPUValidator::ComprehensiveResults comprehensive_results;
        std::vector<CPUGPUValidator::PerformanceComparison> performance_comparisons;
        bool certification_passed;
        std::string certification_summary;
    };
    
    TestSuiteResults run_full_test_suite(PrecisionLevel precision = PrecisionLevel::SCIENTIFIC);
    TestSuiteResults run_regression_tests();
    TestSuiteResults run_stress_tests();
    
    // Report generation
    void generate_html_report(const TestSuiteResults& results, const std::string& filename);
    void generate_json_report(const TestSuiteResults& results, const std::string& filename);
    void generate_csv_report(const TestSuiteResults& results, const std::string& filename);
    
    // Certification
    bool is_system_certified(const TestSuiteResults& results);
    std::string get_certification_status(const TestSuiteResults& results);
    
private:
    std::shared_ptr<CPUGPUValidator> validator_;
    std::vector<std::function<ValidationResult()>> test_functions_;
    
    void register_all_tests();
    void setup_parallel_execution();
    
    // Report generation helpers
    std::string format_validation_result(const ValidationResult& result);
    std::string format_performance_comparison(const CPUGPUValidator::PerformanceComparison& comparison);
};

/**
 * @brief Factory for creating validation instances
 */
class ValidationFactory {
public:
    static std::shared_ptr<CPUGPUValidator> create_validator(int gpu_device_id = 0);
    static std::shared_ptr<ValidationTestRunner> create_test_runner(int gpu_device_id = 0);
    
    // Predefined test configurations
    static PrecisionLevel get_precision_for_certification();
    static size_t get_sample_count_for_precision(PrecisionLevel precision);
    static double get_threshold_for_precision(PrecisionLevel precision);
};

/**
 * @brief Global validation registry for multi-GPU environments
 */
class ValidationRegistry {
public:
    static ValidationRegistry& instance();
    
    void register_validator(int device_id, std::shared_ptr<CPUGPUValidator> validator);
    std::shared_ptr<CPUGPUValidator> get_validator(int device_id);
    
    // Multi-GPU validation
    struct MultiGPUResults {
        std::map<int, CPUGPUValidator::ComprehensiveResults> device_results;
        bool all_devices_passed;
        std::vector<int> failed_devices;
        double worst_precision_across_devices;
    };
    
    MultiGPUResults validate_all_devices(PrecisionLevel precision = PrecisionLevel::SCIENTIFIC);
    
    void cleanup_all_validators();
    
private:
    ValidationRegistry() = default;
    std::map<int, std::shared_ptr<CPUGPUValidator>> validators_;
    std::mutex registry_mutex_;
};

} // namespace validation
} // namespace ecc
} // namespace keyhunt