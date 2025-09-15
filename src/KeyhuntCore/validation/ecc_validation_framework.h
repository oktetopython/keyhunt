/**
 * @file ecc_validation_framework.h
 * @brief Header for comprehensive ECC validation framework with million-scale testing
 * @author KeyhuntCUDA Team
 * 
 * T039: Create comprehensive ECC validation framework with million-scale testing capability
 * 
 * Provides unified validation framework that integrates all ECC components:
 * - Million-scale mathematical property validation
 * - CPU/GPU consistency testing with <1e-10 precision
 * - Performance benchmarking and regression testing
 * - Scientific reporting with statistical analysis
 * - Integration testing across all ECC subsystems
 */

#pragma once

#include "../ecc/secp256k1.h"
#include "../ecc/secp256k1_unified.h"
#include "../ecc/secp256k1_cpu_enhanced.h"
#include "../ecc/secp256k1_point_optimized.h"
#include "../ecc/secp256k1_math_optimized.h"
#include "../ecc/gpu_memory_manager.h"
#include "../crypto/gpu_random.h"
#include "../models/ValidationReport.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>
#include <thread>
#include <mutex>
#include <future>

namespace keyhunt {
namespace ecc {
namespace validation {

/**
 * @brief Validation test categories for comprehensive coverage
 */
enum class ValidationCategory {
    MATHEMATICAL_PROPERTIES,    // Group theory, field operations
    CPU_GPU_CONSISTENCY,       // Cross-platform validation
    PERFORMANCE_BENCHMARKS,    // Speed and efficiency testing
    MEMORY_MANAGEMENT,         // Memory allocation and optimization
    STATISTICAL_ANALYSIS,      // Random testing and distribution
    INTEGRATION_TESTING,       // End-to-end system validation
    REGRESSION_TESTING,        // Performance regression detection
    STRESS_TESTING,           // High-load and edge case testing
    SECURITY_VALIDATION       // Cryptographic security properties
};

/**
 * @brief Validation precision levels for different test requirements
 */
enum class ValidationPrecision {
    FAST,           // Basic validation, ~1000 samples
    STANDARD,       // Normal validation, ~10,000 samples
    THOROUGH,       // Comprehensive validation, ~100,000 samples
    SCIENTIFIC,     // Million-scale validation, ~1,000,000 samples
    EXHAUSTIVE      // Maximum validation, ~10,000,000 samples
};

/**
 * @brief Individual test result structure
 */
struct ValidationTestResult {
    std::string test_name;
    ValidationCategory category;
    bool passed;
    double precision_achieved;      // Actual precision (e.g., 1e-12)
    double performance_score;       // Operations per second or similar metric
    size_t samples_tested;         // Number of test samples
    std::chrono::milliseconds execution_time;
    std::string error_message;     // Empty if passed
    std::vector<double> metrics;   // Additional test-specific metrics
    
    ValidationTestResult() 
        : passed(false), precision_achieved(0.0), performance_score(0.0),
          samples_tested(0), execution_time(0) {}
};

/**
 * @brief Comprehensive validation report aggregating all test results
 */
struct ComprehensiveValidationReport {
    std::vector<ValidationTestResult> test_results;
    std::map<ValidationCategory, size_t> tests_per_category;
    std::map<ValidationCategory, size_t> passed_per_category;
    
    // Overall statistics
    size_t total_tests;
    size_t total_passed;
    double overall_pass_rate;
    double minimum_precision;
    double average_precision;
    size_t total_samples_tested;
    std::chrono::milliseconds total_execution_time;
    
    // Performance metrics
    double cpu_operations_per_second;
    double gpu_operations_per_second;
    double gpu_speedup_factor;
    double memory_efficiency_score;
    
    // System information
    std::string cuda_device_name;
    std::string cuda_compute_capability;
    size_t total_gpu_memory;
    std::string cpu_info;
    std::string test_timestamp;
    
    ComprehensiveValidationReport() 
        : total_tests(0), total_passed(0), overall_pass_rate(0.0),
          minimum_precision(1e-15), average_precision(0.0), total_samples_tested(0),
          total_execution_time(0), cpu_operations_per_second(0.0),
          gpu_operations_per_second(0.0), gpu_speedup_factor(1.0),
          memory_efficiency_score(0.0) {}
};

/**
 * @brief Configuration for validation framework execution
 */
struct ValidationFrameworkConfig {
    ValidationPrecision precision_level;
    std::vector<ValidationCategory> enabled_categories;
    bool parallel_execution;        // Run tests in parallel when possible
    bool generate_detailed_report;  // Include detailed statistics
    bool save_intermediate_results; // Save results during execution
    bool enable_regression_baseline; // Compare against baseline
    double precision_threshold;     // Minimum required precision (e.g., 1e-10)
    size_t max_concurrent_threads;  // Maximum parallel test threads
    std::string output_directory;   // Directory for reports and logs
    
    ValidationFrameworkConfig()
        : precision_level(ValidationPrecision::STANDARD),
          parallel_execution(true), generate_detailed_report(true),
          save_intermediate_results(true), enable_regression_baseline(false),
          precision_threshold(1e-10), max_concurrent_threads(std::thread::hardware_concurrency()),
          output_directory("validation_results") {}
};

/**
 * @brief Main ECC validation framework class
 */
class ECCValidationFramework {
public:
    explicit ECCValidationFramework(const ValidationFrameworkConfig& config = ValidationFrameworkConfig());
    ~ECCValidationFramework();
    
    // Framework initialization and cleanup
    bool initialize(int device_id = 0);
    void cleanup();
    
    // Main validation execution
    ComprehensiveValidationReport run_comprehensive_validation();
    ComprehensiveValidationReport run_category_validation(ValidationCategory category);
    ValidationTestResult run_single_test(const std::string& test_name);
    
    // Mathematical property validation tests
    ValidationTestResult validate_field_operations(size_t sample_count = 100000);
    ValidationTestResult validate_group_properties(size_t sample_count = 100000);
    ValidationTestResult validate_scalar_multiplication(size_t sample_count = 100000);
    ValidationTestResult validate_point_addition_properties(size_t sample_count = 100000);
    ValidationTestResult validate_endomorphism_properties(size_t sample_count = 50000);
    ValidationTestResult validate_curve_equation(size_t sample_count = 100000);
    
    // CPU/GPU consistency validation tests
    ValidationTestResult validate_modular_arithmetic_consistency(size_t sample_count = 100000);
    ValidationTestResult validate_point_operations_consistency(size_t sample_count = 100000);
    ValidationTestResult validate_scalar_multiplication_consistency(size_t sample_count = 50000);
    ValidationTestResult validate_random_generation_consistency(size_t sample_count = 100000);
    ValidationTestResult validate_memory_operations_consistency(size_t sample_count = 50000);
    
    // Performance benchmark tests
    ValidationTestResult benchmark_modular_arithmetic_performance(size_t operation_count = 1000000);
    ValidationTestResult benchmark_point_operations_performance(size_t operation_count = 500000);
    ValidationTestResult benchmark_scalar_multiplication_performance(size_t operation_count = 100000);
    ValidationTestResult benchmark_memory_management_performance(size_t allocation_count = 10000);
    ValidationTestResult benchmark_random_generation_performance(size_t generation_count = 1000000);
    
    // Statistical analysis tests
    ValidationTestResult validate_random_distribution_quality(size_t sample_count = 1000000);
    ValidationTestResult validate_private_key_distribution(size_t key_count = 1000000);
    ValidationTestResult validate_point_distribution(size_t point_count = 500000);
    ValidationTestResult analyze_performance_statistics(size_t measurement_count = 10000);
    
    // Integration tests
    ValidationTestResult validate_end_to_end_key_generation(size_t key_count = 100000);
    ValidationTestResult validate_multi_gpu_coordination(size_t operation_count = 100000);
    ValidationTestResult validate_memory_efficiency_integration(size_t allocation_cycles = 1000);
    ValidationTestResult validate_unified_interface_integration(size_t operation_count = 100000);
    
    // Stress testing
    ValidationTestResult stress_test_continuous_operation(std::chrono::minutes duration = std::chrono::minutes(5));
    ValidationTestResult stress_test_memory_pressure(size_t memory_pressure_gb = 4);
    ValidationTestResult stress_test_concurrent_operations(size_t concurrent_threads = 16);
    ValidationTestResult stress_test_edge_cases(size_t edge_case_count = 50000);
    
    // Regression testing
    bool save_performance_baseline(const std::string& baseline_name);
    ValidationTestResult compare_against_baseline(const std::string& baseline_name);
    ValidationTestResult detect_performance_regressions(double threshold_percent = 5.0);
    
    // Report generation and analysis
    void generate_detailed_report(const ComprehensiveValidationReport& report,
                                 const std::string& filename = "");
    void generate_summary_report(const ComprehensiveValidationReport& report,
                                const std::string& filename = "");
    void generate_performance_charts(const ComprehensiveValidationReport& report,
                                    const std::string& directory = "");
    void generate_statistical_analysis(const ComprehensiveValidationReport& report,
                                      const std::string& filename = "");
    
    // Configuration and status
    ValidationFrameworkConfig get_current_config() const { return config_; }
    void update_config(const ValidationFrameworkConfig& new_config);
    bool is_initialized() const { return initialized_; }
    int get_device_id() const { return device_id_; }
    
    // Progress monitoring
    struct ValidationProgress {
        size_t current_test_index;
        size_t total_tests;
        std::string current_test_name;
        ValidationCategory current_category;
        double completion_percentage;
        std::chrono::milliseconds elapsed_time;
        std::chrono::milliseconds estimated_remaining;
    };
    
    ValidationProgress get_current_progress() const;
    void set_progress_callback(std::function<void(const ValidationProgress&)> callback);

private:
    ValidationFrameworkConfig config_;
    bool initialized_;
    int device_id_;
    
    // Component instances for testing
    std::unique_ptr<unified::UnifiedECCInterface> unified_interface_;
    std::unique_ptr<cpu::EnhancedSecp256k1> cpu_reference_;
    std::unique_ptr<gpu::optimized::OptimizedPointOperations> gpu_point_ops_;
    std::unique_ptr<gpu::optimized::OptimizedModularArithmetic> gpu_arithmetic_;
    std::unique_ptr<gpu::Secp256k1MemoryManager> memory_manager_;
    std::unique_ptr<crypto::gpu::GPURandomGenerator> random_generator_;
    
    // Validation state and progress tracking
    mutable std::mutex progress_mutex_;
    ValidationProgress current_progress_;
    std::function<void(const ValidationProgress&)> progress_callback_;
    
    // Performance baseline storage
    std::map<std::string, ComprehensiveValidationReport> performance_baselines_;
    
    // Test execution helpers
    template<typename TestFunc>
    ValidationTestResult execute_test_with_timing(const std::string& test_name,
                                                 ValidationCategory category,
                                                 TestFunc test_function);
    
    template<typename TestFunc>
    std::vector<ValidationTestResult> execute_parallel_tests(
        const std::vector<std::pair<std::string, TestFunc>>& tests,
        ValidationCategory category);
    
    // Mathematical validation helpers
    bool validate_field_element_properties(const BigInt256& a, const BigInt256& b, 
                                          double& max_error);
    bool validate_group_operation_properties(const Point& p1, const Point& p2, 
                                           const Point& p3, double& max_error);
    bool validate_scalar_properties(const BigInt256& scalar, const Point& point,
                                   double& max_error);
    
    // Consistency validation helpers
    double compare_cpu_gpu_results(const std::vector<BigInt256>& cpu_results,
                                  const std::vector<BigInt256>& gpu_results);
    double compare_point_results(const std::vector<Point>& cpu_results,
                                const std::vector<Point>& gpu_results);
    
    // Statistical analysis helpers
    double calculate_chi_square_statistic(const std::vector<uint64_t>& data);
    double calculate_kolmogorov_smirnov_statistic(const std::vector<double>& data);
    std::vector<double> analyze_distribution_properties(const std::vector<uint64_t>& data);
    
    // Performance analysis helpers
    double calculate_performance_score(double operations_per_second, 
                                      double reference_performance);
    void collect_system_information(ComprehensiveValidationReport& report);
    void update_progress(const std::string& test_name, ValidationCategory category,
                        size_t current_index, size_t total_tests);
    
    // Report generation helpers
    std::string generate_test_summary_table(const std::vector<ValidationTestResult>& results);
    std::string generate_performance_comparison_table(const ComprehensiveValidationReport& report);
    std::string generate_statistical_summary(const ComprehensiveValidationReport& report);
    void save_report_to_file(const std::string& content, const std::string& filename);
    
    // Baseline management
    void save_baseline_to_file(const ComprehensiveValidationReport& report,
                              const std::string& baseline_name);
    bool load_baseline_from_file(ComprehensiveValidationReport& report,
                               const std::string& baseline_name);
};

/**
 * @brief Specialized validation test suites for specific ECC components
 */
namespace test_suites {
    
    /**
     * @brief Mathematical properties validation suite
     */
    class MathematicalValidationSuite {
    public:
        static std::vector<ValidationTestResult> run_field_arithmetic_tests(
            ECCValidationFramework* framework, size_t samples_per_test = 100000);
        
        static std::vector<ValidationTestResult> run_elliptic_curve_tests(
            ECCValidationFramework* framework, size_t samples_per_test = 100000);
        
        static std::vector<ValidationTestResult> run_scalar_multiplication_tests(
            ECCValidationFramework* framework, size_t samples_per_test = 50000);
        
        static std::vector<ValidationTestResult> run_endomorphism_tests(
            ECCValidationFramework* framework, size_t samples_per_test = 50000);
    };
    
    /**
     * @brief Performance benchmarking suite
     */
    class PerformanceBenchmarkSuite {
    public:
        static std::vector<ValidationTestResult> run_arithmetic_benchmarks(
            ECCValidationFramework* framework, size_t operations_per_test = 1000000);
        
        static std::vector<ValidationTestResult> run_point_operation_benchmarks(
            ECCValidationFramework* framework, size_t operations_per_test = 500000);
        
        static std::vector<ValidationTestResult> run_memory_benchmarks(
            ECCValidationFramework* framework, size_t allocations_per_test = 10000);
        
        static std::vector<ValidationTestResult> run_scalability_benchmarks(
            ECCValidationFramework* framework, const std::vector<size_t>& batch_sizes);
    };
    
    /**
     * @brief Statistical analysis suite
     */
    class StatisticalAnalysisSuite {
    public:
        static std::vector<ValidationTestResult> run_randomness_tests(
            ECCValidationFramework* framework, size_t samples_per_test = 1000000);
        
        static std::vector<ValidationTestResult> run_distribution_tests(
            ECCValidationFramework* framework, size_t samples_per_test = 1000000);
        
        static std::vector<ValidationTestResult> run_independence_tests(
            ECCValidationFramework* framework, size_t samples_per_test = 500000);
        
        static std::vector<ValidationTestResult> run_uniformity_tests(
            ECCValidationFramework* framework, size_t samples_per_test = 1000000);
    };
    
    /**
     * @brief Integration testing suite
     */
    class IntegrationTestSuite {
    public:
        static std::vector<ValidationTestResult> run_component_integration_tests(
            ECCValidationFramework* framework, size_t operations_per_test = 100000);
        
        static std::vector<ValidationTestResult> run_multi_device_tests(
            ECCValidationFramework* framework, size_t operations_per_test = 100000);
        
        static std::vector<ValidationTestResult> run_memory_coherence_tests(
            ECCValidationFramework* framework, size_t test_cycles = 1000);
        
        static std::vector<ValidationTestResult> run_error_handling_tests(
            ECCValidationFramework* framework, size_t error_scenarios = 100);
    };
}

/**
 * @brief Utilities for validation data generation and analysis
 */
namespace validation_utils {
    
    /**
     * @brief Test data generator for comprehensive validation
     */
    class ValidationDataGenerator {
    public:
        // Generate test vectors for mathematical validation
        static std::vector<BigInt256> generate_field_elements(size_t count, 
                                                             uint64_t seed = 0);
        static std::vector<Point> generate_curve_points(size_t count, 
                                                       uint64_t seed = 0);
        static std::vector<BigInt256> generate_scalars(size_t count, 
                                                      uint64_t seed = 0);
        
        // Generate edge case test data
        static std::vector<BigInt256> generate_edge_case_scalars(size_t count);
        static std::vector<Point> generate_edge_case_points(size_t count);
        static std::vector<BigInt256> generate_boundary_values(size_t count);
        
        // Generate performance test data
        static std::vector<std::pair<BigInt256, BigInt256>> generate_arithmetic_pairs(
            size_t count, uint64_t seed = 0);
        static std::vector<std::pair<BigInt256, Point>> generate_scalar_mult_pairs(
            size_t count, uint64_t seed = 0);
    };
    
    /**
     * @brief Statistical analysis utilities
     */
    class StatisticalAnalyzer {
    public:
        // Distribution analysis
        static double calculate_entropy(const std::vector<uint8_t>& data);
        static double calculate_mean(const std::vector<double>& data);
        static double calculate_variance(const std::vector<double>& data);
        static double calculate_standard_deviation(const std::vector<double>& data);
        
        // Hypothesis testing
        static bool chi_square_uniformity_test(const std::vector<uint64_t>& data,
                                              double significance_level = 0.05);
        static bool kolmogorov_smirnov_test(const std::vector<double>& data,
                                           double significance_level = 0.05);
        static bool runs_test(const std::vector<uint8_t>& data,
                             double significance_level = 0.05);
        
        // Correlation analysis
        static double calculate_autocorrelation(const std::vector<double>& data, 
                                               int lag = 1);
        static std::vector<double> calculate_autocorrelation_sequence(
            const std::vector<double>& data, int max_lag = 100);
    };
    
    /**
     * @brief Performance measurement utilities
     */
    class PerformanceMeasurer {
    public:
        // Timing utilities
        static std::chrono::nanoseconds measure_execution_time(std::function<void()> operation);
        static double measure_operations_per_second(std::function<void()> operation,
                                                   size_t operation_count,
                                                   std::chrono::seconds duration = std::chrono::seconds(5));
        
        // Memory measurement
        static size_t measure_gpu_memory_usage();
        static size_t measure_cpu_memory_usage();
        static double measure_memory_bandwidth(std::function<void()> memory_operation,
                                              size_t bytes_transferred);
        
        // System resource monitoring
        static double measure_gpu_utilization(std::chrono::seconds duration = std::chrono::seconds(1));
        static double measure_cpu_utilization(std::chrono::seconds duration = std::chrono::seconds(1));
    };
}

/**
 * @brief Global validation framework registry and management
 */
class ValidationFrameworkRegistry {
public:
    static ECCValidationFramework* get_instance(int device_id = 0);
    static void set_global_config(const ValidationFrameworkConfig& config);
    static void cleanup_all_instances();
    
    // Global validation execution
    static ComprehensiveValidationReport run_distributed_validation(
        const std::vector<int>& device_ids,
        ValidationPrecision precision = ValidationPrecision::STANDARD);
    
    // Baseline management
    static bool save_global_baseline(const std::string& baseline_name);
    static bool load_global_baseline(const std::string& baseline_name);
    static std::vector<std::string> list_available_baselines();
    
private:
    static std::unordered_map<int, std::unique_ptr<ECCValidationFramework>> instances_;
    static ValidationFrameworkConfig global_config_;
    static std::mutex registry_mutex_;
};

} // namespace validation
} // namespace ecc
} // namespace keyhunt