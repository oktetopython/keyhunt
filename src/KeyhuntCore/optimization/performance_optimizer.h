/**
 * @file performance_optimizer.h
 * @brief Performance optimization algorithms with adaptive parameter tuning and bottleneck analysis
 * @author KeyhuntCUDA Team
 * 
 * T046: Implement performance optimization algorithms with adaptive parameter tuning and bottleneck analysis
 * 
 * This module provides intelligent performance optimization for the KeyhuntCUDA system:
 * - Real-time bottleneck detection and analysis
 * - Adaptive parameter tuning based on system performance
 * - Hardware-aware optimization strategies
 * - Dynamic load balancing optimization
 * - Memory usage optimization
 * - GPU utilization maximization
 */

#pragma once

#include "../engine/integrated_scanning_engine.h"
#include "../models/GPUConfiguration.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>

namespace keyhunt {
namespace optimization {

/**
 * @brief Performance bottleneck types
 */
enum class BottleneckType {
    GPU_UTILIZATION,        // Low GPU utilization
    MEMORY_BANDWIDTH,       // Memory bandwidth limited
    CPU_PROCESSING,         // CPU processing bottleneck
    IO_OPERATIONS,          // I/O operations bottleneck
    SYNCHRONIZATION,        // Thread synchronization issues
    LOAD_BALANCING,         // Poor load distribution
    MEMORY_ALLOCATION,      // Memory allocation issues
    CACHE_EFFICIENCY,       // Poor cache utilization
    KERNEL_LAUNCH_OVERHEAD, // GPU kernel launch overhead
    DATA_TRANSFER,          // Host-device data transfer
    UNKNOWN                 // Unidentified bottleneck
};

/**
 * @brief Performance bottleneck analysis result
 */
struct BottleneckAnalysis {
    BottleneckType type;
    std::string description;
    double severity_score;          // 0.0 = minor, 1.0 = critical
    std::string recommended_action;
    std::unordered_map<std::string, double> metrics; // Supporting metrics
    std::chrono::system_clock::time_point detected_time;
    
    BottleneckAnalysis() 
        : type(BottleneckType::UNKNOWN)
        , severity_score(0.0)
        , detected_time(std::chrono::system_clock::now())
    {}
};

/**
 * @brief Parameter optimization suggestion
 */
struct OptimizationSuggestion {
    std::string parameter_name;
    std::string current_value;
    std::string suggested_value;
    std::string rationale;
    double expected_improvement;    // Expected performance improvement (0.0-1.0)
    double confidence;              // Confidence in suggestion (0.0-1.0)
    
    OptimizationSuggestion()
        : expected_improvement(0.0)
        , confidence(0.0)
    {}
};

/**
 * @brief Adaptive tuning strategy configuration
 */
struct AdaptiveTuningConfig {
    bool enable_automatic_tuning;      // Enable automatic parameter adjustment
    double tuning_aggressiveness;      // 0.0 = conservative, 1.0 = aggressive
    std::chrono::seconds tuning_interval; // How often to apply tunings
    double minimum_improvement_threshold; // Minimum improvement to apply tuning
    size_t history_window_size;        // Number of measurements to consider
    bool enable_rollback_on_degradation; // Rollback changes if performance degrades
    
    AdaptiveTuningConfig()
        : enable_automatic_tuning(true)
        , tuning_aggressiveness(0.5)
        , tuning_interval(std::chrono::seconds(30))
        , minimum_improvement_threshold(0.05) // 5% improvement
        , history_window_size(10)
        , enable_rollback_on_degradation(true)
    {}
};

/**
 * @brief Performance measurement sample
 */
struct PerformanceSample {
    std::chrono::system_clock::time_point timestamp;
    engine::ScanningEngineMetrics metrics;
    models::GPUConfiguration gpu_config;
    std::unordered_map<std::string, std::string> active_parameters;
    double overall_performance_score;
    
    PerformanceSample()
        : timestamp(std::chrono::system_clock::now())
        , overall_performance_score(0.0)
    {}
};

/**
 * @brief Hardware profile for optimization
 */
struct HardwareProfile {
    std::vector<models::GPUInfo> gpu_devices;
    size_t total_memory_gb;
    size_t cpu_cores;
    std::string architecture_profile; // "gaming", "datacenter", "workstation", etc.
    std::unordered_map<std::string, double> performance_characteristics;
    
    HardwareProfile()
        : total_memory_gb(0)
        , cpu_cores(0)
        , architecture_profile("unknown")
    {}
};

/**
 * @brief Main performance optimizer class
 */
class PerformanceOptimizer {
public:
    PerformanceOptimizer();
    ~PerformanceOptimizer();
    
    // Initialization and configuration
    bool initialize(const AdaptiveTuningConfig& config);
    bool configure_hardware_profile(const HardwareProfile& profile);
    void cleanup();
    
    // Performance monitoring and analysis
    bool start_monitoring(engine::IntegratedScanningEngine* engine);
    bool stop_monitoring();
    void add_performance_sample(const PerformanceSample& sample);
    
    // Bottleneck analysis
    std::vector<BottleneckAnalysis> analyze_current_bottlenecks() const;
    std::vector<BottleneckAnalysis> analyze_historical_bottlenecks() const;
    BottleneckAnalysis identify_primary_bottleneck() const;
    
    // Optimization suggestions
    std::vector<OptimizationSuggestion> generate_optimization_suggestions() const;
    std::vector<OptimizationSuggestion> get_hardware_specific_suggestions() const;
    bool apply_optimization_suggestions(const std::vector<OptimizationSuggestion>& suggestions);
    
    // Adaptive parameter tuning
    bool enable_adaptive_tuning(bool enable = true);
    bool tune_batch_sizes_adaptively();
    bool tune_gpu_parameters_adaptively();
    bool tune_memory_allocation_adaptively();
    bool optimize_load_balancing_adaptively();
    
    // Performance prediction and modeling
    double predict_performance_improvement(const OptimizationSuggestion& suggestion) const;
    double estimate_optimal_batch_size(int device_id) const;
    std::unordered_map<std::string, double> predict_parameter_impact(
        const std::unordered_map<std::string, std::string>& parameter_changes
    ) const;
    
    // Real-time optimization
    bool apply_real_time_optimizations();
    bool detect_and_handle_performance_regression();
    bool optimize_for_current_workload();
    
    // Configuration management
    AdaptiveTuningConfig get_current_config() const;
    bool update_config(const AdaptiveTuningConfig& config);
    bool save_optimization_state(const std::string& filename) const;
    bool load_optimization_state(const std::string& filename);
    
    // Statistics and reporting
    struct OptimizationStatistics {
        size_t total_optimizations_applied;
        size_t successful_optimizations;
        double average_performance_improvement;
        double peak_performance_improvement;
        size_t rollbacks_performed;
        std::chrono::milliseconds total_optimization_time;
        std::unordered_map<BottleneckType, size_t> bottleneck_frequency;
    };
    
    OptimizationStatistics get_optimization_statistics() const;
    std::string generate_optimization_report() const;
    bool export_performance_data(const std::string& filename) const;

private:
    // Configuration and state
    AdaptiveTuningConfig config_;
    HardwareProfile hardware_profile_;
    std::atomic<bool> monitoring_active_;
    std::atomic<bool> adaptive_tuning_enabled_;
    
    // Engine reference
    engine::IntegratedScanningEngine* target_engine_;
    
    // Performance monitoring
    std::vector<PerformanceSample> performance_history_;
    mutable std::mutex history_mutex_;
    std::thread monitoring_thread_;
    std::atomic<bool> monitoring_thread_running_;
    
    // Optimization state
    std::vector<OptimizationSuggestion> applied_optimizations_;
    std::vector<BottleneckAnalysis> detected_bottlenecks_;
    mutable std::mutex optimization_mutex_;
    
    // Statistics
    OptimizationStatistics stats_;
    mutable std::mutex stats_mutex_;
    
    // Adaptive tuning
    std::thread tuning_thread_;
    std::atomic<bool> tuning_thread_running_;
    std::unordered_map<std::string, std::string> parameter_history_;
    
    // Internal methods
    
    // Performance monitoring
    void monitoring_loop();
    void collect_current_metrics();
    double calculate_performance_score(const PerformanceSample& sample) const;
    bool is_performance_regression_detected() const;
    
    // Bottleneck detection
    BottleneckAnalysis analyze_gpu_utilization(const PerformanceSample& sample) const;
    BottleneckAnalysis analyze_memory_bandwidth(const PerformanceSample& sample) const;
    BottleneckAnalysis analyze_cpu_processing(const PerformanceSample& sample) const;
    BottleneckAnalysis analyze_io_operations(const PerformanceSample& sample) const;
    BottleneckAnalysis analyze_synchronization(const PerformanceSample& sample) const;
    BottleneckAnalysis analyze_load_balancing(const PerformanceSample& sample) const;
    BottleneckAnalysis analyze_memory_allocation(const PerformanceSample& sample) const;
    BottleneckAnalysis analyze_cache_efficiency(const PerformanceSample& sample) const;
    
    // Optimization generation
    OptimizationSuggestion suggest_batch_size_optimization() const;
    OptimizationSuggestion suggest_memory_optimization() const;
    OptimizationSuggestion suggest_gpu_configuration_optimization() const;
    OptimizationSuggestion suggest_load_balancing_optimization() const;
    OptimizationSuggestion suggest_thread_configuration_optimization() const;
    
    // Adaptive tuning
    void adaptive_tuning_loop();
    bool apply_adaptive_optimization();
    bool rollback_optimization(const OptimizationSuggestion& optimization);
    double measure_optimization_impact(const OptimizationSuggestion& suggestion);
    
    // Hardware analysis
    bool analyze_hardware_capabilities();
    std::string classify_hardware_profile() const;
    std::unordered_map<std::string, double> get_hardware_performance_characteristics() const;
    
    // Parameter optimization
    double optimize_single_parameter(const std::string& parameter_name, 
                                   const std::vector<std::string>& candidate_values);
    bool is_parameter_safe_to_modify(const std::string& parameter_name) const;
    std::vector<std::string> generate_parameter_candidates(const std::string& parameter_name) const;
    
    // Performance modeling
    double model_performance_with_parameters(
        const std::unordered_map<std::string, std::string>& parameters) const;
    double calculate_bottleneck_severity(BottleneckType type, const PerformanceSample& sample) const;
    
    // Utility methods
    void update_optimization_statistics(const OptimizationSuggestion& suggestion, bool success);
    void cleanup_old_performance_samples();
    bool validate_optimization_suggestion(const OptimizationSuggestion& suggestion) const;
};

/**
 * @brief Optimization utility functions
 */
namespace optimization_utils {
    
    // Performance analysis utilities
    double calculate_gpu_efficiency(const engine::ScanningEngineMetrics& metrics);
    double calculate_memory_efficiency(const engine::ScanningEngineMetrics& metrics);
    double calculate_overall_efficiency(const engine::ScanningEngineMetrics& metrics);
    
    // Bottleneck detection utilities
    BottleneckType identify_primary_bottleneck_type(const engine::ScanningEngineMetrics& metrics);
    double calculate_bottleneck_impact(BottleneckType type, const engine::ScanningEngineMetrics& metrics);
    std::string get_bottleneck_description(BottleneckType type);
    
    // Parameter optimization utilities
    size_t calculate_optimal_batch_size(const models::GPUInfo& gpu_info, size_t available_memory);
    int calculate_optimal_threads_per_block(const models::GPUInfo& gpu_info);
    size_t calculate_optimal_memory_allocation(const models::GPUInfo& gpu_info, size_t total_keys);
    
    // Hardware analysis utilities
    std::string classify_gpu_architecture(const models::GPUInfo& gpu_info);
    double estimate_theoretical_performance(const models::GPUInfo& gpu_info);
    bool is_memory_bandwidth_limited(const engine::ScanningEngineMetrics& metrics, const models::GPUInfo& gpu_info);
    
    // Configuration utilities
    engine::ScanningEngineConfig optimize_config_for_bottleneck(
        const engine::ScanningEngineConfig& base_config,
        BottleneckType bottleneck_type
    );
    
    bool validate_optimization_safety(const OptimizationSuggestion& suggestion);
    double estimate_optimization_risk(const OptimizationSuggestion& suggestion);
}

} // namespace optimization
} // namespace keyhunt