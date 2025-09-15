/**
 * @file bottleneck_analyzer.h
 * @brief Advanced bottleneck analysis and detection system
 * @author KeyhuntCUDA Team
 * 
 * T046: Sophisticated bottleneck analysis with real-time monitoring and predictive analytics
 */

#pragma once

#include "../engine/integrated_scanning_engine.h"
#include "performance_optimizer.h"
#include <vector>
#include <queue>
#include <memory>

namespace keyhunt {
namespace optimization {

/**
 * @brief Real-time bottleneck detection system
 */
class BottleneckAnalyzer {
public:
    /**
     * @brief Bottleneck severity levels
     */
    enum class SeverityLevel {
        NEGLIGIBLE,     // < 5% performance impact
        MINOR,          // 5-15% performance impact
        MODERATE,       // 15-30% performance impact
        MAJOR,          // 30-50% performance impact
        CRITICAL        // > 50% performance impact
    };
    
    /**
     * @brief Detailed bottleneck analysis result
     */
    struct DetailedBottleneckAnalysis {
        BottleneckType type;
        SeverityLevel severity;
        double impact_percentage;
        std::string root_cause;
        std::vector<std::string> contributing_factors;
        std::vector<OptimizationSuggestion> immediate_solutions;
        std::vector<OptimizationSuggestion> long_term_solutions;
        double confidence_score;
        std::chrono::system_clock::time_point detection_time;
        std::chrono::milliseconds time_since_first_detected;
        
        // Detailed metrics
        std::unordered_map<std::string, double> performance_metrics;
        std::unordered_map<std::string, double> resource_utilization;
        std::unordered_map<std::string, std::string> system_state;
    };
    
    /**
     * @brief Bottleneck trend analysis
     */
    struct BottleneckTrend {
        BottleneckType type;
        std::vector<double> severity_history;
        std::vector<std::chrono::system_clock::time_point> timestamps;
        double trend_slope; // Positive = getting worse, negative = improving
        double prediction_accuracy;
        std::chrono::milliseconds predicted_time_to_critical;
    };
    
    BottleneckAnalyzer();
    ~BottleneckAnalyzer();
    
    // Initialization and configuration
    bool initialize(engine::IntegratedScanningEngine* engine);
    bool start_real_time_analysis();
    bool stop_real_time_analysis();
    void cleanup();
    
    // Core analysis methods
    std::vector<DetailedBottleneckAnalysis> analyze_all_bottlenecks();
    DetailedBottleneckAnalysis analyze_specific_bottleneck(BottleneckType type);
    std::vector<BottleneckType> detect_emerging_bottlenecks();
    
    // Advanced analysis
    std::vector<BottleneckTrend> analyze_bottleneck_trends();
    std::vector<DetailedBottleneckAnalysis> predict_future_bottlenecks(
        std::chrono::minutes prediction_horizon = std::chrono::minutes(10)
    );
    
    // Specialized analysis methods
    DetailedBottleneckAnalysis analyze_gpu_bottlenecks();
    DetailedBottleneckAnalysis analyze_memory_bottlenecks();
    DetailedBottleneckAnalysis analyze_cpu_bottlenecks();
    DetailedBottleneckAnalysis analyze_io_bottlenecks();
    DetailedBottleneckAnalysis analyze_synchronization_bottlenecks();
    DetailedBottleneckAnalysis analyze_algorithmic_bottlenecks();
    
    // Cross-component analysis
    struct CrossComponentBottleneck {
        std::vector<BottleneckType> involved_components;
        std::string interaction_pattern;
        double combined_impact;
        std::vector<OptimizationSuggestion> coordinated_solutions;
    };
    
    std::vector<CrossComponentBottleneck> analyze_cross_component_bottlenecks();
    
    // Root cause analysis
    struct RootCauseAnalysis {
        BottleneckType primary_bottleneck;
        std::vector<std::string> causal_chain;
        std::string fundamental_cause;
        double certainty_level;
        std::vector<std::string> evidence;
        std::vector<OptimizationSuggestion> targeted_solutions;
    };
    
    RootCauseAnalysis perform_root_cause_analysis(BottleneckType bottleneck);
    
    // Performance impact assessment
    struct ImpactAssessment {
        BottleneckType bottleneck;
        double current_performance_loss;
        double potential_performance_gain;
        std::chrono::milliseconds time_to_resolve;
        double resource_cost_to_resolve;
        double roi_estimate; // Return on investment for fixing this bottleneck
    };
    
    std::vector<ImpactAssessment> assess_bottleneck_impacts();
    ImpactAssessment prioritize_bottlenecks_by_impact();
    
    // Reporting and visualization
    std::string generate_bottleneck_report();
    std::string generate_trend_analysis_report();
    std::string generate_optimization_roadmap();
    bool export_bottleneck_data(const std::string& filename);

private:
    engine::IntegratedScanningEngine* target_engine_;
    std::atomic<bool> analysis_active_;
    std::thread analysis_thread_;
    std::atomic<bool> analysis_thread_running_;
    
    // Analysis state
    std::deque<engine::ScanningEngineMetrics> metrics_history_;
    std::vector<DetailedBottleneckAnalysis> bottleneck_history_;
    std::unordered_map<BottleneckType, std::chrono::system_clock::time_point> first_detection_times_;
    
    mutable std::mutex analysis_mutex_;
    
    // Configuration
    size_t max_history_size_;
    std::chrono::seconds analysis_interval_;
    double severity_thresholds_[5]; // Thresholds for severity levels
    
    // Internal analysis methods
    void real_time_analysis_loop();
    void collect_metrics_sample();
    void update_bottleneck_trends();
    
    // GPU analysis helpers
    double analyze_gpu_memory_bandwidth_utilization();
    double analyze_gpu_compute_utilization();
    double analyze_gpu_occupancy();
    std::vector<std::string> identify_gpu_kernel_bottlenecks();
    
    // Memory analysis helpers
    double analyze_host_device_transfer_efficiency();
    double analyze_memory_allocation_patterns();
    bool detect_memory_leaks();
    std::vector<std::string> identify_memory_access_patterns();
    
    // CPU analysis helpers
    double analyze_cpu_gpu_synchronization_overhead();
    double analyze_cpu_bound_operations();
    std::vector<std::string> identify_cpu_hotspots();
    
    // I/O analysis helpers
    double analyze_disk_io_patterns();
    double analyze_network_io_bottlenecks();
    std::vector<std::string> identify_io_contention_points();
    
    // Synchronization analysis helpers
    double analyze_lock_contention();
    double analyze_thread_pool_efficiency();
    std::vector<std::string> identify_deadlock_risks();
    
    // Trend analysis helpers
    double calculate_trend_slope(const std::vector<double>& values, 
                               const std::vector<std::chrono::system_clock::time_point>& timestamps);
    double predict_future_severity(const BottleneckTrend& trend, std::chrono::minutes horizon);
    
    // Utility methods
    SeverityLevel categorize_severity(double impact_percentage);
    double calculate_confidence_score(const DetailedBottleneckAnalysis& analysis);
    bool is_bottleneck_persistent(BottleneckType type, std::chrono::minutes duration = std::chrono::minutes(2));
    void cleanup_old_data();
};

/**
 * @brief Utility functions for bottleneck analysis
 */
namespace bottleneck_utils {
    
    // Severity calculation utilities
    double calculate_gpu_bottleneck_severity(const engine::ScanningEngineMetrics& metrics);
    double calculate_memory_bottleneck_severity(const engine::ScanningEngineMetrics& metrics);
    double calculate_cpu_bottleneck_severity(const engine::ScanningEngineMetrics& metrics);
    
    // Impact estimation utilities
    double estimate_performance_impact(BottleneckType type, double severity);
    double estimate_resolution_time(BottleneckType type, BottleneckAnalyzer::SeverityLevel severity);
    double estimate_resolution_cost(BottleneckType type, const OptimizationSuggestion& solution);
    
    // Pattern detection utilities
    bool detect_cyclic_bottleneck_pattern(const std::vector<BottleneckAnalyzer::DetailedBottleneckAnalysis>& history);
    std::vector<BottleneckType> identify_correlated_bottlenecks(const std::vector<BottleneckAnalyzer::DetailedBottleneckAnalysis>& history);
    
    // Solution recommendation utilities
    std::vector<OptimizationSuggestion> generate_targeted_solutions(const BottleneckAnalyzer::DetailedBottleneckAnalysis& analysis);
    OptimizationSuggestion select_best_solution(const std::vector<OptimizationSuggestion>& solutions);
    
    // Reporting utilities
    std::string format_bottleneck_summary(const BottleneckAnalyzer::DetailedBottleneckAnalysis& analysis);
    std::string format_trend_summary(const BottleneckAnalyzer::BottleneckTrend& trend);
    std::string generate_optimization_timeline(const std::vector<OptimizationSuggestion>& solutions);
}

} // namespace optimization
} // namespace keyhunt