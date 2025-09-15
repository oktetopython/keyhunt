/**
 * @file analytics_engine.h
 * @brief Real-time analytics and insights engine
 * @author KeyhuntCUDA Team
 * 
 * T048: Advanced analytics engine for pattern recognition, anomaly detection, and predictive insights
 */

#pragma once

#include "monitoring_system.h"
#include <deque>
#include <unordered_set>
#include <algorithm>
#include <cmath>

namespace keyhunt {
namespace analytics {

/**
 * @brief Analytics data aggregation periods
 */
enum class AggregationPeriod {
    SECOND,         // 1 second aggregation
    MINUTE,         // 1 minute aggregation
    HOUR,           // 1 hour aggregation
    DAY,            // 1 day aggregation
    WEEK,           // 1 week aggregation
    MONTH           // 1 month aggregation
};

/**
 * @brief Aggregation functions for metrics
 */
enum class AggregationFunction {
    SUM,            // Sum of values
    AVERAGE,        // Average of values
    MIN,            // Minimum value
    MAX,            // Maximum value
    COUNT,          // Count of data points
    RATE,           // Rate of change
    PERCENTILE_50,  // Median
    PERCENTILE_95,  // 95th percentile
    PERCENTILE_99,  // 99th percentile
    STDDEV          // Standard deviation
};

/**
 * @brief Aggregated metric data point
 */
struct AggregatedMetric {
    std::string metric_name;
    AggregationPeriod period;
    AggregationFunction function;
    std::chrono::system_clock::time_point timestamp;
    double value;
    size_t data_points_count; // Number of raw data points used for aggregation
    std::unordered_map<std::string, std::string> tags;
    
    AggregatedMetric()
        : period(AggregationPeriod::MINUTE)
        , function(AggregationFunction::AVERAGE)
        , timestamp(std::chrono::system_clock::now())
        , value(0.0)
        , data_points_count(0)
    {}
};

/**
 * @brief Anomaly detection result
 */
struct AnomalyDetection {
    std::string metric_name;
    std::chrono::system_clock::time_point timestamp;
    double actual_value;
    double expected_value;
    double deviation_score;     // Standard deviations from normal
    double anomaly_probability; // 0.0 to 1.0
    std::string anomaly_type;   // "spike", "dip", "trend_change", "pattern_break"
    std::string description;
    
    AnomalyDetection()
        : timestamp(std::chrono::system_clock::now())
        , actual_value(0.0)
        , expected_value(0.0)
        , deviation_score(0.0)
        , anomaly_probability(0.0)
    {}
};

/**
 * @brief Trend analysis result
 */
struct TrendAnalysis {
    std::string metric_name;
    std::chrono::system_clock::time_point analysis_time;
    std::chrono::seconds analysis_period;
    
    enum class TrendDirection {
        INCREASING,
        DECREASING,
        STABLE,
        VOLATILE,
        UNKNOWN
    };
    
    TrendDirection direction;
    double slope;               // Rate of change
    double correlation;         // Correlation coefficient (-1 to 1)
    double volatility;          // Measure of variance
    double confidence_level;    // Confidence in trend analysis
    std::string interpretation; // Human-readable interpretation
    
    TrendAnalysis()
        : analysis_time(std::chrono::system_clock::now())
        , analysis_period(std::chrono::hours(1))
        , direction(TrendDirection::UNKNOWN)
        , slope(0.0)
        , correlation(0.0)
        , volatility(0.0)
        , confidence_level(0.0)
    {}
};

/**
 * @brief Performance insight
 */
struct PerformanceInsight {
    std::string insight_id;
    std::string category;       // "performance", "efficiency", "resource", "error"
    std::string title;
    std::string description;
    std::vector<std::string> related_metrics;
    double impact_score;        // 0.0 to 1.0 (impact on overall performance)
    std::vector<std::string> recommendations;
    std::chrono::system_clock::time_point generated_at;
    bool is_actionable;         // Whether this insight can be acted upon
    
    PerformanceInsight()
        : generated_at(std::chrono::system_clock::now())
        , impact_score(0.0)
        , is_actionable(false)
    {}
};

/**
 * @brief Analytics engine configuration
 */
struct AnalyticsConfig {
    bool enable_real_time_analytics;
    bool enable_anomaly_detection;
    bool enable_trend_analysis;
    bool enable_performance_insights;
    
    std::chrono::seconds analytics_update_interval;
    std::chrono::hours baseline_learning_period;    // Period to learn normal behavior
    double anomaly_sensitivity;                     // 0.0 to 1.0 (higher = more sensitive)
    size_t minimum_data_points_for_analysis;        // Minimum data points needed
    
    std::vector<AggregationPeriod> enabled_aggregation_periods;
    std::vector<AggregationFunction> default_aggregation_functions;
    
    AnalyticsConfig()
        : enable_real_time_analytics(true)
        , enable_anomaly_detection(true)
        , enable_trend_analysis(true)
        , enable_performance_insights(true)
        , analytics_update_interval(std::chrono::seconds(30))
        , baseline_learning_period(std::chrono::hours(6))
        , anomaly_sensitivity(0.7)
        , minimum_data_points_for_analysis(10)
    {
        enabled_aggregation_periods = {
            AggregationPeriod::MINUTE,
            AggregationPeriod::HOUR,
            AggregationPeriod::DAY
        };
        default_aggregation_functions = {
            AggregationFunction::AVERAGE,
            AggregationFunction::MAX,
            AggregationFunction::MIN,
            AggregationFunction::PERCENTILE_95
        };
    }
};

/**
 * @brief Main analytics engine
 */
class AnalyticsEngine {
public:
    AnalyticsEngine(const AnalyticsConfig& config = AnalyticsConfig());
    ~AnalyticsEngine();
    
    // System control
    bool initialize(monitoring::MonitoringSystem* monitoring_system);
    bool start_analytics();
    bool stop_analytics();
    void shutdown();
    
    // Data aggregation
    void process_metric_data(const monitoring::MetricDataPoint& metric);
    std::vector<AggregatedMetric> get_aggregated_data(const std::string& metric_name,
                                                     AggregationPeriod period,
                                                     AggregationFunction function,
                                                     std::chrono::system_clock::time_point start_time,
                                                     std::chrono::system_clock::time_point end_time);
    
    // Anomaly detection
    std::vector<AnomalyDetection> detect_anomalies(const std::string& metric_name = "");
    std::vector<AnomalyDetection> get_recent_anomalies(std::chrono::hours lookback = std::chrono::hours(1));
    bool is_anomaly_detected(const std::string& metric_name, double current_value);
    
    // Trend analysis
    TrendAnalysis analyze_trend(const std::string& metric_name, 
                               std::chrono::seconds analysis_period = std::chrono::hours(1));
    std::vector<TrendAnalysis> analyze_all_trends();
    
    // Performance insights
    std::vector<PerformanceInsight> generate_insights();
    std::vector<PerformanceInsight> get_recent_insights(std::chrono::hours lookback = std::chrono::hours(6));
    PerformanceInsight analyze_system_performance();
    
    // Predictive analytics
    struct Prediction {
        std::string metric_name;
        std::chrono::system_clock::time_point prediction_time;
        std::chrono::minutes prediction_horizon;
        double predicted_value;
        double confidence_interval; // +/- range
        double confidence_level;    // 0.0 to 1.0
        std::string prediction_method; // "linear", "exponential", "seasonal", "ml"
    };
    
    Prediction predict_metric_value(const std::string& metric_name, 
                                   std::chrono::minutes horizon);
    std::vector<Prediction> predict_all_metrics(std::chrono::minutes horizon);
    
    // Correlation analysis
    struct CorrelationResult {
        std::string metric1_name;
        std::string metric2_name;
        double correlation_coefficient; // -1.0 to 1.0
        double p_value;                // Statistical significance
        std::string correlation_strength; // "weak", "moderate", "strong"
        std::string interpretation;
    };
    
    std::vector<CorrelationResult> analyze_metric_correlations();
    CorrelationResult calculate_correlation(const std::string& metric1, const std::string& metric2);
    
    // Configuration
    bool update_config(const AnalyticsConfig& config);
    AnalyticsConfig get_config() const;
    
    // Statistics
    struct AnalyticsStats {
        size_t total_metrics_processed;
        size_t anomalies_detected;
        size_t trends_analyzed;
        size_t insights_generated;
        size_t predictions_made;
        std::chrono::milliseconds average_processing_time;
        std::chrono::milliseconds average_anomaly_detection_time;
        std::chrono::milliseconds average_trend_analysis_time;
    };
    
    AnalyticsStats get_analytics_stats() const;

private:
    AnalyticsConfig config_;
    monitoring::MonitoringSystem* monitoring_system_;
    std::atomic<bool> analytics_active_;
    
    // Data storage
    std::unordered_map<std::string, std::deque<monitoring::MetricDataPoint>> raw_data_;
    std::unordered_map<std::string, std::unordered_map<AggregationPeriod, 
        std::unordered_map<AggregationFunction, std::deque<AggregatedMetric>>>> aggregated_data_;
    mutable std::shared_mutex data_mutex_;
    
    // Baseline models for anomaly detection
    struct BaselineModel {
        std::string metric_name;
        double mean;
        double standard_deviation;
        double min_value;
        double max_value;
        std::deque<double> historical_values;
        size_t samples_count;
        std::chrono::system_clock::time_point last_updated;
    };
    
    std::unordered_map<std::string, BaselineModel> baseline_models_;
    mutable std::shared_mutex baseline_mutex_;
    
    // Detected anomalies and trends
    std::deque<AnomalyDetection> recent_anomalies_;
    std::unordered_map<std::string, TrendAnalysis> current_trends_;
    std::deque<PerformanceInsight> recent_insights_;
    mutable std::shared_mutex analysis_results_mutex_;
    
    // Background threads
    std::thread analytics_thread_;
    std::thread aggregation_thread_;
    std::thread cleanup_thread_;
    std::atomic<bool> threads_running_;
    
    // Statistics
    AnalyticsStats analytics_stats_;
    mutable std::mutex stats_mutex_;
    
    // Internal methods
    void analytics_processing_loop();
    void data_aggregation_loop();
    void cleanup_old_data_loop();
    
    // Data processing
    void update_baseline_model(const std::string& metric_name, double value);
    AggregatedMetric aggregate_data_points(const std::string& metric_name,
                                          const std::vector<monitoring::MetricDataPoint>& data_points,
                                          AggregationPeriod period,
                                          AggregationFunction function);
    
    // Anomaly detection algorithms
    bool detect_statistical_anomaly(const std::string& metric_name, double value);
    bool detect_threshold_anomaly(const std::string& metric_name, double value);
    bool detect_pattern_anomaly(const std::string& metric_name, double value);
    double calculate_anomaly_score(const BaselineModel& model, double value);
    
    // Trend analysis algorithms
    double calculate_linear_regression_slope(const std::vector<double>& values);
    double calculate_correlation_coefficient(const std::vector<double>& x, const std::vector<double>& y);
    double calculate_volatility(const std::vector<double>& values);
    TrendAnalysis::TrendDirection classify_trend_direction(double slope, double volatility);
    
    // Insight generation
    PerformanceInsight generate_performance_degradation_insight();
    PerformanceInsight generate_resource_utilization_insight();
    PerformanceInsight generate_error_rate_insight();
    PerformanceInsight generate_efficiency_insight();
    
    // Prediction algorithms
    double predict_linear_trend(const std::vector<double>& values, size_t steps_ahead);
    double predict_exponential_trend(const std::vector<double>& values, size_t steps_ahead);
    double predict_moving_average(const std::vector<double>& values, size_t window_size, size_t steps_ahead);
    
    // Utility methods
    std::vector<double> extract_values_from_time_series(const std::string& metric_name, 
                                                       std::chrono::seconds period);
    void cleanup_old_data();
    void update_analytics_stats(const std::string& operation, 
                               std::chrono::milliseconds processing_time);
};

/**
 * @brief Dashboard data provider for visualization
 */
class DashboardDataProvider {
public:
    DashboardDataProvider(AnalyticsEngine* analytics_engine, 
                         monitoring::MonitoringSystem* monitoring_system);
    
    // Dashboard data structures
    struct DashboardMetric {
        std::string name;
        std::string display_name;
        std::string unit;
        double current_value;
        double previous_value;
        double change_percentage;
        std::string trend_indicator; // "up", "down", "stable"
        std::vector<double> sparkline_data; // Last N values for mini-chart
    };
    
    struct DashboardSection {
        std::string title;
        std::vector<DashboardMetric> metrics;
        std::vector<AnomalyDetection> alerts;
        std::vector<PerformanceInsight> insights;
    };
    
    struct DashboardData {
        std::chrono::system_clock::time_point last_updated;
        std::vector<DashboardSection> sections;
        std::unordered_map<std::string, std::string> system_status; // "healthy", "warning", "critical"
    };
    
    // Data retrieval
    DashboardData get_dashboard_data();
    DashboardSection get_performance_section();
    DashboardSection get_system_resources_section();
    DashboardSection get_business_metrics_section();
    DashboardSection get_alerts_and_insights_section();
    
    // Export formats
    std::string export_dashboard_as_json();
    std::string export_dashboard_as_html();
    bool export_dashboard_to_file(const std::string& filename, const std::string& format);

private:
    AnalyticsEngine* analytics_engine_;
    monitoring::MonitoringSystem* monitoring_system_;
    
    DashboardMetric create_dashboard_metric(const std::string& metric_name, 
                                           const std::string& display_name,
                                           const std::string& unit);
    std::string determine_trend_indicator(double current_value, double previous_value);
    std::string determine_system_status(const std::vector<DashboardSection>& sections);
};

/**
 * @brief Analytics utility functions
 */
namespace analytics_utils {
    
    // Statistical utilities
    double calculate_mean(const std::vector<double>& values);
    double calculate_standard_deviation(const std::vector<double>& values, double mean);
    double calculate_median(std::vector<double> values);
    double calculate_percentile(std::vector<double> values, double percentile);
    double calculate_z_score(double value, double mean, double std_dev);
    
    // Time series utilities
    std::vector<double> smooth_time_series(const std::vector<double>& values, size_t window_size);
    std::vector<double> detect_outliers(const std::vector<double>& values, double threshold = 2.0);
    bool has_seasonal_pattern(const std::vector<double>& values, size_t period_hint);
    
    // Aggregation utilities
    std::string aggregation_period_to_string(AggregationPeriod period);
    std::string aggregation_function_to_string(AggregationFunction function);
    std::chrono::seconds aggregation_period_to_duration(AggregationPeriod period);
    
    // Formatting utilities
    std::string format_metric_value(double value, const std::string& unit);
    std::string format_percentage_change(double change);
    std::string format_trend_description(const TrendAnalysis& trend);
    std::string format_anomaly_description(const AnomalyDetection& anomaly);
    
    // Configuration utilities
    AnalyticsConfig create_development_analytics_config();
    AnalyticsConfig create_production_analytics_config();
    bool validate_analytics_config(const AnalyticsConfig& config);
}

} // namespace analytics
} // namespace keyhunt