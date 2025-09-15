/**
 * @file analytics_engine.cpp
 * @brief Implementation of real-time analytics and insights engine
 * @author KeyhuntCUDA Team
 * 
 * T048: Implementation of analytics engine for pattern recognition and anomaly detection
 */

#include "analytics_engine.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <random>
#include <thread>

namespace keyhunt {
namespace analytics {

// AnalyticsEngine implementation
AnalyticsEngine::AnalyticsEngine(const AnalyticsConfig& config)
    : config_(config)
    , monitoring_system_(nullptr)
    , analytics_active_(false)
    , threads_running_(false)
{
    // Initialize statistics
    std::lock_guard<std::mutex> lock(stats_mutex_);
    analytics_stats_ = AnalyticsStats();
}

AnalyticsEngine::~AnalyticsEngine() {
    shutdown();
}

bool AnalyticsEngine::initialize(monitoring::MonitoringSystem* monitoring_system) {
    if (!monitoring_system) {
        return false;
    }
    
    monitoring_system_ = monitoring_system;
    
    std::unique_lock<std::shared_mutex> data_lock(data_mutex_);
    std::unique_lock<std::shared_mutex> baseline_lock(baseline_mutex_);
    std::unique_lock<std::shared_mutex> results_lock(analysis_results_mutex_);
    
    // Clear existing data
    raw_data_.clear();
    aggregated_data_.clear();
    baseline_models_.clear();
    recent_anomalies_.clear();
    current_trends_.clear();
    recent_insights_.clear();
    
    return true;
}

bool AnalyticsEngine::start_analytics() {
    if (analytics_active_ || !monitoring_system_) {
        return false;
    }
    
    threads_running_ = true;
    analytics_active_ = true;
    
    // Start background threads
    if (config_.enable_real_time_analytics) {
        analytics_thread_ = std::thread(&AnalyticsEngine::analytics_processing_loop, this);
        aggregation_thread_ = std::thread(&AnalyticsEngine::data_aggregation_loop, this);
    }
    
    cleanup_thread_ = std::thread(&AnalyticsEngine::cleanup_old_data_loop, this);
    
    return true;
}

bool AnalyticsEngine::stop_analytics() {
    if (!analytics_active_) {
        return true;
    }
    
    analytics_active_ = false;
    threads_running_ = false;
    
    // Wait for threads to finish
    if (analytics_thread_.joinable()) {
        analytics_thread_.join();
    }
    if (aggregation_thread_.joinable()) {
        aggregation_thread_.join();
    }
    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }
    
    return true;
}

void AnalyticsEngine::shutdown() {
    stop_analytics();
    
    std::unique_lock<std::shared_mutex> data_lock(data_mutex_);
    std::unique_lock<std::shared_mutex> baseline_lock(baseline_mutex_);
    std::unique_lock<std::shared_mutex> results_lock(analysis_results_mutex_);
    
    raw_data_.clear();
    aggregated_data_.clear();
    baseline_models_.clear();
    recent_anomalies_.clear();
    current_trends_.clear();
    recent_insights_.clear();
}

void AnalyticsEngine::process_metric_data(const monitoring::MetricDataPoint& metric) {
    if (!analytics_active_) {
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::unique_lock<std::shared_mutex> lock(data_mutex_);
    
    // Add to raw data storage
    auto& metric_data = raw_data_[metric.name];
    metric_data.push_back(metric);
    
    // Maintain size limits
    const size_t max_raw_data_points = 10000;
    if (metric_data.size() > max_raw_data_points) {
        metric_data.pop_front();
    }
    
    lock.unlock();
    
    // Update baseline model for anomaly detection
    update_baseline_model(metric.name, metric.value);
    
    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    update_analytics_stats("process_metric", processing_time);
}

std::vector<AggregatedMetric> AnalyticsEngine::get_aggregated_data(
    const std::string& metric_name,
    AggregationPeriod period,
    AggregationFunction function,
    std::chrono::system_clock::time_point start_time,
    std::chrono::system_clock::time_point end_time) {
    
    std::shared_lock<std::shared_mutex> lock(data_mutex_);
    
    auto metric_it = aggregated_data_.find(metric_name);
    if (metric_it == aggregated_data_.end()) {
        return {};
    }
    
    auto period_it = metric_it->second.find(period);
    if (period_it == metric_it->second.end()) {
        return {};
    }
    
    auto function_it = period_it->second.find(function);
    if (function_it == period_it->second.end()) {
        return {};
    }
    
    std::vector<AggregatedMetric> result;
    for (const auto& aggregated_metric : function_it->second) {
        if (aggregated_metric.timestamp >= start_time && aggregated_metric.timestamp <= end_time) {
            result.push_back(aggregated_metric);
        }
    }
    
    return result;
}

std::vector<AnomalyDetection> AnalyticsEngine::detect_anomalies(const std::string& metric_name) {
    if (!config_.enable_anomaly_detection) {
        return {};
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<AnomalyDetection> anomalies;
    
    std::shared_lock<std::shared_mutex> data_lock(data_mutex_);
    std::shared_lock<std::shared_mutex> baseline_lock(baseline_mutex_);
    
    if (metric_name.empty()) {
        // Detect anomalies for all metrics
        for (const auto& pair : raw_data_) {
            if (!pair.second.empty()) {
                auto metric_anomalies = detect_anomalies(pair.first);
                anomalies.insert(anomalies.end(), metric_anomalies.begin(), metric_anomalies.end());
            }
        }
    } else {
        // Detect anomalies for specific metric
        auto raw_it = raw_data_.find(metric_name);
        auto baseline_it = baseline_models_.find(metric_name);
        
        if (raw_it != raw_data_.end() && baseline_it != baseline_models_.end() && 
            !raw_it->second.empty()) {
            
            const auto& latest_point = raw_it->second.back();
            const auto& model = baseline_it->second;
            
            if (detect_statistical_anomaly(metric_name, latest_point.value)) {
                AnomalyDetection anomaly;
                anomaly.metric_name = metric_name;
                anomaly.timestamp = latest_point.timestamp;
                anomaly.actual_value = latest_point.value;
                anomaly.expected_value = model.mean;
                anomaly.deviation_score = calculate_anomaly_score(model, latest_point.value);
                anomaly.anomaly_probability = std::min(1.0, std::abs(anomaly.deviation_score) / 3.0);
                
                // Classify anomaly type
                if (latest_point.value > model.mean + 2 * model.standard_deviation) {
                    anomaly.anomaly_type = "spike";
                } else if (latest_point.value < model.mean - 2 * model.standard_deviation) {
                    anomaly.anomaly_type = "dip";
                } else {
                    anomaly.anomaly_type = "pattern_break";
                }
                
                anomaly.description = "Statistical anomaly detected for metric " + metric_name;
                anomalies.push_back(anomaly);
            }
        }
    }
    
    // Store detected anomalies
    if (!anomalies.empty()) {
        std::unique_lock<std::shared_mutex> results_lock(analysis_results_mutex_);
        for (const auto& anomaly : anomalies) {
            recent_anomalies_.push_back(anomaly);
        }
        
        // Maintain size limits
        const size_t max_anomalies = 1000;
        while (recent_anomalies_.size() > max_anomalies) {
            recent_anomalies_.pop_front();
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    update_analytics_stats("detect_anomalies", processing_time);
    
    return anomalies;
}

std::vector<AnomalyDetection> AnalyticsEngine::get_recent_anomalies(std::chrono::hours lookback) {
    std::shared_lock<std::shared_mutex> lock(analysis_results_mutex_);
    
    auto cutoff_time = std::chrono::system_clock::now() - lookback;
    std::vector<AnomalyDetection> result;
    
    for (const auto& anomaly : recent_anomalies_) {
        if (anomaly.timestamp >= cutoff_time) {
            result.push_back(anomaly);
        }
    }
    
    return result;
}

bool AnalyticsEngine::is_anomaly_detected(const std::string& metric_name, double current_value) {
    std::shared_lock<std::shared_mutex> lock(baseline_mutex_);
    
    auto it = baseline_models_.find(metric_name);
    if (it == baseline_models_.end()) {
        return false; // No baseline to compare against
    }
    
    return detect_statistical_anomaly(metric_name, current_value);
}

TrendAnalysis AnalyticsEngine::analyze_trend(const std::string& metric_name, 
                                           std::chrono::seconds analysis_period) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    TrendAnalysis trend;
    trend.metric_name = metric_name;
    trend.analysis_time = std::chrono::system_clock::now();
    trend.analysis_period = analysis_period;
    
    std::vector<double> values = extract_values_from_time_series(metric_name, analysis_period);
    
    if (values.size() < config_.minimum_data_points_for_analysis) {
        trend.direction = TrendAnalysis::TrendDirection::UNKNOWN;
        trend.confidence_level = 0.0;
        trend.interpretation = "Insufficient data points for trend analysis";
        return trend;
    }
    
    // Calculate trend metrics
    trend.slope = calculate_linear_regression_slope(values);
    trend.volatility = calculate_volatility(values);
    trend.correlation = calculate_correlation_coefficient(
        std::vector<double>(values.size()), values); // Simple index correlation
    
    // Classify trend direction
    trend.direction = classify_trend_direction(trend.slope, trend.volatility);
    
    // Calculate confidence level
    trend.confidence_level = std::min(1.0, std::abs(trend.correlation) * 
                                     (1.0 - trend.volatility / (trend.volatility + 1.0)));
    
    // Generate interpretation
    std::ostringstream oss;
    switch (trend.direction) {
        case TrendAnalysis::TrendDirection::INCREASING:
            oss << "Increasing trend with slope " << std::fixed << std::setprecision(4) << trend.slope;
            break;
        case TrendAnalysis::TrendDirection::DECREASING:
            oss << "Decreasing trend with slope " << std::fixed << std::setprecision(4) << trend.slope;
            break;
        case TrendAnalysis::TrendDirection::STABLE:
            oss << "Stable trend with low volatility (" << std::fixed << std::setprecision(4) << trend.volatility << ")";
            break;
        case TrendAnalysis::TrendDirection::VOLATILE:
            oss << "Volatile pattern with high variance (" << std::fixed << std::setprecision(4) << trend.volatility << ")";
            break;
        default:
            oss << "Trend analysis inconclusive";
            break;
    }
    trend.interpretation = oss.str();
    
    // Store trend analysis result
    std::unique_lock<std::shared_mutex> lock(analysis_results_mutex_);
    current_trends_[metric_name] = trend;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    update_analytics_stats("analyze_trend", processing_time);
    
    return trend;
}

std::vector<TrendAnalysis> AnalyticsEngine::analyze_all_trends() {
    std::shared_lock<std::shared_mutex> data_lock(data_mutex_);
    
    std::vector<TrendAnalysis> trends;
    for (const auto& pair : raw_data_) {
        if (!pair.second.empty()) {
            trends.push_back(analyze_trend(pair.first));
        }
    }
    
    return trends;
}

std::vector<PerformanceInsight> AnalyticsEngine::generate_insights() {
    if (!config_.enable_performance_insights) {
        return {};
    }
    
    std::vector<PerformanceInsight> insights;
    
    // Generate different types of insights
    insights.push_back(generate_performance_degradation_insight());
    insights.push_back(generate_resource_utilization_insight());
    insights.push_back(generate_error_rate_insight());
    insights.push_back(generate_efficiency_insight());
    
    // Store insights
    std::unique_lock<std::shared_mutex> lock(analysis_results_mutex_);
    for (const auto& insight : insights) {
        if (insight.impact_score > 0.1) { // Only store meaningful insights
            recent_insights_.push_back(insight);
        }
    }
    
    // Maintain size limits
    const size_t max_insights = 100;
    while (recent_insights_.size() > max_insights) {
        recent_insights_.pop_front();
    }
    
    return insights;
}

std::vector<PerformanceInsight> AnalyticsEngine::get_recent_insights(std::chrono::hours lookback) {
    std::shared_lock<std::shared_mutex> lock(analysis_results_mutex_);
    
    auto cutoff_time = std::chrono::system_clock::now() - lookback;
    std::vector<PerformanceInsight> result;
    
    for (const auto& insight : recent_insights_) {
        if (insight.generated_at >= cutoff_time) {
            result.push_back(insight);
        }
    }
    
    return result;
}

AnalyticsEngine::Prediction AnalyticsEngine::predict_metric_value(
    const std::string& metric_name, std::chrono::minutes horizon) {
    
    Prediction prediction;
    prediction.metric_name = metric_name;
    prediction.prediction_time = std::chrono::system_clock::now();
    prediction.prediction_horizon = horizon;
    
    std::vector<double> values = extract_values_from_time_series(
        metric_name, std::chrono::hours(1)); // Use 1 hour of data for prediction
    
    if (values.size() < config_.minimum_data_points_for_analysis) {
        prediction.confidence_level = 0.0;
        prediction.prediction_method = "insufficient_data";
        return prediction;
    }
    
    // Use linear trend prediction as default method
    size_t steps_ahead = static_cast<size_t>(horizon.count() / 5); // Assume 5-minute intervals
    prediction.predicted_value = predict_linear_trend(values, steps_ahead);
    prediction.confidence_interval = calculate_volatility(values) * 2.0; // Simple confidence interval
    prediction.confidence_level = std::min(1.0, 1.0 - calculate_volatility(values));
    prediction.prediction_method = "linear";
    
    return prediction;
}

AnalyticsEngine::AnalyticsStats AnalyticsEngine::get_analytics_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return analytics_stats_;
}

// Private methods implementation
void AnalyticsEngine::analytics_processing_loop() {
    while (threads_running_) {
        // Perform periodic analytics tasks
        if (config_.enable_anomaly_detection) {
            detect_anomalies();
        }
        
        if (config_.enable_trend_analysis) {
            analyze_all_trends();
        }
        
        if (config_.enable_performance_insights) {
            generate_insights();
        }
        
        std::this_thread::sleep_for(config_.analytics_update_interval);
    }
}

void AnalyticsEngine::data_aggregation_loop() {
    while (threads_running_) {
        // Perform data aggregation
        std::shared_lock<std::shared_mutex> data_lock(data_mutex_);
        
        for (const auto& pair : raw_data_) {
            const std::string& metric_name = pair.first;
            const auto& data_points = pair.second;
            
            if (data_points.size() < 2) continue;
            
            // Convert to vector for aggregation
            std::vector<monitoring::MetricDataPoint> points(data_points.begin(), data_points.end());
            
            // Aggregate for different periods and functions
            for (auto period : config_.enabled_aggregation_periods) {
                for (auto function : config_.default_aggregation_functions) {
                    auto aggregated = aggregate_data_points(metric_name, points, period, function);
                    
                    // Store aggregated data
                    std::unique_lock<std::shared_mutex> agg_lock(data_mutex_);
                    auto& period_map = aggregated_data_[metric_name][period];
                    auto& function_deque = period_map[function];
                    function_deque.push_back(aggregated);
                    
                    // Maintain size limits
                    const size_t max_aggregated_points = 1000;
                    if (function_deque.size() > max_aggregated_points) {
                        function_deque.pop_front();
                    }
                    agg_lock.unlock();
                }
            }
        }
        
        data_lock.unlock();
        
        std::this_thread::sleep_for(std::chrono::minutes(1)); // Run aggregation every minute
    }
}

void AnalyticsEngine::cleanup_old_data_loop() {
    while (threads_running_) {
        cleanup_old_data();
        std::this_thread::sleep_for(std::chrono::hours(1)); // Run cleanup every hour
    }
}

void AnalyticsEngine::update_baseline_model(const std::string& metric_name, double value) {
    std::unique_lock<std::shared_mutex> lock(baseline_mutex_);
    
    auto& model = baseline_models_[metric_name];
    model.metric_name = metric_name;
    model.last_updated = std::chrono::system_clock::now();
    
    // Add to historical values
    model.historical_values.push_back(value);
    const size_t max_history = 1000;
    if (model.historical_values.size() > max_history) {
        model.historical_values.pop_front();
    }
    
    // Update statistics
    model.samples_count = model.historical_values.size();
    if (model.samples_count > 0) {
        // Calculate mean
        model.mean = std::accumulate(model.historical_values.begin(), 
                                   model.historical_values.end(), 0.0) / model.samples_count;
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double val : model.historical_values) {
            variance += (val - model.mean) * (val - model.mean);
        }
        variance /= model.samples_count;
        model.standard_deviation = std::sqrt(variance);
        
        // Update min/max
        auto minmax = std::minmax_element(model.historical_values.begin(), 
                                        model.historical_values.end());
        model.min_value = *minmax.first;
        model.max_value = *minmax.second;
    }
}

bool AnalyticsEngine::detect_statistical_anomaly(const std::string& metric_name, double value) {
    std::shared_lock<std::shared_mutex> lock(baseline_mutex_);
    
    auto it = baseline_models_.find(metric_name);
    if (it == baseline_models_.end() || it->second.samples_count < config_.minimum_data_points_for_analysis) {
        return false;
    }
    
    const auto& model = it->second;
    double z_score = std::abs(calculate_anomaly_score(model, value));
    
    // Use configurable sensitivity to determine threshold
    double threshold = 3.0 * (1.0 - config_.anomaly_sensitivity); // Higher sensitivity = lower threshold
    return z_score > threshold;
}

double AnalyticsEngine::calculate_anomaly_score(const BaselineModel& model, double value) {
    if (model.standard_deviation == 0.0) {
        return 0.0;
    }
    return (value - model.mean) / model.standard_deviation;
}

double AnalyticsEngine::calculate_linear_regression_slope(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    
    size_t n = values.size();
    double sum_x = n * (n - 1) / 2.0; // Sum of indices
    double sum_y = std::accumulate(values.begin(), values.end(), 0.0);
    double sum_xy = 0.0;
    double sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0; // Sum of squared indices
    
    for (size_t i = 0; i < n; ++i) {
        sum_xy += i * values[i];
    }
    
    double denominator = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denominator) < 1e-10) return 0.0;
    
    return (n * sum_xy - sum_x * sum_y) / denominator;
}

double AnalyticsEngine::calculate_correlation_coefficient(
    const std::vector<double>& x, const std::vector<double>& y) {
    
    if (x.size() != y.size() || x.size() < 2) return 0.0;
    
    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    
    double numerator = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
    
    for (size_t i = 0; i < x.size(); ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }
    
    double denominator = std::sqrt(sum_x2 * sum_y2);
    if (std::abs(denominator) < 1e-10) return 0.0;
    
    return numerator / denominator;
}

double AnalyticsEngine::calculate_volatility(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double variance = 0.0;
    
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    
    variance /= values.size() - 1;
    return std::sqrt(variance);
}

TrendAnalysis::TrendDirection AnalyticsEngine::classify_trend_direction(
    double slope, double volatility) {
    
    const double slope_threshold = 0.01;
    const double volatility_threshold = 0.1;
    
    if (volatility > volatility_threshold) {
        return TrendAnalysis::TrendDirection::VOLATILE;
    } else if (slope > slope_threshold) {
        return TrendAnalysis::TrendDirection::INCREASING;
    } else if (slope < -slope_threshold) {
        return TrendAnalysis::TrendDirection::DECREASING;
    } else {
        return TrendAnalysis::TrendDirection::STABLE;
    }
}

PerformanceInsight AnalyticsEngine::generate_performance_degradation_insight() {
    PerformanceInsight insight;
    insight.insight_id = "perf_degradation_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    insight.category = "performance";
    insight.title = "Performance Degradation Analysis";
    insight.description = "Analysis of system performance trends and degradation patterns";
    
    // Simplified insight generation - in practice this would analyze multiple metrics
    insight.impact_score = 0.5; // Placeholder
    insight.is_actionable = true;
    insight.recommendations.push_back("Monitor key performance metrics closely");
    insight.recommendations.push_back("Consider system optimization if trends continue");
    
    return insight;
}

PerformanceInsight AnalyticsEngine::generate_resource_utilization_insight() {
    PerformanceInsight insight;
    insight.insight_id = "resource_util_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    insight.category = "resource";
    insight.title = "Resource Utilization Analysis";
    insight.description = "Analysis of system resource usage patterns and efficiency";
    
    insight.impact_score = 0.3;
    insight.is_actionable = true;
    insight.recommendations.push_back("Optimize memory allocation patterns");
    insight.recommendations.push_back("Consider load balancing improvements");
    
    return insight;
}

PerformanceInsight AnalyticsEngine::generate_error_rate_insight() {
    PerformanceInsight insight;
    insight.insight_id = "error_rate_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    insight.category = "error";
    insight.title = "Error Rate Analysis";
    insight.description = "Analysis of error patterns and system reliability";
    
    insight.impact_score = 0.7;
    insight.is_actionable = true;
    insight.recommendations.push_back("Investigate recent error spikes");
    insight.recommendations.push_back("Implement additional error handling");
    
    return insight;
}

PerformanceInsight AnalyticsEngine::generate_efficiency_insight() {
    PerformanceInsight insight;
    insight.insight_id = "efficiency_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    insight.category = "efficiency";
    insight.title = "System Efficiency Analysis";
    insight.description = "Analysis of overall system efficiency and optimization opportunities";
    
    insight.impact_score = 0.4;
    insight.is_actionable = true;
    insight.recommendations.push_back("Consider algorithmic optimizations");
    insight.recommendations.push_back("Review resource allocation strategies");
    
    return insight;
}

double AnalyticsEngine::predict_linear_trend(const std::vector<double>& values, size_t steps_ahead) {
    if (values.size() < 2) return 0.0;
    
    double slope = calculate_linear_regression_slope(values);
    double latest_value = values.back();
    
    return latest_value + slope * steps_ahead;
}

std::vector<double> AnalyticsEngine::extract_values_from_time_series(
    const std::string& metric_name, std::chrono::seconds period) {
    
    std::shared_lock<std::shared_mutex> lock(data_mutex_);
    
    auto it = raw_data_.find(metric_name);
    if (it == raw_data_.end()) {
        return {};
    }
    
    auto cutoff_time = std::chrono::system_clock::now() - period;
    std::vector<double> values;
    
    for (const auto& point : it->second) {
        if (point.timestamp >= cutoff_time) {
            values.push_back(point.value);
        }
    }
    
    return values;
}

void AnalyticsEngine::cleanup_old_data() {
    // Cleanup raw data
    std::unique_lock<std::shared_mutex> data_lock(data_mutex_);
    
    auto cutoff_time = std::chrono::system_clock::now() - std::chrono::hours(6);
    
    for (auto& pair : raw_data_) {
        auto& data_points = pair.second;
        
        // Remove old data points
        data_points.erase(
            std::remove_if(data_points.begin(), data_points.end(),
                [cutoff_time](const monitoring::MetricDataPoint& point) {
                    return point.timestamp < cutoff_time;
                }),
            data_points.end()
        );
    }
    
    data_lock.unlock();
    
    // Cleanup analysis results
    std::unique_lock<std::shared_mutex> results_lock(analysis_results_mutex_);
    
    // Remove old anomalies
    recent_anomalies_.erase(
        std::remove_if(recent_anomalies_.begin(), recent_anomalies_.end(),
            [cutoff_time](const AnomalyDetection& anomaly) {
                return anomaly.timestamp < cutoff_time;
            }),
        recent_anomalies_.end()
    );
    
    // Remove old insights
    recent_insights_.erase(
        std::remove_if(recent_insights_.begin(), recent_insights_.end(),
            [cutoff_time](const PerformanceInsight& insight) {
                return insight.generated_at < cutoff_time;
            }),
        recent_insights_.end()
    );
}

void AnalyticsEngine::update_analytics_stats(const std::string& operation, 
                                           std::chrono::milliseconds processing_time) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (operation == "process_metric") {
        analytics_stats_.total_metrics_processed++;
    } else if (operation == "detect_anomalies") {
        analytics_stats_.anomalies_detected++;
        analytics_stats_.average_anomaly_detection_time = processing_time;
    } else if (operation == "analyze_trend") {
        analytics_stats_.trends_analyzed++;
        analytics_stats_.average_trend_analysis_time = processing_time;
    }
    
    analytics_stats_.average_processing_time = processing_time;
}

// Utility functions implementation
namespace analytics_utils {

double calculate_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double calculate_standard_deviation(const std::vector<double>& values, double mean) {
    if (values.size() < 2) return 0.0;
    
    double variance = 0.0;
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= values.size() - 1;
    
    return std::sqrt(variance);
}

double calculate_median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    
    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0;
    } else {
        return values[n/2];
    }
}

double calculate_percentile(std::vector<double> values, double percentile) {
    if (values.empty()) return 0.0;
    
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>((percentile / 100.0) * (values.size() - 1));
    
    return values[std::min(index, values.size() - 1)];
}

std::string aggregation_period_to_string(AggregationPeriod period) {
    switch (period) {
        case AggregationPeriod::SECOND: return "second";
        case AggregationPeriod::MINUTE: return "minute";
        case AggregationPeriod::HOUR: return "hour";
        case AggregationPeriod::DAY: return "day";
        case AggregationPeriod::WEEK: return "week";
        case AggregationPeriod::MONTH: return "month";
        default: return "unknown";
    }
}

std::string aggregation_function_to_string(AggregationFunction function) {
    switch (function) {
        case AggregationFunction::SUM: return "sum";
        case AggregationFunction::AVERAGE: return "average";
        case AggregationFunction::MIN: return "min";
        case AggregationFunction::MAX: return "max";
        case AggregationFunction::COUNT: return "count";
        case AggregationFunction::RATE: return "rate";
        case AggregationFunction::PERCENTILE_50: return "percentile_50";
        case AggregationFunction::PERCENTILE_95: return "percentile_95";
        case AggregationFunction::PERCENTILE_99: return "percentile_99";
        case AggregationFunction::STDDEV: return "stddev";
        default: return "unknown";
    }
}

AnalyticsConfig create_development_analytics_config() {
    AnalyticsConfig config;
    config.enable_real_time_analytics = true;
    config.enable_anomaly_detection = false; // Reduce noise during development
    config.enable_trend_analysis = true;
    config.enable_performance_insights = false;
    config.analytics_update_interval = std::chrono::seconds(60);
    config.baseline_learning_period = std::chrono::hours(1);
    config.anomaly_sensitivity = 0.5;
    return config;
}

AnalyticsConfig create_production_analytics_config() {
    AnalyticsConfig config;
    config.enable_real_time_analytics = true;
    config.enable_anomaly_detection = true;
    config.enable_trend_analysis = true;
    config.enable_performance_insights = true;
    config.analytics_update_interval = std::chrono::seconds(30);
    config.baseline_learning_period = std::chrono::hours(6);
    config.anomaly_sensitivity = 0.7;
    return config;
}

} // namespace analytics_utils

} // namespace analytics
} // namespace keyhunt