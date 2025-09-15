/**
 * @file monitoring_system.cpp
 * @brief Implementation of real-time monitoring and alerting infrastructure
 * @author KeyhuntCUDA Team
 * 
 * T048: Implementation of monitoring system with metrics collection and alerting
 */

#include "monitoring_system.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>
#include <thread>
#include <iomanip>

namespace keyhunt {
namespace monitoring {

// MetricsCollector implementation
MetricsCollector::MetricsCollector(const std::string& collector_name)
    : collector_name_(collector_name)
    , stats_()
{
    // Initialize stats
    std::lock_guard<std::mutex> lock(stats_mutex_);
    for (int i = 0; i < 6; ++i) {
        stats_.metrics_by_type[i] = 0;
    }
}

void MetricsCollector::record_counter(const std::string& name, double value,
                                     const std::unordered_map<std::string, std::string>& tags) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MetricDataPoint metric;
    metric.name = name;
    metric.type = MetricType::COUNTER;
    metric.value = value;
    metric.tags = tags;
    
    on_metric_recorded(metric);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto collection_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    update_stats(metric, collection_time);
}

void MetricsCollector::record_gauge(const std::string& name, double value,
                                   const std::unordered_map<std::string, std::string>& tags) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MetricDataPoint metric;
    metric.name = name;
    metric.type = MetricType::GAUGE;
    metric.value = value;
    metric.tags = tags;
    
    on_metric_recorded(metric);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto collection_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    update_stats(metric, collection_time);
}

void MetricsCollector::record_histogram(const std::string& name, double value,
                                       const std::unordered_map<std::string, std::string>& tags) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MetricDataPoint metric;
    metric.name = name;
    metric.type = MetricType::HISTOGRAM;
    metric.value = value;
    metric.tags = tags;
    
    on_metric_recorded(metric);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto collection_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    update_stats(metric, collection_time);
}

void MetricsCollector::record_timer(const std::string& name, std::chrono::milliseconds duration,
                                   const std::unordered_map<std::string, std::string>& tags) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MetricDataPoint metric;
    metric.name = name;
    metric.type = MetricType::TIMER;
    metric.value = static_cast<double>(duration.count());
    metric.tags = tags;
    
    on_metric_recorded(metric);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto collection_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    update_stats(metric, collection_time);
}

void MetricsCollector::record_rate(const std::string& name, double rate,
                                  const std::unordered_map<std::string, std::string>& tags) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MetricDataPoint metric;
    metric.name = name;
    metric.type = MetricType::RATE;
    metric.value = rate;
    metric.tags = tags;
    
    on_metric_recorded(metric);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto collection_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    update_stats(metric, collection_time);
}

void MetricsCollector::record_percentage(const std::string& name, double percentage,
                                        const std::unordered_map<std::string, std::string>& tags) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MetricDataPoint metric;
    metric.name = name;
    metric.type = MetricType::PERCENTAGE;
    metric.value = percentage;
    metric.tags = tags;
    
    on_metric_recorded(metric);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto collection_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    update_stats(metric, collection_time);
}

void MetricsCollector::record_metrics_batch(const std::vector<MetricDataPoint>& metrics) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (const auto& metric : metrics) {
        on_metric_recorded(metric);
        auto collection_time = std::chrono::milliseconds(1); // Simplified for batch
        update_stats(metric, collection_time);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Update overall timing stats
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_collection_time += total_time;
    if (stats_.total_metrics_recorded > 0) {
        stats_.average_collection_time = std::chrono::milliseconds(
            stats_.total_collection_time.count() / stats_.total_metrics_recorded
        );
    }
}

std::unique_ptr<MetricsCollector::Timer> MetricsCollector::start_timer(
    const std::string& name, const std::unordered_map<std::string, std::string>& tags) {
    return std::make_unique<Timer>(this, name, tags);
}

void MetricsCollector::on_metric_recorded(const MetricDataPoint& metric) {
    // Default implementation - can be overridden by subclasses
    // This could send to monitoring system, log, etc.
}

void MetricsCollector::update_stats(const MetricDataPoint& metric, 
                                   std::chrono::milliseconds collection_time) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_metrics_recorded++;
    stats_.metrics_by_type[static_cast<int>(metric.type)]++;
    stats_.total_collection_time += collection_time;
    
    if (stats_.total_metrics_recorded > 0) {
        stats_.average_collection_time = std::chrono::milliseconds(
            stats_.total_collection_time.count() / stats_.total_metrics_recorded
        );
    }
}

MetricsCollector::CollectorStats MetricsCollector::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// Timer implementation
MetricsCollector::Timer::Timer(MetricsCollector* collector, const std::string& metric_name,
                              const std::unordered_map<std::string, std::string>& tags)
    : collector_(collector)
    , metric_name_(metric_name)
    , tags_(tags)
    , start_time_(std::chrono::high_resolution_clock::now())
    , stopped_(false)
{
}

MetricsCollector::Timer::~Timer() {
    if (!stopped_) {
        stop();
    }
}

void MetricsCollector::Timer::stop() {
    if (!stopped_) {
        auto duration = elapsed();
        collector_->record_timer(metric_name_, duration, tags_);
        stopped_ = true;
    }
}

std::chrono::milliseconds MetricsCollector::Timer::elapsed() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
}

// MonitoringSystem implementation
MonitoringSystem::MonitoringSystem(const MonitoringConfig& config)
    : config_(config)
    , monitoring_active_(false)
    , threads_running_(false)
{
    // Initialize system stats
    std::lock_guard<std::mutex> lock(stats_mutex_);
    system_stats_ = MonitoringStats();
}

MonitoringSystem::~MonitoringSystem() {
    shutdown();
}

bool MonitoringSystem::initialize() {
    std::unique_lock<std::shared_mutex> time_series_lock(time_series_mutex_);
    std::unique_lock<std::shared_mutex> alerts_lock(alerts_mutex_);
    
    // Clear existing data
    time_series_.clear();
    alert_conditions_.clear();
    active_alerts_.clear();
    alert_history_.clear();
    
    return true;
}

bool MonitoringSystem::start_monitoring() {
    if (monitoring_active_) {
        return true;
    }
    
    threads_running_ = true;
    monitoring_active_ = true;
    
    // Start background threads
    if (config_.enable_metrics_collection) {
        metrics_collection_thread_ = std::thread(&MonitoringSystem::metrics_collection_loop, this);
    }
    
    if (config_.enable_alerting) {
        alert_evaluation_thread_ = std::thread(&MonitoringSystem::alert_evaluation_loop, this);
    }
    
    cleanup_thread_ = std::thread(&MonitoringSystem::cleanup_loop, this);
    
    return true;
}

bool MonitoringSystem::stop_monitoring() {
    if (!monitoring_active_) {
        return true;
    }
    
    monitoring_active_ = false;
    threads_running_ = false;
    
    // Wait for threads to finish
    if (metrics_collection_thread_.joinable()) {
        metrics_collection_thread_.join();
    }
    if (alert_evaluation_thread_.joinable()) {
        alert_evaluation_thread_.join();
    }
    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }
    
    return true;
}

void MonitoringSystem::shutdown() {
    stop_monitoring();
    
    std::unique_lock<std::shared_mutex> collectors_lock(collectors_mutex_);
    std::unique_lock<std::shared_mutex> time_series_lock(time_series_mutex_);
    std::unique_lock<std::shared_mutex> alerts_lock(alerts_mutex_);
    
    collectors_.clear();
    time_series_.clear();
    alert_conditions_.clear();
    active_alerts_.clear();
    alert_history_.clear();
}

void MonitoringSystem::record_metric(const MetricDataPoint& metric) {
    if (!monitoring_active_) {
        return;
    }
    
    process_metric(metric);
}

void MonitoringSystem::record_metrics_batch(const std::vector<MetricDataPoint>& metrics) {
    if (!monitoring_active_) {
        return;
    }
    
    for (const auto& metric : metrics) {
        process_metric(metric);
    }
}

std::shared_ptr<MetricsCollector> MonitoringSystem::create_collector(const std::string& name) {
    std::unique_lock<std::shared_mutex> lock(collectors_mutex_);
    
    auto collector = std::make_shared<MetricsCollector>(name);
    collectors_[name] = collector;
    
    return collector;
}

std::shared_ptr<MetricsCollector> MonitoringSystem::get_collector(const std::string& name) {
    std::shared_lock<std::shared_mutex> lock(collectors_mutex_);
    
    auto it = collectors_.find(name);
    if (it != collectors_.end()) {
        return it->second;
    }
    
    return nullptr;
}

std::vector<MetricDataPoint> MonitoringSystem::get_metric_data(
    const std::string& metric_name,
    std::chrono::system_clock::time_point start_time,
    std::chrono::system_clock::time_point end_time) {
    
    std::shared_lock<std::shared_mutex> lock(time_series_mutex_);
    
    auto it = time_series_.find(metric_name);
    if (it == time_series_.end()) {
        return {};
    }
    
    std::vector<MetricDataPoint> result;
    for (const auto& point : it->second.data_points) {
        if (point.timestamp >= start_time && point.timestamp <= end_time) {
            result.push_back(point);
        }
    }
    
    return result;
}

std::vector<std::string> MonitoringSystem::get_available_metrics() const {
    std::shared_lock<std::shared_mutex> lock(time_series_mutex_);
    
    std::vector<std::string> metrics;
    for (const auto& pair : time_series_) {
        metrics.push_back(pair.first);
    }
    
    return metrics;
}

MetricTimeSeries MonitoringSystem::get_time_series(const std::string& metric_name) const {
    std::shared_lock<std::shared_mutex> lock(time_series_mutex_);
    
    auto it = time_series_.find(metric_name);
    if (it != time_series_.end()) {
        return it->second;
    }
    
    return MetricTimeSeries();
}

bool MonitoringSystem::add_alert_condition(const AlertCondition& condition) {
    std::unique_lock<std::shared_mutex> lock(alerts_mutex_);
    
    // Check if condition already exists
    auto it = std::find_if(alert_conditions_.begin(), alert_conditions_.end(),
        [&condition](const AlertCondition& existing) {
            return existing.condition_id == condition.condition_id;
        });
    
    if (it != alert_conditions_.end()) {
        return false; // Already exists
    }
    
    alert_conditions_.push_back(condition);
    return true;
}

bool MonitoringSystem::remove_alert_condition(const std::string& condition_id) {
    std::unique_lock<std::shared_mutex> lock(alerts_mutex_);
    
    auto it = std::remove_if(alert_conditions_.begin(), alert_conditions_.end(),
        [&condition_id](const AlertCondition& condition) {
            return condition.condition_id == condition_id;
        });
    
    if (it != alert_conditions_.end()) {
        alert_conditions_.erase(it, alert_conditions_.end());
        return true;
    }
    
    return false;
}

bool MonitoringSystem::update_alert_condition(const AlertCondition& condition) {
    std::unique_lock<std::shared_mutex> lock(alerts_mutex_);
    
    auto it = std::find_if(alert_conditions_.begin(), alert_conditions_.end(),
        [&condition](AlertCondition& existing) {
            return existing.condition_id == condition.condition_id;
        });
    
    if (it != alert_conditions_.end()) {
        *it = condition;
        return true;
    }
    
    return false;
}

std::vector<AlertCondition> MonitoringSystem::get_alert_conditions() const {
    std::shared_lock<std::shared_mutex> lock(alerts_mutex_);
    return alert_conditions_;
}

bool MonitoringSystem::register_alert_handler(AlertHandler handler) {
    std::unique_lock<std::shared_mutex> lock(alerts_mutex_);
    
    alert_handlers_.push_back(handler);
    return true;
}

void MonitoringSystem::trigger_alert(const AlertNotification& alert) {
    notify_alert_handlers(alert);
    
    std::unique_lock<std::shared_mutex> lock(alerts_mutex_);
    active_alerts_.push_back(alert);
    alert_history_.push_back(alert);
    
    // Update stats
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    system_stats_.total_alerts_triggered++;
}

std::vector<AlertNotification> MonitoringSystem::get_active_alerts() const {
    std::shared_lock<std::shared_mutex> lock(alerts_mutex_);
    return active_alerts_;
}

std::vector<AlertNotification> MonitoringSystem::get_alert_history(
    std::chrono::hours lookback) const {
    
    std::shared_lock<std::shared_mutex> lock(alerts_mutex_);
    
    auto cutoff_time = std::chrono::system_clock::now() - lookback;
    std::vector<AlertNotification> result;
    
    for (const auto& alert : alert_history_) {
        if (alert.triggered_at >= cutoff_time) {
            result.push_back(alert);
        }
    }
    
    return result;
}

bool MonitoringSystem::update_config(const MonitoringConfig& config) {
    config_ = config;
    return true;
}

MonitoringConfig MonitoringSystem::get_config() const {
    return config_;
}

MonitoringSystem::MonitoringStats MonitoringSystem::get_system_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return system_stats_;
}

void MonitoringSystem::process_metric(const MetricDataPoint& metric) {
    std::unique_lock<std::shared_mutex> lock(time_series_mutex_);
    
    // Get or create time series
    auto& time_series = time_series_[metric.name];
    time_series.metric_name = metric.name;
    time_series.type = metric.type;
    
    // Add data point
    time_series.data_points.push_back(metric);
    
    // Maintain size limits
    if (time_series.data_points.size() > time_series.max_data_points) {
        time_series.data_points.pop_front();
    }
    
    // Update system stats
    lock.unlock();
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    system_stats_.total_metrics_collected++;
}

void MonitoringSystem::metrics_collection_loop() {
    while (threads_running_) {
        update_system_stats();
        std::this_thread::sleep_for(config_.metrics_collection_interval);
    }
}

void MonitoringSystem::alert_evaluation_loop() {
    while (threads_running_) {
        std::shared_lock<std::shared_mutex> lock(alerts_mutex_);
        auto conditions = alert_conditions_;
        lock.unlock();
        
        for (auto& condition : conditions) {
            if (evaluate_alert_condition(condition)) {
                // Create alert notification
                AlertNotification alert;
                alert.alert_id = "alert_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
                alert.condition_id = condition.condition_id;
                alert.metric_name = condition.metric_name;
                alert.severity = condition.severity;
                alert.message = condition.description;
                
                trigger_alert(alert);
            }
        }
        
        std::this_thread::sleep_for(config_.alert_evaluation_interval);
    }
}

void MonitoringSystem::cleanup_loop() {
    while (threads_running_) {
        cleanup_old_data();
        std::this_thread::sleep_for(std::chrono::hours(1)); // Run cleanup every hour
    }
}

void MonitoringSystem::cleanup_old_data() {
    std::unique_lock<std::shared_mutex> lock(time_series_mutex_);
    
    auto cutoff_time = std::chrono::system_clock::now() - config_.metrics_retention_period;
    
    for (auto& pair : time_series_) {
        auto& data_points = pair.second.data_points;
        
        // Remove old data points
        data_points.erase(
            std::remove_if(data_points.begin(), data_points.end(),
                [cutoff_time](const MetricDataPoint& point) {
                    return point.timestamp < cutoff_time;
                }),
            data_points.end()
        );
    }
}

bool MonitoringSystem::evaluate_alert_condition(const AlertCondition& condition) {
    // Simplified alert evaluation - can be expanded with more sophisticated logic
    std::shared_lock<std::shared_mutex> lock(time_series_mutex_);
    
    auto it = time_series_.find(condition.metric_name);
    if (it == time_series_.end() || it->second.data_points.empty()) {
        return false;
    }
    
    auto& data_points = it->second.data_points;
    double latest_value = data_points.back().value;
    
    switch (condition.condition_type) {
        case AlertCondition::ConditionType::GREATER_THAN:
            return latest_value > condition.threshold_value;
        case AlertCondition::ConditionType::LESS_THAN:
            return latest_value < condition.threshold_value;
        case AlertCondition::ConditionType::EQUALS:
            return std::abs(latest_value - condition.threshold_value) < 1e-6;
        case AlertCondition::ConditionType::NOT_EQUALS:
            return std::abs(latest_value - condition.threshold_value) >= 1e-6;
        case AlertCondition::ConditionType::RANGE:
            return latest_value >= condition.threshold_value && 
                   latest_value <= condition.threshold_value_2;
        default:
            return false;
    }
}

void MonitoringSystem::notify_alert_handlers(const AlertNotification& alert) {
    std::shared_lock<std::shared_mutex> lock(alerts_mutex_);
    
    for (const auto& handler : alert_handlers_) {
        try {
            handler(alert);
        } catch (const std::exception& e) {
            // Log error handling alert notification
            // In a real implementation, this would use the logging system
        }
    }
}

void MonitoringSystem::update_system_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Update active time series count
    std::shared_lock<std::shared_mutex> ts_lock(time_series_mutex_);
    system_stats_.active_time_series = time_series_.size();
    ts_lock.unlock();
    
    // Update active alert conditions count
    std::shared_lock<std::shared_mutex> alert_lock(alerts_mutex_);
    system_stats_.active_alert_conditions = alert_conditions_.size();
}

// Utility functions implementation
namespace monitoring_utils {

std::string metric_type_to_string(MetricType type) {
    switch (type) {
        case MetricType::COUNTER: return "counter";
        case MetricType::GAUGE: return "gauge";
        case MetricType::HISTOGRAM: return "histogram";
        case MetricType::TIMER: return "timer";
        case MetricType::RATE: return "rate";
        case MetricType::PERCENTAGE: return "percentage";
        default: return "unknown";
    }
}

MetricType string_to_metric_type(const std::string& type_str) {
    if (type_str == "counter") return MetricType::COUNTER;
    if (type_str == "gauge") return MetricType::GAUGE;
    if (type_str == "histogram") return MetricType::HISTOGRAM;
    if (type_str == "timer") return MetricType::TIMER;
    if (type_str == "rate") return MetricType::RATE;
    if (type_str == "percentage") return MetricType::PERCENTAGE;
    return MetricType::GAUGE; // Default
}

std::string alert_severity_to_string(AlertSeverity severity) {
    switch (severity) {
        case AlertSeverity::LOW: return "low";
        case AlertSeverity::MEDIUM: return "medium";
        case AlertSeverity::HIGH: return "high";
        case AlertSeverity::CRITICAL: return "critical";
        default: return "unknown";
    }
}

AlertSeverity string_to_alert_severity(const std::string& severity_str) {
    if (severity_str == "low") return AlertSeverity::LOW;
    if (severity_str == "medium") return AlertSeverity::MEDIUM;
    if (severity_str == "high") return AlertSeverity::HIGH;
    if (severity_str == "critical") return AlertSeverity::CRITICAL;
    return AlertSeverity::MEDIUM; // Default
}

MonitoringConfig create_development_monitoring_config() {
    MonitoringConfig config;
    config.enable_metrics_collection = true;
    config.enable_alerting = false; // Less noise during development
    config.metrics_collection_interval = std::chrono::seconds(5);
    config.max_metrics_in_memory = 10000;
    config.metrics_retention_period = std::chrono::hours(1);
    return config;
}

MonitoringConfig create_production_monitoring_config() {
    MonitoringConfig config;
    config.enable_metrics_collection = true;
    config.enable_alerting = true;
    config.metrics_collection_interval = std::chrono::seconds(10);
    config.alert_evaluation_interval = std::chrono::seconds(30);
    config.max_metrics_in_memory = 100000;
    config.metrics_retention_period = std::chrono::hours(24);
    config.enable_metrics_export = true;
    return config;
}

} // namespace monitoring_utils

} // namespace monitoring
} // namespace keyhunt