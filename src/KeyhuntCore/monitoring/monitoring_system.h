/**
 * @file monitoring_system.h
 * @brief Real-time monitoring and alerting infrastructure
 * @author KeyhuntCUDA Team
 * 
 * T048: Real-time monitoring system with metrics collection, alerting, and analytics
 */

#pragma once

#include "logging_system.h"
#include <chrono>
#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>
#include <functional>

namespace keyhunt {
namespace monitoring {

/**
 * @brief Metric data types
 */
enum class MetricType {
    COUNTER,        // Monotonically increasing value
    GAUGE,          // Instantaneous value that can go up or down
    HISTOGRAM,      // Distribution of values over time
    TIMER,          // Measures duration of events
    RATE,           // Rate of change over time
    PERCENTAGE      // Percentage value (0-100)
};

/**
 * @brief Alert severity levels
 */
enum class AlertSeverity {
    LOW = 0,        // Informational alerts
    MEDIUM = 1,     // Warning alerts
    HIGH = 2,       // Error alerts
    CRITICAL = 3    // Critical system alerts
};

/**
 * @brief Metric data point
 */
struct MetricDataPoint {
    std::chrono::system_clock::time_point timestamp;
    std::string name;
    MetricType type;
    double value;
    std::unordered_map<std::string, std::string> tags; // Labels/dimensions
    
    MetricDataPoint()
        : timestamp(std::chrono::system_clock::now())
        , type(MetricType::GAUGE)
        , value(0.0)
    {}
};

/**
 * @brief Time series data for a metric
 */
struct MetricTimeSeries {
    std::string metric_name;
    MetricType type;
    std::deque<MetricDataPoint> data_points;
    size_t max_data_points;
    std::chrono::seconds retention_period;
    
    MetricTimeSeries()
        : type(MetricType::GAUGE)
        , max_data_points(1000)
        , retention_period(std::chrono::hours(1))
    {}
};

/**
 * @brief Alert condition configuration
 */
struct AlertCondition {
    std::string condition_id;
    std::string metric_name;
    std::string description;
    AlertSeverity severity;
    
    // Threshold conditions
    enum class ConditionType {
        GREATER_THAN,
        LESS_THAN,
        EQUALS,
        NOT_EQUALS,
        RANGE,          // Value within range
        RATE_OF_CHANGE, // Rate of change threshold
        ABSENCE         // Metric not reported
    };
    
    ConditionType condition_type;
    double threshold_value;
    double threshold_value_2; // For range conditions
    std::chrono::seconds evaluation_window;
    std::chrono::seconds cooldown_period; // Minimum time between alerts
    
    // Alert state
    bool is_triggered;
    std::chrono::system_clock::time_point last_triggered;
    size_t trigger_count;
    
    AlertCondition()
        : severity(AlertSeverity::MEDIUM)
        , condition_type(ConditionType::GREATER_THAN)
        , threshold_value(0.0)
        , threshold_value_2(0.0)
        , evaluation_window(std::chrono::seconds(60))
        , cooldown_period(std::chrono::seconds(300))
        , is_triggered(false)
        , trigger_count(0)
    {}
};

/**
 * @brief Alert notification
 */
struct AlertNotification {
    std::string alert_id;
    std::string condition_id;
    std::string metric_name;
    AlertSeverity severity;
    std::string message;
    double current_value;
    double threshold_value;
    std::chrono::system_clock::time_point triggered_at;
    std::unordered_map<std::string, std::string> context;
    
    AlertNotification()
        : severity(AlertSeverity::MEDIUM)
        , current_value(0.0)
        , threshold_value(0.0)
        , triggered_at(std::chrono::system_clock::now())
    {}
};

/**
 * @brief Metrics collector interface
 */
class MetricsCollector {
public:
    MetricsCollector(const std::string& collector_name);
    virtual ~MetricsCollector() = default;
    
    // Metric recording methods
    void record_counter(const std::string& name, double value = 1.0, 
                       const std::unordered_map<std::string, std::string>& tags = {});
    void record_gauge(const std::string& name, double value,
                     const std::unordered_map<std::string, std::string>& tags = {});
    void record_histogram(const std::string& name, double value,
                         const std::unordered_map<std::string, std::string>& tags = {});
    void record_timer(const std::string& name, std::chrono::milliseconds duration,
                     const std::unordered_map<std::string, std::string>& tags = {});
    void record_rate(const std::string& name, double rate,
                    const std::unordered_map<std::string, std::string>& tags = {});
    void record_percentage(const std::string& name, double percentage,
                          const std::unordered_map<std::string, std::string>& tags = {});
    
    // Bulk operations
    void record_metrics_batch(const std::vector<MetricDataPoint>& metrics);
    
    // Timer helper class
    class Timer {
    public:
        Timer(MetricsCollector* collector, const std::string& metric_name,
              const std::unordered_map<std::string, std::string>& tags = {});
        ~Timer();
        
        void stop();
        std::chrono::milliseconds elapsed() const;
        
    private:
        MetricsCollector* collector_;
        std::string metric_name_;
        std::unordered_map<std::string, std::string> tags_;
        std::chrono::high_resolution_clock::time_point start_time_;
        bool stopped_;
    };
    
    // Convenience method for creating timers
    std::unique_ptr<Timer> start_timer(const std::string& name,
                                      const std::unordered_map<std::string, std::string>& tags = {});
    
    // Statistics
    struct CollectorStats {
        size_t total_metrics_recorded;
        size_t metrics_by_type[6]; // One for each MetricType
        size_t unique_metric_names;
        std::chrono::milliseconds total_collection_time;
        std::chrono::milliseconds average_collection_time;
    };
    
    CollectorStats get_stats() const;
    
protected:
    std::string collector_name_;
    CollectorStats stats_;
    mutable std::mutex stats_mutex_;
    
    virtual void on_metric_recorded(const MetricDataPoint& metric);
    void update_stats(const MetricDataPoint& metric, std::chrono::milliseconds collection_time);
};

/**
 * @brief System monitoring configuration
 */
struct MonitoringConfig {
    bool enable_metrics_collection;
    bool enable_alerting;
    std::chrono::seconds metrics_collection_interval;
    std::chrono::seconds alert_evaluation_interval;
    size_t max_metrics_in_memory;
    size_t max_time_series_points;
    std::chrono::hours metrics_retention_period;
    bool enable_metrics_export; // Export to external systems
    std::string metrics_export_endpoint;
    std::string metrics_export_format; // "prometheus", "influxdb", "json"
    
    MonitoringConfig()
        : enable_metrics_collection(true)
        , enable_alerting(true)
        , metrics_collection_interval(std::chrono::seconds(10))
        , alert_evaluation_interval(std::chrono::seconds(30))
        , max_metrics_in_memory(100000)
        , max_time_series_points(1000)
        , metrics_retention_period(std::chrono::hours(24))
        , enable_metrics_export(false)
        , metrics_export_format("prometheus")
    {}
};

/**
 * @brief Main monitoring system
 */
class MonitoringSystem {
public:
    MonitoringSystem(const MonitoringConfig& config = MonitoringConfig());
    ~MonitoringSystem();
    
    // System control
    bool initialize();
    bool start_monitoring();
    bool stop_monitoring();
    void shutdown();
    
    // Metrics collection
    void record_metric(const MetricDataPoint& metric);
    void record_metrics_batch(const std::vector<MetricDataPoint>& metrics);
    std::shared_ptr<MetricsCollector> create_collector(const std::string& name);
    std::shared_ptr<MetricsCollector> get_collector(const std::string& name);
    
    // Time series data
    std::vector<MetricDataPoint> get_metric_data(const std::string& metric_name,
                                                std::chrono::system_clock::time_point start_time,
                                                std::chrono::system_clock::time_point end_time);
    std::vector<std::string> get_available_metrics() const;
    MetricTimeSeries get_time_series(const std::string& metric_name) const;
    
    // Alerting system
    bool add_alert_condition(const AlertCondition& condition);
    bool remove_alert_condition(const std::string& condition_id);
    bool update_alert_condition(const AlertCondition& condition);
    std::vector<AlertCondition> get_alert_conditions() const;
    
    // Alert notifications
    using AlertHandler = std::function<void(const AlertNotification&)>;
    bool register_alert_handler(AlertHandler handler);
    void trigger_alert(const AlertNotification& alert);
    std::vector<AlertNotification> get_active_alerts() const;
    std::vector<AlertNotification> get_alert_history(std::chrono::hours lookback = std::chrono::hours(24)) const;
    
    // Configuration
    bool update_config(const MonitoringConfig& config);
    MonitoringConfig get_config() const;
    
    // System statistics
    struct MonitoringStats {
        size_t total_metrics_collected;
        size_t active_time_series;
        size_t total_alerts_triggered;
        size_t active_alert_conditions;
        std::chrono::milliseconds average_collection_latency;
        std::chrono::milliseconds average_alert_evaluation_time;
        size_t memory_usage_bytes;
    };
    
    MonitoringStats get_system_stats() const;
    
    // Data export
    bool export_metrics_to_file(const std::string& filename, 
                               const std::string& format = "json") const;
    std::string export_metrics_to_string(const std::string& format = "json") const;
    bool export_to_external_system(const std::string& endpoint, 
                                  const std::string& format) const;

private:
    MonitoringConfig config_;
    std::atomic<bool> monitoring_active_;
    
    // Metrics storage
    std::unordered_map<std::string, MetricTimeSeries> time_series_;
    mutable std::shared_mutex time_series_mutex_;
    
    // Collectors
    std::unordered_map<std::string, std::shared_ptr<MetricsCollector>> collectors_;
    mutable std::shared_mutex collectors_mutex_;
    
    // Alerting
    std::vector<AlertCondition> alert_conditions_;
    std::vector<AlertNotification> active_alerts_;
    std::vector<AlertNotification> alert_history_;
    std::vector<AlertHandler> alert_handlers_;
    mutable std::shared_mutex alerts_mutex_;
    
    // Background threads
    std::thread metrics_collection_thread_;
    std::thread alert_evaluation_thread_;
    std::thread cleanup_thread_;
    std::atomic<bool> threads_running_;
    
    // Statistics
    MonitoringStats system_stats_;
    mutable std::mutex stats_mutex_;
    
    // Internal methods
    void metrics_collection_loop();
    void alert_evaluation_loop();
    void cleanup_loop();
    
    void process_metric(const MetricDataPoint& metric);
    void cleanup_old_data();
    bool evaluate_alert_condition(const AlertCondition& condition);
    void notify_alert_handlers(const AlertNotification& alert);
    
    std::string format_metrics_as_json() const;
    std::string format_metrics_as_prometheus() const;
    std::string format_metrics_as_influxdb() const;
    
    void update_system_stats();
};

/**
 * @brief Specialized collectors for different system components
 */
class PerformanceMetricsCollector : public MetricsCollector {
public:
    PerformanceMetricsCollector();
    
    // Performance-specific metrics
    void record_scanning_rate(double keys_per_second);
    void record_gpu_utilization(int device_id, double utilization);
    void record_memory_usage(const std::string& component, size_t bytes);
    void record_operation_latency(const std::string& operation, std::chrono::milliseconds latency);
    void record_error_count(const std::string& component, const std::string& error_type);
    void record_throughput(const std::string& component, double operations_per_second);
    
protected:
    void on_metric_recorded(const MetricDataPoint& metric) override;
};

class SystemMetricsCollector : public MetricsCollector {
public:
    SystemMetricsCollector();
    
    // System resource metrics
    void record_cpu_usage(double percentage);
    void record_memory_usage(size_t used_bytes, size_t total_bytes);
    void record_disk_usage(const std::string& path, size_t used_bytes, size_t total_bytes);
    void record_network_io(size_t bytes_sent, size_t bytes_received);
    void record_gpu_temperature(int device_id, double temperature);
    void record_gpu_power_draw(int device_id, double watts);
    
protected:
    void on_metric_recorded(const MetricDataPoint& metric) override;
};

class BusinessMetricsCollector : public MetricsCollector {
public:
    BusinessMetricsCollector();
    
    // Business/application-specific metrics
    void record_addresses_checked(size_t count);
    void record_matches_found(size_t count, const std::string& address_type);
    void record_checkpoint_created(const std::string& checkpoint_type);
    void record_validation_result(bool passed, const std::string& validation_type);
    void record_optimization_applied(const std::string& optimization_type, double improvement);
    
protected:
    void on_metric_recorded(const MetricDataPoint& metric) override;
};

/**
 * @brief Alert notification handlers
 */
namespace alert_handlers {
    
    // Console alert handler
    class ConsoleAlertHandler {
    public:
        ConsoleAlertHandler(bool use_colors = true);
        void operator()(const AlertNotification& alert);
        
    private:
        bool use_colors_;
        std::string get_color_code(AlertSeverity severity) const;
    };
    
    // File alert handler
    class FileAlertHandler {
    public:
        FileAlertHandler(const std::string& filename);
        void operator()(const AlertNotification& alert);
        
    private:
        std::string filename_;
        std::mutex file_mutex_;
    };
    
    // Email alert handler (requires external SMTP configuration)
    class EmailAlertHandler {
    public:
        EmailAlertHandler(const std::string& smtp_server, int port,
                         const std::string& username, const std::string& password,
                         const std::vector<std::string>& recipients);
        void operator()(const AlertNotification& alert);
        
    private:
        std::string smtp_server_;
        int port_;
        std::string username_;
        std::string password_;
        std::vector<std::string> recipients_;
        
        bool send_email(const std::string& subject, const std::string& body);
    };
    
    // Webhook alert handler
    class WebhookAlertHandler {
    public:
        WebhookAlertHandler(const std::string& webhook_url);
        void operator()(const AlertNotification& alert);
        
    private:
        std::string webhook_url_;
        bool send_webhook(const std::string& payload);
    };
}

/**
 * @brief Monitoring utility functions
 */
namespace monitoring_utils {
    
    // Metric utilities
    std::string metric_type_to_string(MetricType type);
    MetricType string_to_metric_type(const std::string& type_str);
    bool is_valid_metric_name(const std::string& name);
    std::string sanitize_metric_name(const std::string& name);
    
    // Alert utilities
    std::string alert_severity_to_string(AlertSeverity severity);
    AlertSeverity string_to_alert_severity(const std::string& severity_str);
    bool is_alert_condition_valid(const AlertCondition& condition);
    
    // Statistical utilities
    double calculate_average(const std::vector<double>& values);
    double calculate_percentile(const std::vector<double>& values, double percentile);
    double calculate_rate_of_change(const std::vector<MetricDataPoint>& data_points);
    
    // Configuration utilities
    MonitoringConfig create_development_monitoring_config();
    MonitoringConfig create_production_monitoring_config();
    bool validate_monitoring_config(const MonitoringConfig& config);
    
    // Export utilities
    std::string format_timestamp_for_export(const std::chrono::system_clock::time_point& timestamp,
                                           const std::string& format);
    std::string escape_string_for_format(const std::string& str, const std::string& format);
}

} // namespace monitoring
} // namespace keyhunt