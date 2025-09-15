/**
 * @file logging_system.h
 * @brief Comprehensive logging and monitoring infrastructure with real-time analytics
 * @author KeyhuntCUDA Team
 * 
 * T048: Develop comprehensive logging and monitoring infrastructure with real-time analytics and alerting
 * 
 * This module provides enterprise-grade logging and monitoring for KeyhuntCUDA:
 * - Structured logging with multiple output formats (JSON, plaintext, binary)
 * - Real-time log streaming and aggregation
 * - Performance metrics collection and analysis
 * - Alert system with configurable thresholds and notifications
 * - Log rotation and archival management
 * - Distributed logging across multiple components and devices
 * - Integration with external monitoring systems (Prometheus, Grafana, etc.)
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include <chrono>
#include <functional>
#include <fstream>
#include <sstream>

namespace keyhunt {
namespace logging {

/**
 * @brief Log levels for severity classification
 */
enum class LogLevel {
    TRACE = 0,      // Detailed diagnostic information
    DEBUG = 1,      // Debug information for development
    INFO = 2,       // General information messages
    WARNING = 3,    // Warning conditions
    ERROR = 4,      // Error conditions
    CRITICAL = 5,   // Critical error conditions
    ALERT = 6       // Immediate action required
};

/**
 * @brief Log output formats
 */
enum class LogFormat {
    PLAINTEXT,      // Human-readable plaintext
    JSON,           // Structured JSON format
    BINARY,         // Compact binary format
    CSV,            // Comma-separated values
    CUSTOM          // Custom format using formatter function
};

/**
 * @brief Log destinations
 */
enum class LogDestination {
    CONSOLE,        // Standard output/error
    FILE,           // File output
    NETWORK,        // Network streaming (UDP/TCP)
    SYSLOG,         // System log
    DATABASE,       // Database logging
    MEMORY_BUFFER,  // In-memory circular buffer
    CUSTOM          // Custom destination handler
};

/**
 * @brief Structured log entry
 */
struct LogEntry {
    std::chrono::system_clock::time_point timestamp;
    LogLevel level;
    std::string component;          // Component name (e.g., "ecc", "scan", "memory")
    std::string message;            // Log message
    std::string thread_id;          // Thread identifier
    std::string file;               // Source file name
    int line;                       // Source line number
    std::string function;           // Source function name
    
    // Structured data fields
    std::unordered_map<std::string, std::string> fields;
    
    // Performance context
    std::unordered_map<std::string, double> metrics;
    
    // Device context
    int device_id;                  // GPU device ID (-1 for CPU)
    
    LogEntry()
        : timestamp(std::chrono::system_clock::now())
        , level(LogLevel::INFO)
        , line(0)
        , device_id(-1)
    {}
};

/**
 * @brief Logger configuration
 */
struct LoggerConfig {
    LogLevel min_level;                             // Minimum log level to output
    std::vector<LogDestination> destinations;       // Output destinations
    LogFormat format;                               // Output format
    bool include_timestamp;                         // Include timestamp in output
    bool include_thread_id;                         // Include thread ID
    bool include_source_location;                   // Include file/line/function
    bool enable_async_logging;                      // Enable asynchronous logging
    size_t async_buffer_size;                       // Async buffer size
    std::chrono::milliseconds flush_interval;       // Auto-flush interval
    bool enable_compression;                        // Enable log compression
    size_t max_file_size;                          // Max file size before rotation
    size_t max_archive_files;                      // Max number of archived files
    std::string log_directory;                     // Directory for log files
    std::string log_file_prefix;                   // Prefix for log files
    
    LoggerConfig()
        : min_level(LogLevel::INFO)
        , format(LogFormat::PLAINTEXT)
        , include_timestamp(true)
        , include_thread_id(true)
        , include_source_location(false)
        , enable_async_logging(true)
        , async_buffer_size(10000)
        , flush_interval(std::chrono::seconds(5))
        , enable_compression(false)
        , max_file_size(100 * 1024 * 1024) // 100MB
        , max_archive_files(10)
        , log_directory("logs")
        , log_file_prefix("keyhunt")
    {
        destinations.push_back(LogDestination::CONSOLE);
        destinations.push_back(LogDestination::FILE);
    }
};

/**
 * @brief Forward declaration for custom handlers
 */
class Logger;

/**
 * @brief Custom log destination handler
 */
using CustomLogHandler = std::function<void(const LogEntry&, const std::string&)>;

/**
 * @brief Custom log formatter
 */
using CustomLogFormatter = std::function<std::string(const LogEntry&)>;

/**
 * @brief Core logger class
 */
class Logger {
public:
    Logger(const std::string& component_name, const LoggerConfig& config = LoggerConfig());
    ~Logger();
    
    // Core logging methods
    void log(LogLevel level, const std::string& message, 
             const std::string& file = "", int line = 0, const std::string& function = "");
    
    void log_with_fields(LogLevel level, const std::string& message,
                        const std::unordered_map<std::string, std::string>& fields,
                        const std::string& file = "", int line = 0, const std::string& function = "");
    
    void log_with_metrics(LogLevel level, const std::string& message,
                         const std::unordered_map<std::string, double>& metrics,
                         const std::string& file = "", int line = 0, const std::string& function = "");
    
    // Convenience methods for different log levels
    void trace(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    void debug(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    void info(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    void warning(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    void error(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    void critical(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    void alert(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "");
    
    // Configuration management
    bool update_config(const LoggerConfig& config);
    LoggerConfig get_config() const;
    bool set_min_level(LogLevel level);
    bool add_destination(LogDestination dest);
    bool remove_destination(LogDestination dest);
    
    // Custom handlers
    bool set_custom_handler(CustomLogHandler handler);
    bool set_custom_formatter(CustomLogFormatter formatter);
    
    // Control methods
    void flush();
    void start_async_logging();
    void stop_async_logging();
    void rotate_logs();
    
    // Statistics
    struct LoggerStats {
        size_t total_entries_logged;
        size_t entries_by_level[7]; // One for each LogLevel
        size_t entries_dropped;     // Dropped due to buffer overflow
        size_t bytes_written;
        std::chrono::milliseconds total_logging_time;
        std::chrono::milliseconds average_log_time;
        size_t current_buffer_size; // For async logging
        size_t files_rotated;
    };
    
    LoggerStats get_stats() const;
    void reset_stats();

private:
    std::string component_name_;
    LoggerConfig config_;
    
    // Async logging
    std::queue<LogEntry> log_buffer_;
    std::thread async_thread_;
    std::atomic<bool> async_thread_running_;
    std::mutex buffer_mutex_;
    std::condition_variable buffer_cv_;
    
    // Output streams
    std::unique_ptr<std::ofstream> file_stream_;
    std::mutex output_mutex_;
    
    // Custom handlers
    CustomLogHandler custom_handler_;
    CustomLogFormatter custom_formatter_;
    
    // Statistics
    LoggerStats stats_;
    mutable std::mutex stats_mutex_;
    
    // Log rotation
    size_t current_file_size_;
    std::string current_log_file_;
    
    // Internal methods
    void async_logging_loop();
    void process_log_entry(const LogEntry& entry);
    std::string format_log_entry(const LogEntry& entry);
    std::string format_plaintext(const LogEntry& entry);
    std::string format_json(const LogEntry& entry);
    std::string format_csv(const LogEntry& entry);
    
    void write_to_destinations(const LogEntry& entry, const std::string& formatted_message);
    void write_to_console(const LogEntry& entry, const std::string& message);
    void write_to_file(const LogEntry& entry, const std::string& message);
    void write_to_network(const LogEntry& entry, const std::string& message);
    
    bool should_log(LogLevel level) const;
    void update_stats(const LogEntry& entry, std::chrono::milliseconds log_time);
    void rotate_log_file();
    std::string generate_log_filename();
    
    std::string get_thread_id();
    std::string level_to_string(LogLevel level);
    std::string escape_json_string(const std::string& str);
};

/**
 * @brief Global logger manager for coordinating multiple loggers
 */
class LoggerManager {
public:
    static LoggerManager& instance();
    
    // Logger management
    std::shared_ptr<Logger> get_logger(const std::string& component_name);
    std::shared_ptr<Logger> create_logger(const std::string& component_name, const LoggerConfig& config);
    bool remove_logger(const std::string& component_name);
    
    // Global configuration
    bool set_global_config(const LoggerConfig& config);
    bool set_global_min_level(LogLevel level);
    void flush_all_loggers();
    void shutdown();
    
    // Global statistics
    struct GlobalLoggerStats {
        size_t total_loggers;
        size_t total_entries_logged;
        size_t total_bytes_written;
        std::unordered_map<std::string, Logger::LoggerStats> component_stats;
    };
    
    GlobalLoggerStats get_global_stats() const;
    
    // Event broadcasting
    using LogEventHandler = std::function<void(const std::string& component, const LogEntry&)>;
    bool register_event_handler(LogEventHandler handler);
    void broadcast_log_event(const std::string& component, const LogEntry& entry);

private:
    LoggerManager() = default;
    ~LoggerManager() = default;
    
    std::unordered_map<std::string, std::shared_ptr<Logger>> loggers_;
    mutable std::shared_mutex loggers_mutex_;
    
    LoggerConfig global_config_;
    std::vector<LogEventHandler> event_handlers_;
    mutable std::mutex handlers_mutex_;
};

/**
 * @brief Macro helpers for convenient logging with source location
 */
#define LOG_TRACE(logger, message) \
    (logger)->trace((message), __FILE__, __LINE__, __FUNCTION__)

#define LOG_DEBUG(logger, message) \
    (logger)->debug((message), __FILE__, __LINE__, __FUNCTION__)

#define LOG_INFO(logger, message) \
    (logger)->info((message), __FILE__, __LINE__, __FUNCTION__)

#define LOG_WARNING(logger, message) \
    (logger)->warning((message), __FILE__, __LINE__, __FUNCTION__)

#define LOG_ERROR(logger, message) \
    (logger)->error((message), __FILE__, __LINE__, __FUNCTION__)

#define LOG_CRITICAL(logger, message) \
    (logger)->critical((message), __FILE__, __LINE__, __FUNCTION__)

#define LOG_ALERT(logger, message) \
    (logger)->alert((message), __FILE__, __LINE__, __FUNCTION__)

// Macros with fields
#define LOG_INFO_WITH_FIELDS(logger, message, fields) \
    (logger)->log_with_fields(keyhunt::logging::LogLevel::INFO, (message), (fields), __FILE__, __LINE__, __FUNCTION__)

#define LOG_ERROR_WITH_FIELDS(logger, message, fields) \
    (logger)->log_with_fields(keyhunt::logging::LogLevel::ERROR, (message), (fields), __FILE__, __LINE__, __FUNCTION__)

// Macros with metrics
#define LOG_INFO_WITH_METRICS(logger, message, metrics) \
    (logger)->log_with_metrics(keyhunt::logging::LogLevel::INFO, (message), (metrics), __FILE__, __LINE__, __FUNCTION__)

/**
 * @brief Utility functions for logging system
 */
namespace logging_utils {
    
    // Log level utilities
    std::string log_level_to_string(LogLevel level);
    LogLevel string_to_log_level(const std::string& level_str);
    bool is_valid_log_level(int level);
    
    // Formatting utilities
    std::string format_timestamp(const std::chrono::system_clock::time_point& timestamp);
    std::string format_duration(const std::chrono::milliseconds& duration);
    std::string format_bytes(size_t bytes);
    
    // Configuration utilities
    LoggerConfig create_development_config();
    LoggerConfig create_production_config();
    LoggerConfig create_debug_config();
    bool validate_logger_config(const LoggerConfig& config);
    
    // File utilities
    bool create_log_directory(const std::string& directory);
    std::vector<std::string> find_log_files(const std::string& directory, const std::string& prefix);
    bool compress_log_file(const std::string& filename);
    bool cleanup_old_log_files(const std::string& directory, size_t max_files);
    
    // Network utilities
    bool is_valid_network_destination(const std::string& address, int port);
    std::string format_syslog_message(const LogEntry& entry);
}

} // namespace logging
} // namespace keyhunt