/**
 * @file logging_system.cpp
 * @brief Implementation of comprehensive logging system
 * @author KeyhuntCUDA Team
 * 
 * T048: Implementation of logging and monitoring infrastructure
 */

#include "logging_system.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <thread>

namespace keyhunt {
namespace logging {

Logger::Logger(const std::string& component_name, const LoggerConfig& config)
    : component_name_(component_name)
    , config_(config)
    , async_thread_running_(false)
    , current_file_size_(0)
{
    // Initialize statistics
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = LoggerStats();
    
    // Create log directory if needed
    if (std::find(config_.destinations.begin(), config_.destinations.end(), 
                  LogDestination::FILE) != config_.destinations.end()) {
        logging_utils::create_log_directory(config_.log_directory);
        current_log_file_ = generate_log_filename();
    }
    
    // Start async logging if enabled
    if (config_.enable_async_logging) {
        start_async_logging();
    }
}

Logger::~Logger() {
    stop_async_logging();
    flush();
}

void Logger::log(LogLevel level, const std::string& message, 
                const std::string& file, int line, const std::string& function) {
    
    if (!should_log(level)) {
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    LogEntry entry;
    entry.timestamp = std::chrono::system_clock::now();
    entry.level = level;
    entry.component = component_name_;
    entry.message = message;
    entry.thread_id = get_thread_id();
    entry.file = file;
    entry.line = line;
    entry.function = function;
    
    if (config_.enable_async_logging) {
        // Add to async buffer
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (log_buffer_.size() < config_.async_buffer_size) {
            log_buffer_.push(entry);
            buffer_cv_.notify_one();
        } else {
            // Buffer overflow - drop message and increment dropped count
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.entries_dropped++;
        }
    } else {
        // Process immediately
        process_log_entry(entry);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto log_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    update_stats(entry, log_time);
}

void Logger::log_with_fields(LogLevel level, const std::string& message,
                            const std::unordered_map<std::string, std::string>& fields,
                            const std::string& file, int line, const std::string& function) {
    
    if (!should_log(level)) {
        return;
    }
    
    LogEntry entry;
    entry.timestamp = std::chrono::system_clock::now();
    entry.level = level;
    entry.component = component_name_;
    entry.message = message;
    entry.thread_id = get_thread_id();
    entry.file = file;
    entry.line = line;
    entry.function = function;
    entry.fields = fields;
    
    if (config_.enable_async_logging) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (log_buffer_.size() < config_.async_buffer_size) {
            log_buffer_.push(entry);
            buffer_cv_.notify_one();
        }
    } else {
        process_log_entry(entry);
    }
}

void Logger::info(const std::string& message, const std::string& file, int line, const std::string& function) {
    log(LogLevel::INFO, message, file, line, function);
}

void Logger::error(const std::string& message, const std::string& file, int line, const std::string& function) {
    log(LogLevel::ERROR, message, file, line, function);
}

void Logger::warning(const std::string& message, const std::string& file, int line, const std::string& function) {
    log(LogLevel::WARNING, message, file, line, function);
}

void Logger::debug(const std::string& message, const std::string& file, int line, const std::string& function) {
    log(LogLevel::DEBUG, message, file, line, function);
}

void Logger::critical(const std::string& message, const std::string& file, int line, const std::string& function) {
    log(LogLevel::CRITICAL, message, file, line, function);
}

void Logger::start_async_logging() {
    if (!async_thread_running_) {
        async_thread_running_ = true;
        async_thread_ = std::thread(&Logger::async_logging_loop, this);
    }
}

void Logger::stop_async_logging() {
    if (async_thread_running_) {
        async_thread_running_ = false;
        buffer_cv_.notify_all();
        
        if (async_thread_.joinable()) {
            async_thread_.join();
        }
        
        // Process remaining entries
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        while (!log_buffer_.empty()) {
            process_log_entry(log_buffer_.front());
            log_buffer_.pop();
        }
    }
}

void Logger::async_logging_loop() {
    while (async_thread_running_) {
        std::unique_lock<std::mutex> lock(buffer_mutex_);
        
        // Wait for entries or timeout
        buffer_cv_.wait_for(lock, config_.flush_interval, 
            [this] { return !log_buffer_.empty() || !async_thread_running_; });
        
        // Process all available entries
        while (!log_buffer_.empty()) {
            LogEntry entry = log_buffer_.front();
            log_buffer_.pop();
            
            // Unlock while processing to avoid blocking new entries
            lock.unlock();
            process_log_entry(entry);
            lock.lock();
        }
    }
}

void Logger::process_log_entry(const LogEntry& entry) {
    std::string formatted_message = format_log_entry(entry);
    write_to_destinations(entry, formatted_message);
}

std::string Logger::format_log_entry(const LogEntry& entry) {
    switch (config_.format) {
        case LogFormat::JSON:
            return format_json(entry);
        case LogFormat::CSV:
            return format_csv(entry);
        case LogFormat::CUSTOM:
            if (custom_formatter_) {
                return custom_formatter_(entry);
            }
            // Fall through to plaintext if no custom formatter
        default:
            return format_plaintext(entry);
    }
}

std::string Logger::format_plaintext(const LogEntry& entry) {
    std::ostringstream oss;
    
    // Timestamp
    if (config_.include_timestamp) {
        oss << "[" << logging_utils::format_timestamp(entry.timestamp) << "] ";
    }
    
    // Level
    oss << "[" << level_to_string(entry.level) << "] ";
    
    // Component
    oss << "[" << entry.component << "] ";
    
    // Thread ID
    if (config_.include_thread_id) {
        oss << "[" << entry.thread_id << "] ";
    }
    
    // Source location
    if (config_.include_source_location && !entry.file.empty()) {
        std::filesystem::path file_path(entry.file);
        oss << "[" << file_path.filename().string() << ":" << entry.line << "] ";
    }
    
    // Message
    oss << entry.message;
    
    // Fields
    if (!entry.fields.empty()) {
        oss << " {";
        bool first = true;
        for (const auto& field : entry.fields) {
            if (!first) oss << ", ";
            oss << field.first << "=" << field.second;
            first = false;
        }
        oss << "}";
    }
    
    // Metrics
    if (!entry.metrics.empty()) {
        oss << " [metrics: ";
        bool first = true;
        for (const auto& metric : entry.metrics) {
            if (!first) oss << ", ";
            oss << metric.first << "=" << metric.second;
            first = false;
        }
        oss << "]";
    }
    
    return oss.str();
}

std::string Logger::format_json(const LogEntry& entry) {
    std::ostringstream oss;
    oss << "{";
    
    // Timestamp
    oss << "\"timestamp\":\"" << logging_utils::format_timestamp(entry.timestamp) << "\",";
    
    // Level
    oss << "\"level\":\"" << level_to_string(entry.level) << "\",";
    
    // Component
    oss << "\"component\":\"" << entry.component << "\",";
    
    // Message
    oss << "\"message\":\"" << escape_json_string(entry.message) << "\",";
    
    // Thread ID
    oss << "\"thread_id\":\"" << entry.thread_id << "\"";
    
    // Source location
    if (config_.include_source_location && !entry.file.empty()) {
        std::filesystem::path file_path(entry.file);
        oss << ",\"file\":\"" << file_path.filename().string() << "\",";
        oss << "\"line\":" << entry.line << ",";
        oss << "\"function\":\"" << entry.function << "\"";
    }
    
    // Fields
    if (!entry.fields.empty()) {
        oss << ",\"fields\":{";
        bool first = true;
        for (const auto& field : entry.fields) {
            if (!first) oss << ",";
            oss << "\"" << field.first << "\":\"" << escape_json_string(field.second) << "\"";
            first = false;
        }
        oss << "}";
    }
    
    // Metrics
    if (!entry.metrics.empty()) {
        oss << ",\"metrics\":{";
        bool first = true;
        for (const auto& metric : entry.metrics) {
            if (!first) oss << ",";
            oss << "\"" << metric.first << "\":" << metric.second;
            first = false;
        }
        oss << "}";
    }
    
    oss << "}";
    return oss.str();
}

void Logger::write_to_destinations(const LogEntry& entry, const std::string& formatted_message) {
    for (LogDestination dest : config_.destinations) {
        switch (dest) {
            case LogDestination::CONSOLE:
                write_to_console(entry, formatted_message);
                break;
            case LogDestination::FILE:
                write_to_file(entry, formatted_message);
                break;
            case LogDestination::CUSTOM:
                if (custom_handler_) {
                    custom_handler_(entry, formatted_message);
                }
                break;
            default:
                break;
        }
    }
}

void Logger::write_to_console(const LogEntry& entry, const std::string& message) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    
    // Write to stderr for errors, stdout for everything else
    if (entry.level >= LogLevel::ERROR) {
        std::cerr << message << std::endl;
    } else {
        std::cout << message << std::endl;
    }
}

void Logger::write_to_file(const LogEntry& entry, const std::string& message) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    
    // Open file if not already open
    if (!file_stream_ || !file_stream_->is_open()) {
        file_stream_ = std::make_unique<std::ofstream>(current_log_file_, std::ios::app);
        if (!file_stream_->is_open()) {
            std::cerr << "ERROR: Failed to open log file: " << current_log_file_ << std::endl;
            return;
        }
    }
    
    // Write message
    *file_stream_ << message << std::endl;
    
    // Update file size and check for rotation
    current_file_size_ += message.length() + 1; // +1 for newline
    if (current_file_size_ >= config_.max_file_size) {
        rotate_log_file();
    }
    
    // Update bytes written
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    stats_.bytes_written += message.length() + 1;
}

void Logger::flush() {
    if (file_stream_) {
        std::lock_guard<std::mutex> lock(output_mutex_);
        file_stream_->flush();
    }
}

bool Logger::should_log(LogLevel level) const {
    return static_cast<int>(level) >= static_cast<int>(config_.min_level);
}

void Logger::update_stats(const LogEntry& entry, std::chrono::milliseconds log_time) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_entries_logged++;
    stats_.entries_by_level[static_cast<int>(entry.level)]++;
    stats_.total_logging_time += log_time;
    
    if (stats_.total_entries_logged > 0) {
        stats_.average_log_time = std::chrono::milliseconds(
            stats_.total_logging_time.count() / stats_.total_entries_logged
        );
    }
}

std::string Logger::get_thread_id() {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
}

std::string Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        case LogLevel::ALERT: return "ALERT";
        default: return "UNKNOWN";
    }
}

std::string Logger::escape_json_string(const std::string& str) {
    std::string escaped;
    escaped.reserve(str.length() + 10); // Reserve extra space for escapes
    
    for (char c : str) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c; break;
        }
    }
    
    return escaped;
}

void Logger::rotate_log_file() {
    if (file_stream_) {
        file_stream_->close();
        file_stream_.reset();
    }
    
    // Generate new filename
    std::string new_filename = generate_log_filename();
    
    // Rename current file with timestamp
    if (std::filesystem::exists(current_log_file_)) {
        auto timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::ostringstream archived_name;
        archived_name << current_log_file_ << "." << timestamp;
        
        try {
            std::filesystem::rename(current_log_file_, archived_name.str());
            
            // Compress if enabled
            if (config_.enable_compression) {
                logging_utils::compress_log_file(archived_name.str());
            }
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to rotate log file: " << e.what() << std::endl;
        }
    }
    
    current_log_file_ = new_filename;
    current_file_size_ = 0;
    
    // Clean up old files
    logging_utils::cleanup_old_log_files(config_.log_directory, config_.max_archive_files);
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.files_rotated++;
}

std::string Logger::generate_log_filename() {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream filename;
    filename << config_.log_directory << "/" << config_.log_file_prefix 
             << "_" << component_name_ << "_" << timestamp << ".log";
    
    return filename.str();
}

Logger::LoggerStats Logger::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// LoggerManager implementation
LoggerManager& LoggerManager::instance() {
    static LoggerManager instance;
    return instance;
}

std::shared_ptr<Logger> LoggerManager::get_logger(const std::string& component_name) {
    std::shared_lock<std::shared_mutex> lock(loggers_mutex_);
    
    auto it = loggers_.find(component_name);
    if (it != loggers_.end()) {
        return it->second;
    }
    
    // Create new logger with global config
    lock.unlock();
    return create_logger(component_name, global_config_);
}

std::shared_ptr<Logger> LoggerManager::create_logger(const std::string& component_name, const LoggerConfig& config) {
    std::unique_lock<std::shared_mutex> lock(loggers_mutex_);
    
    auto logger = std::make_shared<Logger>(component_name, config);
    loggers_[component_name] = logger;
    
    return logger;
}

void LoggerManager::flush_all_loggers() {
    std::shared_lock<std::shared_mutex> lock(loggers_mutex_);
    
    for (auto& pair : loggers_) {
        pair.second->flush();
    }
}

// Utility functions implementation
namespace logging_utils {

std::string format_timestamp(const std::chrono::system_clock::time_point& timestamp) {
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        timestamp.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}

bool create_log_directory(const std::string& directory) {
    try {
        if (!std::filesystem::exists(directory)) {
            return std::filesystem::create_directories(directory);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to create log directory: " << e.what() << std::endl;
        return false;
    }
}

LoggerConfig create_production_config() {
    LoggerConfig config;
    config.min_level = LogLevel::INFO;
    config.destinations = {LogDestination::FILE};
    config.format = LogFormat::JSON;
    config.enable_async_logging = true;
    config.async_buffer_size = 50000;
    config.max_file_size = 500 * 1024 * 1024; // 500MB
    config.max_archive_files = 30;
    config.enable_compression = true;
    return config;
}

LoggerConfig create_development_config() {
    LoggerConfig config;
    config.min_level = LogLevel::DEBUG;
    config.destinations = {LogDestination::CONSOLE, LogDestination::FILE};
    config.format = LogFormat::PLAINTEXT;
    config.include_source_location = true;
    config.enable_async_logging = false;
    return config;
}

} // namespace logging_utils

} // namespace logging
} // namespace keyhunt