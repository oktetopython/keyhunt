#include "../include/config_manager.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace UnifiedECC {

// ============================================================================
// ConfigValue 实现
// ============================================================================

std::string ConfigValue::as_string() const {
    switch (type_) {
        case STRING: return string_val_;
        case INT: return std::to_string(int_val_);
        case DOUBLE: return std::to_string(double_val_);
        case BOOL: return bool_val_ ? "true" : "false";
        default: return "";
    }
}

int ConfigValue::as_int() const {
    switch (type_) {
        case STRING: return std::stoi(string_val_);
        case INT: return int_val_;
        case DOUBLE: return static_cast<int>(double_val_);
        case BOOL: return bool_val_ ? 1 : 0;
        default: return 0;
    }
}

double ConfigValue::as_double() const {
    switch (type_) {
        case STRING: return std::stod(string_val_);
        case INT: return static_cast<double>(int_val_);
        case DOUBLE: return double_val_;
        case BOOL: return bool_val_ ? 1.0 : 0.0;
        default: return 0.0;
    }
}

bool ConfigValue::as_bool() const {
    switch (type_) {
        case STRING: {
            std::string lower = string_val_;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            return lower == "true" || lower == "1" || lower == "yes" || lower == "on";
        }
        case INT: return int_val_ != 0;
        case DOUBLE: return double_val_ != 0.0;
        case BOOL: return bool_val_;
        default: return false;
    }
}

// ============================================================================
// ConfigManager 实现
// ============================================================================

ConfigManager& ConfigManager::instance() {
    static ConfigManager instance;
    return instance;
}

ConfigManager::ConfigManager() {
    set_defaults();
}

void ConfigManager::set_defaults() {
    // 后端设置
    set(ConfigKeys::BACKEND, std::string("cpu_optimized"));
    set(ConfigKeys::FALLBACK_BACKEND, std::string("cpu_generic"));

    // GPU设置
    set(ConfigKeys::ENABLE_GPU, true);
    set(ConfigKeys::GPU_DEVICE_ID, 0);
    set(ConfigKeys::GPU_MEMORY_LIMIT, 1024); // MB

    // 性能设置
    set(ConfigKeys::ENABLE_BATCH_OPS, true);
    set(ConfigKeys::BATCH_SIZE, 1024);
    set(ConfigKeys::ENABLE_MULTITHREADING, true);
    set(ConfigKeys::THREAD_COUNT, 4);

    // 曲线参数
    set(ConfigKeys::CURVE_NAME, std::string("secp256k1"));

    // 调试和监控
    set(ConfigKeys::ENABLE_PERFORMANCE_MONITORING, false);
    set(ConfigKeys::LOG_LEVEL, std::string("info"));
    set(ConfigKeys::LOG_FILE, std::string("ecc.log"));

    // 安全设置
    set(ConfigKeys::ENABLE_SECURE_MEMORY, true);
    set(ConfigKeys::RANDOM_SEED, 0); // 0表示使用系统随机种子

    // 缓存设置
    set(ConfigKeys::ENABLE_RESULT_CACHE, true);
    set(ConfigKeys::CACHE_SIZE, 1000); // 缓存条目数
}

bool ConfigManager::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        return false;
    }

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        try {
            parse_line(line);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line " << line_number << " in " << filename
                      << ": " << e.what() << std::endl;
            return false;
        }
    }

    return true;
}

bool ConfigManager::load_from_string(const std::string& config_str) {
    std::istringstream stream(config_str);
    std::string line;

    while (std::getline(stream, line)) {
        try {
            parse_line(line);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing config string: " << e.what() << std::endl;
            return false;
        }
    }

    return true;
}

void ConfigManager::load_from_environment() {
    // 从环境变量加载配置
    const char* env_vars[] = {
        "ECC_BACKEND",
        "ECC_ENABLE_GPU",
        "ECC_GPU_DEVICE_ID",
        "ECC_THREAD_COUNT",
        "ECC_CURVE_NAME",
        "ECC_LOG_LEVEL",
        nullptr
    };

    for (int i = 0; env_vars[i] != nullptr; ++i) {
        const char* value = std::getenv(env_vars[i]);
        if (value) {
            std::string key = to_lower(env_vars[i] + 4); // 移除"ECC_"前缀
            set(key, std::string(value));
        }
    }
}

void ConfigManager::parse_line(const std::string& line) {
    std::string trimmed = trim(line);

    // 跳过空行和注释
    if (trimmed.empty() || trimmed[0] == '#') {
        return;
    }

    // 查找等号
    size_t equals_pos = trimmed.find('=');
    if (equals_pos == std::string::npos) {
        throw std::runtime_error("Invalid config line format: " + trimmed);
    }

    std::string key = trim(trimmed.substr(0, equals_pos));
    std::string value = trim(trimmed.substr(equals_pos + 1));

    // 移除引号
    if (!value.empty() && value[0] == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
    }

    set(key, value);
}

std::string ConfigManager::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t");
    if (first == std::string::npos) return "";

    size_t last = str.find_last_not_of(" \t");
    return str.substr(first, last - first + 1);
}

std::string ConfigManager::to_lower(const std::string& str) const {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

void ConfigManager::set(const std::string& key, const ConfigValue& value) {
    config_[key] = value;
}

void ConfigManager::set(const std::string& key, const std::string& value) {
    config_[key] = ConfigValue(value);
}

void ConfigManager::set(const std::string& key, int value) {
    config_[key] = ConfigValue(value);
}

void ConfigManager::set(const std::string& key, double value) {
    config_[key] = ConfigValue(value);
}

void ConfigManager::set(const std::string& key, bool value) {
    config_[key] = ConfigValue(value);
}

ConfigValue ConfigManager::get(const std::string& key) const {
    auto it = config_.find(key);
    if (it != config_.end()) {
        return it->second;
    }
    return ConfigValue(); // 返回默认值
}

std::string ConfigManager::get_string(const std::string& key, const std::string& default_val) const {
    auto it = config_.find(key);
    return (it != config_.end()) ? it->second.as_string() : default_val;
}

int ConfigManager::get_int(const std::string& key, int default_val) const {
    auto it = config_.find(key);
    return (it != config_.end()) ? it->second.as_int() : default_val;
}

double ConfigManager::get_double(const std::string& key, double default_val) const {
    auto it = config_.find(key);
    return (it != config_.end()) ? it->second.as_double() : default_val;
}

bool ConfigManager::get_bool(const std::string& key, bool default_val) const {
    auto it = config_.find(key);
    return (it != config_.end()) ? it->second.as_bool() : default_val;
}

bool ConfigManager::validate() const {
    validation_errors_.clear();

    // 验证后端
    std::string backend = get_string(ConfigKeys::BACKEND);
    if (!ConfigValidator::validate_backend(backend)) {
        validation_errors_.push_back("Invalid backend: " + backend);
    }

    // 验证曲线名称
    std::string curve = get_string(ConfigKeys::CURVE_NAME);
    if (!ConfigValidator::validate_curve_name(curve)) {
        validation_errors_.push_back("Invalid curve name: " + curve);
    }

    // 验证GPU设备ID
    int gpu_id = get_int(ConfigKeys::GPU_DEVICE_ID);
    if (!ConfigValidator::validate_gpu_device_id(gpu_id)) {
        validation_errors_.push_back("Invalid GPU device ID: " + std::to_string(gpu_id));
    }

    // 验证线程数
    int thread_count = get_int(ConfigKeys::THREAD_COUNT);
    if (!ConfigValidator::validate_thread_count(thread_count)) {
        validation_errors_.push_back("Invalid thread count: " + std::to_string(thread_count));
    }

    // 验证日志级别
    std::string log_level = get_string(ConfigKeys::LOG_LEVEL);
    if (!ConfigValidator::validate_log_level(log_level)) {
        validation_errors_.push_back("Invalid log level: " + log_level);
    }

    return validation_errors_.empty();
}

std::vector<std::string> ConfigManager::get_validation_errors() const {
    return validation_errors_;
}

bool ConfigManager::save_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "# Unified ECC Configuration File" << std::endl;
    file << "# Generated automatically" << std::endl;
    file << std::endl;

    for (const auto& pair : config_) {
        file << pair.first << " = " << pair.second.as_string() << std::endl;
    }

    return true;
}

std::vector<std::string> ConfigManager::get_all_keys() const {
    std::vector<std::string> keys;
    for (const auto& pair : config_) {
        keys.push_back(pair.first);
    }
    return keys;
}

void ConfigManager::print_config() const {
    std::cout << "=== Current Configuration ===" << std::endl;
    for (const auto& pair : config_) {
        std::cout << pair.first << " = " << pair.second.as_string() << std::endl;
    }
    std::cout << "=============================" << std::endl;
}

void ConfigManager::clear() {
    config_.clear();
}

void ConfigManager::reset_to_defaults() {
    clear();
    set_defaults();
}

// ============================================================================
// ConfigBuilder 实现
// ============================================================================

ConfigBuilder::ConfigBuilder() = default;

ConfigBuilder& ConfigBuilder::backend(const std::string& backend) {
    config_map_[ConfigKeys::BACKEND] = ConfigValue(backend);
    return *this;
}

ConfigBuilder& ConfigBuilder::fallback_backend(const std::string& backend) {
    config_map_[ConfigKeys::FALLBACK_BACKEND] = ConfigValue(backend);
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_gpu(bool enable) {
    config_map_[ConfigKeys::ENABLE_GPU] = ConfigValue(enable);
    return *this;
}

ConfigBuilder& ConfigBuilder::gpu_device_id(int device_id) {
    config_map_[ConfigKeys::GPU_DEVICE_ID] = ConfigValue(device_id);
    return *this;
}

ConfigBuilder& ConfigBuilder::gpu_memory_limit(size_t limit_mb) {
    config_map_[ConfigKeys::GPU_MEMORY_LIMIT] = ConfigValue(static_cast<int>(limit_mb));
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_batch_ops(bool enable) {
    config_map_[ConfigKeys::ENABLE_BATCH_OPS] = ConfigValue(enable);
    return *this;
}

ConfigBuilder& ConfigBuilder::batch_size(int size) {
    config_map_[ConfigKeys::BATCH_SIZE] = ConfigValue(size);
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_multithreading(bool enable) {
    config_map_[ConfigKeys::ENABLE_MULTITHREADING] = ConfigValue(enable);
    return *this;
}

ConfigBuilder& ConfigBuilder::thread_count(int count) {
    config_map_[ConfigKeys::THREAD_COUNT] = ConfigValue(count);
    return *this;
}

ConfigBuilder& ConfigBuilder::curve_name(const std::string& name) {
    config_map_[ConfigKeys::CURVE_NAME] = ConfigValue(name);
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_performance_monitoring(bool enable) {
    config_map_[ConfigKeys::ENABLE_PERFORMANCE_MONITORING] = ConfigValue(enable);
    return *this;
}

ConfigBuilder& ConfigBuilder::log_level(const std::string& level) {
    config_map_[ConfigKeys::LOG_LEVEL] = ConfigValue(level);
    return *this;
}

ConfigBuilder& ConfigBuilder::log_file(const std::string& file) {
    config_map_[ConfigKeys::LOG_FILE] = ConfigValue(file);
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_secure_memory(bool enable) {
    config_map_[ConfigKeys::ENABLE_SECURE_MEMORY] = ConfigValue(enable);
    return *this;
}

ConfigBuilder& ConfigBuilder::random_seed(uint64_t seed) {
    config_map_[ConfigKeys::RANDOM_SEED] = ConfigValue(static_cast<int>(seed));
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_result_cache(bool enable) {
    config_map_[ConfigKeys::ENABLE_RESULT_CACHE] = ConfigValue(enable);
    return *this;
}

ConfigBuilder& ConfigBuilder::cache_size(size_t size) {
    config_map_[ConfigKeys::CACHE_SIZE] = ConfigValue(static_cast<int>(size));
    return *this;
}

void ConfigBuilder::build() {
    ConfigManager& mgr = ConfigManager::instance();
    for (const auto& pair : config_map_) {
        mgr.set(pair.first, pair.second);
    }
}

// ============================================================================
// ConfigValidator 实现
// ============================================================================

bool ConfigValidator::validate_backend(const std::string& backend) {
    std::vector<std::string> supported = get_supported_backends();
    return std::find(supported.begin(), supported.end(), backend) != supported.end();
}

bool ConfigValidator::validate_curve_name(const std::string& curve) {
    std::vector<std::string> supported = get_supported_curves();
    return std::find(supported.begin(), supported.end(), curve) != supported.end();
}

bool ConfigValidator::validate_gpu_device_id(int device_id) {
    return device_id >= 0 && device_id < 16; // 假设最多16个GPU设备
}

bool ConfigValidator::validate_thread_count(int count) {
    return count > 0 && count <= 1024; // 合理的线程数范围
}

bool ConfigValidator::validate_log_level(const std::string& level) {
    std::vector<std::string> supported = get_supported_log_levels();
    return std::find(supported.begin(), supported.end(), level) != supported.end();
}

std::vector<std::string> ConfigValidator::get_supported_backends() {
    return {"cpu_generic", "cpu_optimized", "cuda_gecc", "cuda_keyhunt", "auto"};
}

std::vector<std::string> ConfigValidator::get_supported_curves() {
    return {"secp256k1", "secp256r1", "ed25519", "custom"};
}

std::vector<std::string> ConfigValidator::get_supported_log_levels() {
    return {"debug", "info", "warning", "error", "fatal"};
}

// ============================================================================
// 便捷配置函数
// ============================================================================

void configure_for_performance() {
    ConfigBuilder()
        .backend("auto")
        .enable_gpu(true)
        .enable_batch_ops(true)
        .batch_size(2048)
        .enable_multithreading(true)
        .thread_count(8)
        .enable_performance_monitoring(true)
        .enable_result_cache(true)
        .cache_size(5000)
        .build();
}

void configure_for_security() {
    ConfigBuilder()
        .backend("cpu_optimized")
        .enable_secure_memory(true)
        .enable_gpu(false) // 避免GPU侧信道攻击
        .enable_result_cache(false) // 避免缓存侧信道
        .random_seed(0) // 使用系统随机种子
        .build();
}

void configure_for_development() {
    ConfigBuilder()
        .backend("cpu_generic")
        .enable_gpu(false)
        .enable_performance_monitoring(true)
        .log_level("debug")
        .enable_batch_ops(false)
        .enable_multithreading(false)
        .build();
}

void configure_for_production() {
    ConfigBuilder()
        .backend("auto")
        .enable_gpu(true)
        .enable_batch_ops(true)
        .enable_secure_memory(true)
        .log_level("warning")
        .enable_performance_monitoring(false)
        .build();
}

} // namespace UnifiedECC