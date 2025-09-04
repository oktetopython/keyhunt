#ifndef UNIFIED_ECC_CONFIG_MANAGER_H
#define UNIFIED_ECC_CONFIG_MANAGER_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>

namespace UnifiedECC {

// ============================================================================
// 配置键定义
// ============================================================================

namespace ConfigKeys {
    // 后端选择
    const std::string BACKEND = "backend";
    const std::string FALLBACK_BACKEND = "fallback_backend";

    // GPU设置
    const std::string ENABLE_GPU = "enable_gpu";
    const std::string GPU_DEVICE_ID = "gpu_device_id";
    const std::string GPU_MEMORY_LIMIT = "gpu_memory_limit";

    // 性能设置
    const std::string ENABLE_BATCH_OPS = "enable_batch_ops";
    const std::string BATCH_SIZE = "batch_size";
    const std::string ENABLE_MULTITHREADING = "enable_multithreading";
    const std::string THREAD_COUNT = "thread_count";

    // 曲线参数
    const std::string CURVE_NAME = "curve_name";
    const std::string CUSTOM_CURVE_PARAMS = "custom_curve_params";

    // 调试和监控
    const std::string ENABLE_PERFORMANCE_MONITORING = "enable_performance_monitoring";
    const std::string LOG_LEVEL = "log_level";
    const std::string LOG_FILE = "log_file";

    // 安全设置
    const std::string ENABLE_SECURE_MEMORY = "enable_secure_memory";
    const std::string RANDOM_SEED = "random_seed";

    // 缓存设置
    const std::string ENABLE_RESULT_CACHE = "enable_result_cache";
    const std::string CACHE_SIZE = "cache_size";
}

// ============================================================================
// 配置值类型
// ============================================================================

class ConfigValue {
public:
    enum Type { STRING, INT, DOUBLE, BOOL };

    ConfigValue() : type_(STRING), string_val_("") {}
    ConfigValue(const std::string& val) : type_(STRING), string_val_(val) {}
    ConfigValue(int val) : type_(INT), int_val_(val) {}
    ConfigValue(double val) : type_(DOUBLE), double_val_(val) {}
    ConfigValue(bool val) : type_(BOOL), bool_val_(val) {}

    Type type() const { return type_; }

    std::string as_string() const;
    int as_int() const;
    double as_double() const;
    bool as_bool() const;

private:
    Type type_;
    std::string string_val_;
    int int_val_;
    double double_val_;
    bool bool_val_;
};

// ============================================================================
// 配置管理器
// ============================================================================

class ConfigManager {
public:
    // 单例模式
    static ConfigManager& instance();

    // 配置加载
    bool load_from_file(const std::string& filename);
    bool load_from_string(const std::string& config_str);
    void load_from_environment();
    void load_defaults();

    // 配置设置
    void set(const std::string& key, const ConfigValue& value);
    void set(const std::string& key, const std::string& value);
    void set(const std::string& key, int value);
    void set(const std::string& key, double value);
    void set(const std::string& key, bool value);

    // 配置获取
    ConfigValue get(const std::string& key) const;
    std::string get_string(const std::string& key, const std::string& default_val = "") const;
    int get_int(const std::string& key, int default_val = 0) const;
    double get_double(const std::string& key, double default_val = 0.0) const;
    bool get_bool(const std::string& key, bool default_val = false) const;

    // 配置验证
    bool validate() const;
    std::vector<std::string> get_validation_errors() const;

    // 配置保存
    bool save_to_file(const std::string& filename) const;

    // 配置信息
    std::vector<std::string> get_all_keys() const;
    void print_config() const;

    // 配置重置
    void clear();
    void reset_to_defaults();

private:
    ConfigManager();
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    std::unordered_map<std::string, ConfigValue> config_;
    mutable std::vector<std::string> validation_errors_;

    // 私有辅助方法
    void parse_line(const std::string& line);
    std::string trim(const std::string& str) const;
    std::string to_lower(const std::string& str) const;
    bool is_valid_key(const std::string& key) const;
    void set_defaults();
};

// ============================================================================
// 配置构建器 (Builder模式)
// ============================================================================

class ConfigBuilder {
public:
    ConfigBuilder();

    // 后端配置
    ConfigBuilder& backend(const std::string& backend);
    ConfigBuilder& fallback_backend(const std::string& backend);

    // GPU配置
    ConfigBuilder& enable_gpu(bool enable = true);
    ConfigBuilder& gpu_device_id(int device_id);
    ConfigBuilder& gpu_memory_limit(size_t limit_mb);

    // 性能配置
    ConfigBuilder& enable_batch_ops(bool enable = true);
    ConfigBuilder& batch_size(int size);
    ConfigBuilder& enable_multithreading(bool enable = true);
    ConfigBuilder& thread_count(int count);

    // 曲线配置
    ConfigBuilder& curve_name(const std::string& name);

    // 调试配置
    ConfigBuilder& enable_performance_monitoring(bool enable = true);
    ConfigBuilder& log_level(const std::string& level);
    ConfigBuilder& log_file(const std::string& file);

    // 安全配置
    ConfigBuilder& enable_secure_memory(bool enable = true);
    ConfigBuilder& random_seed(uint64_t seed);

    // 缓存配置
    ConfigBuilder& enable_result_cache(bool enable = true);
    ConfigBuilder& cache_size(size_t size);

    // 构建配置
    void build();

private:
    std::unordered_map<std::string, ConfigValue> config_map_;
};

// ============================================================================
// 配置验证器
// ============================================================================

class ConfigValidator {
public:
    static bool validate_backend(const std::string& backend);
    static bool validate_curve_name(const std::string& curve);
    static bool validate_gpu_device_id(int device_id);
    static bool validate_thread_count(int count);
    static bool validate_log_level(const std::string& level);

    static std::vector<std::string> get_supported_backends();
    static std::vector<std::string> get_supported_curves();
    static std::vector<std::string> get_supported_log_levels();
};

// ============================================================================
// 便捷函数
// ============================================================================

// 全局配置访问函数
inline ConfigManager& config() {
    return ConfigManager::instance();
}

// 快速配置函数
void configure_for_performance();
void configure_for_security();
void configure_for_development();
void configure_for_production();

} // namespace UnifiedECC

#endif // UNIFIED_ECC_CONFIG_MANAGER_H