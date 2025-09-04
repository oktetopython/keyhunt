#include "../include/config_manager.h"
#include <iostream>
#include <cassert>
#include <fstream>

using namespace UnifiedECC;

void test_basic_config_operations() {
    std::cout << "=== Testing Basic Config Operations ===" << std::endl;

    ConfigManager& config = ConfigManager::instance();

    // 测试设置和获取
    config.set("test_key", "test_value");
    assert(config.get_string("test_key") == "test_value");

    config.set("test_int", 42);
    assert(config.get_int("test_int") == 42);

    config.set("test_bool", true);
    assert(config.get_bool("test_bool") == true);

    config.set("test_double", 3.14);
    assert(std::abs(config.get_double("test_double") - 3.14) < 0.001);

    std::cout << "✅ Basic config operations test passed!" << std::endl;
}

void test_config_builder() {
    std::cout << "\n=== Testing Config Builder ===" << std::endl;

    ConfigBuilder builder;
    builder.backend("cpu_optimized")
           .enable_gpu(true)
           .gpu_device_id(1)
           .enable_batch_ops(true)
           .batch_size(1024)
           .curve_name("secp256k1")
           .enable_performance_monitoring(true)
           .log_level("debug")
           .build();

    ConfigManager& config = ConfigManager::instance();
    assert(config.get_string("backend") == "cpu_optimized");
    assert(config.get_bool("enable_gpu") == true);
    assert(config.get_int("gpu_device_id") == 1);
    assert(config.get_bool("enable_batch_ops") == true);
    assert(config.get_int("batch_size") == 1024);
    assert(config.get_string("curve_name") == "secp256k1");
    assert(config.get_bool("enable_performance_monitoring") == true);
    assert(config.get_string("log_level") == "debug");

    std::cout << "✅ Config builder test passed!" << std::endl;
}

void test_config_validation() {
    std::cout << "\n=== Testing Config Validation ===" << std::endl;

    ConfigManager& config = ConfigManager::instance();

    // 测试有效配置
    config.set("backend", "cpu_optimized");
    config.set("curve_name", "secp256k1");
    config.set("gpu_device_id", 0);
    config.set("thread_count", 4);
    config.set("log_level", "info");

    assert(config.validate() == true);

    // 测试无效配置
    config.set("backend", "invalid_backend");
    assert(config.validate() == false);

    config.set("backend", "cpu_optimized"); // 恢复有效值
    config.set("gpu_device_id", -1);
    assert(config.validate() == false);

    config.set("gpu_device_id", 0); // 恢复有效值
    assert(config.validate() == true);

    std::cout << "✅ Config validation test passed!" << std::endl;
}

void test_config_file_operations() {
    std::cout << "\n=== Testing Config File Operations ===" << std::endl;

    const std::string test_file = "test_config.txt";

    // 创建测试配置文件
    std::ofstream out_file(test_file);
    out_file << "# Test configuration file" << std::endl;
    out_file << "backend = cpu_generic" << std::endl;
    out_file << "enable_gpu = false" << std::endl;
    out_file << "gpu_device_id = 2" << std::endl;
    out_file << "thread_count = 8" << std::endl;
    out_file << "curve_name = secp256r1" << std::endl;
    out_file << "log_level = warning" << std::endl;
    out_file.close();

    // 加载配置文件
    ConfigManager& config = ConfigManager::instance();
    assert(config.load_from_file(test_file) == true);

    // 验证加载的值
    assert(config.get_string("backend") == "cpu_generic");
    assert(config.get_bool("enable_gpu") == false);
    assert(config.get_int("gpu_device_id") == 2);
    assert(config.get_int("thread_count") == 8);
    assert(config.get_string("curve_name") == "secp256r1");
    assert(config.get_string("log_level") == "warning");

    // 保存配置到新文件
    const std::string save_file = "saved_config.txt";
    assert(config.save_to_file(save_file) == true);

    // 清理测试文件
    std::remove(test_file.c_str());
    std::remove(save_file.c_str());

    std::cout << "✅ Config file operations test passed!" << std::endl;
}

void test_preset_configurations() {
    std::cout << "\n=== Testing Preset Configurations ===" << std::endl;

    // 测试性能配置
    configure_for_performance();
    ConfigManager& config = ConfigManager::instance();
    assert(config.get_string("backend") == "auto");
    assert(config.get_bool("enable_gpu") == true);
    assert(config.get_bool("enable_batch_ops") == true);
    assert(config.get_int("batch_size") == 2048);

    // 测试安全配置
    configure_for_security();
    assert(config.get_string("backend") == "cpu_optimized");
    assert(config.get_bool("enable_gpu") == false);
    assert(config.get_bool("enable_secure_memory") == true);

    // 测试开发配置
    configure_for_development();
    assert(config.get_string("backend") == "cpu_generic");
    assert(config.get_bool("enable_gpu") == false);
    assert(config.get_string("log_level") == "debug");

    // 测试生产配置
    configure_for_production();
    assert(config.get_string("backend") == "auto");
    assert(config.get_bool("enable_gpu") == true);
    assert(config.get_string("log_level") == "warning");

    std::cout << "✅ Preset configurations test passed!" << std::endl;
}

void test_config_string_parsing() {
    std::cout << "\n=== Testing Config String Parsing ===" << std::endl;

    std::string config_str = R"(
        # Test configuration string
        backend = cuda_gecc
        enable_gpu = true
        thread_count = 16
        curve_name = "secp256k1"
        log_level = error
    )";

    ConfigManager& config = ConfigManager::instance();
    assert(config.load_from_string(config_str) == true);

    assert(config.get_string("backend") == "cuda_gecc");
    assert(config.get_bool("enable_gpu") == true);
    assert(config.get_int("thread_count") == 16);
    assert(config.get_string("curve_name") == "secp256k1");
    assert(config.get_string("log_level") == "error");

    std::cout << "✅ Config string parsing test passed!" << std::endl;
}

void test_config_printing() {
    std::cout << "\n=== Testing Config Printing ===" << std::endl;

    ConfigManager& config = ConfigManager::instance();
    config.set("test_print_key", "test_print_value");

    // 测试打印功能（不会失败，只是输出到控制台）
    config.print_config();

    std::vector<std::string> keys = config.get_all_keys();
    assert(keys.size() > 0);

    std::cout << "✅ Config printing test passed!" << std::endl;
}

int main() {
    std::cout << "Unified ECC Configuration Manager Test Suite" << std::endl;
    std::cout << "============================================" << std::endl;

    try {
        test_basic_config_operations();
        test_config_builder();
        test_config_validation();
        test_config_file_operations();
        test_preset_configurations();
        test_config_string_parsing();
        test_config_printing();

        std::cout << "\n🎉 All configuration tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}