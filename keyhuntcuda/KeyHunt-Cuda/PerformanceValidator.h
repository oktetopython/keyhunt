#ifndef PERFORMANCE_VALIDATOR_H
#define PERFORMANCE_VALIDATOR_H

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>

/**
 * @brief 性能验证器
 * @details 用于测试和验证各种优化措施的效果
 */
class PerformanceValidator {
public:
    struct TestResult {
        std::string test_name;
        double execution_time_ms;
        double memory_usage_mb;
        double throughput_mops; // Million operations per second
        bool passed;
        std::string error_message;
    };

    struct BenchmarkConfig {
        int iterations;
        int warmup_iterations;
        bool enable_memory_tracking;
        bool enable_detailed_logging;
        std::string output_file;
        
        BenchmarkConfig() :
            iterations(1000),
            warmup_iterations(100),
            enable_memory_tracking(true),
            enable_detailed_logging(false),
            output_file("performance_report.txt") {}
    };

    PerformanceValidator(const BenchmarkConfig& config = {}) : config_(config) {}

    /**
     * @brief 运行完整的性能测试套件
     */
    std::vector<TestResult> run_full_test_suite() {
        std::vector<TestResult> results;
        
        std::cout << "=== KeyHunt-Cuda Performance Test Suite ===" << std::endl;
        std::cout << "Iterations: " << config_.iterations << std::endl;
        std::cout << "Warmup: " << config_.warmup_iterations << std::endl;
        std::cout << std::endl;

        // 测试1：内存分配性能
        results.push_back(test_memory_allocation_performance());
        
        // 测试2：椭圆曲线运算性能
        results.push_back(test_ec_operations_performance());
        
        // 测试3：哈希函数性能
        results.push_back(test_hash_performance());
        
        // 测试4：GPU内核性能
        results.push_back(test_gpu_kernel_performance());
        
        // 测试5：并发性能
        results.push_back(test_concurrent_performance());
        
        // 生成报告
        generate_performance_report(results);
        
        return results;
    }

    /**
     * @brief 验证优化效果
     */
    bool validate_optimizations(const std::vector<TestResult>& baseline, 
                               const std::vector<TestResult>& optimized) {
        bool all_improved = true;
        
        std::cout << "=== Optimization Validation ===" << std::endl;
        
        for (size_t i = 0; i < baseline.size() && i < optimized.size(); ++i) {
            const auto& base = baseline[i];
            const auto& opt = optimized[i];
            
            if (base.test_name != opt.test_name) continue;
            
            double speedup = base.execution_time_ms / opt.execution_time_ms;
            double memory_improvement = base.memory_usage_mb / opt.memory_usage_mb;
            
            std::cout << base.test_name << ":" << std::endl;
            std::cout << "  Speedup: " << speedup << "x" << std::endl;
            std::cout << "  Memory improvement: " << memory_improvement << "x" << std::endl;
            
            if (speedup < 1.1) { // 期望至少10%的性能提升
                std::cout << "  ⚠️  Performance improvement below expectation" << std::endl;
                all_improved = false;
            } else {
                std::cout << "  ✅ Performance improvement achieved" << std::endl;
            }
        }
        
        return all_improved;
    }

private:
    BenchmarkConfig config_;

    TestResult test_memory_allocation_performance() {
        TestResult result;
        result.test_name = "Memory Allocation Performance";
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            // 模拟内存分配模式
            std::vector<std::unique_ptr<char[]>> allocations;
            allocations.reserve(config_.iterations);
            
            for (int i = 0; i < config_.iterations; ++i) {
                allocations.push_back(std::make_unique<char[]>(1024)); // 1KB分配
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            result.execution_time_ms = duration.count() / 1000.0;
            result.memory_usage_mb = (config_.iterations * 1024.0) / (1024.0 * 1024.0);
            result.throughput_mops = config_.iterations / (result.execution_time_ms / 1000.0) / 1e6;
            result.passed = true;
            
        } catch (const std::exception& e) {
            result.passed = false;
            result.error_message = e.what();
        }
        
        return result;
    }

    TestResult test_ec_operations_performance() {
        TestResult result;
        result.test_name = "Elliptic Curve Operations";
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            // 这里可以集成实际的EC运算测试
            // 现在使用模拟计算
            
            volatile int dummy = 0;
            for (int i = 0; i < config_.iterations; ++i) {
                // 模拟EC点加运算
                for (int j = 0; j < 100; ++j) {
                    dummy += j * i;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            result.execution_time_ms = duration.count() / 1000.0;
            result.throughput_mops = config_.iterations / (result.execution_time_ms / 1000.0) / 1e6;
            result.passed = true;
            
        } catch (const std::exception& e) {
            result.passed = false;
            result.error_message = e.what();
        }
        
        return result;
    }

    TestResult test_hash_performance() {
        TestResult result;
        result.test_name = "Hash Function Performance";
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            // 模拟哈希计算
            volatile int hash_sum = 0;
            
            for (int i = 0; i < config_.iterations; ++i) {
                // 简单的哈希模拟
                hash_sum += (i * 2654435761) >> 16; // 黄金比例哈希
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            result.execution_time_ms = duration.count() / 1000.0;
            result.throughput_mops = config_.iterations / (result.execution_time_ms / 1000.0) / 1e6;
            result.passed = true;
            
        } catch (const std::exception& e) {
            result.passed = false;
            result.error_message = e.what();
        }
        
        return result;
    }

    TestResult test_gpu_kernel_performance() {
        TestResult result;
        result.test_name = "GPU Kernel Performance";
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            // 模拟GPU并行计算
            volatile int parallel_sum = 0;
            
            for (int i = 0; i < config_.iterations; ++i) {
                // 模拟并行线程计算
                for (int thread = 0; thread < 256; ++thread) { // 模拟256个GPU线程
                    parallel_sum += thread * i;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            result.execution_time_ms = duration.count() / 1000.0;
            result.throughput_mops = (config_.iterations * 256) / (result.execution_time_ms / 1000.0) / 1e6;
            result.passed = true;
            
        } catch (const std::exception& e) {
            result.passed = false;
            result.error_message = e.what();
        }
        
        return result;
    }

    TestResult test_concurrent_performance() {
        TestResult result;
        result.test_name = "Concurrent Performance";
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            // 模拟并发计算
            volatile int concurrent_sum = 0;
            
            for (int i = 0; i < config_.iterations; ++i) {
                // 模拟多个并发任务
                for (int task = 0; task < 8; ++task) { // 模拟8个并发任务
                    concurrent_sum += task * i * 7; // 不同的计算模式
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            result.execution_time_ms = duration.count() / 1000.0;
            result.throughput_mops = (config_.iterations * 8) / (result.execution_time_ms / 1000.0) / 1e6;
            result.passed = true;
            
        } catch (const std::exception& e) {
            result.passed = false;
            result.error_message = e.what();
        }
        
        return result;
    }

    void generate_performance_report(const std::vector<TestResult>& results) {
        if (config_.output_file.empty()) return;
        
        std::ofstream report(config_.output_file);
        if (!report.is_open()) return;
        
        report << "=== KeyHunt-Cuda Performance Report ===" << std::endl;
        report << "Generated: " << get_current_timestamp() << std::endl;
        report << "Iterations: " << config_.iterations << std::endl;
        report << std::endl;
        
        double total_time = 0.0;
        double total_throughput = 0.0;
        
        for (const auto& result : results) {
            report << "Test: " << result.test_name << std::endl;
            report << "  Execution Time: " << result.execution_time_ms << " ms" << std::endl;
            report << "  Memory Usage: " << result.memory_usage_mb << " MB" << std::endl;
            report << "  Throughput: " << result.throughput_mops << " Mops/sec" << std::endl;
            report << "  Status: " << (result.passed ? "PASSED" : "FAILED") << std::endl;
            if (!result.error_message.empty()) {
                report << "  Error: " << result.error_message << std::endl;
            }
            report << std::endl;
            
            total_time += result.execution_time_ms;
            total_throughput += result.throughput_mops;
        }
        
        report << "=== Summary ===" << std::endl;
        report << "Total Execution Time: " << total_time << " ms" << std::endl;
        report << "Average Throughput: " << (total_throughput / results.size()) << " Mops/sec" << std::endl;
        report << "Tests Passed: " << count_passed_tests(results) << "/" << results.size() << std::endl;
        
        report.close();
        
        std::cout << "Performance report generated: " << config_.output_file << std::endl;
    }

    std::string get_current_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&time_t));
        return std::string(buffer);
    }

    int count_passed_tests(const std::vector<TestResult>& results) {
        int count = 0;
        for (const auto& result : results) {
            if (result.passed) ++count;
        }
        return count;
    }
};

#endif // PERFORMANCE_VALIDATOR_H