/**
 * @file secp256k1_math_optimized.cpp
 * @brief C++ implementation for optimized modular arithmetic
 * @author KeyhuntCUDA Team
 * 
 * T036: Optimize modular arithmetic kernels with assembly-level optimizations for performance
 * 
 * Provides C++ wrapper implementation for assembly-optimized modular arithmetic
 * operations with performance monitoring and architecture-specific optimizations.
 */

#include "secp256k1_math_optimized.h"
#include "gpu_memory_manager.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>

namespace keyhunt {
namespace ecc {
namespace gpu {
namespace optimized {

// OptimizedModularArithmetic Implementation
OptimizedModularArithmetic::OptimizedModularArithmetic(const OptimizationConfig& config)
    : config_(config), initialized_(false), device_id_(0), d_temp_a_(nullptr),
      d_temp_b_(nullptr), d_temp_r_(nullptr), allocated_size_(0),
      compute_stream_(nullptr), memory_stream_(nullptr) {
}

OptimizedModularArithmetic::~OptimizedModularArithmetic() {
    cleanup();
}

bool OptimizedModularArithmetic::initialize(int device_id) {
    if (initialized_) return true;
    
    device_id_ = device_id;
    
    // Set device
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) return false;
    
    // Configure for target architecture
    configure_for_architecture();
    
    // Create CUDA streams
    err = cudaStreamCreate(&compute_stream_);
    if (err != cudaSuccess) return false;
    
    err = cudaStreamCreate(&memory_stream_);
    if (err != cudaSuccess) {
        cudaStreamDestroy(compute_stream_);
        return false;
    }
    
    // Create CUDA events for timing
    err = cudaEventCreate(&start_event_);
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }
    
    err = cudaEventCreate(&stop_event_);
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }
    
    // Allocate initial device memory
    if (!allocate_device_memory(10000)) { // Default allocation for 10K operations
        cleanup();
        return false;
    }
    
    initialized_ = true;
    reset_performance_counters();
    
    return true;
}

void OptimizedModularArithmetic::cleanup() {
    if (!initialized_) return;
    
    free_device_memory();
    
    if (compute_stream_) {
        cudaStreamDestroy(compute_stream_);
        compute_stream_ = nullptr;
    }
    
    if (memory_stream_) {
        cudaStreamDestroy(memory_stream_);
        memory_stream_ = nullptr;
    }
    
    if (start_event_) {
        cudaEventDestroy(start_event_);
        start_event_ = nullptr;
    }
    
    if (stop_event_) {
        cudaEventDestroy(stop_event_);
        stop_event_ = nullptr;
    }
    
    initialized_ = false;
}

cudaError_t OptimizedModularArithmetic::modular_multiply(const BigInt256& a, const BigInt256& b, BigInt256& result) {
    if (!initialized_) return cudaErrorNotInitialized;
    
    // Copy input data to device
    cudaError_t err = cudaMemcpyAsync(d_temp_a_, a.d, 4 * sizeof(uint64_t), 
                                     cudaMemcpyHostToDevice, memory_stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(d_temp_b_, b.d, 4 * sizeof(uint64_t), 
                         cudaMemcpyHostToDevice, memory_stream_);
    if (err != cudaSuccess) return err;
    
    // Wait for memory transfer completion
    cudaStreamSynchronize(memory_stream_);
    
    // Record start time
    cudaEventRecord(start_event_, compute_stream_);
    
    // Launch kernel
    batch_mod_mult_optimized<<<1, 1, 0, compute_stream_>>>(d_temp_r_, d_temp_a_, d_temp_b_, 1);
    
    // Record end time
    cudaEventRecord(stop_event_, compute_stream_);
    
    // Copy result back to host
    err = cudaMemcpyAsync(result.d, d_temp_r_, 4 * sizeof(uint64_t), 
                         cudaMemcpyDeviceToHost, memory_stream_);
    if (err != cudaSuccess) return err;
    
    // Wait for completion
    cudaStreamSynchronize(memory_stream_);
    
    // Update performance metrics
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    metrics_.cycles_per_operation = elapsed_ms * 1000.0; // Convert to microseconds
    
    return cudaSuccess;
}

cudaError_t OptimizedModularArithmetic::batch_modular_multiply(const std::vector<BigInt256>& a_values,
                                                             const std::vector<BigInt256>& b_values,
                                                             std::vector<BigInt256>& results) {
    if (!initialized_) return cudaErrorNotInitialized;
    if (a_values.size() != b_values.size()) return cudaErrorInvalidValue;
    
    size_t count = a_values.size();
    results.resize(count);
    
    // Ensure sufficient device memory
    if (count > allocated_size_) {
        if (!allocate_device_memory(count * 2)) {
            return cudaErrorMemoryAllocation;
        }
    }
    
    // Copy input data to device
    cudaError_t err = copy_to_device(a_values, d_temp_a_);
    if (err != cudaSuccess) return err;
    
    err = copy_to_device(b_values, d_temp_b_);
    if (err != cudaSuccess) return err;
    
    // Calculate launch parameters
    dim3 grid_size = calculate_grid_size(count);
    dim3 block_size = calculate_block_size();
    
    // Record start time
    cudaEventRecord(start_event_, compute_stream_);
    
    // Launch optimized batch kernel
    batch_mod_mult_optimized<<<grid_size, block_size, 0, compute_stream_>>>(
        d_temp_r_, d_temp_a_, d_temp_b_, count);
    
    // Record end time
    cudaEventRecord(stop_event_, compute_stream_);
    
    // Copy results back to host
    err = copy_from_device(d_temp_r_, results, count);
    if (err != cudaSuccess) return err;
    
    // Update performance metrics
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    metrics_.cycles_per_operation = (elapsed_ms * 1000.0) / count; // Microseconds per operation
    
    return cudaSuccess;
}

cudaError_t OptimizedModularArithmetic::batch_modular_square(const std::vector<BigInt256>& a_values,
                                                           std::vector<BigInt256>& results) {
    if (!initialized_) return cudaErrorNotInitialized;
    
    size_t count = a_values.size();
    results.resize(count);
    
    if (count > allocated_size_) {
        if (!allocate_device_memory(count * 2)) {
            return cudaErrorMemoryAllocation;
        }
    }
    
    cudaError_t err = copy_to_device(a_values, d_temp_a_);
    if (err != cudaSuccess) return err;
    
    dim3 grid_size = calculate_grid_size(count);
    dim3 block_size = calculate_block_size();
    
    cudaEventRecord(start_event_, compute_stream_);
    
    batch_mod_sqr_optimized<<<grid_size, block_size, 0, compute_stream_>>>(
        d_temp_r_, d_temp_a_, count);
    
    cudaEventRecord(stop_event_, compute_stream_);
    
    err = copy_from_device(d_temp_r_, results, count);
    if (err != cudaSuccess) return err;
    
    // Update metrics
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    metrics_.cycles_per_operation = (elapsed_ms * 1000.0) / count;
    
    return cudaSuccess;
}

cudaError_t OptimizedModularArithmetic::batch_modular_add(const std::vector<BigInt256>& a_values,
                                                        const std::vector<BigInt256>& b_values,
                                                        std::vector<BigInt256>& results) {
    if (!initialized_) return cudaErrorNotInitialized;
    if (a_values.size() != b_values.size()) return cudaErrorInvalidValue;
    
    size_t count = a_values.size();
    results.resize(count);
    
    if (count > allocated_size_) {
        if (!allocate_device_memory(count * 2)) {
            return cudaErrorMemoryAllocation;
        }
    }
    
    cudaError_t err = copy_to_device(a_values, d_temp_a_);
    if (err != cudaSuccess) return err;
    
    err = copy_to_device(b_values, d_temp_b_);
    if (err != cudaSuccess) return err;
    
    dim3 grid_size = calculate_grid_size(count);
    dim3 block_size = calculate_block_size();
    
    cudaEventRecord(start_event_, compute_stream_);
    
    batch_mod_add_optimized<<<grid_size, block_size, 0, compute_stream_>>>(
        d_temp_r_, d_temp_a_, d_temp_b_, count);
    
    cudaEventRecord(stop_event_, compute_stream_);
    
    err = copy_from_device(d_temp_r_, results, count);
    
    // Update metrics
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    metrics_.cycles_per_operation = (elapsed_ms * 1000.0) / count;
    
    return err;
}

OptimizationMetrics OptimizedModularArithmetic::get_performance_metrics() const {
    return metrics_;
}

void OptimizedModularArithmetic::reset_performance_counters() {
    metrics_ = OptimizationMetrics();
}

void OptimizedModularArithmetic::set_optimization_level(int level) {
    level = std::clamp(level, 0, 3);
    
    switch (level) {
        case 0: // Conservative
            config_.use_ptx_assembly = false;
            config_.use_fused_operations = false;
            config_.use_warp_primitives = false;
            break;
        case 1: // Moderate  
            config_.use_ptx_assembly = true;
            config_.use_fused_operations = false;
            config_.use_warp_primitives = false;
            break;
        case 2: // Aggressive
            config_.use_ptx_assembly = true;
            config_.use_fused_operations = true;
            config_.use_warp_primitives = false;
            break;
        case 3: // Maximum
            config_.use_ptx_assembly = true;
            config_.use_fused_operations = true;
            config_.use_warp_primitives = true;
            config_.use_shared_memory = true;
            break;
    }
}

OptimizedModularArithmetic::BenchmarkResults 
OptimizedModularArithmetic::benchmark_modular_multiply(size_t operation_count) {
    BenchmarkResults results;
    
    // Generate random test data
    std::vector<BigInt256> a_values(operation_count);
    std::vector<BigInt256> b_values(operation_count);
    std::vector<BigInt256> output_values;
    
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    
    for (size_t i = 0; i < operation_count; ++i) {
        for (int j = 0; j < 4; ++j) {
            a_values[i].d[j] = dist(rng);
            b_values[i].d[j] = dist(rng);
        }
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cudaError_t err = batch_modular_multiply(a_values, b_values, output_values);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (err == cudaSuccess) {
        results.execution_time = duration;
        results.total_operations = operation_count;
        results.operations_per_second = (double)operation_count / (duration.count() / 1000.0);
        
        // Estimate memory bandwidth (rough calculation)
        size_t bytes_transferred = operation_count * 3 * 4 * sizeof(uint64_t); // 3 arrays x 4 uint64_t
        results.memory_bandwidth_gbps = (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / 
                                       (duration.count() / 1000.0);
        
        // Compute efficiency estimation (very rough)
        results.compute_efficiency = std::min(1.0, results.operations_per_second / 1000000.0);
    }
    
    return results;
}

bool OptimizedModularArithmetic::allocate_device_memory(size_t max_operations) {
    // Free existing memory
    free_device_memory();
    
    size_t bytes_per_array = max_operations * 4 * sizeof(uint64_t);
    
    // Allocate three arrays: A, B, and Result
    cudaError_t err = cudaMalloc(&d_temp_a_, bytes_per_array);
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&d_temp_b_, bytes_per_array);
    if (err != cudaSuccess) {
        cudaFree(d_temp_a_);
        d_temp_a_ = nullptr;
        return false;
    }
    
    err = cudaMalloc(&d_temp_r_, bytes_per_array);
    if (err != cudaSuccess) {
        cudaFree(d_temp_a_);
        cudaFree(d_temp_b_);
        d_temp_a_ = d_temp_b_ = nullptr;
        return false;
    }
    
    allocated_size_ = max_operations;
    return true;
}

void OptimizedModularArithmetic::free_device_memory() {
    if (d_temp_a_) {
        cudaFree(d_temp_a_);
        d_temp_a_ = nullptr;
    }
    if (d_temp_b_) {
        cudaFree(d_temp_b_);
        d_temp_b_ = nullptr;
    }
    if (d_temp_r_) {
        cudaFree(d_temp_r_);
        d_temp_r_ = nullptr;
    }
    allocated_size_ = 0;
}

cudaError_t OptimizedModularArithmetic::copy_to_device(const std::vector<BigInt256>& host_data, uint64_t* device_ptr) {
    size_t bytes = host_data.size() * 4 * sizeof(uint64_t);
    return cudaMemcpyAsync(device_ptr, host_data.data(), bytes, cudaMemcpyHostToDevice, memory_stream_);
}

cudaError_t OptimizedModularArithmetic::copy_from_device(uint64_t* device_ptr, std::vector<BigInt256>& host_data, size_t count) {
    size_t bytes = count * 4 * sizeof(uint64_t);
    return cudaMemcpyAsync(host_data.data(), device_ptr, bytes, cudaMemcpyDeviceToHost, memory_stream_);
}

dim3 OptimizedModularArithmetic::calculate_grid_size(size_t operation_count) {
    dim3 block_size = calculate_block_size();
    int grid_x = (operation_count + block_size.x - 1) / block_size.x;
    return dim3(grid_x, 1, 1);
}

dim3 OptimizedModularArithmetic::calculate_block_size() {
    // Architecture-specific optimal block sizes
    switch (config_.target_architecture) {
        case 75: return dim3(256, 1, 1); // Turing
        case 80:
        case 86: return dim3(512, 1, 1); // Ampere
        case 90: return dim3(1024, 1, 1); // Hopper
        default: return dim3(256, 1, 1);
    }
}

void OptimizedModularArithmetic::configure_for_architecture() {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id_);
    
    config_.target_architecture = props.major * 10 + props.minor;
    
    // Enable architecture-specific features
    enable_architecture_features();
}

void OptimizedModularArithmetic::enable_architecture_features() {
    switch (config_.target_architecture) {
        case 75: // Turing
            arch_specific::TuringOptimizer::configure_for_turing(config_);
            break;
        case 80:
        case 86: // Ampere
            arch_specific::AmpereOptimizer::configure_for_ampere(config_);
            break;
        case 90: // Hopper
            arch_specific::HopperOptimizer::configure_for_hopper(config_);
            break;
    }
}

// Architecture-specific optimizer implementations
namespace arch_specific {

void TuringOptimizer::configure_for_turing(OptimizationConfig& config) {
    config.use_ptx_assembly = true;
    config.use_fused_operations = true; // Turing has good FMA support
    config.use_warp_primitives = true;
    config.use_shared_memory = false; // Prefer L1 cache
}

dim3 TuringOptimizer::get_optimal_block_size() {
    return dim3(256, 1, 1); // Optimal for Turing's SM count
}

void AmpereOptimizer::configure_for_ampere(OptimizationConfig& config) {
    config.use_ptx_assembly = true;
    config.use_fused_operations = true;
    config.use_warp_primitives = true;
    config.use_shared_memory = true; // Ampere has excellent shared memory
}

dim3 AmpereOptimizer::get_optimal_block_size() {
    return dim3(512, 1, 1); // Take advantage of higher SM count
}

void HopperOptimizer::configure_for_hopper(OptimizationConfig& config) {
    config.use_ptx_assembly = true;
    config.use_fused_operations = true;
    config.use_warp_primitives = true;
    config.use_shared_memory = true;
}

dim3 HopperOptimizer::get_optimal_block_size() {
    return dim3(1024, 1, 1); // Maximum parallelism for Hopper
}

} // namespace arch_specific

// Performance tester implementation
PerformanceTester::PerformanceTester() {
    optimized_impl_ = std::make_unique<OptimizedModularArithmetic>();
    reference_impl_ = std::make_unique<cpu::Secp256k1>();
}

PerformanceTester::~PerformanceTester() = default;

bool PerformanceTester::validate_optimized_operations(size_t test_count) {
    if (!optimized_impl_->initialize() || !reference_impl_->initialize()) {
        return false;
    }
    
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    
    for (size_t i = 0; i < test_count; ++i) {
        BigInt256 a, b;
        for (int j = 0; j < 4; ++j) {
            a.d[j] = dist(rng);
            b.d[j] = dist(rng);
        }
        
        BigInt256 gpu_result, cpu_result;
        
        // Compute using GPU
        if (optimized_impl_->modular_multiply(a, b, gpu_result) != cudaSuccess) {
            return false;
        }
        
        // Compute using CPU reference
        cpu_result = reference_impl_->scalar_multiply(a, Point(b, BigInt256(1))).x;
        
        // Compare results (allowing for small numerical differences)
        for (int j = 0; j < 4; ++j) {
            if (gpu_result.d[j] != cpu_result.d[j]) {
                return false; // Results don't match
            }
        }
    }
    
    return true; // All tests passed
}

PerformanceTester::PerformanceComparison 
PerformanceTester::benchmark_against_reference(size_t operation_count) {
    PerformanceComparison comparison = {};
    
    // Generate test data
    std::vector<BigInt256> test_scalars(operation_count);
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    
    for (size_t i = 0; i < operation_count; ++i) {
        for (int j = 0; j < 4; ++j) {
            test_scalars[i].d[j] = dist(rng);
        }
    }
    
    // Benchmark GPU implementation
    auto gpu_results = optimized_impl_->benchmark_modular_multiply(operation_count);
    comparison.optimized_ops_per_sec = gpu_results.operations_per_second;
    
    // Benchmark CPU implementation (simplified)
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < std::min(operation_count, size_t(1000)); ++i) {
        Point result = reference_impl_->scalar_multiply(test_scalars[i], constants::GENERATOR);
        (void)result; // Suppress unused variable warning
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    comparison.reference_ops_per_sec = 1000.0 / (duration.count() / 1000.0);
    comparison.speedup_factor = comparison.optimized_ops_per_sec / comparison.reference_ops_per_sec;
    comparison.correctness_passed = validate_optimized_operations(100);
    
    return comparison;
}

// Global registry implementation
std::unordered_map<int, std::unique_ptr<OptimizedModularArithmetic>> OptimizationRegistry::instances_;
int OptimizationRegistry::global_optimization_level_ = 2; // Default to aggressive
std::mutex OptimizationRegistry::registry_mutex_;

OptimizedModularArithmetic* OptimizationRegistry::get_instance(int device_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = instances_.find(device_id);
    if (it != instances_.end()) {
        return it->second.get();
    }
    
    // Create new instance
    auto config = get_optimal_config_for_device(device_id);
    auto instance = std::make_unique<OptimizedModularArithmetic>(config);
    instance->initialize(device_id);
    instance->set_optimization_level(global_optimization_level_);
    
    OptimizedModularArithmetic* ptr = instance.get();
    instances_[device_id] = std::move(instance);
    
    return ptr;
}

OptimizationConfig OptimizationRegistry::get_optimal_config_for_device(int device_id) {
    OptimizationConfig config;
    
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_id) == cudaSuccess) {
        config.target_architecture = props.major * 10 + props.minor;
        
        // Set defaults based on architecture
        if (config.target_architecture >= 75) {
            config.use_ptx_assembly = true;
            config.use_fused_operations = true;
            config.use_warp_primitives = true;
        }
    }
    
    return config;
}

} // namespace optimized
} // namespace gpu  
} // namespace ecc
} // namespace keyhunt