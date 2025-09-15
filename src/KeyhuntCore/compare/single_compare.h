/**
 * @file single_compare.h
 * @brief GPU-optimized single target address comparison interface
 * @author KeyhuntCUDA Team
 * 
 * T049: Build single target address comparison in src/KeyhuntCore/compare/single_compare.cu
 * 
 * Provides high-performance CUDA kernels and C++ interface for comparing generated
 * Hash160 values against a single target Bitcoin address. Optimized for maximum
 * throughput when searching for one specific address.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>

namespace keyhunt {
namespace compare {

/**
 * @brief Configuration for single target comparison operations
 */
struct SingleTargetCompareConfig {
    // Kernel selection
    enum class KernelType {
        BASIC,              // Basic comparison kernel
        EARLY_EXIT,         // Early termination on first match
        WARP_OPTIMIZED,     // Warp-level optimizations
        VECTORIZED,         // Vectorized memory operations
        SHARED_MEMORY,      // Shared memory optimization
        STREAMING          // Streaming for large datasets
    };
    
    KernelType kernel_type;                    // Selected kernel type
    
    // Performance parameters
    size_t threads_per_block;                  // CUDA threads per block (256-1024)
    size_t blocks_per_grid;                    // CUDA blocks per grid
    size_t max_batch_size;                     // Maximum Hash160 values per batch
    
    // Memory optimization
    bool enable_memory_coalescing;             // Optimize memory access patterns
    bool enable_constant_memory;               // Use constant memory for target
    bool enable_shared_memory_caching;         // Cache target in shared memory
    
    // Streaming parameters (for large datasets)
    size_t streaming_chunk_size;               // Chunk size for streaming mode
    size_t max_concurrent_streams;             // Maximum concurrent CUDA streams
    
    // Early exit optimization
    bool enable_early_termination;             // Stop on first match found
    bool enable_match_counting;                // Count total matches found
    
    SingleTargetCompareConfig()
        : kernel_type(KernelType::BASIC)
        , threads_per_block(256)
        , blocks_per_grid(256)
        , max_batch_size(1048576)  // 1M Hash160 values
        , enable_memory_coalescing(true)
        , enable_constant_memory(true)
        , enable_shared_memory_caching(false)
        , streaming_chunk_size(65536)  // 64K per chunk
        , max_concurrent_streams(4)
        , enable_early_termination(false)
        , enable_match_counting(true) {}
};

/**
 * @brief Single target comparison performance metrics
 */
struct SingleTargetCompareMetrics {
    // Throughput metrics
    double comparisons_per_second;             // Hash160 comparisons/sec
    double memory_bandwidth_gb_s;              // Memory bandwidth utilization
    double gpu_utilization_percent;            // GPU compute utilization
    
    // Timing metrics
    std::chrono::microseconds kernel_launch_time;     // Kernel launch overhead
    std::chrono::microseconds memory_transfer_time;   // Host-Device transfer time
    std::chrono::microseconds computation_time;       // Actual computation time
    std::chrono::microseconds total_operation_time;   // Total end-to-end time
    
    // Match statistics
    size_t total_comparisons_performed;        // Total Hash160 values compared
    size_t total_matches_found;                // Total matches discovered
    size_t false_positive_count;               // False positives (should be 0)
    
    // Memory usage
    size_t gpu_memory_used_bytes;              // GPU memory consumption
    size_t cpu_memory_used_bytes;              // CPU memory consumption
    
    // Batch statistics
    size_t total_batches_processed;            // Number of batches processed
    size_t average_batch_size;                 // Average Hash160 values per batch
    double batch_processing_efficiency;        // Batch processing efficiency (0-1)
    
    SingleTargetCompareMetrics()
        : comparisons_per_second(0.0)
        , memory_bandwidth_gb_s(0.0)
        , gpu_utilization_percent(0.0)
        , kernel_launch_time(0)
        , memory_transfer_time(0)
        , computation_time(0)
        , total_operation_time(0)
        , total_comparisons_performed(0)
        , total_matches_found(0)
        , false_positive_count(0)
        , gpu_memory_used_bytes(0)
        , cpu_memory_used_bytes(0)
        , total_batches_processed(0)
        , average_batch_size(0)
        , batch_processing_efficiency(0.0) {}
};

/**
 * @brief Result of a single target comparison operation
 */
struct SingleTargetCompareResult {
    bool target_found;                         // True if target was found
    size_t match_index;                        // Index where match was found
    uint64_t private_key_index;                // Associated private key index
    uint8_t matching_hash160[20];              // The matching Hash160 value
    std::chrono::system_clock::time_point found_time;  // When match was discovered
    int device_id;                             // GPU device where found
    
    SingleTargetCompareResult()
        : target_found(false)
        , match_index(SIZE_MAX)
        , private_key_index(0)
        , device_id(-1) {
        found_time = std::chrono::system_clock::now();
        memset(matching_hash160, 0, 20);
    }
};

/**
 * @brief High-performance single target comparison class
 * 
 * Provides GPU-accelerated comparison of Hash160 values against a single target
 * Bitcoin address. Optimized for maximum throughput when searching for one
 * specific address across large key ranges.
 */
class SingleTargetCompare {
public:
    SingleTargetCompare();
    ~SingleTargetCompare();
    
    // Initialization and configuration
    bool initialize(int device_id = 0);
    bool configure(const SingleTargetCompareConfig& config);
    void cleanup();
    
    // Target management
    bool set_target_hash160(const uint8_t hash160[20]);
    bool set_target_address(const std::string& bitcoin_address);
    bool clear_target();
    bool has_target() const { return target_set_; }
    
    // Get target information
    void get_target_hash160(uint8_t hash160[20]) const;
    std::string get_target_address() const { return target_address_; }
    
    // Comparison operations
    SingleTargetCompareResult compare_batch(
        const uint8_t* hash160_values,
        size_t hash_count,
        const uint64_t* private_key_indices = nullptr
    );
    
    std::vector<SingleTargetCompareResult> compare_batch_all_matches(
        const uint8_t* hash160_values,
        size_t hash_count,
        const uint64_t* private_key_indices = nullptr
    );
    
    // Streaming comparison for large datasets
    bool start_streaming_comparison(
        size_t total_hash_count,
        const uint64_t* private_key_indices = nullptr
    );
    
    SingleTargetCompareResult process_streaming_chunk(
        const uint8_t* hash160_chunk,
        size_t chunk_size,
        uint64_t chunk_offset
    );
    
    bool finish_streaming_comparison();
    
    // Asynchronous operations
    bool compare_batch_async(
        const uint8_t* hash160_values,
        size_t hash_count,
        cudaStream_t stream = nullptr,
        const uint64_t* private_key_indices = nullptr
    );
    
    bool wait_for_async_completion(cudaStream_t stream = nullptr);
    SingleTargetCompareResult get_async_result();
    
    // Performance monitoring
    SingleTargetCompareMetrics get_performance_metrics() const;
    void reset_performance_counters();
    
    // Benchmark and validation
    SingleTargetCompareMetrics benchmark_performance(
        size_t test_hash_count = 1000000,
        size_t iterations = 10
    );
    
    bool validate_correctness(
        const uint8_t* test_hash160_values,
        size_t test_count,
        const std::vector<bool>& expected_results
    );
    
    // Configuration access
    SingleTargetCompareConfig get_config() const { return config_; }
    int get_device_id() const { return device_id_; }
    
    // Memory management
    size_t get_gpu_memory_usage() const;
    size_t get_cpu_memory_usage() const;
    bool allocate_gpu_memory(size_t max_batch_size);
    void deallocate_gpu_memory();
    
    // Utility methods
    static std::string hash160_to_hex_string(const uint8_t hash160[20]);
    static bool hex_string_to_hash160(const std::string& hex, uint8_t hash160[20]);
    static bool address_to_hash160(const std::string& address, uint8_t hash160[20]);
    static std::string hash160_to_address(const uint8_t hash160[20], bool p2pkh = true);

private:
    // Device management
    int device_id_;
    bool is_initialized_;
    bool target_set_;
    
    // Configuration
    SingleTargetCompareConfig config_;
    
    // Target data
    uint8_t target_hash160_[20];               // Target Hash160 value
    std::string target_address_;               // Target Bitcoin address
    
    // GPU memory buffers
    struct GPUBuffers {
        uint32_t* hash160_input;               // Input Hash160 values (as uint32_t[5])
        uint32_t* match_results;               // Match results
        uint32_t* match_found_flag;            // Global match found flag
        uint32_t* match_index;                 // Index where match was found
        uint64_t* private_key_indices;         // Private key indices
        
        size_t allocated_hash_count;           // Number of hashes allocated for
        size_t allocated_size_bytes;           // Total allocated GPU memory
        
        GPUBuffers()
            : hash160_input(nullptr)
            , match_results(nullptr)
            , match_found_flag(nullptr)
            , match_index(nullptr)
            , private_key_indices(nullptr)
            , allocated_hash_count(0)
            , allocated_size_bytes(0) {}
    } gpu_buffers_;
    
    // CUDA streams for async operations
    cudaStream_t computation_stream_;
    cudaStream_t memory_stream_;
    bool streams_created_;
    
    // Performance tracking
    mutable SingleTargetCompareMetrics metrics_;
    std::chrono::high_resolution_clock::time_point last_operation_start_;
    
    // Streaming state
    bool streaming_active_;
    size_t streaming_total_count_;
    size_t streaming_processed_count_;
    
    // Internal methods
    
    // GPU memory management
    bool setup_gpu_buffers(size_t max_batch_size);
    void cleanup_gpu_buffers();
    bool resize_gpu_buffers_if_needed(size_t required_hash_count);
    
    // Target management
    bool upload_target_to_gpu();
    bool validate_target_hash160() const;
    
    // Kernel launching
    bool launch_comparison_kernel(
        const uint32_t* input_hash160,
        uint32_t* match_results,
        size_t hash_count,
        cudaStream_t stream = nullptr
    );
    
    bool launch_early_exit_kernel(
        const uint32_t* input_hash160,
        uint32_t* match_found,
        uint32_t* match_index,
        size_t hash_count,
        cudaStream_t stream = nullptr
    );
    
    bool launch_streaming_kernel(
        const uint32_t* input_hash160,
        uint32_t* match_results,
        const uint64_t* private_key_indices,
        size_t hash_count,
        uint64_t batch_offset,
        cudaStream_t stream = nullptr
    );
    
    // Performance monitoring
    void start_performance_timing();
    void end_performance_timing(size_t operations_performed);
    void update_memory_usage_metrics();
    
    // Utility methods
    dim3 calculate_grid_dimensions(size_t hash_count) const;
    dim3 calculate_block_dimensions() const;
    size_t calculate_shared_memory_size() const;
    
    // Validation and error checking
    bool validate_input_parameters(const uint8_t* hash160_values, size_t hash_count) const;
    bool check_cuda_errors(const char* operation_name) const;
    
    // Address format detection and conversion
    static bool is_valid_bitcoin_address(const std::string& address);
    static bool extract_hash160_from_p2pkh(const std::string& address, uint8_t hash160[20]);
    static bool extract_hash160_from_bech32(const std::string& address, uint8_t hash160[20]);
};

/**
 * @brief Factory for creating optimized single target comparers
 */
class SingleTargetCompareFactory {
public:
    enum class PerformanceProfile {
        MAXIMUM_SPEED,      // Optimize for maximum comparison speed
        BALANCED,           // Balance speed and memory usage
        LOW_MEMORY,         // Minimize memory usage
        STREAMING          // Optimize for large datasets that don't fit in memory
    };
    
    static std::unique_ptr<SingleTargetCompare> create_comparer(
        PerformanceProfile profile = PerformanceProfile::BALANCED,
        int device_id = 0
    );
    
    static SingleTargetCompareConfig get_recommended_config(
        PerformanceProfile profile,
        size_t expected_batch_size,
        size_t available_gpu_memory_mb
    );
    
    static SingleTargetCompareConfig get_config_for_architecture(
        int compute_capability_major,
        int compute_capability_minor,
        size_t multiprocessor_count
    );
};

/**
 * @brief Utility functions for single target comparison
 */
namespace single_compare_utils {
    
    // Performance benchmarking
    struct BenchmarkResults {
        double peak_comparisons_per_second;    // Peak comparison rate
        double average_comparisons_per_second; // Average comparison rate
        double memory_bandwidth_utilization;   // Memory bandwidth usage (0-1)
        size_t optimal_batch_size;             // Optimal batch size for this GPU
        double gpu_efficiency;                 // GPU utilization efficiency (0-1)
    };
    
    BenchmarkResults benchmark_single_target_compare(
        SingleTargetCompare& comparer,
        const std::vector<size_t>& batch_sizes = {1024, 4096, 16384, 65536, 262144, 1048576}
    );
    
    // Test data generation
    std::vector<uint8_t> generate_random_hash160_values(size_t count);
    std::vector<uint8_t> generate_test_hash160_with_target(
        size_t count,
        const uint8_t target_hash160[20],
        const std::vector<size_t>& target_positions
    );
    
    // Validation utilities
    bool validate_hash160_format(const uint8_t hash160[20]);
    bool validate_comparison_results(
        const std::vector<uint8_t>& hash160_values,
        const uint8_t target_hash160[20],
        const std::vector<bool>& results
    );
    
    // Performance optimization hints
    SingleTargetCompareConfig optimize_config_for_gpu(
        int device_id,
        const SingleTargetCompareConfig& base_config
    );
    
    size_t estimate_optimal_batch_size(
        int device_id,
        size_t available_memory_mb
    );
}

} // namespace compare
} // namespace keyhunt