/**
 * @file multi_target_compare.h
 * @brief GPU-accelerated multi-target address comparison with Bloom filter optimization
 * @author KeyhuntCUDA Team
 * 
 * T050: Create multi-target comparison with Bloom Filter in src/KeyhuntCore/compare/bloom_filter.cu
 * 
 * Provides high-performance CUDA kernels and C++ interface for comparing generated
 * Hash160 values against multiple target Bitcoin addresses. Uses Bloom filter for
 * initial fast filtering followed by exact matching for confirmed hits.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <cuda_runtime.h>

namespace keyhunt {
namespace compare {

/**
 * @brief Configuration for multi-target comparison operations
 */
struct MultiTargetCompareConfig {
    // Kernel selection
    enum class KernelType {
        BASIC,              // Basic multi-target comparison
        WARP_OPTIMIZED,     // Warp-level optimizations
        EARLY_EXIT,         // Early termination on first match
        SHARED_MEMORY,      // Shared memory optimization
        BATCH_PROCESSING,   // Multiple hashes per thread
        STREAMING,          // Streaming for large datasets
        STATISTICS         // Statistics collection enabled
    };
    
    KernelType kernel_type;                    // Selected kernel type
    
    // Performance parameters
    size_t threads_per_block;                  // CUDA threads per block (256-1024)
    size_t blocks_per_grid;                    // CUDA blocks per grid
    size_t max_batch_size;                     // Maximum Hash160 values per batch
    size_t hashes_per_thread;                  // Hashes per thread (batch processing)
    
    // Bloom filter configuration
    bool enable_bloom_filter;                  // Use Bloom filter for initial filtering
    size_t bloom_filter_size_bits;             // Bloom filter size in bits
    size_t bloom_filter_hash_functions;        // Number of hash functions
    double bloom_filter_false_positive_rate;   // Target false positive rate
    
    // Target management
    size_t max_targets_constant_memory;        // Maximum targets in constant memory
    size_t max_targets_global_memory;          // Maximum targets in global memory
    bool enable_target_caching;                // Cache targets in shared memory
    
    // Memory optimization
    bool enable_memory_coalescing;             // Optimize memory access patterns
    bool enable_shared_memory_optimization;    // Use shared memory for frequently accessed data
    bool enable_vectorized_operations;         // Use vectorized memory operations
    
    // Performance monitoring
    bool enable_statistics_collection;         // Collect performance statistics
    bool enable_early_termination;             // Stop on first match found
    bool enable_match_counting;                // Count total matches found
    
    // Streaming parameters
    size_t streaming_chunk_size;               // Chunk size for streaming mode
    size_t max_concurrent_streams;             // Maximum concurrent CUDA streams
    
    MultiTargetCompareConfig()
        : kernel_type(KernelType::BASIC)
        , threads_per_block(256)
        , blocks_per_grid(256)
        , max_batch_size(1048576)  // 1M Hash160 values
        , hashes_per_thread(4)
        , enable_bloom_filter(true)
        , bloom_filter_size_bits(1048576)  // 1M bits = 128KB
        , bloom_filter_hash_functions(7)
        , bloom_filter_false_positive_rate(0.001)  // 0.1%
        , max_targets_constant_memory(64)
        , max_targets_global_memory(1000000)  // 1M targets
        , enable_target_caching(true)
        , enable_memory_coalescing(true)
        , enable_shared_memory_optimization(true)
        , enable_vectorized_operations(true)
        , enable_statistics_collection(false)
        , enable_early_termination(false)
        , enable_match_counting(true)
        , streaming_chunk_size(65536)  // 64K per chunk
        , max_concurrent_streams(4) {}
};

/**
 * @brief Multi-target comparison performance metrics
 */
struct MultiTargetCompareMetrics {
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
    size_t matches_per_target[64];             // Matches per target (up to 64 targets)
    
    // Bloom filter statistics
    size_t bloom_filter_hits;                  // Bloom filter positive hits
    size_t bloom_filter_false_positives;       // False positive count
    double bloom_filter_efficiency;            // Efficiency (1 - FP rate)
    size_t bloom_filter_memory_bytes;          // Bloom filter memory usage
    
    // Memory usage
    size_t gpu_memory_used_bytes;              // GPU memory consumption
    size_t cpu_memory_used_bytes;              // CPU memory consumption
    size_t target_storage_bytes;               // Target storage memory
    
    // Performance efficiency
    size_t total_batches_processed;            // Number of batches processed
    size_t average_batch_size;                 // Average Hash160 values per batch
    double batch_processing_efficiency;        // Batch processing efficiency (0-1)
    
    // Target distribution
    size_t active_target_count;                // Number of active targets
    size_t targets_in_constant_memory;         // Targets stored in constant memory
    size_t targets_in_global_memory;           // Targets stored in global memory
    
    MultiTargetCompareMetrics()
        : comparisons_per_second(0.0)
        , memory_bandwidth_gb_s(0.0)
        , gpu_utilization_percent(0.0)
        , kernel_launch_time(0)
        , memory_transfer_time(0)
        , computation_time(0)
        , total_operation_time(0)
        , total_comparisons_performed(0)
        , total_matches_found(0)
        , bloom_filter_hits(0)
        , bloom_filter_false_positives(0)
        , bloom_filter_efficiency(0.0)
        , bloom_filter_memory_bytes(0)
        , gpu_memory_used_bytes(0)
        , cpu_memory_used_bytes(0)
        , target_storage_bytes(0)
        , total_batches_processed(0)
        , average_batch_size(0)
        , batch_processing_efficiency(0.0)
        , active_target_count(0)
        , targets_in_constant_memory(0)
        , targets_in_global_memory(0) {
        memset(matches_per_target, 0, sizeof(matches_per_target));
    }
};

/**
 * @brief Result of a multi-target comparison operation
 */
struct MultiTargetCompareResult {
    bool targets_found;                        // True if any targets were found
    size_t total_matches;                      // Total number of matches found
    std::vector<size_t> match_indices;         // Indices where matches were found
    std::vector<size_t> target_indices;        // Which targets matched at each index
    std::vector<uint64_t> private_key_indices; // Associated private key indices
    std::vector<uint8_t> matching_hash160s;    // The matching Hash160 values (20 bytes each)
    std::chrono::system_clock::time_point found_time;  // When matches were discovered
    int device_id;                             // GPU device where found
    
    // Statistics
    size_t bloom_filter_hits;                  // Bloom filter positive results
    size_t bloom_filter_false_positives;       // False positives encountered
    double processing_time_ms;                 // Processing time in milliseconds
    
    MultiTargetCompareResult()
        : targets_found(false)
        , total_matches(0)
        , device_id(-1)
        , bloom_filter_hits(0)
        , bloom_filter_false_positives(0)
        , processing_time_ms(0.0) {
        found_time = std::chrono::system_clock::now();
    }
};

/**
 * @brief Target address information
 */
struct TargetAddressInfo {
    std::string address;                       // Bitcoin address string
    uint8_t hash160[20];                       // Hash160 value
    size_t target_index;                       // Internal target index
    size_t match_count;                        // Number of times this target was matched
    std::chrono::system_clock::time_point added_time;  // When target was added
    
    TargetAddressInfo() : target_index(SIZE_MAX), match_count(0) {
        memset(hash160, 0, 20);
        added_time = std::chrono::system_clock::now();
    }
};

/**
 * @brief High-performance multi-target comparison class
 * 
 * Provides GPU-accelerated comparison of Hash160 values against multiple target
 * Bitcoin addresses. Uses Bloom filter for efficient initial filtering followed
 * by exact matching for confirmed hits.
 */
class MultiTargetCompare {
public:
    MultiTargetCompare();
    ~MultiTargetCompare();
    
    // Initialization and configuration
    bool initialize(int device_id = 0);
    bool configure(const MultiTargetCompareConfig& config);
    void cleanup();
    
    // Target management
    bool add_target_hash160(const uint8_t hash160[20]);
    bool add_target_address(const std::string& bitcoin_address);
    bool add_target_addresses(const std::vector<std::string>& addresses);
    bool remove_target_address(const std::string& bitcoin_address);
    bool clear_targets();
    
    size_t get_target_count() const { return targets_.size(); }
    std::vector<TargetAddressInfo> get_target_list() const;
    
    // Bloom filter management
    bool rebuild_bloom_filter();
    bool optimize_bloom_filter_parameters();
    double get_bloom_filter_false_positive_rate() const;
    size_t get_bloom_filter_memory_usage() const;
    
    // Comparison operations
    MultiTargetCompareResult compare_batch(
        const uint8_t* hash160_values,
        size_t hash_count,
        const uint64_t* private_key_indices = nullptr
    );
    
    std::vector<MultiTargetCompareResult> compare_batch_detailed(
        const uint8_t* hash160_values,
        size_t hash_count,
        const uint64_t* private_key_indices = nullptr
    );
    
    // Streaming comparison for large datasets
    bool start_streaming_comparison(
        size_t total_hash_count,
        const uint64_t* private_key_indices = nullptr
    );
    
    MultiTargetCompareResult process_streaming_chunk(
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
    MultiTargetCompareResult get_async_result();
    
    // Performance monitoring
    MultiTargetCompareMetrics get_performance_metrics() const;
    void reset_performance_counters();
    
    // Benchmark and validation
    MultiTargetCompareMetrics benchmark_performance(
        size_t test_hash_count = 1000000,
        size_t iterations = 10
    );
    
    bool validate_correctness(
        const uint8_t* test_hash160_values,
        size_t test_count,
        const std::vector<std::vector<bool>>& expected_results
    );
    
    // Configuration access
    MultiTargetCompareConfig get_config() const { return config_; }
    int get_device_id() const { return device_id_; }
    
    // Memory management
    size_t get_gpu_memory_usage() const;
    size_t get_cpu_memory_usage() const;
    bool allocate_gpu_memory(size_t max_batch_size);
    void deallocate_gpu_memory();
    
    // Statistics and analysis
    void print_target_statistics() const;
    void print_bloom_filter_statistics() const;
    void export_performance_data(const std::string& filename) const;
    
    // Utility methods
    static std::string hash160_to_hex_string(const uint8_t hash160[20]);
    static bool hex_string_to_hash160(const std::string& hex, uint8_t hash160[20]);
    static bool address_to_hash160(const std::string& address, uint8_t hash160[20]);
    static std::string hash160_to_address(const uint8_t hash160[20], bool p2pkh = true);

private:
    // Device management
    int device_id_;
    bool is_initialized_;
    
    // Configuration
    MultiTargetCompareConfig config_;
    
    // Target management
    std::vector<TargetAddressInfo> targets_;
    std::unordered_map<std::string, size_t> address_to_index_;
    std::vector<uint32_t> target_hash160_words_;  // Targets as uint32_t arrays
    
    // Bloom filter
    std::unique_ptr<class BloomFilterGPU> bloom_filter_;
    bool bloom_filter_uploaded_;
    
    // GPU memory buffers
    struct GPUBuffers {
        uint32_t* hash160_input;               // Input Hash160 values (as uint32_t[5])
        uint32_t* match_results;               // Match results
        uint32_t* target_indices;              // Target indices for matches
        uint64_t* private_key_indices;         // Private key indices
        
        // Early exit buffers
        uint32_t* match_found_flag;            // Global match found flag
        uint32_t* match_index;                 // Index where match was found
        uint32_t* found_target_index;          // Target index that matched
        
        // Statistics buffers
        uint64_t* bloom_hits_counter;          // Bloom filter hits
        uint64_t* exact_matches_counter;       // Exact matches
        
        // Target storage
        uint32_t* target_hashes_global;        // Targets in global memory
        uint32_t* bloom_filter_data;           // Bloom filter bit array
        
        size_t allocated_hash_count;           // Number of hashes allocated for
        size_t allocated_size_bytes;           // Total allocated GPU memory
        
        GPUBuffers()
            : hash160_input(nullptr)
            , match_results(nullptr)
            , target_indices(nullptr)
            , private_key_indices(nullptr)
            , match_found_flag(nullptr)
            , match_index(nullptr)
            , found_target_index(nullptr)
            , bloom_hits_counter(nullptr)
            , exact_matches_counter(nullptr)
            , target_hashes_global(nullptr)
            , bloom_filter_data(nullptr)
            , allocated_hash_count(0)
            , allocated_size_bytes(0) {}
    } gpu_buffers_;
    
    // CUDA streams for async operations
    cudaStream_t computation_stream_;
    cudaStream_t memory_stream_;
    bool streams_created_;
    
    // Performance tracking
    mutable MultiTargetCompareMetrics metrics_;
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
    bool upload_targets_to_gpu();
    bool rebuild_target_storage();
    size_t pack_targets_for_gpu();
    
    // Bloom filter management
    bool setup_bloom_filter();
    bool upload_bloom_filter_to_gpu();
    bool update_bloom_filter_with_targets();
    
    // Kernel launching
    bool launch_comparison_kernel(
        const uint32_t* input_hash160,
        uint32_t* match_results,
        uint32_t* target_indices,
        size_t hash_count,
        cudaStream_t stream = nullptr
    );
    
    bool launch_early_exit_kernel(
        const uint32_t* input_hash160,
        uint32_t* match_found,
        uint32_t* match_index,
        uint32_t* target_index,
        size_t hash_count,
        cudaStream_t stream = nullptr
    );
    
    bool launch_streaming_kernel(
        const uint32_t* input_hash160,
        uint32_t* match_results,
        uint32_t* target_indices,
        const uint64_t* private_key_indices,
        size_t hash_count,
        uint64_t batch_offset,
        cudaStream_t stream = nullptr
    );
    
    bool launch_statistics_kernel(
        const uint32_t* input_hash160,
        uint32_t* match_results,
        uint32_t* target_indices,
        uint64_t* bloom_hits,
        uint64_t* exact_matches,
        size_t hash_count,
        cudaStream_t stream = nullptr
    );
    
    // Performance monitoring
    void start_performance_timing();
    void end_performance_timing(size_t operations_performed);
    void update_memory_usage_metrics();
    void collect_bloom_filter_statistics();
    
    // Utility methods
    dim3 calculate_grid_dimensions(size_t hash_count) const;
    dim3 calculate_block_dimensions() const;
    size_t calculate_shared_memory_size() const;
    
    // Validation and error checking
    bool validate_input_parameters(const uint8_t* hash160_values, size_t hash_count) const;
    bool check_cuda_errors(const char* operation_name) const;
    
    // Target validation and conversion
    bool validate_target_hash160(const uint8_t hash160[20]) const;
    size_t find_target_index(const uint8_t hash160[20]) const;
    
    // Address format handling
    static bool is_valid_bitcoin_address(const std::string& address);
    static bool extract_hash160_from_p2pkh(const std::string& address, uint8_t hash160[20]);
    static bool extract_hash160_from_bech32(const std::string& address, uint8_t hash160[20]);
};

/**
 * @brief GPU-accelerated Bloom filter implementation
 */
class BloomFilterGPU {
public:
    BloomFilterGPU(size_t size_bits, size_t num_hash_functions);
    ~BloomFilterGPU();
    
    bool add_hash160(const uint8_t hash160[20]);
    bool add_hash160_batch(const std::vector<uint8_t>& hash160_values);
    bool might_contain(const uint8_t hash160[20]) const;
    
    bool upload_to_gpu();
    bool download_from_gpu();
    void clear();
    
    size_t size_bits() const { return size_bits_; }
    size_t size_bytes() const { return size_bytes_; }
    size_t hash_functions() const { return num_hash_functions_; }
    double false_positive_probability(size_t num_elements) const;
    
    // GPU memory management
    void cleanup_gpu();
    size_t get_gpu_memory_usage() const;
    bool is_uploaded_to_gpu() const { return gpu_uploaded_; }

private:
    size_t size_bits_;
    size_t size_bytes_;
    size_t num_hash_functions_;
    std::vector<uint32_t> bit_array_;          // Host bit array (32-bit words)
    
    // GPU memory
    uint32_t* gpu_bit_array_;
    bool gpu_uploaded_;
    
    // Hash functions for Hash160 values
    uint32_t hash_function(const uint8_t hash160[20], size_t hash_index) const;
    void set_bit(size_t bit_index);
    bool get_bit(size_t bit_index) const;
};

/**
 * @brief Factory for creating optimized multi-target comparers
 */
class MultiTargetCompareFactory {
public:
    enum class PerformanceProfile {
        MAXIMUM_SPEED,      // Optimize for maximum comparison speed
        BALANCED,           // Balance speed, memory usage, and accuracy
        LOW_MEMORY,         // Minimize memory usage
        HIGH_ACCURACY,      // Minimize false positives
        STREAMING          // Optimize for large datasets
    };
    
    static std::unique_ptr<MultiTargetCompare> create_comparer(
        PerformanceProfile profile = PerformanceProfile::BALANCED,
        int device_id = 0
    );
    
    static MultiTargetCompareConfig get_recommended_config(
        PerformanceProfile profile,
        size_t target_count,
        size_t expected_batch_size,
        size_t available_gpu_memory_mb
    );
    
    static MultiTargetCompareConfig get_config_for_architecture(
        int compute_capability_major,
        int compute_capability_minor,
        size_t multiprocessor_count,
        size_t target_count
    );
};

/**
 * @brief Utility functions for multi-target comparison
 */
namespace multi_target_utils {
    
    // Performance benchmarking
    struct BenchmarkResults {
        double peak_comparisons_per_second;    // Peak comparison rate
        double average_comparisons_per_second; // Average comparison rate
        double memory_bandwidth_utilization;   // Memory bandwidth usage (0-1)
        double bloom_filter_efficiency;        // Bloom filter effectiveness (0-1)
        size_t optimal_batch_size;             // Optimal batch size for this GPU
        size_t optimal_target_count;           // Optimal target count for performance
        double gpu_efficiency;                 // GPU utilization efficiency (0-1)
    };
    
    BenchmarkResults benchmark_multi_target_compare(
        MultiTargetCompare& comparer,
        const std::vector<size_t>& target_counts = {10, 100, 1000, 10000},
        const std::vector<size_t>& batch_sizes = {1024, 4096, 16384, 65536, 262144}
    );
    
    // Test data generation
    std::vector<uint8_t> generate_random_hash160_values(size_t count);
    std::vector<uint8_t> generate_test_hash160_with_targets(
        size_t count,
        const std::vector<uint8_t>& target_hash160s,
        const std::vector<std::vector<size_t>>& target_positions
    );
    
    std::vector<std::string> generate_random_bitcoin_addresses(size_t count);
    
    // Validation utilities
    bool validate_multi_target_results(
        const std::vector<uint8_t>& hash160_values,
        const std::vector<uint8_t>& target_hash160s,
        const MultiTargetCompareResult& results
    );
    
    // Performance optimization
    MultiTargetCompareConfig optimize_config_for_targets(
        int device_id,
        size_t target_count,
        const MultiTargetCompareConfig& base_config
    );
    
    size_t estimate_optimal_bloom_filter_size(
        size_t target_count,
        double desired_false_positive_rate
    );
    
    size_t estimate_optimal_hash_functions(
        size_t bloom_filter_size_bits,
        size_t target_count
    );
}

} // namespace compare
} // namespace keyhunt