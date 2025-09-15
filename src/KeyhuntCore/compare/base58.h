/**
 * @file base58.h
 * @brief GPU-accelerated Base58 encoding for Bitcoin address generation
 * @author KeyhuntCUDA Team
 * 
 * T051: Implement Base58 encoding for address generation in src/KeyhuntCore/compare/base58.cu
 * 
 * Provides high-performance CUDA kernels and C++ interface for Base58 and Base58Check
 * encoding/decoding operations. Optimized for batch processing of Bitcoin addresses
 * with maximum GPU throughput.
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
 * @brief Configuration for Base58 encoding operations
 */
struct Base58Config {
    // Performance parameters
    size_t threads_per_block;                  // CUDA threads per block (256-1024)
    size_t blocks_per_grid;                    // CUDA blocks per grid
    size_t max_batch_size;                     // Maximum items per batch
    
    // Memory optimization
    bool enable_shared_memory_optimization;    // Use shared memory for alphabet
    bool enable_memory_coalescing;             // Optimize memory access patterns
    bool enable_vectorized_operations;         // Use vectorized operations where possible
    
    // Output configuration
    size_t max_input_length;                   // Maximum input data length
    size_t max_output_length;                  // Maximum output string length
    bool enable_null_termination;              // Add null terminators to strings
    
    // Validation
    bool enable_input_validation;              // Validate input data
    bool enable_checksum_validation;           // Validate checksums (Base58Check)
    
    Base58Config()
        : threads_per_block(256)
        , blocks_per_grid(256)
        , max_batch_size(100000)  // 100K items per batch
        , enable_shared_memory_optimization(true)
        , enable_memory_coalescing(true)
        , enable_vectorized_operations(true)
        , max_input_length(64)
        , max_output_length(64)
        , enable_null_termination(true)
        , enable_input_validation(true)
        , enable_checksum_validation(true) {}
};

/**
 * @brief Performance metrics for Base58 operations
 */
struct Base58Metrics {
    // Throughput metrics
    double encodings_per_second;               // Encodings/sec
    double decodings_per_second;               // Decodings/sec
    double memory_bandwidth_gb_s;              // Memory bandwidth utilization
    double gpu_utilization_percent;            // GPU compute utilization
    
    // Timing metrics
    std::chrono::microseconds kernel_launch_time;     // Kernel launch overhead
    std::chrono::microseconds memory_transfer_time;   // Host-Device transfer time
    std::chrono::microseconds computation_time;       // Actual computation time
    std::chrono::microseconds total_operation_time;   // Total end-to-end time
    
    // Operation statistics
    size_t total_encodings_performed;          // Total encodings completed
    size_t total_decodings_performed;          // Total decodings completed
    size_t successful_operations;              // Successful operations
    size_t failed_operations;                  // Failed operations
    
    // Data statistics
    size_t total_input_bytes;                  // Total input data processed
    size_t total_output_bytes;                 // Total output data generated
    double average_input_length;               // Average input length
    double average_output_length;              // Average output length
    
    // Memory usage
    size_t gpu_memory_used_bytes;              // GPU memory consumption
    size_t cpu_memory_used_bytes;              // CPU memory consumption
    
    // Error statistics
    size_t invalid_input_count;                // Invalid input data count
    size_t checksum_errors;                    // Checksum validation errors
    size_t buffer_overflow_errors;             // Buffer overflow errors
    
    Base58Metrics()
        : encodings_per_second(0.0)
        , decodings_per_second(0.0)
        , memory_bandwidth_gb_s(0.0)
        , gpu_utilization_percent(0.0)
        , kernel_launch_time(0)
        , memory_transfer_time(0)
        , computation_time(0)
        , total_operation_time(0)
        , total_encodings_performed(0)
        , total_decodings_performed(0)
        , successful_operations(0)
        , failed_operations(0)
        , total_input_bytes(0)
        , total_output_bytes(0)
        , average_input_length(0.0)
        , average_output_length(0.0)
        , gpu_memory_used_bytes(0)
        , cpu_memory_used_bytes(0)
        , invalid_input_count(0)
        , checksum_errors(0)
        , buffer_overflow_errors(0) {}
};

/**
 * @brief Result of Base58 encoding/decoding operations
 */
struct Base58Result {
    bool success;                              // Operation success flag
    std::vector<std::string> encoded_strings;  // Encoded strings (for encoding ops)
    std::vector<std::vector<uint8_t>> decoded_data;  // Decoded data (for decoding ops)
    std::vector<size_t> output_lengths;        // Output lengths for each item
    
    // Error information
    std::vector<bool> item_success;            // Success flag for each item
    std::vector<std::string> error_messages;   // Error messages for failed items
    
    // Performance data
    double processing_time_ms;                 // Processing time in milliseconds
    size_t items_processed;                    // Number of items processed
    size_t items_successful;                   // Number of successful items
    
    Base58Result()
        : success(false)
        , processing_time_ms(0.0)
        , items_processed(0)
        , items_successful(0) {}
};

/**
 * @brief High-performance Base58 encoder/decoder class
 * 
 * Provides GPU-accelerated Base58 and Base58Check encoding/decoding for Bitcoin
 * address generation and validation. Optimized for batch processing with maximum
 * throughput on modern GPU architectures.
 */
class Base58Encoder {
public:
    Base58Encoder();
    ~Base58Encoder();
    
    // Initialization and configuration
    bool initialize(int device_id = 0);
    bool configure(const Base58Config& config);
    void cleanup();
    
    // Basic Base58 encoding operations
    Base58Result encode_batch(
        const std::vector<std::vector<uint8_t>>& input_data
    );
    
    Base58Result encode_batch(
        const uint8_t* input_data,
        const size_t* input_lengths,
        size_t count
    );
    
    std::string encode_single(
        const uint8_t* input_data,
        size_t input_length
    );
    
    // Base58Check encoding operations
    Base58Result encode_check_batch(
        const std::vector<std::vector<uint8_t>>& input_data
    );
    
    Base58Result encode_check_batch(
        const uint8_t* input_data,
        const size_t* input_lengths,
        size_t count
    );
    
    std::string encode_check_single(
        const uint8_t* input_data,
        size_t input_length
    );
    
    // Base58 decoding operations
    Base58Result decode_batch(
        const std::vector<std::string>& input_strings
    );
    
    Base58Result decode_batch(
        const char* input_data,
        const size_t* input_lengths,
        size_t count
    );
    
    std::vector<uint8_t> decode_single(
        const std::string& input_string
    );
    
    // Base58Check decoding operations
    Base58Result decode_check_batch(
        const std::vector<std::string>& input_strings
    );
    
    std::vector<uint8_t> decode_check_single(
        const std::string& input_string
    );
    
    // Bitcoin address generation
    Base58Result generate_p2pkh_addresses_batch(
        const uint8_t* hash160_values,
        size_t count
    );
    
    Base58Result generate_p2sh_addresses_batch(
        const uint8_t* script_hashes,
        size_t count
    );
    
    std::string generate_p2pkh_address_single(
        const uint8_t hash160[20]
    );
    
    std::string generate_p2sh_address_single(
        const uint8_t script_hash[20]
    );
    
    // Address validation
    bool validate_bitcoin_address(const std::string& address);
    bool validate_bitcoin_addresses_batch(const std::vector<std::string>& addresses);
    
    // Performance monitoring
    Base58Metrics get_performance_metrics() const;
    void reset_performance_counters();
    
    // Benchmark and validation
    Base58Metrics benchmark_performance(
        size_t test_data_count = 100000,
        size_t iterations = 10
    );
    
    bool validate_correctness(
        const std::vector<std::vector<uint8_t>>& test_data,
        const std::vector<std::string>& expected_results
    );
    
    // Configuration access
    Base58Config get_config() const { return config_; }
    int get_device_id() const { return device_id_; }
    
    // Memory management
    size_t get_gpu_memory_usage() const;
    size_t get_cpu_memory_usage() const;
    bool allocate_gpu_memory(size_t max_batch_size);
    void deallocate_gpu_memory();
    
    // Utility methods
    static bool is_valid_base58_character(char c);
    static bool is_valid_base58_string(const std::string& str);
    static size_t estimate_encoded_length(size_t input_length);
    static size_t estimate_decoded_length(size_t encoded_length);
    
    // Constants
    static const char* BASE58_ALPHABET;
    static const size_t MAX_BITCOIN_ADDRESS_LENGTH = 35;
    static const size_t MIN_BITCOIN_ADDRESS_LENGTH = 26;

private:
    // Device management
    int device_id_;
    bool is_initialized_;
    
    // Configuration
    Base58Config config_;
    
    // GPU memory buffers
    struct GPUBuffers {
        // Input buffers
        uint8_t* input_data;                   // Input data buffer
        uint32_t* input_lengths;               // Input length array
        char* input_strings;                   // Input strings buffer (for decoding)
        
        // Output buffers
        char* output_strings;                  // Output strings buffer
        uint8_t* output_data;                  // Output data buffer
        uint32_t* output_lengths;              // Output length array
        
        // Working buffers
        uint8_t* working_buffer;               // Temporary working space
        uint32_t* error_flags;                 // Error flags for each item
        
        size_t allocated_batch_size;           // Allocated batch size
        size_t allocated_size_bytes;           // Total allocated GPU memory
        
        GPUBuffers()
            : input_data(nullptr)
            , input_lengths(nullptr)
            , input_strings(nullptr)
            , output_strings(nullptr)
            , output_data(nullptr)
            , output_lengths(nullptr)
            , working_buffer(nullptr)
            , error_flags(nullptr)
            , allocated_batch_size(0)
            , allocated_size_bytes(0) {}
    } gpu_buffers_;
    
    // CUDA streams for async operations
    cudaStream_t computation_stream_;
    cudaStream_t memory_stream_;
    bool streams_created_;
    
    // Performance tracking
    mutable Base58Metrics metrics_;
    std::chrono::high_resolution_clock::time_point last_operation_start_;
    
    // Internal methods
    
    // GPU memory management
    bool setup_gpu_buffers(size_t max_batch_size);
    void cleanup_gpu_buffers();
    bool resize_gpu_buffers_if_needed(size_t required_batch_size);
    
    // Kernel launching
    bool launch_encode_kernel(
        const uint8_t* input_data,
        const uint32_t* input_lengths,
        char* output_strings,
        uint32_t* output_lengths,
        size_t count,
        cudaStream_t stream = nullptr
    );
    
    bool launch_encode_check_kernel(
        const uint8_t* input_data,
        const uint32_t* input_lengths,
        char* output_strings,
        uint32_t* output_lengths,
        size_t count,
        cudaStream_t stream = nullptr
    );
    
    bool launch_decode_kernel(
        const char* input_strings,
        const uint32_t* input_lengths,
        uint8_t* output_data,
        uint32_t* output_lengths,
        size_t count,
        cudaStream_t stream = nullptr
    );
    
    bool launch_p2pkh_address_kernel(
        const uint8_t* hash160_values,
        char* address_output,
        size_t count,
        cudaStream_t stream = nullptr
    );
    
    bool launch_optimized_encode_kernel(
        const uint8_t* input_data,
        const uint32_t* input_lengths,
        char* output_strings,
        uint32_t* output_lengths,
        size_t count,
        cudaStream_t stream = nullptr
    );
    
    // Data preparation and validation
    bool prepare_input_data(
        const std::vector<std::vector<uint8_t>>& input_data,
        std::vector<uint8_t>& packed_data,
        std::vector<uint32_t>& lengths
    );
    
    bool prepare_string_data(
        const std::vector<std::string>& input_strings,
        std::vector<char>& packed_strings,
        std::vector<uint32_t>& lengths
    );
    
    bool validate_input_parameters(
        size_t count,
        const uint32_t* lengths = nullptr
    ) const;
    
    // Result processing
    Base58Result process_encode_results(
        const char* output_strings,
        const uint32_t* output_lengths,
        const uint32_t* error_flags,
        size_t count
    );
    
    Base58Result process_decode_results(
        const uint8_t* output_data,
        const uint32_t* output_lengths,
        const uint32_t* error_flags,
        size_t count
    );
    
    // Performance monitoring
    void start_performance_timing();
    void end_performance_timing(size_t operations_performed, bool is_encoding);
    void update_memory_usage_metrics();
    
    // Utility methods
    dim3 calculate_grid_dimensions(size_t count) const;
    dim3 calculate_block_dimensions() const;
    size_t calculate_shared_memory_size() const;
    
    // Validation and error checking
    bool check_cuda_errors(const char* operation_name) const;
    static bool validate_base58_input(const std::string& input);
    static bool validate_checksum(const std::vector<uint8_t>& data);
    
    // Hash functions (for Base58Check)
    static std::vector<uint8_t> sha256_double_hash(const std::vector<uint8_t>& data);
    static void sha256_double_hash(const uint8_t* input, size_t length, uint8_t* output);
};

/**
 * @brief Factory for creating optimized Base58 encoders
 */
class Base58EncoderFactory {
public:
    enum class PerformanceProfile {
        MAXIMUM_SPEED,      // Optimize for maximum encoding speed
        BALANCED,           // Balance speed and memory usage
        LOW_MEMORY,         // Minimize memory usage
        HIGH_ACCURACY      // Maximum validation and error checking
    };
    
    static std::unique_ptr<Base58Encoder> create_encoder(
        PerformanceProfile profile = PerformanceProfile::BALANCED,
        int device_id = 0
    );
    
    static Base58Config get_recommended_config(
        PerformanceProfile profile,
        size_t expected_batch_size,
        size_t available_gpu_memory_mb
    );
    
    static Base58Config get_config_for_architecture(
        int compute_capability_major,
        int compute_capability_minor,
        size_t multiprocessor_count
    );
};

/**
 * @brief Utility functions for Base58 operations
 */
namespace base58_utils {
    
    // Performance benchmarking
    struct BenchmarkResults {
        double peak_encodings_per_second;      // Peak encoding rate
        double peak_decodings_per_second;      // Peak decoding rate
        double average_encodings_per_second;   // Average encoding rate
        double average_decodings_per_second;   // Average decoding rate
        double memory_bandwidth_utilization;   // Memory bandwidth usage (0-1)
        size_t optimal_batch_size;             // Optimal batch size for this GPU
        double gpu_efficiency;                 // GPU utilization efficiency (0-1)
    };
    
    BenchmarkResults benchmark_base58_encoder(
        Base58Encoder& encoder,
        const std::vector<size_t>& batch_sizes = {100, 1000, 10000, 100000}
    );
    
    // Test data generation
    std::vector<std::vector<uint8_t>> generate_random_test_data(
        size_t count,
        size_t min_length = 1,
        size_t max_length = 32
    );
    
    std::vector<uint8_t> generate_random_hash160_values(size_t count);
    std::vector<std::string> generate_test_bitcoin_addresses(size_t count);
    
    // Validation utilities
    bool validate_base58_encoding_correctness(
        const std::vector<std::vector<uint8_t>>& input_data,
        const std::vector<std::string>& encoded_results
    );
    
    bool validate_base58_decoding_correctness(
        const std::vector<std::string>& input_strings,
        const std::vector<std::vector<uint8_t>>& decoded_results
    );
    
    // Performance optimization
    Base58Config optimize_config_for_gpu(
        int device_id,
        const Base58Config& base_config
    );
    
    size_t estimate_optimal_batch_size(
        int device_id,
        size_t available_memory_mb
    );
    
    // Address utilities
    bool extract_hash160_from_address(const std::string& address, uint8_t hash160[20]);
    std::string hash160_to_address(const uint8_t hash160[20], uint8_t version_byte = 0x00);
    bool validate_bitcoin_address_format(const std::string& address);
    
    // Checksum utilities
    bool verify_base58check_checksum(const std::string& encoded);
    std::vector<uint8_t> calculate_base58check_checksum(const std::vector<uint8_t>& data);
}

} // namespace compare
} // namespace keyhunt