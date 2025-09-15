/**
 * @file bitcoin_address_generator.h
 * @brief Bitcoin address generation and comparison pipeline with GPU optimization
 * @author KeyhuntCUDA Team
 * 
 * T044: Develop Bitcoin address generation and comparison pipeline with GPU-optimized hash operations
 * 
 * Provides comprehensive Bitcoin address generation from public keys with support for:
 * - P2PKH (Pay-to-Public-Key-Hash) addresses
 * - P2SH (Pay-to-Script-Hash) addresses  
 * - Bech32 (P2WPKH/P2WSH) addresses
 * - GPU-optimized hash operations (SHA256, RIPEMD160)
 * - Batch processing for high throughput
 * - Multi-format address comparison
 */

#pragma once

#include "../ecc/secp256k1.h"
#include "../models/TargetAddress.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <cuda_runtime.h>

namespace keyhunt {
namespace compare {

/**
 * @brief Bitcoin address format types
 */
enum class AddressFormat {
    P2PKH,          // Pay-to-Public-Key-Hash (1...)
    P2SH,           // Pay-to-Script-Hash (3...)
    P2WPKH_V0,      // Witness v0 Public Key Hash (bc1q...)
    P2WSH_V0,       // Witness v0 Script Hash (bc1q...)
    P2TR,           // Taproot (bc1p...)
    UNKNOWN
};

/**
 * @brief Address generation context for batch operations
 */
struct AddressGenerationContext {
    // Public key data (33 or 65 bytes per key)
    std::vector<uint8_t> public_keys;          // Compressed/uncompressed public keys
    std::vector<bool> compression_flags;       // True for compressed keys
    size_t key_count;                          // Number of public keys
    
    // Hash intermediate results
    std::vector<uint8_t> sha256_hashes;        // SHA256(public_key) results
    std::vector<uint8_t> ripemd160_hashes;     // RIPEMD160(SHA256(public_key)) results
    std::vector<uint8_t> hash160_results;      // Final Hash160 results
    
    // Generated addresses
    std::vector<std::string> p2pkh_addresses;  // P2PKH addresses
    std::vector<std::string> p2sh_addresses;   // P2SH addresses (if applicable)
    std::vector<std::string> bech32_addresses; // Bech32 addresses
    
    // Performance metrics
    size_t total_addresses_generated;          // Total addresses generated
    std::chrono::milliseconds generation_time; // Time taken for generation
    double addresses_per_second;               // Generation rate
    
    AddressGenerationContext() 
        : key_count(0), total_addresses_generated(0)
        , generation_time(0), addresses_per_second(0.0) {}
};

/**
 * @brief GPU-optimized hash operations configuration
 */
struct HashOperationConfig {
    size_t batch_size;                         // Keys per batch
    size_t threads_per_block;                  // CUDA threads per block
    size_t blocks_per_grid;                    // CUDA blocks per grid
    bool enable_shared_memory_optimization;    // Use shared memory optimization
    bool enable_texture_memory;                // Use texture memory for constants
    size_t max_concurrent_batches;             // Maximum concurrent batches
    
    // Hash algorithm selection
    bool enable_parallel_sha256;               // Parallel SHA256 implementation
    bool enable_parallel_ripemd160;            // Parallel RIPEMD160 implementation
    bool enable_fused_hash_operations;         // Fuse SHA256+RIPEMD160
    
    HashOperationConfig()
        : batch_size(65536), threads_per_block(256), blocks_per_grid(256)
        , enable_shared_memory_optimization(true), enable_texture_memory(true)
        , max_concurrent_batches(4), enable_parallel_sha256(true)
        , enable_parallel_ripemd160(true), enable_fused_hash_operations(true) {}
};

/**
 * @brief Address comparison and matching configuration
 */
struct AddressComparisonConfig {
    // Target address sets for different formats
    std::unordered_set<std::string> target_p2pkh;     // P2PKH targets
    std::unordered_set<std::string> target_p2sh;      // P2SH targets
    std::unordered_set<std::string> target_bech32;    // Bech32 targets
    
    // Comparison optimization
    bool enable_bloom_filter;                  // Use Bloom filter for initial filtering
    bool enable_gpu_comparison;                // GPU-based address comparison
    bool enable_prefix_optimization;           // Optimize for address prefixes
    size_t bloom_filter_size;                  // Bloom filter size (bits)
    double bloom_filter_false_positive_rate;   // Target false positive rate
    
    // Performance settings
    bool enable_batch_comparison;              // Compare addresses in batches
    size_t comparison_batch_size;              // Addresses per comparison batch
    
    AddressComparisonConfig()
        : enable_bloom_filter(true), enable_gpu_comparison(true)
        , enable_prefix_optimization(true), bloom_filter_size(1048576)  // 1MB
        , bloom_filter_false_positive_rate(0.001)  // 0.1%
        , enable_batch_comparison(true), comparison_batch_size(32768) {}
};

/**
 * @brief Address generation and comparison performance metrics
 */
struct AddressGenerationMetrics {
    // Generation performance
    size_t total_public_keys_processed;        // Total public keys processed
    size_t total_addresses_generated;          // Total addresses generated
    double average_generation_rate;            // Keys/sec average
    double peak_generation_rate;               // Keys/sec peak
    
    // Hash operation performance
    double sha256_operations_per_second;       // SHA256 ops/sec
    double ripemd160_operations_per_second;    // RIPEMD160 ops/sec
    double hash160_operations_per_second;      // Hash160 ops/sec
    
    // Comparison performance
    size_t total_comparisons_performed;        // Total address comparisons
    double comparison_rate;                    // Comparisons/sec
    size_t bloom_filter_hits;                  // Bloom filter positive hits
    size_t bloom_filter_false_positives;       // False positive count
    
    // Memory utilization
    size_t gpu_memory_used;                    // GPU memory usage (bytes)
    size_t cpu_memory_used;                    // CPU memory usage (bytes)
    double memory_bandwidth_utilization;       // Memory bandwidth usage
    
    // Match statistics
    size_t total_matches_found;                // Total address matches
    size_t p2pkh_matches;                      // P2PKH matches
    size_t p2sh_matches;                       // P2SH matches
    size_t bech32_matches;                     // Bech32 matches
    
    AddressGenerationMetrics()
        : total_public_keys_processed(0), total_addresses_generated(0)
        , average_generation_rate(0.0), peak_generation_rate(0.0)
        , sha256_operations_per_second(0.0), ripemd160_operations_per_second(0.0)
        , hash160_operations_per_second(0.0), total_comparisons_performed(0)
        , comparison_rate(0.0), bloom_filter_hits(0), bloom_filter_false_positives(0)
        , gpu_memory_used(0), cpu_memory_used(0), memory_bandwidth_utilization(0.0)
        , total_matches_found(0), p2pkh_matches(0), p2sh_matches(0), bech32_matches(0) {}
};

/**
 * @brief Address match result
 */
struct AddressMatch {
    ecc::Point public_key;                     // Matching public key
    ecc::BigInt256 private_key;                // Associated private key (if available)
    std::string address;                       // Matched address string
    AddressFormat format;                      // Address format
    uint8_t hash160[20];                       // Hash160 value
    
    // Discovery information
    std::chrono::system_clock::time_point found_time;  // When match was found
    size_t batch_id;                           // Batch ID where found
    int device_id;                             // GPU device ID
    double confidence_score;                   // Match confidence (0-1)
    
    AddressMatch() 
        : format(AddressFormat::UNKNOWN), batch_id(0), device_id(-1)
        , confidence_score(1.0) {
        found_time = std::chrono::system_clock::now();
        memset(hash160, 0, 20);
    }
};

/**
 * @brief Main Bitcoin address generator class
 */
class BitcoinAddressGenerator {
public:
    BitcoinAddressGenerator();
    ~BitcoinAddressGenerator();
    
    // Initialization and configuration
    bool initialize(int device_id = 0);
    void configure_hash_operations(const HashOperationConfig& config);
    void configure_address_comparison(const AddressComparisonConfig& config);
    void cleanup();
    
    // Address generation methods
    bool generate_addresses_from_public_keys(
        const std::vector<ecc::Point>& public_keys,
        AddressGenerationContext& context,
        bool compressed = true
    );
    
    bool generate_addresses_batch(
        const uint8_t* public_keys_data,
        size_t key_count,
        bool compressed,
        AddressGenerationContext& context
    );
    
    // Individual address generation
    std::string generate_p2pkh_address(const ecc::Point& public_key, bool compressed = true);
    std::string generate_p2sh_address(const uint8_t script_hash[20]);
    std::string generate_bech32_address(const uint8_t hash[20], int witness_version = 0);
    
    // Hash operations
    bool compute_hash160_batch(
        const uint8_t* public_keys_data,
        size_t key_count,
        bool compressed,
        uint8_t* hash160_results
    );
    
    bool compute_sha256_batch(
        const uint8_t* input_data,
        size_t data_count,
        size_t input_size,
        uint8_t* sha256_results
    );
    
    bool compute_ripemd160_batch(
        const uint8_t* input_data,
        size_t data_count,
        uint8_t* ripemd160_results
    );
    
    // Address comparison and matching
    bool set_target_addresses(const std::vector<std::string>& addresses);
    bool add_target_address(const std::string& address);
    bool remove_target_address(const std::string& address);
    
    std::vector<AddressMatch> compare_addresses(
        const AddressGenerationContext& context,
        const std::vector<ecc::BigInt256>& private_keys = {}
    );
    
    bool compare_addresses_gpu(
        const uint8_t* hash160_data,
        size_t hash_count,
        const ecc::BigInt256* private_keys,
        std::vector<AddressMatch>& matches
    );
    
    // Utility methods
    static AddressFormat detect_address_format(const std::string& address);
    static bool validate_address(const std::string& address);
    static std::string hash160_to_p2pkh_address(const uint8_t hash160[20]);
    static std::string hash160_to_bech32_address(const uint8_t hash160[20], int witness_version = 0);
    
    // Base58 encoding/decoding
    static std::string encode_base58(const uint8_t* data, size_t length);
    static std::vector<uint8_t> decode_base58(const std::string& encoded);
    static std::string encode_base58_check(const uint8_t* data, size_t length);
    static std::vector<uint8_t> decode_base58_check(const std::string& encoded);
    
    // Bech32 encoding/decoding
    static std::string encode_bech32(const std::string& hrp, const std::vector<uint8_t>& data);
    static std::pair<std::string, std::vector<uint8_t>> decode_bech32(const std::string& encoded);
    
    // Performance monitoring
    AddressGenerationMetrics get_current_metrics() const;
    void reset_performance_counters();
    
    // Configuration access
    HashOperationConfig get_hash_config() const { return hash_config_; }
    AddressComparisonConfig get_comparison_config() const { return comparison_config_; }
    
    // GPU resource management
    bool allocate_gpu_memory(size_t max_batch_size);
    void deallocate_gpu_memory();
    size_t get_gpu_memory_usage() const;
    
    // Callbacks for address matches
    void set_match_callback(std::function<void(const AddressMatch&)> callback);

private:
    // GPU device management
    int device_id_;
    bool is_initialized_;
    
    // Configuration
    HashOperationConfig hash_config_;
    AddressComparisonConfig comparison_config_;
    
    // GPU memory management
    struct GPUMemoryBuffers {
        uint8_t* public_keys_buffer;           // Public keys input buffer
        uint8_t* sha256_buffer;                // SHA256 results buffer
        uint8_t* ripemd160_buffer;             // RIPEMD160 results buffer
        uint8_t* hash160_buffer;               // Hash160 results buffer
        uint8_t* target_hash160_buffer;        // Target hash160 values
        bool* match_results_buffer;            // Match results
        
        size_t allocated_size;                 // Total allocated size
        
        GPUMemoryBuffers() 
            : public_keys_buffer(nullptr), sha256_buffer(nullptr)
            , ripemd160_buffer(nullptr), hash160_buffer(nullptr)
            , target_hash160_buffer(nullptr), match_results_buffer(nullptr)
            , allocated_size(0) {}
    } gpu_buffers_;
    
    // Target address management
    std::unordered_map<std::string, AddressFormat> target_addresses_;
    std::vector<uint8_t> target_hash160_values_;  // Packed Hash160 values
    
    // Bloom filter for fast address filtering
    std::unique_ptr<class BloomFilter> bloom_filter_;
    
    // Performance metrics
    mutable std::mutex metrics_mutex_;
    AddressGenerationMetrics current_metrics_;
    std::chrono::high_resolution_clock::time_point last_metrics_update_;
    
    // CUDA streams for overlapping operations
    cudaStream_t computation_stream_;
    cudaStream_t memory_stream_;
    
    // Match callback
    std::function<void(const AddressMatch&)> match_callback_;
    
    // Internal methods
    
    // Hash computation kernels (implemented in .cu file)
    bool launch_sha256_kernel(
        const uint8_t* input, size_t count, size_t input_size,
        uint8_t* output, cudaStream_t stream = nullptr
    );
    
    bool launch_ripemd160_kernel(
        const uint8_t* input, size_t count,
        uint8_t* output, cudaStream_t stream = nullptr
    );
    
    bool launch_hash160_kernel(
        const uint8_t* public_keys, size_t count, bool compressed,
        uint8_t* hash160_output, cudaStream_t stream = nullptr
    );
    
    // Address comparison kernels
    bool launch_address_comparison_kernel(
        const uint8_t* hash160_input, size_t input_count,
        const uint8_t* target_hash160, size_t target_count,
        bool* match_results, cudaStream_t stream = nullptr
    );
    
    // Utility methods
    bool setup_gpu_buffers(size_t max_batch_size);
    bool cleanup_gpu_buffers();
    bool setup_target_address_data();
    void update_performance_metrics();
    
    // Address format detection and validation
    static bool is_valid_p2pkh_address(const std::string& address);
    static bool is_valid_p2sh_address(const std::string& address);
    static bool is_valid_bech32_address(const std::string& address);
    
    // Hash160 extraction from addresses
    bool extract_hash160_from_address(const std::string& address, uint8_t hash160[20]);
    
    // Base58 implementation details
    static const char base58_alphabet[];
    static const int base58_map[256];
    static void init_base58_map();
    
    // Bech32 implementation details
    static uint32_t bech32_polymod(const std::vector<uint8_t>& values);
    static std::vector<uint8_t> bech32_hrp_expand(const std::string& hrp);
    static std::vector<uint8_t> convert_bits(
        const std::vector<uint8_t>& data, int from_bits, int to_bits, bool pad = true
    );
};

/**
 * @brief High-performance Bloom filter for address filtering
 */
class BloomFilter {
public:
    BloomFilter(size_t size_bits, double false_positive_rate);
    ~BloomFilter();
    
    void add(const uint8_t* data, size_t length);
    void add(const std::string& str);
    bool might_contain(const uint8_t* data, size_t length) const;
    bool might_contain(const std::string& str) const;
    
    void clear();
    size_t size() const { return size_bits_; }
    size_t hash_functions() const { return num_hash_functions_; }
    double false_positive_probability() const;
    
    // GPU acceleration
    bool upload_to_gpu();
    bool download_from_gpu();
    void cleanup_gpu();

private:
    size_t size_bits_;
    size_t size_bytes_;
    size_t num_hash_functions_;
    std::vector<uint8_t> bit_array_;
    
    // GPU memory for Bloom filter
    uint8_t* gpu_bit_array_;
    bool gpu_uploaded_;
    
    // Hash functions
    uint32_t hash1(const uint8_t* data, size_t length) const;
    uint32_t hash2(const uint8_t* data, size_t length) const;
    uint32_t get_hash(const uint8_t* data, size_t length, size_t i) const;
};

/**
 * @brief Factory for creating optimized address generators
 */
class AddressGeneratorFactory {
public:
    enum class OptimizationStrategy {
        BALANCED,           // Balanced speed/memory usage
        SPEED_OPTIMIZED,    // Maximum speed
        MEMORY_OPTIMIZED,   // Minimum memory usage
        POWER_EFFICIENT     // Power-efficient operation
    };
    
    static std::unique_ptr<BitcoinAddressGenerator> create_generator(
        OptimizationStrategy strategy = OptimizationStrategy::BALANCED,
        int device_id = 0
    );
    
    static HashOperationConfig get_recommended_hash_config(
        OptimizationStrategy strategy,
        size_t available_memory_mb
    );
    
    static AddressComparisonConfig get_recommended_comparison_config(
        OptimizationStrategy strategy,
        size_t target_address_count
    );
};

/**
 * @brief Utility functions for address generation and comparison
 */
namespace address_utils {
    
    // Address validation utilities
    bool validate_bitcoin_address_format(const std::string& address);
    AddressFormat classify_bitcoin_address(const std::string& address);
    std::string normalize_bitcoin_address(const std::string& address);
    
    // Performance benchmarking
    struct BenchmarkResults {
        double hash160_rate_mkeys_per_sec;     // Hash160 rate (million keys/sec)
        double address_gen_rate_maddr_per_sec; // Address generation rate (million addr/sec)
        double comparison_rate_mcomp_per_sec;  // Comparison rate (million comp/sec)
        size_t memory_bandwidth_gb_s;          // Memory bandwidth (GB/s)
        double gpu_utilization_percent;        // GPU utilization percentage
    };
    
    BenchmarkResults benchmark_address_generation(
        BitcoinAddressGenerator& generator,
        size_t test_key_count = 1000000
    );
    
    // Address set management
    class AddressSet {
    public:
        void add_address(const std::string& address);
        void add_addresses(const std::vector<std::string>& addresses);
        bool contains_address(const std::string& address) const;
        void remove_address(const std::string& address);
        void clear();
        
        size_t size() const;
        std::vector<std::string> get_all_addresses() const;
        std::vector<std::string> get_addresses_by_format(AddressFormat format) const;
        
        // Export Hash160 values for GPU processing
        std::vector<uint8_t> export_hash160_values() const;
        
    private:
        std::unordered_map<AddressFormat, std::unordered_set<std::string>> addresses_by_format_;
        std::unordered_map<std::string, uint8_t[20]> address_to_hash160_;
    };
}

} // namespace compare
} // namespace keyhunt