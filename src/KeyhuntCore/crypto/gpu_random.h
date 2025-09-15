/**
 * @file gpu_random.h
 * @brief Header for GPU-based cryptographically secure random number generation
 * @author KeyhuntCUDA Team
 * 
 * T040: Develop GPU random number generation with cryptographically secure entropy sources
 * 
 * Provides high-performance cryptographically secure random number generation for:
 * - Private key generation and testing
 * - Statistical validation and Monte Carlo methods
 * - Cryptographic nonces and initialization vectors
 * - Performance benchmarking with controlled randomness
 */

#pragma once

#include "secp256k1.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <string>

namespace keyhunt {
namespace crypto {
namespace gpu {

/**
 * @brief Cryptographically secure pseudo-random number generator states
 * Supports multiple PRNG algorithms optimized for different use cases
 */
enum class PRNGAlgorithm {
    CURAND_XORWOW,          // CUDA's default, fast for general use
    CURAND_MRG32K3A,        // High-quality, longer period
    CURAND_MTGP32,          // Mersenne Twister for GPU, very high quality
    CURAND_PHILOX,          // Counter-based, stateless, cryptographically secure
    CURAND_SOBOL32,         // Quasi-random sequence for Monte Carlo
    CUSTOM_ChaCha20,        // Custom ChaCha20 implementation for max security
    CUSTOM_AES_CTR,         // Custom AES-CTR implementation
    CUSTOM_BLAKE2B          // Custom BLAKE2b-based PRNG
};

/**
 * @brief Entropy source configuration for seeding PRNGs
 */
enum class EntropySource {
    HARDWARE_RNG,           // Hardware random number generator (/dev/random)
    SYSTEM_ENTROPY,         // System entropy pool (/dev/urandom)  
    CRYPTO_API,             // Operating system crypto API
    TIME_BASED,             // High-resolution time + system state
    COMBINED_SOURCES,       // Combination of multiple sources
    DETERMINISTIC_SEED      // Fixed seed for reproducible testing
};

/**
 * @brief Configuration for GPU random number generation
 */
struct GPURandomConfig {
    PRNGAlgorithm algorithm;        // PRNG algorithm to use
    EntropySource entropy_source;   // Source of entropy for seeding
    size_t states_per_device;       // Number of PRNG states per GPU device
    size_t random_buffer_size;      // Size of device random number buffer
    bool enable_validation;         // Enable statistical validation
    bool use_shared_memory;         // Use shared memory for state caching
    uint32_t threads_per_block;     // Threads per block for generation
    
    GPURandomConfig() 
        : algorithm(PRNGAlgorithm::CURAND_PHILOX), 
          entropy_source(EntropySource::COMBINED_SOURCES),
          states_per_device(65536), random_buffer_size(1024*1024),
          enable_validation(true), use_shared_memory(true),
          threads_per_block(256) {}
};

/**
 * @brief CUDA kernel function declarations for random number generation
 */

// Initialize PRNG states with entropy
__global__ void initialize_random_states(curandState* states, 
                                        unsigned long long seed, 
                                        size_t num_states);

// Generate random 32-bit integers
__global__ void generate_random_uint32(curandState* states,
                                      uint32_t* output,
                                      size_t num_values,
                                      size_t num_states);

// Generate random 64-bit integers  
__global__ void generate_random_uint64(curandState* states,
                                      uint64_t* output,
                                      size_t num_values,
                                      size_t num_states);

// Generate cryptographically secure random BigInt256 values
__global__ void generate_random_bigint256(curandState* states,
                                         uint64_t* output,
                                         size_t num_values,
                                         size_t num_states);

// Generate random values in specific range [min, max)
__global__ void generate_random_range(curandState* states,
                                     uint64_t* output,
                                     uint64_t min_val,
                                     uint64_t max_val,
                                     size_t num_values,
                                     size_t num_states);

// Generate random private keys for secp256k1 (must be < curve order n)
__global__ void generate_random_private_keys(curandState* states,
                                            uint64_t* private_keys,
                                            const uint64_t* curve_order,
                                            size_t num_keys,
                                            size_t num_states);

// Generate random secp256k1 points for testing
__global__ void generate_random_points(curandState* states,
                                      uint64_t* points_x,
                                      uint64_t* points_y,
                                      size_t num_points,
                                      size_t num_states);

// ChaCha20-based cryptographically secure generation
__global__ void generate_chacha20_random(uint32_t* key,
                                        uint32_t* nonce,
                                        uint32_t counter,
                                        uint32_t* output,
                                        size_t num_blocks);

// AES-CTR based cryptographically secure generation
__global__ void generate_aes_ctr_random(uint32_t* key,
                                       uint32_t* iv,
                                       uint32_t counter,
                                       uint32_t* output,
                                       size_t num_blocks);

/**
 * @brief High-level C++ interface for GPU random number generation
 */
class GPURandomGenerator {
public:
    explicit GPURandomGenerator(const GPURandomConfig& config = GPURandomConfig());
    ~GPURandomGenerator();
    
    // Initialization and cleanup
    bool initialize(int device_id = 0);
    void cleanup();
    
    // Entropy collection and seeding
    bool collect_system_entropy(std::vector<uint8_t>& entropy_data, size_t bytes = 1024);
    bool seed_from_entropy_source();
    bool seed_from_data(const std::vector<uint8_t>& entropy_data);
    void seed_deterministic(uint64_t seed); // For testing reproducibility
    
    // Random number generation
    cudaError_t generate_uint32(std::vector<uint32_t>& output, size_t count);
    cudaError_t generate_uint64(std::vector<uint64_t>& output, size_t count);
    cudaError_t generate_bigint256(std::vector<ecc::BigInt256>& output, size_t count);
    
    // Specialized generation for cryptographic applications
    cudaError_t generate_private_keys(std::vector<ecc::BigInt256>& private_keys, size_t count);
    cudaError_t generate_random_points(std::vector<ecc::Point>& points, size_t count);
    cudaError_t generate_range(std::vector<uint64_t>& output, 
                              uint64_t min_val, uint64_t max_val, size_t count);
    
    // High-performance batch generation
    cudaError_t generate_batch_uint64(uint64_t* device_output, size_t count);
    cudaError_t generate_batch_bigint256(uint64_t* device_output, size_t count);
    cudaError_t generate_batch_private_keys(uint64_t* device_private_keys, size_t count);
    
    // Cryptographically secure algorithms
    cudaError_t generate_chacha20(const std::vector<uint8_t>& key,
                                 const std::vector<uint8_t>& nonce,
                                 std::vector<uint8_t>& output,
                                 size_t bytes);
                                 
    cudaError_t generate_aes_ctr(const std::vector<uint8_t>& key,
                                const std::vector<uint8_t>& iv,
                                std::vector<uint8_t>& output,
                                size_t bytes);
    
    // Statistical validation and testing
    struct RandomnessMetrics {
        double entropy_estimate;        // Estimated entropy per bit
        double chi_square_statistic;    // Chi-square test statistic
        double kolmogorov_smirnov_p;   // K-S test p-value
        double autocorrelation_max;     // Maximum autocorrelation
        size_t samples_tested;          // Number of samples in validation
        bool passes_fips_140_2;        // FIPS 140-2 randomness tests
        bool passes_nist_sp800_22;     // NIST SP 800-22 statistical tests
    };
    
    RandomnessMetrics validate_randomness(size_t sample_size = 1000000);
    bool run_statistical_tests();
    
    // Performance monitoring
    struct PerformanceMetrics {
        double generation_rate_mbps;    // Generation rate in MB/s
        double samples_per_second;      // Samples generated per second
        double gpu_utilization;         // GPU utilization percentage
        double memory_bandwidth_gbps;   // Memory bandwidth utilization
        std::chrono::milliseconds avg_generation_time;
    };
    
    PerformanceMetrics get_performance_metrics() const;
    void reset_performance_counters();
    
    // Configuration management
    GPURandomConfig get_current_config() const { return config_; }
    bool update_config(const GPURandomConfig& new_config);
    
    // Device management
    int get_device_id() const { return device_id_; }
    size_t get_num_states() const { return config_.states_per_device; }
    size_t get_buffer_size() const { return config_.random_buffer_size; }

private:
    GPURandomConfig config_;
    bool initialized_;
    int device_id_;
    
    // CUDA resources
    curandState* d_random_states_;      // Device PRNG states
    uint64_t* d_random_buffer_;         // Device random number buffer
    cudaStream_t random_stream_;        // CUDA stream for generation
    cudaStream_t memory_stream_;        // CUDA stream for memory operations
    
    // Performance tracking
    mutable PerformanceMetrics metrics_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    
    // Internal entropy management
    std::mt19937_64 host_rng_;          // Host RNG for entropy mixing
    std::vector<uint8_t> entropy_pool_; // Collected entropy
    uint64_t generation_counter_;       // Counter for stateless PRNGs
    
    // Memory management
    bool allocate_device_memory();
    void free_device_memory();
    
    // Entropy collection helpers
    bool collect_hardware_entropy(std::vector<uint8_t>& entropy);
    bool collect_system_entropy_internal(std::vector<uint8_t>& entropy);
    bool collect_time_entropy(std::vector<uint8_t>& entropy);
    uint64_t combine_entropy_sources(const std::vector<uint8_t>& entropy);
    
    // PRNG initialization helpers
    bool initialize_curand_states(uint64_t seed);
    bool initialize_custom_states(uint64_t seed);
    
    // Launch parameter optimization
    dim3 calculate_grid_size(size_t num_elements);
    dim3 calculate_block_size();
    
    // Statistical testing helpers
    double calculate_entropy(const std::vector<uint32_t>& data);
    double chi_square_test(const std::vector<uint32_t>& data);
    double kolmogorov_smirnov_test(const std::vector<double>& data);
    std::vector<double> autocorrelation_analysis(const std::vector<uint32_t>& data);
    bool fips_140_2_tests(const std::vector<uint8_t>& data);
};

/**
 * @brief Specialized entropy collectors for different platforms
 */
namespace entropy {
    
    /**
     * @brief Hardware random number generator interface
     */
    class HardwareRNG {
    public:
        static bool is_available();
        static bool collect_entropy(std::vector<uint8_t>& output, size_t bytes);
        static std::string get_hardware_info();
    };
    
    /**
     * @brief System entropy pool interface
     */
    class SystemEntropy {
    public:
        static bool collect_from_urandom(std::vector<uint8_t>& output, size_t bytes);
        static bool collect_from_random(std::vector<uint8_t>& output, size_t bytes);
        static size_t get_available_entropy();
    };
    
    /**
     * @brief Time-based entropy collector
     */
    class TimeEntropy {
    public:
        static uint64_t collect_high_resolution_time();
        static uint64_t collect_system_state_entropy();
        static std::vector<uint8_t> collect_timing_jitter(size_t iterations = 10000);
    };
    
    /**
     * @brief Entropy mixing and conditioning
     */
    class EntropyConditioner {
    public:
        // Von Neumann bias correction
        static std::vector<uint8_t> von_neumann_correct(const std::vector<uint8_t>& input);
        
        // Cryptographic hash-based conditioning
        static std::vector<uint8_t> sha256_condition(const std::vector<uint8_t>& input);
        static std::vector<uint8_t> blake2b_condition(const std::vector<uint8_t>& input);
        
        // Entropy mixing from multiple sources
        static std::vector<uint8_t> mix_entropy_sources(
            const std::vector<std::vector<uint8_t>>& sources);
    };
}

/**
 * @brief Custom cryptographically secure PRNG implementations
 */
namespace custom_prng {
    
    /**
     * @brief ChaCha20 stream cipher for random generation
     */
    class ChaCha20PRNG {
    public:
        explicit ChaCha20PRNG(const std::vector<uint8_t>& key, 
                             const std::vector<uint8_t>& nonce);
        
        void generate_bytes(std::vector<uint8_t>& output, size_t bytes);
        void generate_uint64(std::vector<uint64_t>& output, size_t count);
        
        // CUDA kernel implementation
        static cudaError_t generate_gpu(const uint32_t* key, const uint32_t* nonce,
                                      uint32_t* output, size_t num_blocks,
                                      cudaStream_t stream = nullptr);
        
    private:
        uint32_t state_[16];
        uint64_t counter_;
    };
    
    /**
     * @brief AES-CTR mode for random generation
     */
    class AESCTRPRNG {
    public:
        explicit AESCTRPRNG(const std::vector<uint8_t>& key, 
                           const std::vector<uint8_t>& iv);
        
        void generate_bytes(std::vector<uint8_t>& output, size_t bytes);
        void generate_uint64(std::vector<uint64_t>& output, size_t count);
        
        // CUDA kernel implementation
        static cudaError_t generate_gpu(const uint32_t* key, const uint32_t* iv,
                                      uint32_t* output, size_t num_blocks,
                                      cudaStream_t stream = nullptr);
        
    private:
        uint32_t round_keys_[60];  // Expanded AES keys
        uint32_t counter_[4];
        size_t key_bits_;
    };
    
    /**
     * @brief BLAKE2b-based PRNG
     */
    class Blake2bPRNG {
    public:
        explicit Blake2bPRNG(const std::vector<uint8_t>& seed);
        
        void generate_bytes(std::vector<uint8_t>& output, size_t bytes);
        void generate_uint64(std::vector<uint64_t>& output, size_t count);
        
        // Reseed the PRNG
        void reseed(const std::vector<uint8_t>& new_seed);
        
    private:
        uint64_t state_[8];
        uint64_t counter_;
        void blake2b_hash(const uint8_t* input, size_t len, uint8_t* output);
    };
}

/**
 * @brief Statistical testing suite for random number validation
 */
class RandomnessValidator {
public:
    RandomnessValidator();
    
    // NIST SP 800-22 statistical tests
    bool frequency_test(const std::vector<uint8_t>& data, double& p_value);
    bool block_frequency_test(const std::vector<uint8_t>& data, int block_length, double& p_value);
    bool runs_test(const std::vector<uint8_t>& data, double& p_value);
    bool longest_run_test(const std::vector<uint8_t>& data, double& p_value);
    bool discrete_fourier_transform_test(const std::vector<uint8_t>& data, double& p_value);
    bool non_overlapping_template_test(const std::vector<uint8_t>& data, double& p_value);
    bool overlapping_template_test(const std::vector<uint8_t>& data, double& p_value);
    bool maurers_universal_test(const std::vector<uint8_t>& data, double& p_value);
    bool linear_complexity_test(const std::vector<uint8_t>& data, double& p_value);
    bool serial_test(const std::vector<uint8_t>& data, double& p_value1, double& p_value2);
    bool approximate_entropy_test(const std::vector<uint8_t>& data, double& p_value);
    bool cumulative_sums_test(const std::vector<uint8_t>& data, double& p_value);
    bool random_excursions_test(const std::vector<uint8_t>& data, double& p_value);
    bool random_excursions_variant_test(const std::vector<uint8_t>& data, double& p_value);
    
    // FIPS 140-2 tests
    bool fips_monobit_test(const std::vector<uint8_t>& data);
    bool fips_poker_test(const std::vector<uint8_t>& data);
    bool fips_runs_test(const std::vector<uint8_t>& data);
    bool fips_long_run_test(const std::vector<uint8_t>& data);
    
    // Comprehensive validation
    struct ValidationReport {
        std::map<std::string, double> test_results;
        std::map<std::string, bool> test_passed;
        bool overall_passed;
        double confidence_level;
        size_t sample_size;
    };
    
    ValidationReport run_comprehensive_tests(const std::vector<uint8_t>& data);
    
private:
    double significance_level_;  // Alpha level for statistical tests
    
    // Helper functions
    double igamc(double a, double x);  // Incomplete gamma function
    double erfc(double x);             // Complementary error function
    std::vector<double> discrete_fourier_transform(const std::vector<int>& data);
};

/**
 * @brief Global registry for GPU random generators
 */
class GPURandomRegistry {
public:
    static GPURandomGenerator* get_instance(int device_id = 0);
    static void set_global_config(const GPURandomConfig& config);
    static void cleanup_all_instances();
    
    // Performance monitoring across all instances
    static std::vector<GPURandomGenerator::PerformanceMetrics> get_all_metrics();
    static void reset_all_performance_counters();
    
    // Global entropy collection
    static bool collect_global_entropy(std::vector<uint8_t>& entropy, size_t bytes);

private:
    static std::unordered_map<int, std::unique_ptr<GPURandomGenerator>> instances_;
    static GPURandomConfig global_config_;
    static std::mutex registry_mutex_;
};

} // namespace gpu
} // namespace crypto  
} // namespace keyhunt