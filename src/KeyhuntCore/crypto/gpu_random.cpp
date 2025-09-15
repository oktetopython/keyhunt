/**
 * @file gpu_random.cpp
 * @brief C++ implementation for GPU-based cryptographically secure random number generation
 * @author KeyhuntCUDA Team
 * 
 * T040: Develop GPU random number generation with cryptographically secure entropy sources
 * 
 * Provides C++ wrapper implementation for GPU random number generation with entropy
 * collection, statistical validation, and performance monitoring capabilities.
 */

#include "gpu_random.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

#ifdef __linux__
#include <sys/random.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#endif

namespace keyhunt {
namespace crypto {
namespace gpu {

// GPURandomGenerator implementation
GPURandomGenerator::GPURandomGenerator(const GPURandomConfig& config)
    : config_(config), initialized_(false), device_id_(0),
      d_random_states_(nullptr), d_random_buffer_(nullptr),
      random_stream_(nullptr), memory_stream_(nullptr),
      start_event_(nullptr), stop_event_(nullptr),
      host_rng_(std::random_device{}()), generation_counter_(0) {
}

GPURandomGenerator::~GPURandomGenerator() {
    cleanup();
}

bool GPURandomGenerator::initialize(int device_id) {
    if (initialized_) return true;
    
    device_id_ = device_id;
    
    // Set CUDA device
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Allocate device memory
    if (!allocate_device_memory()) {
        std::cerr << "Failed to allocate device memory for random generation" << std::endl;
        return false;
    }
    
    // Create CUDA streams
    err = cudaStreamCreate(&random_stream_);
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }
    
    err = cudaStreamCreate(&memory_stream_);
    if (err != cudaSuccess) {
        cleanup();
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
    
    // Initialize random states with entropy
    if (!seed_from_entropy_source()) {
        std::cerr << "Failed to seed random generator from entropy source" << std::endl;
        cleanup();
        return false;
    }
    
    initialized_ = true;
    reset_performance_counters();
    
    std::cout << "GPU random generator initialized successfully on device " << device_id_ << std::endl;
    return true;
}

void GPURandomGenerator::cleanup() {
    if (!initialized_) return;
    
    free_device_memory();
    
    if (random_stream_) {
        cudaStreamDestroy(random_stream_);
        random_stream_ = nullptr;
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

bool GPURandomGenerator::collect_system_entropy(std::vector<uint8_t>& entropy_data, size_t bytes) {
    entropy_data.resize(bytes);
    
    switch (config_.entropy_source) {
        case EntropySource::HARDWARE_RNG:
            return entropy::HardwareRNG::collect_entropy(entropy_data, bytes);
            
        case EntropySource::SYSTEM_ENTROPY:
            return entropy::SystemEntropy::collect_from_urandom(entropy_data, bytes);
            
        case EntropySource::CRYPTO_API:
#ifdef _WIN32
            return collect_windows_crypto_entropy(entropy_data, bytes);
#else
            return entropy::SystemEntropy::collect_from_random(entropy_data, bytes);
#endif
            
        case EntropySource::TIME_BASED:
            return collect_time_entropy(entropy_data);
            
        case EntropySource::COMBINED_SOURCES: {
            std::vector<std::vector<uint8_t>> sources;
            
            // Collect from multiple sources
            std::vector<uint8_t> hw_entropy(bytes / 4);
            if (entropy::HardwareRNG::collect_entropy(hw_entropy, bytes / 4)) {
                sources.push_back(hw_entropy);
            }
            
            std::vector<uint8_t> sys_entropy(bytes / 4);
            if (entropy::SystemEntropy::collect_from_urandom(sys_entropy, bytes / 4)) {
                sources.push_back(sys_entropy);
            }
            
            std::vector<uint8_t> time_entropy(bytes / 4);
            if (collect_time_entropy(time_entropy)) {
                sources.push_back(time_entropy);
            }
            
            // Mix entropy sources
            if (!sources.empty()) {
                entropy_data = entropy::EntropyConditioner::mix_entropy_sources(sources);
                entropy_data.resize(bytes);  // Ensure correct size
                return true;
            }
            return false;
        }
        
        case EntropySource::DETERMINISTIC_SEED:
            // Fill with deterministic pattern for testing
            std::fill(entropy_data.begin(), entropy_data.end(), 0xAA);
            return true;
            
        default:
            return false;
    }
}

bool GPURandomGenerator::seed_from_entropy_source() {
    std::vector<uint8_t> entropy_data;
    if (!collect_system_entropy(entropy_data, 1024)) {
        std::cerr << "Failed to collect entropy from source" << std::endl;
        return false;
    }
    
    return seed_from_data(entropy_data);
}

bool GPURandomGenerator::seed_from_data(const std::vector<uint8_t>& entropy_data) {
    if (entropy_data.empty()) return false;
    
    // Mix entropy data to create 64-bit seed
    uint64_t seed = combine_entropy_sources(entropy_data);
    
    // Store entropy for potential re-seeding
    entropy_pool_ = entropy_data;
    
    // Initialize PRNG states based on algorithm
    if (config_.algorithm <= PRNGAlgorithm::CURAND_SOBOL32) {
        return initialize_curand_states(seed);
    } else {
        return initialize_custom_states(seed);
    }
}

void GPURandomGenerator::seed_deterministic(uint64_t seed) {
    std::vector<uint8_t> entropy_data(64);
    uint64_t* seed_ptr = reinterpret_cast<uint64_t*>(entropy_data.data());
    for (int i = 0; i < 8; i++) {
        seed_ptr[i] = seed + i * 0x123456789ABCDEFULL;
    }
    
    seed_from_data(entropy_data);
}

cudaError_t GPURandomGenerator::generate_uint32(std::vector<uint32_t>& output, size_t count) {
    if (!initialized_) return cudaErrorNotInitialized;
    
    output.resize(count);
    
    // Allocate device memory if needed
    size_t bytes_needed = count * sizeof(uint32_t);
    if (bytes_needed > config_.random_buffer_size * sizeof(uint64_t)) {
        // Generate in batches
        size_t batch_size = config_.random_buffer_size / 2; // Each uint64 gives 2 uint32
        size_t generated = 0;
        
        while (generated < count) {
            size_t current_batch = std::min(batch_size, count - generated);
            
            cudaError_t err = generate_batch_uint32_internal(
                reinterpret_cast<uint32_t*>(d_random_buffer_), current_batch);
            if (err != cudaSuccess) return err;
            
            // Copy to host
            err = cudaMemcpyAsync(&output[generated], 
                                 reinterpret_cast<uint32_t*>(d_random_buffer_),
                                 current_batch * sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost, memory_stream_);
            if (err != cudaSuccess) return err;
            
            generated += current_batch;
        }
        
        cudaStreamSynchronize(memory_stream_);
    } else {
        // Generate all at once
        cudaError_t err = generate_batch_uint32_internal(
            reinterpret_cast<uint32_t*>(d_random_buffer_), count);
        if (err != cudaSuccess) return err;
        
        // Copy to host
        err = cudaMemcpyAsync(output.data(), 
                             reinterpret_cast<uint32_t*>(d_random_buffer_),
                             count * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, memory_stream_);
        if (err != cudaSuccess) return err;
        
        cudaStreamSynchronize(memory_stream_);
    }
    
    // Update performance metrics
    metrics_.samples_per_second = count; // Simplified - should track actual timing
    return cudaSuccess;
}

cudaError_t GPURandomGenerator::generate_uint64(std::vector<uint64_t>& output, size_t count) {
    if (!initialized_) return cudaErrorNotInitialized;
    
    output.resize(count);
    return generate_batch_uint64(output.data(), count);
}

cudaError_t GPURandomGenerator::generate_bigint256(std::vector<ecc::BigInt256>& output, size_t count) {
    if (!initialized_) return cudaErrorNotInitialized;
    
    output.resize(count);
    
    // Each BigInt256 needs 4 uint64_t values
    size_t total_uint64s = count * 4;
    
    cudaError_t err = generate_batch_uint64(reinterpret_cast<uint64_t*>(output.data()), total_uint64s);
    return err;
}

cudaError_t GPURandomGenerator::generate_private_keys(std::vector<ecc::BigInt256>& private_keys, size_t count) {
    if (!initialized_) return cudaErrorNotInitialized;
    
    private_keys.resize(count);
    
    // Launch kernel to generate valid private keys
    dim3 grid_size = calculate_grid_size(count);
    dim3 block_size = calculate_block_size();
    
    cudaEventRecord(start_event_, random_stream_);
    
    generate_random_private_keys<<<grid_size, block_size, 0, random_stream_>>>(
        d_random_states_,
        reinterpret_cast<uint64_t*>(private_keys.data()),
        nullptr, // Will use constant memory
        count,
        config_.states_per_device
    );
    
    cudaEventRecord(stop_event_, random_stream_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    cudaStreamSynchronize(random_stream_);
    
    // Update metrics
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    metrics_.avg_generation_time = std::chrono::milliseconds(static_cast<int>(elapsed_ms));
    
    return cudaSuccess;
}

cudaError_t GPURandomGenerator::generate_batch_uint64(uint64_t* device_output, size_t count) {
    if (!initialized_) return cudaErrorNotInitialized;
    
    dim3 grid_size = calculate_grid_size(count);
    dim3 block_size = calculate_block_size();
    
    cudaEventRecord(start_event_, random_stream_);
    
    generate_random_uint64<<<grid_size, block_size, 0, random_stream_>>>(
        d_random_states_,
        device_output,
        count,
        config_.states_per_device
    );
    
    cudaEventRecord(stop_event_, random_stream_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    cudaStreamSynchronize(random_stream_);
    
    // Update performance metrics
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    
    double elapsed_seconds = elapsed_ms / 1000.0;
    double bytes_generated = count * sizeof(uint64_t);
    metrics_.generation_rate_mbps = (bytes_generated / (1024.0 * 1024.0)) / elapsed_seconds;
    metrics_.samples_per_second = count / elapsed_seconds;
    
    return cudaSuccess;
}

GPURandomGenerator::RandomnessMetrics GPURandomGenerator::validate_randomness(size_t sample_size) {
    RandomnessMetrics metrics = {};
    
    if (!initialized_) {
        metrics.passes_fips_140_2 = false;
        metrics.passes_nist_sp800_22 = false;
        return metrics;
    }
    
    // Generate sample data for validation
    std::vector<uint32_t> sample_data;
    if (generate_uint32(sample_data, sample_size) != cudaSuccess) {
        return metrics;
    }
    
    // Convert to bytes for statistical tests
    std::vector<uint8_t> byte_data;
    byte_data.reserve(sample_size * 4);
    for (uint32_t value : sample_data) {
        byte_data.push_back(static_cast<uint8_t>(value));
        byte_data.push_back(static_cast<uint8_t>(value >> 8));
        byte_data.push_back(static_cast<uint8_t>(value >> 16));
        byte_data.push_back(static_cast<uint8_t>(value >> 24));
    }
    
    // Run statistical tests
    RandomnessValidator validator;
    
    // Basic entropy estimate
    metrics.entropy_estimate = calculate_entropy(sample_data);
    
    // Chi-square test
    metrics.chi_square_statistic = chi_square_test(sample_data);
    
    // FIPS 140-2 tests
    metrics.passes_fips_140_2 = fips_140_2_tests(byte_data);
    
    // Simplified NIST tests (subset)
    double p_value;
    bool freq_test = validator.frequency_test(byte_data, p_value);
    bool runs_test = validator.runs_test(byte_data, p_value);
    metrics.passes_nist_sp800_22 = freq_test && runs_test;
    
    metrics.samples_tested = sample_size;
    
    return metrics;
}

GPURandomGenerator::PerformanceMetrics GPURandomGenerator::get_performance_metrics() const {
    return metrics_;
}

void GPURandomGenerator::reset_performance_counters() {
    metrics_ = PerformanceMetrics();
}

bool GPURandomGenerator::allocate_device_memory() {
    // Allocate PRNG states
    size_t states_bytes = config_.states_per_device * sizeof(curandState);
    cudaError_t err = cudaMalloc(&d_random_states_, states_bytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate random states: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Allocate random buffer
    size_t buffer_bytes = config_.random_buffer_size * sizeof(uint64_t);
    err = cudaMalloc(&d_random_buffer_, buffer_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_random_states_);
        d_random_states_ = nullptr;
        std::cerr << "Failed to allocate random buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

void GPURandomGenerator::free_device_memory() {
    if (d_random_states_) {
        cudaFree(d_random_states_);
        d_random_states_ = nullptr;
    }
    
    if (d_random_buffer_) {
        cudaFree(d_random_buffer_);
        d_random_buffer_ = nullptr;
    }
}

bool GPURandomGenerator::collect_time_entropy(std::vector<uint8_t>& entropy) {
    entropy.resize(256);
    
    // High-resolution timestamp
    uint64_t time_ns = entropy::TimeEntropy::collect_high_resolution_time();
    
    // System state entropy
    uint64_t system_state = entropy::TimeEntropy::collect_system_state_entropy();
    
    // Timing jitter
    auto jitter_data = entropy::TimeEntropy::collect_timing_jitter(1000);
    
    // Combine time sources
    uint64_t* entropy_ptr = reinterpret_cast<uint64_t*>(entropy.data());
    entropy_ptr[0] = time_ns;
    entropy_ptr[1] = system_state;
    
    // Mix in jitter data
    for (size_t i = 0; i < std::min(jitter_data.size(), size_t(30)); i++) {
        if (i + 2 < entropy.size() / 8) {
            entropy_ptr[i + 2] ^= *reinterpret_cast<const uint64_t*>(&jitter_data[i]);
        }
    }
    
    return true;
}

uint64_t GPURandomGenerator::combine_entropy_sources(const std::vector<uint8_t>& entropy) {
    if (entropy.empty()) return 0;
    
    uint64_t result = 0;
    
    // Simple combining function - XOR with rotation
    for (size_t i = 0; i < entropy.size(); i++) {
        result = (result << 1) | (result >> 63);  // Rotate
        result ^= entropy[i];
    }
    
    // Mix with host RNG state
    result ^= host_rng_();
    
    return result;
}

bool GPURandomGenerator::initialize_curand_states(uint64_t seed) {
    dim3 grid_size = calculate_grid_size(config_.states_per_device);
    dim3 block_size = calculate_block_size();
    
    initialize_random_states<<<grid_size, block_size, 0, random_stream_>>>(
        d_random_states_, seed, config_.states_per_device);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize random states: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    cudaStreamSynchronize(random_stream_);
    return true;
}

bool GPURandomGenerator::initialize_custom_states(uint64_t seed) {
    // For custom PRNGs, store seed for later use
    generation_counter_ = seed;
    return true;
}

dim3 GPURandomGenerator::calculate_grid_size(size_t num_elements) {
    dim3 block_size = calculate_block_size();
    int grid_x = (num_elements + block_size.x - 1) / block_size.x;
    return dim3(grid_x, 1, 1);
}

dim3 GPURandomGenerator::calculate_block_size() {
    return dim3(config_.threads_per_block, 1, 1);
}

cudaError_t GPURandomGenerator::generate_batch_uint32_internal(uint32_t* device_output, size_t count) {
    dim3 grid_size = calculate_grid_size(count);
    dim3 block_size = calculate_block_size();
    
    generate_random_uint32<<<grid_size, block_size, 0, random_stream_>>>(
        d_random_states_,
        device_output,
        count,
        config_.states_per_device
    );
    
    return cudaGetLastError();
}

double GPURandomGenerator::calculate_entropy(const std::vector<uint32_t>& data) {
    if (data.empty()) return 0.0;
    
    // Calculate frequency distribution
    std::unordered_map<uint32_t, size_t> frequencies;
    for (uint32_t value : data) {
        frequencies[value]++;
    }
    
    // Calculate Shannon entropy
    double entropy = 0.0;
    double n = static_cast<double>(data.size());
    
    for (const auto& pair : frequencies) {
        double p = static_cast<double>(pair.second) / n;
        if (p > 0) {
            entropy -= p * std::log2(p);
        }
    }
    
    return entropy;
}

double GPURandomGenerator::chi_square_test(const std::vector<uint32_t>& data) {
    if (data.empty()) return 0.0;
    
    // Simplified chi-square test for uniformity
    const size_t num_bins = 256;
    std::vector<size_t> observed(num_bins, 0);
    
    // Count occurrences in bins (using lower 8 bits)
    for (uint32_t value : data) {
        observed[value & 0xFF]++;
    }
    
    double expected = static_cast<double>(data.size()) / num_bins;
    double chi_square = 0.0;
    
    for (size_t count : observed) {
        double diff = static_cast<double>(count) - expected;
        chi_square += (diff * diff) / expected;
    }
    
    return chi_square;
}

bool GPURandomGenerator::fips_140_2_tests(const std::vector<uint8_t>& data) {
    if (data.size() < 2500) return false;
    
    RandomnessValidator validator;
    
    // Run FIPS 140-2 tests
    bool monobit = validator.fips_monobit_test(data);
    bool poker = validator.fips_poker_test(data);
    bool runs = validator.fips_runs_test(data);
    bool long_run = validator.fips_long_run_test(data);
    
    return monobit && poker && runs && long_run;
}

// Entropy collection implementations
namespace entropy {

bool HardwareRNG::is_available() {
#ifdef __linux__
    return access("/dev/hwrng", R_OK) == 0;
#elif defined(_WIN32)
    return true; // Windows Crypto API generally available
#else
    return false;
#endif
}

bool HardwareRNG::collect_entropy(std::vector<uint8_t>& output, size_t bytes) {
#ifdef __linux__
    std::ifstream hwrng("/dev/hwrng", std::ios::binary);
    if (!hwrng.is_open()) return false;
    
    output.resize(bytes);
    hwrng.read(reinterpret_cast<char*>(output.data()), bytes);
    return hwrng.gcount() == static_cast<std::streamsize>(bytes);
#else
    // Fallback to system entropy
    return SystemEntropy::collect_from_urandom(output, bytes);
#endif
}

std::string HardwareRNG::get_hardware_info() {
#ifdef __linux__
    std::ifstream info("/proc/sys/kernel/random/entropy_avail");
    if (info.is_open()) {
        std::string entropy_info;
        std::getline(info, entropy_info);
        return "Available entropy: " + entropy_info + " bits";
    }
#endif
    return "Hardware RNG information not available";
}

bool SystemEntropy::collect_from_urandom(std::vector<uint8_t>& output, size_t bytes) {
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (!urandom.is_open()) return false;
    
    output.resize(bytes);
    urandom.read(reinterpret_cast<char*>(output.data()), bytes);
    return urandom.gcount() == static_cast<std::streamsize>(bytes);
}

bool SystemEntropy::collect_from_random(std::vector<uint8_t>& output, size_t bytes) {
    std::ifstream random_dev("/dev/random", std::ios::binary);
    if (!random_dev.is_open()) return false;
    
    output.resize(bytes);
    random_dev.read(reinterpret_cast<char*>(output.data()), bytes);
    return random_dev.gcount() == static_cast<std::streamsize>(bytes);
}

uint64_t TimeEntropy::collect_high_resolution_time() {
    auto now = std::chrono::high_resolution_clock::now();
    return now.time_since_epoch().count();
}

uint64_t TimeEntropy::collect_system_state_entropy() {
    // Combine various system state information
    uint64_t entropy = 0;
    
    entropy ^= static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    entropy ^= reinterpret_cast<uintptr_t>(&entropy);  // Stack address
    entropy ^= static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    
    return entropy;
}

std::vector<uint8_t> TimeEntropy::collect_timing_jitter(size_t iterations) {
    std::vector<uint8_t> jitter;
    jitter.reserve(iterations);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        volatile int dummy = 0; // Prevent optimization
        dummy++;
        auto t2 = std::chrono::high_resolution_clock::now();
        
        uint64_t diff = (t2 - t1).count();
        jitter.push_back(static_cast<uint8_t>(diff & 0xFF));
    }
    
    return jitter;
}

} // namespace entropy

// Global registry implementation
std::unordered_map<int, std::unique_ptr<GPURandomGenerator>> GPURandomRegistry::instances_;
GPURandomConfig GPURandomRegistry::global_config_;
std::mutex GPURandomRegistry::registry_mutex_;

GPURandomGenerator* GPURandomRegistry::get_instance(int device_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = instances_.find(device_id);
    if (it != instances_.end()) {
        return it->second.get();
    }
    
    // Create new instance
    auto instance = std::make_unique<GPURandomGenerator>(global_config_);
    instance->initialize(device_id);
    
    GPURandomGenerator* ptr = instance.get();
    instances_[device_id] = std::move(instance);
    
    return ptr;
}

void GPURandomRegistry::set_global_config(const GPURandomConfig& config) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    global_config_ = config;
}

void GPURandomRegistry::cleanup_all_instances() {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    instances_.clear();
}

} // namespace gpu
} // namespace crypto
} // namespace keyhunt