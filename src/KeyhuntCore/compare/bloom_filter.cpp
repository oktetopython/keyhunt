/**
 * @file bloom_filter.cpp
 * @brief High-performance Bloom filter implementation for address filtering
 * @author KeyhuntCUDA Team
 * 
 * T044: Supporting Bloom filter for GPU-optimized address comparison
 */

#include "bitcoin_address_generator.h"
#include <cstring>
#include <cmath>
#include <iostream>

namespace keyhunt {
namespace compare {

BloomFilter::BloomFilter(size_t size_bits, double false_positive_rate)
    : size_bits_(size_bits)
    , size_bytes_((size_bits + 7) / 8)
    , gpu_bit_array_(nullptr)
    , gpu_uploaded_(false)
{
    // Calculate optimal number of hash functions
    num_hash_functions_ = static_cast<size_t>(-log(false_positive_rate) / (log(2) * log(2)) * size_bits / size_bits);
    num_hash_functions_ = std::max(1UL, std::min(num_hash_functions_, 10UL)); // Limit to reasonable range
    
    // Initialize bit array
    bit_array_.resize(size_bytes_, 0);
    
    std::cout << "BloomFilter initialized:" << std::endl;
    std::cout << "  Size: " << size_bits_ << " bits (" << size_bytes_ << " bytes)" << std::endl;
    std::cout << "  Hash functions: " << num_hash_functions_ << std::endl;
    std::cout << "  Target FP rate: " << false_positive_rate << std::endl;
}

BloomFilter::~BloomFilter() {
    cleanup_gpu();
}

void BloomFilter::add(const uint8_t* data, size_t length) {
    for (size_t i = 0; i < num_hash_functions_; i++) {
        uint32_t hash = get_hash(data, length, i);
        size_t bit_index = hash % size_bits_;
        size_t byte_index = bit_index / 8;
        size_t bit_offset = bit_index % 8;
        
        bit_array_[byte_index] |= (1 << bit_offset);
    }
}

void BloomFilter::add(const std::string& str) {
    add(reinterpret_cast<const uint8_t*>(str.c_str()), str.length());
}

bool BloomFilter::might_contain(const uint8_t* data, size_t length) const {
    for (size_t i = 0; i < num_hash_functions_; i++) {
        uint32_t hash = get_hash(data, length, i);
        size_t bit_index = hash % size_bits_;
        size_t byte_index = bit_index / 8;
        size_t bit_offset = bit_index % 8;
        
        if (!(bit_array_[byte_index] & (1 << bit_offset))) {
            return false; // Definitely not present
        }
    }
    return true; // Might be present
}

bool BloomFilter::might_contain(const std::string& str) const {
    return might_contain(reinterpret_cast<const uint8_t*>(str.c_str()), str.length());
}

void BloomFilter::clear() {
    std::fill(bit_array_.begin(), bit_array_.end(), 0);
    
    if (gpu_uploaded_) {
        // Clear GPU memory as well
        cudaMemset(gpu_bit_array_, 0, size_bytes_);
    }
}

double BloomFilter::false_positive_probability() const {
    // Count set bits
    size_t set_bits = 0;
    for (uint8_t byte : bit_array_) {
        set_bits += __builtin_popcount(byte);
    }
    
    // Calculate actual false positive probability
    double ratio = static_cast<double>(set_bits) / size_bits_;
    return std::pow(ratio, num_hash_functions_);
}

bool BloomFilter::upload_to_gpu() {
    if (gpu_uploaded_) {
        return true;
    }
    
    try {
        cudaError_t err = cudaMalloc(&gpu_bit_array_, size_bytes_);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory for Bloom filter: " 
                      << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMemcpy(gpu_bit_array_, bit_array_.data(), size_bytes_, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to upload Bloom filter to GPU: " 
                      << cudaGetErrorString(err) << std::endl;
            cudaFree(gpu_bit_array_);
            gpu_bit_array_ = nullptr;
            return false;
        }
        
        gpu_uploaded_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in upload_to_gpu: " << e.what() << std::endl;
        return false;
    }
}

bool BloomFilter::download_from_gpu() {
    if (!gpu_uploaded_) {
        return true;
    }
    
    try {
        cudaError_t err = cudaMemcpy(bit_array_.data(), gpu_bit_array_, size_bytes_, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to download Bloom filter from GPU: " 
                      << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in download_from_gpu: " << e.what() << std::endl;
        return false;
    }
}

void BloomFilter::cleanup_gpu() {
    if (gpu_bit_array_) {
        cudaFree(gpu_bit_array_);
        gpu_bit_array_ = nullptr;
    }
    gpu_uploaded_ = false;
}

uint32_t BloomFilter::hash1(const uint8_t* data, size_t length) const {
    // Simple FNV-1a hash
    uint32_t hash = 2166136261U;
    for (size_t i = 0; i < length; i++) {
        hash ^= data[i];
        hash *= 16777619U;
    }
    return hash;
}

uint32_t BloomFilter::hash2(const uint8_t* data, size_t length) const {
    // Simple djb2 hash
    uint32_t hash = 5381;
    for (size_t i = 0; i < length; i++) {
        hash = ((hash << 5) + hash) + data[i];
    }
    return hash;
}

uint32_t BloomFilter::get_hash(const uint8_t* data, size_t length, size_t i) const {
    // Double hashing: hash_i = hash1 + i * hash2
    return hash1(data, length) + i * hash2(data, length);
}

// Address Generator Factory Implementation

std::unique_ptr<BitcoinAddressGenerator> AddressGeneratorFactory::create_generator(
    OptimizationStrategy strategy, int device_id) {
    
    auto generator = std::make_unique<BitcoinAddressGenerator>();
    
    if (!generator->initialize(device_id)) {
        return nullptr;
    }
    
    // Configure based on strategy
    HashOperationConfig hash_config;
    AddressComparisonConfig comparison_config;
    
    switch (strategy) {
        case OptimizationStrategy::SPEED_OPTIMIZED:
            hash_config.batch_size = 131072;  // 128K
            hash_config.threads_per_block = 512;
            hash_config.blocks_per_grid = 512;
            hash_config.enable_shared_memory_optimization = true;
            hash_config.enable_fused_hash_operations = true;
            
            comparison_config.enable_gpu_comparison = true;
            comparison_config.enable_bloom_filter = true;
            comparison_config.bloom_filter_size = 2097152;  // 2MB
            break;
            
        case OptimizationStrategy::MEMORY_OPTIMIZED:
            hash_config.batch_size = 32768;   // 32K
            hash_config.threads_per_block = 256;
            hash_config.blocks_per_grid = 128;
            hash_config.enable_shared_memory_optimization = false;
            
            comparison_config.enable_bloom_filter = true;
            comparison_config.bloom_filter_size = 524288;   // 512KB
            break;
            
        case OptimizationStrategy::POWER_EFFICIENT:
            hash_config.batch_size = 16384;   // 16K
            hash_config.threads_per_block = 128;
            hash_config.blocks_per_grid = 64;
            hash_config.enable_shared_memory_optimization = false;
            hash_config.enable_fused_hash_operations = false;
            
            comparison_config.enable_gpu_comparison = false;
            comparison_config.enable_bloom_filter = true;
            comparison_config.bloom_filter_size = 262144;   // 256KB
            break;
            
        default: // BALANCED
            hash_config.batch_size = 65536;   // 64K
            hash_config.threads_per_block = 256;
            hash_config.blocks_per_grid = 256;
            hash_config.enable_shared_memory_optimization = true;
            hash_config.enable_fused_hash_operations = true;
            
            comparison_config.enable_gpu_comparison = true;
            comparison_config.enable_bloom_filter = true;
            comparison_config.bloom_filter_size = 1048576;  // 1MB
            break;
    }
    
    generator->configure_hash_operations(hash_config);
    generator->configure_address_comparison(comparison_config);
    
    return generator;
}

HashOperationConfig AddressGeneratorFactory::get_recommended_hash_config(
    OptimizationStrategy strategy, size_t available_memory_mb) {
    
    HashOperationConfig config;
    
    // Adjust batch size based on available memory
    size_t max_batch_size = (available_memory_mb * 1024 * 1024) / (65 + 32 + 20 + 20); // Key + SHA + RIPE + Hash160
    max_batch_size = std::min(max_batch_size, size_t(262144)); // Max 256K
    max_batch_size = std::max(max_batch_size, size_t(1024));   // Min 1K
    
    config.batch_size = max_batch_size;
    
    switch (strategy) {
        case OptimizationStrategy::SPEED_OPTIMIZED:
            config.threads_per_block = 512;
            config.blocks_per_grid = 512;
            config.enable_shared_memory_optimization = true;
            config.enable_fused_hash_operations = true;
            break;
            
        case OptimizationStrategy::MEMORY_OPTIMIZED:
            config.batch_size = std::min(config.batch_size, size_t(32768));
            config.threads_per_block = 256;
            config.blocks_per_grid = 128;
            config.enable_shared_memory_optimization = false;
            break;
            
        default:
            config.threads_per_block = 256;
            config.blocks_per_grid = 256;
            config.enable_shared_memory_optimization = true;
            break;
    }
    
    return config;
}

AddressComparisonConfig AddressGeneratorFactory::get_recommended_comparison_config(
    OptimizationStrategy strategy, size_t target_address_count) {
    
    AddressComparisonConfig config;
    
    // Adjust Bloom filter size based on target count
    size_t bloom_bits = target_address_count * 10; // 10 bits per element
    bloom_bits = std::max(bloom_bits, size_t(65536));   // Min 64KB
    bloom_bits = std::min(bloom_bits, size_t(16777216)); // Max 16MB
    
    config.bloom_filter_size = bloom_bits;
    config.bloom_filter_false_positive_rate = 0.001; // 0.1%
    
    switch (strategy) {
        case OptimizationStrategy::SPEED_OPTIMIZED:
            config.enable_gpu_comparison = true;
            config.enable_bloom_filter = true;
            config.comparison_batch_size = 65536;
            break;
            
        case OptimizationStrategy::MEMORY_OPTIMIZED:
            config.enable_gpu_comparison = false;
            config.enable_bloom_filter = true;
            config.bloom_filter_size = std::min(config.bloom_filter_size, size_t(1048576)); // Max 1MB
            config.comparison_batch_size = 16384;
            break;
            
        case OptimizationStrategy::POWER_EFFICIENT:
            config.enable_gpu_comparison = false;
            config.enable_bloom_filter = true;
            config.bloom_filter_size = std::min(config.bloom_filter_size, size_t(524288)); // Max 512KB
            config.comparison_batch_size = 8192;
            break;
            
        default: // BALANCED
            config.enable_gpu_comparison = true;
            config.enable_bloom_filter = true;
            config.comparison_batch_size = 32768;
            break;
    }
    
    return config;
}

// Address utilities namespace implementation

namespace address_utils {

bool validate_bitcoin_address_format(const std::string& address) {
    return BitcoinAddressGenerator::validate_address(address);
}

AddressFormat classify_bitcoin_address(const std::string& address) {
    return BitcoinAddressGenerator::detect_address_format(address);
}

std::string normalize_bitcoin_address(const std::string& address) {
    // Remove whitespace and convert to standard case
    std::string normalized;
    for (char c : address) {
        if (!std::isspace(c)) {
            normalized += c;
        }
    }
    return normalized;
}

BenchmarkResults benchmark_address_generation(
    BitcoinAddressGenerator& generator, size_t test_key_count) {
    
    BenchmarkResults results;
    
    try {
        // Generate test public keys
        std::vector<ecc::Point> test_keys;
        test_keys.reserve(test_key_count);
        
        for (size_t i = 0; i < test_key_count; i++) {
            ecc::Point key;
            // Generate simple test key (placeholder)
            key.x.d[0] = i;
            key.y.d[0] = i * 2;
            test_keys.push_back(key);
        }
        
        // Benchmark Hash160 generation
        auto start_time = std::chrono::high_resolution_clock::now();
        
        AddressGenerationContext context;
        bool success = generator.generate_addresses_from_public_keys(test_keys, context, true);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (success && duration.count() > 0) {
            results.hash160_rate_mkeys_per_sec = (test_key_count / 1000000.0) / (duration.count() / 1000.0);
            results.address_gen_rate_maddr_per_sec = results.hash160_rate_mkeys_per_sec; // Same rate
        }
        
        // Benchmark comparison if targets exist
        std::vector<std::string> dummy_targets = {"1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"};
        generator.set_target_addresses(dummy_targets);
        
        start_time = std::chrono::high_resolution_clock::now();
        auto matches = generator.compare_addresses(context);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (duration.count() > 0) {
            results.comparison_rate_mcomp_per_sec = (test_key_count / 1000000.0) / (duration.count() / 1000.0);
        }
        
        // Estimate memory bandwidth and GPU utilization
        auto metrics = generator.get_current_metrics();
        results.memory_bandwidth_gb_s = 500; // Placeholder - would need actual measurement
        results.gpu_utilization_percent = 75.0; // Placeholder
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << std::endl;
    }
    
    return results;
}

// AddressSet implementation

void AddressSet::add_address(const std::string& address) {
    AddressFormat format = BitcoinAddressGenerator::detect_address_format(address);
    if (format != AddressFormat::UNKNOWN) {
        addresses_by_format_[format].insert(address);
        
        // Extract and store Hash160
        uint8_t hash160[20];
        // This would need proper implementation to extract Hash160 from address
        memset(hash160, 0, 20);
        memcpy(address_to_hash160_[address], hash160, 20);
    }
}

void AddressSet::add_addresses(const std::vector<std::string>& addresses) {
    for (const auto& address : addresses) {
        add_address(address);
    }
}

bool AddressSet::contains_address(const std::string& address) const {
    AddressFormat format = BitcoinAddressGenerator::detect_address_format(address);
    if (format == AddressFormat::UNKNOWN) {
        return false;
    }
    
    auto format_it = addresses_by_format_.find(format);
    if (format_it == addresses_by_format_.end()) {
        return false;
    }
    
    return format_it->second.find(address) != format_it->second.end();
}

void AddressSet::remove_address(const std::string& address) {
    AddressFormat format = BitcoinAddressGenerator::detect_address_format(address);
    if (format != AddressFormat::UNKNOWN) {
        addresses_by_format_[format].erase(address);
        address_to_hash160_.erase(address);
    }
}

void AddressSet::clear() {
    addresses_by_format_.clear();
    address_to_hash160_.clear();
}

size_t AddressSet::size() const {
    size_t total = 0;
    for (const auto& [format, addresses] : addresses_by_format_) {
        total += addresses.size();
    }
    return total;
}

std::vector<std::string> AddressSet::get_all_addresses() const {
    std::vector<std::string> all_addresses;
    for (const auto& [format, addresses] : addresses_by_format_) {
        all_addresses.insert(all_addresses.end(), addresses.begin(), addresses.end());
    }
    return all_addresses;
}

std::vector<std::string> AddressSet::get_addresses_by_format(AddressFormat format) const {
    std::vector<std::string> addresses;
    auto it = addresses_by_format_.find(format);
    if (it != addresses_by_format_.end()) {
        addresses.assign(it->second.begin(), it->second.end());
    }
    return addresses;
}

std::vector<uint8_t> AddressSet::export_hash160_values() const {
    std::vector<uint8_t> hash160_data;
    hash160_data.reserve(address_to_hash160_.size() * 20);
    
    for (const auto& [address, hash160] : address_to_hash160_) {
        hash160_data.insert(hash160_data.end(), hash160, hash160 + 20);
    }
    
    return hash160_data;
}

} // namespace address_utils

} // namespace compare
} // namespace keyhunt