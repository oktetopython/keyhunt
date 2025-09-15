/**
 * @file bitcoin_address_generator.cpp
 * @brief Bitcoin address generation and comparison pipeline implementation
 * @author KeyhuntCUDA Team
 * 
 * T044: Develop Bitcoin address generation and comparison pipeline with GPU-optimized hash operations
 */

#include "bitcoin_address_generator.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <random>
#include <xxhash.h>

namespace keyhunt {
namespace compare {

// Base58 alphabet and mapping
const char BitcoinAddressGenerator::base58_alphabet[] = 
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

int BitcoinAddressGenerator::base58_map[256];
static bool base58_map_initialized = false;

BitcoinAddressGenerator::BitcoinAddressGenerator()
    : device_id_(0)
    , is_initialized_(false)
    , computation_stream_(nullptr)
    , memory_stream_(nullptr)
{
    if (!base58_map_initialized) {
        init_base58_map();
        base58_map_initialized = true;
    }
    
    last_metrics_update_ = std::chrono::high_resolution_clock::now();
}

BitcoinAddressGenerator::~BitcoinAddressGenerator() {
    cleanup();
}

bool BitcoinAddressGenerator::initialize(int device_id) {
    try {
        device_id_ = device_id;
        
        // Set CUDA device
        cudaError_t err = cudaSetDevice(device_id_);
        if (err != cudaSuccess) {
            std::cerr << "ERROR: Failed to set CUDA device " << device_id_ << ": " 
                      << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Create CUDA streams
        err = cudaStreamCreate(&computation_stream_);
        if (err != cudaSuccess) {
            std::cerr << "ERROR: Failed to create computation stream: " 
                      << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaStreamCreate(&memory_stream_);
        if (err != cudaSuccess) {
            std::cerr << "ERROR: Failed to create memory stream: " 
                      << cudaGetErrorString(err) << std::endl;
            cudaStreamDestroy(computation_stream_);
            return false;
        }
        
        // Initialize Bloom filter
        if (comparison_config_.enable_bloom_filter) {
            bloom_filter_ = std::make_unique<BloomFilter>(
                comparison_config_.bloom_filter_size,
                comparison_config_.bloom_filter_false_positive_rate
            );
        }
        
        // Allocate initial GPU buffers
        if (!allocate_gpu_memory(hash_config_.batch_size)) {
            std::cerr << "ERROR: Failed to allocate GPU memory" << std::endl;
            cleanup();
            return false;
        }
        
        is_initialized_ = true;
        
        std::cout << "BitcoinAddressGenerator initialized successfully on device " << device_id_ << std::endl;
        std::cout << "  Hash batch size: " << hash_config_.batch_size << std::endl;
        std::cout << "  Bloom filter: " << (comparison_config_.enable_bloom_filter ? "Enabled" : "Disabled") << std::endl;
        std::cout << "  GPU comparison: " << (comparison_config_.enable_gpu_comparison ? "Enabled" : "Disabled") << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in BitcoinAddressGenerator::initialize: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

void BitcoinAddressGenerator::configure_hash_operations(const HashOperationConfig& config) {
    hash_config_ = config;
    
    // Reallocate GPU buffers if batch size changed
    if (is_initialized_ && config.batch_size != gpu_buffers_.allocated_size) {
        deallocate_gpu_memory();
        allocate_gpu_memory(config.batch_size);
    }
}

void BitcoinAddressGenerator::configure_address_comparison(const AddressComparisonConfig& config) {
    comparison_config_ = config;
    
    // Recreate Bloom filter if settings changed
    if (is_initialized_ && comparison_config_.enable_bloom_filter) {
        bloom_filter_ = std::make_unique<BloomFilter>(
            comparison_config_.bloom_filter_size,
            comparison_config_.bloom_filter_false_positive_rate
        );
        
        // Re-add existing target addresses to new Bloom filter
        for (const auto& [address, format] : target_addresses_) {
            bloom_filter_->add(address);
        }
    }
}

void BitcoinAddressGenerator::cleanup() {
    if (!is_initialized_) return;
    
    // Clean up GPU resources
    cleanup_gpu_buffers();
    
    if (computation_stream_) {
        cudaStreamDestroy(computation_stream_);
        computation_stream_ = nullptr;
    }
    
    if (memory_stream_) {
        cudaStreamDestroy(memory_stream_);
        memory_stream_ = nullptr;
    }
    
    // Clean up Bloom filter
    if (bloom_filter_) {
        bloom_filter_.reset();
    }
    
    is_initialized_ = false;
}

bool BitcoinAddressGenerator::generate_addresses_from_public_keys(
    const std::vector<ecc::Point>& public_keys,
    AddressGenerationContext& context,
    bool compressed) {
    
    if (!is_initialized_) {
        std::cerr << "ERROR: Address generator not initialized" << std::endl;
        return false;
    }
    
    if (public_keys.empty()) {
        return true;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        context.key_count = public_keys.size();
        context.compression_flags.assign(context.key_count, compressed);
        
        // Convert public keys to raw format
        size_t key_size = compressed ? 33 : 65;
        context.public_keys.resize(context.key_count * key_size);
        
        for (size_t i = 0; i < context.key_count; i++) {
            uint8_t* key_data = &context.public_keys[i * key_size];
            
            if (compressed) {
                // Compressed format: 0x02/0x03 + x coordinate
                key_data[0] = (public_keys[i].y.d[0] & 1) ? 0x03 : 0x02;
                memcpy(key_data + 1, public_keys[i].x.d, 32);
            } else {
                // Uncompressed format: 0x04 + x + y coordinates
                key_data[0] = 0x04;
                memcpy(key_data + 1, public_keys[i].x.d, 32);
                memcpy(key_data + 33, public_keys[i].y.d, 32);
            }
        }
        
        // Generate addresses in batches
        bool success = generate_addresses_batch(
            context.public_keys.data(),
            context.key_count,
            compressed,
            context
        );
        
        auto end_time = std::chrono::high_resolution_clock::now();
        context.generation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (context.generation_time.count() > 0) {
            context.addresses_per_second = (context.total_addresses_generated * 1000.0) / context.generation_time.count();
        }
        
        // Update performance metrics
        update_performance_metrics();
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in generate_addresses_from_public_keys: " << e.what() << std::endl;
        return false;
    }
}

bool BitcoinAddressGenerator::generate_addresses_batch(
    const uint8_t* public_keys_data,
    size_t key_count,
    bool compressed,
    AddressGenerationContext& context) {
    
    try {
        context.hash160_results.resize(key_count * 20);
        context.p2pkh_addresses.clear();
        context.p2pkh_addresses.reserve(key_count);
        
        if (comparison_config_.target_bech32.size() > 0) {
            context.bech32_addresses.clear();
            context.bech32_addresses.reserve(key_count);
        }
        
        // Process in batches to manage GPU memory
        size_t batch_size = hash_config_.batch_size;
        size_t processed = 0;
        
        while (processed < key_count) {
            size_t current_batch = std::min(batch_size, key_count - processed);
            size_t key_size = compressed ? 33 : 65;
            
            // Compute Hash160 for this batch
            bool hash_success = compute_hash160_batch(
                public_keys_data + (processed * key_size),
                current_batch,
                compressed,
                context.hash160_results.data() + (processed * 20)
            );
            
            if (!hash_success) {
                std::cerr << "ERROR: Hash160 computation failed for batch starting at " << processed << std::endl;
                return false;
            }
            
            // Generate addresses from Hash160 values
            for (size_t i = 0; i < current_batch; i++) {
                const uint8_t* hash160 = context.hash160_results.data() + ((processed + i) * 20);
                
                // Generate P2PKH address
                std::string p2pkh_addr = hash160_to_p2pkh_address(hash160);
                context.p2pkh_addresses.push_back(p2pkh_addr);
                
                // Generate Bech32 address if needed
                if (!comparison_config_.target_bech32.empty()) {
                    std::string bech32_addr = hash160_to_bech32_address(hash160, 0);
                    context.bech32_addresses.push_back(bech32_addr);
                }
                
                context.total_addresses_generated++;
            }
            
            processed += current_batch;
            
            // Update metrics periodically
            {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                current_metrics_.total_public_keys_processed = processed;
                current_metrics_.total_addresses_generated = context.total_addresses_generated;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in generate_addresses_batch: " << e.what() << std::endl;
        return false;
    }
}

std::string BitcoinAddressGenerator::generate_p2pkh_address(const ecc::Point& public_key, bool compressed) {
    try {
        // Serialize public key
        uint8_t key_data[65];
        size_t key_size;
        
        if (compressed) {
            key_size = 33;
            key_data[0] = (public_key.y.d[0] & 1) ? 0x03 : 0x02;
            memcpy(key_data + 1, public_key.x.d, 32);
        } else {
            key_size = 65;
            key_data[0] = 0x04;
            memcpy(key_data + 1, public_key.x.d, 32);
            memcpy(key_data + 33, public_key.y.d, 32);
        }
        
        // Compute Hash160 (SHA256 followed by RIPEMD160)
        uint8_t hash160[20];
        
        // For individual operations, use CPU implementation
        // In production, this would use optimized hash functions
        
        // Placeholder: Simple hash computation
        // This should be replaced with proper SHA256 and RIPEMD160
        memset(hash160, 0, 20);
        for (size_t i = 0; i < key_size; i++) {
            hash160[i % 20] ^= key_data[i];
        }
        
        return hash160_to_p2pkh_address(hash160);
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in generate_p2pkh_address: " << e.what() << std::endl;
        return "";
    }
}

bool BitcoinAddressGenerator::compute_hash160_batch(
    const uint8_t* public_keys_data,
    size_t key_count,
    bool compressed,
    uint8_t* hash160_results) {
    
    try {
        if (!is_initialized_) {
            return false;
        }
        
        // Use GPU implementation if available
        if (hash_config_.enable_fused_hash_operations) {
            return launch_hash160_kernel(
                public_keys_data, key_count, compressed,
                hash160_results, computation_stream_
            );
        }
        
        // Fallback: Two-stage hash computation
        size_t key_size = compressed ? 33 : 65;
        std::vector<uint8_t> sha256_results(key_count * 32);
        
        // Compute SHA256
        bool sha_success = compute_sha256_batch(
            public_keys_data, key_count, key_size,
            sha256_results.data()
        );
        
        if (!sha_success) {
            return false;
        }
        
        // Compute RIPEMD160
        bool ripemd_success = compute_ripemd160_batch(
            sha256_results.data(), key_count,
            hash160_results
        );
        
        return ripemd_success;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in compute_hash160_batch: " << e.what() << std::endl;
        return false;
    }
}

bool BitcoinAddressGenerator::set_target_addresses(const std::vector<std::string>& addresses) {
    try {
        target_addresses_.clear();
        target_hash160_values_.clear();
        
        if (bloom_filter_) {
            bloom_filter_->clear();
        }
        
        for (const auto& address : addresses) {
            if (!add_target_address(address)) {
                std::cerr << "WARNING: Failed to add target address: " << address << std::endl;
            }
        }
        
        setup_target_address_data();
        
        std::cout << "Set " << target_addresses_.size() << " target addresses" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in set_target_addresses: " << e.what() << std::endl;
        return false;
    }
}

bool BitcoinAddressGenerator::add_target_address(const std::string& address) {
    try {
        if (!validate_address(address)) {
            return false;
        }
        
        AddressFormat format = detect_address_format(address);
        if (format == AddressFormat::UNKNOWN) {
            return false;
        }
        
        target_addresses_[address] = format;
        
        // Add to appropriate target set
        switch (format) {
            case AddressFormat::P2PKH:
                comparison_config_.target_p2pkh.insert(address);
                break;
            case AddressFormat::P2SH:
                comparison_config_.target_p2sh.insert(address);
                break;
            case AddressFormat::P2WPKH_V0:
                comparison_config_.target_bech32.insert(address);
                break;
            default:
                break;
        }
        
        // Extract Hash160 and add to target data
        uint8_t hash160[20];
        if (extract_hash160_from_address(address, hash160)) {
            target_hash160_values_.insert(target_hash160_values_.end(), hash160, hash160 + 20);
        }
        
        // Add to Bloom filter
        if (bloom_filter_) {
            bloom_filter_->add(address);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in add_target_address: " << e.what() << std::endl;
        return false;
    }
}

std::vector<AddressMatch> BitcoinAddressGenerator::compare_addresses(
    const AddressGenerationContext& context,
    const std::vector<ecc::BigInt256>& private_keys) {
    
    std::vector<AddressMatch> matches;
    
    try {
        if (target_addresses_.empty()) {
            return matches;
        }
        
        // GPU-accelerated comparison if enabled
        if (comparison_config_.enable_gpu_comparison && !context.hash160_results.empty()) {
            const ecc::BigInt256* priv_keys_ptr = private_keys.empty() ? nullptr : private_keys.data();
            
            bool gpu_success = compare_addresses_gpu(
                context.hash160_results.data(),
                context.key_count,
                priv_keys_ptr,
                matches
            );
            
            if (gpu_success) {
                update_performance_metrics();
                return matches;
            }
        }
        
        // Fallback: CPU comparison
        for (size_t i = 0; i < context.p2pkh_addresses.size(); i++) {
            const std::string& address = context.p2pkh_addresses[i];
            
            // Quick Bloom filter check
            if (bloom_filter_ && !bloom_filter_->might_contain(address)) {
                continue;
            }
            
            // Exact match check
            auto it = target_addresses_.find(address);
            if (it != target_addresses_.end()) {
                AddressMatch match;
                match.address = address;
                match.format = it->second;
                match.batch_id = i;
                match.device_id = device_id_;
                
                if (i < private_keys.size()) {
                    match.private_key = private_keys[i];
                }
                
                // Extract Hash160 from context
                if (i * 20 < context.hash160_results.size()) {
                    memcpy(match.hash160, &context.hash160_results[i * 20], 20);
                }
                
                matches.push_back(match);
                
                // Call match callback if set
                if (match_callback_) {
                    match_callback_(match);
                }
            }
        }
        
        // Check Bech32 addresses if available
        for (size_t i = 0; i < context.bech32_addresses.size(); i++) {
            const std::string& address = context.bech32_addresses[i];
            
            if (bloom_filter_ && !bloom_filter_->might_contain(address)) {
                continue;
            }
            
            auto it = target_addresses_.find(address);
            if (it != target_addresses_.end()) {
                AddressMatch match;
                match.address = address;
                match.format = it->second;
                match.batch_id = i;
                match.device_id = device_id_;
                
                if (i < private_keys.size()) {
                    match.private_key = private_keys[i];
                }
                
                if (i * 20 < context.hash160_results.size()) {
                    memcpy(match.hash160, &context.hash160_results[i * 20], 20);
                }
                
                matches.push_back(match);
                
                if (match_callback_) {
                    match_callback_(match);
                }
            }
        }
        
        // Update metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            current_metrics_.total_comparisons_performed += context.key_count;
            current_metrics_.total_matches_found += matches.size();
            
            for (const auto& match : matches) {
                switch (match.format) {
                    case AddressFormat::P2PKH:
                        current_metrics_.p2pkh_matches++;
                        break;
                    case AddressFormat::P2SH:
                        current_metrics_.p2sh_matches++;
                        break;
                    case AddressFormat::P2WPKH_V0:
                        current_metrics_.bech32_matches++;
                        break;
                    default:
                        break;
                }
            }
        }
        
        return matches;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in compare_addresses: " << e.what() << std::endl;
        return matches;
    }
}

AddressFormat BitcoinAddressGenerator::detect_address_format(const std::string& address) {
    if (address.empty()) {
        return AddressFormat::UNKNOWN;
    }
    
    if (address.length() >= 26 && address.length() <= 35) {
        if (address[0] == '1') {
            return AddressFormat::P2PKH;
        } else if (address[0] == '3') {
            return AddressFormat::P2SH;
        }
    }
    
    if (address.length() >= 14 && (address.substr(0, 4) == "bc1q" || address.substr(0, 4) == "tb1q")) {
        return AddressFormat::P2WPKH_V0;
    }
    
    if (address.length() >= 14 && (address.substr(0, 4) == "bc1p" || address.substr(0, 4) == "tb1p")) {
        return AddressFormat::P2TR;
    }
    
    return AddressFormat::UNKNOWN;
}

bool BitcoinAddressGenerator::validate_address(const std::string& address) {
    AddressFormat format = detect_address_format(address);
    
    switch (format) {
        case AddressFormat::P2PKH:
        case AddressFormat::P2SH:
            return is_valid_p2pkh_address(address) || is_valid_p2sh_address(address);
        case AddressFormat::P2WPKH_V0:
        case AddressFormat::P2WSH_V0:
        case AddressFormat::P2TR:
            return is_valid_bech32_address(address);
        default:
            return false;
    }
}

std::string BitcoinAddressGenerator::hash160_to_p2pkh_address(const uint8_t hash160[20]) {
    uint8_t payload[21];
    payload[0] = 0x00; // P2PKH version byte
    memcpy(payload + 1, hash160, 20);
    
    return encode_base58_check(payload, 21);
}

std::string BitcoinAddressGenerator::hash160_to_bech32_address(const uint8_t hash160[20], int witness_version) {
    std::vector<uint8_t> data(hash160, hash160 + 20);
    std::vector<uint8_t> converted = convert_bits(data, 8, 5, true);
    
    std::vector<uint8_t> spec;
    spec.push_back(witness_version);
    spec.insert(spec.end(), converted.begin(), converted.end());
    
    return encode_bech32("bc", spec);
}

// Placeholder implementations for hash operations
// These would be implemented in the corresponding .cu file with proper CUDA kernels

bool BitcoinAddressGenerator::launch_sha256_kernel(
    const uint8_t* input, size_t count, size_t input_size,
    uint8_t* output, cudaStream_t stream) {
    
    // Placeholder: This would launch actual SHA256 CUDA kernel
    std::cerr << "WARNING: SHA256 kernel not implemented - using placeholder" << std::endl;
    
    // Simple placeholder hash
    for (size_t i = 0; i < count; i++) {
        const uint8_t* in_data = input + (i * input_size);
        uint8_t* out_data = output + (i * 32);
        
        // Extremely simple hash placeholder
        uint32_t hash = 0;
        for (size_t j = 0; j < input_size; j++) {
            hash = hash * 31 + in_data[j];
        }
        
        memset(out_data, 0, 32);
        memcpy(out_data, &hash, sizeof(hash));
    }
    
    return true;
}

bool BitcoinAddressGenerator::launch_ripemd160_kernel(
    const uint8_t* input, size_t count,
    uint8_t* output, cudaStream_t stream) {
    
    // Placeholder: This would launch actual RIPEMD160 CUDA kernel
    std::cerr << "WARNING: RIPEMD160 kernel not implemented - using placeholder" << std::endl;
    
    // Simple placeholder
    for (size_t i = 0; i < count; i++) {
        const uint8_t* in_data = input + (i * 32);
        uint8_t* out_data = output + (i * 20);
        
        // Simple reduction to 20 bytes
        for (int j = 0; j < 20; j++) {
            out_data[j] = in_data[j] ^ in_data[j + 12];
        }
    }
    
    return true;
}

bool BitcoinAddressGenerator::launch_hash160_kernel(
    const uint8_t* public_keys, size_t count, bool compressed,
    uint8_t* hash160_output, cudaStream_t stream) {
    
    // Placeholder: This would launch fused Hash160 CUDA kernel
    std::cerr << "WARNING: Hash160 kernel not implemented - using placeholder" << std::endl;
    
    size_t key_size = compressed ? 33 : 65;
    
    for (size_t i = 0; i < count; i++) {
        const uint8_t* key_data = public_keys + (i * key_size);
        uint8_t* hash_out = hash160_output + (i * 20);
        
        // Extremely simple hash placeholder
        for (int j = 0; j < 20; j++) {
            hash_out[j] = 0;
            for (size_t k = 0; k < key_size; k++) {
                hash_out[j] ^= key_data[k];
            }
            hash_out[j] ^= (j + i) & 0xFF;
        }
    }
    
    return true;
}

// Additional utility method implementations

void BitcoinAddressGenerator::init_base58_map() {
    memset(base58_map, -1, sizeof(base58_map));
    for (int i = 0; i < 58; i++) {
        base58_map[static_cast<uint8_t>(base58_alphabet[i])] = i;
    }
}

AddressGenerationMetrics BitcoinAddressGenerator::get_current_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_metrics_;
}

void BitcoinAddressGenerator::reset_performance_counters() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    current_metrics_ = AddressGenerationMetrics();
    last_metrics_update_ = std::chrono::high_resolution_clock::now();
}

bool BitcoinAddressGenerator::allocate_gpu_memory(size_t max_batch_size) {
    try {
        deallocate_gpu_memory(); // Clean up any existing allocation
        
        size_t key_size = 65; // Max size for uncompressed keys
        size_t total_memory_needed = 
            max_batch_size * key_size +     // public keys
            max_batch_size * 32 +           // SHA256 results
            max_batch_size * 20 +           // RIPEMD160 results
            max_batch_size * 20 +           // Hash160 results
            target_hash160_values_.size() + // target hash160 values
            max_batch_size;                 // match results (bool array)
        
        std::cout << "Allocating GPU memory: " << (total_memory_needed / 1024 / 1024) << " MB" << std::endl;
        
        // Allocate GPU buffers
        cudaError_t err;
        
        err = cudaMalloc(&gpu_buffers_.public_keys_buffer, max_batch_size * key_size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate public keys buffer: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&gpu_buffers_.sha256_buffer, max_batch_size * 32);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate SHA256 buffer: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&gpu_buffers_.ripemd160_buffer, max_batch_size * 20);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate RIPEMD160 buffer: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&gpu_buffers_.hash160_buffer, max_batch_size * 20);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate Hash160 buffer: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&gpu_buffers_.target_hash160_buffer, target_hash160_values_.size());
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate target Hash160 buffer: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&gpu_buffers_.match_results_buffer, max_batch_size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate match results buffer: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        gpu_buffers_.allocated_size = max_batch_size;
        
        // Upload target Hash160 values
        if (!target_hash160_values_.empty()) {
            err = cudaMemcpy(gpu_buffers_.target_hash160_buffer, 
                           target_hash160_values_.data(), 
                           target_hash160_values_.size(),
                           cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "Failed to upload target Hash160 values: " << cudaGetErrorString(err) << std::endl;
                return false;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in allocate_gpu_memory: " << e.what() << std::endl;
        return false;
    }
}

bool BitcoinAddressGenerator::cleanup_gpu_buffers() {
    bool success = true;
    
    if (gpu_buffers_.public_keys_buffer) {
        cudaFree(gpu_buffers_.public_keys_buffer);
        gpu_buffers_.public_keys_buffer = nullptr;
    }
    
    if (gpu_buffers_.sha256_buffer) {
        cudaFree(gpu_buffers_.sha256_buffer);
        gpu_buffers_.sha256_buffer = nullptr;
    }
    
    if (gpu_buffers_.ripemd160_buffer) {
        cudaFree(gpu_buffers_.ripemd160_buffer);
        gpu_buffers_.ripemd160_buffer = nullptr;
    }
    
    if (gpu_buffers_.hash160_buffer) {
        cudaFree(gpu_buffers_.hash160_buffer);
        gpu_buffers_.hash160_buffer = nullptr;
    }
    
    if (gpu_buffers_.target_hash160_buffer) {
        cudaFree(gpu_buffers_.target_hash160_buffer);
        gpu_buffers_.target_hash160_buffer = nullptr;
    }
    
    if (gpu_buffers_.match_results_buffer) {
        cudaFree(gpu_buffers_.match_results_buffer);
        gpu_buffers_.match_results_buffer = nullptr;
    }
    
    gpu_buffers_.allocated_size = 0;
    return success;
}

void BitcoinAddressGenerator::update_performance_metrics() {
    auto now = std::chrono::high_resolution_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_metrics_update_);
    
    if (time_diff.count() >= 1000) { // Update every second
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        // Calculate rates
        double time_sec = time_diff.count() / 1000.0;
        if (time_sec > 0) {
            current_metrics_.average_generation_rate = 
                current_metrics_.total_public_keys_processed / time_sec;
            current_metrics_.comparison_rate = 
                current_metrics_.total_comparisons_performed / time_sec;
        }
        
        last_metrics_update_ = now;
    }
}

// Placeholder implementations for Base58 and Bech32 encoding
// These would need proper implementation for production use

std::string BitcoinAddressGenerator::encode_base58_check(const uint8_t* data, size_t length) {
    // Placeholder implementation
    std::ostringstream oss;
    oss << "1";
    for (size_t i = 0; i < std::min(length, size_t(25)); i++) {
        oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(data[i]);
    }
    return oss.str();
}

std::string BitcoinAddressGenerator::encode_bech32(const std::string& hrp, const std::vector<uint8_t>& data) {
    // Placeholder implementation
    std::ostringstream oss;
    oss << hrp << "1q";
    for (size_t i = 0; i < std::min(data.size(), size_t(20)); i++) {
        oss << std::hex << static_cast<int>(data[i]);
    }
    return oss.str();
}

std::vector<uint8_t> BitcoinAddressGenerator::convert_bits(
    const std::vector<uint8_t>& data, int from_bits, int to_bits, bool pad) {
    // Placeholder implementation
    return data; // Would need proper bit conversion
}

// Placeholder validation methods
bool BitcoinAddressGenerator::is_valid_p2pkh_address(const std::string& address) {
    return address.length() >= 26 && address.length() <= 35 && address[0] == '1';
}

bool BitcoinAddressGenerator::is_valid_p2sh_address(const std::string& address) {
    return address.length() >= 26 && address.length() <= 35 && address[0] == '3';
}

bool BitcoinAddressGenerator::is_valid_bech32_address(const std::string& address) {
    return address.length() >= 14 && (address.substr(0, 4) == "bc1q" || address.substr(0, 4) == "bc1p");
}

bool BitcoinAddressGenerator::extract_hash160_from_address(const std::string& address, uint8_t hash160[20]) {
    // Placeholder implementation
    AddressFormat format = detect_address_format(address);
    if (format == AddressFormat::UNKNOWN) {
        return false;
    }
    
    // Simple hash based on address string for placeholder
    uint32_t hash = 0;
    for (char c : address) {
        hash = hash * 31 + static_cast<uint32_t>(c);
    }
    
    memset(hash160, 0, 20);
    memcpy(hash160, &hash, sizeof(hash));
    
    return true;
}

} // namespace compare
} // namespace keyhunt