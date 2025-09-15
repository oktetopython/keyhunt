/**
 * @file single_compare.cpp
 * @brief CPU interface for GPU-optimized single target address comparison
 * @author KeyhuntCUDA Team
 * 
 * T049: Build single target address comparison in src/KeyhuntCore/compare/single_compare.cu
 * 
 * Provides the CPU-side implementation and interface for GPU-accelerated
 * single target Bitcoin address comparison. Handles memory management,
 * kernel launching, and performance monitoring.
 */

#include "single_compare.h"
#include "../utils/logger.h"
#include "../utils/timer.h"
#include <cuda_runtime.h>
#include <cassert>
#include <algorithm>
#include <iomanip>
#include <sstream>

// External CUDA kernel declarations
extern "C" {
    // Set target hash in constant memory
    cudaError_t cuda_set_target_hash160(const uint32_t* target_hash);
    
    // Launch comparison kernels
    cudaError_t cuda_launch_single_target_compare_kernel(
        const uint32_t* hash160_input,
        uint32_t* match_results,
        size_t input_count,
        dim3 grid_size,
        dim3 block_size,
        cudaStream_t stream
    );
    
    cudaError_t cuda_launch_single_target_compare_early_exit_kernel(
        const uint32_t* hash160_input,
        uint32_t* match_found,
        uint32_t* match_index,
        size_t input_count,
        dim3 grid_size,
        dim3 block_size,
        cudaStream_t stream
    );
    
    cudaError_t cuda_launch_single_target_compare_streaming_kernel(
        const uint32_t* hash160_input,
        uint32_t* match_results,
        const uint64_t* private_key_indices,
        size_t input_count,
        uint64_t batch_offset,
        dim3 grid_size,
        dim3 block_size,
        cudaStream_t stream
    );
}

namespace keyhunt {
namespace compare {

SingleTargetCompare::SingleTargetCompare()
    : device_id_(-1)
    , is_initialized_(false)
    , target_set_(false)
    , computation_stream_(nullptr)
    , memory_stream_(nullptr)
    , streams_created_(false)
    , streaming_active_(false)
    , streaming_total_count_(0)
    , streaming_processed_count_(0) {
    
    memset(target_hash160_, 0, 20);
    target_address_.clear();
}

SingleTargetCompare::~SingleTargetCompare() {
    cleanup();
}

bool SingleTargetCompare::initialize(int device_id) {
    if (is_initialized_) {
        cleanup();
    }
    
    device_id_ = device_id;
    
    // Set CUDA device
    cudaError_t cuda_error = cudaSetDevice(device_id_);
    if (cuda_error != cudaSuccess) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to set CUDA device " + std::to_string(device_id_) + 
            ": " + cudaGetErrorString(cuda_error));
        return false;
    }
    
    // Create CUDA streams for async operations
    cuda_error = cudaStreamCreate(&computation_stream_);
    if (cuda_error != cudaSuccess) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to create computation stream: " + std::string(cudaGetErrorString(cuda_error)));
        return false;
    }
    
    cuda_error = cudaStreamCreate(&memory_stream_);
    if (cuda_error != cudaSuccess) {
        cudaStreamDestroy(computation_stream_);
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to create memory stream: " + std::string(cudaGetErrorString(cuda_error)));
        return false;
    }
    
    streams_created_ = true;
    is_initialized_ = true;
    
    utils::Logger::log(utils::LogLevel::Info, 
        "SingleTargetCompare initialized on device " + std::to_string(device_id_));
    
    return true;
}

bool SingleTargetCompare::configure(const SingleTargetCompareConfig& config) {
    if (!is_initialized_) {
        utils::Logger::log(utils::LogLevel::Error, 
            "SingleTargetCompare not initialized - call initialize() first");
        return false;
    }
    
    config_ = config;
    
    // Allocate GPU memory based on configuration
    if (!allocate_gpu_memory(config_.max_batch_size)) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to allocate GPU memory for batch size " + std::to_string(config_.max_batch_size));
        return false;
    }
    
    utils::Logger::log(utils::LogLevel::Info, 
        "SingleTargetCompare configured with batch size " + std::to_string(config_.max_batch_size));
    
    return true;
}

void SingleTargetCompare::cleanup() {
    if (streams_created_) {
        cudaStreamDestroy(computation_stream_);
        cudaStreamDestroy(memory_stream_);
        streams_created_ = false;
    }
    
    deallocate_gpu_memory();
    
    target_set_ = false;
    is_initialized_ = false;
    streaming_active_ = false;
    
    memset(target_hash160_, 0, 20);
    target_address_.clear();
}

bool SingleTargetCompare::set_target_hash160(const uint8_t hash160[20]) {
    if (!is_initialized_) {
        utils::Logger::log(utils::LogLevel::Error, 
            "SingleTargetCompare not initialized");
        return false;
    }
    
    // Copy target hash160
    memcpy(target_hash160_, hash160, 20);
    
    // Upload to GPU constant memory
    if (!upload_target_to_gpu()) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to upload target hash160 to GPU");
        return false;
    }
    
    target_set_ = true;
    target_address_ = hash160_to_address(hash160, true);  // Convert to P2PKH address
    
    utils::Logger::log(utils::LogLevel::Info, 
        "Target hash160 set: " + hash160_to_hex_string(hash160));
    
    return true;
}

bool SingleTargetCompare::set_target_address(const std::string& bitcoin_address) {
    if (!is_valid_bitcoin_address(bitcoin_address)) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Invalid Bitcoin address format: " + bitcoin_address);
        return false;
    }
    
    uint8_t hash160[20];
    if (!address_to_hash160(bitcoin_address, hash160)) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to extract hash160 from address: " + bitcoin_address);
        return false;
    }
    
    target_address_ = bitcoin_address;
    return set_target_hash160(hash160);
}

bool SingleTargetCompare::clear_target() {
    target_set_ = false;
    memset(target_hash160_, 0, 20);
    target_address_.clear();
    
    utils::Logger::log(utils::LogLevel::Info, "Target cleared");
    return true;
}

void SingleTargetCompare::get_target_hash160(uint8_t hash160[20]) const {
    memcpy(hash160, target_hash160_, 20);
}

SingleTargetCompareResult SingleTargetCompare::compare_batch(
    const uint8_t* hash160_values,
    size_t hash_count,
    const uint64_t* private_key_indices) {
    
    SingleTargetCompareResult result;
    
    if (!target_set_) {
        utils::Logger::log(utils::LogLevel::Error, 
            "No target set - call set_target_hash160() or set_target_address() first");
        return result;
    }
    
    if (!validate_input_parameters(hash160_values, hash_count)) {
        return result;
    }
    
    start_performance_timing();
    
    // Resize GPU buffers if needed
    if (!resize_gpu_buffers_if_needed(hash_count)) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to resize GPU buffers for " + std::to_string(hash_count) + " hashes");
        return result;
    }
    
    // Copy input data to GPU
    size_t hash160_bytes = hash_count * 20;  // 20 bytes per hash160
    cudaError_t cuda_error = cudaMemcpy(gpu_buffers_.hash160_input, hash160_values, 
                                       hash160_bytes, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to copy hash160 data to GPU: " + std::string(cudaGetErrorString(cuda_error)));
        return result;
    }
    
    // Copy private key indices if provided
    if (private_key_indices && gpu_buffers_.private_key_indices) {
        size_t indices_bytes = hash_count * sizeof(uint64_t);
        cuda_error = cudaMemcpy(gpu_buffers_.private_key_indices, private_key_indices, 
                               indices_bytes, cudaMemcpyHostToDevice);
        if (cuda_error != cudaSuccess) {
            utils::Logger::log(utils::LogLevel::Error, 
                "Failed to copy private key indices to GPU: " + std::string(cudaGetErrorString(cuda_error)));
            return result;
        }
    }
    
    // Launch appropriate kernel based on configuration
    bool kernel_success = false;
    if (config_.enable_early_termination) {
        // Reset match flags
        uint32_t zero = 0;
        cudaMemcpy(gpu_buffers_.match_found_flag, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_buffers_.match_index, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        kernel_success = launch_early_exit_kernel(
            gpu_buffers_.hash160_input,
            gpu_buffers_.match_found_flag,
            gpu_buffers_.match_index,
            hash_count,
            computation_stream_
        );
    } else {
        kernel_success = launch_comparison_kernel(
            gpu_buffers_.hash160_input,
            gpu_buffers_.match_results,
            hash_count,
            computation_stream_
        );
    }
    
    if (!kernel_success) {
        utils::Logger::log(utils::LogLevel::Error, "Failed to launch comparison kernel");
        return result;
    }
    
    // Wait for kernel completion
    cuda_error = cudaStreamSynchronize(computation_stream_);
    if (cuda_error != cudaSuccess) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Kernel execution failed: " + std::string(cudaGetErrorString(cuda_error)));
        return result;
    }
    
    // Copy results back from GPU
    if (config_.enable_early_termination) {
        uint32_t match_found = 0;
        uint32_t match_index = 0;
        
        cudaMemcpy(&match_found, gpu_buffers_.match_found_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&match_index, gpu_buffers_.match_index, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        if (match_found) {
            result.target_found = true;
            result.match_index = match_index;
            result.device_id = device_id_;
            
            // Copy the matching hash160
            if (match_index < hash_count) {
                memcpy(result.matching_hash160, &hash160_values[match_index * 20], 20);
                
                if (private_key_indices) {
                    result.private_key_index = private_key_indices[match_index];
                }
            }
        }
    } else {
        // Copy all match results and check for matches
        std::vector<uint32_t> match_results(hash_count);
        size_t results_bytes = hash_count * sizeof(uint32_t);
        
        cuda_error = cudaMemcpy(match_results.data(), gpu_buffers_.match_results, 
                               results_bytes, cudaMemcpyDeviceToHost);
        if (cuda_error != cudaSuccess) {
            utils::Logger::log(utils::LogLevel::Error, 
                "Failed to copy match results from GPU: " + std::string(cudaGetErrorString(cuda_error)));
            return result;
        }
        
        // Find first match
        for (size_t i = 0; i < hash_count; ++i) {
            if (match_results[i] != 0) {
                result.target_found = true;
                result.match_index = i;
                result.device_id = device_id_;
                
                memcpy(result.matching_hash160, &hash160_values[i * 20], 20);
                
                if (private_key_indices) {
                    result.private_key_index = private_key_indices[i];
                }
                break;
            }
        }
    }
    
    end_performance_timing(hash_count);
    
    if (result.target_found) {
        utils::Logger::log(utils::LogLevel::Info, 
            "Target found at index " + std::to_string(result.match_index) + 
            " with hash160: " + hash160_to_hex_string(result.matching_hash160));
    }
    
    return result;
}

std::vector<SingleTargetCompareResult> SingleTargetCompare::compare_batch_all_matches(
    const uint8_t* hash160_values,
    size_t hash_count,
    const uint64_t* private_key_indices) {
    
    std::vector<SingleTargetCompareResult> results;
    
    if (!target_set_) {
        utils::Logger::log(utils::LogLevel::Error, "No target set");
        return results;
    }
    
    if (!validate_input_parameters(hash160_values, hash_count)) {
        return results;
    }
    
    start_performance_timing();
    
    // Use the basic comparison kernel to get all matches
    config_.enable_early_termination = false;
    
    if (!resize_gpu_buffers_if_needed(hash_count)) {
        return results;
    }
    
    // Copy input data to GPU
    size_t hash160_bytes = hash_count * 20;
    cudaMemcpy(gpu_buffers_.hash160_input, hash160_values, hash160_bytes, cudaMemcpyHostToDevice);
    
    if (private_key_indices && gpu_buffers_.private_key_indices) {
        size_t indices_bytes = hash_count * sizeof(uint64_t);
        cudaMemcpy(gpu_buffers_.private_key_indices, private_key_indices, 
                   indices_bytes, cudaMemcpyHostToDevice);
    }
    
    // Launch comparison kernel
    if (!launch_comparison_kernel(gpu_buffers_.hash160_input, gpu_buffers_.match_results, 
                                  hash_count, computation_stream_)) {
        return results;
    }
    
    cudaStreamSynchronize(computation_stream_);
    
    // Copy all results back
    std::vector<uint32_t> match_results(hash_count);
    size_t results_bytes = hash_count * sizeof(uint32_t);
    cudaMemcpy(match_results.data(), gpu_buffers_.match_results, 
               results_bytes, cudaMemcpyDeviceToHost);
    
    // Collect all matches
    for (size_t i = 0; i < hash_count; ++i) {
        if (match_results[i] != 0) {
            SingleTargetCompareResult result;
            result.target_found = true;
            result.match_index = i;
            result.device_id = device_id_;
            
            memcpy(result.matching_hash160, &hash160_values[i * 20], 20);
            
            if (private_key_indices) {
                result.private_key_index = private_key_indices[i];
            }
            
            results.push_back(result);
        }
    }
    
    end_performance_timing(hash_count);
    
    utils::Logger::log(utils::LogLevel::Info, 
        "Found " + std::to_string(results.size()) + " matches in batch of " + 
        std::to_string(hash_count) + " hashes");
    
    return results;
}

SingleTargetCompareMetrics SingleTargetCompare::get_performance_metrics() const {
    return metrics_;
}

void SingleTargetCompare::reset_performance_counters() {
    metrics_ = SingleTargetCompareMetrics();
}

bool SingleTargetCompare::allocate_gpu_memory(size_t max_batch_size) {
    deallocate_gpu_memory();  // Clean up any existing allocation
    
    size_t hash160_buffer_size = max_batch_size * 5 * sizeof(uint32_t);  // 20 bytes as 5 uint32_t
    size_t match_results_size = max_batch_size * sizeof(uint32_t);
    size_t indices_size = max_batch_size * sizeof(uint64_t);
    
    // Allocate hash160 input buffer
    cudaError_t cuda_error = cudaMalloc(&gpu_buffers_.hash160_input, hash160_buffer_size);
    if (cuda_error != cudaSuccess) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to allocate hash160 input buffer: " + std::string(cudaGetErrorString(cuda_error)));
        return false;
    }
    
    // Allocate match results buffer
    cuda_error = cudaMalloc(&gpu_buffers_.match_results, match_results_size);
    if (cuda_error != cudaSuccess) {
        cleanup_gpu_buffers();
        return false;
    }
    
    // Allocate match found flag and index
    cuda_error = cudaMalloc(&gpu_buffers_.match_found_flag, sizeof(uint32_t));
    if (cuda_error != cudaSuccess) {
        cleanup_gpu_buffers();
        return false;
    }
    
    cuda_error = cudaMalloc(&gpu_buffers_.match_index, sizeof(uint32_t));
    if (cuda_error != cudaSuccess) {
        cleanup_gpu_buffers();
        return false;
    }
    
    // Allocate private key indices buffer
    cuda_error = cudaMalloc(&gpu_buffers_.private_key_indices, indices_size);
    if (cuda_error != cudaSuccess) {
        cleanup_gpu_buffers();
        return false;
    }
    
    gpu_buffers_.allocated_hash_count = max_batch_size;
    gpu_buffers_.allocated_size_bytes = hash160_buffer_size + match_results_size + 
                                       2 * sizeof(uint32_t) + indices_size;
    
    utils::Logger::log(utils::LogLevel::Info, 
        "Allocated " + std::to_string(gpu_buffers_.allocated_size_bytes / (1024 * 1024)) + 
        " MB GPU memory for " + std::to_string(max_batch_size) + " hashes");
    
    return true;
}

void SingleTargetCompare::deallocate_gpu_memory() {
    cleanup_gpu_buffers();
}

void SingleTargetCompare::cleanup_gpu_buffers() {
    if (gpu_buffers_.hash160_input) {
        cudaFree(gpu_buffers_.hash160_input);
        gpu_buffers_.hash160_input = nullptr;
    }
    
    if (gpu_buffers_.match_results) {
        cudaFree(gpu_buffers_.match_results);
        gpu_buffers_.match_results = nullptr;
    }
    
    if (gpu_buffers_.match_found_flag) {
        cudaFree(gpu_buffers_.match_found_flag);
        gpu_buffers_.match_found_flag = nullptr;
    }
    
    if (gpu_buffers_.match_index) {
        cudaFree(gpu_buffers_.match_index);
        gpu_buffers_.match_index = nullptr;
    }
    
    if (gpu_buffers_.private_key_indices) {
        cudaFree(gpu_buffers_.private_key_indices);
        gpu_buffers_.private_key_indices = nullptr;
    }
    
    gpu_buffers_.allocated_hash_count = 0;
    gpu_buffers_.allocated_size_bytes = 0;
}

bool SingleTargetCompare::upload_target_to_gpu() {
    // Convert hash160 to uint32_t array for constant memory
    uint32_t target_words[5];
    for (int i = 0; i < 5; ++i) {
        target_words[i] = *reinterpret_cast<const uint32_t*>(&target_hash160_[i * 4]);
    }
    
    // Upload to GPU constant memory
    cudaError_t cuda_error = cuda_set_target_hash160(target_words);
    if (cuda_error != cudaSuccess) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to set target in constant memory: " + std::string(cudaGetErrorString(cuda_error)));
        return false;
    }
    
    return true;
}

bool SingleTargetCompare::launch_comparison_kernel(
    const uint32_t* input_hash160,
    uint32_t* match_results,
    size_t hash_count,
    cudaStream_t stream) {
    
    dim3 grid_size = calculate_grid_dimensions(hash_count);
    dim3 block_size = calculate_block_dimensions();
    
    cudaError_t cuda_error = cuda_launch_single_target_compare_kernel(
        input_hash160, match_results, hash_count, grid_size, block_size, stream);
    
    if (cuda_error != cudaSuccess) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to launch comparison kernel: " + std::string(cudaGetErrorString(cuda_error)));
        return false;
    }
    
    return true;
}

bool SingleTargetCompare::launch_early_exit_kernel(
    const uint32_t* input_hash160,
    uint32_t* match_found,
    uint32_t* match_index,
    size_t hash_count,
    cudaStream_t stream) {
    
    dim3 grid_size = calculate_grid_dimensions(hash_count);
    dim3 block_size = calculate_block_dimensions();
    
    cudaError_t cuda_error = cuda_launch_single_target_compare_early_exit_kernel(
        input_hash160, match_found, match_index, hash_count, grid_size, block_size, stream);
    
    if (cuda_error != cudaSuccess) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Failed to launch early exit kernel: " + std::string(cudaGetErrorString(cuda_error)));
        return false;
    }
    
    return true;
}

dim3 SingleTargetCompare::calculate_grid_dimensions(size_t hash_count) const {
    size_t blocks_needed = (hash_count + config_.threads_per_block - 1) / config_.threads_per_block;
    size_t blocks_per_grid = std::min(blocks_needed, config_.blocks_per_grid);
    return dim3(static_cast<unsigned int>(blocks_per_grid));
}

dim3 SingleTargetCompare::calculate_block_dimensions() const {
    return dim3(static_cast<unsigned int>(config_.threads_per_block));
}

bool SingleTargetCompare::resize_gpu_buffers_if_needed(size_t required_hash_count) {
    if (required_hash_count <= gpu_buffers_.allocated_hash_count) {
        return true;  // Current allocation is sufficient
    }
    
    // Need to reallocate with larger size
    size_t new_size = std::max(required_hash_count, gpu_buffers_.allocated_hash_count * 2);
    
    utils::Logger::log(utils::LogLevel::Info, 
        "Resizing GPU buffers from " + std::to_string(gpu_buffers_.allocated_hash_count) + 
        " to " + std::to_string(new_size) + " hashes");
    
    return allocate_gpu_memory(new_size);
}

void SingleTargetCompare::start_performance_timing() {
    last_operation_start_ = std::chrono::high_resolution_clock::now();
}

void SingleTargetCompare::end_performance_timing(size_t operations_performed) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - last_operation_start_);
    
    metrics_.total_operation_time = duration;
    metrics_.total_comparisons_performed += operations_performed;
    
    if (duration.count() > 0) {
        metrics_.comparisons_per_second = static_cast<double>(operations_performed) * 1000000.0 / 
                                         static_cast<double>(duration.count());
    }
    
    update_memory_usage_metrics();
}

void SingleTargetCompare::update_memory_usage_metrics() {
    metrics_.gpu_memory_used_bytes = gpu_buffers_.allocated_size_bytes;
    
    // Estimate memory bandwidth
    size_t bytes_transferred = metrics_.total_comparisons_performed * 20;  // 20 bytes per hash160
    double time_seconds = static_cast<double>(metrics_.total_operation_time.count()) / 1000000.0;
    
    if (time_seconds > 0.0) {
        metrics_.memory_bandwidth_gb_s = static_cast<double>(bytes_transferred) / (time_seconds * 1e9);
    }
}

bool SingleTargetCompare::validate_input_parameters(const uint8_t* hash160_values, size_t hash_count) const {
    if (!hash160_values) {
        utils::Logger::log(utils::LogLevel::Error, "Invalid hash160_values pointer");
        return false;
    }
    
    if (hash_count == 0) {
        utils::Logger::log(utils::LogLevel::Error, "Hash count cannot be zero");
        return false;
    }
    
    if (hash_count > config_.max_batch_size) {
        utils::Logger::log(utils::LogLevel::Error, 
            "Hash count " + std::to_string(hash_count) + 
            " exceeds maximum batch size " + std::to_string(config_.max_batch_size));
        return false;
    }
    
    return true;
}

std::string SingleTargetCompare::hash160_to_hex_string(const uint8_t hash160[20]) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 20; ++i) {
        oss << std::setw(2) << static_cast<unsigned int>(hash160[i]);
    }
    return oss.str();
}

bool SingleTargetCompare::hex_string_to_hash160(const std::string& hex, uint8_t hash160[20]) {
    if (hex.length() != 40) {
        return false;  // Hash160 should be exactly 40 hex characters
    }
    
    for (int i = 0; i < 20; ++i) {
        std::string byte_hex = hex.substr(i * 2, 2);
        char* end_ptr;
        unsigned long byte_val = std::strtoul(byte_hex.c_str(), &end_ptr, 16);
        
        if (end_ptr != byte_hex.c_str() + 2 || byte_val > 255) {
            return false;
        }
        
        hash160[i] = static_cast<uint8_t>(byte_val);
    }
    
    return true;
}

bool SingleTargetCompare::address_to_hash160(const std::string& address, uint8_t hash160[20]) {
    // This is a simplified implementation - in practice would need full Base58 decoding
    // For now, return false to indicate this needs to be implemented
    utils::Logger::log(utils::LogLevel::Warning, 
        "address_to_hash160 not fully implemented - use set_target_hash160 directly");
    return false;
}

std::string SingleTargetCompare::hash160_to_address(const uint8_t hash160[20], bool p2pkh) {
    // This is a simplified implementation - in practice would need full Base58 encoding
    // For now, return hex representation
    return hash160_to_hex_string(hash160);
}

bool SingleTargetCompare::is_valid_bitcoin_address(const std::string& address) {
    // Simplified validation - check basic format
    if (address.empty() || address.length() < 26 || address.length() > 35) {
        return false;
    }
    
    // Check for valid prefixes
    return (address[0] == '1' || address[0] == '3' || 
            (address.length() > 3 && address.substr(0, 4) == "bc1q"));
}

size_t SingleTargetCompare::get_gpu_memory_usage() const {
    return gpu_buffers_.allocated_size_bytes;
}

size_t SingleTargetCompare::get_cpu_memory_usage() const {
    return sizeof(*this) + target_address_.capacity();
}

} // namespace compare
} // namespace keyhunt