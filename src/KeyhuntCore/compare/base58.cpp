/**
 * @file base58.cpp
 * @brief C++ implementation for Base58Encoder class
 * @author KeyhuntCUDA Team
 * 
 * T051: Implement Base58 encoding wrapper for CUDA kernels
 * 
 * Provides C++ wrapper implementation for Base58Encoder class that interfaces
 * with the CUDA kernels in base58.cu for high-performance Base58 operations.
 */

#include "base58.h"
#include <cuda_runtime.h>
#include <memory>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace keyhunt {
namespace compare {

// External CUDA kernel declarations
extern "C" {
    void launch_base58_encode_batch(
        const uint8_t* input_data,
        const uint32_t* input_lengths,
        char* output_data,
        uint32_t* output_lengths,
        uint32_t max_output_len,
        uint32_t count,
        cudaStream_t stream
    );
    
    void launch_base58check_encode_batch(
        const uint8_t* input_data,
        const uint32_t* input_lengths,
        char* output_data,
        uint32_t* output_lengths,
        uint32_t max_output_len,
        uint32_t count,
        uint8_t version_byte,
        cudaStream_t stream
    );
    
    void launch_bitcoin_p2pkh_address_batch(
        const uint8_t* hash160_values,
        char* output_addresses,
        uint32_t* output_lengths,
        uint32_t max_address_len,
        uint32_t count,
        uint8_t version_byte,
        cudaStream_t stream
    );
    
    void launch_base58_decode_batch(
        const char* input_strings,
        const uint32_t* input_lengths,
        uint8_t* output_data,
        uint32_t* output_lengths,
        uint32_t max_output_len,
        uint32_t count,
        cudaStream_t stream
    );
}

// Base58Encoder Implementation
Base58Encoder::Base58Encoder() 
    : device_id_(0), is_initialized_(false) {
    // Initialize GPU buffers structure
    gpu_buffers_ = GPUBuffers();
    streams_created_ = false;
}

Base58Encoder::~Base58Encoder() {
    cleanup();
}

bool Base58Encoder::initialize(int device_id) {
    if (is_initialized_) {
        cleanup();
    }
    
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        return false;
    }
    
    error = cudaStreamCreate(&computation_stream_);
    if (error != cudaSuccess) {
        return false;
    }
    
    error = cudaStreamCreate(&memory_stream_);
    if (error != cudaSuccess) {
        cudaStreamDestroy(computation_stream_);
        computation_stream_ = nullptr;
        return false;
    }
    
    streams_created_ = true;
    
    device_id_ = device_id;
    is_initialized_ = true;
    return true;
}

bool Base58Encoder::configure(const Base58Config& config) {
    if (!is_initialized_) {
        return false;
    }
    
    config_ = config;
    return true;
}

void Base58Encoder::cleanup() {
    if (streams_created_) {
        if (computation_stream_) {
            cudaStreamDestroy(computation_stream_);
            computation_stream_ = nullptr;
        }
        if (memory_stream_) {
            cudaStreamDestroy(memory_stream_);
            memory_stream_ = nullptr;
        }
        streams_created_ = false;
    }
    
    // Cleanup GPU buffers
    if (gpu_buffers_.input_data) {
        cudaFree(gpu_buffers_.input_data);
        gpu_buffers_.input_data = nullptr;
    }
    if (gpu_buffers_.input_lengths) {
        cudaFree(gpu_buffers_.input_lengths);
        gpu_buffers_.input_lengths = nullptr;
    }
    if (gpu_buffers_.input_strings) {
        cudaFree(gpu_buffers_.input_strings);
        gpu_buffers_.input_strings = nullptr;
    }
    if (gpu_buffers_.output_strings) {
        cudaFree(gpu_buffers_.output_strings);
        gpu_buffers_.output_strings = nullptr;
    }
    if (gpu_buffers_.output_data) {
        cudaFree(gpu_buffers_.output_data);
        gpu_buffers_.output_data = nullptr;
    }
    if (gpu_buffers_.output_lengths) {
        cudaFree(gpu_buffers_.output_lengths);
        gpu_buffers_.output_lengths = nullptr;
    }
    if (gpu_buffers_.working_buffer) {
        cudaFree(gpu_buffers_.working_buffer);
        gpu_buffers_.working_buffer = nullptr;
    }
    if (gpu_buffers_.error_flags) {
        cudaFree(gpu_buffers_.error_flags);
        gpu_buffers_.error_flags = nullptr;
    }
    
    gpu_buffers_.allocated_batch_size = 0;
    gpu_buffers_.allocated_size_bytes = 0;
    
    is_initialized_ = false;
}

// Basic Base58 encoding operations
Base58Result Base58Encoder::encode_batch(
    const std::vector<std::vector<uint8_t>>& input_data) {
    
    Base58Result result;
    if (!is_initialized_ || input_data.empty()) {
        return result;
    }
    
    // Prepare contiguous memory for batch processing
    size_t total_input_size = 0;
    std::vector<uint32_t> input_lengths;
    std::vector<uint8_t> flat_input;
    
    for (const auto& data : input_data) {
        input_lengths.push_back(static_cast<uint32_t>(data.size()));
        flat_input.insert(flat_input.end(), data.begin(), data.end());
        total_input_size += data.size();
    }
    
    // Allocate device memory
    uint8_t* d_input = nullptr;
    uint32_t* d_input_lengths = nullptr;
    char* d_output = nullptr;
    uint32_t* d_output_lengths = nullptr;
    
    cudaMalloc(&d_input, total_input_size);
    cudaMalloc(&d_input_lengths, input_lengths.size() * sizeof(uint32_t));
    cudaMalloc(&d_output, input_data.size() * config_.max_output_length);
    cudaMalloc(&d_output_lengths, input_data.size() * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_input, flat_input.data(), total_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_lengths, input_lengths.data(), input_lengths.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    launch_base58_encode_batch(
        d_input,
        d_input_lengths,
        d_output,
        d_output_lengths,
        config_.max_output_length,
        static_cast<uint32_t>(input_data.size()),
        computation_stream_
    );
    
    // Copy results back
    std::vector<char> output_buffer(input_data.size() * config_.max_output_length);
    std::vector<uint32_t> output_lengths_buffer(input_data.size());
    
    cudaMemcpy(output_buffer.data(), d_output, output_buffer.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_lengths_buffer.data(), d_output_lengths, output_lengths_buffer.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Process results
    for (size_t i = 0; i < input_data.size(); ++i) {
        size_t start_idx = i * config_.max_output_length;
        size_t length = output_lengths_buffer[i];
        
        if (length > 0 && length <= config_.max_output_length) {
            std::string encoded(output_buffer.data() + start_idx, length);
            result.encoded_strings.push_back(encoded);
        }
    }
    
    // Cleanup device memory
    cudaFree(d_input);
    cudaFree(d_input_lengths);
    cudaFree(d_output);
    cudaFree(d_output_lengths);
    
    return result;
}

Base58Result Base58Encoder::encode_batch(
    const uint8_t* input_data,
    const size_t* input_lengths,
    size_t count) {
    
    Base58Result result;
    if (!is_initialized_ || count == 0) {
        return result;
    }
    
    // Convert sizes to uint32_t for CUDA
    std::vector<uint32_t> lengths_uint32(count);
    for (size_t i = 0; i < count; ++i) {
        lengths_uint32[i] = static_cast<uint32_t>(input_lengths[i]);
    }
    
    // Calculate total input size
    size_t total_input_size = 0;
    for (size_t i = 0; i < count; ++i) {
        total_input_size += input_lengths[i];
    }
    
    // Allocate device memory
    uint8_t* d_input = nullptr;
    uint32_t* d_input_lengths = nullptr;
    char* d_output = nullptr;
    uint32_t* d_output_lengths = nullptr;
    
    cudaMalloc(&d_input, total_input_size);
    cudaMalloc(&d_input_lengths, count * sizeof(uint32_t));
    cudaMalloc(&d_output, count * config_.max_output_length);
    cudaMalloc(&d_output_lengths, count * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_input, input_data, total_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_lengths, lengths_uint32.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    launch_base58_encode_batch(
        d_input,
        d_input_lengths,
        d_output,
        d_output_lengths,
        config_.max_output_length,
        static_cast<uint32_t>(count),
        computation_stream_
    );
    
    // Copy results back
    std::vector<char> output_buffer(count * config_.max_output_length);
    std::vector<uint32_t> output_lengths_buffer(count);
    
    cudaMemcpy(output_buffer.data(), d_output, output_buffer.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_lengths_buffer.data(), d_output_lengths, output_lengths_buffer.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Process results
    for (size_t i = 0; i < count; ++i) {
        size_t start_idx = i * config_.max_output_length;
        size_t length = output_lengths_buffer[i];
        
        if (length > 0 && length <= config_.max_output_length) {
            std::string encoded(output_buffer.data() + start_idx, length);
            result.encoded_strings.push_back(encoded);
        }
    }
    
    // Cleanup device memory
    cudaFree(d_input);
    cudaFree(d_input_lengths);
    cudaFree(d_output);
    cudaFree(d_output_lengths);
    
    return result;
}

std::string Base58Encoder::encode_single(
    const uint8_t* input_data,
    size_t input_length) {
    
    if (!is_initialized_) {
        return "";
    }
    
    // Use batch operation with single item
    size_t length = input_length;
    Base58Result result = encode_batch(input_data, &length, 1);
    
    if (!result.encoded_strings.empty()) {
        return result.encoded_strings[0];
    }
    
    return "";
}

// Base58Check encoding operations
Base58Result Base58Encoder::encode_check_batch(
    const std::vector<std::vector<uint8_t>>& input_data) {
    
    Base58Result result;
    if (!is_initialized_ || input_data.empty()) {
        return result;
    }
    
    // Prepare data for batch processing
    size_t total_input_size = 0;
    std::vector<uint32_t> input_lengths;
    std::vector<uint8_t> flat_input;
    
    for (const auto& data : input_data) {
        input_lengths.push_back(static_cast<uint32_t>(data.size()));
        flat_input.insert(flat_input.end(), data.begin(), data.end());
        total_input_size += data.size();
    }
    
    // Allocate device memory
    uint8_t* d_input = nullptr;
    uint32_t* d_input_lengths = nullptr;
    char* d_output = nullptr;
    uint32_t* d_output_lengths = nullptr;
    
    cudaMalloc(&d_input, total_input_size);
    cudaMalloc(&d_input_lengths, input_lengths.size() * sizeof(uint32_t));
    cudaMalloc(&d_output, input_data.size() * config_.max_output_length);
    cudaMalloc(&d_output_lengths, input_data.size() * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_input, flat_input.data(), total_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_lengths, input_lengths.data(), input_lengths.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch Base58Check kernel (version byte 0 for mainnet P2PKH)
    launch_base58check_encode_batch(
        d_input,
        d_input_lengths,
        d_output,
        d_output_lengths,
        config_.max_output_length,
        static_cast<uint32_t>(input_data.size()),
        0x00, // Mainnet version byte
        computation_stream_
    );
    
    // Copy results back and process
    std::vector<char> output_buffer(input_data.size() * config_.max_output_length);
    std::vector<uint32_t> output_lengths_buffer(input_data.size());
    
    cudaMemcpy(output_buffer.data(), d_output, output_buffer.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_lengths_buffer.data(), d_output_lengths, output_lengths_buffer.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        size_t start_idx = i * config_.max_output_length;
        size_t length = output_lengths_buffer[i];
        
        if (length > 0) {
            std::string encoded(output_buffer.data() + start_idx, length);
            result.encoded_strings.push_back(encoded);
        }
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_input_lengths);
    cudaFree(d_output);
    cudaFree(d_output_lengths);
    
    return result;
}

// Base58Check encoding operations - pointer version
Base58Result Base58Encoder::encode_check_batch(
    const uint8_t* input_data,
    const size_t* input_lengths,
    size_t count
) {
    
    Base58Result result;
    if (!is_initialized_ || count == 0) {
        return result;
    }
    
    // Convert sizes to uint32_t for CUDA
    std::vector<uint32_t> lengths_uint32(count);
    for (size_t i = 0; i < count; ++i) {
        lengths_uint32[i] = static_cast<uint32_t>(input_lengths[i]);
    }
    
    // Calculate total input size
    size_t total_input_size = 0;
    for (size_t i = 0; i < count; ++i) {
        total_input_size += input_lengths[i];
    }
    
    // Allocate device memory
    uint8_t* d_input = nullptr;
    uint32_t* d_input_lengths = nullptr;
    char* d_output = nullptr;
    uint32_t* d_output_lengths = nullptr;
    
    cudaMalloc(&d_input, total_input_size);
    cudaMalloc(&d_input_lengths, count * sizeof(uint32_t));
    cudaMalloc(&d_output, count * config_.max_output_length);
    cudaMalloc(&d_output_lengths, count * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_input, input_data, total_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_lengths, lengths_uint32.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch Base58Check kernel (version byte 0 for mainnet P2PKH)
    launch_base58check_encode_batch(
        d_input,
        d_input_lengths,
        d_output,
        d_output_lengths,
        config_.max_output_length,
        static_cast<uint32_t>(count),
        0x00, // Mainnet version byte
        computation_stream_
    );
    
    // Copy results back
    std::vector<char> output_buffer(count * config_.max_output_length);
    std::vector<uint32_t> output_lengths_buffer(count);
    
    cudaMemcpy(output_buffer.data(), d_output, output_buffer.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_lengths_buffer.data(), d_output_lengths, output_lengths_buffer.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Process results
    for (size_t i = 0; i < count; ++i) {
        size_t start_idx = i * config_.max_output_length;
        size_t length = output_lengths_buffer[i];
        
        if (length > 0) {
            std::string encoded(output_buffer.data() + start_idx, length);
            result.encoded_strings.push_back(encoded);
        }
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_input_lengths);
    cudaFree(d_output);
    cudaFree(d_output_lengths);
    
    result.success = !result.encoded_strings.empty();
    result.items_processed = count;
    result.items_successful = result.encoded_strings.size();
    
    return result;
}

std::string Base58Encoder::encode_check_single(
    const uint8_t* input_data,
    size_t input_length
) {
    
    if (!is_initialized_) {
        return "";
    }
    
    // Use batch operation with single item
    size_t length = input_length;
    Base58Result result = encode_check_batch(input_data, &length, 1);
    
    if (!result.encoded_strings.empty()) {
        return result.encoded_strings[0];
    }
    
    return "";
}

// Base58 decoding operations
Base58Result Base58Encoder::decode_batch(
    const std::vector<std::string>& input_strings
) {
    
    Base58Result result;
    if (!is_initialized_ || input_strings.empty()) {
        return result;
    }
    
    // Prepare data for batch processing
    size_t total_input_size = 0;
    std::vector<uint32_t> input_lengths;
    std::vector<char> flat_input;
    
    for (const auto& str : input_strings) {
        input_lengths.push_back(static_cast<uint32_t>(str.size()));
        flat_input.insert(flat_input.end(), str.begin(), str.end());
        total_input_size += str.size();
    }
    
    // Allocate device memory
    char* d_input = nullptr;
    uint32_t* d_input_lengths = nullptr;
    uint8_t* d_output = nullptr;
    uint32_t* d_output_lengths = nullptr;
    
    cudaMalloc(&d_input, total_input_size);
    cudaMalloc(&d_input_lengths, input_lengths.size() * sizeof(uint32_t));
    cudaMalloc(&d_output, input_strings.size() * config_.max_input_length);
    cudaMalloc(&d_output_lengths, input_strings.size() * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_input, flat_input.data(), total_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_lengths, input_lengths.data(), input_lengths.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch decoding kernel
    launch_base58_decode_batch(
        d_input,
        d_input_lengths,
        d_output,
        d_output_lengths,
        config_.max_input_length,
        static_cast<uint32_t>(input_strings.size()),
        computation_stream_
    );
    
    // Copy results back
    std::vector<uint8_t> output_buffer(input_strings.size() * config_.max_input_length);
    std::vector<uint32_t> output_lengths_buffer(input_strings.size());
    
    cudaMemcpy(output_buffer.data(), d_output, output_buffer.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_lengths_buffer.data(), d_output_lengths, output_lengths_buffer.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Process results
    for (size_t i = 0; i < input_strings.size(); ++i) {
        size_t start_idx = i * config_.max_input_length;
        size_t length = output_lengths_buffer[i];
        
        if (length > 0) {
            std::vector<uint8_t> decoded(output_buffer.data() + start_idx, output_buffer.data() + start_idx + length);
            result.decoded_data.push_back(decoded);
        }
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_input_lengths);
    cudaFree(d_output);
    cudaFree(d_output_lengths);
    
    result.success = !result.decoded_data.empty();
    result.items_processed = input_strings.size();
    result.items_successful = result.decoded_data.size();
    
    return result;
}

Base58Result Base58Encoder::decode_batch(
    const char* input_data,
    const size_t* input_lengths,
    size_t count
) {
    
    Base58Result result;
    if (!is_initialized_ || count == 0) {
        return result;
    }
    
    // Convert sizes to uint32_t for CUDA
    std::vector<uint32_t> lengths_uint32(count);
    for (size_t i = 0; i < count; ++i) {
        lengths_uint32[i] = static_cast<uint32_t>(input_lengths[i]);
    }
    
    // Calculate total input size
    size_t total_input_size = 0;
    for (size_t i = 0; i < count; ++i) {
        total_input_size += input_lengths[i];
    }
    
    // Allocate device memory
    char* d_input = nullptr;
    uint32_t* d_input_lengths = nullptr;
    uint8_t* d_output = nullptr;
    uint32_t* d_output_lengths = nullptr;
    
    cudaMalloc(&d_input, total_input_size);
    cudaMalloc(&d_input_lengths, count * sizeof(uint32_t));
    cudaMalloc(&d_output, count * config_.max_input_length);
    cudaMalloc(&d_output_lengths, count * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_input, input_data, total_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_lengths, lengths_uint32.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch decoding kernel
    launch_base58_decode_batch(
        d_input,
        d_input_lengths,
        d_output,
        d_output_lengths,
        config_.max_input_length,
        static_cast<uint32_t>(count),
        computation_stream_
    );
    
    // Copy results back
    std::vector<uint8_t> output_buffer(count * config_.max_input_length);
    std::vector<uint32_t> output_lengths_buffer(count);
    
    cudaMemcpy(output_buffer.data(), d_output, output_buffer.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_lengths_buffer.data(), d_output_lengths, output_lengths_buffer.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Process results
    for (size_t i = 0; i < count; ++i) {
        size_t start_idx = i * config_.max_input_length;
        size_t length = output_lengths_buffer[i];
        
        if (length > 0) {
            std::vector<uint8_t> decoded(output_buffer.data() + start_idx, output_buffer.data() + start_idx + length);
            result.decoded_data.push_back(decoded);
        }
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_input_lengths);
    cudaFree(d_output);
    cudaFree(d_output_lengths);
    
    result.success = !result.decoded_data.empty();
    result.items_processed = count;
    result.items_successful = result.decoded_data.size();
    
    return result;
}

std::vector<uint8_t> Base58Encoder::decode_single(
    const std::string& input_string
) {
    
    if (!is_initialized_ || input_string.empty()) {
        return {};
    }
    
    // Use batch operation with single item
    const char* input_data = input_string.c_str();
    size_t length = input_string.size();
    Base58Result result = decode_batch(input_data, &length, 1);
    
    if (!result.decoded_data.empty()) {
        return result.decoded_data[0];
    }
    
    return {};
}

// Base58Check decoding operations
Base58Result Base58Encoder::decode_check_batch(
    const std::vector<std::string>& input_strings
) {
    
    Base58Result result;
    if (!is_initialized_ || input_strings.empty()) {
        return result;
    }
    
    // First decode normally
    Base58Result decode_result = decode_batch(input_strings);
    
    if (!decode_result.success) {
        return result;
    }
    
    // Then validate checksums
    for (size_t i = 0; i < decode_result.decoded_data.size(); ++i) {
        const auto& decoded = decode_result.decoded_data[i];
        
        if (decoded.size() >= 4) { // Minimum size for checksum
            // Extract data and checksum
            std::vector<uint8_t> data(decoded.begin(), decoded.end() - 4);
            std::vector<uint8_t> checksum(decoded.end() - 4, decoded.end());
            
            // Calculate expected checksum
            auto expected_checksum = sha256_double_hash(data);
            expected_checksum.resize(4); // First 4 bytes
            
            if (expected_checksum == checksum) {
                result.decoded_data.push_back(data);
                result.item_success.push_back(true);
            } else {
                result.item_success.push_back(false);
                result.error_messages.push_back("Checksum validation failed");
            }
        } else {
            result.item_success.push_back(false);
            result.error_messages.push_back("Invalid data length for Base58Check");
        }
    }
    
    result.success = !result.decoded_data.empty();
    result.items_processed = input_strings.size();
    result.items_successful = result.decoded_data.size();
    
    return result;
}

std::vector<uint8_t> Base58Encoder::decode_check_single(
    const std::string& input_string
) {
    
    if (!is_initialized_ || input_string.empty()) {
        return {};
    }
    
    // Use batch operation with single item
    std::vector<std::string> inputs = {input_string};
    Base58Result result = decode_check_batch(inputs);
    
    if (!result.decoded_data.empty()) {
        return result.decoded_data[0];
    }
    
    return {};
}

// Bitcoin address generation
Base58Result Base58Encoder::generate_p2pkh_addresses_batch(
    const uint8_t* hash160_values,
    size_t count
) {
    
    Base58Result result;
    if (!is_initialized_ || count == 0) {
        return result;
    }
    
    // Allocate device memory
    uint8_t* d_hash160 = nullptr;
    char* d_output = nullptr;
    uint32_t* d_output_lengths = nullptr;
    
    cudaMalloc(&d_hash160, count * 20); // 20 bytes per hash160
    cudaMalloc(&d_output, count * config_.max_output_length);
    cudaMalloc(&d_output_lengths, count * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_hash160, hash160_values, count * 20, cudaMemcpyHostToDevice);
    
    // Launch Bitcoin address kernel
    launch_bitcoin_p2pkh_address_batch(
        d_hash160,
        d_output,
        d_output_lengths,
        config_.max_output_length,
        static_cast<uint32_t>(count),
        0x00, // Mainnet version byte
        computation_stream_
    );
    
    // Copy results back
    std::vector<char> output_buffer(count * config_.max_output_length);
    std::vector<uint32_t> output_lengths_buffer(count);
    
    cudaMemcpy(output_buffer.data(), d_output, output_buffer.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_lengths_buffer.data(), d_output_lengths, output_lengths_buffer.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Process results
    for (size_t i = 0; i < count; ++i) {
        size_t start_idx = i * config_.max_output_length;
        size_t length = output_lengths_buffer[i];
        
        if (length > 0) {
            std::string address(output_buffer.data() + start_idx, length);
            result.encoded_strings.push_back(address);
        }
    }
    
    // Cleanup
    cudaFree(d_hash160);
    cudaFree(d_output);
    cudaFree(d_output_lengths);
    
    result.success = !result.encoded_strings.empty();
    result.items_processed = count;
    result.items_successful = result.encoded_strings.size();
    
    return result;
}

Base58Result Base58Encoder::generate_p2sh_addresses_batch(
    const uint8_t* script_hashes,
    size_t count
) {
    
    Base58Result result;
    if (!is_initialized_ || count == 0) {
        return result;
    }
    
    // For P2SH addresses, use version byte 0x05
    // The implementation is similar to P2PKH but with different version byte
    
    // Allocate device memory
    uint8_t* d_script_hash = nullptr;
    char* d_output = nullptr;
    uint32_t* d_output_lengths = nullptr;
    
    cudaMalloc(&d_script_hash, count * 20); // 20 bytes per script hash
    cudaMalloc(&d_output, count * config_.max_output_length);
    cudaMalloc(&d_output_lengths, count * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_script_hash, script_hashes, count * 20, cudaMemcpyHostToDevice);
    
    // Launch Bitcoin address kernel with P2SH version byte
    launch_bitcoin_p2pkh_address_batch(
        d_script_hash,
        d_output,
        d_output_lengths,
        config_.max_output_length,
        static_cast<uint32_t>(count),
        0x05, // P2SH version byte
        computation_stream_
    );
    
    // Copy results back
    std::vector<char> output_buffer(count * config_.max_output_length);
    std::vector<uint32_t> output_lengths_buffer(count);
    
    cudaMemcpy(output_buffer.data(), d_output, output_buffer.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_lengths_buffer.data(), d_output_lengths, output_lengths_buffer.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Process results
    for (size_t i = 0; i < count; ++i) {
        size_t start_idx = i * config_.max_output_length;
        size_t length = output_lengths_buffer[i];
        
        if (length > 0) {
            std::string address(output_buffer.data() + start_idx, length);
            result.encoded_strings.push_back(address);
        }
    }
    
    // Cleanup
    cudaFree(d_script_hash);
    cudaFree(d_output);
    cudaFree(d_output_lengths);
    
    result.success = !result.encoded_strings.empty();
    result.items_processed = count;
    result.items_successful = result.encoded_strings.size();
    
    return result;
}

std::string Base58Encoder::generate_p2pkh_address_single(
    const uint8_t hash160[20]
) {
    
    if (!is_initialized_) {
        return "";
    }
    
    // Use batch operation with single item
    Base58Result result = generate_p2pkh_addresses_batch(hash160, 1);
    
    if (!result.encoded_strings.empty()) {
        return result.encoded_strings[0];
    }
    
    return "";
}

std::string Base58Encoder::generate_p2sh_address_single(
    const uint8_t script_hash[20]
) {
    
    if (!is_initialized_) {
        return "";
    }
    
    // Use batch operation with single item
    Base58Result result = generate_p2sh_addresses_batch(script_hash, 1);
    
    if (!result.encoded_strings.empty()) {
        return result.encoded_strings[0];
    }
    
    return "";
}

// Address validation
bool Base58Encoder::validate_bitcoin_address(const std::string& address) {
    
    if (address.empty()) {
        return false;
    }
    
    // Basic format validation
    if (address.length() < MIN_BITCOIN_ADDRESS_LENGTH || 
        address.length() > MAX_BITCOIN_ADDRESS_LENGTH) {
        return false;
    }
    
    // Check if all characters are valid Base58
    for (char c : address) {
        if (!is_valid_base58_character(c)) {
            return false;
        }
    }
    
    // Decode and validate checksum
    auto decoded = decode_check_single(address);
    if (decoded.empty()) {
        return false;
    }
    
    // Check version byte (P2PKH: 0x00, P2SH: 0x05)
    if (decoded.size() < 1) {
        return false;
    }
    
    uint8_t version_byte = decoded[0];
    return (version_byte == 0x00 || version_byte == 0x05);
}

bool Base58Encoder::validate_bitcoin_addresses_batch(const std::vector<std::string>& addresses) {
    
    if (addresses.empty()) {
        return false;
    }
    
    // Decode all addresses
    Base58Result result = decode_check_batch(addresses);
    
    if (!result.success) {
        return false;
    }
    
    // Check if all addresses have valid version bytes
    for (const auto& decoded : result.decoded_data) {
        if (decoded.empty() || (decoded[0] != 0x00 && decoded[0] != 0x05)) {
            return false;
        }
    }
    
    return true;
}

// Performance monitoring
Base58Metrics Base58Encoder::get_performance_metrics() const {
    return metrics_;
}

void Base58Encoder::reset_performance_counters() {
    metrics_ = Base58Metrics();
}

// Benchmark and validation
Base58Metrics Base58Encoder::benchmark_performance(
    size_t test_data_count,
    size_t iterations
) {
    
    Base58Metrics benchmark_metrics;
    
    if (!is_initialized_ || test_data_count == 0 || iterations == 0) {
        return benchmark_metrics;
    }
    
    // Generate test data
    std::vector<std::vector<uint8_t>> test_data;
    for (size_t i = 0; i < test_data_count; ++i) {
        std::vector<uint8_t> data(20); // 20 bytes for hash160
        for (size_t j = 0; j < 20; ++j) {
            data[j] = static_cast<uint8_t>((i + j) % 256);
        }
        test_data.push_back(data);
    }
    
    // Run benchmark
    for (size_t iter = 0; iter < iterations; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        
        Base58Result result = encode_check_batch(test_data);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (result.success) {
            benchmark_metrics.encodings_per_second += 
                (test_data_count * 1000000.0) / duration.count();
            benchmark_metrics.successful_operations += test_data_count;
        } else {
            benchmark_metrics.failed_operations += test_data_count;
        }
        
        benchmark_metrics.total_encodings_performed += test_data_count;
        benchmark_metrics.total_input_bytes += test_data_count * 20;
        benchmark_metrics.total_output_bytes += test_data_count * 35; // Approx address length
    }
    
    // Calculate averages
    if (iterations > 0) {
        benchmark_metrics.encodings_per_second /= iterations;
    }
    
    return benchmark_metrics;
}

bool Base58Encoder::validate_correctness(
    const std::vector<std::vector<uint8_t>>& test_data,
    const std::vector<std::string>& expected_results
) {
    
    if (test_data.size() != expected_results.size()) {
        return false;
    }
    
    if (!is_initialized_ || test_data.empty()) {
        return false;
    }
    
    // Encode the test data
    Base58Result result = encode_check_batch(test_data);
    
    if (!result.success || result.encoded_strings.size() != test_data.size()) {
        return false;
    }
    
    // Compare with expected results
    for (size_t i = 0; i < test_data.size(); ++i) {
        if (result.encoded_strings[i] != expected_results[i]) {
            return false;
        }
    }
    
    return true;
}

// Memory management
size_t Base58Encoder::get_gpu_memory_usage() const {
    return 0; // Placeholder - would track actual GPU memory usage
}

size_t Base58Encoder::get_cpu_memory_usage() const {
    return 0; // Placeholder - would track actual CPU memory usage
}

bool Base58Encoder::allocate_gpu_memory(size_t max_batch_size) {
    // Implementation would allocate persistent GPU memory
    return true;
}

void Base58Encoder::deallocate_gpu_memory() {
    // Implementation would free persistent GPU memory
}

// Utility methods
bool Base58Encoder::is_valid_base58_character(char c) {
    static const std::string valid_chars = 
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    return valid_chars.find(c) != std::string::npos;
}

bool Base58Encoder::is_valid_base58_string(const std::string& str) {
    for (char c : str) {
        if (!is_valid_base58_character(c)) {
            return false;
        }
    }
    return true;
}

size_t Base58Encoder::estimate_encoded_length(size_t input_length) {
    // Base58 encoding expands data by ~1.38x
    return static_cast<size_t>(input_length * 1.38) + 4; // +4 for safety
}

size_t Base58Encoder::estimate_decoded_length(size_t encoded_length) {
    // Base58 decoding compresses data by ~0.72x
    return static_cast<size_t>(encoded_length * 0.72) + 1; // +1 for safety
}

// Constants
const char* Base58Encoder::BASE58_ALPHABET = 
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Hash functions (for Base58Check)
std::vector<uint8_t> Base58Encoder::sha256_double_hash(const std::vector<uint8_t>& data) {
    // Placeholder implementation - would use proper SHA256
    std::vector<uint8_t> result(32);
    for (size_t i = 0; i < 32; ++i) {
        result[i] = static_cast<uint8_t>(i);
    }
    return result;
}

void Base58Encoder::sha256_double_hash(const uint8_t* input, size_t length, uint8_t* output) {
    // Placeholder implementation
    for (size_t i = 0; i < 32; ++i) {
        output[i] = static_cast<uint8_t>(i);
    }
}

// Base58EncoderFactory Implementation
std::unique_ptr<Base58Encoder> Base58EncoderFactory::create_encoder(
    PerformanceProfile profile, int device_id) {
    
    auto encoder = std::make_unique<Base58Encoder>();
    
    if (!encoder->initialize(device_id)) {
        return nullptr;
    }
    
    Base58Config config;
    
    // Configure based on performance profile
    switch (profile) {
        case PerformanceProfile::MAXIMUM_SPEED:
            config.threads_per_block = 1024;
            config.blocks_per_grid = 256;
            config.max_batch_size = 100000;
            config.enable_shared_memory_optimization = true;
            config.enable_memory_coalescing = true;
            config.enable_vectorized_operations = true;
            config.enable_input_validation = false;
            config.enable_checksum_validation = false;
            break;
        case PerformanceProfile::BALANCED:
            config.threads_per_block = 512;
            config.blocks_per_grid = 128;
            config.max_batch_size = 50000;
            config.enable_shared_memory_optimization = true;
            config.enable_memory_coalescing = true;
            config.enable_vectorized_operations = true;
            config.enable_input_validation = true;
            config.enable_checksum_validation = true;
            break;
        case PerformanceProfile::LOW_MEMORY:
            config.threads_per_block = 256;
            config.blocks_per_grid = 64;
            config.max_batch_size = 10000;
            config.enable_shared_memory_optimization = false;
            config.enable_memory_coalescing = true;
            config.enable_vectorized_operations = false;
            config.enable_input_validation = true;
            config.enable_checksum_validation = true;
            break;
        case PerformanceProfile::HIGH_ACCURACY:
            config.threads_per_block = 256;
            config.blocks_per_grid = 64;
            config.max_batch_size = 10000;
            config.enable_shared_memory_optimization = true;
            config.enable_memory_coalescing = true;
            config.enable_vectorized_operations = true;
            config.enable_input_validation = true;
            config.enable_checksum_validation = true;
            break;
    }
    
    if (!encoder->configure(config)) {
        return nullptr;
    }
    
    return encoder;
}

Base58Config Base58EncoderFactory::get_recommended_config(
    PerformanceProfile profile,
    size_t expected_batch_size,
    size_t available_gpu_memory_mb
) {
    
    Base58Config config;
    
    // Adjust configuration based on available memory
    size_t max_possible_batch = (available_gpu_memory_mb * 1024 * 1024) / 
                               (config.max_input_length + config.max_output_length + sizeof(uint32_t) * 2);
    
    switch (profile) {
        case PerformanceProfile::MAXIMUM_SPEED:
            config.threads_per_block = 1024;
            config.blocks_per_grid = 256;
            config.max_batch_size = std::min(expected_batch_size, max_possible_batch);
            config.enable_shared_memory_optimization = true;
            config.enable_memory_coalescing = true;
            config.enable_vectorized_operations = true;
            config.enable_input_validation = false;
            config.enable_checksum_validation = false;
            break;
        case PerformanceProfile::BALANCED:
            config.threads_per_block = 512;
            config.blocks_per_grid = 128;
            config.max_batch_size = std::min(expected_batch_size, max_possible_batch);
            config.enable_shared_memory_optimization = true;
            config.enable_memory_coalescing = true;
            config.enable_vectorized_operations = true;
            config.enable_input_validation = true;
            config.enable_checksum_validation = true;
            break;
        case PerformanceProfile::LOW_MEMORY:
            config.threads_per_block = 256;
            config.blocks_per_grid = 64;
            config.max_batch_size = std::min(expected_batch_size, max_possible_batch / 2);
            config.enable_shared_memory_optimization = false;
            config.enable_memory_coalescing = true;
            config.enable_vectorized_operations = false;
            config.enable_input_validation = true;
            config.enable_checksum_validation = true;
            break;
        case PerformanceProfile::HIGH_ACCURACY:
            config.threads_per_block = 256;
            config.blocks_per_grid = 64;
            config.max_batch_size = std::min(expected_batch_size, max_possible_batch / 4);
            config.enable_shared_memory_optimization = true;
            config.enable_memory_coalescing = true;
            config.enable_vectorized_operations = true;
            config.enable_input_validation = true;
            config.enable_checksum_validation = true;
            break;
    }
    
    return config;
}

Base58Config Base58EncoderFactory::get_config_for_architecture(
    int compute_capability_major,
    int compute_capability_minor,
    size_t multiprocessor_count
) {
    
    Base58Config config;
    
    // Optimize for different GPU architectures
    if (compute_capability_major >= 8) { // Ampere or newer
        config.threads_per_block = 1024;
        config.blocks_per_grid = multiprocessor_count * 4;
        config.enable_shared_memory_optimization = true;
        config.enable_memory_coalescing = true;
        config.enable_vectorized_operations = true;
    } else if (compute_capability_major >= 7) { // Volta/Turing
        config.threads_per_block = 1024;
        config.blocks_per_grid = multiprocessor_count * 2;
        config.enable_shared_memory_optimization = true;
        config.enable_memory_coalescing = true;
        config.enable_vectorized_operations = true;
    } else if (compute_capability_major >= 6) { // Pascal
        config.threads_per_block = 1024;
        config.blocks_per_grid = multiprocessor_count;
        config.enable_shared_memory_optimization = true;
        config.enable_memory_coalescing = true;
        config.enable_vectorized_operations = false;
    } else { // Older architectures
        config.threads_per_block = 512;
        config.blocks_per_grid = multiprocessor_count;
        config.enable_shared_memory_optimization = false;
        config.enable_memory_coalescing = true;
        config.enable_vectorized_operations = false;
    }
    
    return config;
}
} // namespace compare
} // namespace keyhunt
