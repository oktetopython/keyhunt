/**
 * @file private_key_scanner.cpp
 * @brief Implementation of private key range scanning framework with batch processing and GPU optimization
 * @author KeyhuntCUDA Team
 * 
 * T041: Implement private key range scanning framework with batch processing and GPU optimization
 * 
 * Integrates BitCrack concepts with KeyhuntCore ECC operations for high-performance
 * GPU-accelerated Bitcoin private key scanning with scientific validation.
 */

#include "private_key_scanner.h"
#include "../validation/ecc_validation_framework.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>

namespace keyhunt {
namespace scan {

// PrivateKeyScanner implementation
PrivateKeyScanner::PrivateKeyScanner() 
    : is_scanning_(false)
    , is_paused_(false)
    , should_stop_(false)
    , device_id_(-1)
    , next_batch_id_(0)
{
}

PrivateKeyScanner::~PrivateKeyScanner() {
    cleanup();
}

bool PrivateKeyScanner::initialize(const ScanningConfiguration& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    try {
        // Validate configuration
        if (!validate_configuration()) {
            std::cerr << "ERROR: Invalid scanning configuration" << std::endl;
            return false;
        }
        
        // Store configuration
        config_ = config;
        
        // Initialize GPU device (default to device 0)
        if (!initialize_gpu(0)) {
            std::cerr << "ERROR: Failed to initialize GPU device 0" << std::endl;
            return false;
        }
        
        // Initialize memory manager with T035 GPU memory optimization
        gpu::MemoryManagerConfig mem_config;
        mem_config.pool_size = config_.max_gpu_memory_usage;
        mem_config.enable_pooling = config_.enable_memory_pooling;
        mem_config.strategy = gpu::MemoryStrategy::COALESCED; // BitCrack optimization
        
        memory_manager_ = std::make_unique<gpu::Secp256k1MemoryManager>(mem_config);
        if (!memory_manager_->initialize(device_id_)) {
            std::cerr << "ERROR: Failed to initialize GPU memory manager" << std::endl;
            return false;
        }
        
        // Initialize unified ECC interface with T034 CPU/GPU consistency
        ecc::unified::UnifiedECCConfig ecc_config;
        ecc_config.preferred_backend = ecc::unified::ComputeBackend::GPU;
        ecc_config.enable_validation = true;
        ecc_config.precision_threshold = 1e-10;
        
        ecc_interface_ = std::make_unique<ecc::unified::UnifiedECCInterface>(ecc_config);
        if (!ecc_interface_->initialize(device_id_)) {
            std::cerr << "ERROR: Failed to initialize unified ECC interface" << std::endl;
            return false;
        }
        
        // Create CUDA streams for asynchronous execution
        cuda_streams_.resize(config_.cuda_streams);
        for (size_t i = 0; i < config_.cuda_streams; i++) {
            cudaError_t err = cudaStreamCreate(&cuda_streams_[i]);
            if (err != cudaSuccess) {
                std::cerr << "ERROR: Failed to create CUDA stream " << i << ": " << cudaGetErrorString(err) << std::endl;
                return false;
            }
        }
        
        // Apply BitCrack optimizations from T040 analysis
        apply_bitcrack_optimizations();
        
        // Initialize metrics
        metrics_ = ScanningMetrics();
        scan_start_time_ = std::chrono::high_resolution_clock::now();
        
        std::cout << "PrivateKeyScanner initialized successfully" << std::endl;
        std::cout << "  Device ID: " << device_id_ << std::endl;
        std::cout << "  Keys per batch: " << config_.keys_per_batch << std::endl;
        std::cout << "  Threads per block: " << config_.threads_per_block << std::endl;
        std::cout << "  Blocks per grid: " << config_.blocks_per_grid << std::endl;
        std::cout << "  CUDA streams: " << config_.cuda_streams << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in PrivateKeyScanner::initialize: " << e.what() << std::endl;
        return false;
    }
}

bool PrivateKeyScanner::initialize_gpu(int device_id) {
    // Check CUDA device availability
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "ERROR: No CUDA devices available" << std::endl;
        return false;
    }
    
    if (device_id >= device_count) {
        std::cerr << "ERROR: Invalid device ID " << device_id << " (max: " << device_count - 1 << ")" << std::endl;
        return false;
    }
    
    // Set CUDA device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: Failed to set CUDA device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    device_id_ = device_id;
    
    // Get device properties for optimization
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        std::cerr << "WARNING: Failed to get device properties" << std::endl;
        return true; // Continue without properties
    }
    
    std::cout << "Initialized GPU device " << device_id << ": " << props.name << std::endl;
    std::cout << "  Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "  Global Memory: " << (props.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << props.multiProcessorCount << std::endl;
    
    return true;
}

void PrivateKeyScanner::cleanup() {
    // Stop scanning if active
    if (is_scanning_) {
        stop_scanning();
    }
    
    // Wait for worker threads to complete
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    // Clean up CUDA streams
    for (cudaStream_t stream : cuda_streams_) {
        cudaStreamDestroy(stream);
    }
    cuda_streams_.clear();
    
    // Clean up batches
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        
        // Deallocate memory for all batches
        for (auto& batch : active_batches_) {
            deallocate_batch_memory(batch.get());
        }
        active_batches_.clear();
        
        while (!pending_batches_.empty()) {
            auto batch = std::move(pending_batches_.front());
            pending_batches_.pop();
            deallocate_batch_memory(batch.get());
        }
        
        while (!completed_batches_.empty()) {
            auto batch = std::move(completed_batches_.front());
            completed_batches_.pop();
            deallocate_batch_memory(batch.get());
        }
    }
    
    // Clean up components
    if (memory_manager_) {
        memory_manager_->cleanup();
        memory_manager_.reset();
    }
    
    if (ecc_interface_) {
        ecc_interface_->cleanup();
        ecc_interface_.reset();
    }
    
    device_id_ = -1;
}

bool PrivateKeyScanner::start_scanning(const models::PrivateKeyRange& range) {
    if (is_scanning_) {
        std::cerr << "ERROR: Scanning already in progress" << std::endl;
        return false;
    }
    
    if (!validate_private_key_range(range)) {
        std::cerr << "ERROR: Invalid private key range" << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    try {
        // Update configuration with range parameters
        config_.start_key = range.start_key;
        config_.end_key = range.end_key;
        config_.current_key = range.start_key;
        
        // Initialize metrics
        metrics_ = ScanningMetrics();
        metrics_.keys_remaining = config_.end_key - config_.start_key;
        scan_start_time_ = std::chrono::high_resolution_clock::now();
        
        // Reset scanning state
        is_scanning_ = true;
        is_paused_ = false;
        should_stop_ = false;
        next_batch_id_ = 0;
        
        // Create worker threads for batch processing
        size_t worker_count = std::min(config_.cuda_streams, size_t(4)); // Max 4 workers
        worker_threads_.reserve(worker_count);
        
        for (size_t i = 0; i < worker_count; i++) {
            worker_threads_.emplace_back(&PrivateKeyScanner::batch_worker_thread, this);
        }
        
        // Generate initial batches
        size_t initial_batch_count = config_.cuda_streams * 2; // Queue 2x streams
        for (size_t i = 0; i < initial_batch_count && !is_range_completed(); i++) {
            auto start_key = get_next_batch_start_key();
            auto batch = create_scan_batch(start_key, config_.keys_per_batch, next_batch_id_++);
            
            if (batch && allocate_batch_memory(batch.get())) {
                std::lock_guard<std::mutex> batch_lock(batch_mutex_);
                pending_batches_.push(std::move(batch));
                batch_cv_.notify_one();
            }
        }
        
        std::cout << "Started scanning private key range:" << std::endl;
        std::cout << "  Start: " << config_.start_key.to_hex() << std::endl;
        std::cout << "  End:   " << config_.end_key.to_hex() << std::endl;
        std::cout << "  Keys per batch: " << config_.keys_per_batch << std::endl;
        std::cout << "  Worker threads: " << worker_count << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in start_scanning: " << e.what() << std::endl;
        is_scanning_ = false;
        return false;
    }
}

bool PrivateKeyScanner::pause_scanning() {
    if (!is_scanning_ || is_paused_) {
        return false;
    }
    
    is_paused_ = true;
    std::cout << "Scanning paused" << std::endl;
    
    // Save checkpoint if enabled
    if (config_.enable_checkpointing) {
        save_checkpoint();
    }
    
    return true;
}

bool PrivateKeyScanner::resume_scanning() {
    if (!is_scanning_ || !is_paused_) {
        return false;
    }
    
    is_paused_ = false;
    batch_cv_.notify_all(); // Wake up worker threads
    
    std::cout << "Scanning resumed" << std::endl;
    return true;
}

bool PrivateKeyScanner::stop_scanning() {
    if (!is_scanning_) {
        return false;
    }
    
    std::cout << "Stopping scanning..." << std::endl;
    
    should_stop_ = true;
    is_scanning_ = false;
    is_paused_ = false;
    batch_cv_.notify_all();
    
    // Wait for worker threads to complete
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    // Save final checkpoint
    if (config_.enable_checkpointing) {
        save_checkpoint();
    }
    
    std::cout << "Scanning stopped" << std::endl;
    return true;
}

ScanningMetrics PrivateKeyScanner::get_current_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

double PrivateKeyScanner::get_progress_percentage() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    if (config_.end_key <= config_.start_key) {
        return 0.0;
    }
    
    ecc::BigInt256 total_range = config_.end_key - config_.start_key;
    ecc::BigInt256 scanned = config_.current_key - config_.start_key;
    
    // Convert to double for percentage calculation
    double progress = static_cast<double>(scanned.to_uint64()) / static_cast<double>(total_range.to_uint64());
    return std::min(100.0, std::max(0.0, progress * 100.0));
}

std::vector<PrivateKeyScanner::ScanMatch> PrivateKeyScanner::get_matches() const {
    std::lock_guard<std::mutex> lock(matches_mutex_);
    return found_matches_;
}

size_t PrivateKeyScanner::get_match_count() const {
    std::lock_guard<std::mutex> lock(matches_mutex_);
    return found_matches_.size();
}

bool PrivateKeyScanner::set_target_addresses(const std::vector<std::string>& addresses) {
    if (is_scanning_) {
        std::cerr << "ERROR: Cannot change target addresses while scanning" << std::endl;
        return false;
    }
    
    // Validate addresses
    for (const auto& addr : addresses) {
        if (!scanning_utils::AddressConverter::is_valid_bitcoin_address(addr)) {
            std::cerr << "ERROR: Invalid Bitcoin address: " << addr << std::endl;
            return false;
        }
    }
    
    std::lock_guard<std::mutex> lock(targets_mutex_);
    target_addresses_ = addresses;
    target_address_hashes_ = convert_addresses_to_binary(addresses);
    
    std::cout << "Set " << addresses.size() << " target addresses" << std::endl;
    return true;
}

// Private methods implementation

void PrivateKeyScanner::batch_worker_thread() {
    while (!should_stop_) {
        std::unique_lock<std::mutex> lock(batch_mutex_);
        
        // Wait for available batch or stop signal
        batch_cv_.wait(lock, [this] { 
            return should_stop_ || (!pending_batches_.empty() && !is_paused_);
        });
        
        if (should_stop_) {
            break;
        }
        
        if (pending_batches_.empty() || is_paused_) {
            continue;
        }
        
        // Get next batch to process
        auto batch = std::move(pending_batches_.front());
        pending_batches_.pop();
        active_batches_.push_back(std::move(batch));
        
        ScanBatch* batch_ptr = active_batches_.back().get();
        lock.unlock();
        
        // Process the batch
        try {
            batch_ptr->state = ScanBatch::BatchState::PROCESSING;
            batch_ptr->start_time = std::chrono::high_resolution_clock::now();
            
            bool success = process_scan_batch(batch_ptr);
            
            batch_ptr->end_time = std::chrono::high_resolution_clock::now();
            batch_ptr->state = success ? ScanBatch::BatchState::COMPLETED : ScanBatch::BatchState::FAILED;
            
            // Update metrics
            update_scanning_metrics();
            
            // Generate new batch if needed
            if (!is_range_completed() && active_batches_.size() < config_.cuda_streams) {
                auto start_key = get_next_batch_start_key();
                auto new_batch = create_scan_batch(start_key, config_.keys_per_batch, next_batch_id_++);
                
                if (new_batch && allocate_batch_memory(new_batch.get())) {
                    std::lock_guard<std::mutex> batch_lock(batch_mutex_);
                    pending_batches_.push(std::move(new_batch));
                    batch_cv_.notify_one();
                }
            }
            
        } catch (const std::exception& e) {
            handle_batch_error(batch_ptr, e.what());
        }
        
        // Move completed batch
        lock.lock();
        auto it = std::find_if(active_batches_.begin(), active_batches_.end(),
                              [batch_ptr](const std::unique_ptr<ScanBatch>& b) { return b.get() == batch_ptr; });
        if (it != active_batches_.end()) {
            completed_batches_.push(std::move(*it));
            active_batches_.erase(it);
        }
    }
}

bool PrivateKeyScanner::process_scan_batch(ScanBatch* batch) {
    if (!batch || batch->key_count == 0) {
        return false;
    }
    
    try {
        // Select CUDA stream for this batch
        cudaStream_t stream = cuda_streams_[batch->batch_id % cuda_streams_.size()];
        
        // Step 1: Generate private keys for batch
        if (!generate_private_keys_for_batch(batch)) {
            std::cerr << "ERROR: Failed to generate private keys for batch " << batch->batch_id << std::endl;
            return false;
        }
        
        // Step 2: Compute corresponding public keys using T037 optimized operations
        if (!compute_public_keys_for_batch(batch)) {
            std::cerr << "ERROR: Failed to compute public keys for batch " << batch->batch_id << std::endl;
            return false;
        }
        
        // Step 3: Generate Bitcoin addresses from public keys
        if (!generate_addresses_for_batch(batch)) {
            std::cerr << "ERROR: Failed to generate addresses for batch " << batch->batch_id << std::endl;
            return false;
        }
        
        // Step 4: Check addresses against target list
        if (!check_addresses_for_batch(batch)) {
            std::cerr << "ERROR: Failed to check addresses for batch " << batch->batch_id << std::endl;
            return false;
        }
        
        // Synchronize stream to ensure completion
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            handle_cuda_error(err, "cudaStreamSynchronize");
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in process_scan_batch: " << e.what() << std::endl;
        return false;
    }
}

bool PrivateKeyScanner::generate_private_keys_for_batch(ScanBatch* batch) {
    cudaStream_t stream = cuda_streams_[batch->batch_id % cuda_streams_.size()];
    
    // Copy start key to device
    ecc::BigInt256* d_start_key;
    cudaError_t err = cudaMalloc(&d_start_key, sizeof(ecc::BigInt256));
    if (err != cudaSuccess) {
        handle_cuda_error(err, "cudaMalloc for start key");
        return false;
    }
    
    err = cudaMemcpyAsync(d_start_key, &batch->start_key, sizeof(ecc::BigInt256), 
                         cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_start_key);
        handle_cuda_error(err, "cudaMemcpy for start key");
        return false;
    }
    
    // Launch kernel to generate private keys
    launch_generate_private_keys_kernel(
        batch->d_private_keys,
        d_start_key,
        batch->key_count,
        1, // stride = 1 for sequential keys
        stream
    );
    
    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_start_key);
        handle_cuda_error(err, "launch_generate_private_keys_kernel");
        return false;
    }
    
    cudaFree(d_start_key);
    return true;
}

bool PrivateKeyScanner::compute_public_keys_for_batch(ScanBatch* batch) {
    cudaStream_t stream = cuda_streams_[batch->batch_id % cuda_streams_.size()];
    
    // Launch kernel to compute public keys using T037 optimized point operations
    launch_compute_public_keys_kernel(
        batch->d_private_keys,
        batch->d_public_keys,
        batch->key_count,
        stream
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        handle_cuda_error(err, "launch_compute_public_keys_kernel");
        return false;
    }
    
    return true;
}

bool PrivateKeyScanner::generate_addresses_for_batch(ScanBatch* batch) {
    cudaStream_t stream = cuda_streams_[batch->batch_id % cuda_streams_.size()];
    
    // Launch kernel to generate Bitcoin addresses
    launch_generate_addresses_kernel(
        batch->d_public_keys,
        batch->d_addresses,
        batch->key_count,
        stream
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        handle_cuda_error(err, "launch_generate_addresses_kernel");
        return false;
    }
    
    return true;
}

bool PrivateKeyScanner::check_addresses_for_batch(ScanBatch* batch) {
    if (target_address_hashes_.empty()) {
        return true; // No targets to check against
    }
    
    cudaStream_t stream = cuda_streams_[batch->batch_id % cuda_streams_.size()];
    
    // Allocate result arrays on device
    bool* d_matches;
    size_t* d_match_indices;
    
    cudaError_t err = cudaMalloc(&d_matches, batch->key_count * sizeof(bool));
    if (err != cudaSuccess) {
        handle_cuda_error(err, "cudaMalloc for matches");
        return false;
    }
    
    err = cudaMalloc(&d_match_indices, batch->key_count * sizeof(size_t));
    if (err != cudaSuccess) {
        cudaFree(d_matches);
        handle_cuda_error(err, "cudaMalloc for match indices");
        return false;
    }
    
    // Copy target addresses to device
    uint8_t* d_targets;
    size_t target_count = target_addresses_.size();
    size_t address_size = 25; // Standard Bitcoin address hash size
    
    err = cudaMalloc(&d_targets, target_count * address_size);
    if (err != cudaSuccess) {
        cudaFree(d_matches);
        cudaFree(d_match_indices);
        handle_cuda_error(err, "cudaMalloc for targets");
        return false;
    }
    
    err = cudaMemcpyAsync(d_targets, target_address_hashes_.data(), 
                         target_count * address_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_matches);
        cudaFree(d_match_indices);
        cudaFree(d_targets);
        handle_cuda_error(err, "cudaMemcpy for targets");
        return false;
    }
    
    // Launch address checking kernel
    launch_check_addresses_kernel(
        batch->d_addresses,
        d_targets,
        d_matches,
        d_match_indices,
        batch->key_count,
        target_count,
        stream
    );
    
    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_matches);
        cudaFree(d_match_indices);
        cudaFree(d_targets);
        handle_cuda_error(err, "launch_check_addresses_kernel");
        return false;
    }
    
    // Copy results back to host
    std::vector<bool> h_matches(batch->key_count);
    std::vector<size_t> h_match_indices(batch->key_count);
    
    err = cudaMemcpyAsync(h_matches.data(), d_matches, batch->key_count * sizeof(bool),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_matches);
        cudaFree(d_match_indices);
        cudaFree(d_targets);
        handle_cuda_error(err, "cudaMemcpy matches back to host");
        return false;
    }
    
    err = cudaMemcpyAsync(h_match_indices.data(), d_match_indices, batch->key_count * sizeof(size_t),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_matches);
        cudaFree(d_match_indices);
        cudaFree(d_targets);
        handle_cuda_error(err, "cudaMemcpy match indices back to host");
        return false;
    }
    
    // Wait for completion
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(d_matches);
        cudaFree(d_match_indices);
        cudaFree(d_targets);
        handle_cuda_error(err, "cudaStreamSynchronize");
        return false;
    }
    
    // Process matches
    for (size_t i = 0; i < batch->key_count; i++) {
        if (h_matches[i]) {
            // Found a match!
            ScanMatch match;
            match.private_key = batch->start_key + ecc::BigInt256(i);
            match.address = target_addresses_[h_match_indices[i]];
            match.found_time = std::chrono::system_clock::now();
            match.batch_id = batch->batch_id;
            match.device_id = device_id_;
            
            // Store match
            {
                std::lock_guard<std::mutex> lock(matches_mutex_);
                found_matches_.push_back(match);
            }
            
            // Call match callback if set
            if (match_callback_) {
                match_callback_(match);
            }
            
            batch->found_match = true;
            batch->matched_private_key = match.private_key;
            batch->matched_address = match.address;
            
            std::cout << "🎉 MATCH FOUND! 🎉" << std::endl;
            std::cout << "  Private Key: " << match.private_key.to_hex() << std::endl;
            std::cout << "  Address: " << match.address << std::endl;
            std::cout << "  Batch ID: " << match.batch_id << std::endl;
        }
    }
    
    // Clean up device memory
    cudaFree(d_matches);
    cudaFree(d_match_indices);
    cudaFree(d_targets);
    
    return true;
}

std::unique_ptr<ScanBatch> PrivateKeyScanner::create_scan_batch(
    const ecc::BigInt256& start_key,
    size_t key_count,
    size_t batch_id) {
    
    auto batch = std::make_unique<ScanBatch>();
    batch->batch_id = batch_id;
    batch->device_id = device_id_;
    batch->start_key = start_key;
    batch->end_key = start_key + ecc::BigInt256(key_count);
    batch->key_count = key_count;
    batch->state = ScanBatch::BatchState::PENDING;
    
    return batch;
}

bool PrivateKeyScanner::allocate_batch_memory(ScanBatch* batch) {
    if (!batch || batch->key_count == 0) {
        return false;
    }
    
    cudaError_t err;
    size_t address_size = 25; // Bitcoin address hash size
    
    // Allocate private key array
    err = cudaMalloc(&batch->d_private_keys, batch->key_count * sizeof(ecc::BigInt256));
    if (err != cudaSuccess) {
        handle_cuda_error(err, "cudaMalloc for private keys");
        return false;
    }
    
    // Allocate public key array
    err = cudaMalloc(&batch->d_public_keys, batch->key_count * sizeof(ecc::Point));
    if (err != cudaSuccess) {
        cudaFree(batch->d_private_keys);
        handle_cuda_error(err, "cudaMalloc for public keys");
        return false;
    }
    
    // Allocate address array
    err = cudaMalloc(&batch->d_addresses, batch->key_count * address_size);
    if (err != cudaSuccess) {
        cudaFree(batch->d_private_keys);
        cudaFree(batch->d_public_keys);
        handle_cuda_error(err, "cudaMalloc for addresses");
        return false;
    }
    
    return true;
}

void PrivateKeyScanner::deallocate_batch_memory(ScanBatch* batch) {
    if (!batch) return;
    
    if (batch->d_private_keys) {
        cudaFree(batch->d_private_keys);
        batch->d_private_keys = nullptr;
    }
    
    if (batch->d_public_keys) {
        cudaFree(batch->d_public_keys);
        batch->d_public_keys = nullptr;
    }
    
    if (batch->d_addresses) {
        cudaFree(batch->d_addresses);
        batch->d_addresses = nullptr;
    }
}

ecc::BigInt256 PrivateKeyScanner::get_next_batch_start_key() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    ecc::BigInt256 start_key = config_.current_key;
    advance_current_position(config_.keys_per_batch);
    
    return start_key;
}

void PrivateKeyScanner::advance_current_position(size_t key_count) {
    config_.current_key = config_.current_key + ecc::BigInt256(key_count);
}

bool PrivateKeyScanner::is_range_completed() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_.current_key >= config_.end_key;
}

void PrivateKeyScanner::update_scanning_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - scan_start_time_);
    
    metrics_.elapsed_time = elapsed;
    
    // Update keys scanned
    {
        std::lock_guard<std::mutex> config_lock(config_mutex_);
        metrics_.keys_scanned = config_.current_key - config_.start_key;
        metrics_.keys_remaining = config_.end_key - config_.current_key;
        metrics_.progress_percentage = get_progress_percentage();
    }
    
    // Calculate throughput
    if (elapsed.count() > 0) {
        double seconds = elapsed.count() / 1000.0;
        metrics_.keys_per_second = static_cast<double>(metrics_.keys_scanned.to_uint64()) / seconds;
        
        // Update average
        throughput_history_.push_back(metrics_.keys_per_second);
        if (throughput_history_.size() > 100) { // Keep last 100 measurements
            throughput_history_.erase(throughput_history_.begin());
        }
        
        double sum = std::accumulate(throughput_history_.begin(), throughput_history_.end(), 0.0);
        metrics_.average_keys_per_second = sum / throughput_history_.size();
        metrics_.peak_keys_per_second = *std::max_element(throughput_history_.begin(), throughput_history_.end());
    }
    
    // Call progress callback if set
    if (progress_callback_) {
        progress_callback_(metrics_);
    }
}

void PrivateKeyScanner::apply_bitcrack_optimizations() {
    // Load BitCrack concepts from T040 analysis
    bitcrack_analysis::BitCrackAnalysisFramework analyzer;
    if (analyzer.initialize()) {
        bitcrack_concepts_ = analyzer.analyze_core_concepts();
        
        // Apply thread management optimizations
        config_.threads_per_block = bitcrack_concepts_.thread_mgmt.threads_per_block;
        config_.blocks_per_grid = bitcrack_concepts_.thread_mgmt.blocks_per_grid;
        
        // Apply range partitioning optimizations
        if (bitcrack_concepts_.range_partition.partitioning_strategy == 
            bitcrack_analysis::BitCrackScanningConcepts::RangePartitioning::Strategy::BLOCK_BASED) {
            config_.keys_per_batch = bitcrack_concepts_.range_partition.partition_size;
        }
        
        // Apply memory optimization settings
        config_.enable_coalesced_access = bitcrack_concepts_.memory_opt.use_coalesced_access;
        config_.use_shared_memory = bitcrack_concepts_.memory_opt.enable_shared_memory;
        
        std::cout << "Applied BitCrack optimizations:" << std::endl;
        std::cout << "  Threads per block: " << config_.threads_per_block << std::endl;
        std::cout << "  Blocks per grid: " << config_.blocks_per_grid << std::endl;
        std::cout << "  Keys per batch: " << config_.keys_per_batch << std::endl;
        std::cout << "  Coalesced access: " << (config_.enable_coalesced_access ? "Enabled" : "Disabled") << std::endl;
    }
}

bool PrivateKeyScanner::validate_configuration() const {
    // Validate basic parameters
    if (config_.keys_per_batch == 0 || config_.keys_per_batch > 100000000) {
        std::cerr << "ERROR: Invalid keys_per_batch: " << config_.keys_per_batch << std::endl;
        return false;
    }
    
    if (config_.threads_per_block == 0 || config_.threads_per_block > 1024) {
        std::cerr << "ERROR: Invalid threads_per_block: " << config_.threads_per_block << std::endl;
        return false;
    }
    
    if (config_.blocks_per_grid == 0 || config_.blocks_per_grid > 65535) {
        std::cerr << "ERROR: Invalid blocks_per_grid: " << config_.blocks_per_grid << std::endl;
        return false;
    }
    
    if (config_.cuda_streams == 0 || config_.cuda_streams > 32) {
        std::cerr << "ERROR: Invalid cuda_streams: " << config_.cuda_streams << std::endl;
        return false;
    }
    
    return true;
}

bool PrivateKeyScanner::validate_private_key_range(const models::PrivateKeyRange& range) const {
    if (range.end_key <= range.start_key) {
        std::cerr << "ERROR: Invalid range: end_key must be greater than start_key" << std::endl;
        return false;
    }
    
    // Check for valid secp256k1 range
    ecc::BigInt256 secp256k1_order;
    // secp256k1 curve order: 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    secp256k1_order.from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    
    if (range.start_key.is_zero() || range.start_key >= secp256k1_order) {
        std::cerr << "ERROR: start_key out of valid secp256k1 range" << std::endl;
        return false;
    }
    
    if (range.end_key >= secp256k1_order) {
        std::cerr << "ERROR: end_key out of valid secp256k1 range" << std::endl;
        return false;
    }
    
    return true;
}

void PrivateKeyScanner::handle_cuda_error(cudaError_t error, const std::string& operation) {
    std::cerr << "CUDA Error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.kernel_errors++;
}

void PrivateKeyScanner::handle_batch_error(ScanBatch* batch, const std::string& error_message) {
    std::cerr << "Batch " << batch->batch_id << " error: " << error_message << std::endl;
    
    batch->state = ScanBatch::BatchState::FAILED;
    batch->end_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.validation_errors++;
}

std::vector<uint8_t> PrivateKeyScanner::convert_addresses_to_binary(const std::vector<std::string>& addresses) const {
    std::vector<uint8_t> binary_data;
    size_t address_size = 25; // Standard Bitcoin address hash size
    
    binary_data.reserve(addresses.size() * address_size);
    
    for (const auto& address : addresses) {
        auto hash160 = scanning_utils::AddressConverter::extract_hash160(address);
        if (hash160.size() == 20) {
            // Pad to 25 bytes (20 + 5 for type and checksum info)
            binary_data.insert(binary_data.end(), hash160.begin(), hash160.end());
            binary_data.insert(binary_data.end(), 5, 0); // Padding
        }
    }
    
    return binary_data;
}

// ScanningFrameworkFactory implementation
std::unique_ptr<PrivateKeyScanner> ScanningFrameworkFactory::create_scanner(
    ScanningStrategy strategy,
    const ScanningConfiguration& base_config) {
    
    auto scanner = std::make_unique<PrivateKeyScanner>();
    ScanningConfiguration config = base_config;
    
    // Apply strategy-specific optimizations
    switch (strategy) {
        case ScanningStrategy::BITCRACK_OPTIMIZED:
            // BitCrack optimizations will be applied automatically in initialize()
            config.enable_coalesced_access = true;
            config.use_shared_memory = true;
            config.enable_async_execution = true;
            break;
            
        case ScanningStrategy::ADAPTIVE_BATCHING:
            config.enable_occupancy_optimization = true;
            config.target_gpu_utilization = 0.95;
            break;
            
        case ScanningStrategy::LINEAR_SEQUENTIAL:
            config.keys_per_batch = 1024 * 256; // Smaller batches for sequential
            config.cuda_streams = 2; // Fewer streams
            break;
            
        default:
            break;
    }
    
    if (!scanner->initialize(config)) {
        return nullptr;
    }
    
    return scanner;
}

// Utility implementations
namespace scanning_utils {

std::vector<models::PrivateKeyRange> RangeSubdivider::subdivide_range(
    const models::PrivateKeyRange& range,
    size_t subdivision_count) {
    
    std::vector<models::PrivateKeyRange> subdivisions;
    
    if (subdivision_count == 0 || range.end_key <= range.start_key) {
        return subdivisions;
    }
    
    ecc::BigInt256 total_range = range.end_key - range.start_key;
    ecc::BigInt256 subdivision_size = total_range / ecc::BigInt256(subdivision_count);
    
    for (size_t i = 0; i < subdivision_count; i++) {
        models::PrivateKeyRange sub_range;
        sub_range.start_key = range.start_key + (subdivision_size * ecc::BigInt256(i));
        sub_range.end_key = (i == subdivision_count - 1) ? 
                           range.end_key : 
                           (sub_range.start_key + subdivision_size);
        sub_range.name = "Subdivision_" + std::to_string(i);
        
        subdivisions.push_back(sub_range);
    }
    
    return subdivisions;
}

bool AddressConverter::is_valid_bitcoin_address(const std::string& address) {
    // Basic validation - starts with 1, 3, or bc1
    if (address.empty()) return false;
    
    char first_char = address[0];
    return (first_char == '1' || first_char == '3' || 
            (address.length() > 3 && address.substr(0, 3) == "bc1"));
}

std::vector<uint8_t> AddressConverter::extract_hash160(const std::string& address) {
    // Simplified implementation - would need proper Base58 decoding
    std::vector<uint8_t> hash160(20, 0);
    
    // For testing purposes, generate deterministic hash from address string
    std::hash<std::string> hasher;
    size_t hash_value = hasher(address);
    
    for (int i = 0; i < 20; i++) {
        hash160[i] = static_cast<uint8_t>((hash_value >> (i * 8)) & 0xFF);
    }
    
    return hash160;
}

} // namespace scanning_utils

} // namespace scan
} // namespace keyhunt