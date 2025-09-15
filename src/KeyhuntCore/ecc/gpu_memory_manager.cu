/**
 * @file gpu_memory_manager.cu
 * @brief Implementation of GPU memory management and optimization
 * @author KeyhuntCUDA Team
 * 
 * T035: Implement GPU memory management and optimization for secp256k1 operations
 * 
 * Provides comprehensive GPU memory management with memory pooling, batch optimization,
 * CUDA stream management, and performance monitoring for secp256k1 operations.
 */

#include "gpu_memory_manager.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <sstream>

namespace keyhunt {
namespace ecc {
namespace gpu {

// ManagedStream Implementation
ManagedStream::ManagedStream(int priority) : stream_(nullptr), start_event_(nullptr), 
                                           end_event_(nullptr), operation_count_(0), total_time_(0) {
    cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority);
    cudaEventCreate(&start_event_);
    cudaEventCreate(&end_event_);
}

ManagedStream::~ManagedStream() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
    if (start_event_) cudaEventDestroy(start_event_);
    if (end_event_) cudaEventDestroy(end_event_);
}

bool ManagedStream::is_ready() const {
    return cudaStreamQuery(stream_) == cudaSuccess;
}

void ManagedStream::synchronize() {
    cudaStreamSynchronize(stream_);
}

void ManagedStream::record_operation_start() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    last_start_ = std::chrono::steady_clock::now();
    cudaEventRecord(start_event_, stream_);
}

void ManagedStream::record_operation_end() {
    cudaEventRecord(end_event_, stream_);
    cudaEventSynchronize(end_event_);
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - last_start_);
    
    total_time_ += duration;
    operation_count_++;
}

double ManagedStream::get_utilization() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    auto current_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_start_);
    
    if (total_elapsed.count() == 0) return 0.0;
    return static_cast<double>(total_time_.count()) / total_elapsed.count();
}

size_t ManagedStream::get_operation_count() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return operation_count_;
}

// GPUMemoryPool Implementation
GPUMemoryPool::GPUMemoryPool(const MemoryPoolConfig& config, int device_id) 
    : config_(config), device_id_(device_id), initialized_(false), pool_ptr_(nullptr),
      pool_size_(0), next_alloc_id_(1) {
}

GPUMemoryPool::~GPUMemoryPool() {
    cleanup();
}

bool GPUMemoryPool::initialize() {
    if (initialized_) return true;
    
    // Set device
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) return false;
    
    // Allocate initial pool
    pool_size_ = config_.initial_size;
    
    switch (config_.strategy) {
        case MemoryStrategy::PINNED:
            err = cudaMallocHost(&pool_ptr_, pool_size_);
            break;
        case MemoryStrategy::MANAGED:
            err = cudaMallocManaged(&pool_ptr_, pool_size_);
            break;
        case MemoryStrategy::UNIFIED:
            err = cudaMallocManaged(&pool_ptr_, pool_size_, cudaMemAttachGlobal);
            break;
        default:
            err = cudaMalloc(&pool_ptr_, pool_size_);
            break;
    }
    
    if (err != cudaSuccess) {
        return false;
    }
    
    // Initialize with single large free block
    blocks_.emplace_back(pool_ptr_, pool_size_, 0, config_.strategy);
    blocks_[0].is_free = true;
    
    initialized_ = true;
    update_stats();
    
    return true;
}

void GPUMemoryPool::cleanup() {
    if (!initialized_) return;
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (pool_ptr_) {
        switch (config_.strategy) {
            case MemoryStrategy::PINNED:
                cudaFreeHost(pool_ptr_);
                break;
            default:
                cudaFree(pool_ptr_);
                break;
        }
        pool_ptr_ = nullptr;
    }
    
    blocks_.clear();
    allocation_map_.clear();
    pool_size_ = 0;
    initialized_ = false;
}

void* GPUMemoryPool::allocate(size_t size, MemoryStrategy strategy) {
    if (!initialized_) return nullptr;
    
    // Align size to configured alignment
    size = (size + config_.alignment - 1) & ~(config_.alignment - 1);
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    auto start_time = std::chrono::steady_clock::now();
    
    void* ptr = allocate_from_pool(size, strategy);
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    stats_.total_alloc_time += duration;
    if (ptr) {
        stats_.num_allocations++;
        stats_.current_usage += size;
        if (stats_.current_usage > stats_.peak_usage) {
            stats_.peak_usage = stats_.current_usage;
        }
        allocation_map_[ptr] = size;
    }
    
    update_stats();
    return ptr;
}

void GPUMemoryPool::deallocate(void* ptr) {
    if (!ptr || !initialized_) return;
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    auto start_time = std::chrono::steady_clock::now();
    
    auto it = allocation_map_.find(ptr);
    if (it == allocation_map_.end()) return; // Invalid pointer
    
    size_t size = it->second;
    allocation_map_.erase(it);
    
    // Mark block as free
    for (auto& block : blocks_) {
        if (block.ptr == ptr) {
            block.is_free = true;
            break;
        }
    }
    
    // Merge adjacent free blocks if enabled
    if (config_.enable_defrag) {
        merge_free_blocks();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    stats_.total_free_time += duration;
    stats_.num_deallocations++;
    stats_.current_usage -= size;
    
    update_stats();
}

std::vector<void*> GPUMemoryPool::allocate_batch(const std::vector<size_t>& sizes, MemoryStrategy strategy) {
    std::vector<void*> ptrs;
    ptrs.reserve(sizes.size());
    
    for (size_t size : sizes) {
        ptrs.push_back(allocate(size, strategy));
    }
    
    return ptrs;
}

void GPUMemoryPool::deallocate_batch(const std::vector<void*>& ptrs) {
    for (void* ptr : ptrs) {
        deallocate(ptr);
    }
}

void* GPUMemoryPool::allocate_from_pool(size_t size, MemoryStrategy strategy) {
    // Find suitable free block
    for (auto& block : blocks_) {
        if (block.is_free && block.size >= size) {
            // Split block if necessary
            if (block.size > size + config_.block_size) {
                // Create new block for remainder
                size_t remaining_size = block.size - size;
                size_t new_offset = block.offset + size;
                void* new_ptr = static_cast<char*>(block.ptr) + size;
                
                blocks_.emplace_back(new_ptr, remaining_size, new_offset, strategy);
                blocks_.back().is_free = true;
                
                // Update current block
                block.size = size;
            }
            
            block.is_free = false;
            block.strategy = strategy;
            block.alloc_time = std::chrono::steady_clock::now();
            block.alloc_id = next_alloc_id_.fetch_add(1);
            
            return block.ptr;
        }
    }
    
    // No suitable block found, try to expand pool
    if (expand_pool()) {
        return allocate_from_pool(size, strategy);
    }
    
    return nullptr; // Allocation failed
}

bool GPUMemoryPool::expand_pool() {
    if (pool_size_ >= config_.max_size) return false;
    
    size_t new_size = static_cast<size_t>(pool_size_ * config_.growth_factor);
    new_size = std::min(new_size, config_.max_size);
    
    void* new_pool = nullptr;
    cudaError_t err;
    
    switch (config_.strategy) {
        case MemoryStrategy::PINNED:
            err = cudaMallocHost(&new_pool, new_size);
            break;
        case MemoryStrategy::MANAGED:
            err = cudaMallocManaged(&new_pool, new_size);
            break;
        default:
            err = cudaMalloc(&new_pool, new_size);
            break;
    }
    
    if (err != cudaSuccess) return false;
    
    // Copy existing data
    cudaMemcpy(new_pool, pool_ptr_, pool_size_, cudaMemcpyDeviceToDevice);
    
    // Free old pool
    switch (config_.strategy) {
        case MemoryStrategy::PINNED:
            cudaFreeHost(pool_ptr_);
            break;
        default:
            cudaFree(pool_ptr_);
            break;
    }
    
    // Update block pointers
    char* offset_diff = static_cast<char*>(new_pool) - static_cast<char*>(pool_ptr_);
    for (auto& block : blocks_) {
        block.ptr = static_cast<char*>(block.ptr) + offset_diff;
    }
    
    // Update allocation map
    std::map<void*, size_t> new_allocation_map;
    for (const auto& pair : allocation_map_) {
        void* new_ptr = static_cast<char*>(pair.first) + offset_diff;
        new_allocation_map[new_ptr] = pair.second;
    }
    allocation_map_ = std::move(new_allocation_map);
    
    // Add new free block for expanded space
    size_t expansion_size = new_size - pool_size_;
    void* expansion_ptr = static_cast<char*>(new_pool) + pool_size_;
    blocks_.emplace_back(expansion_ptr, expansion_size, pool_size_, config_.strategy);
    blocks_.back().is_free = true;
    
    pool_ptr_ = new_pool;
    pool_size_ = new_size;
    
    return true;
}

void GPUMemoryPool::merge_free_blocks() {
    // Sort blocks by offset
    std::sort(blocks_.begin(), blocks_.end(), 
              [](const MemoryBlock& a, const MemoryBlock& b) {
                  return a.offset < b.offset;
              });
    
    // Merge adjacent free blocks
    for (size_t i = 0; i < blocks_.size() - 1; ) {
        if (blocks_[i].is_free && blocks_[i + 1].is_free && 
            blocks_[i].offset + blocks_[i].size == blocks_[i + 1].offset) {
            
            // Merge blocks
            blocks_[i].size += blocks_[i + 1].size;
            blocks_.erase(blocks_.begin() + i + 1);
        } else {
            i++;
        }
    }
}

void GPUMemoryPool::defragment() {
    if (!initialized_ || !config_.enable_defrag) return;
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    merge_free_blocks();
}

void GPUMemoryPool::reset() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Mark all blocks as free
    for (auto& block : blocks_) {
        block.is_free = true;
    }
    
    allocation_map_.clear();
    merge_free_blocks();
    
    stats_.current_usage = 0;
    stats_.num_allocations = 0;
}

double GPUMemoryPool::get_fragmentation_ratio() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (pool_size_ == 0) return 0.0;
    
    size_t free_memory = 0;
    size_t largest_free_block = 0;
    
    for (const auto& block : blocks_) {
        if (block.is_free) {
            free_memory += block.size;
            largest_free_block = std::max(largest_free_block, block.size);
        }
    }
    
    if (free_memory == 0) return 0.0;
    return 1.0 - (static_cast<double>(largest_free_block) / free_memory);
}

MemoryStats GPUMemoryPool::get_stats() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return stats_;
}

void GPUMemoryPool::update_stats() {
    stats_.total_allocated = pool_size_;
    stats_.total_free = pool_size_ - stats_.current_usage;
    stats_.fragmentation_ratio = static_cast<size_t>(get_fragmentation_ratio() * 100);
    
    if (stats_.total_allocated > 0) {
        stats_.allocation_efficiency = static_cast<double>(stats_.current_usage) / stats_.total_allocated;
    }
}

// BatchOptimizer Implementation
BatchOptimizer::BatchOptimizer(const BatchConfig& config, GPUMemoryPool* memory_pool)
    : config_(config), memory_pool_(memory_pool), initialized_(false) {
}

BatchOptimizer::~BatchOptimizer() {
    cleanup();
}

bool BatchOptimizer::initialize() {
    if (initialized_) return true;
    
    // Create managed streams
    streams_.reserve(config_.num_streams);
    for (int i = 0; i < config_.num_streams; ++i) {
        streams_.push_back(std::make_unique<ManagedStream>(i)); // Higher priority for lower indices
        available_streams_.push(streams_[i].get());
    }
    
    initialized_ = true;
    last_update_ = std::chrono::steady_clock::now();
    
    return true;
}

void BatchOptimizer::cleanup() {
    if (!initialized_) return;
    
    std::lock_guard<std::mutex> lock(stream_mutex_);
    
    // Synchronize all streams
    for (auto& stream : streams_) {
        stream->synchronize();
    }
    
    streams_.clear();
    while (!available_streams_.empty()) {
        available_streams_.pop();
    }
    
    initialized_ = false;
}

size_t BatchOptimizer::calculate_optimal_batch_size(size_t total_operations, size_t element_size) {
    if (!initialized_) return config_.min_batch_size;
    
    // Calculate memory constraint
    size_t available_memory = memory_pool_->get_available_memory();
    size_t max_elements_by_memory = available_memory / element_size;
    
    // Start with configured optimal size
    size_t optimal_size = config_.optimal_batch_size;
    
    // Adjust based on memory constraints
    optimal_size = std::min(optimal_size, max_elements_by_memory);
    optimal_size = std::min(optimal_size, config_.max_batch_size);
    optimal_size = std::max(optimal_size, config_.min_batch_size);
    
    // Adjust based on total operations
    if (total_operations < optimal_size) {
        optimal_size = total_operations;
    }
    
    return optimal_size;
}

std::vector<size_t> BatchOptimizer::split_into_batches(size_t total_operations, size_t max_memory) {
    std::vector<size_t> batch_sizes;
    
    size_t element_size = sizeof(BigInt256); // Estimate for secp256k1 scalar
    size_t max_elements_per_batch = max_memory / element_size;
    
    size_t remaining = total_operations;
    while (remaining > 0) {
        size_t batch_size = calculate_optimal_batch_size(remaining, element_size);
        batch_size = std::min(batch_size, max_elements_per_batch);
        batch_size = std::min(batch_size, remaining);
        
        batch_sizes.push_back(batch_size);
        remaining -= batch_size;
    }
    
    return batch_sizes;
}

ManagedStream* BatchOptimizer::get_available_stream() {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    
    if (available_streams_.empty()) {
        return nullptr; // No streams available
    }
    
    ManagedStream* stream = available_streams_.front();
    available_streams_.pop();
    return stream;
}

void BatchOptimizer::return_stream(ManagedStream* stream) {
    if (!stream) return;
    
    std::lock_guard<std::mutex> lock(stream_mutex_);
    available_streams_.push(stream);
}

BatchOptimizer::BatchMetrics BatchOptimizer::get_metrics() const {
    // Calculate current metrics based on stream performance
    BatchMetrics metrics = current_metrics_;
    
    if (initialized_) {
        size_t total_operations = 0;
        double total_utilization = 0.0;
        
        for (const auto& stream : streams_) {
            total_operations += stream->get_operation_count();
            total_utilization += stream->get_utilization();
        }
        
        if (!streams_.empty()) {
            metrics.gpu_utilization = total_utilization / streams_.size();
        }
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_update_);
        if (elapsed.count() > 0) {
            metrics.operations_per_second = total_operations / elapsed.count();
        }
    }
    
    return metrics;
}

// Layout implementations
size_t layouts::ScalarArray::calculate_size(size_t scalar_count) {
    return scalar_count * 4 * sizeof(uint64_t); // 4 x uint64_t per scalar
}

void* layouts::ScalarArray::allocate_optimized(GPUMemoryPool* pool, size_t scalar_count) {
    size_t size = calculate_size(scalar_count);
    return pool->allocate(size, MemoryStrategy::COALESCED);
}

size_t layouts::PointArray::calculate_size(size_t point_count) {
    return point_count * 3 * 4 * sizeof(uint64_t); // 3 coordinates x 4 uint64_t per coordinate
}

void* layouts::PointArray::allocate_optimized(GPUMemoryPool* pool, size_t point_count) {
    size_t size = calculate_size(point_count);
    return pool->allocate(size, MemoryStrategy::COALESCED);
}

size_t layouts::PrecomputedTable::calculate_size(size_t window_size) {
    size_t table_entries = (1 << window_size) - 1; // 2^w - 1 precomputed points
    return table_entries * 2 * 4 * sizeof(uint64_t); // 2 coordinates x 4 uint64_t per coordinate
}

void* layouts::PrecomputedTable::allocate_optimized(GPUMemoryPool* pool, size_t window_size) {
    size_t size = calculate_size(window_size);
    return pool->allocate(size, MemoryStrategy::COALESCED);
}

// Global registry implementation
std::unordered_map<int, std::unique_ptr<Secp256k1MemoryManager>> MemoryManagerRegistry::managers_;
Secp256k1MemoryManager::Config MemoryManagerRegistry::global_config_;
std::mutex MemoryManagerRegistry::registry_mutex_;

Secp256k1MemoryManager* MemoryManagerRegistry::get_manager(int device_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = managers_.find(device_id);
    if (it != managers_.end()) {
        return it->second.get();
    }
    
    // Create new manager with global config
    auto manager = std::make_unique<Secp256k1MemoryManager>(global_config_);
    manager->initialize();
    
    Secp256k1MemoryManager* ptr = manager.get();
    managers_[device_id] = std::move(manager);
    
    return ptr;
}

} // namespace gpu
} // namespace ecc
} // namespace keyhunt