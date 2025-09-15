/**
 * @file secp256k1_memory_manager.cpp
 * @brief High-level secp256k1 memory manager implementation
 * @author KeyhuntCUDA Team
 * 
 * T035: Implement GPU memory management and optimization for secp256k1 operations
 * 
 * Provides high-level memory management interface for secp256k1 operations
 * with optimized layouts, batch processing, and performance monitoring.
 */

#include "gpu_memory_manager.h"
#include <algorithm>
#include <cstring>

namespace keyhunt {
namespace ecc {
namespace gpu {

// Secp256k1MemoryManager Implementation
Secp256k1MemoryManager::Secp256k1MemoryManager(const Config& config) 
    : config_(config), initialized_(false), total_device_memory_(0), available_device_memory_(0) {
}

Secp256k1MemoryManager::~Secp256k1MemoryManager() {
    cleanup();
}

bool Secp256k1MemoryManager::initialize() {
    if (initialized_) return true;
    
    // Set device and query properties
    cudaError_t err = cudaSetDevice(config_.device_id);
    if (err != cudaSuccess) return false;
    
    if (!query_device_properties()) return false;
    
    // Initialize memory pool
    memory_pool_ = std::make_unique<GPUMemoryPool>(config_.pool_config, config_.device_id);
    if (!memory_pool_->initialize()) return false;
    
    // Initialize batch optimizer
    batch_optimizer_ = std::make_unique<BatchOptimizer>(config_.batch_config, memory_pool_.get());
    if (!batch_optimizer_->initialize()) return false;
    
    // Create compute streams
    const int num_compute_streams = 2;
    compute_streams_.reserve(num_compute_streams);
    for (int i = 0; i < num_compute_streams; ++i) {
        compute_streams_.push_back(std::make_unique<ManagedStream>(i));
    }
    
    // Create transfer streams
    const int num_transfer_streams = 2;
    transfer_streams_.reserve(num_transfer_streams);
    for (int i = 0; i < num_transfer_streams; ++i) {
        transfer_streams_.push_back(std::make_unique<ManagedStream>(num_compute_streams + i));
    }
    
    // Set up peer access if enabled
    if (config_.enable_peer_access) {
        setup_peer_access();
    }
    
    initialized_ = true;
    last_report_time_ = std::chrono::steady_clock::now();
    
    return true;
}

void Secp256k1MemoryManager::cleanup() {
    if (!initialized_) return;
    
    // Synchronize all streams
    for (auto& stream : compute_streams_) {
        stream->synchronize();
    }
    for (auto& stream : transfer_streams_) {
        stream->synchronize();
    }
    
    // Cleanup components
    if (batch_optimizer_) {
        batch_optimizer_->cleanup();
        batch_optimizer_.reset();
    }
    
    if (memory_pool_) {
        memory_pool_->cleanup();
        memory_pool_.reset();
    }
    
    compute_streams_.clear();
    transfer_streams_.clear();
    
    initialized_ = false;
}

layouts::ScalarArray Secp256k1MemoryManager::allocate_scalars(size_t count) {
    if (!initialized_) return layouts::ScalarArray{nullptr, 0, 0};
    
    size_t size = layouts::ScalarArray::calculate_size(count);
    void* ptr = memory_pool_->allocate(size, MemoryStrategy::COALESCED);
    
    if (!ptr) return layouts::ScalarArray{nullptr, 0, 0};
    
    layouts::ScalarArray array;
    array.data = static_cast<uint64_t*>(ptr);
    array.count = count;
    array.stride = 4 * sizeof(uint64_t); // 4 x uint64_t per scalar
    
    return array;
}

layouts::PointArray Secp256k1MemoryManager::allocate_points(size_t count) {
    if (!initialized_) return layouts::PointArray{nullptr, nullptr, nullptr, 0, 0};
    
    size_t size = layouts::PointArray::calculate_size(count);
    void* ptr = memory_pool_->allocate(size, MemoryStrategy::COALESCED);
    
    if (!ptr) return layouts::PointArray{nullptr, nullptr, nullptr, 0, 0};
    
    layouts::PointArray array;
    uint64_t* base_ptr = static_cast<uint64_t*>(ptr);
    array.x_coords = base_ptr;
    array.y_coords = base_ptr + count * 4; // Offset by count * 4 uint64_t
    array.z_coords = base_ptr + count * 8; // Offset by count * 8 uint64_t
    array.count = count;
    array.stride = 4 * sizeof(uint64_t); // 4 x uint64_t per coordinate
    
    return array;
}

layouts::PrecomputedTable Secp256k1MemoryManager::allocate_precomputed_table(size_t window_size) {
    if (!initialized_) return layouts::PrecomputedTable{nullptr, 0, 0};
    
    size_t size = layouts::PrecomputedTable::calculate_size(window_size);
    void* ptr = memory_pool_->allocate(size, MemoryStrategy::COALESCED);
    
    if (!ptr) return layouts::PrecomputedTable{nullptr, 0, 0};
    
    layouts::PrecomputedTable table;
    table.table_data = static_cast<uint64_t*>(ptr);
    table.table_size = size;
    table.window_size = window_size;
    
    return table;
}

void Secp256k1MemoryManager::deallocate_scalars(const layouts::ScalarArray& array) {
    if (array.data) {
        memory_pool_->deallocate(array.data);
    }
}

void Secp256k1MemoryManager::deallocate_points(const layouts::PointArray& array) {
    if (array.x_coords) {
        memory_pool_->deallocate(array.x_coords);
    }
}

void Secp256k1MemoryManager::deallocate_precomputed_table(const layouts::PrecomputedTable& table) {
    if (table.table_data) {
        memory_pool_->deallocate(table.table_data);
    }
}

std::vector<size_t> Secp256k1MemoryManager::optimize_batch_sizes(size_t total_scalars, size_t available_memory) {
    if (!initialized_) return {total_scalars};
    
    return batch_optimizer_->split_into_batches(total_scalars, available_memory);
}

ManagedStream* Secp256k1MemoryManager::get_compute_stream() {
    if (!initialized_ || compute_streams_.empty()) return nullptr;
    
    std::lock_guard<std::mutex> lock(stream_mutex_);
    
    // Find available compute stream
    for (auto& stream : compute_streams_) {
        if (stream->is_ready()) {
            return stream.get();
        }
    }
    
    // Return first stream if none are ready (will block)
    return compute_streams_[0].get();
}

ManagedStream* Secp256k1MemoryManager::get_transfer_stream() {
    if (!initialized_ || transfer_streams_.empty()) return nullptr;
    
    std::lock_guard<std::mutex> lock(stream_mutex_);
    
    // Find available transfer stream
    for (auto& stream : transfer_streams_) {
        if (stream->is_ready()) {
            return stream.get();
        }
    }
    
    // Return first stream if none are ready (will block)
    return transfer_streams_[0].get();
}

cudaError_t Secp256k1MemoryManager::copy_scalars_to_device(const std::vector<BigInt256>& host_scalars,
                                                          const layouts::ScalarArray& device_array,
                                                          ManagedStream* stream) {
    if (!initialized_ || host_scalars.empty() || !device_array.data) {
        return cudaErrorInvalidValue;
    }
    
    if (host_scalars.size() > device_array.count) {
        return cudaErrorInvalidValue; // Not enough space in device array
    }
    
    // Copy scalar data with proper memory layout
    size_t copy_size = host_scalars.size() * 4 * sizeof(uint64_t);
    
    cudaStream_t cuda_stream = stream ? stream->get_stream() : cudaStreamDefault;
    
    if (stream) stream->record_operation_start();
    
    cudaError_t err = cudaMemcpyAsync(device_array.data, host_scalars.data(), copy_size,
                                     cudaMemcpyHostToDevice, cuda_stream);
    
    if (stream) stream->record_operation_end();
    
    return err;
}

cudaError_t Secp256k1MemoryManager::copy_points_from_device(const layouts::PointArray& device_array,
                                                           std::vector<Point>& host_points,
                                                           ManagedStream* stream) {
    if (!initialized_ || !device_array.x_coords || device_array.count == 0) {
        return cudaErrorInvalidValue;
    }
    
    host_points.resize(device_array.count);
    
    cudaStream_t cuda_stream = stream ? stream->get_stream() : cudaStreamDefault;
    
    if (stream) stream->record_operation_start();
    
    // Copy x coordinates
    size_t coord_size = device_array.count * 4 * sizeof(uint64_t);
    
    // Temporary host buffer for device data
    std::vector<uint64_t> temp_coords(device_array.count * 4 * 3); // x, y, z coordinates
    
    cudaError_t err = cudaMemcpyAsync(temp_coords.data(), device_array.x_coords, coord_size,
                                     cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) {
        if (stream) stream->record_operation_end();
        return err;
    }
    
    err = cudaMemcpyAsync(temp_coords.data() + device_array.count * 4, device_array.y_coords, coord_size,
                         cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) {
        if (stream) stream->record_operation_end();
        return err;
    }
    
    err = cudaMemcpyAsync(temp_coords.data() + device_array.count * 8, device_array.z_coords, coord_size,
                         cudaMemcpyDeviceToHost, cuda_stream);
    
    if (err == cudaSuccess) {
        // Synchronize to ensure data is available
        cudaStreamSynchronize(cuda_stream);
        
        // Convert to Point structures
        for (size_t i = 0; i < device_array.count; ++i) {
            Point& point = host_points[i];
            
            // Copy x coordinate
            std::memcpy(point.x.d, &temp_coords[i * 4], 4 * sizeof(uint64_t));
            
            // Copy y coordinate  
            std::memcpy(point.y.d, &temp_coords[(device_array.count + i) * 4], 4 * sizeof(uint64_t));
            
            // Copy z coordinate
            std::memcpy(point.z.d, &temp_coords[(device_array.count * 2 + i) * 4], 4 * sizeof(uint64_t));
            
            // Set infinity flag
            point.is_infinity = point.z.is_zero();
        }
    }
    
    if (stream) stream->record_operation_end();
    
    return err;
}

Secp256k1MemoryManager::PerformanceReport Secp256k1MemoryManager::generate_performance_report() const {
    if (!initialized_) return PerformanceReport{};
    
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_report_time_);
    
    // If less than 1 second elapsed, return cached report
    if (elapsed.count() < 1 && cached_report_.memory_stats.total_allocated > 0) {
        return cached_report_;
    }
    
    PerformanceReport report;
    
    // Get memory statistics
    report.memory_stats = memory_pool_->get_stats();
    
    // Get batch optimizer metrics
    if (batch_optimizer_) {
        report.batch_metrics = batch_optimizer_->get_metrics();
    }
    
    // Calculate memory bandwidth
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    
    if (total_memory > 0) {
        report.effective_memory_usage = static_cast<double>(total_memory - free_memory) / total_memory;
    }
    
    // Estimate memory bandwidth (simplified calculation)
    if (report.memory_stats.total_alloc_time.count() > 0) {
        double transfer_rate = static_cast<double>(report.memory_stats.current_usage) / 
                              (report.memory_stats.total_alloc_time.count() / 1000.0); // bytes/sec
        report.memory_bandwidth_gbps = transfer_rate / (1024.0 * 1024.0 * 1024.0); // GB/s
    }
    
    report.operations_per_second = report.batch_metrics.operations_per_second;
    
    // Cache the report
    cached_report_ = report;
    last_report_time_ = current_time;
    
    return report;
}

void Secp256k1MemoryManager::reset_performance_counters() {
    if (memory_pool_) {
        memory_pool_->reset();
    }
    
    cached_report_ = PerformanceReport{};
    last_report_time_ = std::chrono::steady_clock::now();
}

size_t Secp256k1MemoryManager::get_device_memory_size() const {
    return total_device_memory_;
}

size_t Secp256k1MemoryManager::get_available_device_memory() const {
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    return free_memory;
}

int Secp256k1MemoryManager::get_device_compute_capability() const {
    return device_props_.major * 10 + device_props_.minor;
}

bool Secp256k1MemoryManager::query_device_properties() {
    cudaError_t err = cudaGetDeviceProperties(&device_props_, config_.device_id);
    if (err != cudaSuccess) return false;
    
    // Get memory information
    size_t free_memory;
    err = cudaMemGetInfo(&free_memory, &total_device_memory_);
    if (err != cudaSuccess) return false;
    
    available_device_memory_ = free_memory;
    
    return true;
}

void Secp256k1MemoryManager::setup_peer_access() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    for (int i = 0; i < device_count; ++i) {
        if (i != config_.device_id) {
            int can_access;
            cudaDeviceCanAccessPeer(&can_access, config_.device_id, i);
            if (can_access) {
                cudaDeviceEnablePeerAccess(i, 0);
            }
        }
    }
}

size_t Secp256k1MemoryManager::calculate_memory_requirement(size_t operations) {
    // Estimate memory requirement for secp256k1 operations
    size_t scalar_memory = layouts::ScalarArray::calculate_size(operations);
    size_t point_memory = layouts::PointArray::calculate_size(operations);
    size_t temp_memory = point_memory; // Temporary storage for intermediate results
    
    return scalar_memory + point_memory + temp_memory;
}

} // namespace gpu
} // namespace ecc
} // namespace keyhunt