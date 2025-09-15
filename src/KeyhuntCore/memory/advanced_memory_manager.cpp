/**
 * @file advanced_memory_manager.cpp
 * @brief Implementation of advanced memory management system
 * @author KeyhuntCUDA Team
 * 
 * T047: Implement advanced memory management system with GPU memory pools and smart allocation strategies
 */

#include "advanced_memory_manager.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

namespace keyhunt {
namespace memory {

AdvancedMemoryManager::AdvancedMemoryManager()
    : next_allocation_id_(1)
    , background_tasks_running_(false)
    , memory_compression_enabled_(false)
    , memory_prefetching_enabled_(true)
    , memory_overcommit_enabled_(false)
{
    // Initialize default strategies for each pool type
    pool_strategies_[MemoryPoolType::SMALL_OBJECTS] = AllocationStrategy::SLAB_ALLOCATOR;
    pool_strategies_[MemoryPoolType::MEDIUM_OBJECTS] = AllocationStrategy::BEST_FIT;
    pool_strategies_[MemoryPoolType::LARGE_OBJECTS] = AllocationStrategy::FIRST_FIT;
    pool_strategies_[MemoryPoolType::HUGE_OBJECTS] = AllocationStrategy::WORST_FIT;
    pool_strategies_[MemoryPoolType::ECC_OPERATIONS] = AllocationStrategy::POOL_ALLOCATOR;
    pool_strategies_[MemoryPoolType::HASH_OPERATIONS] = AllocationStrategy::POOL_ALLOCATOR;
    pool_strategies_[MemoryPoolType::SCANNING_BUFFERS] = AllocationStrategy::LINEAR_ALLOCATOR;
    pool_strategies_[MemoryPoolType::TEMPORARY_STORAGE] = AllocationStrategy::STACK_ALLOCATOR;
    pool_strategies_[MemoryPoolType::PERSISTENT_CACHE] = AllocationStrategy::BUDDY_SYSTEM;
}

AdvancedMemoryManager::~AdvancedMemoryManager() {
    cleanup();
}

bool AdvancedMemoryManager::initialize(const std::vector<models::GPUInfo>& gpu_devices) {
    try {
        std::cout << "Initializing Advanced Memory Manager..." << std::endl;
        
        gpu_devices_ = gpu_devices;
        
        // Initialize memory pools for each device
        for (const auto& gpu : gpu_devices_) {
            if (!initialize_device_pools(gpu.device_id)) {
                std::cerr << "ERROR: Failed to initialize memory pools for device " << gpu.device_id << std::endl;
                return false;
            }
            
            // Set default memory limit (90% of total memory)
            size_t total_memory = get_total_device_memory(gpu.device_id);
            device_memory_limits_[gpu.device_id] = static_cast<size_t>(total_memory * 0.9);
            
            std::cout << "Device " << gpu.device_id << " (" << gpu.device_name << "):" << std::endl;
            std::cout << "  Total memory: " << memory_utils::format_memory_size(total_memory) << std::endl;
            std::cout << "  Memory limit: " << memory_utils::format_memory_size(device_memory_limits_[gpu.device_id]) << std::endl;
        }
        
        // Initialize global statistics
        {
            std::lock_guard<std::mutex> lock(statistics_mutex_);
            global_statistics_ = MemoryStatistics();
            
            // Calculate total memory across all devices
            for (const auto& gpu : gpu_devices_) {
                global_statistics_.total_device_memory += get_total_device_memory(gpu.device_id);
                device_statistics_[gpu.device_id] = MemoryStatistics();
            }
        }
        
        std::cout << "Advanced Memory Manager initialized successfully" << std::endl;
        std::cout << "  Total GPU devices: " << gpu_devices_.size() << std::endl;
        std::cout << "  Total device memory: " << memory_utils::format_memory_size(global_statistics_.total_device_memory) << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in AdvancedMemoryManager::initialize: " << e.what() << std::endl;
        return false;
    }
}

bool AdvancedMemoryManager::configure_pool(const MemoryPoolConfig& config) {
    try {
        // Apply configuration to all devices
        for (const auto& gpu : gpu_devices_) {
            if (!create_pool(config.type, config, gpu.device_id)) {
                std::cerr << "ERROR: Failed to configure pool type " << static_cast<int>(config.type) 
                          << " for device " << gpu.device_id << std::endl;
                return false;
            }
        }
        
        std::cout << "Memory pool configured successfully:" << std::endl;
        std::cout << "  Type: " << static_cast<int>(config.type) << std::endl;
        std::cout << "  Initial size: " << memory_utils::format_memory_size(config.initial_size) << std::endl;
        std::cout << "  Max size: " << memory_utils::format_memory_size(config.max_size) << std::endl;
        std::cout << "  Strategy: " << static_cast<int>(config.allocation_strategy) << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in configure_pool: " << e.what() << std::endl;
        return false;
    }
}

bool AdvancedMemoryManager::start_background_tasks() {
    if (background_tasks_running_) {
        return true;
    }
    
    try {
        background_tasks_running_ = true;
        
        // Start defragmentation thread
        defragmentation_thread_ = std::thread(&AdvancedMemoryManager::defragmentation_task_loop, this);
        
        // Start monitoring thread
        monitoring_thread_ = std::thread(&AdvancedMemoryManager::monitoring_task_loop, this);
        
        // Start optimization thread
        optimization_thread_ = std::thread(&AdvancedMemoryManager::optimization_task_loop, this);
        
        std::cout << "Memory management background tasks started" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to start background tasks: " << e.what() << std::endl;
        background_tasks_running_ = false;
        return false;
    }
}

bool AdvancedMemoryManager::stop_background_tasks() {
    if (!background_tasks_running_) {
        return true;
    }
    
    std::cout << "Stopping memory management background tasks..." << std::endl;
    
    background_tasks_running_ = false;
    
    // Join all background threads
    if (defragmentation_thread_.joinable()) {
        defragmentation_thread_.join();
    }
    
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    if (optimization_thread_.joinable()) {
        optimization_thread_.join();
    }
    
    std::cout << "Background tasks stopped successfully" << std::endl;
    return true;
}

void AdvancedMemoryManager::cleanup() {
    std::cout << "Cleaning up Advanced Memory Manager..." << std::endl;
    
    // Stop background tasks
    stop_background_tasks();
    
    // Deallocate all active allocations
    {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        for (auto& [id, allocation] : active_allocations_) {
            try {
                if (allocation.device_ptr) {
                    cudaSetDevice(allocation.device_id);
                    cudaFree(allocation.device_ptr);
                }
                if (allocation.host_ptr && allocation.is_pinned) {
                    cudaFreeHost(allocation.host_ptr);
                }
            } catch (...) {
                // Ignore cleanup errors
            }
        }
        active_allocations_.clear();
    }
    
    // Destroy all memory pools
    for (auto& [device_id, pools] : memory_pools_) {
        pools.clear();
    }
    memory_pools_.clear();
    
    std::cout << "Advanced Memory Manager cleanup completed" << std::endl;
}

MemoryAllocation AdvancedMemoryManager::allocate(const MemoryAllocationRequest& request) {
    try {
        // Validate request
        if (!validate_allocation_request(request)) {
            MemoryAllocation failed_allocation;
            return failed_allocation;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Try allocation from appropriate pool
        MemoryAllocation allocation = allocate_from_pool(request);
        
        // If pool allocation failed, try fallback strategies
        if (!allocation.device_ptr && request.allow_fallback_device) {
            allocation = fallback_allocation(request);
        }
        
        if (allocation.device_ptr) {
            // Record successful allocation
            allocation.allocation_id = next_allocation_id_++;
            allocation.allocation_time = std::chrono::system_clock::now();
            
            {
                std::lock_guard<std::mutex> lock(allocation_mutex_);
                active_allocations_[allocation.allocation_id] = allocation;
            }
            
            // Update statistics
            update_statistics(allocation);
            
            // Call allocation callback if set
            if (allocation_callback_) {
                allocation_callback_(allocation);
            }
            
            // Calculate allocation time
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            std::lock_guard<std::mutex> stats_lock(statistics_mutex_);
            global_statistics_.average_allocation_time = 
                std::chrono::microseconds(
                    (global_statistics_.average_allocation_time.count() * global_statistics_.total_allocations + duration.count()) 
                    / (global_statistics_.total_allocations + 1)
                );
            global_statistics_.peak_allocation_time = std::max(global_statistics_.peak_allocation_time, duration);
        } else {
            // Record failed allocation
            std::lock_guard<std::mutex> stats_lock(statistics_mutex_);
            global_statistics_.failed_allocations++;
        }
        
        return allocation;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in allocate: " << e.what() << std::endl;
        
        std::lock_guard<std::mutex> stats_lock(statistics_mutex_);
        global_statistics_.failed_allocations++;
        
        return MemoryAllocation();
    }
}

bool AdvancedMemoryManager::deallocate(const MemoryAllocation& allocation) {
    return deallocate(allocation.allocation_id);
}

bool AdvancedMemoryManager::deallocate(uint64_t allocation_id) {
    try {
        MemoryAllocation allocation;
        
        // Find and remove allocation from active list
        {
            std::lock_guard<std::mutex> lock(allocation_mutex_);
            auto it = active_allocations_.find(allocation_id);
            if (it == active_allocations_.end()) {
                std::cerr << "ERROR: Allocation ID " << allocation_id << " not found" << std::endl;
                return false;
            }
            
            allocation = it->second;
            active_allocations_.erase(it);
        }
        
        // Free the memory
        bool success = false;
        
        try {
            cudaSetDevice(allocation.device_id);
            
            if (allocation.is_managed) {
                cudaFree(allocation.device_ptr);
            } else if (allocation.is_pinned && allocation.host_ptr) {
                cudaFreeHost(allocation.host_ptr);
            } else if (allocation.device_ptr) {
                // Try to deallocate from pool first
                auto pool = get_pool(allocation.pool_type, allocation.device_id);
                if (pool && pool->deallocate(allocation.device_ptr)) {
                    success = true;
                } else {
                    // Fallback to direct CUDA free
                    cudaFree(allocation.device_ptr);
                    success = true;
                }
            }
            
            if (!success) {
                cudaError_t error = cudaGetLastError();
                if (error == cudaSuccess) {
                    success = true;
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception during deallocation: " << e.what() << std::endl;
        }
        
        if (success) {
            // Update statistics
            update_deallocation_statistics(allocation_id);
            
            // Call deallocation callback if set
            if (deallocation_callback_) {
                deallocation_callback_(allocation_id);
            }
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in deallocate: " << e.what() << std::endl;
        return false;
    }
}

MemoryAllocation AdvancedMemoryManager::allocate_ecc_memory(size_t size, int device_id) {
    MemoryAllocationRequest request;
    request.size = size;
    request.device_id = device_id;
    request.pool_type = MemoryPoolType::ECC_OPERATIONS;
    request.allocation_tag = "ecc_operations";
    request.alignment = 256; // ECC operations benefit from aligned memory
    request.prefer_contiguous = true;
    
    return allocate(request);
}

MemoryAllocation AdvancedMemoryManager::allocate_hash_memory(size_t size, int device_id) {
    MemoryAllocationRequest request;
    request.size = size;
    request.device_id = device_id;
    request.pool_type = MemoryPoolType::HASH_OPERATIONS;
    request.allocation_tag = "hash_operations";
    request.alignment = 128; // Hash operations alignment
    
    return allocate(request);
}

MemoryAllocation AdvancedMemoryManager::allocate_scanning_buffer(size_t size, int device_id) {
    MemoryAllocationRequest request;
    request.size = size;
    request.device_id = device_id;
    request.pool_type = MemoryPoolType::SCANNING_BUFFERS;
    request.allocation_tag = "scanning_buffers";
    request.alignment = 256;
    request.prefer_contiguous = true;
    
    return allocate(request);
}

MemoryAllocation AdvancedMemoryManager::allocate_temporary(size_t size, int device_id) {
    MemoryAllocationRequest request;
    request.size = size;
    request.device_id = device_id;
    request.pool_type = MemoryPoolType::TEMPORARY_STORAGE;
    request.allocation_tag = "temporary";
    request.max_wait_time = std::chrono::milliseconds(100); // Fast allocation for temporary
    
    return allocate(request);
}

MemoryAllocation AdvancedMemoryManager::allocate_pinned_host(size_t size) {
    try {
        MemoryAllocation allocation;
        allocation.requested_size = size;
        allocation.allocated_size = memory_utils::align_to_boundary(size, 4096); // Page alignment
        allocation.is_pinned = true;
        allocation.allocation_tag = "pinned_host";
        
        cudaError_t error = cudaHostAlloc(&allocation.host_ptr, allocation.allocated_size, cudaHostAllocDefault);
        
        if (error == cudaSuccess && allocation.host_ptr) {
            allocation.allocation_id = next_allocation_id_++;
            allocation.allocation_time = std::chrono::system_clock::now();
            
            {
                std::lock_guard<std::mutex> lock(allocation_mutex_);
                active_allocations_[allocation.allocation_id] = allocation;
            }
            
            update_statistics(allocation);
            
            if (allocation_callback_) {
                allocation_callback_(allocation);
            }
        } else {
            std::cerr << "ERROR: Failed to allocate pinned host memory: " 
                      << cudaGetErrorString(error) << std::endl;
        }
        
        return allocation;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in allocate_pinned_host: " << e.what() << std::endl;
        return MemoryAllocation();
    }
}

MemoryAllocation AdvancedMemoryManager::allocate_managed(size_t size, int device_id) {
    try {
        MemoryAllocation allocation;
        allocation.requested_size = size;
        allocation.allocated_size = memory_utils::align_to_boundary(size, 256);
        allocation.device_id = device_id;
        allocation.is_managed = true;
        allocation.allocation_tag = "managed_memory";
        
        cudaSetDevice(device_id);
        cudaError_t error = cudaMallocManaged(&allocation.device_ptr, allocation.allocated_size);
        
        if (error == cudaSuccess && allocation.device_ptr) {
            allocation.allocation_id = next_allocation_id_++;
            allocation.allocation_time = std::chrono::system_clock::now();
            
            {
                std::lock_guard<std::mutex> lock(allocation_mutex_);
                active_allocations_[allocation.allocation_id] = allocation;
            }
            
            update_statistics(allocation);
            
            if (allocation_callback_) {
                allocation_callback_(allocation);
            }
        } else {
            std::cerr << "ERROR: Failed to allocate managed memory: " 
                      << cudaGetErrorString(error) << std::endl;
        }
        
        return allocation;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in allocate_managed: " << e.what() << std::endl;
        return MemoryAllocation();
    }
}

bool AdvancedMemoryManager::create_pool(MemoryPoolType type, const MemoryPoolConfig& config, int device_id) {
    try {
        if (!is_device_valid(device_id)) {
            std::cerr << "ERROR: Invalid device ID: " << device_id << std::endl;
            return false;
        }
        
        // Create memory pool
        auto pool = std::make_unique<MemoryPool>(type, config, device_id);
        
        if (!pool) {
            std::cerr << "ERROR: Failed to create memory pool" << std::endl;
            return false;
        }
        
        // Store pool
        memory_pools_[device_id][type] = std::move(pool);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in create_pool: " << e.what() << std::endl;
        return false;
    }
}

MemoryStatistics AdvancedMemoryManager::get_memory_statistics(int device_id) const {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    
    if (device_id == -1) {
        return global_statistics_;
    } else {
        auto it = device_statistics_.find(device_id);
        if (it != device_statistics_.end()) {
            return it->second;
        } else {
            return MemoryStatistics();
        }
    }
}

std::string AdvancedMemoryManager::generate_memory_report() const {
    std::ostringstream report;
    
    report << "KeyhuntCUDA Advanced Memory Management Report\n";
    report << "============================================\n\n";
    
    // Global statistics
    auto global_stats = get_memory_statistics(-1);
    report << "Global Memory Statistics:\n";
    report << "  Total device memory: " << memory_utils::format_memory_size(global_stats.total_device_memory) << "\n";
    report << "  Allocated device memory: " << memory_utils::format_memory_size(global_stats.allocated_device_memory) << "\n";
    report << "  Pinned memory size: " << memory_utils::format_memory_size(global_stats.pinned_memory_size) << "\n";
    report << "  Total allocations: " << global_stats.total_allocations << "\n";
    report << "  Failed allocations: " << global_stats.failed_allocations << "\n";
    report << "  Average allocation time: " << global_stats.average_allocation_time.count() << " μs\n";
    report << "  Overall fragmentation: " << std::fixed << std::setprecision(2) 
           << (global_stats.overall_fragmentation_ratio * 100) << "%\n\n";
    
    // Per-device statistics
    for (const auto& gpu : gpu_devices_) {
        auto device_stats = get_memory_statistics(gpu.device_id);
        report << "Device " << gpu.device_id << " (" << gpu.device_name << "):\n";
        report << "  Allocated memory: " << memory_utils::format_memory_size(device_stats.allocated_device_memory) << "\n";
        report << "  Free memory: " << memory_utils::format_memory_size(
            get_available_device_memory(gpu.device_id)) << "\n";
        report << "  Allocations: " << device_stats.total_allocations << "\n";
        report << "  Fragmentation: " << std::fixed << std::setprecision(2) 
               << (device_stats.overall_fragmentation_ratio * 100) << "%\n";
        
        // Pool statistics
        if (memory_pools_.find(gpu.device_id) != memory_pools_.end()) {
            const auto& pools = memory_pools_.at(gpu.device_id);
            for (const auto& [pool_type, pool] : pools) {
                if (pool) {
                    report << "  Pool " << static_cast<int>(pool_type) 
                           << ": " << memory_utils::format_memory_size(pool->get_allocated_size())
                           << "/" << memory_utils::format_memory_size(pool->get_total_size()) << "\n";
                }
            }
        }
        report << "\n";
    }
    
    // Optimization recommendations
    auto recommendations = generate_optimization_recommendations();
    if (!recommendations.empty()) {
        report << "Optimization Recommendations:\n";
        for (const auto& rec : recommendations) {
            report << "  • " << rec << "\n";
        }
        report << "\n";
    }
    
    report << "Report generated at: " 
           << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
    
    return report.str();
}

// Private method implementations

bool AdvancedMemoryManager::initialize_device_pools(int device_id) {
    try {
        cudaSetDevice(device_id);
        
        // Create default pools for this device
        return create_default_pools(device_id);
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to initialize device pools for device " << device_id 
                  << ": " << e.what() << std::endl;
        return false;
    }
}

bool AdvancedMemoryManager::create_default_pools(int device_id) {
    std::vector<std::pair<MemoryPoolType, MemoryPoolConfig>> default_configs = {
        {MemoryPoolType::SMALL_OBJECTS, {MemoryPoolType::SMALL_OBJECTS, 10*1024*1024, 100*1024*1024, 4096}},
        {MemoryPoolType::MEDIUM_OBJECTS, {MemoryPoolType::MEDIUM_OBJECTS, 50*1024*1024, 500*1024*1024, 64*1024}},
        {MemoryPoolType::LARGE_OBJECTS, {MemoryPoolType::LARGE_OBJECTS, 100*1024*1024, 1024*1024*1024, 1024*1024}},
        {MemoryPoolType::ECC_OPERATIONS, {MemoryPoolType::ECC_OPERATIONS, 50*1024*1024, 300*1024*1024, 256*1024}},
        {MemoryPoolType::HASH_OPERATIONS, {MemoryPoolType::HASH_OPERATIONS, 30*1024*1024, 200*1024*1024, 128*1024}},
        {MemoryPoolType::SCANNING_BUFFERS, {MemoryPoolType::SCANNING_BUFFERS, 100*1024*1024, 800*1024*1024, 1024*1024}},
        {MemoryPoolType::TEMPORARY_STORAGE, {MemoryPoolType::TEMPORARY_STORAGE, 20*1024*1024, 100*1024*1024, 32*1024}}
    };
    
    for (const auto& [type, config] : default_configs) {
        if (!create_pool(type, config, device_id)) {
            std::cerr << "ERROR: Failed to create default pool type " << static_cast<int>(type) << std::endl;
            return false;
        }
    }
    
    return true;
}

MemoryPool* AdvancedMemoryManager::get_pool(MemoryPoolType type, int device_id) {
    auto device_it = memory_pools_.find(device_id);
    if (device_it == memory_pools_.end()) {
        return nullptr;
    }
    
    auto pool_it = device_it->second.find(type);
    if (pool_it == device_it->second.end()) {
        return nullptr;
    }
    
    return pool_it->second.get();
}

MemoryAllocation AdvancedMemoryManager::allocate_from_pool(const MemoryAllocationRequest& request) {
    auto pool = get_pool(request.pool_type, request.device_id);
    if (!pool) {
        // Try to create pool on demand if it doesn't exist
        MemoryPoolConfig default_config;
        default_config.type = request.pool_type;
        default_config.allocation_strategy = pool_strategies_[request.pool_type];
        
        if (create_pool(request.pool_type, default_config, request.device_id)) {
            pool = get_pool(request.pool_type, request.device_id);
        }
    }
    
    if (pool) {
        return pool->allocate(request.size, request.alignment, request.allocation_tag);
    }
    
    return MemoryAllocation();
}

bool AdvancedMemoryManager::validate_allocation_request(const MemoryAllocationRequest& request) const {
    if (request.size == 0) {
        return false;
    }
    
    if (!is_device_valid(request.device_id)) {
        return false;
    }
    
    // Check if we have enough memory available
    size_t available = get_available_device_memory(request.device_id);
    if (available < request.size) {
        return false;
    }
    
    return true;
}

void AdvancedMemoryManager::update_statistics(const MemoryAllocation& allocation) {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    
    // Update global statistics
    global_statistics_.total_allocations++;
    global_statistics_.allocated_device_memory += allocation.allocated_size;
    
    if (allocation.is_pinned) {
        global_statistics_.pinned_memory_size += allocation.allocated_size;
    }
    
    if (allocation.is_managed) {
        global_statistics_.managed_memory_size += allocation.allocated_size;
    }
    
    // Update device statistics
    device_statistics_[allocation.device_id].total_allocations++;
    device_statistics_[allocation.device_id].allocated_device_memory += allocation.allocated_size;
    
    // Update pool statistics
    auto& pool_stats = global_statistics_.pool_allocated_sizes;
    pool_stats[allocation.pool_type] += allocation.allocated_size;
}

void AdvancedMemoryManager::update_deallocation_statistics(uint64_t allocation_id) {
    // Implementation for deallocation statistics update
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    global_statistics_.total_deallocations++;
}

size_t AdvancedMemoryManager::get_available_device_memory(int device_id) const {
    try {
        cudaSetDevice(device_id);
        size_t free_mem, total_mem;
        cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
        
        if (error == cudaSuccess) {
            return free_mem;
        }
    } catch (...) {
        // Handle error silently
    }
    
    return 0;
}

size_t AdvancedMemoryManager::get_total_device_memory(int device_id) const {
    try {
        cudaSetDevice(device_id);
        size_t free_mem, total_mem;
        cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
        
        if (error == cudaSuccess) {
            return total_mem;
        }
    } catch (...) {
        // Handle error silently
    }
    
    return 0;
}

bool AdvancedMemoryManager::is_device_valid(int device_id) const {
    return std::find_if(gpu_devices_.begin(), gpu_devices_.end(),
        [device_id](const models::GPUInfo& gpu) {
            return gpu.device_id == device_id;
        }) != gpu_devices_.end();
}

std::vector<std::string> AdvancedMemoryManager::generate_optimization_recommendations() const {
    std::vector<std::string> recommendations;
    
    auto global_stats = get_memory_statistics(-1);
    
    // Check fragmentation
    if (global_stats.overall_fragmentation_ratio > 0.3) {
        recommendations.push_back("High memory fragmentation detected - consider running defragmentation");
    }
    
    // Check allocation failure rate
    if (global_stats.total_allocations > 0) {
        double failure_rate = static_cast<double>(global_stats.failed_allocations) / global_stats.total_allocations;
        if (failure_rate > 0.05) { // 5% failure rate
            recommendations.push_back("High allocation failure rate - consider increasing memory limits or pool sizes");
        }
    }
    
    // Check memory utilization
    if (global_stats.total_device_memory > 0) {
        double utilization = static_cast<double>(global_stats.allocated_device_memory) / global_stats.total_device_memory;
        if (utilization > 0.9) {
            recommendations.push_back("Very high memory utilization - consider memory optimization strategies");
        }
    }
    
    return recommendations;
}

void AdvancedMemoryManager::defragmentation_task_loop() {
    while (background_tasks_running_) {
        try {
            // Check each device for fragmentation
            for (const auto& gpu : gpu_devices_) {
                for (const auto& [pool_type, _] : memory_pools_[gpu.device_id]) {
                    if (should_defragment_pool(pool_type, gpu.device_id)) {
                        auto pool = get_pool(pool_type, gpu.device_id);
                        if (pool) {
                            pool->defragment();
                        }
                    }
                }
            }
            
            // Sleep for defragmentation interval
            std::this_thread::sleep_for(std::chrono::minutes(5));
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in defragmentation loop: " << e.what() << std::endl;
        }
    }
}

void AdvancedMemoryManager::monitoring_task_loop() {
    while (background_tasks_running_) {
        try {
            // Monitor memory pressure on each device
            for (const auto& gpu : gpu_devices_) {
                if (detect_memory_pressure(gpu.device_id)) {
                    MemoryPressureEvent event;
                    event.device_id = gpu.device_id;
                    event.pressure_level = calculate_pressure_level(gpu.device_id);
                    event.available_memory = get_available_device_memory(gpu.device_id);
                    
                    handle_memory_pressure(event);
                }
            }
            
            // Clean up expired allocations
            cleanup_expired_allocations();
            
            // Sleep for monitoring interval
            std::this_thread::sleep_for(std::chrono::seconds(30));
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in monitoring loop: " << e.what() << std::endl;
        }
    }
}

void AdvancedMemoryManager::optimization_task_loop() {
    while (background_tasks_running_) {
        try {
            // Optimize allocation strategies based on usage patterns
            optimize_allocation_strategies();
            
            // Balance memory across devices if needed
            balance_memory_across_devices();
            
            // Sleep for optimization interval
            std::this_thread::sleep_for(std::chrono::minutes(10));
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in optimization loop: " << e.what() << std::endl;
        }
    }
}

// Additional implementation methods would continue here...
// For brevity, showing key methods that demonstrate the memory management approach

} // namespace memory
} // namespace keyhunt