/**
 * @file advanced_memory_manager.h
 * @brief Advanced memory management system with GPU memory pools and smart allocation strategies
 * @author KeyhuntCUDA Team
 * 
 * T047: Implement advanced memory management system with GPU memory pools and smart allocation strategies
 * 
 * This module provides intelligent memory management for the KeyhuntCUDA system:
 * - GPU memory pools with different allocation strategies
 * - Smart allocation based on usage patterns and performance metrics
 * - Memory fragmentation detection and defragmentation
 * - Automatic memory pressure handling and optimization
 * - Cross-device memory management for multi-GPU setups
 * - Memory usage analytics and optimization recommendations
 */

#pragma once

#include "../models/GPUConfiguration.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <queue>
#include <functional>

namespace keyhunt {
namespace memory {

/**
 * @brief Memory allocation strategy types
 */
enum class AllocationStrategy {
    FIRST_FIT,          // First available block
    BEST_FIT,           // Best size match
    WORST_FIT,          // Largest available block
    BUDDY_SYSTEM,       // Buddy system allocation
    SLAB_ALLOCATOR,     // Slab-based allocation
    POOL_ALLOCATOR,     // Fixed-size pool allocation
    STACK_ALLOCATOR,    // LIFO stack allocation
    LINEAR_ALLOCATOR,   // Linear allocation
    ADAPTIVE_STRATEGY   // Dynamically choose strategy
};

/**
 * @brief Memory pool types
 */
enum class MemoryPoolType {
    SMALL_OBJECTS,      // < 1KB allocations
    MEDIUM_OBJECTS,     // 1KB - 1MB allocations
    LARGE_OBJECTS,      // 1MB - 100MB allocations
    HUGE_OBJECTS,       // > 100MB allocations
    ECC_OPERATIONS,     // ECC-specific memory
    HASH_OPERATIONS,    // Hash computation memory
    SCANNING_BUFFERS,   // Private key scanning buffers
    TEMPORARY_STORAGE,  // Short-lived temporary memory
    PERSISTENT_CACHE    // Long-lived cache memory
};

/**
 * @brief Memory allocation request
 */
struct MemoryAllocationRequest {
    size_t size;
    size_t alignment;
    MemoryPoolType pool_type;
    int device_id;
    std::string allocation_tag;
    std::chrono::milliseconds max_wait_time;
    bool allow_fallback_device;
    bool prefer_contiguous;
    
    MemoryAllocationRequest()
        : size(0)
        , alignment(256) // 256-byte alignment by default
        , pool_type(MemoryPoolType::MEDIUM_OBJECTS)
        , device_id(0)
        , max_wait_time(std::chrono::milliseconds(1000))
        , allow_fallback_device(true)
        , prefer_contiguous(true)
    {}
};

/**
 * @brief Memory allocation result
 */
struct MemoryAllocation {
    void* device_ptr;
    void* host_ptr; // For pinned memory
    size_t allocated_size;
    size_t requested_size;
    int device_id;
    MemoryPoolType pool_type;
    AllocationStrategy strategy_used;
    std::chrono::system_clock::time_point allocation_time;
    std::string allocation_tag;
    uint64_t allocation_id;
    bool is_pinned;
    bool is_managed; // CUDA managed memory
    
    MemoryAllocation()
        : device_ptr(nullptr)
        , host_ptr(nullptr)
        , allocated_size(0)
        , requested_size(0)
        , device_id(0)
        , pool_type(MemoryPoolType::MEDIUM_OBJECTS)
        , strategy_used(AllocationStrategy::FIRST_FIT)
        , allocation_time(std::chrono::system_clock::now())
        , allocation_id(0)
        , is_pinned(false)
        , is_managed(false)
    {}
};

/**
 * @brief Memory pool configuration
 */
struct MemoryPoolConfig {
    MemoryPoolType type;
    size_t initial_size;
    size_t max_size;
    size_t chunk_size;
    AllocationStrategy allocation_strategy;
    bool enable_defragmentation;
    std::chrono::seconds defrag_interval;
    double fragmentation_threshold; // 0.0 to 1.0
    bool enable_preallocation;
    bool enable_memory_warming; // Pre-touch memory pages
    
    MemoryPoolConfig()
        : type(MemoryPoolType::MEDIUM_OBJECTS)
        , initial_size(100 * 1024 * 1024) // 100MB
        , max_size(1024 * 1024 * 1024) // 1GB
        , chunk_size(1024 * 1024) // 1MB
        , allocation_strategy(AllocationStrategy::BEST_FIT)
        , enable_defragmentation(true)
        , defrag_interval(std::chrono::seconds(300)) // 5 minutes
        , fragmentation_threshold(0.3) // 30% fragmentation
        , enable_preallocation(true)
        , enable_memory_warming(false)
    {}
};

/**
 * @brief Memory usage statistics
 */
struct MemoryStatistics {
    // Total memory
    size_t total_device_memory;
    size_t total_host_memory;
    
    // Allocated memory
    size_t allocated_device_memory;
    size_t allocated_host_memory;
    size_t pinned_memory_size;
    size_t managed_memory_size;
    
    // Pool statistics
    std::unordered_map<MemoryPoolType, size_t> pool_allocated_sizes;
    std::unordered_map<MemoryPoolType, size_t> pool_free_sizes;
    std::unordered_map<MemoryPoolType, size_t> pool_fragmentation_bytes;
    
    // Performance metrics
    size_t total_allocations;
    size_t total_deallocations;
    size_t failed_allocations;
    std::chrono::microseconds average_allocation_time;
    std::chrono::microseconds peak_allocation_time;
    
    // Fragmentation metrics
    double overall_fragmentation_ratio;
    size_t largest_free_block;
    size_t total_free_blocks;
    
    MemoryStatistics()
        : total_device_memory(0)
        , total_host_memory(0)
        , allocated_device_memory(0)
        , allocated_host_memory(0)
        , pinned_memory_size(0)
        , managed_memory_size(0)
        , total_allocations(0)
        , total_deallocations(0)
        , failed_allocations(0)
        , average_allocation_time(std::chrono::microseconds(0))
        , peak_allocation_time(std::chrono::microseconds(0))
        , overall_fragmentation_ratio(0.0)
        , largest_free_block(0)
        , total_free_blocks(0)
    {}
};

/**
 * @brief Memory pressure event
 */
struct MemoryPressureEvent {
    int device_id;
    double pressure_level; // 0.0 to 1.0
    size_t available_memory;
    size_t requested_memory;
    std::chrono::system_clock::time_point timestamp;
    std::vector<std::string> contributing_factors;
    
    MemoryPressureEvent()
        : device_id(0)
        , pressure_level(0.0)
        , available_memory(0)
        , requested_memory(0)
        , timestamp(std::chrono::system_clock::now())
    {}
};

/**
 * @brief Forward declaration for memory pool
 */
class MemoryPool;

/**
 * @brief Main advanced memory manager class
 */
class AdvancedMemoryManager {
public:
    AdvancedMemoryManager();
    ~AdvancedMemoryManager();
    
    // Initialization and configuration
    bool initialize(const std::vector<models::GPUInfo>& gpu_devices);
    bool configure_pool(const MemoryPoolConfig& config);
    bool start_background_tasks();
    bool stop_background_tasks();
    void cleanup();
    
    // Memory allocation and deallocation
    MemoryAllocation allocate(const MemoryAllocationRequest& request);
    bool deallocate(const MemoryAllocation& allocation);
    bool deallocate(uint64_t allocation_id);
    
    // Specialized allocation methods
    MemoryAllocation allocate_ecc_memory(size_t size, int device_id = 0);
    MemoryAllocation allocate_hash_memory(size_t size, int device_id = 0);
    MemoryAllocation allocate_scanning_buffer(size_t size, int device_id = 0);
    MemoryAllocation allocate_temporary(size_t size, int device_id = 0);
    MemoryAllocation allocate_pinned_host(size_t size);
    MemoryAllocation allocate_managed(size_t size, int device_id = 0);
    
    // Memory pool management
    bool create_pool(MemoryPoolType type, const MemoryPoolConfig& config, int device_id);
    bool destroy_pool(MemoryPoolType type, int device_id);
    bool resize_pool(MemoryPoolType type, int device_id, size_t new_size);
    std::vector<MemoryPoolType> get_available_pools(int device_id) const;
    
    // Memory optimization
    bool defragment_pools(int device_id = -1); // -1 for all devices
    bool compact_pool(MemoryPoolType type, int device_id);
    bool optimize_allocation_strategies();
    bool warm_memory_pools(); // Pre-touch memory for better performance
    
    // Memory pressure handling
    bool handle_memory_pressure(const MemoryPressureEvent& event);
    bool free_unused_memory(int device_id = -1);
    bool reduce_pool_sizes(double reduction_ratio = 0.2);
    std::vector<MemoryAllocation> get_eviction_candidates();
    
    // Cross-device memory management
    bool enable_peer_access(int src_device, int dst_device);
    bool copy_memory_cross_device(const MemoryAllocation& src, const MemoryAllocation& dst);
    MemoryAllocation migrate_allocation(const MemoryAllocation& allocation, int target_device);
    bool balance_memory_across_devices();
    
    // Memory analytics and monitoring
    MemoryStatistics get_memory_statistics(int device_id = -1) const;
    std::vector<MemoryPressureEvent> get_recent_pressure_events() const;
    double calculate_memory_efficiency(int device_id = -1) const;
    std::vector<std::string> generate_optimization_recommendations() const;
    
    // Memory usage tracking
    std::unordered_map<std::string, size_t> get_allocation_by_tag() const;
    std::vector<MemoryAllocation> get_active_allocations() const;
    std::vector<MemoryAllocation> get_allocations_by_pool(MemoryPoolType type) const;
    bool set_allocation_callback(std::function<void(const MemoryAllocation&)> callback);
    bool set_deallocation_callback(std::function<void(uint64_t)> callback);
    
    // Configuration and tuning
    bool set_allocation_strategy(MemoryPoolType type, AllocationStrategy strategy);
    bool set_defragmentation_threshold(MemoryPoolType type, double threshold);
    bool enable_memory_compression(bool enable = true);
    bool set_memory_limit(int device_id, size_t limit);
    
    // Advanced features
    bool enable_memory_prefetching(bool enable = true);
    bool set_numa_policy(const std::string& policy); // "interleave", "bind", "preferred"
    bool enable_memory_overcommit(bool enable = true);
    MemoryAllocation allocate_with_hint(const MemoryAllocationRequest& request, void* hint_address);
    
    // Reporting and debugging
    std::string generate_memory_report() const;
    bool export_memory_usage_data(const std::string& filename) const;
    bool validate_memory_integrity() const;
    std::vector<std::string> detect_memory_leaks() const;

private:
    // Device and pool management
    std::vector<models::GPUInfo> gpu_devices_;
    std::unordered_map<int, std::unordered_map<MemoryPoolType, std::unique_ptr<MemoryPool>>> memory_pools_;
    
    // Allocation tracking
    std::unordered_map<uint64_t, MemoryAllocation> active_allocations_;
    std::atomic<uint64_t> next_allocation_id_;
    mutable std::mutex allocation_mutex_;
    
    // Statistics and monitoring
    MemoryStatistics global_statistics_;
    std::unordered_map<int, MemoryStatistics> device_statistics_;
    std::deque<MemoryPressureEvent> pressure_event_history_;
    mutable std::mutex statistics_mutex_;
    
    // Background tasks
    std::atomic<bool> background_tasks_running_;
    std::thread defragmentation_thread_;
    std::thread monitoring_thread_;
    std::thread optimization_thread_;
    
    // Configuration
    std::unordered_map<MemoryPoolType, AllocationStrategy> pool_strategies_;
    std::unordered_map<int, size_t> device_memory_limits_;
    bool memory_compression_enabled_;
    bool memory_prefetching_enabled_;
    bool memory_overcommit_enabled_;
    
    // Callbacks
    std::function<void(const MemoryAllocation&)> allocation_callback_;
    std::function<void(uint64_t)> deallocation_callback_;
    
    // Internal methods
    
    // Pool management
    bool initialize_device_pools(int device_id);
    MemoryPool* get_pool(MemoryPoolType type, int device_id);
    bool create_default_pools(int device_id);
    
    // Allocation helpers
    MemoryAllocation allocate_from_pool(const MemoryAllocationRequest& request);
    MemoryAllocation fallback_allocation(const MemoryAllocationRequest& request);
    bool validate_allocation_request(const MemoryAllocationRequest& request) const;
    AllocationStrategy select_optimal_strategy(const MemoryAllocationRequest& request) const;
    
    // Memory pressure detection
    bool detect_memory_pressure(int device_id);
    double calculate_pressure_level(int device_id) const;
    void handle_out_of_memory(int device_id);
    
    // Background task implementations
    void defragmentation_task_loop();
    void monitoring_task_loop();
    void optimization_task_loop();
    
    // Analytics helpers
    void update_statistics(const MemoryAllocation& allocation);
    void update_deallocation_statistics(uint64_t allocation_id);
    double calculate_fragmentation_ratio(int device_id) const;
    
    // Optimization helpers
    bool should_defragment_pool(MemoryPoolType type, int device_id) const;
    std::vector<MemoryAllocation> find_relocatable_allocations(MemoryPoolType type, int device_id);
    bool attempt_allocation_migration(const MemoryAllocation& allocation);
    
    // Utility methods
    size_t get_available_device_memory(int device_id) const;
    size_t get_total_device_memory(int device_id) const;
    bool is_device_valid(int device_id) const;
    MemoryPoolType classify_allocation_size(size_t size) const;
    void cleanup_expired_allocations();
};

/**
 * @brief Memory pool implementation
 */
class MemoryPool {
public:
    MemoryPool(MemoryPoolType type, const MemoryPoolConfig& config, int device_id);
    ~MemoryPool();
    
    // Allocation and deallocation
    MemoryAllocation allocate(size_t size, size_t alignment, const std::string& tag);
    bool deallocate(void* ptr);
    bool deallocate(uint64_t allocation_id);
    
    // Pool management
    bool resize(size_t new_size);
    bool defragment();
    bool compact();
    double get_fragmentation_ratio() const;
    
    // Statistics
    size_t get_total_size() const;
    size_t get_allocated_size() const;
    size_t get_free_size() const;
    size_t get_largest_free_block() const;
    size_t get_allocation_count() const;
    
    // Configuration
    void set_allocation_strategy(AllocationStrategy strategy);
    AllocationStrategy get_allocation_strategy() const;
    void set_defragmentation_threshold(double threshold);

private:
    struct FreeBlock {
        void* ptr;
        size_t size;
        bool operator<(const FreeBlock& other) const {
            return size < other.size; // For best-fit strategy
        }
    };
    
    struct AllocatedBlock {
        void* ptr;
        size_t size;
        uint64_t allocation_id;
        std::chrono::system_clock::time_point allocation_time;
        std::string tag;
    };
    
    MemoryPoolType pool_type_;
    MemoryPoolConfig config_;
    int device_id_;
    
    void* pool_base_ptr_;
    size_t pool_size_;
    
    std::vector<FreeBlock> free_blocks_;
    std::unordered_map<void*, AllocatedBlock> allocated_blocks_;
    std::unordered_map<uint64_t, void*> allocation_id_map_;
    
    mutable std::mutex pool_mutex_;
    std::atomic<uint64_t> next_allocation_id_;
    
    // Internal allocation strategies
    void* allocate_first_fit(size_t size, size_t alignment);
    void* allocate_best_fit(size_t size, size_t alignment);
    void* allocate_worst_fit(size_t size, size_t alignment);
    void* allocate_buddy_system(size_t size, size_t alignment);
    
    // Helper methods
    void coalesce_free_blocks();
    bool split_block(std::vector<FreeBlock>::iterator& it, size_t size);
    void* align_pointer(void* ptr, size_t alignment);
    size_t align_size(size_t size, size_t alignment);
    bool is_valid_allocation(void* ptr) const;
};

/**
 * @brief Memory management utility functions
 */
namespace memory_utils {
    
    // Size utilities
    size_t align_to_boundary(size_t size, size_t boundary);
    size_t next_power_of_two(size_t size);
    std::string format_memory_size(size_t bytes);
    size_t parse_memory_size(const std::string& size_str); // e.g., "1.5GB" -> bytes
    
    // GPU memory utilities
    bool check_cuda_memory_error(cudaError_t error);
    size_t get_gpu_memory_info(int device_id, size_t* free_mem = nullptr);
    bool set_memory_growth_policy(int device_id, bool enable_growth);
    
    // Performance utilities
    double calculate_memory_bandwidth_utilization(size_t bytes_transferred, 
                                                std::chrono::microseconds duration,
                                                size_t peak_bandwidth);
    bool warm_memory_region(void* ptr, size_t size);
    bool prefetch_memory_region(void* ptr, size_t size, int device_id);
    
    // Analysis utilities
    std::vector<std::pair<void*, size_t>> analyze_memory_layout(const std::vector<MemoryAllocation>& allocations);
    double calculate_external_fragmentation(const std::vector<size_t>& free_blocks);
    double calculate_internal_fragmentation(const std::vector<std::pair<size_t, size_t>>& allocated_requested_pairs);
    
    // Optimization utilities
    AllocationStrategy recommend_strategy_for_pattern(const std::vector<size_t>& allocation_sizes);
    size_t recommend_pool_size(MemoryPoolType type, const std::vector<size_t>& historical_usage);
    bool should_enable_defragmentation(double fragmentation_ratio, size_t pool_size);
}

} // namespace memory
} // namespace keyhunt