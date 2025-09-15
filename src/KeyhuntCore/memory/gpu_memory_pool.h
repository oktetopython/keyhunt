/**
 * @file gpu_memory_pool.h
 * @brief Specialized GPU memory pool implementations with different allocation strategies
 * @author KeyhuntCUDA Team
 * 
 * T047: GPU-specific memory pools with advanced allocation algorithms
 */

#pragma once

#include "advanced_memory_manager.h"
#include <cuda_runtime.h>
#include <vector>
#include <set>
#include <stack>
#include <queue>

namespace keyhunt {
namespace memory {

/**
 * @brief Buddy system memory allocator
 */
class BuddyAllocator {
public:
    BuddyAllocator(void* base_ptr, size_t total_size);
    ~BuddyAllocator();
    
    void* allocate(size_t size);
    bool deallocate(void* ptr);
    double get_fragmentation_ratio() const;
    size_t get_largest_free_block() const;
    
private:
    struct Block {
        void* ptr;
        size_t size;
        bool is_free;
        size_t level; // Level in buddy tree
        
        Block(void* p, size_t s, size_t l) 
            : ptr(p), size(s), is_free(true), level(l) {}
    };
    
    void* base_ptr_;
    size_t total_size_;
    size_t min_block_size_;
    size_t max_level_;
    
    std::vector<std::set<Block*>> free_lists_; // One set per level
    std::unordered_map<void*, Block*> allocated_blocks_;
    
    size_t size_to_level(size_t size) const;
    size_t level_to_size(size_t level) const;
    Block* find_buddy(Block* block) const;
    void coalesce(Block* block);
    Block* split_block(Block* block, size_t target_level);
};

/**
 * @brief Slab allocator for fixed-size objects
 */
class SlabAllocator {
public:
    SlabAllocator(void* base_ptr, size_t total_size, size_t object_size);
    ~SlabAllocator();
    
    void* allocate();
    bool deallocate(void* ptr);
    size_t get_free_count() const;
    size_t get_total_count() const;
    double get_utilization() const;
    
private:
    void* base_ptr_;
    size_t total_size_;
    size_t object_size_;
    size_t object_count_;
    
    std::stack<void*> free_objects_;
    std::unordered_set<void*> allocated_objects_;
    
    void initialize_free_list();
};

/**
 * @brief Stack allocator for LIFO allocations
 */
class StackAllocator {
public:
    StackAllocator(void* base_ptr, size_t total_size);
    ~StackAllocator();
    
    void* allocate(size_t size, size_t alignment = 1);
    bool deallocate(void* ptr); // Must be last allocated
    void reset(); // Reset to beginning
    size_t get_used_size() const;
    size_t get_remaining_size() const;
    
private:
    struct Marker {
        void* ptr;
        size_t offset;
    };
    
    void* base_ptr_;
    size_t total_size_;
    size_t current_offset_;
    std::stack<Marker> markers_;
    
    void* align_pointer(void* ptr, size_t alignment);
};

/**
 * @brief Linear allocator for sequential allocations
 */
class LinearAllocator {
public:
    LinearAllocator(void* base_ptr, size_t total_size);
    ~LinearAllocator();
    
    void* allocate(size_t size, size_t alignment = 1);
    void reset(); // Clear all allocations
    size_t get_used_size() const;
    size_t get_remaining_size() const;
    
    // Markers for partial resets
    struct Marker {
        size_t offset;
    };
    
    Marker get_marker() const;
    void reset_to_marker(const Marker& marker);
    
private:
    void* base_ptr_;
    size_t total_size_;
    size_t current_offset_;
    
    void* align_pointer(void* ptr, size_t alignment);
};

/**
 * @brief Pool allocator for fixed-size blocks with multiple sizes
 */
class PoolAllocator {
public:
    struct PoolConfig {
        size_t block_size;
        size_t block_count;
    };
    
    PoolAllocator(void* base_ptr, size_t total_size, const std::vector<PoolConfig>& configs);
    ~PoolAllocator();
    
    void* allocate(size_t size);
    bool deallocate(void* ptr);
    size_t get_pool_count() const;
    double get_overall_utilization() const;
    
private:
    struct Pool {
        size_t block_size;
        size_t block_count;
        std::queue<void*> free_blocks;
        std::unordered_set<void*> allocated_blocks;
        void* pool_base;
        
        Pool(void* base, size_t bsize, size_t bcount) 
            : block_size(bsize), block_count(bcount), pool_base(base) {
            // Initialize free blocks
            for (size_t i = 0; i < block_count; ++i) {
                free_blocks.push(static_cast<char*>(base) + i * block_size);
            }
        }
    };
    
    void* base_ptr_;
    size_t total_size_;
    std::vector<std::unique_ptr<Pool>> pools_;
    
    Pool* find_suitable_pool(size_t size);
    void* allocate_from_pool(Pool* pool);
    bool deallocate_from_pool(Pool* pool, void* ptr);
};

/**
 * @brief GPU-specific memory pool with CUDA optimizations
 */
class GPUMemoryPool {
public:
    GPUMemoryPool(int device_id, size_t initial_size, AllocationStrategy strategy);
    ~GPUMemoryPool();
    
    // Allocation and deallocation
    void* allocate(size_t size, size_t alignment = 256);
    bool deallocate(void* ptr);
    
    // GPU-specific operations
    bool prefetch_to_device();
    bool prefetch_to_host();
    bool set_memory_advice(cudaMemoryAdvise advice);
    bool warm_up_pool(); // Touch all memory to ensure physical allocation
    
    // Stream-ordered operations
    void* allocate_async(size_t size, cudaStream_t stream);
    bool deallocate_async(void* ptr, cudaStream_t stream);
    
    // Memory mapping and access patterns
    bool set_preferred_location(int device_id);
    bool set_accessed_by(int device_id);
    bool optimize_for_access_pattern(const std::string& pattern); // "sequential", "random", "streaming"
    
    // Performance monitoring
    struct PerformanceMetrics {
        size_t allocations_per_second;
        size_t deallocations_per_second;
        std::chrono::microseconds average_allocation_time;
        double cache_hit_ratio;
        size_t memory_bandwidth_utilization;
    };
    
    PerformanceMetrics get_performance_metrics() const;
    
    // Pool management
    bool resize(size_t new_size);
    bool defragment();
    double get_fragmentation_ratio() const;
    size_t get_total_size() const;
    size_t get_free_size() const;
    size_t get_allocated_size() const;
    
    // Advanced features
    bool enable_memory_pooling_hints(bool enable = true);
    bool set_allocation_granularity(size_t granularity);
    bool enable_oversubscription(bool enable = true);

private:
    int device_id_;
    size_t pool_size_;
    void* pool_base_;
    AllocationStrategy strategy_;
    
    // Strategy-specific allocators
    std::unique_ptr<BuddyAllocator> buddy_allocator_;
    std::unique_ptr<SlabAllocator> slab_allocator_;
    std::unique_ptr<StackAllocator> stack_allocator_;
    std::unique_ptr<LinearAllocator> linear_allocator_;
    std::unique_ptr<PoolAllocator> pool_allocator_;
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics metrics_;
    std::chrono::high_resolution_clock::time_point last_metrics_update_;
    
    // Stream management for async operations
    std::unordered_map<cudaStream_t, std::vector<void*>> stream_allocations_;
    
    // Memory access optimization
    cudaMemoryAdvise current_advice_;
    int preferred_location_;
    std::set<int> accessing_devices_;
    
    // Internal methods
    bool initialize_pool();
    bool setup_allocator();
    void update_performance_metrics();
    bool validate_allocation(void* ptr, size_t size) const;
    void* allocate_with_strategy(size_t size, size_t alignment);
    bool deallocate_with_strategy(void* ptr);
};

/**
 * @brief Memory pool factory for creating optimized pools
 */
class MemoryPoolFactory {
public:
    static std::unique_ptr<GPUMemoryPool> create_ecc_pool(int device_id, size_t size);
    static std::unique_ptr<GPUMemoryPool> create_hash_pool(int device_id, size_t size);
    static std::unique_ptr<GPUMemoryPool> create_scanning_pool(int device_id, size_t size);
    static std::unique_ptr<GPUMemoryPool> create_temporary_pool(int device_id, size_t size);
    static std::unique_ptr<GPUMemoryPool> create_cache_pool(int device_id, size_t size);
    
    // Create pool based on usage pattern analysis
    static std::unique_ptr<GPUMemoryPool> create_optimized_pool(
        int device_id, 
        size_t size, 
        const std::vector<size_t>& allocation_history
    );
    
    // Create pool based on hardware characteristics
    static std::unique_ptr<GPUMemoryPool> create_hardware_optimized_pool(
        const models::GPUInfo& gpu_info, 
        MemoryPoolType pool_type
    );

private:
    static AllocationStrategy recommend_strategy_for_usage(const std::vector<size_t>& allocation_sizes);
    static size_t calculate_optimal_pool_size(const models::GPUInfo& gpu_info, MemoryPoolType pool_type);
};

/**
 * @brief Memory pool monitoring and analytics
 */
class MemoryPoolMonitor {
public:
    MemoryPoolMonitor();
    ~MemoryPoolMonitor();
    
    // Registration and monitoring
    bool register_pool(const std::string& name, GPUMemoryPool* pool);
    bool unregister_pool(const std::string& name);
    
    // Analytics
    struct PoolAnalytics {
        std::string pool_name;
        double utilization_ratio;
        double fragmentation_ratio;
        size_t allocation_rate; // allocations per second
        size_t deallocation_rate; // deallocations per second
        std::chrono::microseconds average_allocation_time;
        std::vector<size_t> allocation_size_histogram;
        double cache_efficiency;
    };
    
    std::vector<PoolAnalytics> get_all_analytics() const;
    PoolAnalytics get_pool_analytics(const std::string& name) const;
    
    // Optimization recommendations
    struct OptimizationRecommendation {
        std::string pool_name;
        std::string recommendation_type;
        std::string description;
        double expected_improvement;
        std::string action_required;
    };
    
    std::vector<OptimizationRecommendation> generate_recommendations() const;
    
    // Alerting
    bool set_utilization_threshold(const std::string& pool_name, double threshold);
    bool set_fragmentation_threshold(const std::string& pool_name, double threshold);
    bool set_alert_callback(std::function<void(const std::string&, const std::string&)> callback);
    
    // Reporting
    std::string generate_monitoring_report() const;
    bool export_analytics_data(const std::string& filename) const;

private:
    std::unordered_map<std::string, GPUMemoryPool*> monitored_pools_;
    std::unordered_map<std::string, double> utilization_thresholds_;
    std::unordered_map<std::string, double> fragmentation_thresholds_;
    std::function<void(const std::string&, const std::string&)> alert_callback_;
    
    mutable std::mutex monitor_mutex_;
    std::thread monitoring_thread_;
    std::atomic<bool> monitoring_active_;
    
    void monitoring_loop();
    void check_thresholds();
    void update_analytics();
};

} // namespace memory
} // namespace keyhunt