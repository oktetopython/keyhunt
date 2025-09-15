/**
 * @file gpu_memory_manager.h
 * @brief GPU memory management and optimization for secp256k1 operations
 * @author KeyhuntCUDA Team
 * 
 * T035: Implement GPU memory management and optimization for secp256k1 operations
 * 
 * Provides comprehensive GPU memory management including memory pooling,
 * batch optimization, CUDA stream management, and performance monitoring
 * for large-scale secp256k1 operations.
 */

#pragma once

#include "secp256k1.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <queue>
#include <unordered_map>

namespace keyhunt {
namespace ecc {
namespace gpu {

/**
 * @brief Memory allocation strategies for different use cases
 */
enum class MemoryStrategy {
    COALESCED,         // Optimize for coalesced memory access
    PITCHED,           // Use pitched memory for 2D operations
    UNIFIED,           // Use unified memory for CPU-GPU coordination
    PAGEABLE,          // Standard pageable memory
    PINNED,            // Pinned (page-locked) memory for fast transfers
    MANAGED            // CUDA managed memory with automatic migration
};

/**
 * @brief Memory pool configuration
 */
struct MemoryPoolConfig {
    size_t initial_size;      // Initial pool size in bytes
    size_t max_size;          // Maximum pool size in bytes
    size_t block_size;        // Standard block size
    size_t alignment;         // Memory alignment requirements
    MemoryStrategy strategy;  // Memory allocation strategy
    bool enable_defrag;       // Enable memory defragmentation
    double growth_factor;     // Pool growth factor (1.5 = 50% growth)
    
    MemoryPoolConfig() : initial_size(64 * 1024 * 1024), max_size(2ULL * 1024 * 1024 * 1024),
                        block_size(4096), alignment(256), strategy(MemoryStrategy::PINNED),
                        enable_defrag(true), growth_factor(1.5) {}
};

/**
 * @brief Memory allocation statistics
 */
struct MemoryStats {
    size_t total_allocated;      // Total memory allocated
    size_t total_free;           // Total free memory
    size_t peak_usage;           // Peak memory usage
    size_t current_usage;        // Current memory usage
    size_t num_allocations;      // Number of active allocations
    size_t num_deallocations;    // Total number of deallocations
    size_t fragmentation_ratio;  // Fragmentation ratio (0-100)
    double allocation_efficiency; // Allocation efficiency (0.0-1.0)
    std::chrono::milliseconds total_alloc_time; // Total allocation time
    std::chrono::milliseconds total_free_time;  // Total deallocation time
    
    MemoryStats() : total_allocated(0), total_free(0), peak_usage(0), current_usage(0),
                   num_allocations(0), num_deallocations(0), fragmentation_ratio(0),
                   allocation_efficiency(1.0), total_alloc_time(0), total_free_time(0) {}
};

/**
 * @brief Memory block descriptor
 */
struct MemoryBlock {
    void* ptr;                // Memory pointer
    size_t size;              // Block size
    size_t offset;            // Offset from pool start
    bool is_free;             // Free status
    MemoryStrategy strategy;  // Allocation strategy used
    std::chrono::steady_clock::time_point alloc_time; // Allocation timestamp
    uint64_t alloc_id;        // Unique allocation ID
    
    MemoryBlock(void* p, size_t s, size_t o, MemoryStrategy st) 
        : ptr(p), size(s), offset(o), is_free(false), strategy(st), 
          alloc_time(std::chrono::steady_clock::now()), alloc_id(0) {}
};

/**
 * @brief CUDA stream wrapper with performance tracking
 */
class ManagedStream {
public:
    ManagedStream(int priority = 0);
    ~ManagedStream();
    
    cudaStream_t get_stream() const { return stream_; }
    bool is_ready() const;
    void synchronize();
    
    // Performance tracking
    void record_operation_start();
    void record_operation_end();
    double get_utilization() const;
    size_t get_operation_count() const;
    
private:
    cudaStream_t stream_;
    cudaEvent_t start_event_;
    cudaEvent_t end_event_;
    mutable std::mutex metrics_mutex_;
    size_t operation_count_;
    std::chrono::milliseconds total_time_;
    std::chrono::steady_clock::time_point last_start_;
};

/**
 * @brief GPU memory pool for efficient allocation/deallocation
 */
class GPUMemoryPool {
public:
    GPUMemoryPool(const MemoryPoolConfig& config, int device_id = 0);
    ~GPUMemoryPool();
    
    bool initialize();
    void cleanup();
    
    // Memory allocation/deallocation
    void* allocate(size_t size, MemoryStrategy strategy = MemoryStrategy::PINNED);
    void deallocate(void* ptr);
    
    // Bulk operations
    std::vector<void*> allocate_batch(const std::vector<size_t>& sizes, MemoryStrategy strategy);
    void deallocate_batch(const std::vector<void*>& ptrs);
    
    // Memory management
    void defragment();
    void reset();
    double get_fragmentation_ratio() const;
    
    // Statistics and monitoring
    MemoryStats get_stats() const;
    size_t get_available_memory() const;
    size_t get_largest_free_block() const;
    
    // Configuration
    void set_growth_factor(double factor);
    void enable_defragmentation(bool enable);
    
private:
    MemoryPoolConfig config_;
    int device_id_;
    bool initialized_;
    
    void* pool_ptr_;
    size_t pool_size_;
    std::vector<MemoryBlock> blocks_;
    std::map<void*, size_t> allocation_map_;
    
    mutable std::mutex pool_mutex_;
    std::atomic<uint64_t> next_alloc_id_;
    
    MemoryStats stats_;
    
    // Internal methods
    void* allocate_from_pool(size_t size, MemoryStrategy strategy);
    bool expand_pool();
    void merge_free_blocks();
    void update_stats();
};

/**
 * @brief Batch operation optimizer for secp256k1 operations
 */
class BatchOptimizer {
public:
    struct BatchConfig {
        size_t min_batch_size;      // Minimum batch size for GPU execution
        size_t optimal_batch_size;  // Optimal batch size for current GPU
        size_t max_batch_size;      // Maximum batch size
        size_t memory_limit;        // Memory limit for batches
        bool enable_streaming;      // Enable CUDA streaming
        int num_streams;            // Number of concurrent streams
        
        BatchConfig() : min_batch_size(1000), optimal_batch_size(10000),
                       max_batch_size(100000), memory_limit(512 * 1024 * 1024),
                       enable_streaming(true), num_streams(4) {}
    };
    
    BatchOptimizer(const BatchConfig& config, GPUMemoryPool* memory_pool);
    ~BatchOptimizer();
    
    bool initialize();
    void cleanup();
    
    // Batch size optimization
    size_t calculate_optimal_batch_size(size_t total_operations, size_t element_size);
    std::vector<size_t> split_into_batches(size_t total_operations, size_t max_memory);
    
    // Memory layout optimization
    void optimize_memory_layout(void** ptrs, size_t count, size_t element_size);
    
    // Stream management
    ManagedStream* get_available_stream();
    void return_stream(ManagedStream* stream);
    
    // Performance monitoring
    struct BatchMetrics {
        size_t operations_per_second;
        double memory_bandwidth;
        double gpu_utilization;
        size_t optimal_batch_size_current;
    };
    
    BatchMetrics get_metrics() const;
    void update_optimal_parameters();

private:
    BatchConfig config_;
    GPUMemoryPool* memory_pool_;
    bool initialized_;
    
    std::vector<std::unique_ptr<ManagedStream>> streams_;
    std::queue<ManagedStream*> available_streams_;
    mutable std::mutex stream_mutex_;
    
    // Performance tracking
    mutable BatchMetrics current_metrics_;
    std::chrono::steady_clock::time_point last_update_;
};

/**
 * @brief secp256k1 specific GPU memory layouts
 */
namespace layouts {
    
    /**
     * @brief Scalar array layout optimized for coalesced access
     */
    struct ScalarArray {
        uint64_t* data;       // Scalar data (4 x uint64_t per scalar)
        size_t count;         // Number of scalars
        size_t stride;        // Stride between scalars
        
        static size_t calculate_size(size_t scalar_count);
        static void* allocate_optimized(GPUMemoryPool* pool, size_t scalar_count);
    };
    
    /**
     * @brief Point array layout for elliptic curve points
     */
    struct PointArray {
        uint64_t* x_coords;   // X coordinates
        uint64_t* y_coords;   // Y coordinates  
        uint64_t* z_coords;   // Z coordinates (projective)
        size_t count;         // Number of points
        size_t stride;        // Stride between coordinates
        
        static size_t calculate_size(size_t point_count);
        static void* allocate_optimized(GPUMemoryPool* pool, size_t point_count);
    };
    
    /**
     * @brief Precomputed table layout for fast scalar multiplication
     */
    struct PrecomputedTable {
        uint64_t* table_data; // Precomputed points
        size_t table_size;    // Size of precomputed table
        size_t window_size;   // Window size for windowed method
        
        static size_t calculate_size(size_t window_size);
        static void* allocate_optimized(GPUMemoryPool* pool, size_t window_size);
    };
}

/**
 * @brief High-level GPU memory manager for secp256k1 operations
 */
class Secp256k1MemoryManager {
public:
    struct Config {
        MemoryPoolConfig pool_config;
        BatchOptimizer::BatchConfig batch_config;
        int device_id;
        bool enable_peer_access;
        bool enable_memory_profiling;
        
        Config() : device_id(0), enable_peer_access(true), enable_memory_profiling(false) {}
    };
    
    Secp256k1MemoryManager(const Config& config);
    ~Secp256k1MemoryManager();
    
    bool initialize();
    void cleanup();
    
    // Memory allocation for secp256k1 operations
    layouts::ScalarArray allocate_scalars(size_t count);
    layouts::PointArray allocate_points(size_t count);
    layouts::PrecomputedTable allocate_precomputed_table(size_t window_size);
    
    void deallocate_scalars(const layouts::ScalarArray& array);
    void deallocate_points(const layouts::PointArray& array);
    void deallocate_precomputed_table(const layouts::PrecomputedTable& table);
    
    // Batch operations
    std::vector<size_t> optimize_batch_sizes(size_t total_scalars, size_t available_memory);
    ManagedStream* get_compute_stream();
    ManagedStream* get_transfer_stream();
    
    // Memory transfer utilities
    cudaError_t copy_scalars_to_device(const std::vector<BigInt256>& host_scalars, 
                                      const layouts::ScalarArray& device_array,
                                      ManagedStream* stream = nullptr);
    
    cudaError_t copy_points_from_device(const layouts::PointArray& device_array,
                                       std::vector<Point>& host_points,
                                       ManagedStream* stream = nullptr);
    
    // Performance monitoring
    struct PerformanceReport {
        MemoryStats memory_stats;
        BatchOptimizer::BatchMetrics batch_metrics;
        double memory_bandwidth_gbps;
        double effective_memory_usage;
        size_t operations_per_second;
    };
    
    PerformanceReport generate_performance_report() const;
    void reset_performance_counters();
    
    // Device information
    size_t get_device_memory_size() const;
    size_t get_available_device_memory() const;
    int get_device_compute_capability() const;
    
private:
    Config config_;
    bool initialized_;
    
    std::unique_ptr<GPUMemoryPool> memory_pool_;
    std::unique_ptr<BatchOptimizer> batch_optimizer_;
    
    // Device properties
    cudaDeviceProp device_props_;
    size_t total_device_memory_;
    size_t available_device_memory_;
    
    // Stream management
    std::vector<std::unique_ptr<ManagedStream>> compute_streams_;
    std::vector<std::unique_ptr<ManagedStream>> transfer_streams_;
    mutable std::mutex stream_mutex_;
    
    // Performance tracking
    mutable std::chrono::steady_clock::time_point last_report_time_;
    mutable PerformanceReport cached_report_;
    
    // Internal methods
    bool query_device_properties();
    void setup_peer_access();
    size_t calculate_memory_requirement(size_t operations);
};

/**
 * @brief Global memory manager factory and registry
 */
class MemoryManagerRegistry {
public:
    static Secp256k1MemoryManager* get_manager(int device_id = 0);
    static bool register_manager(int device_id, std::unique_ptr<Secp256k1MemoryManager> manager);
    static void cleanup_all();
    
    // Global configuration
    static void set_global_config(const Secp256k1MemoryManager::Config& config);
    static Secp256k1MemoryManager::Config get_global_config();
    
private:
    static std::unordered_map<int, std::unique_ptr<Secp256k1MemoryManager>> managers_;
    static Secp256k1MemoryManager::Config global_config_;
    static std::mutex registry_mutex_;
};

} // namespace gpu
} // namespace ecc
} // namespace keyhunt