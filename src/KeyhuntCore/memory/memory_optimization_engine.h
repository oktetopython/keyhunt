/**
 * @file memory_optimization_engine.h
 * @brief Memory optimization engine with predictive allocation and intelligent caching
 * @author KeyhuntCUDA Team
 * 
 * T047: Advanced memory optimization engine for predictive allocation patterns and intelligent caching strategies
 */

#pragma once

#include "gpu_memory_manager.h"
#include <unordered_map>
#include <deque>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

namespace keyhunt {
namespace memory {

/**
 * @brief Memory access pattern types
 */
enum class AccessPattern {
    SEQUENTIAL,         // Sequential access pattern
    RANDOM,            // Random access pattern
    STRIDED,           // Strided access pattern
    TEMPORAL_LOCALITY, // High temporal locality
    SPATIAL_LOCALITY,  // High spatial locality
    MIXED              // Mixed access patterns
};

/**
 * @brief Memory usage prediction model
 */
struct MemoryUsagePrediction {
    std::chrono::system_clock::time_point prediction_time;
    size_t predicted_memory_need;
    std::chrono::milliseconds prediction_horizon;
    double confidence_level;        // 0.0 to 1.0
    std::string workload_phase;    // "initialization", "scanning", "comparison", etc.
    AccessPattern expected_pattern;
    
    MemoryUsagePrediction()
        : prediction_time(std::chrono::system_clock::now())
        , predicted_memory_need(0)
        , prediction_horizon(std::chrono::milliseconds(0))
        , confidence_level(0.0)
        , workload_phase("unknown")
        , expected_pattern(AccessPattern::MIXED)
    {}
};

/**
 * @brief Cache optimization configuration
 */
struct CacheOptimizationConfig {
    size_t max_cache_size_per_category;    // Maximum cache size per category
    std::chrono::seconds cache_timeout;     // Cache entry timeout
    double cache_hit_target_ratio;          // Target cache hit ratio
    bool enable_predictive_caching;         // Enable predictive cache loading
    bool enable_cache_warming;              // Enable cache pre-warming
    double cache_utilization_threshold;     // Cleanup threshold
    size_t max_prefetch_queue_size;         // Maximum prefetch queue size
    
    CacheOptimizationConfig()
        : max_cache_size_per_category(256 * 1024 * 1024) // 256MB per category
        , cache_timeout(std::chrono::seconds(300)) // 5 minutes
        , cache_hit_target_ratio(0.8) // 80% hit ratio target
        , enable_predictive_caching(true)
        , enable_cache_warming(true)
        , cache_utilization_threshold(0.9) // 90% utilization
        , max_prefetch_queue_size(100)
    {}
};

/**
 * @brief Memory workload analyzer
 */
class MemoryWorkloadAnalyzer {
public:
    MemoryWorkloadAnalyzer();
    ~MemoryWorkloadAnalyzer();
    
    // Workload analysis
    bool start_analysis();
    bool stop_analysis();
    void record_allocation(size_t size, const std::string& category, const AllocationHint& hint);
    void record_deallocation(size_t size, const std::string& category);
    void record_memory_access(void* ptr, AccessPattern pattern);
    
    // Pattern recognition
    struct WorkloadPattern {
        std::string phase_name;
        std::vector<size_t> typical_allocation_sizes;
        std::vector<std::string> active_categories;
        AccessPattern dominant_access_pattern;
        std::chrono::milliseconds phase_duration;
        size_t peak_memory_usage;
        double allocation_frequency; // Allocations per second
    };
    
    std::vector<WorkloadPattern> identify_workload_patterns();
    WorkloadPattern predict_next_phase() const;
    
    // Memory usage prediction
    MemoryUsagePrediction predict_memory_usage(std::chrono::minutes horizon);
    std::vector<MemoryUsagePrediction> predict_usage_timeline(std::chrono::hours horizon, 
                                                             std::chrono::minutes resolution);
    
    // Analysis results
    struct AnalysisResults {
        std::unordered_map<std::string, size_t> category_usage;
        std::unordered_map<AccessPattern, double> access_pattern_frequency;
        std::vector<std::pair<size_t, size_t>> size_distribution; // size, frequency
        double temporal_locality_score;
        double spatial_locality_score;
        std::chrono::milliseconds average_allocation_lifetime;
    };
    
    AnalysisResults get_analysis_results() const;

private:
    std::atomic<bool> analysis_active_;
    std::thread analysis_thread_;
    
    // Allocation tracking
    struct AllocationRecord {
        std::chrono::system_clock::time_point timestamp;
        size_t size;
        std::string category;
        AllocationHint hint;
        bool is_deallocation;
    };
    
    std::deque<AllocationRecord> allocation_history_;
    mutable std::mutex history_mutex_;
    
    // Access pattern tracking
    struct AccessRecord {
        std::chrono::system_clock::time_point timestamp;
        void* ptr;
        AccessPattern pattern;
    };
    
    std::deque<AccessRecord> access_history_;
    mutable std::mutex access_mutex_;
    
    // Pattern recognition state
    std::vector<WorkloadPattern> identified_patterns_;
    std::string current_phase_;
    mutable std::mutex pattern_mutex_;
    
    // Internal methods
    void analysis_thread_loop();
    void analyze_current_workload();
    WorkloadPattern extract_pattern_from_history(const std::vector<AllocationRecord>& records);
    double calculate_pattern_similarity(const WorkloadPattern& p1, const WorkloadPattern& p2);
    void cleanup_old_records();
};

/**
 * @brief Intelligent cache manager with predictive capabilities
 */
class IntelligentCacheManager {
public:
    IntelligentCacheManager(const CacheOptimizationConfig& config);
    ~IntelligentCacheManager();
    
    // Cache operations
    bool initialize();
    void* get_from_cache(const std::string& category, size_t size, const AllocationHint& hint);
    bool put_in_cache(const std::string& category, void* ptr, size_t size, const AllocationHint& hint);
    bool invalidate_cache_entry(void* ptr);
    void flush_category_cache(const std::string& category);
    void flush_all_cache();
    
    // Predictive caching
    bool prefetch_for_pattern(const MemoryWorkloadAnalyzer::WorkloadPattern& pattern);
    bool warm_cache_for_phase(const std::string& phase_name);
    void set_workload_analyzer(MemoryWorkloadAnalyzer* analyzer);
    
    // Cache optimization
    bool optimize_cache_sizes();
    bool rebalance_cache_across_categories();
    void evict_least_useful_entries();
    
    // Statistics and monitoring
    struct CacheStatistics {
        std::unordered_map<std::string, size_t> hits_by_category;
        std::unordered_map<std::string, size_t> misses_by_category;
        std::unordered_map<std::string, size_t> cache_size_by_category;
        size_t total_cache_memory_used;
        double overall_hit_ratio;
        size_t prefetch_hits;
        size_t prefetch_waste;
        std::chrono::milliseconds average_cache_lookup_time;
    };
    
    CacheStatistics get_cache_statistics() const;
    std::string generate_cache_report() const;
    void cleanup();

private:
    CacheOptimizationConfig config_;
    MemoryWorkloadAnalyzer* workload_analyzer_;
    
    // Cache storage
    struct CacheEntry {
        void* ptr;
        size_t size;
        std::chrono::system_clock::time_point cache_time;
        std::chrono::system_clock::time_point last_access;
        AllocationHint hint;
        size_t access_count;
        bool is_prefetched;
    };
    
    std::unordered_map<std::string, std::deque<CacheEntry>> category_caches_;
    mutable std::shared_mutex cache_mutex_;
    
    // Prefetch management
    std::queue<std::pair<std::string, size_t>> prefetch_queue_;
    std::thread prefetch_thread_;
    std::atomic<bool> prefetch_thread_running_;
    mutable std::mutex prefetch_mutex_;
    
    // Cache optimization
    std::thread optimization_thread_;
    std::atomic<bool> optimization_thread_running_;
    std::chrono::seconds optimization_interval_;
    
    // Statistics
    CacheStatistics cache_stats_;
    mutable std::mutex stats_mutex_;
    
    // Internal methods
    void prefetch_thread_loop();
    void optimization_thread_loop();
    CacheEntry* find_best_match(const std::string& category, size_t size);
    bool should_cache_entry(const std::string& category, size_t size, const AllocationHint& hint);
    void cleanup_expired_entries();
    void update_access_statistics(const std::string& category, bool hit);
    double calculate_entry_utility(const CacheEntry& entry) const;
};

/**
 * @brief Memory optimization engine coordinating all optimization strategies
 */
class MemoryOptimizationEngine {
public:
    MemoryOptimizationEngine();
    ~MemoryOptimizationEngine();
    
    // Initialization
    bool initialize(MultiGPUMemoryManager* memory_manager, 
                   const CacheOptimizationConfig& cache_config = CacheOptimizationConfig());
    void cleanup();
    
    // Core optimization operations
    bool start_optimization();
    bool stop_optimization();
    bool optimize_memory_layout();
    bool optimize_allocation_strategy();
    bool optimize_cache_configuration();
    
    // Predictive optimization
    bool enable_predictive_optimization(bool enable = true);
    bool apply_workload_based_optimizations();
    bool prepare_for_predicted_workload(const MemoryWorkloadAnalyzer::WorkloadPattern& pattern);
    
    // Real-time adaptation
    bool adapt_to_current_workload();
    bool detect_and_optimize_bottlenecks();
    bool balance_memory_across_devices();
    
    // Configuration and monitoring
    struct OptimizationMetrics {
        double memory_utilization_efficiency;
        double cache_hit_ratio;
        double allocation_success_ratio;
        double fragmentation_level;
        size_t total_memory_saved;
        std::chrono::milliseconds average_allocation_latency;
        size_t optimization_actions_taken;
    };
    
    OptimizationMetrics get_optimization_metrics() const;
    std::string generate_optimization_report() const;
    bool export_optimization_data(const std::string& filename) const;
    
    // Advanced features
    bool enable_machine_learning_optimization(bool enable = true);
    bool train_optimization_models();
    bool apply_learned_optimizations();

private:
    MultiGPUMemoryManager* memory_manager_;
    std::unique_ptr<MemoryWorkloadAnalyzer> workload_analyzer_;
    std::unique_ptr<IntelligentCacheManager> cache_manager_;
    
    // Optimization state
    std::atomic<bool> optimization_active_;
    std::thread optimization_thread_;
    std::atomic<bool> optimization_thread_running_;
    
    // Machine learning state
    std::atomic<bool> ml_optimization_enabled_;
    std::vector<std::pair<MemoryWorkloadAnalyzer::WorkloadPattern, AllocationStrategy>> learned_strategies_;
    
    // Metrics tracking
    OptimizationMetrics current_metrics_;
    mutable std::mutex metrics_mutex_;
    
    // Internal methods
    void optimization_thread_loop();
    void collect_optimization_metrics();
    void apply_real_time_optimizations();
    
    // Strategy optimization
    AllocationStrategy recommend_strategy_for_workload(const MemoryWorkloadAnalyzer::WorkloadPattern& pattern);
    MemoryPoolConfig optimize_pool_config_for_workload(const MemoryWorkloadAnalyzer::WorkloadPattern& pattern);
    
    // Bottleneck detection
    enum class MemoryBottleneckType {
        FRAGMENTATION,
        CACHE_MISSES,
        ALLOCATION_LATENCY,
        BANDWIDTH_SATURATION,
        POOL_EXHAUSTION
    };
    
    std::vector<MemoryBottleneckType> detect_memory_bottlenecks();
    bool optimize_for_bottleneck(MemoryBottleneckType bottleneck);
    
    // Machine learning helpers
    void collect_training_data();
    bool update_learned_strategies();
    double evaluate_strategy_performance(AllocationStrategy strategy, 
                                        const MemoryWorkloadAnalyzer::WorkloadPattern& pattern);
};

/**
 * @brief Memory optimization utility functions
 */
namespace optimization_utils {
    
    // Workload analysis utilities
    AccessPattern classify_access_pattern(const std::vector<void*>& access_sequence);
    double calculate_temporal_locality(const std::vector<std::chrono::system_clock::time_point>& access_times);
    double calculate_spatial_locality(const std::vector<void*>& access_addresses);
    
    // Prediction utilities
    size_t predict_memory_requirement(const MemoryWorkloadAnalyzer::WorkloadPattern& pattern, 
                                     std::chrono::milliseconds time_horizon);
    double estimate_cache_hit_ratio(const std::vector<size_t>& allocation_sizes, size_t cache_size);
    
    // Optimization utilities
    AllocationStrategy recommend_strategy_for_pattern(AccessPattern pattern, double fragmentation);
    size_t calculate_optimal_cache_size(const std::vector<size_t>& allocation_history, 
                                       double target_hit_ratio);
    MemoryPoolConfig adapt_config_for_workload_shift(const MemoryPoolConfig& current_config,
                                                    const MemoryWorkloadAnalyzer::WorkloadPattern& new_pattern);
}

} // namespace memory
} // namespace keyhunt