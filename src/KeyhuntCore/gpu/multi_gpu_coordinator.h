/**
 * @file multi_gpu_coordinator.h
 * @brief Multi-GPU coordination system with dynamic load balancing and work distribution
 * @author KeyhuntCUDA Team
 * 
 * T043: Implement multi-GPU coordination system with dynamic load balancing and work distribution
 * 
 * Provides comprehensive multi-GPU coordination for private key scanning operations with
 * dynamic load balancing, work stealing, performance monitoring, and fault tolerance.
 */

#pragma once

#include "../models/GPUConfiguration.h"
#include "../models/PrivateKeyRange.h"
#include "../scan/private_key_scanner.h"
#include "../scan/checkpoint_manager.h"
#include "../ecc/secp256k1.h"
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <map>
#include <chrono>
#include <functional>
#include <cuda_runtime.h>
#include <nccl.h>

namespace keyhunt {
namespace gpu {
namespace coordination {

/**
 * @brief GPU device information and capabilities
 */
struct GPUDeviceInfo {
    int device_id;                              // CUDA device ID
    std::string device_name;                    // Device name (e.g., "GeForce RTX 4090")
    size_t total_memory;                        // Total GPU memory (bytes)
    size_t free_memory;                         // Available GPU memory (bytes)
    int compute_capability_major;               // Compute capability major version
    int compute_capability_minor;               // Compute capability minor version
    int multiprocessor_count;                   // Number of SMs
    int max_threads_per_block;                  // Maximum threads per block
    int max_blocks_per_multiprocessor;          // Maximum blocks per SM
    size_t shared_memory_per_block;             // Shared memory per block
    size_t constant_memory_size;                // Constant memory size
    
    // Performance characteristics
    double memory_bandwidth_gb_s;              // Memory bandwidth (GB/s)
    double peak_flops;                          // Peak FLOPS performance
    double power_consumption_watts;             // Power consumption (watts)
    double temperature_celsius;                 // Current temperature
    
    // Runtime status
    bool is_available;                          // Device availability
    bool is_busy;                              // Device busy status
    std::chrono::system_clock::time_point last_activity; // Last activity time
    size_t current_workload_size;              // Current workload size
    double current_utilization;                 // Current GPU utilization (0-1)
    
    GPUDeviceInfo() 
        : device_id(-1), total_memory(0), free_memory(0)
        , compute_capability_major(0), compute_capability_minor(0)
        , multiprocessor_count(0), max_threads_per_block(0)
        , max_blocks_per_multiprocessor(0), shared_memory_per_block(0)
        , constant_memory_size(0), memory_bandwidth_gb_s(0.0)
        , peak_flops(0.0), power_consumption_watts(0.0)
        , temperature_celsius(0.0), is_available(false)
        , is_busy(false), current_workload_size(0)
        , current_utilization(0.0)
    {
        last_activity = std::chrono::system_clock::now();
    }
};

/**
 * @brief Work unit for distribution across GPUs
 */
struct WorkUnit {
    size_t work_id;                             // Unique work unit ID
    ecc::BigInt256 range_start;                 // Starting private key
    ecc::BigInt256 range_end;                   // Ending private key (exclusive)
    size_t estimated_key_count;                 // Estimated number of keys
    
    // Assignment information
    int assigned_device_id;                     // Assigned GPU device (-1 if unassigned)
    std::chrono::system_clock::time_point assigned_time; // Assignment timestamp
    std::chrono::system_clock::time_point deadline;     // Expected completion time
    
    // Progress tracking
    enum class WorkStatus {
        PENDING,        // Waiting to be assigned
        ASSIGNED,       // Assigned to a device
        PROCESSING,     // Currently being processed
        COMPLETED,      // Successfully completed
        FAILED,         // Processing failed
        CANCELLED,      // Work unit cancelled
        STOLEN          // Work stolen by another device
    };
    
    WorkStatus status;                          // Current work status
    double progress_percentage;                 // Completion percentage (0-100)
    size_t keys_processed;                      // Keys processed so far
    
    // Performance metrics
    std::chrono::milliseconds processing_time;  // Actual processing time
    double keys_per_second;                     // Processing speed
    
    // Results
    std::vector<scan::PrivateKeyScanner::ScanMatch> matches; // Found matches
    std::vector<std::string> error_messages;    // Error messages (if failed)
    
    WorkUnit() 
        : work_id(0), estimated_key_count(0), assigned_device_id(-1)
        , status(WorkStatus::PENDING), progress_percentage(0.0)
        , keys_processed(0), processing_time(0), keys_per_second(0.0)
    {
        assigned_time = std::chrono::system_clock::now();
        deadline = assigned_time + std::chrono::hours(1); // Default 1-hour deadline
    }
};

/**
 * @brief Load balancing strategy configuration
 */
struct LoadBalancingConfig {
    enum class Strategy {
        ROUND_ROBIN,            // Simple round-robin assignment
        PERFORMANCE_WEIGHTED,   // Assign based on device performance
        DYNAMIC_ADAPTIVE,       // Adaptive assignment based on current load
        WORK_STEALING,          // Work stealing with load balancing
        HETEROGENEOUS_AWARE     // Heterogeneous GPU-aware balancing
    };
    
    Strategy strategy;                          // Load balancing strategy
    double rebalance_threshold;                 // Rebalancing trigger threshold
    std::chrono::seconds rebalance_interval;    // Rebalancing check interval
    bool enable_work_stealing;                  // Enable work stealing
    double work_stealing_threshold;             // Work stealing trigger threshold
    size_t max_work_steal_attempts;             // Maximum steal attempts
    
    // Performance weighting parameters
    double memory_weight;                       // Memory size weight
    double compute_weight;                      // Compute capability weight
    double bandwidth_weight;                    // Memory bandwidth weight
    double utilization_weight;                  // Current utilization weight
    
    LoadBalancingConfig() 
        : strategy(Strategy::DYNAMIC_ADAPTIVE)
        , rebalance_threshold(0.2)  // 20% load imbalance
        , rebalance_interval(std::chrono::seconds(30))
        , enable_work_stealing(true)
        , work_stealing_threshold(0.5)  // 50% completion difference
        , max_work_steal_attempts(3)
        , memory_weight(0.3), compute_weight(0.4)
        , bandwidth_weight(0.2), utilization_weight(0.1)
    {}
};

/**
 * @brief Multi-GPU performance metrics
 */
struct MultiGPUMetrics {
    // Overall performance
    double total_keys_per_second;               // Combined throughput
    double average_keys_per_second;             // Average per-device throughput
    double peak_keys_per_second;                // Peak achieved throughput
    
    // Load balancing metrics
    double load_balance_coefficient;            // Load balance quality (0-1)
    double work_distribution_efficiency;       // Work distribution efficiency
    size_t work_stealing_events;               // Number of work stealing events
    size_t rebalancing_events;                  // Number of rebalancing events
    
    // Device utilization
    std::vector<double> device_utilizations;    // Per-device utilization
    double average_device_utilization;          // Average device utilization
    double utilization_variance;                // Utilization variance
    
    // Communication metrics
    size_t inter_gpu_communications;            // Inter-GPU communication count
    double communication_overhead_percentage;   // Communication overhead
    std::chrono::milliseconds average_comm_latency; // Average communication latency
    
    // Fault tolerance
    size_t device_failures;                     // Device failure count
    size_t work_redistributions;                // Work redistribution count
    std::chrono::milliseconds recovery_time;    // Average recovery time
    
    // Progress tracking
    size_t total_work_units_completed;          // Total completed work units
    size_t total_work_units_failed;             // Total failed work units
    double overall_progress_percentage;         // Overall scanning progress
    
    MultiGPUMetrics() 
        : total_keys_per_second(0.0), average_keys_per_second(0.0)
        , peak_keys_per_second(0.0), load_balance_coefficient(0.0)
        , work_distribution_efficiency(0.0), work_stealing_events(0)
        , rebalancing_events(0), average_device_utilization(0.0)
        , utilization_variance(0.0), inter_gpu_communications(0)
        , communication_overhead_percentage(0.0), average_comm_latency(0)
        , device_failures(0), work_redistributions(0), recovery_time(0)
        , total_work_units_completed(0), total_work_units_failed(0)
        , overall_progress_percentage(0.0)
    {}
};

/**
 * @brief NCCL communication context for multi-GPU operations
 */
struct NCCLContext {
    ncclComm_t* nccl_comms;                     // NCCL communicators
    cudaStream_t* cuda_streams;                 // CUDA streams for communication
    int device_count;                           // Number of devices
    bool is_initialized;                        // Initialization status
    
    // Communication buffers
    void** send_buffers;                        // Send buffers for each device
    void** recv_buffers;                        // Receive buffers for each device
    size_t buffer_size;                         // Buffer size per device
    
    NCCLContext() 
        : nccl_comms(nullptr), cuda_streams(nullptr)
        , device_count(0), is_initialized(false)
        , send_buffers(nullptr), recv_buffers(nullptr)
        , buffer_size(0)
    {}
};

/**
 * @brief Main multi-GPU coordinator class
 */
class MultiGPUCoordinator {
public:
    MultiGPUCoordinator();
    ~MultiGPUCoordinator();
    
    // Initialization and configuration
    bool initialize();
    bool initialize_devices(const std::vector<int>& device_ids = {});
    void cleanup();
    
    // Device management
    bool add_device(int device_id);
    bool remove_device(int device_id);
    std::vector<GPUDeviceInfo> get_available_devices() const;
    std::vector<GPUDeviceInfo> get_active_devices() const;
    bool is_device_active(int device_id) const;
    
    // Load balancing configuration
    void configure_load_balancing(const LoadBalancingConfig& config);
    LoadBalancingConfig get_load_balancing_config() const;
    
    // Work distribution and coordination
    bool distribute_work(
        const models::PrivateKeyRange& range,
        const std::vector<std::string>& target_addresses,
        size_t work_unit_size = 10000000  // 10M keys per unit
    );
    
    bool start_coordinated_scanning();
    bool pause_coordinated_scanning();
    bool resume_coordinated_scanning();
    bool stop_coordinated_scanning();
    
    // Dynamic load balancing
    bool trigger_load_rebalancing();
    bool enable_automatic_load_balancing();
    bool disable_automatic_load_balancing();
    bool is_automatic_load_balancing_enabled() const;
    
    // Work stealing
    bool enable_work_stealing();
    bool disable_work_stealing();
    bool attempt_work_steal(int requesting_device_id, int target_device_id);
    
    // Performance monitoring
    MultiGPUMetrics get_current_metrics() const;
    std::vector<MultiGPUMetrics> get_metrics_history() const;
    void reset_performance_counters();
    
    // Progress and status
    bool is_scanning() const { return is_scanning_; }
    bool is_paused() const { return is_paused_; }
    double get_overall_progress() const;
    std::chrono::milliseconds get_estimated_completion_time() const;
    
    // Results and matches
    std::vector<scan::PrivateKeyScanner::ScanMatch> get_all_matches() const;
    std::map<int, std::vector<scan::PrivateKeyScanner::ScanMatch>> get_matches_by_device() const;
    size_t get_total_match_count() const;
    
    // Fault tolerance and recovery
    bool handle_device_failure(int device_id);
    bool redistribute_work_from_failed_device(int device_id);
    bool recover_from_device_failure();
    
    // Checkpoint integration
    bool set_checkpoint_manager(std::shared_ptr<scan::checkpoint::CheckpointManager> checkpoint_manager);
    bool save_coordinated_checkpoint(const std::string& filename = "");
    bool load_coordinated_checkpoint(const std::string& filename);
    
    // Communication and synchronization
    bool synchronize_all_devices();
    bool broadcast_configuration_to_all_devices();
    bool gather_results_from_all_devices();
    
    // Advanced features
    bool enable_heterogeneous_optimization();
    bool configure_memory_optimization_across_devices();
    bool enable_peer_to_peer_access();
    
    // Callbacks and notifications
    void set_progress_callback(std::function<void(const MultiGPUMetrics&)> callback);
    void set_match_callback(std::function<void(const scan::PrivateKeyScanner::ScanMatch&, int device_id)> callback);
    void set_device_failure_callback(std::function<void(int device_id, const std::string& error)> callback);
    void set_rebalancing_callback(std::function<void(const std::string& reason)> callback);
    
    // Statistics and analysis
    struct CoordinationStatistics {
        std::chrono::milliseconds total_coordination_time;
        size_t total_work_units_processed;
        double coordination_efficiency;
        double communication_efficiency;
        std::map<int, double> device_efficiency_scores;
        std::vector<std::string> optimization_recommendations;
    };
    
    CoordinationStatistics get_coordination_statistics() const;
    std::vector<std::string> get_performance_recommendations() const;

private:
    // Core coordination state
    std::atomic<bool> is_scanning_;
    std::atomic<bool> is_paused_;
    std::atomic<bool> should_stop_;
    
    // Device management
    mutable std::mutex devices_mutex_;
    std::vector<GPUDeviceInfo> available_devices_;
    std::vector<int> active_device_ids_;
    std::map<int, std::unique_ptr<scan::PrivateKeyScanner>> device_scanners_;
    
    // Work management
    mutable std::mutex work_mutex_;
    std::queue<std::unique_ptr<WorkUnit>> pending_work_queue_;
    std::map<size_t, std::unique_ptr<WorkUnit>> active_work_units_;
    std::map<size_t, std::unique_ptr<WorkUnit>> completed_work_units_;
    std::atomic<size_t> next_work_id_;
    
    // Load balancing
    LoadBalancingConfig load_balance_config_;
    std::atomic<bool> auto_load_balancing_enabled_;
    std::thread load_balance_thread_;
    std::atomic<bool> should_stop_load_balancing_;
    
    // Performance monitoring
    mutable std::mutex metrics_mutex_;
    MultiGPUMetrics current_metrics_;
    std::vector<MultiGPUMetrics> metrics_history_;
    std::chrono::high_resolution_clock::time_point coordination_start_time_;
    
    // Communication infrastructure
    std::unique_ptr<NCCLContext> nccl_context_;
    bool nccl_initialized_;
    
    // Threading and synchronization
    std::vector<std::thread> device_worker_threads_;
    std::condition_variable work_available_cv_;
    std::condition_variable coordination_cv_;
    
    // Checkpoint integration
    std::shared_ptr<scan::checkpoint::CheckpointManager> checkpoint_manager_;
    
    // Callbacks
    std::function<void(const MultiGPUMetrics&)> progress_callback_;
    std::function<void(const scan::PrivateKeyScanner::ScanMatch&, int)> match_callback_;
    std::function<void(int, const std::string&)> device_failure_callback_;
    std::function<void(const std::string&)> rebalancing_callback_;
    
    // Results aggregation
    mutable std::mutex results_mutex_;
    std::map<int, std::vector<scan::PrivateKeyScanner::ScanMatch>> device_matches_;
    
    // Internal coordination methods
    
    // Device discovery and management
    bool discover_available_devices();
    GPUDeviceInfo query_device_info(int device_id);
    bool initialize_device_scanner(int device_id);
    void cleanup_device_scanner(int device_id);
    
    // Work distribution strategies
    std::unique_ptr<WorkUnit> create_work_unit(
        const ecc::BigInt256& start_key, 
        const ecc::BigInt256& end_key,
        size_t work_id
    );
    
    bool assign_work_round_robin();
    bool assign_work_performance_weighted();
    bool assign_work_dynamic_adaptive();
    
    // Load balancing implementation
    void load_balance_worker_thread();
    bool analyze_load_imbalance();
    bool rebalance_work_distribution();
    double calculate_device_performance_score(int device_id) const;
    
    // Work stealing implementation
    bool can_steal_work_from_device(int source_device_id, int target_device_id) const;
    std::unique_ptr<WorkUnit> steal_work_from_device(int source_device_id);
    bool redistribute_stolen_work(std::unique_ptr<WorkUnit> work_unit, int new_device_id);
    
    // Device worker threads
    void device_worker_thread(int device_id);
    bool process_work_unit_on_device(WorkUnit* work_unit, int device_id);
    
    // Performance monitoring
    void update_coordination_metrics();
    void update_device_utilization(int device_id);
    double calculate_load_balance_coefficient() const;
    double calculate_work_distribution_efficiency() const;
    
    // Communication and NCCL
    bool initialize_nccl_context();
    void cleanup_nccl_context();
    bool broadcast_data_to_all_devices(const void* data, size_t size);
    bool gather_data_from_all_devices(void* data, size_t size_per_device);
    bool all_reduce_metrics();
    
    // Fault tolerance
    bool detect_device_failures();
    bool handle_failed_work_units(int failed_device_id);
    bool reassign_work_from_failed_device(int failed_device_id);
    
    // Optimization strategies
    bool optimize_memory_layout_across_devices();
    bool optimize_communication_patterns();
    bool apply_heterogeneous_device_optimizations();
    
    // Checkpoint coordination
    scan::checkpoint::ExtendedCheckpointData create_coordinated_checkpoint_data() const;
    bool restore_coordination_from_checkpoint(const scan::checkpoint::ExtendedCheckpointData& data);
    
    // Statistics and analysis
    CoordinationStatistics calculate_coordination_statistics() const;
    std::vector<std::string> analyze_performance_bottlenecks() const;
};

/**
 * @brief Multi-GPU coordinator factory for different configurations
 */
class MultiGPUCoordinatorFactory {
public:
    enum class CoordinationStrategy {
        BASIC_PARALLEL,         // Basic parallel execution
        LOAD_BALANCED,         // Load-balanced execution
        WORK_STEALING,         // Work-stealing execution
        ADAPTIVE_DYNAMIC,      // Adaptive dynamic coordination
        HETEROGENEOUS_AWARE    // Heterogeneous GPU-aware coordination
    };
    
    static std::unique_ptr<MultiGPUCoordinator> create_coordinator(
        CoordinationStrategy strategy,
        const std::vector<int>& device_ids = {}
    );
    
    static LoadBalancingConfig get_recommended_load_balancing_config(
        CoordinationStrategy strategy,
        const std::vector<GPUDeviceInfo>& devices
    );
    
    static std::vector<CoordinationStrategy> get_available_strategies();
    static std::string get_strategy_description(CoordinationStrategy strategy);
};

/**
 * @brief Utility functions for multi-GPU coordination
 */
namespace coordination_utils {
    
    /**
     * @brief Device topology analyzer
     */
    class TopologyAnalyzer {
    public:
        struct DeviceTopology {
            std::map<std::pair<int, int>, bool> peer_access_matrix;
            std::map<std::pair<int, int>, double> bandwidth_matrix;
            std::vector<std::vector<int>> numa_groups;
            std::string topology_description;
        };
        
        static DeviceTopology analyze_device_topology(const std::vector<int>& device_ids);
        static bool optimize_device_placement(std::vector<int>& device_ids, const DeviceTopology& topology);
        static std::vector<int> get_optimal_device_pairing(const std::vector<int>& device_ids);
    };
    
    /**
     * @brief Workload estimation and partitioning
     */
    class WorkloadEstimator {
    public:
        static size_t estimate_optimal_work_unit_size(
            const models::PrivateKeyRange& range,
            const std::vector<GPUDeviceInfo>& devices
        );
        
        static std::vector<models::PrivateKeyRange> partition_range_for_devices(
            const models::PrivateKeyRange& range,
            const std::vector<GPUDeviceInfo>& devices
        );
        
        static double estimate_completion_time(
            const models::PrivateKeyRange& range,
            const std::vector<GPUDeviceInfo>& devices
        );
    };
    
    /**
     * @brief Communication pattern optimizer
     */
    class CommunicationOptimizer {
    public:
        static std::vector<std::pair<int, int>> optimize_communication_graph(
            const std::vector<int>& device_ids,
            const TopologyAnalyzer::DeviceTopology& topology
        );
        
        static size_t calculate_optimal_buffer_size(const std::vector<GPUDeviceInfo>& devices);
        static bool enable_optimal_peer_access(const std::vector<int>& device_ids);
    };
}

} // namespace coordination
} // namespace gpu
} // namespace keyhunt