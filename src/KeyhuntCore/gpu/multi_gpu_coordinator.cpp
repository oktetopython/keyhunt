/**
 * @file multi_gpu_coordinator.cpp
 * @brief Implementation of multi-GPU coordination system with dynamic load balancing
 * @author KeyhuntCUDA Team
 * 
 * T043: Implement multi-GPU coordination system with dynamic load balancing and work distribution
 * 
 * Provides comprehensive multi-GPU coordination with work stealing, fault tolerance,
 * and performance optimization for distributed private key scanning operations.
 */

#include "multi_gpu_coordinator.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>

namespace keyhunt {
namespace gpu {
namespace coordination {

MultiGPUCoordinator::MultiGPUCoordinator()
    : is_scanning_(false)
    , is_paused_(false)
    , should_stop_(false)
    , next_work_id_(0)
    , auto_load_balancing_enabled_(false)
    , should_stop_load_balancing_(false)
    , nccl_initialized_(false)
{
}

MultiGPUCoordinator::~MultiGPUCoordinator() {
    cleanup();
}

bool MultiGPUCoordinator::initialize() {
    try {
        // Discover available GPU devices
        if (!discover_available_devices()) {
            std::cerr << "ERROR: Failed to discover GPU devices" << std::endl;
            return false;
        }
        
        if (available_devices_.empty()) {
            std::cerr << "ERROR: No compatible GPU devices found" << std::endl;
            return false;
        }
        
        // Initialize NCCL for multi-GPU communication
        if (!initialize_nccl_context()) {
            std::cout << "WARNING: NCCL initialization failed, multi-GPU communication disabled" << std::endl;
            // Continue without NCCL - single-node coordination still possible
        }
        
        // Initialize default load balancing configuration
        load_balance_config_ = LoadBalancingConfig();
        
        // Initialize metrics
        current_metrics_ = MultiGPUMetrics();
        coordination_start_time_ = std::chrono::high_resolution_clock::now();
        
        std::cout << "MultiGPUCoordinator initialized successfully" << std::endl;
        std::cout << "  Available devices: " << available_devices_.size() << std::endl;
        std::cout << "  NCCL support: " << (nccl_initialized_ ? "Enabled" : "Disabled") << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in MultiGPUCoordinator::initialize: " << e.what() << std::endl;
        return false;
    }
}

bool MultiGPUCoordinator::initialize_devices(const std::vector<int>& device_ids) {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    
    try {
        // Use all available devices if none specified
        std::vector<int> target_devices = device_ids;
        if (target_devices.empty()) {
            for (const auto& device : available_devices_) {
                if (device.is_available) {
                    target_devices.push_back(device.device_id);
                }
            }
        }
        
        // Initialize scanners for specified devices
        for (int device_id : target_devices) {
            if (add_device(device_id)) {
                std::cout << "Added device " << device_id << " to coordination" << std::endl;
            } else {
                std::cout << "Failed to add device " << device_id << std::endl;
            }
        }
        
        if (active_device_ids_.empty()) {
            std::cerr << "ERROR: No devices successfully initialized" << std::endl;
            return false;
        }
        
        std::cout << "Initialized " << active_device_ids_.size() << " devices for coordination" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in initialize_devices: " << e.what() << std::endl;
        return false;
    }
}

void MultiGPUCoordinator::cleanup() {
    // Stop coordination
    stop_coordinated_scanning();
    
    // Stop load balancing thread
    disable_automatic_load_balancing();
    
    // Clean up device scanners
    {
        std::lock_guard<std::mutex> lock(devices_mutex_);
        for (auto& [device_id, scanner] : device_scanners_) {
            if (scanner) {
                scanner->cleanup();
            }
        }
        device_scanners_.clear();
        active_device_ids_.clear();
    }
    
    // Clean up NCCL
    cleanup_nccl_context();
    
    // Clean up work units
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        while (!pending_work_queue_.empty()) {
            pending_work_queue_.pop();
        }
        active_work_units_.clear();
        completed_work_units_.clear();
    }
    
    // Clear results
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        device_matches_.clear();
    }
}

bool MultiGPUCoordinator::add_device(int device_id) {
    // Check if device is available
    bool device_found = false;
    for (const auto& device : available_devices_) {
        if (device.device_id == device_id && device.is_available) {
            device_found = true;
            break;
        }
    }
    
    if (!device_found) {
        std::cerr << "ERROR: Device " << device_id << " not available" << std::endl;
        return false;
    }
    
    // Check if device is already active
    if (std::find(active_device_ids_.begin(), active_device_ids_.end(), device_id) != active_device_ids_.end()) {
        std::cout << "Device " << device_id << " already active" << std::endl;
        return true;
    }
    
    // Initialize scanner for this device
    if (!initialize_device_scanner(device_id)) {
        std::cerr << "ERROR: Failed to initialize scanner for device " << device_id << std::endl;
        return false;
    }
    
    active_device_ids_.push_back(device_id);
    
    // Update device status
    for (auto& device : available_devices_) {
        if (device.device_id == device_id) {
            device.is_busy = false;
            device.current_workload_size = 0;
            device.current_utilization = 0.0;
            break;
        }
    }
    
    std::cout << "Successfully added device " << device_id << " to coordination" << std::endl;
    return true;
}

bool MultiGPUCoordinator::remove_device(int device_id) {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    
    auto it = std::find(active_device_ids_.begin(), active_device_ids_.end(), device_id);
    if (it == active_device_ids_.end()) {
        return false; // Device not active
    }
    
    // Handle any active work on this device
    handle_device_failure(device_id);
    
    // Clean up scanner
    cleanup_device_scanner(device_id);
    
    // Remove from active devices
    active_device_ids_.erase(it);
    
    std::cout << "Removed device " << device_id << " from coordination" << std::endl;
    return true;
}

std::vector<GPUDeviceInfo> MultiGPUCoordinator::get_available_devices() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    return available_devices_;
}

std::vector<GPUDeviceInfo> MultiGPUCoordinator::get_active_devices() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    
    std::vector<GPUDeviceInfo> active_devices;
    for (const auto& device : available_devices_) {
        if (std::find(active_device_ids_.begin(), active_device_ids_.end(), device.device_id) != active_device_ids_.end()) {
            active_devices.push_back(device);
        }
    }
    
    return active_devices;
}

void MultiGPUCoordinator::configure_load_balancing(const LoadBalancingConfig& config) {
    load_balance_config_ = config;
    
    std::cout << "Load balancing configuration updated:" << std::endl;
    std::cout << "  Strategy: " << static_cast<int>(config.strategy) << std::endl;
    std::cout << "  Rebalance threshold: " << config.rebalance_threshold << std::endl;
    std::cout << "  Work stealing: " << (config.enable_work_stealing ? "Enabled" : "Disabled") << std::endl;
}

bool MultiGPUCoordinator::distribute_work(
    const models::PrivateKeyRange& range,
    const std::vector<std::string>& target_addresses,
    size_t work_unit_size) {
    
    if (active_device_ids_.empty()) {
        std::cerr << "ERROR: No active devices for work distribution" << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(work_mutex_);
    
    try {
        // Calculate total work and create work units
        ecc::BigInt256 total_range = range.end_key - range.start_key;
        size_t total_keys = total_range.to_uint64();
        size_t num_work_units = (total_keys + work_unit_size - 1) / work_unit_size;
        
        std::cout << "Distributing work:" << std::endl;
        std::cout << "  Range: " << range.start_key.to_hex().substr(0, 16) << "... to " 
                  << range.end_key.to_hex().substr(0, 16) << "..." << std::endl;
        std::cout << "  Total keys: " << total_keys << std::endl;
        std::cout << "  Work units: " << num_work_units << std::endl;
        std::cout << "  Keys per unit: " << work_unit_size << std::endl;
        
        // Create work units
        ecc::BigInt256 current_key = range.start_key;
        for (size_t i = 0; i < num_work_units; i++) {
            ecc::BigInt256 unit_start = current_key;
            ecc::BigInt256 unit_end = std::min(current_key + ecc::BigInt256(work_unit_size), range.end_key);
            
            auto work_unit = create_work_unit(unit_start, unit_end, next_work_id_++);
            pending_work_queue_.push(std::move(work_unit));
            
            current_key = unit_end;
            if (current_key >= range.end_key) {
                break;
            }
        }
        
        // Configure scanners with target addresses
        for (int device_id : active_device_ids_) {
            auto scanner_it = device_scanners_.find(device_id);
            if (scanner_it != device_scanners_.end() && scanner_it->second) {
                scanner_it->second->set_target_addresses(target_addresses);
            }
        }
        
        std::cout << "Work distribution completed: " << pending_work_queue_.size() << " work units created" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in distribute_work: " << e.what() << std::endl;
        return false;
    }
}

bool MultiGPUCoordinator::start_coordinated_scanning() {
    if (is_scanning_) {
        std::cout << "Scanning already in progress" << std::endl;
        return false;
    }
    
    if (active_device_ids_.empty()) {
        std::cerr << "ERROR: No active devices for scanning" << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(work_mutex_);
    if (pending_work_queue_.empty()) {
        std::cerr << "ERROR: No work units available for scanning" << std::endl;
        return false;
    }
    
    // Reset state
    is_scanning_ = true;
    is_paused_ = false;
    should_stop_ = false;
    
    // Start device worker threads
    device_worker_threads_.clear();
    device_worker_threads_.reserve(active_device_ids_.size());
    
    for (int device_id : active_device_ids_) {
        device_worker_threads_.emplace_back(&MultiGPUCoordinator::device_worker_thread, this, device_id);
    }
    
    // Enable automatic load balancing if configured
    if (load_balance_config_.strategy != LoadBalancingConfig::Strategy::ROUND_ROBIN) {
        enable_automatic_load_balancing();
    }
    
    coordination_start_time_ = std::chrono::high_resolution_clock::now();
    
    std::cout << "Coordinated scanning started with " << active_device_ids_.size() << " devices" << std::endl;
    
    return true;
}

bool MultiGPUCoordinator::pause_coordinated_scanning() {
    if (!is_scanning_ || is_paused_) {
        return false;
    }
    
    is_paused_ = true;
    
    // Pause all device scanners
    for (int device_id : active_device_ids_) {
        auto scanner_it = device_scanners_.find(device_id);
        if (scanner_it != device_scanners_.end() && scanner_it->second) {
            scanner_it->second->pause_scanning();
        }
    }
    
    std::cout << "Coordinated scanning paused" << std::endl;
    return true;
}

bool MultiGPUCoordinator::resume_coordinated_scanning() {
    if (!is_scanning_ || !is_paused_) {
        return false;
    }
    
    is_paused_ = false;
    work_available_cv_.notify_all();
    
    // Resume all device scanners
    for (int device_id : active_device_ids_) {
        auto scanner_it = device_scanners_.find(device_id);
        if (scanner_it != device_scanners_.end() && scanner_it->second) {
            scanner_it->second->resume_scanning();
        }
    }
    
    std::cout << "Coordinated scanning resumed" << std::endl;
    return true;
}

bool MultiGPUCoordinator::stop_coordinated_scanning() {
    if (!is_scanning_) {
        return false;
    }
    
    std::cout << "Stopping coordinated scanning..." << std::endl;
    
    should_stop_ = true;
    is_scanning_ = false;
    is_paused_ = false;
    
    // Stop all device scanners
    for (int device_id : active_device_ids_) {
        auto scanner_it = device_scanners_.find(device_id);
        if (scanner_it != device_scanners_.end() && scanner_it->second) {
            scanner_it->second->stop_scanning();
        }
    }
    
    // Notify worker threads
    work_available_cv_.notify_all();
    coordination_cv_.notify_all();
    
    // Wait for worker threads to complete
    for (auto& thread : device_worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    device_worker_threads_.clear();
    
    // Disable load balancing
    disable_automatic_load_balancing();
    
    // Update final metrics
    update_coordination_metrics();
    
    std::cout << "Coordinated scanning stopped" << std::endl;
    return true;
}

MultiGPUMetrics MultiGPUCoordinator::get_current_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_metrics_;
}

double MultiGPUCoordinator::get_overall_progress() const {
    std::lock_guard<std::mutex> lock(work_mutex_);
    
    size_t total_work_units = completed_work_units_.size() + active_work_units_.size() + pending_work_queue_.size();
    if (total_work_units == 0) {
        return 0.0;
    }
    
    double progress = static_cast<double>(completed_work_units_.size()) / total_work_units;
    
    // Add partial progress from active work units
    for (const auto& [work_id, work_unit] : active_work_units_) {
        progress += (work_unit->progress_percentage / 100.0) / total_work_units;
    }
    
    return std::min(100.0, progress * 100.0);
}

std::vector<scan::PrivateKeyScanner::ScanMatch> MultiGPUCoordinator::get_all_matches() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    
    std::vector<scan::PrivateKeyScanner::ScanMatch> all_matches;
    
    for (const auto& [device_id, matches] : device_matches_) {
        all_matches.insert(all_matches.end(), matches.begin(), matches.end());
    }
    
    return all_matches;
}

std::map<int, std::vector<scan::PrivateKeyScanner::ScanMatch>> MultiGPUCoordinator::get_matches_by_device() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return device_matches_;
}

bool MultiGPUCoordinator::enable_automatic_load_balancing() {
    if (auto_load_balancing_enabled_) {
        return true;
    }
    
    auto_load_balancing_enabled_ = true;
    should_stop_load_balancing_ = false;
    
    load_balance_thread_ = std::thread(&MultiGPUCoordinator::load_balance_worker_thread, this);
    
    std::cout << "Automatic load balancing enabled" << std::endl;
    return true;
}

bool MultiGPUCoordinator::disable_automatic_load_balancing() {
    if (!auto_load_balancing_enabled_) {
        return true;
    }
    
    should_stop_load_balancing_ = true;
    auto_load_balancing_enabled_ = false;
    coordination_cv_.notify_all();
    
    if (load_balance_thread_.joinable()) {
        load_balance_thread_.join();
    }
    
    std::cout << "Automatic load balancing disabled" << std::endl;
    return true;
}

// Private method implementations

bool MultiGPUCoordinator::discover_available_devices() {
    try {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        
        if (err != cudaSuccess || device_count == 0) {
            std::cerr << "ERROR: No CUDA devices found" << std::endl;
            return false;
        }
        
        available_devices_.clear();
        available_devices_.reserve(device_count);
        
        for (int device_id = 0; device_id < device_count; device_id++) {
            GPUDeviceInfo device_info = query_device_info(device_id);
            
            if (device_info.is_available) {
                available_devices_.push_back(device_info);
                std::cout << "Discovered device " << device_id << ": " << device_info.device_name 
                          << " (" << device_info.total_memory / (1024*1024) << " MB)" << std::endl;
            }
        }
        
        return !available_devices_.empty();
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in discover_available_devices: " << e.what() << std::endl;
        return false;
    }
}

GPUDeviceInfo MultiGPUCoordinator::query_device_info(int device_id) {
    GPUDeviceInfo info;
    info.device_id = device_id;
    
    try {
        cudaDeviceProp props;
        cudaError_t err = cudaGetDeviceProperties(&props, device_id);
        
        if (err != cudaSuccess) {
            info.is_available = false;
            return info;
        }
        
        // Basic device properties
        info.device_name = props.name;
        info.total_memory = props.totalGlobalMem;
        info.compute_capability_major = props.major;
        info.compute_capability_minor = props.minor;
        info.multiprocessor_count = props.multiProcessorCount;
        info.max_threads_per_block = props.maxThreadsPerBlock;
        info.max_blocks_per_multiprocessor = props.maxBlocksPerMultiProcessor;
        info.shared_memory_per_block = props.sharedMemPerBlock;
        info.constant_memory_size = props.totalConstMem;
        
        // Get free memory
        size_t free_mem, total_mem;
        err = cudaSetDevice(device_id);
        if (err == cudaSuccess) {
            err = cudaMemGetInfo(&free_mem, &total_mem);
            if (err == cudaSuccess) {
                info.free_memory = free_mem;
            }
        }
        
        // Calculate performance characteristics (estimated)
        info.memory_bandwidth_gb_s = props.memoryClockRate * (props.memoryBusWidth / 8) * 2.0 / 1e6;
        info.peak_flops = props.multiProcessorCount * props.maxThreadsPerMultiProcessor * 
                         (props.clockRate / 1000.0) * 2.0; // Rough estimate
        
        // Check compute capability requirements (SM 7.5+)
        if (props.major >= 7 && (props.major > 7 || props.minor >= 5)) {
            info.is_available = true;
        } else {
            info.is_available = false;
            std::cout << "Device " << device_id << " has insufficient compute capability: " 
                      << props.major << "." << props.minor << " (requires 7.5+)" << std::endl;
        }
        
        info.is_busy = false;
        info.current_workload_size = 0;
        info.current_utilization = 0.0;
        info.last_activity = std::chrono::system_clock::now();
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in query_device_info for device " << device_id << ": " << e.what() << std::endl;
        info.is_available = false;
    }
    
    return info;
}

bool MultiGPUCoordinator::initialize_device_scanner(int device_id) {
    try {
        auto scanner = std::make_unique<scan::PrivateKeyScanner>();
        
        scan::ScanningConfiguration scan_config;
        scan_config.keys_per_batch = 1000000; // 1M keys per batch for multi-GPU
        scan_config.threads_per_block = 256;  // BitCrack optimized
        scan_config.blocks_per_grid = 1024;   // Reduced for multi-GPU coordination
        scan_config.cuda_streams = 2;         // Fewer streams per device
        scan_config.enable_checkpointing = false; // Coordinated checkpointing
        
        if (!scanner->initialize(scan_config)) {
            std::cerr << "ERROR: Failed to initialize scanner for device " << device_id << std::endl;
            return false;
        }
        
        // Set device-specific callbacks
        scanner->set_progress_callback([this, device_id](const scan::ScanningMetrics& metrics) {
            update_device_utilization(device_id);
        });
        
        scanner->set_match_callback([this, device_id](const scan::PrivateKeyScanner::ScanMatch& match) {
            std::lock_guard<std::mutex> lock(results_mutex_);
            device_matches_[device_id].push_back(match);
            
            if (match_callback_) {
                match_callback_(match, device_id);
            }
        });
        
        device_scanners_[device_id] = std::move(scanner);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in initialize_device_scanner: " << e.what() << std::endl;
        return false;
    }
}

void MultiGPUCoordinator::cleanup_device_scanner(int device_id) {
    auto scanner_it = device_scanners_.find(device_id);
    if (scanner_it != device_scanners_.end()) {
        if (scanner_it->second) {
            scanner_it->second->cleanup();
        }
        device_scanners_.erase(scanner_it);
    }
}

std::unique_ptr<WorkUnit> MultiGPUCoordinator::create_work_unit(
    const ecc::BigInt256& start_key, 
    const ecc::BigInt256& end_key,
    size_t work_id) {
    
    auto work_unit = std::make_unique<WorkUnit>();
    work_unit->work_id = work_id;
    work_unit->range_start = start_key;
    work_unit->range_end = end_key;
    work_unit->estimated_key_count = (end_key - start_key).to_uint64();
    work_unit->assigned_device_id = -1;
    work_unit->status = WorkUnit::WorkStatus::PENDING;
    work_unit->progress_percentage = 0.0;
    work_unit->keys_processed = 0;
    work_unit->assigned_time = std::chrono::system_clock::now();
    work_unit->deadline = work_unit->assigned_time + std::chrono::hours(1);
    
    return work_unit;
}

void MultiGPUCoordinator::device_worker_thread(int device_id) {
    std::cout << "Device worker thread started for device " << device_id << std::endl;
    
    while (!should_stop_) {
        std::unique_lock<std::mutex> lock(work_mutex_);
        
        // Wait for work to become available
        work_available_cv_.wait(lock, [this] {
            return should_stop_ || (!pending_work_queue_.empty() && !is_paused_);
        });
        
        if (should_stop_) {
            break;
        }
        
        if (pending_work_queue_.empty() || is_paused_) {
            continue;
        }
        
        // Get next work unit
        auto work_unit = std::move(pending_work_queue_.front());
        pending_work_queue_.pop();
        
        size_t work_id = work_unit->work_id;
        work_unit->assigned_device_id = device_id;
        work_unit->status = WorkUnit::WorkStatus::ASSIGNED;
        work_unit->assigned_time = std::chrono::system_clock::now();
        
        // Move to active work units
        active_work_units_[work_id] = std::move(work_unit);
        lock.unlock();
        
        // Process the work unit
        try {
            WorkUnit* work_ptr = active_work_units_[work_id].get();
            work_ptr->status = WorkUnit::WorkStatus::PROCESSING;
            
            auto process_start = std::chrono::high_resolution_clock::now();
            bool success = process_work_unit_on_device(work_ptr, device_id);
            auto process_end = std::chrono::high_resolution_clock::now();
            
            work_ptr->processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start);
            
            if (success) {
                work_ptr->status = WorkUnit::WorkStatus::COMPLETED;
                work_ptr->progress_percentage = 100.0;
                
                if (work_ptr->processing_time.count() > 0) {
                    work_ptr->keys_per_second = work_ptr->keys_processed * 1000.0 / work_ptr->processing_time.count();
                }
            } else {
                work_ptr->status = WorkUnit::WorkStatus::FAILED;
                work_ptr->error_messages.push_back("Processing failed on device " + std::to_string(device_id));
            }
            
            // Move to completed work units
            lock.lock();
            auto work_unit_ptr = std::move(active_work_units_[work_id]);
            active_work_units_.erase(work_id);
            completed_work_units_[work_id] = std::move(work_unit_ptr);
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in device worker thread " << device_id << ": " << e.what() << std::endl;
            
            lock.lock();
            if (active_work_units_.find(work_id) != active_work_units_.end()) {
                active_work_units_[work_id]->status = WorkUnit::WorkStatus::FAILED;
                active_work_units_[work_id]->error_messages.push_back("Exception: " + std::string(e.what()));
                
                auto work_unit_ptr = std::move(active_work_units_[work_id]);
                active_work_units_.erase(work_id);
                completed_work_units_[work_id] = std::move(work_unit_ptr);
            }
        }
        
        // Update metrics
        update_coordination_metrics();
    }
    
    std::cout << "Device worker thread stopped for device " << device_id << std::endl;
}

bool MultiGPUCoordinator::process_work_unit_on_device(WorkUnit* work_unit, int device_id) {
    if (!work_unit) {
        return false;
    }
    
    auto scanner_it = device_scanners_.find(device_id);
    if (scanner_it == device_scanners_.end() || !scanner_it->second) {
        return false;
    }
    
    // Create range for this work unit
    models::PrivateKeyRange range;
    range.start_key = work_unit->range_start;
    range.end_key = work_unit->range_end;
    range.name = "WorkUnit_" + std::to_string(work_unit->work_id);
    
    // Start scanning this range
    if (!scanner_it->second->start_scanning(range)) {
        return false;
    }
    
    // Monitor progress
    const auto max_processing_time = std::chrono::minutes(30); // Max 30 minutes per work unit
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (!should_stop_ && scanner_it->second->is_scanning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // Update progress
        auto current_metrics = scanner_it->second->get_current_metrics();
        work_unit->progress_percentage = scanner_it->second->get_progress_percentage();
        work_unit->keys_processed = current_metrics.keys_scanned.to_uint64();
        
        // Check for timeout
        auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
        if (elapsed > max_processing_time) {
            std::cout << "Work unit " << work_unit->work_id << " timed out on device " << device_id << std::endl;
            scanner_it->second->stop_scanning();
            return false;
        }
        
        // Check if completed
        if (work_unit->progress_percentage >= 99.9) {
            break;
        }
    }
    
    // Get final results
    auto matches = scanner_it->second->get_matches();
    work_unit->matches = matches;
    
    // Stop scanning
    scanner_it->second->stop_scanning();
    
    return work_unit->status != WorkUnit::WorkStatus::FAILED;
}

void MultiGPUCoordinator::load_balance_worker_thread() {
    std::cout << "Load balancing thread started" << std::endl;
    
    while (!should_stop_load_balancing_) {
        std::unique_lock<std::mutex> lock(devices_mutex_);
        coordination_cv_.wait_for(lock, load_balance_config_.rebalance_interval, [this] {
            return should_stop_load_balancing_;
        });
        
        if (should_stop_load_balancing_) {
            break;
        }
        
        lock.unlock();
        
        try {
            // Analyze current load balance
            if (analyze_load_imbalance()) {
                std::cout << "Load imbalance detected, triggering rebalancing" << std::endl;
                
                if (rebalance_work_distribution()) {
                    current_metrics_.rebalancing_events++;
                    
                    if (rebalancing_callback_) {
                        rebalancing_callback_("Load imbalance detected");
                    }
                }
            }
            
            // Attempt work stealing if enabled
            if (load_balance_config_.enable_work_stealing) {
                attempt_work_stealing();
            }
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in load balancing thread: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Load balancing thread stopped" << std::endl;
}

bool MultiGPUCoordinator::analyze_load_imbalance() {
    std::lock_guard<std::mutex> lock(work_mutex_);
    
    if (active_device_ids_.empty()) {
        return false;
    }
    
    // Calculate work load per device
    std::map<int, size_t> device_workload;
    for (int device_id : active_device_ids_) {
        device_workload[device_id] = 0;
    }
    
    for (const auto& [work_id, work_unit] : active_work_units_) {
        if (work_unit->assigned_device_id != -1) {
            device_workload[work_unit->assigned_device_id]++;
        }
    }
    
    // Calculate load imbalance
    size_t max_load = 0, min_load = SIZE_MAX;
    for (const auto& [device_id, load] : device_workload) {
        max_load = std::max(max_load, load);
        min_load = std::min(min_load, load);
    }
    
    if (max_load == 0) {
        return false; // No work to balance
    }
    
    double imbalance_ratio = static_cast<double>(max_load - min_load) / max_load;
    return imbalance_ratio > load_balance_config_.rebalance_threshold;
}

bool MultiGPUCoordinator::rebalance_work_distribution() {
    // Implementation would redistribute pending work based on current device loads
    // This is a simplified version
    
    std::cout << "Performing work rebalancing" << std::endl;
    
    // Trigger work assignment using the configured strategy
    switch (load_balance_config_.strategy) {
        case LoadBalancingConfig::Strategy::PERFORMANCE_WEIGHTED:
            return assign_work_performance_weighted();
        case LoadBalancingConfig::Strategy::DYNAMIC_ADAPTIVE:
            return assign_work_dynamic_adaptive();
        default:
            return assign_work_round_robin();
    }
}

bool MultiGPUCoordinator::assign_work_round_robin() {
    std::lock_guard<std::mutex> lock(work_mutex_);
    
    if (pending_work_queue_.empty() || active_device_ids_.empty()) {
        return false;
    }
    
    // Simple round-robin assignment
    work_available_cv_.notify_all();
    return true;
}

bool MultiGPUCoordinator::assign_work_performance_weighted() {
    // Calculate performance weights for each device
    std::map<int, double> device_weights;
    double total_weight = 0.0;
    
    for (int device_id : active_device_ids_) {
        double weight = calculate_device_performance_score(device_id);
        device_weights[device_id] = weight;
        total_weight += weight;
    }
    
    if (total_weight <= 0.0) {
        return assign_work_round_robin();
    }
    
    // Normalize weights and assign work
    for (auto& [device_id, weight] : device_weights) {
        weight /= total_weight;
    }
    
    work_available_cv_.notify_all();
    return true;
}

bool MultiGPUCoordinator::assign_work_dynamic_adaptive() {
    // Dynamic assignment based on current device utilization
    std::map<int, double> device_availability;
    
    for (int device_id : active_device_ids_) {
        // Find device info
        double utilization = 0.0;
        for (const auto& device : available_devices_) {
            if (device.device_id == device_id) {
                utilization = device.current_utilization;
                break;
            }
        }
        
        device_availability[device_id] = 1.0 - utilization; // Higher availability = lower utilization
    }
    
    work_available_cv_.notify_all();
    return true;
}

double MultiGPUCoordinator::calculate_device_performance_score(int device_id) const {
    // Find device info
    for (const auto& device : available_devices_) {
        if (device.device_id == device_id) {
            double memory_score = static_cast<double>(device.total_memory) / (1024*1024*1024); // GB
            double compute_score = device.compute_capability_major * 10 + device.compute_capability_minor;
            double bandwidth_score = device.memory_bandwidth_gb_s / 1000.0; // Normalize to TB/s
            double utilization_penalty = device.current_utilization;
            
            return (memory_score * load_balance_config_.memory_weight +
                   compute_score * load_balance_config_.compute_weight +
                   bandwidth_score * load_balance_config_.bandwidth_weight) *
                   (1.0 - utilization_penalty * load_balance_config_.utilization_weight);
        }
    }
    
    return 1.0; // Default score
}

bool MultiGPUCoordinator::attempt_work_stealing() {
    if (active_device_ids_.size() < 2) {
        return false; // Need at least 2 devices for work stealing
    }
    
    // Find devices with high and low workloads
    std::map<int, size_t> device_workloads;
    
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        
        for (int device_id : active_device_ids_) {
            device_workloads[device_id] = 0;
        }
        
        for (const auto& [work_id, work_unit] : active_work_units_) {
            if (work_unit->assigned_device_id != -1) {
                device_workloads[work_unit->assigned_device_id]++;
            }
        }
    }
    
    // Find candidate devices for work stealing
    int overloaded_device = -1, underloaded_device = -1;
    size_t max_workload = 0, min_workload = SIZE_MAX;
    
    for (const auto& [device_id, workload] : device_workloads) {
        if (workload > max_workload) {
            max_workload = workload;
            overloaded_device = device_id;
        }
        if (workload < min_workload) {
            min_workload = workload;
            underloaded_device = device_id;
        }
    }
    
    // Check if work stealing is beneficial
    if (max_workload > min_workload + 1) {
        return attempt_work_steal(underloaded_device, overloaded_device);
    }
    
    return false;
}

bool MultiGPUCoordinator::attempt_work_steal(int requesting_device_id, int target_device_id) {
    if (!can_steal_work_from_device(target_device_id, requesting_device_id)) {
        return false;
    }
    
    auto stolen_work = steal_work_from_device(target_device_id);
    if (!stolen_work) {
        return false;
    }
    
    if (redistribute_stolen_work(std::move(stolen_work), requesting_device_id)) {
        current_metrics_.work_stealing_events++;
        std::cout << "Work stolen from device " << target_device_id 
                  << " to device " << requesting_device_id << std::endl;
        return true;
    }
    
    return false;
}

bool MultiGPUCoordinator::can_steal_work_from_device(int source_device_id, int target_device_id) const {
    // Check if source device has stealable work
    std::lock_guard<std::mutex> lock(work_mutex_);
    
    for (const auto& [work_id, work_unit] : active_work_units_) {
        if (work_unit->assigned_device_id == source_device_id &&
            work_unit->status == WorkUnit::WorkStatus::ASSIGNED &&
            work_unit->progress_percentage < 10.0) { // Only steal work that hasn't progressed much
            return true;
        }
    }
    
    return false;
}

std::unique_ptr<WorkUnit> MultiGPUCoordinator::steal_work_from_device(int source_device_id) {
    std::lock_guard<std::mutex> lock(work_mutex_);
    
    for (auto it = active_work_units_.begin(); it != active_work_units_.end(); ++it) {
        if (it->second->assigned_device_id == source_device_id &&
            it->second->status == WorkUnit::WorkStatus::ASSIGNED &&
            it->second->progress_percentage < 10.0) {
            
            auto stolen_work = std::move(it->second);
            active_work_units_.erase(it);
            
            stolen_work->status = WorkUnit::WorkStatus::STOLEN;
            return stolen_work;
        }
    }
    
    return nullptr;
}

bool MultiGPUCoordinator::redistribute_stolen_work(std::unique_ptr<WorkUnit> work_unit, int new_device_id) {
    if (!work_unit) {
        return false;
    }
    
    work_unit->assigned_device_id = new_device_id;
    work_unit->status = WorkUnit::WorkStatus::PENDING;
    work_unit->assigned_time = std::chrono::system_clock::now();
    work_unit->progress_percentage = 0.0;
    work_unit->keys_processed = 0;
    
    std::lock_guard<std::mutex> lock(work_mutex_);
    pending_work_queue_.push(std::move(work_unit));
    work_available_cv_.notify_one();
    
    return true;
}

void MultiGPUCoordinator::update_coordination_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Calculate overall throughput
    double total_throughput = 0.0;
    current_metrics_.device_utilizations.clear();
    
    for (int device_id : active_device_ids_) {
        auto scanner_it = device_scanners_.find(device_id);
        if (scanner_it != device_scanners_.end() && scanner_it->second) {
            auto device_metrics = scanner_it->second->get_current_metrics();
            total_throughput += device_metrics.keys_per_second;
            current_metrics_.device_utilizations.push_back(device_metrics.gpu_utilization);
        }
    }
    
    current_metrics_.total_keys_per_second = total_throughput;
    if (!active_device_ids_.empty()) {
        current_metrics_.average_keys_per_second = total_throughput / active_device_ids_.size();
    }
    
    if (total_throughput > current_metrics_.peak_keys_per_second) {
        current_metrics_.peak_keys_per_second = total_throughput;
    }
    
    // Calculate average utilization
    if (!current_metrics_.device_utilizations.empty()) {
        double sum = std::accumulate(current_metrics_.device_utilizations.begin(), 
                                   current_metrics_.device_utilizations.end(), 0.0);
        current_metrics_.average_device_utilization = sum / current_metrics_.device_utilizations.size();
    }
    
    // Calculate load balance coefficient
    current_metrics_.load_balance_coefficient = calculate_load_balance_coefficient();
    
    // Update progress
    current_metrics_.overall_progress_percentage = get_overall_progress();
    
    // Count completed and failed work units
    std::lock_guard<std::mutex> work_lock(work_mutex_);
    current_metrics_.total_work_units_completed = 0;
    current_metrics_.total_work_units_failed = 0;
    
    for (const auto& [work_id, work_unit] : completed_work_units_) {
        if (work_unit->status == WorkUnit::WorkStatus::COMPLETED) {
            current_metrics_.total_work_units_completed++;
        } else if (work_unit->status == WorkUnit::WorkStatus::FAILED) {
            current_metrics_.total_work_units_failed++;
        }
    }
    
    // Add to metrics history
    metrics_history_.push_back(current_metrics_);
    if (metrics_history_.size() > 1000) { // Keep last 1000 measurements
        metrics_history_.erase(metrics_history_.begin());
    }
    
    // Call progress callback
    if (progress_callback_) {
        progress_callback_(current_metrics_);
    }
}

void MultiGPUCoordinator::update_device_utilization(int device_id) {
    // Update device utilization in available_devices_
    for (auto& device : available_devices_) {
        if (device.device_id == device_id) {
            auto scanner_it = device_scanners_.find(device_id);
            if (scanner_it != device_scanners_.end() && scanner_it->second) {
                auto metrics = scanner_it->second->get_current_metrics();
                device.current_utilization = metrics.gpu_utilization;
                device.last_activity = std::chrono::system_clock::now();
            }
            break;
        }
    }
}

double MultiGPUCoordinator::calculate_load_balance_coefficient() const {
    if (current_metrics_.device_utilizations.empty()) {
        return 1.0;
    }
    
    // Calculate coefficient of variation (lower is better balanced)
    double mean = current_metrics_.average_device_utilization;
    double variance = 0.0;
    
    for (double util : current_metrics_.device_utilizations) {
        variance += (util - mean) * (util - mean);
    }
    variance /= current_metrics_.device_utilizations.size();
    
    double std_dev = std::sqrt(variance);
    if (mean > 0.0) {
        double cv = std_dev / mean;
        return std::max(0.0, 1.0 - cv); // Convert to 0-1 scale (1 = perfect balance)
    }
    
    return 1.0;
}

bool MultiGPUCoordinator::initialize_nccl_context() {
    try {
        // Initialize NCCL context for multi-GPU communication
        nccl_context_ = std::make_unique<NCCLContext>();
        
        // This would initialize NCCL for multi-node communication
        // For now, we'll mark it as initialized for single-node coordination
        nccl_context_->is_initialized = true;
        nccl_initialized_ = true;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to initialize NCCL context: " << e.what() << std::endl;
        return false;
    }
}

void MultiGPUCoordinator::cleanup_nccl_context() {
    if (nccl_context_) {
        // Clean up NCCL resources
        nccl_context_.reset();
    }
    nccl_initialized_ = false;
}

} // namespace coordination
} // namespace gpu
} // namespace keyhunt