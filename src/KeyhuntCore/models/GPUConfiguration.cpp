/**
 * @file GPUConfiguration.cpp
 * @brief GPU configuration and management implementation
 * @author KeyhuntCUDA Team
 * 
 * Implements GPU hardware detection, configuration management, performance
 * optimization, and multi-GPU coordination with scientific precision.
 */

#include "keyhunt/models/GPUConfiguration.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <cassert>
#include <thread>

namespace keyhunt {
namespace models {

// GPUDevice implementation

GPUDevice::GPUDevice(int device_id)
    : device_id_(device_id)
    , compute_capability_(ComputeCapability::UNKNOWN)
    , performance_class_(PerformanceClass::LOW_END)
    , status_(DeviceStatus::AVAILABLE)
    , total_memory_bytes_(0)
    , free_memory_bytes_(0)
    , multiprocessor_count_(0)
    , max_threads_per_block_(1024)
    , max_blocks_per_grid_(65535)
    , warp_size_(32)
    , base_clock_mhz_(0)
    , memory_clock_mhz_(0)
    , boost_clock_mhz_(0)
    , theoretical_gflops_(0.0)
    , memory_bandwidth_gb_s_(0)
    , efficiency_rating_(0.0)
    , temperature_celsius_(0.0)
    , power_usage_watts_(0.0)
    , utilization_percentage_(0.0)
    , memory_utilization_percentage_(0.0)
    , enabled_(true)
    , thread_block_size_(256)
    , grid_size_(0)
    , load_factor_(1.0)
    , supports_unified_memory_(false)
    , supports_concurrent_kernels_(false)
    , supports_double_precision_(false)
{
    // Generate placeholder UUID
    std::ostringstream uuid_stream;
    uuid_stream << "GPU-" << std::setfill('0') << std::setw(8) << std::hex << device_id;
    uuid_ = uuid_stream.str();
    
    // Set default name
    name_ = "Unknown GPU Device " + std::to_string(device_id);
    
    // Initialize device if possible
    initialize_device();
}

bool GPUDevice::initialize_device() {
    try {
        // In production, this would use CUDA runtime API
        // For now, set up mock data based on device_id
        
        // Mock device specifications based on device ID
        switch (device_id_ % 3) {
            case 0: // High-end device
                name_ = "NVIDIA GeForce RTX 4080";
                compute_capability_ = ComputeCapability::ADA_LOVELACE;
                performance_class_ = PerformanceClass::HIGH_END;
                total_memory_bytes_ = 16ULL * 1024 * 1024 * 1024; // 16GB
                multiprocessor_count_ = 76;
                base_clock_mhz_ = 2210;
                memory_clock_mhz_ = 11400;
                theoretical_gflops_ = 48400.0;
                memory_bandwidth_gb_s_ = 717;
                efficiency_rating_ = 0.85;
                supports_unified_memory_ = true;
                supports_concurrent_kernels_ = true;
                supports_double_precision_ = true;
                break;
                
            case 1: // Mid-range device
                name_ = "NVIDIA GeForce RTX 3060";
                compute_capability_ = ComputeCapability::AMPERE;
                performance_class_ = PerformanceClass::MID_RANGE;
                total_memory_bytes_ = 12ULL * 1024 * 1024 * 1024; // 12GB
                multiprocessor_count_ = 28;
                base_clock_mhz_ = 1777;
                memory_clock_mhz_ = 7500;
                theoretical_gflops_ = 12740.0;
                memory_bandwidth_gb_s_ = 360;
                efficiency_rating_ = 0.75;
                supports_unified_memory_ = true;
                supports_concurrent_kernels_ = true;
                supports_double_precision_ = false;
                break;
                
            case 2: // Lower-end device
                name_ = "NVIDIA GeForce GTX 1660";
                compute_capability_ = ComputeCapability::TURING;
                performance_class_ = PerformanceClass::LOW_END;
                total_memory_bytes_ = 6ULL * 1024 * 1024 * 1024; // 6GB
                multiprocessor_count_ = 22;
                base_clock_mhz_ = 1785;
                memory_clock_mhz_ = 4000;
                theoretical_gflops_ = 5027.0;
                memory_bandwidth_gb_s_ = 192;
                efficiency_rating_ = 0.65;
                supports_unified_memory_ = false;
                supports_concurrent_kernels_ = true;
                supports_double_precision_ = false;
                break;
        }
        
        free_memory_bytes_ = total_memory_bytes_ * 0.9; // Assume 90% free
        boost_clock_mhz_ = static_cast<int>(base_clock_mhz_ * 1.1);
        
        status_ = DeviceStatus::AVAILABLE;
        return true;
        
    } catch (const std::exception&) {
        status_ = DeviceStatus::ERROR;
        return false;
    }
}

bool GPUDevice::query_device_properties() {
    // In production, this would query actual CUDA device properties
    // For now, return success if device is initialized
    return status_ != DeviceStatus::ERROR;
}

bool GPUDevice::update_runtime_metrics() {
    // Mock runtime metrics updates
    if (status_ == DeviceStatus::ERROR) return false;
    
    // Simulate some variation in metrics
    static int update_counter = 0;
    update_counter++;
    
    double variation = 0.1 * std::sin(update_counter * 0.1);
    
    utilization_percentage_ = 75.0 + 20.0 * variation;
    utilization_percentage_ = std::max(0.0, std::min(100.0, utilization_percentage_));
    
    temperature_celsius_ = 65.0 + 10.0 * variation;
    temperature_celsius_ = std::max(30.0, std::min(85.0, temperature_celsius_));
    
    power_usage_watts_ = 200.0 + 50.0 * variation;
    power_usage_watts_ = std::max(50.0, std::min(350.0, power_usage_watts_));
    
    memory_utilization_percentage_ = 60.0 + 25.0 * variation;
    memory_utilization_percentage_ = std::max(0.0, std::min(100.0, memory_utilization_percentage_));
    
    // Update free memory based on utilization
    size_t used_memory = static_cast<size_t>(total_memory_bytes_ * (memory_utilization_percentage_ / 100.0));
    free_memory_bytes_ = total_memory_bytes_ - used_memory;
    
    return true;
}

void GPUDevice::optimize_for_scanning() {
    if (!is_suitable_for_keyhunt()) return;
    
    // Calculate optimal thread block size based on compute capability
    switch (compute_capability_) {
        case ComputeCapability::TURING:
        case ComputeCapability::AMPERE:
        case ComputeCapability::ADA_LOVELACE:
        case ComputeCapability::HOPPER:
            thread_block_size_ = 512;  // Optimal for newer architectures
            break;
        case ComputeCapability::PASCAL:
            thread_block_size_ = 256;  // Good balance for Pascal
            break;
        default:
            thread_block_size_ = 128;  // Conservative for older architectures
            break;
    }
    
    // Calculate efficiency-based load factor
    if (performance_class_ == PerformanceClass::HIGH_END) {
        load_factor_ = 1.0;
    } else if (performance_class_ == PerformanceClass::MID_RANGE) {
        load_factor_ = 0.8;
    } else {
        load_factor_ = 0.6;
    }
}

void GPUDevice::calculate_optimal_grid_size(size_t total_work) {
    if (total_work == 0) {
        grid_size_ = 0;
        return;
    }
    
    // Calculate based on multiprocessor count and thread block size
    int max_blocks = multiprocessor_count_ * 4; // 4 blocks per SM is often optimal
    int required_blocks = static_cast<int>((total_work + thread_block_size_ - 1) / thread_block_size_);
    
    grid_size_ = std::min(max_blocks, required_blocks);
    grid_size_ = std::min(grid_size_, max_blocks_per_grid_);
}

bool GPUDevice::validate_configuration() const {
    // Check if device is suitable
    if (!is_suitable_for_keyhunt()) return false;
    
    // Check thread block size
    if (thread_block_size_ <= 0 || thread_block_size_ > max_threads_per_block_) return false;
    
    // Check grid size
    if (grid_size_ < 0 || grid_size_ > max_blocks_per_grid_) return false;
    
    // Check load factor
    if (load_factor_ < 0.0 || load_factor_ > 2.0) return false;
    
    return true;
}

bool GPUDevice::is_suitable_for_keyhunt() const {
    // Check minimum requirements for Keyhunt
    if (compute_capability_ < ComputeCapability::PASCAL) return false;
    if (total_memory_bytes_ < 2ULL * 1024 * 1024 * 1024) return false; // Minimum 2GB
    if (multiprocessor_count_ < 10) return false;
    
    return status_ == DeviceStatus::AVAILABLE && enabled_;
}

std::string GPUDevice::get_capability_string() const {
    switch (compute_capability_) {
        case ComputeCapability::KEPLER: return "3.0 (Kepler)";
        case ComputeCapability::MAXWELL: return "5.0 (Maxwell)";
        case ComputeCapability::PASCAL: return "6.1 (Pascal)";
        case ComputeCapability::TURING: return "7.5 (Turing)";
        case ComputeCapability::AMPERE: return "8.6 (Ampere)";
        case ComputeCapability::ADA_LOVELACE: return "8.9 (Ada Lovelace)";
        case ComputeCapability::HOPPER: return "9.0 (Hopper)";
        default: return "Unknown";
    }
}

std::string GPUDevice::get_performance_summary() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    
    oss << name_ << " (ID: " << device_id_ << ")\n";
    oss << "  Compute Capability: " << get_capability_string() << "\n";
    oss << "  Memory: " << (total_memory_bytes_ / (1024.0 * 1024 * 1024)) << " GB\n";
    oss << "  Performance: " << (theoretical_gflops_ / 1000.0) << " TFLOPS\n";
    oss << "  Multiprocessors: " << multiprocessor_count_ << "\n";
    oss << "  Status: " << (enabled_ ? "Enabled" : "Disabled") << "\n";
    
    return oss.str();
}

std::string GPUDevice::to_json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    oss << "{\n";
    oss << "  \"device_id\": " << device_id_ << ",\n";
    oss << "  \"name\": \"" << name_ << "\",\n";
    oss << "  \"uuid\": \"" << uuid_ << "\",\n";
    oss << "  \"compute_capability\": " << static_cast<int>(compute_capability_) << ",\n";
    oss << "  \"performance_class\": " << static_cast<int>(performance_class_) << ",\n";
    oss << "  \"status\": " << static_cast<int>(status_) << ",\n";
    oss << "  \"total_memory_bytes\": " << total_memory_bytes_ << ",\n";
    oss << "  \"free_memory_bytes\": " << free_memory_bytes_ << ",\n";
    oss << "  \"multiprocessor_count\": " << multiprocessor_count_ << ",\n";
    oss << "  \"theoretical_gflops\": " << theoretical_gflops_ << ",\n";
    oss << "  \"memory_bandwidth_gb_s\": " << memory_bandwidth_gb_s_ << ",\n";
    oss << "  \"temperature_celsius\": " << temperature_celsius_ << ",\n";
    oss << "  \"power_usage_watts\": " << power_usage_watts_ << ",\n";
    oss << "  \"utilization_percentage\": " << utilization_percentage_ << ",\n";
    oss << "  \"enabled\": " << (enabled_ ? "true" : "false") << ",\n";
    oss << "  \"thread_block_size\": " << thread_block_size_ << ",\n";
    oss << "  \"grid_size\": " << grid_size_ << ",\n";
    oss << "  \"load_factor\": " << load_factor_ << "\n";
    oss << "}";
    
    return oss.str();
}

// GPUConfiguration implementation

GPUConfiguration::GPUConfiguration()
    : load_balancing_strategy_(LoadBalancingStrategy::PERFORMANCE_WEIGHTED)
    , communication_pattern_(CommunicationPattern::HOST_MEDIATED)
    , status_(ConfigurationStatus::NOT_VALIDATED)
    , nccl_enabled_(true)
    , nccl_initialized_(false)
    , last_update_time_(std::chrono::steady_clock::now())
{
    initialize_default_settings();
}

void GPUConfiguration::initialize_default_settings() {
    // Auto-detect available devices
    auto_detect_devices();
    
    // Set default load balancing
    load_balancing_strategy_ = LoadBalancingStrategy::PERFORMANCE_WEIGHTED;
    
    // Initialize NCCL if available
    if (nccl_enabled_) {
        initialize_nccl();
    }
}

void GPUConfiguration::add_device(const GPUDevice& device) {
    // Check if device already exists
    for (auto& existing_device : devices_) {
        if (existing_device.get_device_id() == device.get_device_id()) {
            existing_device = device;
            return;
        }
    }
    
    devices_.push_back(device);
}

void GPUConfiguration::add_device(int device_id) {
    GPUDevice device(device_id);
    add_device(device);
}

void GPUConfiguration::remove_device(int device_id) {
    devices_.erase(
        std::remove_if(devices_.begin(), devices_.end(),
            [device_id](const GPUDevice& device) {
                return device.get_device_id() == device_id;
            }),
        devices_.end());
}

void GPUConfiguration::enable_device(int device_id, bool enabled) {
    for (auto& device : devices_) {
        if (device.get_device_id() == device_id) {
            device.set_enabled(enabled);
            break;
        }
    }
}

GPUDevice* GPUConfiguration::get_device(int device_id) {
    for (auto& device : devices_) {
        if (device.get_device_id() == device_id) {
            return &device;
        }
    }
    return nullptr;
}

const GPUDevice* GPUConfiguration::get_device(int device_id) const {
    for (const auto& device : devices_) {
        if (device.get_device_id() == device_id) {
            return &device;
        }
    }
    return nullptr;
}

std::vector<GPUDevice*> GPUConfiguration::get_enabled_devices() {
    std::vector<GPUDevice*> enabled_devices;
    
    for (auto& device : devices_) {
        if (device.is_enabled() && device.is_suitable_for_keyhunt()) {
            enabled_devices.push_back(&device);
        }
    }
    
    return enabled_devices;
}

std::vector<int> GPUConfiguration::get_enabled_device_ids() const {
    std::vector<int> device_ids;
    
    for (const auto& device : devices_) {
        if (device.is_enabled() && device.is_suitable_for_keyhunt()) {
            device_ids.push_back(device.get_device_id());
        }
    }
    
    return device_ids;
}

void GPUConfiguration::set_custom_load_weights(const std::unordered_map<int, double>& weights) {
    custom_load_weights_ = weights;
    if (load_balancing_strategy_ == LoadBalancingStrategy::CUSTOM) {
        calculate_load_distribution();
    }
}

size_t GPUConfiguration::get_total_memory() const {
    size_t total = 0;
    for (const auto& device : devices_) {
        if (device.is_enabled()) {
            total += device.get_total_memory();
        }
    }
    return total;
}

size_t GPUConfiguration::get_available_memory() const {
    size_t available = 0;
    for (const auto& device : devices_) {
        if (device.is_enabled()) {
            available += device.get_free_memory();
        }
    }
    return available;
}

double GPUConfiguration::get_total_theoretical_performance() const {
    double total_performance = 0.0;
    for (const auto& device : devices_) {
        if (device.is_enabled()) {
            total_performance += device.get_theoretical_performance();
        }
    }
    return total_performance;
}

std::unordered_map<int, double> GPUConfiguration::get_load_distribution() const {
    return calculated_load_dist_;
}

void GPUConfiguration::optimize_configuration() {
    // Optimize each enabled device
    for (auto& device : devices_) {
        if (device.is_enabled()) {
            device.optimize_for_scanning();
        }
    }
    
    // Calculate optimal load distribution
    calculate_load_distribution();
    
    // Validate the configuration
    validate_configuration();
}

void GPUConfiguration::calculate_load_distribution() {
    calculated_load_dist_.clear();
    
    auto enabled_devices = get_enabled_devices();
    if (enabled_devices.empty()) return;
    
    double total_weight = 0.0;
    std::unordered_map<int, double> weights;
    
    // Calculate weights based on strategy
    for (const auto* device : enabled_devices) {
        double weight = 0.0;
        
        switch (load_balancing_strategy_) {
            case LoadBalancingStrategy::EQUAL_DISTRIBUTION:
                weight = 1.0;
                break;
                
            case LoadBalancingStrategy::PERFORMANCE_WEIGHTED:
                weight = calculate_performance_weight(*device);
                break;
                
            case LoadBalancingStrategy::MEMORY_WEIGHTED:
                weight = calculate_memory_weight(*device);
                break;
                
            case LoadBalancingStrategy::CUSTOM:
                weight = custom_load_weights_.count(device->get_device_id()) 
                    ? custom_load_weights_.at(device->get_device_id()) : 1.0;
                break;
                
            case LoadBalancingStrategy::ADAPTIVE:
                // For adaptive, use performance weighting as base
                weight = calculate_performance_weight(*device);
                // TODO: Adjust based on runtime performance
                break;
        }
        
        weights[device->get_device_id()] = weight;
        total_weight += weight;
    }
    
    // Normalize to get distribution percentages
    for (const auto& entry : weights) {
        calculated_load_dist_[entry.first] = entry.second / total_weight;
    }
}

void GPUConfiguration::auto_detect_devices() {
    // In production, this would use CUDA runtime to detect devices
    // For now, create mock devices
    
    devices_.clear();
    
    // Create 2-4 mock devices
    int device_count = 2 + (std::hash<void*>{}(this) % 3); // 2-4 devices
    
    for (int i = 0; i < device_count; ++i) {
        add_device(i);
    }
    
    add_warning("Auto-detection used mock devices. In production, would query CUDA runtime.");
}

bool GPUConfiguration::validate_configuration() {
    clear_errors();
    clear_warnings();
    
    // Check if we have any enabled devices
    auto enabled_devices = get_enabled_devices();
    if (enabled_devices.empty()) {
        add_error("No enabled devices suitable for Keyhunt");
        status_ = ConfigurationStatus::INVALID_DEVICES;
        return false;
    }
    
    // Check device compatibility
    if (!check_device_compatibility()) {
        status_ = ConfigurationStatus::CAPABILITY_MISMATCH;
        return false;
    }
    
    // Check available memory
    if (get_available_memory() < 1ULL * 1024 * 1024 * 1024) { // 1GB minimum
        add_error("Insufficient available GPU memory");
        status_ = ConfigurationStatus::INSUFFICIENT_MEMORY;
        return false;
    }
    
    // Validate individual devices
    for (const auto& device : devices_) {
        if (device.is_enabled() && !device.validate_configuration()) {
            add_error("Invalid configuration for device " + std::to_string(device.get_device_id()));
            status_ = ConfigurationStatus::INVALID_DEVICES;
            return false;
        }
    }
    
    status_ = ConfigurationStatus::VALID;
    return true;
}

bool GPUConfiguration::initialize_nccl() {
    if (!nccl_enabled_) return false;
    
    // In production, this would initialize NCCL
    // For now, just set the flag
    nccl_initialized_ = true;
    
    auto enabled_devices = get_enabled_devices();
    if (enabled_devices.size() > 1) {
        add_warning("NCCL mock initialization for " + std::to_string(enabled_devices.size()) + " devices");
    }
    
    return true;
}

void GPUConfiguration::update_all_device_metrics() {
    last_update_time_ = std::chrono::steady_clock::now();
    
    for (auto& device : devices_) {
        if (device.is_enabled()) {
            device.update_runtime_metrics();
        }
    }
}

std::unordered_map<int, double> GPUConfiguration::get_device_utilizations() const {
    std::unordered_map<int, double> utilizations;
    
    for (const auto& device : devices_) {
        if (device.is_enabled()) {
            utilizations[device.get_device_id()] = device.get_utilization();
        }
    }
    
    return utilizations;
}

std::unordered_map<int, double> GPUConfiguration::get_device_temperatures() const {
    std::unordered_map<int, double> temperatures;
    
    for (const auto& device : devices_) {
        if (device.is_enabled()) {
            temperatures[device.get_device_id()] = device.get_temperature();
        }
    }
    
    return temperatures;
}

bool GPUConfiguration::has_suitable_devices() const {
    for (const auto& device : devices_) {
        if (device.is_suitable_for_keyhunt()) {
            return true;
        }
    }
    return false;
}

std::string GPUConfiguration::get_configuration_summary() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    
    auto enabled_devices = get_enabled_device_ids();
    
    oss << "GPU Configuration Summary:\n";
    oss << "  Total Devices: " << devices_.size() << "\n";
    oss << "  Enabled Devices: " << enabled_devices.size() << "\n";
    oss << "  Total Memory: " << (get_total_memory() / (1024.0 * 1024 * 1024)) << " GB\n";
    oss << "  Available Memory: " << (get_available_memory() / (1024.0 * 1024 * 1024)) << " GB\n";
    oss << "  Total Performance: " << (get_total_theoretical_performance() / 1000.0) << " TFLOPS\n";
    oss << "  Load Balancing: ";
    
    switch (load_balancing_strategy_) {
        case LoadBalancingStrategy::EQUAL_DISTRIBUTION: oss << "Equal Distribution"; break;
        case LoadBalancingStrategy::PERFORMANCE_WEIGHTED: oss << "Performance Weighted"; break;
        case LoadBalancingStrategy::MEMORY_WEIGHTED: oss << "Memory Weighted"; break;
        case LoadBalancingStrategy::ADAPTIVE: oss << "Adaptive"; break;
        case LoadBalancingStrategy::CUSTOM: oss << "Custom"; break;
    }
    oss << "\n";
    
    oss << "  NCCL: " << (nccl_initialized_ ? "Initialized" : "Not Available") << "\n";
    oss << "  Status: ";
    
    switch (status_) {
        case ConfigurationStatus::VALID: oss << "Valid"; break;
        case ConfigurationStatus::INVALID_DEVICES: oss << "Invalid Devices"; break;
        case ConfigurationStatus::INSUFFICIENT_MEMORY: oss << "Insufficient Memory"; break;
        case ConfigurationStatus::CAPABILITY_MISMATCH: oss << "Capability Mismatch"; break;
        case ConfigurationStatus::DRIVER_INCOMPATIBLE: oss << "Driver Incompatible"; break;
        case ConfigurationStatus::NOT_VALIDATED: oss << "Not Validated"; break;
    }
    oss << "\n";
    
    if (!errors_.empty()) {
        oss << "  Errors: " << errors_.size() << "\n";
    }
    
    if (!warnings_.empty()) {
        oss << "  Warnings: " << warnings_.size() << "\n";
    }
    
    return oss.str();
}

std::string GPUConfiguration::to_json() const {
    std::ostringstream oss;
    
    oss << "{\n";
    oss << "  \"device_count\": " << devices_.size() << ",\n";
    oss << "  \"load_balancing_strategy\": " << static_cast<int>(load_balancing_strategy_) << ",\n";
    oss << "  \"communication_pattern\": " << static_cast<int>(communication_pattern_) << ",\n";
    oss << "  \"status\": " << static_cast<int>(status_) << ",\n";
    oss << "  \"nccl_enabled\": " << (nccl_enabled_ ? "true" : "false") << ",\n";
    oss << "  \"nccl_initialized\": " << (nccl_initialized_ ? "true" : "false") << ",\n";
    oss << "  \"total_memory_bytes\": " << get_total_memory() << ",\n";
    oss << "  \"available_memory_bytes\": " << get_available_memory() << ",\n";
    oss << "  \"total_theoretical_gflops\": " << get_total_theoretical_performance() << ",\n";
    
    // Devices array
    oss << "  \"devices\": [\n";
    for (size_t i = 0; i < devices_.size(); ++i) {
        if (i > 0) oss << ",\n";
        oss << "    " << devices_[i].to_json();
    }
    oss << "\n  ],\n";
    
    // Load distribution
    oss << "  \"load_distribution\": {";
    bool first = true;
    for (const auto& entry : calculated_load_dist_) {
        if (!first) oss << ", ";
        oss << "\"" << entry.first << "\": " << entry.second;
        first = false;
    }
    oss << "},\n";
    
    // Errors and warnings
    oss << "  \"errors\": [";
    for (size_t i = 0; i < errors_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "\"" << errors_[i] << "\"";
    }
    oss << "],\n";
    
    oss << "  \"warnings\": [";
    for (size_t i = 0; i < warnings_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "\"" << warnings_[i] << "\"";
    }
    oss << "]\n";
    
    oss << "}";
    
    return oss.str();
}

// Helper method implementations

void GPUConfiguration::add_error(const std::string& error) {
    errors_.push_back(error);
}

void GPUConfiguration::add_warning(const std::string& warning) {
    warnings_.push_back(warning);
}

double GPUConfiguration::calculate_performance_weight(const GPUDevice& device) const {
    // Weight based on theoretical performance and efficiency
    return device.get_theoretical_performance() * device.get_efficiency_rating() * device.get_load_factor();
}

double GPUConfiguration::calculate_memory_weight(const GPUDevice& device) const {
    // Weight based on available memory
    return static_cast<double>(device.get_free_memory());
}

bool GPUConfiguration::check_device_compatibility() const {
    auto enabled_devices = get_enabled_devices();
    if (enabled_devices.size() <= 1) return true; // Single device is always compatible
    
    // Check if all devices have compatible compute capabilities
    auto first_capability = enabled_devices[0]->get_compute_capability();
    
    for (size_t i = 1; i < enabled_devices.size(); ++i) {
        auto current_capability = enabled_devices[i]->get_compute_capability();
        
        // Allow some flexibility in compute capability matching
        if (std::abs(static_cast<int>(current_capability) - static_cast<int>(first_capability)) > 100) {
            add_warning("Devices have significantly different compute capabilities");
        }
    }
    
    return true;
}

// GPUPerformanceMonitor implementation

GPUPerformanceMonitor::GPUPerformanceMonitor(GPUConfiguration* config, 
                                           std::chrono::milliseconds update_interval_ms)
    : gpu_config_(config)
    , update_interval_(update_interval_ms)
    , monitoring_active_(false)
    , temperature_threshold_(80.0)
    , utilization_threshold_(95.0)
{
}

void GPUPerformanceMonitor::start_monitoring() {
    if (monitoring_active_ || !gpu_config_) return;
    
    monitoring_active_ = true;
    std::thread monitoring_thread(&GPUPerformanceMonitor::monitoring_loop, this);
    monitoring_thread.detach();
}

void GPUPerformanceMonitor::stop_monitoring() {
    monitoring_active_ = false;
}

void GPUPerformanceMonitor::monitoring_loop() {
    while (monitoring_active_) {
        collect_metrics();
        std::this_thread::sleep_for(update_interval_);
    }
}

void GPUPerformanceMonitor::collect_metrics() {
    if (!gpu_config_) return;
    
    // Update device metrics
    gpu_config_->update_all_device_metrics();
    
    // Record timestamp
    timestamps_.push_back(std::chrono::steady_clock::now());
    
    // Collect metrics from each device
    auto enabled_devices = gpu_config_->get_enabled_devices();
    for (const auto* device : enabled_devices) {
        int device_id = device->get_device_id();
        
        utilization_history_[device_id].push_back(device->get_utilization());
        temperature_history_[device_id].push_back(device->get_temperature());
        power_history_[device_id].push_back(device->get_power_usage());
        memory_history_[device_id].push_back(device->get_used_memory());
    }
    
    // Limit history size to prevent unbounded growth
    const size_t MAX_HISTORY_SIZE = 3600; // 1 hour at 1-second intervals
    
    if (timestamps_.size() > MAX_HISTORY_SIZE) {
        timestamps_.erase(timestamps_.begin());
        
        for (auto& history : utilization_history_) {
            if (history.second.size() > MAX_HISTORY_SIZE) {
                history.second.erase(history.second.begin());
            }
        }
        
        for (auto& history : temperature_history_) {
            if (history.second.size() > MAX_HISTORY_SIZE) {
                history.second.erase(history.second.begin());
            }
        }
        
        for (auto& history : power_history_) {
            if (history.second.size() > MAX_HISTORY_SIZE) {
                history.second.erase(history.second.begin());
            }
        }
        
        for (auto& history : memory_history_) {
            if (history.second.size() > MAX_HISTORY_SIZE) {
                history.second.erase(history.second.begin());
            }
        }
    }
}

std::unordered_map<int, double> GPUPerformanceMonitor::get_current_utilizations() const {
    if (!gpu_config_) return {};
    return gpu_config_->get_device_utilizations();
}

std::unordered_map<int, double> GPUPerformanceMonitor::get_current_temperatures() const {
    if (!gpu_config_) return {};
    return gpu_config_->get_device_temperatures();
}

std::unordered_map<int, double> GPUPerformanceMonitor::get_average_utilizations() const {
    std::unordered_map<int, double> averages;
    
    for (const auto& history : utilization_history_) {
        if (history.second.empty()) continue;
        
        double sum = std::accumulate(history.second.begin(), history.second.end(), 0.0);
        averages[history.first] = sum / history.second.size();
    }
    
    return averages;
}

std::vector<std::string> GPUPerformanceMonitor::get_active_alerts() const {
    std::vector<std::string> alerts;
    
    if (!gpu_config_) return alerts;
    
    auto enabled_devices = gpu_config_->get_enabled_devices();
    for (const auto* device : enabled_devices) {
        int device_id = device->get_device_id();
        
        if (check_temperature_alert(device_id)) {
            alerts.push_back("High temperature alert for GPU " + std::to_string(device_id) + 
                           " (" + std::to_string(device->get_temperature()) + "°C)");
        }
        
        if (check_utilization_alert(device_id)) {
            alerts.push_back("High utilization alert for GPU " + std::to_string(device_id) + 
                           " (" + std::to_string(device->get_utilization()) + "%)");
        }
    }
    
    return alerts;
}

bool GPUPerformanceMonitor::check_temperature_alert(int device_id) const {
    if (!gpu_config_) return false;
    
    const GPUDevice* device = gpu_config_->get_device(device_id);
    return device && device->get_temperature() > temperature_threshold_;
}

bool GPUPerformanceMonitor::check_utilization_alert(int device_id) const {
    if (!gpu_config_) return false;
    
    const GPUDevice* device = gpu_config_->get_device(device_id);
    return device && device->get_utilization() > utilization_threshold_;
}

std::string GPUPerformanceMonitor::export_metrics_csv() const {
    std::ostringstream oss;
    
    if (timestamps_.empty()) return "";
    
    // Header
    oss << "timestamp";
    
    for (const auto& history : utilization_history_) {
        oss << ",gpu" << history.first << "_utilization";
        oss << ",gpu" << history.first << "_temperature";
        oss << ",gpu" << history.first << "_power";
        oss << ",gpu" << history.first << "_memory";
    }
    oss << "\n";
    
    // Data
    for (size_t i = 0; i < timestamps_.size(); ++i) {
        auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            timestamps_[i].time_since_epoch()).count();
        oss << timestamp_ms;
        
        for (const auto& history : utilization_history_) {
            int device_id = history.first;
            
            // Utilization
            if (i < history.second.size()) {
                oss << "," << history.second[i];
            } else {
                oss << ",";
            }
            
            // Temperature
            auto temp_it = temperature_history_.find(device_id);
            if (temp_it != temperature_history_.end() && i < temp_it->second.size()) {
                oss << "," << temp_it->second[i];
            } else {
                oss << ",";
            }
            
            // Power
            auto power_it = power_history_.find(device_id);
            if (power_it != power_history_.end() && i < power_it->second.size()) {
                oss << "," << power_it->second[i];
            } else {
                oss << ",";
            }
            
            // Memory
            auto mem_it = memory_history_.find(device_id);
            if (mem_it != memory_history_.end() && i < mem_it->second.size()) {
                oss << "," << mem_it->second[i];
            } else {
                oss << ",";
            }
        }
        oss << "\n";
    }
    
    return oss.str();
}

void GPUPerformanceMonitor::clear_history() {
    timestamps_.clear();
    utilization_history_.clear();
    temperature_history_.clear();
    power_history_.clear();
    memory_history_.clear();
}

} // namespace models
} // namespace keyhunt