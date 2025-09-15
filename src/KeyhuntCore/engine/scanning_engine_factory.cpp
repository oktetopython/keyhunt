/**
 * @file scanning_engine_factory.cpp
 * @brief Factory implementation for creating configured scanning engines
 * @author KeyhuntCUDA Team
 * 
 * T045: Factory patterns and engine utilities for integrated scanning engine
 */

#include "integrated_scanning_engine.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace keyhunt {
namespace engine {

std::unique_ptr<IntegratedScanningEngine> ScanningEngineFactory::create_engine(EngineProfile profile) {
    auto engine = std::make_unique<IntegratedScanningEngine>();
    
    // Get available GPU devices
    std::vector<int> available_gpus;
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess) {
        for (int i = 0; i < device_count; i++) {
            available_gpus.push_back(i);
        }
    }
    
    // Get recommended configuration for profile
    auto config = get_recommended_config(profile, available_gpus, 8); // 8GB default
    
    if (!engine->initialize(config)) {
        std::cerr << "ERROR: Failed to initialize engine with " << get_profile_description(profile) << " profile" << std::endl;
        return nullptr;
    }
    
    std::cout << "Created scanning engine with " << get_profile_description(profile) << " profile" << std::endl;
    return engine;
}

ScanningEngineConfig ScanningEngineFactory::get_recommended_config(
    EngineProfile profile,
    const std::vector<int>& available_gpu_devices,
    size_t available_memory_gb) {
    
    ScanningEngineConfig config;
    config.gpu_device_ids = available_gpu_devices;
    
    switch (profile) {
        case EngineProfile::HIGH_PERFORMANCE:
            // Maximum performance configuration
            config.ecc_implementation = ecc::ECC_Implementation::UNIFIED;
            config.enable_ecc_validation = false; // Disable for max speed
            config.ecc_batch_size = 131072; // 128K
            config.keys_per_batch = 10000000; // 10M
            config.enable_distributed_scanning = true;
            config.generate_all_address_formats = true;
            
            // Hash operations - speed optimized
            config.hash_config.batch_size = 131072; // 128K
            config.hash_config.threads_per_block = 512;
            config.hash_config.blocks_per_grid = 512;
            config.hash_config.enable_shared_memory_optimization = true;
            config.hash_config.enable_fused_hash_operations = true;
            config.hash_config.max_concurrent_batches = 8;
            
            // Address comparison - speed optimized  
            config.comparison_config.enable_gpu_comparison = true;
            config.comparison_config.enable_bloom_filter = true;
            config.comparison_config.bloom_filter_size = 4194304; // 4MB
            config.comparison_config.comparison_batch_size = 65536; // 64K
            
            // GPU coordination - aggressive
            config.enable_gpu_coordination = true;
            config.load_balance_config.strategy = gpu::coordination::LoadBalancingConfig::Strategy::DYNAMIC_ADAPTIVE;
            config.load_balance_config.enable_work_stealing = true;
            config.load_balance_config.rebalance_threshold = 0.1; // 10%
            config.load_balance_config.rebalance_interval = std::chrono::seconds(10);
            
            // Checkpointing - less frequent for speed
            config.enable_automatic_checkpointing = true;
            config.checkpoint_interval = std::chrono::seconds(600); // 10 minutes
            
            config.enable_performance_monitoring = true;
            config.enable_scientific_validation = false;
            break;
            
        case EngineProfile::MEMORY_CONSTRAINED:
            // Minimal memory usage configuration
            config.ecc_implementation = ecc::ECC_Implementation::CPU;
            config.enable_ecc_validation = true;
            config.ecc_batch_size = 16384; // 16K
            config.keys_per_batch = 1000000; // 1M
            config.enable_distributed_scanning = available_gpu_devices.size() > 1;
            config.generate_all_address_formats = false; // P2PKH only
            
            // Hash operations - memory optimized
            config.hash_config.batch_size = 16384; // 16K
            config.hash_config.threads_per_block = 128;
            config.hash_config.blocks_per_grid = 64;
            config.hash_config.enable_shared_memory_optimization = false;
            config.hash_config.max_concurrent_batches = 2;
            
            // Address comparison - memory optimized
            config.comparison_config.enable_gpu_comparison = false;
            config.comparison_config.enable_bloom_filter = true;
            config.comparison_config.bloom_filter_size = 262144; // 256KB
            config.comparison_config.comparison_batch_size = 8192; // 8K
            
            // GPU coordination - conservative
            config.enable_gpu_coordination = false;
            
            // Checkpointing - more frequent to save progress
            config.enable_automatic_checkpointing = true;
            config.checkpoint_interval = std::chrono::seconds(120); // 2 minutes
            
            config.enable_performance_monitoring = false;
            break;
            
        case EngineProfile::LOW_POWER:
            // Power-efficient configuration
            config.ecc_implementation = ecc::ECC_Implementation::CPU;
            config.enable_ecc_validation = true;
            config.ecc_batch_size = 32768; // 32K
            config.keys_per_batch = 2000000; // 2M
            config.enable_distributed_scanning = false;
            config.generate_all_address_formats = true;
            
            // Hash operations - power efficient
            config.hash_config.batch_size = 32768; // 32K
            config.hash_config.threads_per_block = 128;
            config.hash_config.blocks_per_grid = 128;
            config.hash_config.enable_shared_memory_optimization = false;
            config.hash_config.enable_fused_hash_operations = false;
            config.hash_config.max_concurrent_batches = 1;
            
            // Address comparison - power efficient
            config.comparison_config.enable_gpu_comparison = false;
            config.comparison_config.enable_bloom_filter = true;
            config.comparison_config.bloom_filter_size = 524288; // 512KB
            config.comparison_config.comparison_batch_size = 16384; // 16K
            
            // GPU coordination - minimal
            config.enable_gpu_coordination = false;
            
            // Checkpointing - balanced
            config.enable_automatic_checkpointing = true;
            config.checkpoint_interval = std::chrono::seconds(300); // 5 minutes
            
            config.enable_performance_monitoring = true;
            break;
            
        case EngineProfile::SCIENTIFIC:
            // Scientific validation focused configuration
            config.ecc_implementation = ecc::ECC_Implementation::UNIFIED;
            config.enable_ecc_validation = true;
            config.ecc_batch_size = 65536; // 64K
            config.keys_per_batch = 1000000; // 1M
            config.enable_distributed_scanning = true;
            config.generate_all_address_formats = true;
            
            // Hash operations - validation focused
            config.hash_config.batch_size = 65536; // 64K
            config.hash_config.threads_per_block = 256;
            config.hash_config.blocks_per_grid = 256;
            config.hash_config.enable_shared_memory_optimization = true;
            config.hash_config.enable_fused_hash_operations = true;
            
            // Address comparison - thorough validation
            config.comparison_config.enable_gpu_comparison = true;
            config.comparison_config.enable_bloom_filter = true;
            config.comparison_config.bloom_filter_size = 1048576; // 1MB
            config.comparison_config.comparison_batch_size = 32768; // 32K
            
            // GPU coordination - with validation
            config.enable_gpu_coordination = true;
            config.load_balance_config.strategy = gpu::coordination::LoadBalancingConfig::Strategy::PERFORMANCE_WEIGHTED;
            config.load_balance_config.enable_work_stealing = true;
            
            // Comprehensive validation
            config.enable_performance_monitoring = true;
            config.enable_scientific_validation = true;
            config.validation_sample_size = 1000000; // 1M sample
            
            // Frequent checkpointing for data integrity
            config.enable_automatic_checkpointing = true;
            config.checkpoint_interval = std::chrono::seconds(180); // 3 minutes
            break;
            
        case EngineProfile::EXPERIMENTAL:
            // Experimental features enabled
            config.ecc_implementation = ecc::ECC_Implementation::GPU;
            config.enable_ecc_validation = true;
            config.ecc_batch_size = 65536; // 64K
            config.keys_per_batch = 5000000; // 5M
            config.enable_distributed_scanning = true;
            config.generate_all_address_formats = true;
            
            // Hash operations - experimental optimizations
            config.hash_config.batch_size = 65536; // 64K
            config.hash_config.threads_per_block = 256;
            config.hash_config.blocks_per_grid = 256;
            config.hash_config.enable_shared_memory_optimization = true;
            config.hash_config.enable_texture_memory = true;
            config.hash_config.enable_fused_hash_operations = true;
            config.hash_config.max_concurrent_batches = 6;
            
            // Address comparison - experimental
            config.comparison_config.enable_gpu_comparison = true;
            config.comparison_config.enable_bloom_filter = true;
            config.comparison_config.bloom_filter_size = 2097152; // 2MB
            config.comparison_config.enable_prefix_optimization = true;
            
            // GPU coordination - experimental strategies
            config.enable_gpu_coordination = true;
            config.load_balance_config.strategy = gpu::coordination::LoadBalancingConfig::Strategy::HETEROGENEOUS_AWARE;
            config.load_balance_config.enable_work_stealing = true;
            
            config.enable_performance_monitoring = true;
            config.enable_scientific_validation = true;
            config.validation_sample_size = 500000; // 500K sample
            
            config.enable_automatic_checkpointing = true;
            config.checkpoint_interval = std::chrono::seconds(240); // 4 minutes
            break;
            
        default: // BALANCED
            config.ecc_implementation = ecc::ECC_Implementation::UNIFIED;
            config.enable_ecc_validation = true;
            config.ecc_batch_size = 65536; // 64K
            config.keys_per_batch = 5000000; // 5M
            config.enable_distributed_scanning = available_gpu_devices.size() > 1;
            config.generate_all_address_formats = true;
            
            // Hash operations - balanced
            config.hash_config.batch_size = 65536; // 64K
            config.hash_config.threads_per_block = 256;
            config.hash_config.blocks_per_grid = 256;
            config.hash_config.enable_shared_memory_optimization = true;
            config.hash_config.enable_fused_hash_operations = true;
            config.hash_config.max_concurrent_batches = 4;
            
            // Address comparison - balanced
            config.comparison_config.enable_gpu_comparison = true;
            config.comparison_config.enable_bloom_filter = true;
            config.comparison_config.bloom_filter_size = 1048576; // 1MB
            config.comparison_config.comparison_batch_size = 32768; // 32K
            
            // GPU coordination - balanced
            config.enable_gpu_coordination = available_gpu_devices.size() > 1;
            config.load_balance_config.strategy = gpu::coordination::LoadBalancingConfig::Strategy::DYNAMIC_ADAPTIVE;
            config.load_balance_config.enable_work_stealing = true;
            config.load_balance_config.rebalance_threshold = 0.2; // 20%
            config.load_balance_config.rebalance_interval = std::chrono::seconds(30);
            
            config.enable_performance_monitoring = true;
            config.enable_scientific_validation = false;
            
            config.enable_automatic_checkpointing = true;
            config.checkpoint_interval = std::chrono::seconds(300); // 5 minutes
            break;
    }
    
    // Adjust based on available memory
    if (available_memory_gb < 4) {
        // Reduce batch sizes for low memory systems
        config.ecc_batch_size = std::min(config.ecc_batch_size, size_t(32768));
        config.hash_config.batch_size = std::min(config.hash_config.batch_size, size_t(32768));
        config.keys_per_batch = std::min(config.keys_per_batch, size_t(2000000));
        config.comparison_config.bloom_filter_size = std::min(config.comparison_config.bloom_filter_size, size_t(524288));
    } else if (available_memory_gb > 16) {
        // Increase batch sizes for high memory systems
        if (profile == EngineProfile::HIGH_PERFORMANCE) {
            config.ecc_batch_size = std::max(config.ecc_batch_size, size_t(262144));
            config.hash_config.batch_size = std::max(config.hash_config.batch_size, size_t(262144));
            config.keys_per_batch = std::max(config.keys_per_batch, size_t(20000000));
        }
    }
    
    // Adjust for number of available GPUs
    if (available_gpu_devices.size() > 4) {
        // Many GPUs - enable aggressive coordination
        config.enable_gpu_coordination = true;
        config.load_balance_config.max_work_steal_attempts = 5;
        config.hash_config.max_concurrent_batches = std::min(
            config.hash_config.max_concurrent_batches, 
            available_gpu_devices.size() * 2
        );
    }
    
    return config;
}

std::vector<std::string> ScanningEngineFactory::get_profile_descriptions() {
    return {
        "HIGH_PERFORMANCE: Maximum speed configuration with aggressive optimizations",
        "BALANCED: Balanced speed, memory, and power usage (recommended for most users)",
        "LOW_POWER: Power-efficient configuration for extended operation",
        "MEMORY_CONSTRAINED: Minimal memory usage for resource-limited systems",
        "SCIENTIFIC: Scientific validation focused with comprehensive verification",
        "EXPERIMENTAL: Latest experimental features and optimizations"
    };
}

std::string ScanningEngineFactory::get_profile_description(EngineProfile profile) {
    switch (profile) {
        case EngineProfile::HIGH_PERFORMANCE:
            return "High Performance: Maximum speed configuration";
        case EngineProfile::BALANCED:
            return "Balanced: Balanced speed/memory/power usage";
        case EngineProfile::LOW_POWER:
            return "Low Power: Power-efficient configuration";
        case EngineProfile::MEMORY_CONSTRAINED:
            return "Memory Constrained: Minimal memory usage";
        case EngineProfile::SCIENTIFIC:
            return "Scientific: Validation focused configuration";
        case EngineProfile::EXPERIMENTAL:
            return "Experimental: Latest features and optimizations";
        default:
            return "Unknown Profile";
    }
}

// Engine utilities namespace implementation
namespace engine_utils {

double calculate_scanning_efficiency(const ScanningEngineMetrics& metrics) {
    if (metrics.total_scan_time.count() <= 0) {
        return 0.0;
    }
    
    // Calculate various efficiency factors
    double time_efficiency = 1.0;
    if (metrics.overall_progress_percentage > 0) {
        double expected_time = (metrics.total_scan_time.count() * 100.0) / metrics.overall_progress_percentage;
        time_efficiency = metrics.total_scan_time.count() / expected_time;
    }
    
    double resource_efficiency = metrics.resource_utilization_efficiency;
    if (resource_efficiency <= 0) {
        resource_efficiency = metrics.gpu_utilization_average;
    }
    
    double computational_efficiency = metrics.computational_efficiency;
    if (computational_efficiency <= 0) {
        // Calculate based on key processing rate
        double theoretical_max_rate = metrics.active_gpu_count * 1000000.0; // 1M keys/sec per GPU estimate
        if (theoretical_max_rate > 0) {
            computational_efficiency = std::min(1.0, metrics.keys_per_second_average / theoretical_max_rate);
        }
    }
    
    // Combined efficiency score
    return (time_efficiency * 0.3 + resource_efficiency * 0.4 + computational_efficiency * 0.3);
}

double estimate_completion_time(const ScanningEngineMetrics& metrics, const ecc::BigInt256& remaining_keyspace) {
    if (metrics.keys_per_second_average <= 0) {
        return -1.0; // Cannot estimate
    }
    
    uint64_t remaining_keys = remaining_keyspace.to_uint64();
    if (remaining_keys == 0) {
        return 0.0; // Already complete
    }
    
    double seconds_remaining = static_cast<double>(remaining_keys) / metrics.keys_per_second_average;
    return seconds_remaining;
}

std::vector<std::string> analyze_performance_bottlenecks(const ScanningEngineMetrics& metrics) {
    std::vector<std::string> bottlenecks;
    
    // Low GPU utilization
    if (metrics.gpu_utilization_average < 0.7) {
        bottlenecks.push_back("Low GPU utilization - consider increasing batch sizes or reducing CPU-GPU synchronization");
    }
    
    // Poor load balancing
    if (metrics.gpu_load_balance_coefficient < 0.8) {
        bottlenecks.push_back("Poor load balancing - enable work stealing or adjust load balancing strategy");
    }
    
    // Low address generation rate
    if (metrics.address_generation_rate < metrics.keys_per_second_average * 0.8) {
        bottlenecks.push_back("Address generation bottleneck - optimize hash operations or increase hash batch size");
    }
    
    // High error rate
    if (metrics.total_errors_encountered > metrics.total_keys_processed.to_uint64() / 1000) {
        bottlenecks.push_back("High error rate detected - check device stability and reduce batch sizes");
    }
    
    // Memory bandwidth issues
    if (metrics.memory_bandwidth_utilization < 0.5) {
        bottlenecks.push_back("Low memory bandwidth utilization - consider optimizing data access patterns");
    }
    
    // Excessive recovery operations
    if (metrics.recovery_operations_performed > 10) {
        bottlenecks.push_back("Frequent recovery operations - investigate system stability issues");
    }
    
    if (bottlenecks.empty()) {
        bottlenecks.push_back("No significant performance bottlenecks detected");
    }
    
    return bottlenecks;
}

ScanningEngineConfig optimize_config_for_hardware(const ScanningEngineConfig& base_config) {
    ScanningEngineConfig optimized = base_config;
    
    // Query GPU devices for capabilities
    std::vector<gpu::coordination::GPUDeviceInfo> gpu_info;
    for (int device_id : base_config.gpu_device_ids) {
        cudaDeviceProp props;
        if (cudaGetDeviceProperties(&props, device_id) == cudaSuccess) {
            gpu::coordination::GPUDeviceInfo info;
            info.device_id = device_id;
            info.device_name = props.name;
            info.total_memory = props.totalGlobalMem;
            info.compute_capability_major = props.major;
            info.compute_capability_minor = props.minor;
            info.multiprocessor_count = props.multiProcessorCount;
            
            gpu_info.push_back(info);
        }
    }
    
    if (!gpu_info.empty()) {
        // Optimize based on GPU capabilities
        size_t min_memory = SIZE_MAX;
        int min_compute_capability = INT_MAX;
        
        for (const auto& info : gpu_info) {
            min_memory = std::min(min_memory, info.total_memory);
            min_compute_capability = std::min(min_compute_capability, 
                info.compute_capability_major * 10 + info.compute_capability_minor);
        }
        
        // Adjust batch sizes based on minimum memory
        size_t memory_gb = min_memory / (1024 * 1024 * 1024);
        if (memory_gb < 4) {
            optimized.hash_config.batch_size = std::min(optimized.hash_config.batch_size, size_t(32768));
            optimized.keys_per_batch = std::min(optimized.keys_per_batch, size_t(2000000));
        } else if (memory_gb > 16) {
            optimized.hash_config.batch_size = std::max(optimized.hash_config.batch_size, size_t(131072));
            optimized.keys_per_batch = std::max(optimized.keys_per_batch, size_t(10000000));
        }
        
        // Adjust thread configuration based on compute capability
        if (min_compute_capability >= 75) { // Turing or newer
            optimized.hash_config.threads_per_block = 256;
            optimized.hash_config.enable_shared_memory_optimization = true;
            optimized.hash_config.enable_fused_hash_operations = true;
        } else if (min_compute_capability >= 60) { // Pascal or newer
            optimized.hash_config.threads_per_block = 256;
            optimized.hash_config.enable_shared_memory_optimization = true;
        } else {
            // Older architectures
            optimized.hash_config.threads_per_block = 128;
            optimized.hash_config.enable_shared_memory_optimization = false;
        }
    }
    
    return optimized;
}

bool validate_target_addresses(const std::vector<std::string>& addresses) {
    if (addresses.empty()) {
        return false;
    }
    
    for (const auto& address : addresses) {
        if (!compare::BitcoinAddressGenerator::validate_address(address)) {
            std::cerr << "ERROR: Invalid target address: " << address << std::endl;
            return false;
        }
    }
    
    return true;
}

ecc::BigInt256 calculate_keyspace_size(const models::PrivateKeyRange& range) {
    if (range.start_key >= range.end_key) {
        ecc::BigInt256 zero;
        zero.set_zero();
        return zero;
    }
    
    return range.end_key - range.start_key;
}

bool export_matches_to_csv(const std::vector<ScanningMatch>& matches, const std::string& filename) {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        // CSV header
        file << "PrivateKey,Address,AddressFormat,FoundTime,BatchID,DeviceID,CPUValidated,GPUValidated,Confidence\n";
        
        // CSV data
        for (const auto& match : matches) {
            auto time_t = std::chrono::system_clock::to_time_t(match.found_time);
            
            file << match.private_key.to_hex() << ","
                 << match.address << ","
                 << static_cast<int>(match.address_format) << ","
                 << time_t << ","
                 << match.batch_id << ","
                 << match.device_id << ","
                 << (match.cpu_validated ? "true" : "false") << ","
                 << (match.gpu_validated ? "true" : "false") << ","
                 << std::fixed << std::setprecision(3) << match.validation_confidence << "\n";
        }
        
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in export_matches_to_csv: " << e.what() << std::endl;
        return false;
    }
}

bool export_matches_to_json(const std::vector<ScanningMatch>& matches, const std::string& filename) {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        file << "{\n  \"matches\": [\n";
        
        for (size_t i = 0; i < matches.size(); i++) {
            const auto& match = matches[i];
            auto time_t = std::chrono::system_clock::to_time_t(match.found_time);
            
            file << "    {\n";
            file << "      \"private_key\": \"" << match.private_key.to_hex() << "\",\n";
            file << "      \"address\": \"" << match.address << "\",\n";
            file << "      \"address_format\": " << static_cast<int>(match.address_format) << ",\n";
            file << "      \"found_time\": " << time_t << ",\n";
            file << "      \"batch_id\": " << match.batch_id << ",\n";
            file << "      \"device_id\": " << match.device_id << ",\n";
            file << "      \"cpu_validated\": " << (match.cpu_validated ? "true" : "false") << ",\n";
            file << "      \"gpu_validated\": " << (match.gpu_validated ? "true" : "false") << ",\n";
            file << "      \"validation_confidence\": " << std::fixed << std::setprecision(3) << match.validation_confidence << "\n";
            file << "    }";
            
            if (i < matches.size() - 1) {
                file << ",";
            }
            file << "\n";
        }
        
        file << "  ]\n}\n";
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in export_matches_to_json: " << e.what() << std::endl;
        return false;
    }
}

std::string generate_scanning_report(const ScanningEngineMetrics& metrics, const ScanningStatistics& stats) {
    std::ostringstream report;
    
    report << "KeyhuntCUDA Scanning Report\n";
    report << "==========================\n\n";
    
    // Basic statistics
    report << "Scanning Overview:\n";
    report << "  Status: " << static_cast<int>(metrics.current_status) << "\n";
    report << "  Progress: " << std::fixed << std::setprecision(2) << metrics.overall_progress_percentage << "%\n";
    report << "  Runtime: " << (metrics.total_scan_time.count() / 1000) << " seconds\n";
    report << "  Keys processed: " << metrics.total_keys_processed.to_hex() << "\n\n";
    
    // Performance metrics
    report << "Performance Metrics:\n";
    report << "  Current speed: " << std::fixed << std::setprecision(0) << metrics.keys_per_second_current << " keys/sec\n";
    report << "  Average speed: " << std::fixed << std::setprecision(0) << metrics.keys_per_second_average << " keys/sec\n";
    report << "  Peak speed: " << std::fixed << std::setprecision(0) << metrics.keys_per_second_peak << " keys/sec\n";
    report << "  Scanning efficiency: " << std::fixed << std::setprecision(3) << metrics.scanning_efficiency << "\n\n";
    
    // GPU information
    report << "GPU Utilization:\n";
    report << "  Active GPUs: " << metrics.active_gpu_count << "\n";
    report << "  Average utilization: " << std::fixed << std::setprecision(1) << (metrics.gpu_utilization_average * 100) << "%\n";
    report << "  Load balance coefficient: " << std::fixed << std::setprecision(3) << metrics.gpu_load_balance_coefficient << "\n";
    report << "  Work stealing events: " << metrics.gpu_work_stealing_events << "\n\n";
    
    // Match results
    report << "Results:\n";
    report << "  Total matches found: " << metrics.total_matches_found << "\n";
    report << "  P2PKH matches: " << metrics.p2pkh_matches << "\n";
    report << "  P2SH matches: " << metrics.p2sh_matches << "\n";
    report << "  Bech32 matches: " << metrics.bech32_matches << "\n\n";
    
    // Error and recovery information
    if (metrics.total_errors_encountered > 0) {
        report << "Error Information:\n";
        report << "  Total errors: " << metrics.total_errors_encountered << "\n";
        report << "  Recovery operations: " << metrics.recovery_operations_performed << "\n";
        report << "  Device failures: " << metrics.device_failures_detected << "\n";
        report << "  Recovery time: " << metrics.total_recovery_time.count() << " ms\n\n";
    }
    
    // Resource utilization
    report << "Resource Utilization:\n";
    report << "  GPU memory used: " << (metrics.total_gpu_memory_used / 1024 / 1024) << " MB\n";
    report << "  CPU memory used: " << (metrics.total_cpu_memory_used / 1024 / 1024) << " MB\n";
    report << "  Memory bandwidth: " << std::fixed << std::setprecision(1) << (metrics.memory_bandwidth_utilization * 100) << "%\n";
    report << "  Checkpoint files: " << metrics.checkpoint_files_created << "\n";
    report << "  Checkpoint data: " << (metrics.checkpoint_data_size / 1024 / 1024) << " MB\n\n";
    
    report << "Report generated at: " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
    
    return report.str();
}

bool verify_match_authenticity(const ScanningMatch& match) {
    // Basic verification
    if (match.private_key.is_zero() || match.address.empty()) {
        return false;
    }
    
    // Verify validation confidence is reasonable
    if (match.validation_confidence < 0.5) {
        return false;
    }
    
    // At least one validation method should have passed
    if (!match.cpu_validated && !match.gpu_validated) {
        return false;
    }
    
    // Address format should be valid
    if (match.address_format == compare::AddressFormat::UNKNOWN) {
        return false;
    }
    
    return true;
}

double calculate_match_probability(const models::PrivateKeyRange& range, size_t target_count) {
    if (target_count == 0) {
        return 0.0;
    }
    
    // Calculate keyspace size
    ecc::BigInt256 keyspace_size = calculate_keyspace_size(range);
    if (keyspace_size.is_zero()) {
        return 0.0;
    }
    
    // Approximate probability calculation
    // P(at least one match) ≈ 1 - (1 - targets/total_keyspace)^range_size
    // For small probabilities: P ≈ targets * range_size / total_keyspace
    
    uint64_t range_size = keyspace_size.to_uint64();
    const uint64_t total_bitcoin_keyspace = UINT64_MAX; // Approximation for 2^256
    
    double probability = static_cast<double>(target_count * range_size) / static_cast<double>(total_bitcoin_keyspace);
    return std::min(1.0, probability);
}

} // namespace engine_utils

} // namespace engine
} // namespace keyhunt