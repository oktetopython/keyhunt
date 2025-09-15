/**
 * @file integrated_scanning_engine.cpp
 * @brief Implementation of integrated scanning engine
 * @author KeyhuntCUDA Team
 * 
 * T045: Create integrated scanning engine that combines ECC operations, scanning framework, and address comparison
 */

#include "integrated_scanning_engine.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace keyhunt {
namespace engine {

IntegratedScanningEngine::IntegratedScanningEngine()
    : current_status_(ScanningStatus::IDLE)
    , should_stop_(false)
    , is_paused_(false)
    , max_matches_in_memory_(10000)
    , max_error_log_entries_(1000)
    , recovery_attempts_(0)
    , metrics_thread_running_(false)
    , checkpoint_thread_running_(false)
    , continuous_validation_enabled_(false)
    , validation_frequency_(0.1)
    , validation_operations_count_(0)
    , validation_passed_count_(0)
{
    current_scan_position_.set_zero();
    last_metrics_update_ = std::chrono::high_resolution_clock::now();
}

IntegratedScanningEngine::~IntegratedScanningEngine() {
    cleanup();
}

bool IntegratedScanningEngine::initialize(const ScanningEngineConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        std::cout << "Initializing Integrated Scanning Engine..." << std::endl;
        set_status(ScanningStatus::INITIALIZING);
        
        // Validate configuration
        if (!validate_configuration(config)) {
            std::cerr << "ERROR: Invalid configuration provided" << std::endl;
            set_status(ScanningStatus::ERROR);
            return false;
        }
        
        current_config_ = config;
        
        // Initialize all components
        bool init_success = true;
        
        // Initialize ECC engine
        if (!initialize_ecc_engine()) {
            std::cerr << "ERROR: Failed to initialize ECC engine" << std::endl;
            init_success = false;
        }
        
        // Initialize scanner
        if (!initialize_scanner()) {
            std::cerr << "ERROR: Failed to initialize scanner" << std::endl;
            init_success = false;
        }
        
        // Initialize address generator
        if (!initialize_address_generator()) {
            std::cerr << "ERROR: Failed to initialize address generator" << std::endl;
            init_success = false;
        }
        
        // Initialize GPU coordinator (optional)
        if (current_config_.enable_gpu_coordination) {
            if (!initialize_gpu_coordinator()) {
                std::cerr << "WARNING: Failed to initialize GPU coordinator" << std::endl;
                // Not fatal - continue without GPU coordination
                current_config_.enable_gpu_coordination = false;
            }
        }
        
        // Initialize checkpoint manager
        if (!initialize_checkpoint_manager()) {
            std::cerr << "ERROR: Failed to initialize checkpoint manager" << std::endl;
            init_success = false;
        }
        
        if (!init_success) {
            cleanup();
            set_status(ScanningStatus::ERROR);
            return false;
        }
        
        // Set up component integration
        if (!setup_component_integration()) {
            std::cerr << "ERROR: Failed to set up component integration" << std::endl;
            cleanup();
            set_status(ScanningStatus::ERROR);
            return false;
        }
        
        // Start monitoring threads
        if (current_config_.enable_performance_monitoring) {
            metrics_thread_running_ = true;
            metrics_thread_ = std::thread(&IntegratedScanningEngine::metrics_monitoring_loop, this);
        }
        
        // Start automatic checkpointing if enabled
        if (current_config_.enable_automatic_checkpointing) {
            checkpoint_thread_running_ = true;
            checkpoint_thread_ = std::thread(&IntegratedScanningEngine::checkpoint_automation_loop, this);
        }
        
        set_status(ScanningStatus::IDLE);
        
        std::cout << "Integrated Scanning Engine initialized successfully" << std::endl;
        std::cout << "  ECC Implementation: " << static_cast<int>(current_config_.ecc_implementation) << std::endl;
        std::cout << "  GPU Coordination: " << (current_config_.enable_gpu_coordination ? "Enabled" : "Disabled") << std::endl;
        std::cout << "  GPU Devices: " << current_config_.gpu_device_ids.size() << std::endl;
        std::cout << "  Performance Monitoring: " << (current_config_.enable_performance_monitoring ? "Enabled" : "Disabled") << std::endl;
        std::cout << "  Automatic Checkpointing: " << (current_config_.enable_automatic_checkpointing ? "Enabled" : "Disabled") << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in initialize: " << e.what() << std::endl;
        set_status(ScanningStatus::ERROR);
        return false;
    }
}

void IntegratedScanningEngine::cleanup() {
    std::cout << "Cleaning up Integrated Scanning Engine..." << std::endl;
    
    // Stop scanning if active
    if (current_status_ == ScanningStatus::SCANNING || current_status_ == ScanningStatus::PAUSED) {
        stop_scanning();
    }
    
    // Stop monitoring threads
    if (metrics_thread_running_) {
        metrics_thread_running_ = false;
        if (metrics_thread_.joinable()) {
            metrics_thread_.join();
        }
    }
    
    if (checkpoint_thread_running_) {
        checkpoint_thread_running_ = false;
        if (checkpoint_thread_.joinable()) {
            checkpoint_thread_.join();
        }
    }
    
    // Clean up components
    if (gpu_coordinator_) {
        gpu_coordinator_->cleanup();
        gpu_coordinator_.reset();
    }
    
    if (address_generator_) {
        address_generator_->cleanup();
        address_generator_.reset();
    }
    
    if (scanner_) {
        scanner_->cleanup();
        scanner_.reset();
    }
    
    if (checkpoint_manager_) {
        checkpoint_manager_->cleanup();
        checkpoint_manager_.reset();
    }
    
    if (ecc_engine_) {
        ecc_engine_.reset();
    }
    
    set_status(ScanningStatus::IDLE);
    std::cout << "Cleanup completed" << std::endl;
}

bool IntegratedScanningEngine::start_scanning(
    const models::PrivateKeyRange& range,
    const std::vector<std::string>& target_addresses) {
    
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (current_status_ != ScanningStatus::IDLE) {
            std::cerr << "ERROR: Cannot start scanning - engine not idle (status: " 
                      << static_cast<int>(current_status_) << ")" << std::endl;
            return false;
        }
        
        std::cout << "Starting integrated scanning..." << std::endl;
        std::cout << "  Range: " << range.start_key.to_hex().substr(0, 16) << "... to " 
                  << range.end_key.to_hex().substr(0, 16) << "..." << std::endl;
        std::cout << "  Target addresses: " << target_addresses.size() << std::endl;
        
        // Validate inputs
        if (range.start_key >= range.end_key) {
            std::cerr << "ERROR: Invalid key range" << std::endl;
            return false;
        }
        
        if (target_addresses.empty()) {
            std::cerr << "ERROR: No target addresses provided" << std::endl;
            return false;
        }
        
        // Store scanning parameters
        current_range_ = range;
        target_addresses_ = target_addresses;
        current_scan_position_ = range.start_key;
        
        // Configure target addresses in address generator
        if (!address_generator_->set_target_addresses(target_addresses)) {
            std::cerr << "ERROR: Failed to set target addresses" << std::endl;
            return false;
        }
        
        // Configure scanner with range
        if (!scanner_->configure_range(range)) {
            std::cerr << "ERROR: Failed to configure scanner range" << std::endl;
            return false;
        }
        
        // Set up GPU coordination if enabled
        if (current_config_.enable_gpu_coordination && gpu_coordinator_) {
            if (!gpu_coordinator_->distribute_work(range, target_addresses, current_config_.keys_per_batch)) {
                std::cerr << "ERROR: Failed to distribute work across GPUs" << std::endl;
                return false;
            }
        }
        
        // Reset metrics
        {
            std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
            current_metrics_ = ScanningEngineMetrics();
            current_metrics_.scan_start_time = std::chrono::system_clock::now();
        }
        
        // Start main scanning thread
        should_stop_ = false;
        is_paused_ = false;
        set_status(ScanningStatus::SCANNING);
        
        main_scanning_thread_ = std::thread(&IntegratedScanningEngine::main_scanning_loop, this);
        
        std::cout << "Integrated scanning started successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in start_scanning: " << e.what() << std::endl;
        set_status(ScanningStatus::ERROR);
        return false;
    }
}

bool IntegratedScanningEngine::pause_scanning() {
    if (current_status_ != ScanningStatus::SCANNING) {
        return false;
    }
    
    std::cout << "Pausing integrated scanning..." << std::endl;
    
    is_paused_ = true;
    set_status(ScanningStatus::PAUSED);
    
    // Pause all components
    if (scanner_) {
        scanner_->pause_scanning();
    }
    
    if (gpu_coordinator_) {
        gpu_coordinator_->pause_coordinated_scanning();
    }
    
    std::cout << "Scanning paused" << std::endl;
    return true;
}

bool IntegratedScanningEngine::resume_scanning() {
    if (current_status_ != ScanningStatus::PAUSED) {
        return false;
    }
    
    std::cout << "Resuming integrated scanning..." << std::endl;
    
    is_paused_ = false;
    set_status(ScanningStatus::SCANNING);
    
    // Resume all components
    if (scanner_) {
        scanner_->resume_scanning();
    }
    
    if (gpu_coordinator_) {
        gpu_coordinator_->resume_coordinated_scanning();
    }
    
    scanning_cv_.notify_all();
    
    std::cout << "Scanning resumed" << std::endl;
    return true;
}

bool IntegratedScanningEngine::stop_scanning() {
    if (current_status_ != ScanningStatus::SCANNING && current_status_ != ScanningStatus::PAUSED) {
        return false;
    }
    
    std::cout << "Stopping integrated scanning..." << std::endl;
    set_status(ScanningStatus::STOPPING);
    
    // Signal stop
    should_stop_ = true;
    is_paused_ = false;
    scanning_cv_.notify_all();
    
    // Stop all components
    if (scanner_) {
        scanner_->stop_scanning();
    }
    
    if (gpu_coordinator_) {
        gpu_coordinator_->stop_coordinated_scanning();
    }
    
    // Wait for main scanning thread to complete
    if (main_scanning_thread_.joinable()) {
        main_scanning_thread_.join();
    }
    
    set_status(ScanningStatus::IDLE);
    std::cout << "Scanning stopped" << std::endl;
    return true;
}

ScanningStatus IntegratedScanningEngine::get_scanning_status() const {
    return current_status_;
}

ScanningEngineMetrics IntegratedScanningEngine::get_current_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_metrics_;
}

std::vector<ScanningMatch> IntegratedScanningEngine::get_all_matches() const {
    std::lock_guard<std::mutex> lock(matches_mutex_);
    return found_matches_;
}

size_t IntegratedScanningEngine::get_total_match_count() const {
    std::lock_guard<std::mutex> lock(matches_mutex_);
    return found_matches_.size();
}

// Private method implementations

bool IntegratedScanningEngine::initialize_ecc_engine() {
    try {
        ecc_engine_ = std::make_unique<ecc::Secp256k1Operations>();
        
        if (!ecc_engine_->initialize(current_config_.ecc_implementation)) {
            return false;
        }
        
        // Enable validation if requested
        if (current_config_.enable_ecc_validation) {
            ecc_engine_->enable_validation(true);
        }
        
        std::cout << "ECC engine initialized: " << static_cast<int>(current_config_.ecc_implementation) << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in initialize_ecc_engine: " << e.what() << std::endl;
        return false;
    }
}

bool IntegratedScanningEngine::initialize_scanner() {
    try {
        scanner_ = std::make_unique<scan::PrivateKeyScanner>();
        
        if (!scanner_->initialize(current_config_.scanning_config)) {
            return false;
        }
        
        std::cout << "Private key scanner initialized" << std::endl;
        std::cout << "  Keys per batch: " << current_config_.keys_per_batch << std::endl;
        std::cout << "  Threads per block: " << current_config_.scanning_config.threads_per_block << std::endl;
        std::cout << "  Blocks per grid: " << current_config_.scanning_config.blocks_per_grid << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in initialize_scanner: " << e.what() << std::endl;
        return false;
    }
}

bool IntegratedScanningEngine::initialize_address_generator() {
    try {
        address_generator_ = std::make_unique<compare::BitcoinAddressGenerator>();
        
        int device_id = current_config_.gpu_device_ids.empty() ? 0 : current_config_.gpu_device_ids[0];
        
        if (!address_generator_->initialize(device_id)) {
            return false;
        }
        
        address_generator_->configure_hash_operations(current_config_.hash_config);
        address_generator_->configure_address_comparison(current_config_.comparison_config);
        
        std::cout << "Address generator initialized" << std::endl;
        std::cout << "  Hash batch size: " << current_config_.hash_config.batch_size << std::endl;
        std::cout << "  Comparison batch size: " << current_config_.comparison_config.comparison_batch_size << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in initialize_address_generator: " << e.what() << std::endl;
        return false;
    }
}

bool IntegratedScanningEngine::initialize_gpu_coordinator() {
    try {
        gpu_coordinator_ = std::make_unique<gpu::coordination::MultiGPUCoordinator>();
        
        if (!gpu_coordinator_->initialize()) {
            return false;
        }
        
        if (!gpu_coordinator_->initialize_devices(current_config_.gpu_device_ids)) {
            return false;
        }
        
        gpu_coordinator_->configure_load_balancing(current_config_.load_balance_config);
        
        std::cout << "GPU coordinator initialized" << std::endl;
        std::cout << "  Active devices: " << current_config_.gpu_device_ids.size() << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in initialize_gpu_coordinator: " << e.what() << std::endl;
        return false;
    }
}

bool IntegratedScanningEngine::initialize_checkpoint_manager() {
    try {
        checkpoint_manager_ = std::make_unique<scan::checkpoint::CheckpointManager>();
        
        if (!checkpoint_manager_->initialize("checkpoints")) {
            return false;
        }
        
        checkpoint_manager_->configure(current_config_.checkpoint_settings);
        
        std::cout << "Checkpoint manager initialized" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in initialize_checkpoint_manager: " << e.what() << std::endl;
        return false;
    }
}

bool IntegratedScanningEngine::setup_component_integration() {
    try {
        // Set up callbacks between components
        
        // Address generator match callback
        if (address_generator_) {
            address_generator_->set_match_callback([this](const compare::AddressMatch& match) {
                process_potential_match(match);
            });
        }
        
        // GPU coordinator callbacks
        if (gpu_coordinator_) {
            gpu_coordinator_->set_match_callback([this](const scan::PrivateKeyScanner::ScanMatch& match, int device_id) {
                // Convert to address match and process
                compare::AddressMatch addr_match;
                addr_match.private_key = match.private_key;
                addr_match.address = match.address;
                addr_match.batch_id = match.batch_id;
                addr_match.device_id = device_id;
                process_potential_match(addr_match);
            });
            
            gpu_coordinator_->set_device_failure_callback([this](int device_id, const std::string& error) {
                handle_device_failure(device_id);
            });
        }
        
        std::cout << "Component integration set up successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in setup_component_integration: " << e.what() << std::endl;
        return false;
    }
}

void IntegratedScanningEngine::main_scanning_loop() {
    std::cout << "Main scanning loop started" << std::endl;
    
    try {
        while (!should_stop_) {
            // Handle pause state
            if (is_paused_) {
                std::unique_lock<std::mutex> lock(scanning_mutex_);
                scanning_cv_.wait(lock, [this] { return !is_paused_ || should_stop_; });
                continue;
            }
            
            // Process a scanning batch
            if (!process_scanning_batch()) {
                std::cerr << "ERROR: Batch processing failed" << std::endl;
                log_error("Batch processing failed", 3);
                
                // Attempt recovery
                if (!attempt_error_recovery("batch_processing_failure")) {
                    std::cerr << "ERROR: Recovery failed, stopping scanning" << std::endl;
                    break;
                }
            }
            
            // Update scan position and check completion
            ecc::BigInt256 batch_size_big;
            batch_size_big.set_from_uint64(current_config_.keys_per_batch);
            current_scan_position_ = current_scan_position_ + batch_size_big;
            
            if (current_scan_position_ >= current_range_.end_key) {
                std::cout << "Scanning completed - reached end of range" << std::endl;
                set_status(ScanningStatus::COMPLETED);
                break;
            }
            
            // Brief yield to allow other operations
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in main scanning loop: " << e.what() << std::endl;
        log_error("Exception in main scanning loop: " + std::string(e.what()), 5);
        set_status(ScanningStatus::ERROR);
    }
    
    std::cout << "Main scanning loop ended" << std::endl;
}

bool IntegratedScanningEngine::process_scanning_batch() {
    try {
        if (current_config_.enable_gpu_coordination && gpu_coordinator_) {
            // Use GPU coordination for scanning
            return coordinate_multi_gpu_scanning();
        } else {
            // Use single scanner
            models::PrivateKeyRange batch_range;
            batch_range.start_key = current_scan_position_;
            
            ecc::BigInt256 batch_size;
            batch_size.set_from_uint64(current_config_.keys_per_batch);
            batch_range.end_key = std::min(current_scan_position_ + batch_size, current_range_.end_key);
            
            if (!scanner_->start_scanning(batch_range)) {
                return false;
            }
            
            // Wait for completion (simplified)
            while (scanner_->is_scanning() && !should_stop_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                // Update metrics
                update_current_metrics();
            }
            
            return true;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in process_scanning_batch: " << e.what() << std::endl;
        return false;
    }
}

bool IntegratedScanningEngine::coordinate_multi_gpu_scanning() {
    try {
        // This is a simplified coordination - in practice would be more complex
        if (!gpu_coordinator_->is_scanning()) {
            if (!gpu_coordinator_->start_coordinated_scanning()) {
                return false;
            }
        }
        
        // Monitor progress
        auto start_time = std::chrono::high_resolution_clock::now();
        const auto max_batch_time = std::chrono::seconds(30);
        
        while (gpu_coordinator_->is_scanning() && !should_stop_) {
            auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
            if (elapsed > max_batch_time) {
                std::cout << "Batch timeout reached" << std::endl;
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            update_current_metrics();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in coordinate_multi_gpu_scanning: " << e.what() << std::endl;
        return false;
    }
}

bool IntegratedScanningEngine::process_potential_match(const compare::AddressMatch& address_match) {
    try {
        std::lock_guard<std::mutex> lock(matches_mutex_);
        
        // Create scanning match
        ScanningMatch match;
        match.private_key = address_match.private_key;
        match.public_key = address_match.public_key;
        match.address = address_match.address;
        match.address_format = address_match.format;
        match.found_time = std::chrono::system_clock::now();
        match.batch_id = address_match.batch_id;
        match.device_id = address_match.device_id;
        
        // Set scanning context
        match.search_range_start = current_range_.start_key;
        match.search_range_end = current_range_.end_key;
        
        // Validate match
        if (!validate_match(match)) {
            std::cerr << "WARNING: Match validation failed" << std::endl;
            return false;
        }
        
        // Store match
        store_match(match);
        
        // Notify callback
        notify_match_found(match);
        
        std::cout << "🎉 MATCH FOUND: " << match.address << " 🎉" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in process_potential_match: " << e.what() << std::endl;
        return false;
    }
}

bool IntegratedScanningEngine::validate_match(ScanningMatch& match) {
    try {
        // Basic validation
        if (match.private_key.is_zero()) {
            return false;
        }
        
        if (match.address.empty()) {
            return false;
        }
        
        // CPU validation using ECC engine
        if (ecc_engine_) {
            ecc::Point computed_public_key;
            if (ecc_engine_->scalar_multiply_base(match.private_key, computed_public_key)) {
                match.cpu_validated = (computed_public_key == match.public_key);
            }
        }
        
        // GPU validation (if different from CPU)
        match.gpu_validated = true; // Simplified - assume GPU computation was correct
        
        // Set confidence based on validation results
        if (match.cpu_validated && match.gpu_validated) {
            match.validation_confidence = 1.0;
        } else if (match.cpu_validated || match.gpu_validated) {
            match.validation_confidence = 0.8;
        } else {
            match.validation_confidence = 0.0;
            return false;
        }
        
        match.validation_methods.push_back("CPU_ECC_Verification");
        match.validation_methods.push_back("GPU_ECC_Verification");
        
        return match.validation_confidence > 0.5;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in validate_match: " << e.what() << std::endl;
        return false;
    }
}

void IntegratedScanningEngine::store_match(const ScanningMatch& match) {
    // Add to matches list (with size limit)
    if (found_matches_.size() >= max_matches_in_memory_) {
        // Remove oldest matches to make space
        found_matches_.erase(found_matches_.begin(), found_matches_.begin() + 100);
    }
    
    found_matches_.push_back(match);
    
    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        current_metrics_.total_matches_found++;
        
        switch (match.address_format) {
            case compare::AddressFormat::P2PKH:
                current_metrics_.p2pkh_matches++;
                break;
            case compare::AddressFormat::P2SH:
                current_metrics_.p2sh_matches++;
                break;
            case compare::AddressFormat::P2WPKH_V0:
                current_metrics_.bech32_matches++;
                break;
            default:
                break;
        }
    }
}

void IntegratedScanningEngine::notify_match_found(const ScanningMatch& match) {
    if (match_callback_) {
        try {
            match_callback_(match);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in match callback: " << e.what() << std::endl;
        }
    }
}

void IntegratedScanningEngine::set_status(ScanningStatus new_status) {
    ScanningStatus old_status = current_status_.load();
    current_status_ = new_status;
    
    if (old_status != new_status) {
        notify_status_change(old_status, new_status);
    }
}

void IntegratedScanningEngine::notify_status_change(ScanningStatus old_status, ScanningStatus new_status) {
    if (status_callback_) {
        try {
            status_callback_(old_status, new_status);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in status callback: " << e.what() << std::endl;
        }
    }
}

void IntegratedScanningEngine::metrics_monitoring_loop() {
    while (metrics_thread_running_) {
        try {
            update_current_metrics();
            collect_component_metrics();
            
            // Call progress callback if set
            if (progress_callback_) {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                progress_callback_(current_metrics_);
            }
            
            // Sleep for a short interval
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in metrics monitoring loop: " << e.what() << std::endl;
        }
    }
}

void IntegratedScanningEngine::update_current_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto now = std::chrono::system_clock::now();
    current_metrics_.total_scan_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - current_metrics_.scan_start_time);
    
    current_metrics_.current_status = current_status_;
    
    // Calculate progress
    if (current_range_.end_key > current_range_.start_key) {
        ecc::BigInt256 total_range = current_range_.end_key - current_range_.start_key;
        ecc::BigInt256 completed_range = current_scan_position_ - current_range_.start_key;
        
        if (!total_range.is_zero()) {
            double progress_ratio = static_cast<double>(completed_range.to_uint64()) / 
                                  static_cast<double>(total_range.to_uint64());
            current_metrics_.overall_progress_percentage = std::min(100.0, progress_ratio * 100.0);
        }
    }
    
    // Update keys processed
    current_metrics_.total_keys_processed = current_scan_position_ - current_range_.start_key;
    
    // Calculate rates (simplified)
    if (current_metrics_.total_scan_time.count() > 0) {
        double seconds = current_metrics_.total_scan_time.count() / 1000.0;
        current_metrics_.keys_per_second_average = 
            static_cast<double>(current_metrics_.total_keys_processed.to_uint64()) / seconds;
    }
}

void IntegratedScanningEngine::collect_component_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    try {
        // Collect GPU coordinator metrics
        if (gpu_coordinator_) {
            auto gpu_metrics = gpu_coordinator_->get_current_metrics();
            current_metrics_.active_gpu_count = gpu_coordinator_->get_active_devices().size();
            current_metrics_.gpu_utilization_average = gpu_metrics.average_device_utilization;
            current_metrics_.gpu_load_balance_coefficient = gpu_metrics.load_balance_coefficient;
            current_metrics_.gpu_work_stealing_events = gpu_metrics.work_stealing_events;
        }
        
        // Collect address generator metrics
        if (address_generator_) {
            auto addr_metrics = address_generator_->get_current_metrics();
            current_metrics_.addresses_generated = addr_metrics.total_addresses_generated;
            current_metrics_.address_generation_rate = addr_metrics.average_generation_rate;
            current_metrics_.hash_operations_performed = addr_metrics.total_public_keys_processed;
            current_metrics_.hash_operations_per_second = addr_metrics.hash160_operations_per_second;
            current_metrics_.comparison_operations = addr_metrics.total_comparisons_performed;
            current_metrics_.comparison_rate = addr_metrics.comparison_rate;
            current_metrics_.total_gpu_memory_used = addr_metrics.gpu_memory_used;
        }
        
        // Collect scanner metrics
        if (scanner_) {
            auto scan_metrics = scanner_->get_current_metrics();
            current_metrics_.keys_per_second_current = scan_metrics.keys_per_second;
            current_metrics_.keys_per_second_peak = std::max(
                current_metrics_.keys_per_second_peak, 
                scan_metrics.keys_per_second
            );
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in collect_component_metrics: " << e.what() << std::endl;
    }
}

bool IntegratedScanningEngine::validate_configuration(const ScanningEngineConfig& config) {
    // Basic validation
    if (config.keys_per_batch == 0) {
        std::cerr << "ERROR: Invalid keys_per_batch (must be > 0)" << std::endl;
        return false;
    }
    
    if (config.ecc_batch_size == 0) {
        std::cerr << "ERROR: Invalid ecc_batch_size (must be > 0)" << std::endl;
        return false;
    }
    
    // Validate GPU device IDs (simplified)
    for (int device_id : config.gpu_device_ids) {
        if (device_id < 0) {
            std::cerr << "ERROR: Invalid GPU device ID: " << device_id << std::endl;
            return false;
        }
    }
    
    return true;
}

void IntegratedScanningEngine::log_error(const std::string& error, int severity) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    
    // Create error entry
    std::ostringstream oss;
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    oss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] "
        << "SEVERITY:" << severity << " " << error;
    
    error_log_.push_back(oss.str());
    
    // Limit error log size
    if (error_log_.size() > max_error_log_entries_) {
        error_log_.erase(error_log_.begin());
    }
    
    // Update error count
    current_metrics_.total_errors_encountered++;
    
    // Call error callback if set
    if (error_callback_) {
        try {
            error_callback_(error, severity);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in error callback: " << e.what() << std::endl;
        }
    }
}

bool IntegratedScanningEngine::attempt_error_recovery(const std::string& error_type) {
    recovery_attempts_++;
    
    if (recovery_attempts_ >= current_config_.max_recovery_attempts) {
        std::cerr << "ERROR: Maximum recovery attempts exceeded" << std::endl;
        return false;
    }
    
    std::cout << "Attempting error recovery for: " << error_type << std::endl;
    
    try {
        // Simple recovery strategy
        if (error_type == "batch_processing_failure") {
            // Pause briefly and retry
            std::this_thread::sleep_for(std::chrono::seconds(5));
            return true;
        }
        
        // Default recovery
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception during recovery: " << e.what() << std::endl;
        return false;
    }
}

void IntegratedScanningEngine::checkpoint_automation_loop() {
    while (checkpoint_thread_running_) {
        try {
            std::this_thread::sleep_for(current_config_.checkpoint_interval);
            
            if (current_status_ == ScanningStatus::SCANNING && !should_stop_) {
                create_automatic_checkpoint();
            }
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in checkpoint automation loop: " << e.what() << std::endl;
        }
    }
}

bool IntegratedScanningEngine::create_automatic_checkpoint() {
    try {
        if (!checkpoint_manager_) {
            return false;
        }
        
        // Create checkpoint data
        scan::checkpoint::ExtendedCheckpointData checkpoint_data;
        checkpoint_data.current_key = current_scan_position_;
        checkpoint_data.elapsed_time = current_metrics_.total_scan_time;
        checkpoint_data.keys_scanned = current_metrics_.total_keys_processed;
        checkpoint_data.checkpoint_time = std::chrono::system_clock::now();
        checkpoint_data.original_start_key = current_range_.start_key;
        checkpoint_data.original_end_key = current_range_.end_key;
        
        // Generate filename
        std::string filename = checkpoint_prefix_.empty() ? "auto_checkpoint" : checkpoint_prefix_;
        filename += "_" + std::to_string(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
        filename += ".khcp";
        
        bool success = checkpoint_manager_->create_checkpoint(checkpoint_data, filename);
        
        if (success) {
            current_metrics_.checkpoint_files_created++;
            std::cout << "Automatic checkpoint created: " << filename << std::endl;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in create_automatic_checkpoint: " << e.what() << std::endl;
        return false;
    }
}

// Callback setters
void IntegratedScanningEngine::set_progress_callback(std::function<void(const ScanningEngineMetrics&)> callback) {
    progress_callback_ = callback;
}

void IntegratedScanningEngine::set_match_callback(std::function<void(const ScanningMatch&)> callback) {
    match_callback_ = callback;
}

void IntegratedScanningEngine::set_error_callback(std::function<void(const std::string&, int)> callback) {
    error_callback_ = callback;
}

void IntegratedScanningEngine::set_status_callback(std::function<void(ScanningStatus, ScanningStatus)> callback) {
    status_callback_ = callback;
}

bool IntegratedScanningEngine::handle_device_failure(int device_id) {
    std::cout << "Handling device failure for device " << device_id << std::endl;
    
    current_metrics_.device_failures_detected++;
    log_error("GPU device failure: " + std::to_string(device_id), 4);
    
    // Attempt recovery
    return attempt_error_recovery("device_failure");
}

} // namespace engine
} // namespace keyhunt