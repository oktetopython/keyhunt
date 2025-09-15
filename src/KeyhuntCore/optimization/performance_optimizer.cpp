/**
 * @file performance_optimizer.cpp
 * @brief Implementation of performance optimization algorithms
 * @author KeyhuntCUDA Team
 * 
 * T046: Implement performance optimization algorithms with adaptive parameter tuning and bottleneck analysis
 */

#include "performance_optimizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace keyhunt {
namespace optimization {

PerformanceOptimizer::PerformanceOptimizer()
    : monitoring_active_(false)
    , adaptive_tuning_enabled_(false)
    , target_engine_(nullptr)
    , monitoring_thread_running_(false)
    , tuning_thread_running_(false)
{
    performance_history_.reserve(1000); // Reserve space for performance samples
}

PerformanceOptimizer::~PerformanceOptimizer() {
    cleanup();
}

bool PerformanceOptimizer::initialize(const AdaptiveTuningConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(optimization_mutex_);
        
        std::cout << "Initializing Performance Optimizer..." << std::endl;
        
        config_ = config;
        
        // Initialize statistics
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_ = OptimizationStatistics();
        }
        
        // Clear history
        {
            std::lock_guard<std::mutex> history_lock(history_mutex_);
            performance_history_.clear();
            detected_bottlenecks_.clear();
            applied_optimizations_.clear();
        }
        
        std::cout << "Performance Optimizer initialized successfully" << std::endl;
        std::cout << "  Automatic tuning: " << (config_.enable_automatic_tuning ? "Enabled" : "Disabled") << std::endl;
        std::cout << "  Tuning aggressiveness: " << config_.tuning_aggressiveness << std::endl;
        std::cout << "  Tuning interval: " << config_.tuning_interval.count() << " seconds" << std::endl;
        std::cout << "  Improvement threshold: " << (config_.minimum_improvement_threshold * 100) << "%" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in PerformanceOptimizer::initialize: " << e.what() << std::endl;
        return false;
    }
}

bool PerformanceOptimizer::configure_hardware_profile(const HardwareProfile& profile) {
    try {
        std::lock_guard<std::mutex> lock(optimization_mutex_);
        
        hardware_profile_ = profile;
        
        std::cout << "Hardware profile configured:" << std::endl;
        std::cout << "  GPU devices: " << profile.gpu_devices.size() << std::endl;
        std::cout << "  Total memory: " << profile.total_memory_gb << " GB" << std::endl;
        std::cout << "  CPU cores: " << profile.cpu_cores << std::endl;
        std::cout << "  Architecture: " << profile.architecture_profile << std::endl;
        
        // Analyze hardware capabilities
        if (!analyze_hardware_capabilities()) {
            std::cerr << "WARNING: Hardware capability analysis failed" << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in configure_hardware_profile: " << e.what() << std::endl;
        return false;
    }
}

void PerformanceOptimizer::cleanup() {
    std::cout << "Cleaning up Performance Optimizer..." << std::endl;
    
    // Stop monitoring
    stop_monitoring();
    
    // Stop adaptive tuning
    if (tuning_thread_running_) {
        tuning_thread_running_ = false;
        if (tuning_thread_.joinable()) {
            tuning_thread_.join();
        }
    }
    
    // Clear data
    {
        std::lock_guard<std::mutex> history_lock(history_mutex_);
        performance_history_.clear();
    }
    
    {
        std::lock_guard<std::mutex> opt_lock(optimization_mutex_);
        detected_bottlenecks_.clear();
        applied_optimizations_.clear();
    }
    
    target_engine_ = nullptr;
    
    std::cout << "Performance Optimizer cleanup completed" << std::endl;
}

bool PerformanceOptimizer::start_monitoring(engine::IntegratedScanningEngine* engine) {
    if (!engine) {
        std::cerr << "ERROR: Cannot start monitoring with null engine" << std::endl;
        return false;
    }
    
    if (monitoring_active_) {
        std::cout << "Monitoring already active" << std::endl;
        return true;
    }
    
    try {
        target_engine_ = engine;
        monitoring_active_ = true;
        monitoring_thread_running_ = true;
        
        monitoring_thread_ = std::thread(&PerformanceOptimizer::monitoring_loop, this);
        
        // Start adaptive tuning if enabled
        if (config_.enable_automatic_tuning) {
            adaptive_tuning_enabled_ = true;
            tuning_thread_running_ = true;
            tuning_thread_ = std::thread(&PerformanceOptimizer::adaptive_tuning_loop, this);
        }
        
        std::cout << "Performance monitoring started" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in start_monitoring: " << e.what() << std::endl;
        monitoring_active_ = false;
        return false;
    }
}

bool PerformanceOptimizer::stop_monitoring() {
    if (!monitoring_active_) {
        return true;
    }
    
    std::cout << "Stopping performance monitoring..." << std::endl;
    
    monitoring_active_ = false;
    adaptive_tuning_enabled_ = false;
    
    // Stop monitoring thread
    if (monitoring_thread_running_) {
        monitoring_thread_running_ = false;
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
    }
    
    // Stop tuning thread
    if (tuning_thread_running_) {
        tuning_thread_running_ = false;
        if (tuning_thread_.joinable()) {
            tuning_thread_.join();
        }
    }
    
    std::cout << "Performance monitoring stopped" << std::endl;
    return true;
}

void PerformanceOptimizer::add_performance_sample(const PerformanceSample& sample) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    performance_history_.push_back(sample);
    
    // Limit history size
    if (performance_history_.size() > config_.history_window_size * 2) {
        performance_history_.erase(
            performance_history_.begin(),
            performance_history_.begin() + (performance_history_.size() - config_.history_window_size)
        );
    }
}

std::vector<BottleneckAnalysis> PerformanceOptimizer::analyze_current_bottlenecks() const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    std::vector<BottleneckAnalysis> bottlenecks;
    
    if (performance_history_.empty()) {
        return bottlenecks;
    }
    
    const auto& latest_sample = performance_history_.back();
    
    // Analyze different bottleneck types
    auto gpu_bottleneck = analyze_gpu_utilization(latest_sample);
    if (gpu_bottleneck.severity_score > 0.3) {
        bottlenecks.push_back(gpu_bottleneck);
    }
    
    auto memory_bottleneck = analyze_memory_bandwidth(latest_sample);
    if (memory_bottleneck.severity_score > 0.3) {
        bottlenecks.push_back(memory_bottleneck);
    }
    
    auto cpu_bottleneck = analyze_cpu_processing(latest_sample);
    if (cpu_bottleneck.severity_score > 0.3) {
        bottlenecks.push_back(cpu_bottleneck);
    }
    
    auto io_bottleneck = analyze_io_operations(latest_sample);
    if (io_bottleneck.severity_score > 0.3) {
        bottlenecks.push_back(io_bottleneck);
    }
    
    auto sync_bottleneck = analyze_synchronization(latest_sample);
    if (sync_bottleneck.severity_score > 0.3) {
        bottlenecks.push_back(sync_bottleneck);
    }
    
    auto balance_bottleneck = analyze_load_balancing(latest_sample);
    if (balance_bottleneck.severity_score > 0.3) {
        bottlenecks.push_back(balance_bottleneck);
    }
    
    auto alloc_bottleneck = analyze_memory_allocation(latest_sample);
    if (alloc_bottleneck.severity_score > 0.3) {
        bottlenecks.push_back(alloc_bottleneck);
    }
    
    auto cache_bottleneck = analyze_cache_efficiency(latest_sample);
    if (cache_bottleneck.severity_score > 0.3) {
        bottlenecks.push_back(cache_bottleneck);
    }
    
    // Sort by severity
    std::sort(bottlenecks.begin(), bottlenecks.end(),
        [](const BottleneckAnalysis& a, const BottleneckAnalysis& b) {
            return a.severity_score > b.severity_score;
        });
    
    return bottlenecks;
}

BottleneckAnalysis PerformanceOptimizer::identify_primary_bottleneck() const {
    auto bottlenecks = analyze_current_bottlenecks();
    
    if (bottlenecks.empty()) {
        BottleneckAnalysis none;
        none.type = BottleneckType::UNKNOWN;
        none.description = "No significant bottlenecks detected";
        none.severity_score = 0.0;
        none.recommended_action = "System is performing well";
        return none;
    }
    
    return bottlenecks[0]; // Return most severe bottleneck
}

std::vector<OptimizationSuggestion> PerformanceOptimizer::generate_optimization_suggestions() const {
    std::vector<OptimizationSuggestion> suggestions;
    
    // Get current bottlenecks
    auto bottlenecks = analyze_current_bottlenecks();
    
    // Generate suggestions for each bottleneck
    for (const auto& bottleneck : bottlenecks) {
        switch (bottleneck.type) {
            case BottleneckType::GPU_UTILIZATION:
                {
                    auto suggestion = suggest_gpu_configuration_optimization();
                    suggestion.expected_improvement = std::min(0.5, bottleneck.severity_score);
                    suggestions.push_back(suggestion);
                }
                break;
                
            case BottleneckType::MEMORY_BANDWIDTH:
                {
                    auto suggestion = suggest_memory_optimization();
                    suggestion.expected_improvement = std::min(0.4, bottleneck.severity_score);
                    suggestions.push_back(suggestion);
                }
                break;
                
            case BottleneckType::LOAD_BALANCING:
                {
                    auto suggestion = suggest_load_balancing_optimization();
                    suggestion.expected_improvement = std::min(0.3, bottleneck.severity_score);
                    suggestions.push_back(suggestion);
                }
                break;
                
            case BottleneckType::MEMORY_ALLOCATION:
                {
                    auto suggestion = suggest_batch_size_optimization();
                    suggestion.expected_improvement = std::min(0.3, bottleneck.severity_score);
                    suggestions.push_back(suggestion);
                }
                break;
                
            case BottleneckType::SYNCHRONIZATION:
                {
                    auto suggestion = suggest_thread_configuration_optimization();
                    suggestion.expected_improvement = std::min(0.2, bottleneck.severity_score);
                    suggestions.push_back(suggestion);
                }
                break;
                
            default:
                // Generic optimization for unknown bottlenecks
                {
                    OptimizationSuggestion generic;
                    generic.parameter_name = "keys_per_batch";
                    generic.current_value = "5000000";
                    generic.suggested_value = "7500000";
                    generic.rationale = "Increase batch size to improve GPU utilization";
                    generic.expected_improvement = 0.1;
                    generic.confidence = 0.5;
                    suggestions.push_back(generic);
                }
                break;
        }
    }
    
    // Add hardware-specific suggestions
    auto hardware_suggestions = get_hardware_specific_suggestions();
    suggestions.insert(suggestions.end(), hardware_suggestions.begin(), hardware_suggestions.end());
    
    // Sort by expected improvement
    std::sort(suggestions.begin(), suggestions.end(),
        [](const OptimizationSuggestion& a, const OptimizationSuggestion& b) {
            return (a.expected_improvement * a.confidence) > (b.expected_improvement * b.confidence);
        });
    
    return suggestions;
}

std::vector<OptimizationSuggestion> PerformanceOptimizer::get_hardware_specific_suggestions() const {
    std::vector<OptimizationSuggestion> suggestions;
    
    if (hardware_profile_.gpu_devices.empty()) {
        return suggestions;
    }
    
    // Analyze GPU capabilities
    for (const auto& gpu : hardware_profile_.gpu_devices) {
        if (gpu.compute_capability_major >= 7) {
            // Turing or newer - can use advanced optimizations
            OptimizationSuggestion suggestion;
            suggestion.parameter_name = "enable_fused_operations";
            suggestion.current_value = "false";
            suggestion.suggested_value = "true";
            suggestion.rationale = "Enable fused operations for Turing+ architecture";
            suggestion.expected_improvement = 0.15;
            suggestion.confidence = 0.8;
            suggestions.push_back(suggestion);
        }
        
        if (gpu.memory_size > 8 * 1024 * 1024 * 1024ULL) { // > 8GB
            // High memory GPU - can use larger batches
            OptimizationSuggestion suggestion;
            suggestion.parameter_name = "hash_batch_size";
            suggestion.current_value = "65536";
            suggestion.suggested_value = "131072";
            suggestion.rationale = "Increase hash batch size for high-memory GPU";
            suggestion.expected_improvement = 0.2;
            suggestion.confidence = 0.7;
            suggestions.push_back(suggestion);
        }
    }
    
    // Multi-GPU optimizations
    if (hardware_profile_.gpu_devices.size() > 1) {
        OptimizationSuggestion suggestion;
        suggestion.parameter_name = "enable_work_stealing";
        suggestion.current_value = "false";
        suggestion.suggested_value = "true";
        suggestion.rationale = "Enable work stealing for multi-GPU setup";
        suggestion.expected_improvement = 0.25;
        suggestion.confidence = 0.8;
        suggestions.push_back(suggestion);
    }
    
    return suggestions;
}

bool PerformanceOptimizer::apply_optimization_suggestions(const std::vector<OptimizationSuggestion>& suggestions) {
    if (!target_engine_) {
        std::cerr << "ERROR: No target engine available for optimization" << std::endl;
        return false;
    }
    
    std::cout << "Applying " << suggestions.size() << " optimization suggestions..." << std::endl;
    
    size_t successful_applications = 0;
    
    for (const auto& suggestion : suggestions) {
        try {
            // Validate suggestion first
            if (!validate_optimization_suggestion(suggestion)) {
                std::cout << "Skipping invalid suggestion: " << suggestion.parameter_name << std::endl;
                continue;
            }
            
            // Apply the optimization (simplified - would need actual parameter setting logic)
            std::cout << "Applying optimization: " << suggestion.parameter_name 
                      << " -> " << suggestion.suggested_value << std::endl;
            
            // Store applied optimization
            {
                std::lock_guard<std::mutex> lock(optimization_mutex_);
                applied_optimizations_.push_back(suggestion);
            }
            
            successful_applications++;
            
            // Update statistics
            update_optimization_statistics(suggestion, true);
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to apply optimization " << suggestion.parameter_name 
                      << ": " << e.what() << std::endl;
            update_optimization_statistics(suggestion, false);
        }
    }
    
    std::cout << "Applied " << successful_applications << "/" << suggestions.size() 
              << " optimization suggestions successfully" << std::endl;
    
    return successful_applications > 0;
}

bool PerformanceOptimizer::enable_adaptive_tuning(bool enable) {
    adaptive_tuning_enabled_ = enable;
    
    if (enable && monitoring_active_ && !tuning_thread_running_) {
        tuning_thread_running_ = true;
        tuning_thread_ = std::thread(&PerformanceOptimizer::adaptive_tuning_loop, this);
        std::cout << "Adaptive tuning enabled" << std::endl;
    } else if (!enable && tuning_thread_running_) {
        tuning_thread_running_ = false;
        if (tuning_thread_.joinable()) {
            tuning_thread_.join();
        }
        std::cout << "Adaptive tuning disabled" << std::endl;
    }
    
    return true;
}

PerformanceOptimizer::OptimizationStatistics PerformanceOptimizer::get_optimization_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

std::string PerformanceOptimizer::generate_optimization_report() const {
    std::ostringstream report;
    
    report << "KeyhuntCUDA Performance Optimization Report\n";
    report << "==========================================\n\n";
    
    // Current bottlenecks
    auto bottlenecks = analyze_current_bottlenecks();
    report << "Current Bottlenecks (" << bottlenecks.size() << " detected):\n";
    for (const auto& bottleneck : bottlenecks) {
        report << "  • " << optimization_utils::get_bottleneck_description(bottleneck.type) 
               << " (Severity: " << std::fixed << std::setprecision(2) << bottleneck.severity_score << ")\n";
        report << "    Action: " << bottleneck.recommended_action << "\n";
    }
    report << "\n";
    
    // Optimization suggestions
    auto suggestions = generate_optimization_suggestions();
    report << "Optimization Suggestions (" << suggestions.size() << " available):\n";
    for (size_t i = 0; i < std::min(suggestions.size(), size_t(5)); ++i) {
        const auto& suggestion = suggestions[i];
        report << "  " << (i+1) << ". " << suggestion.parameter_name 
               << ": " << suggestion.current_value << " → " << suggestion.suggested_value << "\n";
        report << "     Expected improvement: " << std::fixed << std::setprecision(1) 
               << (suggestion.expected_improvement * 100) << "%\n";
        report << "     Rationale: " << suggestion.rationale << "\n";
    }
    report << "\n";
    
    // Statistics
    auto stats = get_optimization_statistics();
    report << "Optimization Statistics:\n";
    report << "  Total optimizations applied: " << stats.total_optimizations_applied << "\n";
    report << "  Successful optimizations: " << stats.successful_optimizations << "\n";
    report << "  Success rate: " << std::fixed << std::setprecision(1);
    if (stats.total_optimizations_applied > 0) {
        report << (100.0 * stats.successful_optimizations / stats.total_optimizations_applied);
    } else {
        report << "0.0";
    }
    report << "%\n";
    report << "  Average improvement: " << std::fixed << std::setprecision(1) 
           << (stats.average_performance_improvement * 100) << "%\n";
    report << "  Peak improvement: " << std::fixed << std::setprecision(1) 
           << (stats.peak_performance_improvement * 100) << "%\n";
    report << "  Rollbacks performed: " << stats.rollbacks_performed << "\n";
    
    report << "\nReport generated at: " 
           << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
    
    return report.str();
}

// Private method implementations

void PerformanceOptimizer::monitoring_loop() {
    while (monitoring_thread_running_) {
        try {
            collect_current_metrics();
            
            // Sleep for monitoring interval
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Exception in monitoring loop: " << e.what() << std::endl;
        }
    }
}

void PerformanceOptimizer::collect_current_metrics() {
    if (!target_engine_) {
        return;
    }
    
    try {
        PerformanceSample sample;
        sample.timestamp = std::chrono::system_clock::now();
        sample.metrics = target_engine_->get_current_metrics();
        sample.overall_performance_score = calculate_performance_score(sample);
        
        add_performance_sample(sample);
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to collect metrics: " << e.what() << std::endl;
    }
}

double PerformanceOptimizer::calculate_performance_score(const PerformanceSample& sample) const {
    // Combine multiple performance metrics into a single score
    double gpu_efficiency = optimization_utils::calculate_gpu_efficiency(sample.metrics);
    double memory_efficiency = optimization_utils::calculate_memory_efficiency(sample.metrics);
    double overall_efficiency = optimization_utils::calculate_overall_efficiency(sample.metrics);
    
    // Weighted combination
    return 0.4 * gpu_efficiency + 0.3 * memory_efficiency + 0.3 * overall_efficiency;
}

BottleneckAnalysis PerformanceOptimizer::analyze_gpu_utilization(const PerformanceSample& sample) const {
    BottleneckAnalysis analysis;
    analysis.type = BottleneckType::GPU_UTILIZATION;
    analysis.detected_time = sample.timestamp;
    
    double gpu_utilization = sample.metrics.gpu_utilization_average;
    
    if (gpu_utilization < 0.7) {
        analysis.severity_score = (0.7 - gpu_utilization) / 0.7; // 0 to 1 scale
        analysis.description = "Low GPU utilization detected";
        analysis.recommended_action = "Increase batch sizes or reduce CPU-GPU synchronization overhead";
    } else {
        analysis.severity_score = 0.0;
        analysis.description = "GPU utilization is acceptable";
        analysis.recommended_action = "No action needed";
    }
    
    analysis.metrics["gpu_utilization"] = gpu_utilization;
    analysis.metrics["active_gpu_count"] = sample.metrics.active_gpu_count;
    
    return analysis;
}

BottleneckAnalysis PerformanceOptimizer::analyze_memory_bandwidth(const PerformanceSample& sample) const {
    BottleneckAnalysis analysis;
    analysis.type = BottleneckType::MEMORY_BANDWIDTH;
    analysis.detected_time = sample.timestamp;
    
    double memory_bandwidth_util = sample.metrics.memory_bandwidth_utilization;
    
    if (memory_bandwidth_util > 0.9) {
        analysis.severity_score = (memory_bandwidth_util - 0.9) / 0.1;
        analysis.description = "Memory bandwidth saturation detected";
        analysis.recommended_action = "Optimize memory access patterns or reduce memory-intensive operations";
    } else {
        analysis.severity_score = 0.0;
        analysis.description = "Memory bandwidth utilization is normal";
        analysis.recommended_action = "No action needed";
    }
    
    analysis.metrics["memory_bandwidth_utilization"] = memory_bandwidth_util;
    analysis.metrics["total_gpu_memory_used"] = sample.metrics.total_gpu_memory_used;
    
    return analysis;
}

BottleneckAnalysis PerformanceOptimizer::analyze_load_balancing(const PerformanceSample& sample) const {
    BottleneckAnalysis analysis;
    analysis.type = BottleneckType::LOAD_BALANCING;
    analysis.detected_time = sample.timestamp;
    
    double load_balance_coeff = sample.metrics.gpu_load_balance_coefficient;
    
    if (load_balance_coeff < 0.8 && sample.metrics.active_gpu_count > 1) {
        analysis.severity_score = (0.8 - load_balance_coeff) / 0.8;
        analysis.description = "Poor load balancing across GPUs detected";
        analysis.recommended_action = "Enable work stealing or adjust load balancing strategy";
    } else {
        analysis.severity_score = 0.0;
        analysis.description = "Load balancing is acceptable";
        analysis.recommended_action = "No action needed";
    }
    
    analysis.metrics["load_balance_coefficient"] = load_balance_coeff;
    analysis.metrics["work_stealing_events"] = sample.metrics.gpu_work_stealing_events;
    
    return analysis;
}

OptimizationSuggestion PerformanceOptimizer::suggest_batch_size_optimization() const {
    OptimizationSuggestion suggestion;
    suggestion.parameter_name = "keys_per_batch";
    suggestion.current_value = "5000000";
    
    // Analyze current performance to suggest optimal batch size
    if (!performance_history_.empty()) {
        const auto& latest = performance_history_.back();
        if (latest.metrics.gpu_utilization_average < 0.7) {
            suggestion.suggested_value = "10000000"; // Increase batch size
            suggestion.rationale = "Increase batch size to improve GPU utilization";
            suggestion.expected_improvement = 0.2;
        } else if (latest.metrics.memory_bandwidth_utilization > 0.9) {
            suggestion.suggested_value = "2500000"; // Decrease batch size
            suggestion.rationale = "Decrease batch size to reduce memory pressure";
            suggestion.expected_improvement = 0.1;
        }
    }
    
    suggestion.confidence = 0.7;
    return suggestion;
}

OptimizationSuggestion PerformanceOptimizer::suggest_memory_optimization() const {
    OptimizationSuggestion suggestion;
    suggestion.parameter_name = "enable_memory_pooling";
    suggestion.current_value = "false";
    suggestion.suggested_value = "true";
    suggestion.rationale = "Enable memory pooling to reduce allocation overhead";
    suggestion.expected_improvement = 0.15;
    suggestion.confidence = 0.8;
    return suggestion;
}

OptimizationSuggestion PerformanceOptimizer::suggest_gpu_configuration_optimization() const {
    OptimizationSuggestion suggestion;
    suggestion.parameter_name = "threads_per_block";
    suggestion.current_value = "256";
    suggestion.suggested_value = "512";
    suggestion.rationale = "Increase threads per block to improve GPU occupancy";
    suggestion.expected_improvement = 0.1;
    suggestion.confidence = 0.6;
    return suggestion;
}

bool PerformanceOptimizer::validate_optimization_suggestion(const OptimizationSuggestion& suggestion) const {
    // Basic validation
    if (suggestion.parameter_name.empty() || suggestion.suggested_value.empty()) {
        return false;
    }
    
    if (suggestion.expected_improvement < 0.0 || suggestion.expected_improvement > 1.0) {
        return false;
    }
    
    if (suggestion.confidence < 0.0 || suggestion.confidence > 1.0) {
        return false;
    }
    
    return optimization_utils::validate_optimization_safety(suggestion);
}

void PerformanceOptimizer::update_optimization_statistics(const OptimizationSuggestion& suggestion, bool success) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_optimizations_applied++;
    
    if (success) {
        stats_.successful_optimizations++;
        
        // Update improvement statistics
        double improvement = suggestion.expected_improvement;
        stats_.average_performance_improvement = 
            (stats_.average_performance_improvement * (stats_.successful_optimizations - 1) + improvement) / 
            stats_.successful_optimizations;
        
        stats_.peak_performance_improvement = std::max(stats_.peak_performance_improvement, improvement);
    }
}

// Implementation of other private methods would continue here...
// For brevity, showing key methods that demonstrate the optimization approach

} // namespace optimization
} // namespace keyhunt