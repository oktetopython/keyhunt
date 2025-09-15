/**
 * @file ExperimentalResults.cpp
 * @brief Scientific experimental results implementation for performance analysis
 * @author KeyhuntCUDA Team
 * 
 * Implements scientific experimental results collection, statistical analysis,
 * and performance reporting with scientific precision requirements.
 */

#include "keyhunt/models/ExperimentalResults.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace keyhunt {
namespace models {

ExperimentalResults::ExperimentalResults(const std::string& experiment_id, MeasurementType measurement_type)
    : experiment_id_(experiment_id.empty() ? generate_experiment_id() : experiment_id)
    , measurement_type_(measurement_type)
    , status_(ResultStatus::SUCCESS)
    , confidence_level_(ConfidenceLevel::STANDARD_95)
    , accuracy_percentage_(0.0)
{
    initialize_defaults();
}

void ExperimentalResults::initialize_defaults() {
    start_time_ = std::chrono::system_clock::now();
    end_time_ = start_time_;
    
    // Set default system information
    system_info_ = "Unknown System";
    cuda_version_ = "Unknown CUDA Version";
    
    // Initialize default metrics
    metrics_["keys_per_second"] = 0.0;
    metrics_["gpu_utilization"] = 0.0;
    metrics_["memory_usage"] = 0.0;
    metrics_["power_consumption"] = 0.0;
    
    // Initialize default parameters
    numeric_parameters_["sample_size"] = 1000.0;
    numeric_parameters_["precision_threshold"] = 1e-10;
    parameters_["test_mode"] = "standard";
}

std::string ExperimentalResults::generate_experiment_id() const {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(100, 999);
    
    std::ostringstream oss;
    oss << "exp_" << timestamp << "_" << dis(gen);
    return oss.str();
}

std::chrono::milliseconds ExperimentalResults::get_duration() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_);
}

void ExperimentalResults::add_gpu_model(const std::string& model) {
    if (std::find(gpu_models_.begin(), gpu_models_.end(), model) == gpu_models_.end()) {
        gpu_models_.push_back(model);
    }
}

void ExperimentalResults::start_experiment() {
    start_time_ = std::chrono::system_clock::now();
    status_ = ResultStatus::SUCCESS;
    clear_errors();
    clear_warnings();
}

void ExperimentalResults::end_experiment() {
    end_time_ = std::chrono::system_clock::now();
    calculate_statistics();
}

void ExperimentalResults::interrupt_experiment() {
    end_time_ = std::chrono::system_clock::now();
    status_ = ResultStatus::INTERRUPTED;
    add_warning("Experiment was interrupted before completion");
}

void ExperimentalResults::set_parameter(const std::string& name, const std::string& value) {
    parameters_[name] = value;
}

void ExperimentalResults::set_numeric_parameter(const std::string& name, double value) {
    numeric_parameters_[name] = value;
}

void ExperimentalResults::record_metric(const std::string& name, double value) {
    metrics_[name] = value;
    
    // Also add to time series for tracking
    time_series_[name].push_back(value);
}

void ExperimentalResults::record_time_series_point(const std::string& series_name, double value) {
    time_series_[series_name].push_back(value);
}

void ExperimentalResults::record_time_series(const std::string& series_name, const std::vector<double>& values) {
    time_series_[series_name] = values;
}

void ExperimentalResults::update_metrics(const std::unordered_map<std::string, double>& metrics) {
    for (const auto& entry : metrics) {
        record_metric(entry.first, entry.second);
    }
}

void ExperimentalResults::record_validation_result(const std::string& test_name, bool passed) {
    validation_results_[test_name] = passed;
}

void ExperimentalResults::calculate_statistics() {
    // Calculate statistics for all time series
    for (const auto& series : time_series_) {
        calculate_metric_statistics(series.first);
    }
    
    // Calculate overall accuracy based on validation results
    if (!validation_results_.empty()) {
        size_t passed_count = 0;
        for (const auto& result : validation_results_) {
            if (result.second) passed_count++;
        }
        accuracy_percentage_ = (static_cast<double>(passed_count) / validation_results_.size()) * 100.0;
    }
    
    // Calculate experiment duration statistics
    auto duration_ms = get_duration().count();
    statistics_["duration_ms"] = static_cast<double>(duration_ms);
    statistics_["duration_seconds"] = duration_ms / 1000.0;
    statistics_["duration_minutes"] = duration_ms / 60000.0;
}

void ExperimentalResults::calculate_metric_statistics(const std::string& metric_name) {
    auto it = time_series_.find(metric_name);
    if (it == time_series_.end() || it->second.empty()) return;
    
    const auto& values = it->second;
    
    // Calculate basic statistics
    double mean = calculate_mean(values);
    double stddev = calculate_stddev(values, mean);
    
    std::string prefix = metric_name + "_";
    statistics_[prefix + "mean"] = mean;
    statistics_[prefix + "stddev"] = stddev;
    statistics_[prefix + "min"] = *std::min_element(values.begin(), values.end());
    statistics_[prefix + "max"] = *std::max_element(values.begin(), values.end());
    statistics_[prefix + "count"] = static_cast<double>(values.size());
    
    // Calculate percentiles
    std::vector<double> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    statistics_[prefix + "median"] = calculate_percentile(sorted_values, 50.0);
    statistics_[prefix + "p25"] = calculate_percentile(sorted_values, 25.0);
    statistics_[prefix + "p75"] = calculate_percentile(sorted_values, 75.0);
    statistics_[prefix + "p95"] = calculate_percentile(sorted_values, 95.0);
    statistics_[prefix + "p99"] = calculate_percentile(sorted_values, 99.0);
    
    // Calculate confidence intervals
    double lower, upper;
    calculate_confidence_interval(values, lower, upper);
    statistics_[prefix + "ci_lower"] = lower;
    statistics_[prefix + "ci_upper"] = upper;
}

double ExperimentalResults::get_metric_average(const std::string& metric_name) const {
    std::string key = metric_name + "_mean";
    auto it = statistics_.find(key);
    return it != statistics_.end() ? it->second : 0.0;
}

double ExperimentalResults::get_metric_stddev(const std::string& metric_name) const {
    std::string key = metric_name + "_stddev";
    auto it = statistics_.find(key);
    return it != statistics_.end() ? it->second : 0.0;
}

double ExperimentalResults::get_metric_min(const std::string& metric_name) const {
    std::string key = metric_name + "_min";
    auto it = statistics_.find(key);
    return it != statistics_.end() ? it->second : 0.0;
}

double ExperimentalResults::get_metric_max(const std::string& metric_name) const {
    std::string key = metric_name + "_max";
    auto it = statistics_.find(key);
    return it != statistics_.end() ? it->second : 0.0;
}

void ExperimentalResults::add_error(const std::string& error_message) {
    errors_.push_back(error_message);
    if (status_ == ResultStatus::SUCCESS) {
        status_ = ResultStatus::FAILED;
    }
}

void ExperimentalResults::add_warning(const std::string& warning_message) {
    warnings_.push_back(warning_message);
}

bool ExperimentalResults::compare_with(const ExperimentalResults& other, double tolerance) const {
    // Compare key metrics within tolerance
    for (const auto& metric : metrics_) {
        auto other_it = other.metrics_.find(metric.first);
        if (other_it == other.metrics_.end()) continue;
        
        double diff = std::abs(metric.second - other_it->second);
        double relative_diff = diff / std::max(std::abs(metric.second), std::abs(other_it->second));
        
        if (relative_diff > tolerance) {
            return false;
        }
    }
    
    return true;
}

double ExperimentalResults::calculate_improvement_percentage(const ExperimentalResults& baseline) const {
    // Calculate improvement based on keys_per_second metric
    auto baseline_kps = baseline.metrics_.find("keys_per_second");
    auto current_kps = metrics_.find("keys_per_second");
    
    if (baseline_kps == baseline.metrics_.end() || current_kps == metrics_.end()) {
        return 0.0;
    }
    
    if (baseline_kps->second == 0.0) return 0.0;
    
    return ((current_kps->second - baseline_kps->second) / baseline_kps->second) * 100.0;
}

std::string ExperimentalResults::generate_performance_summary() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    oss << "Performance Summary for " << experiment_id_ << ":\n";
    oss << "  Type: " << get_measurement_type_string() << "\n";
    oss << "  Status: " << get_status_string() << "\n";
    oss << "  Duration: " << format_duration(get_duration()) << "\n";
    
    // Key performance metrics
    auto kps_it = metrics_.find("keys_per_second");
    if (kps_it != metrics_.end()) {
        oss << "  Performance: " << kps_it->second / 1000000.0 << " M keys/s\n";
    }
    
    auto gpu_it = metrics_.find("gpu_utilization");
    if (gpu_it != metrics_.end()) {
        oss << "  GPU Utilization: " << gpu_it->second << "%\n";
    }
    
    auto mem_it = metrics_.find("memory_usage");
    if (mem_it != metrics_.end()) {
        oss << "  Memory Usage: " << mem_it->second / 1024.0 << " GB\n";
    }
    
    if (accuracy_percentage_ > 0.0) {
        oss << "  Accuracy: " << accuracy_percentage_ << "%\n";
    }
    
    if (!errors_.empty()) {
        oss << "  Errors: " << errors_.size() << "\n";
    }
    
    if (!warnings_.empty()) {
        oss << "  Warnings: " << warnings_.size() << "\n";
    }
    
    return oss.str();
}

std::string ExperimentalResults::export_csv() const {
    std::ostringstream oss;
    
    // Header
    oss << "experiment_id,measurement_type,status,duration_ms,accuracy_percentage";
    for (const auto& metric : metrics_) {
        oss << "," << metric.first;
    }
    oss << "\n";
    
    // Data
    oss << experiment_id_ << "," << get_measurement_type_string() << "," << get_status_string();
    oss << "," << get_duration().count() << "," << accuracy_percentage_;
    
    for (const auto& metric : metrics_) {
        oss << "," << metric.second;
    }
    oss << "\n";
    
    return oss.str();
}

std::string ExperimentalResults::export_metrics_csv() const {
    std::ostringstream oss;
    
    // Header
    oss << "metric_name,value\n";
    
    // Data
    for (const auto& metric : metrics_) {
        oss << metric.first << "," << metric.second << "\n";
    }
    
    return oss.str();
}

std::string ExperimentalResults::export_time_series_csv() const {
    std::ostringstream oss;
    
    if (time_series_.empty()) return "";
    
    // Find maximum series length
    size_t max_length = 0;
    for (const auto& series : time_series_) {
        max_length = std::max(max_length, series.second.size());
    }
    
    // Header
    oss << "index";
    for (const auto& series : time_series_) {
        oss << "," << series.first;
    }
    oss << "\n";
    
    // Data
    for (size_t i = 0; i < max_length; ++i) {
        oss << i;
        for (const auto& series : time_series_) {
            if (i < series.second.size()) {
                oss << "," << series.second[i];
            } else {
                oss << ",";
            }
        }
        oss << "\n";
    }
    
    return oss.str();
}

std::string ExperimentalResults::to_json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    oss << "{\n";
    oss << "  \"experiment_id\": \"" << experiment_id_ << "\",\n";
    oss << "  \"description\": \"" << description_ << "\",\n";
    oss << "  \"measurement_type\": \"" << get_measurement_type_string() << "\",\n";
    oss << "  \"status\": \"" << get_status_string() << "\",\n";
    oss << "  \"confidence_level\": " << static_cast<int>(confidence_level_) << ",\n";
    
    // Timing
    auto start_time_t = std::chrono::system_clock::to_time_t(start_time_);
    auto end_time_t = std::chrono::system_clock::to_time_t(end_time_);
    oss << "  \"start_time\": \"" << std::put_time(std::gmtime(&start_time_t), "%Y-%m-%dT%H:%M:%SZ") << "\",\n";
    oss << "  \"end_time\": \"" << std::put_time(std::gmtime(&end_time_t), "%Y-%m-%dT%H:%M:%SZ") << "\",\n";
    oss << "  \"duration_ms\": " << get_duration().count() << ",\n";
    
    // System info
    oss << "  \"system_info\": \"" << system_info_ << "\",\n";
    oss << "  \"cuda_version\": \"" << cuda_version_ << "\",\n";
    oss << "  \"gpu_models\": [";
    for (size_t i = 0; i < gpu_models_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "\"" << gpu_models_[i] << "\"";
    }
    oss << "],\n";
    
    // Metrics
    oss << "  \"metrics\": {";
    bool first = true;
    for (const auto& metric : metrics_) {
        if (!first) oss << ", ";
        oss << "\"" << metric.first << "\": " << metric.second;
        first = false;
    }
    oss << "},\n";
    
    // Statistics
    oss << "  \"statistics\": {";
    first = true;
    for (const auto& stat : statistics_) {
        if (!first) oss << ", ";
        oss << "\"" << stat.first << "\": " << stat.second;
        first = false;
    }
    oss << "},\n";
    
    // Validation results
    oss << "  \"accuracy_percentage\": " << accuracy_percentage_ << ",\n";
    oss << "  \"validation_results\": {";
    first = true;
    for (const auto& result : validation_results_) {
        if (!first) oss << ", ";
        oss << "\"" << result.first << "\": " << (result.second ? "true" : "false");
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

std::string ExperimentalResults::to_detailed_report() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    
    oss << "========================================\n";
    oss << "EXPERIMENTAL RESULTS DETAILED REPORT\n";
    oss << "========================================\n\n";
    
    oss << "Experiment ID: " << experiment_id_ << "\n";
    oss << "Description: " << description_ << "\n";
    oss << "Measurement Type: " << get_measurement_type_string() << "\n";
    oss << "Status: " << get_status_string() << "\n";
    oss << "Duration: " << format_duration(get_duration()) << "\n";
    oss << "Confidence Level: " << static_cast<int>(confidence_level_) << "%\n\n";
    
    // System Information
    oss << "SYSTEM CONFIGURATION:\n";
    oss << "  System Info: " << system_info_ << "\n";
    oss << "  CUDA Version: " << cuda_version_ << "\n";
    oss << "  GPU Models: ";
    for (size_t i = 0; i < gpu_models_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << gpu_models_[i];
    }
    oss << "\n\n";
    
    // Performance Metrics
    oss << "PERFORMANCE METRICS:\n";
    for (const auto& metric : metrics_) {
        oss << "  " << metric.first << ": " << metric.second;
        if (metric.first == "keys_per_second") {
            oss << " (" << metric.second / 1000000.0 << " M keys/s)";
        } else if (metric.first.find("percentage") != std::string::npos || 
                   metric.first.find("utilization") != std::string::npos) {
            oss << "%";
        }
        oss << "\n";
    }
    oss << "\n";
    
    // Statistical Analysis
    if (!statistics_.empty()) {
        oss << "STATISTICAL ANALYSIS:\n";
        for (const auto& stat : statistics_) {
            oss << "  " << stat.first << ": " << stat.second << "\n";
        }
        oss << "\n";
    }
    
    // Validation Results
    if (!validation_results_.empty()) {
        oss << "VALIDATION RESULTS:\n";
        oss << "  Overall Accuracy: " << accuracy_percentage_ << "%\n";
        for (const auto& result : validation_results_) {
            oss << "  " << result.first << ": " << (result.second ? "PASS" : "FAIL") << "\n";
        }
        oss << "\n";
    }
    
    // Errors and Warnings
    if (!errors_.empty()) {
        oss << "ERRORS:\n";
        for (size_t i = 0; i < errors_.size(); ++i) {
            oss << "  " << (i + 1) << ". " << errors_[i] << "\n";
        }
        oss << "\n";
    }
    
    if (!warnings_.empty()) {
        oss << "WARNINGS:\n";
        for (size_t i = 0; i < warnings_.size(); ++i) {
            oss << "  " << (i + 1) << ". " << warnings_[i] << "\n";
        }
        oss << "\n";
    }
    
    oss << "========================================\n";
    
    return oss.str();
}

ExperimentalResults ExperimentalResults::from_json(const std::string& json) {
    // Basic JSON parsing - production would use proper JSON library
    ExperimentalResults results("", MeasurementType::PERFORMANCE_BENCHMARK);
    
    auto extract_string = [&json](const std::string& key) -> std::string {
        std::string search_key = "\"" + key + "\": \"";
        size_t pos = json.find(search_key);
        if (pos == std::string::npos) return "";
        
        pos += search_key.length();
        size_t end_pos = json.find("\"", pos);
        if (end_pos == std::string::npos) return "";
        
        return json.substr(pos, end_pos - pos);
    };
    
    auto extract_number = [&json](const std::string& key) -> double {
        std::string search_key = "\"" + key + "\": ";
        size_t pos = json.find(search_key);
        if (pos == std::string::npos) return 0.0;
        
        pos += search_key.length();
        size_t end_pos = json.find_first_of(",\n}", pos);
        if (end_pos == std::string::npos) return 0.0;
        
        std::string value_str = json.substr(pos, end_pos - pos);
        return std::stod(value_str);
    };
    
    try {
        results.experiment_id_ = extract_string("experiment_id");
        results.description_ = extract_string("description");
        results.system_info_ = extract_string("system_info");
        results.cuda_version_ = extract_string("cuda_version");
        
        results.accuracy_percentage_ = extract_number("accuracy_percentage");
        
        // Set default values for other fields
        results.status_ = ResultStatus::SUCCESS;
        results.confidence_level_ = ConfidenceLevel::STANDARD_95;
        
        return results;
        
    } catch (const std::exception& e) {
        throw std::invalid_argument("Failed to parse experimental results JSON: " + std::string(e.what()));
    }
}

bool ExperimentalResults::validate_results() const {
    // Check if experiment has sufficient data
    if (!has_sufficient_data()) return false;
    
    // Check if critical metrics are within reasonable ranges
    auto kps_it = metrics_.find("keys_per_second");
    if (kps_it != metrics_.end() && kps_it->second < 0.0) return false;
    
    auto gpu_it = metrics_.find("gpu_utilization");
    if (gpu_it != metrics_.end() && (gpu_it->second < 0.0 || gpu_it->second > 100.0)) return false;
    
    // Check accuracy percentage
    if (accuracy_percentage_ < 0.0 || accuracy_percentage_ > 100.0) return false;
    
    return status_ != ResultStatus::FAILED;
}

std::string ExperimentalResults::get_validation_summary() const {
    std::ostringstream oss;
    
    oss << "Validation Summary:\n";
    oss << "  Data Sufficiency: " << (has_sufficient_data() ? "PASS" : "FAIL") << "\n";
    oss << "  Metric Ranges: " << (validate_results() ? "PASS" : "FAIL") << "\n";
    oss << "  Overall Status: " << get_status_string() << "\n";
    
    if (!validation_results_.empty()) {
        size_t passed = 0;
        for (const auto& result : validation_results_) {
            if (result.second) passed++;
        }
        oss << "  Test Results: " << passed << "/" << validation_results_.size() << " passed\n";
    }
    
    return oss.str();
}

bool ExperimentalResults::has_sufficient_data() const {
    return !metrics_.empty() && get_duration().count() > 0;
}

size_t ExperimentalResults::get_sample_count() const {
    if (time_series_.empty()) return 0;
    
    size_t max_count = 0;
    for (const auto& series : time_series_) {
        max_count = std::max(max_count, series.second.size());
    }
    return max_count;
}

// Helper method implementations

std::string ExperimentalResults::format_duration(std::chrono::milliseconds duration) const {
    auto ms = duration.count();
    if (ms < 1000) {
        return std::to_string(ms) + "ms";
    } else if (ms < 60000) {
        return std::to_string(ms / 1000.0) + "s";
    } else {
        return std::to_string(ms / 60000.0) + "min";
    }
}

std::string ExperimentalResults::get_measurement_type_string() const {
    switch (measurement_type_) {
        case MeasurementType::PERFORMANCE_BENCHMARK: return "performance_benchmark";
        case MeasurementType::SCIENTIFIC_VALIDATION: return "scientific_validation";
        case MeasurementType::STRESS_TEST: return "stress_test";
        case MeasurementType::ACCURACY_TEST: return "accuracy_test";
        case MeasurementType::SCALABILITY_TEST: return "scalability_test";
        case MeasurementType::MEMORY_ANALYSIS: return "memory_analysis";
        case MeasurementType::POWER_ANALYSIS: return "power_analysis";
        default: return "unknown";
    }
}

std::string ExperimentalResults::get_status_string() const {
    switch (status_) {
        case ResultStatus::SUCCESS: return "success";
        case ResultStatus::PARTIAL_SUCCESS: return "partial_success";
        case ResultStatus::FAILED: return "failed";
        case ResultStatus::INTERRUPTED: return "interrupted";
        case ResultStatus::INVALID_PARAMETERS: return "invalid_parameters";
        case ResultStatus::RESOURCE_ERROR: return "resource_error";
        default: return "unknown";
    }
}

double ExperimentalResults::calculate_mean(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double ExperimentalResults::calculate_stddev(const std::vector<double>& values, double mean) const {
    if (values.size() <= 1) return 0.0;
    
    double variance = 0.0;
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= (values.size() - 1);
    
    return std::sqrt(variance);
}

double ExperimentalResults::calculate_percentile(const std::vector<double>& values, double percentile) const {
    if (values.empty()) return 0.0;
    
    double index = (percentile / 100.0) * (values.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));
    
    if (lower == upper || upper >= values.size()) {
        return values[lower];
    }
    
    double weight = index - lower;
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

void ExperimentalResults::calculate_confidence_interval(const std::vector<double>& values, 
                                                       double& lower, double& upper) const {
    if (values.empty()) {
        lower = upper = 0.0;
        return;
    }
    
    double mean = calculate_mean(values);
    double stddev = calculate_stddev(values, mean);
    
    // Use t-distribution critical value for 95% confidence
    double t_critical = 1.96; // Approximation for large samples
    double margin = t_critical * (stddev / std::sqrt(values.size()));
    
    lower = mean - margin;
    upper = mean + margin;
}

// ExperimentalResultsSet implementation

ExperimentalResultsSet::ExperimentalResultsSet(const std::string& set_name)
    : set_name_(set_name.empty() ? "DefaultResultSet" : set_name)
{
}

void ExperimentalResultsSet::add_result(const ExperimentalResults& result) {
    results_.push_back(std::make_unique<ExperimentalResults>(result));
}

void ExperimentalResultsSet::add_result(std::unique_ptr<ExperimentalResults> result) {
    if (result) {
        results_.push_back(std::move(result));
    }
}

void ExperimentalResultsSet::remove_result(const std::string& experiment_id) {
    results_.erase(
        std::remove_if(results_.begin(), results_.end(),
            [&experiment_id](const std::unique_ptr<ExperimentalResults>& result) {
                return result && result->get_experiment_id() == experiment_id;
            }),
        results_.end());
}

ExperimentalResults* ExperimentalResultsSet::find_result(const std::string& experiment_id) {
    for (auto& result : results_) {
        if (result && result->get_experiment_id() == experiment_id) {
            return result.get();
        }
    }
    return nullptr;
}

std::vector<ExperimentalResults*> ExperimentalResultsSet::find_by_type(ExperimentalResults::MeasurementType type) {
    std::vector<ExperimentalResults*> matching_results;
    
    for (auto& result : results_) {
        if (result && result->get_measurement_type() == type) {
            matching_results.push_back(result.get());
        }
    }
    
    return matching_results;
}

std::vector<ExperimentalResults*> ExperimentalResultsSet::find_successful_results() {
    std::vector<ExperimentalResults*> successful_results;
    
    for (auto& result : results_) {
        if (result && result->is_successful()) {
            successful_results.push_back(result.get());
        }
    }
    
    return successful_results;
}

std::string ExperimentalResultsSet::generate_comparative_report() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    
    oss << "COMPARATIVE EXPERIMENTAL RESULTS REPORT\n";
    oss << "Set Name: " << set_name_ << "\n";
    oss << "Total Results: " << results_.size() << "\n\n";
    
    if (results_.empty()) {
        oss << "No results to compare.\n";
        return oss.str();
    }
    
    // Performance comparison
    auto best_performance = find_best_performing("keys_per_second");
    if (best_performance) {
        auto kps_it = best_performance->get_metrics().find("keys_per_second");
        if (kps_it != best_performance->get_metrics().end()) {
            oss << "Best Performance: " << best_performance->get_experiment_id();
            oss << " (" << kps_it->second / 1000000.0 << " M keys/s)\n\n";
        }
    }
    
    // Summary statistics
    auto avg_kps = calculate_set_average("keys_per_second");
    auto stddev_kps = calculate_set_stddev("keys_per_second");
    
    oss << "Average Performance: " << avg_kps / 1000000.0 << " M keys/s\n";
    oss << "Performance Std Dev: " << stddev_kps / 1000000.0 << " M keys/s\n\n";
    
    return oss.str();
}

double ExperimentalResultsSet::calculate_set_average(const std::string& metric_name) const {
    auto values = extract_metric_values(metric_name);
    if (values.empty()) return 0.0;
    
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double ExperimentalResultsSet::calculate_set_stddev(const std::string& metric_name) const {
    auto values = extract_metric_values(metric_name);
    if (values.size() <= 1) return 0.0;
    
    double mean = calculate_set_average(metric_name);
    double variance = 0.0;
    
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    
    variance /= (values.size() - 1);
    return std::sqrt(variance);
}

std::vector<double> ExperimentalResultsSet::extract_metric_values(const std::string& metric_name) const {
    std::vector<double> values;
    
    for (const auto& result : results_) {
        if (!result) continue;
        
        auto it = result->get_metrics().find(metric_name);
        if (it != result->get_metrics().end()) {
            values.push_back(it->second);
        }
    }
    
    return values;
}

ExperimentalResults* ExperimentalResultsSet::find_best_performing(const std::string& metric_name) const {
    ExperimentalResults* best = nullptr;
    double best_value = -std::numeric_limits<double>::infinity();
    
    for (const auto& result : results_) {
        if (!result) continue;
        
        auto it = result->get_metrics().find(metric_name);
        if (it != result->get_metrics().end() && it->second > best_value) {
            best_value = it->second;
            best = result.get();
        }
    }
    
    return best;
}

std::string ExperimentalResultsSet::to_json() const {
    std::ostringstream oss;
    
    oss << "{\n";
    oss << "  \"set_name\": \"" << set_name_ << "\",\n";
    oss << "  \"result_count\": " << results_.size() << ",\n";
    oss << "  \"results\": [\n";
    
    for (size_t i = 0; i < results_.size(); ++i) {
        if (i > 0) oss << ",\n";
        if (results_[i]) {
            oss << "    " << results_[i]->to_json();
        }
    }
    
    oss << "\n  ]\n";
    oss << "}";
    
    return oss.str();
}

} // namespace models
} // namespace keyhunt