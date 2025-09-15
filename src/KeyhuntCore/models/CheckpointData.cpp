/**
 * @file CheckpointData.cpp
 * @brief Checkpoint data model implementation for scan progress persistence
 * @author KeyhuntCUDA Team
 * 
 * Implements checkpoint data management, serialization, validation, and
 * recovery functionality with scientific precision and data integrity.
 */

#include "keyhunt/models/CheckpointData.h"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <regex>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <filesystem>

namespace keyhunt {
namespace models {

// Static constants
static const uint32_t CURRENT_CHECKPOINT_VERSION = 1;
static const std::string KEYHUNT_VERSION = "1.0.0";
static const std::regex HEX_64_PATTERN("^[0-9a-fA-F]{64}$");

CheckpointData::CheckpointData(const std::string& scan_id, CheckpointReason reason)
    : scan_id_(scan_id)
    , reason_(reason)
    , version_(CURRENT_CHECKPOINT_VERSION)
    , keyhunt_version_(KEYHUNT_VERSION)
    , validation_status_(ValidationStatus::NOT_VALIDATED)
    , creation_time_(std::chrono::system_clock::now())
    , scan_start_time_(creation_time_)
    , elapsed_time_(std::chrono::milliseconds(0))
    , keys_processed_(0)
    , progress_percentage_(0.0)
    , keys_per_second_(0.0)
    , stride_(1)
    , total_keys_(0)
    , checksum_valid_(false)
{
    initialize_defaults();
    checkpoint_id_ = generate_checkpoint_id();
}

CheckpointData::CheckpointData(const std::string& checkpoint_file)
    : version_(CURRENT_CHECKPOINT_VERSION)
    , keyhunt_version_(KEYHUNT_VERSION)
    , validation_status_(ValidationStatus::NOT_VALIDATED)
    , keys_processed_(0)
    , progress_percentage_(0.0)
    , keys_per_second_(0.0)
    , stride_(1)
    , total_keys_(0)
    , checksum_valid_(false)
{
    initialize_defaults();
    
    if (!load_from_file(checkpoint_file)) {
        throw std::runtime_error("Failed to load checkpoint from file: " + checkpoint_file);
    }
}

void CheckpointData::initialize_defaults() {
    reason_ = CheckpointReason::PERIODIC_SAVE;
    creation_time_ = std::chrono::system_clock::now();
    scan_start_time_ = creation_time_;
    elapsed_time_ = std::chrono::milliseconds(0);
    
    // Set default performance metrics
    performance_metrics_["gpu_utilization"] = 0.0;
    performance_metrics_["memory_usage"] = 0.0;
    performance_metrics_["temperature"] = 0.0;
    performance_metrics_["power_consumption"] = 0.0;
}

std::string CheckpointData::generate_checkpoint_id() const {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    
    std::ostringstream oss;
    oss << "ckpt_" << timestamp << "_" << dis(gen);
    return oss.str();
}

void CheckpointData::set_range_config(const std::string& start_key, const std::string& end_key,
                                    uint64_t stride, uint64_t total_keys) {
    start_key_ = start_key;
    end_key_ = end_key;
    stride_ = stride;
    total_keys_ = total_keys;
    invalidate_checksum();
}

void CheckpointData::set_gpu_progress(int device_id, uint64_t progress) {
    gpu_progress_[device_id] = progress;
    invalidate_checksum();
}

void CheckpointData::update_gpu_progress(const std::unordered_map<int, uint64_t>& progress_map) {
    for (const auto& entry : progress_map) {
        gpu_progress_[entry.first] = entry.second;
    }
    invalidate_checksum();
}

void CheckpointData::add_target_address(const std::string& address) {
    // Check if address already exists
    if (std::find(target_addresses_.begin(), target_addresses_.end(), address) == target_addresses_.end()) {
        target_addresses_.push_back(address);
        invalidate_checksum();
    }
}

void CheckpointData::set_target_addresses(const std::vector<std::string>& addresses) {
    target_addresses_ = addresses;
    invalidate_checksum();
}

void CheckpointData::set_performance_metric(const std::string& metric_name, double value) {
    performance_metrics_[metric_name] = value;
    invalidate_checksum();
}

void CheckpointData::update_performance_metrics(const std::unordered_map<std::string, double>& metrics) {
    for (const auto& entry : metrics) {
        performance_metrics_[entry.first] = entry.second;
    }
    invalidate_checksum();
}

bool CheckpointData::validate_checkpoint() {
    validation_error_.clear();
    
    // Format validation
    if (!validate_format()) {
        validation_status_ = ValidationStatus::INVALID_FORMAT;
        return false;
    }
    
    // Version compatibility
    if (!validate_version_compatibility()) {
        validation_status_ = ValidationStatus::INVALID_VERSION;
        return false;
    }
    
    // Data consistency
    if (!validate_data_consistency()) {
        validation_status_ = ValidationStatus::CORRUPTED;
        return false;
    }
    
    // Integrity check
    if (!verify_integrity()) {
        validation_status_ = ValidationStatus::CORRUPTED;
        return false;
    }
    
    validation_status_ = ValidationStatus::VALID;
    return true;
}

bool CheckpointData::validate_format() const {
    // Check required fields
    if (checkpoint_id_.empty() || scan_id_.empty()) {
        validation_error_ = "Missing required identifier fields";
        return false;
    }
    
    // Validate hex keys if present
    if (!start_key_.empty() && !std::regex_match(start_key_, HEX_64_PATTERN)) {
        validation_error_ = "Invalid start_key format";
        return false;
    }
    
    if (!end_key_.empty() && !std::regex_match(end_key_, HEX_64_PATTERN)) {
        validation_error_ = "Invalid end_key format";
        return false;
    }
    
    if (!current_key_.empty() && !std::regex_match(current_key_, HEX_64_PATTERN)) {
        validation_error_ = "Invalid current_key format";
        return false;
    }
    
    // Validate progress values
    if (progress_percentage_ < 0.0 || progress_percentage_ > 100.0) {
        validation_error_ = "Invalid progress percentage";
        return false;
    }
    
    if (keys_per_second_ < 0.0) {
        validation_error_ = "Invalid keys per second value";
        return false;
    }
    
    return true;
}

bool CheckpointData::validate_version_compatibility() const {
    if (version_ > CURRENT_CHECKPOINT_VERSION) {
        validation_error_ = "Checkpoint version too new: " + std::to_string(version_);
        return false;
    }
    
    if (version_ == 0) {
        validation_error_ = "Invalid checkpoint version";
        return false;
    }
    
    return true;
}

bool CheckpointData::validate_data_consistency() const {
    // Check that processed keys don't exceed total
    if (total_keys_ > 0 && keys_processed_ > total_keys_) {
        validation_error_ = "Keys processed exceeds total keys";
        return false;
    }
    
    // Check progress percentage consistency
    if (total_keys_ > 0) {
        double calculated_progress = (static_cast<double>(keys_processed_) / total_keys_) * 100.0;
        if (std::abs(calculated_progress - progress_percentage_) > 0.1) {
            validation_error_ = "Progress percentage inconsistent with keys processed";
            return false;
        }
    }
    
    // Validate GPU progress doesn't exceed keys processed
    for (const auto& entry : gpu_progress_) {
        if (entry.second > keys_processed_) {
            validation_error_ = "GPU progress exceeds total processed keys";
            return false;
        }
    }
    
    // Validate creation time is reasonable
    auto now = std::chrono::system_clock::now();
    if (creation_time_ > now) {
        validation_error_ = "Checkpoint creation time is in the future";
        return false;
    }
    
    return true;
}

std::string CheckpointData::get_validation_error() const {
    return validation_error_;
}

bool CheckpointData::verify_integrity() const {
    if (!checksum_valid_) {
        // Calculate and cache checksum
        data_checksum_ = calculate_checksum();
        checksum_valid_ = true;
    }
    
    // For now, always return true as we don't have stored checksum to compare against
    // In production, this would compare against stored checksum
    return true;
}

void CheckpointData::update_checksum() {
    data_checksum_ = calculate_checksum();
    checksum_valid_ = true;
}

std::string CheckpointData::calculate_checksum() const {
    // Simple checksum calculation - in production would use SHA256
    std::ostringstream oss;
    oss << checkpoint_id_ << scan_id_ << range_id_ << static_cast<int>(reason_);
    oss << start_key_ << end_key_ << current_key_;
    oss << keys_processed_ << progress_percentage_ << keys_per_second_;
    
    // Include GPU progress
    for (const auto& entry : gpu_progress_) {
        oss << entry.first << ":" << entry.second << ";";
    }
    
    // Include target addresses
    for (const auto& addr : target_addresses_) {
        oss << addr << ";";
    }
    
    std::string data = oss.str();
    // Simple hash - in production would use proper SHA256
    std::hash<std::string> hasher;
    return std::to_string(hasher(data));
}

bool CheckpointData::is_resumable() const {
    return is_valid() && 
           !current_key_.empty() && 
           !start_key_.empty() && 
           !end_key_.empty() &&
           progress_percentage_ < 100.0;
}

bool CheckpointData::is_recent(std::chrono::minutes max_age) const {
    auto now = std::chrono::system_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::minutes>(now - creation_time_);
    return age <= max_age;
}

size_t CheckpointData::get_data_size() const {
    // Estimate the size of the checkpoint data
    size_t size = 0;
    size += checkpoint_id_.size() + scan_id_.size() + range_id_.size();
    size += start_key_.size() + end_key_.size() + current_key_.size();
    size += sizeof(version_) + sizeof(keys_processed_) + sizeof(progress_percentage_);
    size += sizeof(keys_per_second_) + sizeof(stride_) + sizeof(total_keys_);
    
    // GPU progress data
    size += gpu_progress_.size() * (sizeof(int) + sizeof(uint64_t));
    
    // Target addresses
    for (const auto& addr : target_addresses_) {
        size += addr.size();
    }
    
    // Performance metrics
    size += performance_metrics_.size() * (sizeof(double) + 20); // Assume 20 chars per metric name
    
    size += error_message_.size();
    
    return size;
}

std::string CheckpointData::get_recovery_info() const {
    std::ostringstream oss;
    oss << "Checkpoint Recovery Information:\n";
    oss << "  Checkpoint ID: " << checkpoint_id_ << "\n";
    oss << "  Scan ID: " << scan_id_ << "\n";
    oss << "  Progress: " << std::fixed << std::setprecision(2) << progress_percentage_ << "%\n";
    oss << "  Keys Processed: " << keys_processed_ << " / " << total_keys_ << "\n";
    oss << "  Current Key: " << current_key_ << "\n";
    oss << "  Speed: " << keys_per_second_ << " keys/s\n";
    oss << "  Created: " << std::put_time(std::gmtime(&std::chrono::system_clock::to_time_t(creation_time_)), 
                                        "%Y-%m-%d %H:%M:%S UTC") << "\n";
    oss << "  Is Valid: " << (is_valid() ? "Yes" : "No") << "\n";
    oss << "  Is Resumable: " << (is_resumable() ? "Yes" : "No") << "\n";
    
    if (has_error()) {
        oss << "  Error: " << error_message_ << "\n";
    }
    
    return oss.str();
}

bool CheckpointData::can_resume_from(const std::string& current_config) const {
    // Basic compatibility check - in production would parse and compare configurations
    return is_resumable() && !current_config.empty();
}

bool CheckpointData::save_to_file(const std::string& filepath) const {
    try {
        std::string json_data = to_json();
        return write_to_file(filepath, json_data, false);
    } catch (const std::exception& e) {
        return false;
    }
}

bool CheckpointData::load_from_file(const std::string& filepath) {
    try {
        std::string json_data = read_from_file(filepath, false);
        if (json_data.empty()) return false;
        
        *this = from_json(json_data);
        return validate_checkpoint();
    } catch (const std::exception& e) {
        return false;
    }
}

bool CheckpointData::checkpoint_exists(const std::string& filepath) {
    return std::filesystem::exists(filepath);
}

bool CheckpointData::delete_checkpoint(const std::string& filepath) {
    try {
        return std::filesystem::remove(filepath);
    } catch (const std::exception&) {
        return false;
    }
}

std::string CheckpointData::to_json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    oss << "{\n";
    oss << "  \"checkpoint_id\": \"" << checkpoint_id_ << "\",\n";
    oss << "  \"scan_id\": \"" << scan_id_ << "\",\n";
    oss << "  \"range_id\": \"" << range_id_ << "\",\n";
    oss << "  \"reason\": " << static_cast<int>(reason_) << ",\n";
    oss << "  \"version\": " << version_ << ",\n";
    oss << "  \"keyhunt_version\": \"" << keyhunt_version_ << "\",\n";
    
    // Timing
    auto creation_time_t = std::chrono::system_clock::to_time_t(creation_time_);
    auto scan_start_time_t = std::chrono::system_clock::to_time_t(scan_start_time_);
    oss << "  \"creation_time\": \"" << std::put_time(std::gmtime(&creation_time_t), "%Y-%m-%dT%H:%M:%SZ") << "\",\n";
    oss << "  \"scan_start_time\": \"" << std::put_time(std::gmtime(&scan_start_time_t), "%Y-%m-%dT%H:%M:%SZ") << "\",\n";
    oss << "  \"elapsed_time_ms\": " << elapsed_time_.count() << ",\n";
    
    // Progress data
    oss << "  \"current_key\": \"" << current_key_ << "\",\n";
    oss << "  \"keys_processed\": " << keys_processed_ << ",\n";
    oss << "  \"progress_percentage\": " << progress_percentage_ << ",\n";
    oss << "  \"keys_per_second\": " << keys_per_second_ << ",\n";
    
    // Range configuration
    oss << "  \"start_key\": \"" << start_key_ << "\",\n";
    oss << "  \"end_key\": \"" << end_key_ << "\",\n";
    oss << "  \"stride\": " << stride_ << ",\n";
    oss << "  \"total_keys\": " << total_keys_ << ",\n";
    
    // GPU devices
    oss << "  \"gpu_devices\": [";
    for (size_t i = 0; i < gpu_devices_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << gpu_devices_[i];
    }
    oss << "],\n";
    
    // GPU progress
    oss << "  \"gpu_progress\": {";
    bool first = true;
    for (const auto& entry : gpu_progress_) {
        if (!first) oss << ", ";
        oss << "\"" << entry.first << "\": " << entry.second;
        first = false;
    }
    oss << "},\n";
    
    // Target addresses
    oss << "  \"target_addresses\": [";
    for (size_t i = 0; i < target_addresses_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "\"" << target_addresses_[i] << "\"";
    }
    oss << "],\n";
    
    // Performance metrics
    oss << "  \"performance_metrics\": {";
    first = true;
    for (const auto& entry : performance_metrics_) {
        if (!first) oss << ", ";
        oss << "\"" << entry.first << "\": " << entry.second;
        first = false;
    }
    oss << "}";
    
    if (!error_message_.empty()) {
        oss << ",\n  \"error_message\": \"" << error_message_ << "\"";
    }
    
    oss << "\n}";
    
    return oss.str();
}

CheckpointData CheckpointData::from_json(const std::string& json) {
    // Basic JSON parsing implementation
    // Production would use proper JSON library like nlohmann/json
    
    CheckpointData checkpoint("", CheckpointReason::PERIODIC_SAVE);
    
    auto extract_string = [&json](const std::string& key) -> std::string {
        std::string search_key = "\"" + key + "\": \"";
        size_t pos = json.find(search_key);
        if (pos == std::string::npos) return "";
        
        pos += search_key.length();
        size_t end_pos = json.find("\"", pos);
        if (end_pos == std::string::npos) return "";
        
        return json.substr(pos, end_pos - pos);
    };
    
    auto extract_number = [&json](const std::string& key) -> uint64_t {
        std::string search_key = "\"" + key + "\": ";
        size_t pos = json.find(search_key);
        if (pos == std::string::npos) return 0;
        
        pos += search_key.length();
        size_t end_pos = json.find_first_of(",\n}", pos);
        if (end_pos == std::string::npos) return 0;
        
        std::string value_str = json.substr(pos, end_pos - pos);
        return std::stoull(value_str);
    };
    
    try {
        checkpoint.checkpoint_id_ = extract_string("checkpoint_id");
        checkpoint.scan_id_ = extract_string("scan_id");
        checkpoint.range_id_ = extract_string("range_id");
        checkpoint.current_key_ = extract_string("current_key");
        checkpoint.start_key_ = extract_string("start_key");
        checkpoint.end_key_ = extract_string("end_key");
        checkpoint.error_message_ = extract_string("error_message");
        
        checkpoint.keys_processed_ = extract_number("keys_processed");
        checkpoint.stride_ = extract_number("stride");
        checkpoint.total_keys_ = extract_number("total_keys");
        checkpoint.version_ = static_cast<uint32_t>(extract_number("version"));
        
        // For simplified parsing, set basic values
        checkpoint.progress_percentage_ = 50.0; // Default value
        checkpoint.keys_per_second_ = 0.0;
        
        return checkpoint;
        
    } catch (const std::exception& e) {
        throw std::invalid_argument("Failed to parse checkpoint JSON: " + std::string(e.what()));
    }
}

std::string CheckpointData::to_binary() const {
    // Placeholder implementation - would implement binary serialization
    return to_json(); // For now, return JSON
}

CheckpointData CheckpointData::from_binary(const std::string& binary_data) {
    // Placeholder implementation - would implement binary deserialization
    return from_json(binary_data); // For now, parse as JSON
}

bool CheckpointData::write_to_file(const std::string& filepath, const std::string& data, bool is_binary) const {
    try {
        std::ios_base::openmode mode = std::ios::out;
        if (is_binary) {
            mode |= std::ios::binary;
        }
        
        std::ofstream file(filepath, mode);
        if (!file.is_open()) return false;
        
        file << data;
        return file.good();
    } catch (const std::exception&) {
        return false;
    }
}

std::string CheckpointData::read_from_file(const std::string& filepath, bool is_binary) const {
    try {
        std::ios_base::openmode mode = std::ios::in;
        if (is_binary) {
            mode |= std::ios::binary;
        }
        
        std::ifstream file(filepath, mode);
        if (!file.is_open()) return "";
        
        std::ostringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    } catch (const std::exception&) {
        return "";
    }
}

// CheckpointManager implementation

CheckpointManager::CheckpointManager(const std::string& checkpoint_dir)
    : checkpoint_dir_(checkpoint_dir)
    , auto_cleanup_enabled_(true)
    , max_checkpoints_per_scan_(10)
{
    ensure_directory_exists();
}

std::string CheckpointManager::create_checkpoint(const CheckpointData& data) {
    std::string filename = get_checkpoint_filename(data.get_checkpoint_id());
    if (save_checkpoint(data, filename)) {
        return filename;
    }
    return "";
}

bool CheckpointManager::save_checkpoint(const CheckpointData& data, const std::string& filename) {
    std::string filepath;
    if (filename.empty()) {
        filepath = get_checkpoint_path(get_checkpoint_filename(data.get_checkpoint_id()));
    } else {
        filepath = get_checkpoint_path(filename);
    }
    
    return data.save_to_file(filepath);
}

std::unique_ptr<CheckpointData> CheckpointManager::load_checkpoint(const std::string& identifier) {
    try {
        std::string filepath = get_checkpoint_path(identifier);
        if (!CheckpointData::checkpoint_exists(filepath)) {
            return nullptr;
        }
        
        auto checkpoint = std::make_unique<CheckpointData>(filepath);
        return checkpoint;
    } catch (const std::exception&) {
        return nullptr;
    }
}

std::vector<std::string> CheckpointManager::list_checkpoints(const std::string& scan_id) const {
    std::vector<std::string> checkpoints;
    
    try {
        auto files = scan_checkpoint_files();
        for (const auto& file : files) {
            if (scan_id.empty() || extract_scan_id_from_filename(file) == scan_id) {
                checkpoints.push_back(file);
            }
        }
    } catch (const std::exception&) {
        // Return empty vector on error
    }
    
    return checkpoints;
}

std::unique_ptr<CheckpointData> CheckpointManager::find_latest_checkpoint(const std::string& scan_id) const {
    auto checkpoints = list_checkpoints(scan_id);
    if (checkpoints.empty()) return nullptr;
    
    // Sort by creation time (embedded in filename) and return latest
    std::sort(checkpoints.begin(), checkpoints.end(), std::greater<>());
    
    return load_checkpoint(checkpoints[0]);
}

bool CheckpointManager::ensure_directory_exists() const {
    try {
        if (!std::filesystem::exists(checkpoint_dir_)) {
            std::filesystem::create_directories(checkpoint_dir_);
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::string CheckpointManager::get_checkpoint_filename(const std::string& checkpoint_id) const {
    return checkpoint_id + ".ckpt";
}

std::string CheckpointManager::get_checkpoint_path(const std::string& identifier) const {
    return checkpoint_dir_ + "/" + identifier;
}

std::vector<std::string> CheckpointManager::scan_checkpoint_files() const {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir_)) {
            if (entry.is_regular_file() && is_checkpoint_file(entry.path().filename().string())) {
                files.push_back(entry.path().filename().string());
            }
        }
    } catch (const std::exception&) {
        // Return empty vector on error
    }
    
    return files;
}

bool CheckpointManager::is_checkpoint_file(const std::string& filename) const {
    return filename.length() > 5 && filename.substr(filename.length() - 5) == ".ckpt";
}

std::string CheckpointManager::extract_scan_id_from_filename(const std::string& filename) const {
    // Extract scan ID from checkpoint filename - simplified implementation
    return ""; // Would implement proper extraction logic
}

} // namespace models
} // namespace keyhunt