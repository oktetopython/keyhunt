/**
 * @file PrivateKeyRange.cpp
 * @brief Private Key Range data model implementation for Bitcoin key scanning
 * @author KeyhuntCUDA Team
 * 
 * Implements the core functionality for private key range management,
 * validation, and progress tracking with scientific precision requirements.
 */

#include "keyhunt/models/PrivateKeyRange.h"
#include <sstream>
#include <iomanip>
#include <regex>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cassert>

namespace keyhunt {
namespace models {

// Static regex for hex validation
static const std::regex HEX_64_PATTERN("^[0-9a-fA-F]{64}$");

PrivateKeyRange::PrivateKeyRange(const std::string& start_key, 
                               const std::string& end_key, 
                               uint64_t stride)
    : start_key_(start_key)
    , end_key_(end_key)
    , stride_(stride)
    , total_keys_(0)
    , status_(Status::CONFIGURED)
    , estimated_time_(0.0)
    , keys_processed_(0)
    , start_time_(std::chrono::steady_clock::now())
    , last_update_(start_time_)
{
    // Validate input parameters
    validate_hex_key(start_key_);
    validate_hex_key(end_key_);
    
    if (stride == 0) {
        throw std::invalid_argument("Stride must be greater than 0");
    }
    
    // Generate unique range ID
    range_id_ = generate_range_id();
    
    // Calculate total keys in range
    calculate_total_keys();
}

void PrivateKeyRange::validate_hex_key(const std::string& key) const {
    if (!std::regex_match(key, HEX_64_PATTERN)) {
        throw std::invalid_argument("Invalid hex key format: must be 64 hexadecimal characters");
    }
}

std::string PrivateKeyRange::generate_range_id() {
    // Generate unique range ID using timestamp and random component
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    
    std::ostringstream oss;
    oss << "range_" << timestamp << "_" << dis(gen);
    return oss.str();
}

void PrivateKeyRange::calculate_total_keys() {
    // Convert hex strings to uint64_t for calculation
    // This is a simplified calculation - actual implementation would use 256-bit arithmetic
    try {
        // Extract last 16 hex chars (64 bits) for simplified calculation
        std::string start_suffix = start_key_.substr(48, 16);
        std::string end_suffix = end_key_.substr(48, 16);
        
        uint64_t start_val = std::stoull(start_suffix, nullptr, 16);
        uint64_t end_val = std::stoull(end_suffix, nullptr, 16);
        
        if (end_val <= start_val) {
            throw std::invalid_argument("End key must be greater than start key");
        }
        
        total_keys_ = (end_val - start_val + 1) / stride_;
        
        // Estimate time based on performance targets (1000M keys/s for Turing)
        const double KEYS_PER_SECOND = 1000000000.0; // 1000M keys/s
        estimated_time_ = static_cast<double>(total_keys_) / KEYS_PER_SECOND;
        
    } catch (const std::exception& e) {
        throw std::invalid_argument("Failed to calculate range: " + std::string(e.what()));
    }
}

double PrivateKeyRange::get_progress_percentage() const {
    if (total_keys_ == 0) return 0.0;
    return (static_cast<double>(keys_processed_) / static_cast<double>(total_keys_)) * 100.0;
}

std::chrono::milliseconds PrivateKeyRange::get_elapsed_time() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
}

double PrivateKeyRange::get_keys_per_second() const {
    auto elapsed = get_elapsed_time();
    if (elapsed.count() == 0) return 0.0;
    
    double elapsed_seconds = elapsed.count() / 1000.0;
    return static_cast<double>(keys_processed_) / elapsed_seconds;
}

void PrivateKeyRange::set_keys_processed(uint64_t processed) {
    keys_processed_ = processed;
    last_update_ = std::chrono::steady_clock::now();
}

bool PrivateKeyRange::is_valid() const {
    try {
        // Check hex format
        validate_hex_key(start_key_);
        validate_hex_key(end_key_);
        
        // Check stride
        if (stride_ == 0) return false;
        
        // Check range logic
        std::string start_suffix = start_key_.substr(48, 16);
        std::string end_suffix = end_key_.substr(48, 16);
        
        uint64_t start_val = std::stoull(start_suffix, nullptr, 16);
        uint64_t end_val = std::stoull(end_suffix, nullptr, 16);
        
        return end_val > start_val;
        
    } catch (const std::exception&) {
        return false;
    }
}

std::string PrivateKeyRange::get_validation_error() const {
    if (is_valid()) return "";
    
    std::ostringstream oss;
    
    // Check hex format
    if (!std::regex_match(start_key_, HEX_64_PATTERN)) {
        oss << "Invalid start_key format (must be 64 hex characters); ";
    }
    if (!std::regex_match(end_key_, HEX_64_PATTERN)) {
        oss << "Invalid end_key format (must be 64 hex characters); ";
    }
    
    // Check stride
    if (stride_ == 0) {
        oss << "Invalid stride (must be > 0); ";
    }
    
    // Check range logic
    try {
        std::string start_suffix = start_key_.substr(48, 16);
        std::string end_suffix = end_key_.substr(48, 16);
        
        uint64_t start_val = std::stoull(start_suffix, nullptr, 16);
        uint64_t end_val = std::stoull(end_suffix, nullptr, 16);
        
        if (end_val <= start_val) {
            oss << "Invalid range (end_key must be > start_key); ";
        }
    } catch (const std::exception&) {
        oss << "Failed to parse key values; ";
    }
    
    std::string result = oss.str();
    if (!result.empty() && result.back() == ' ') {
        result.pop_back(); // Remove trailing space
        result.pop_back(); // Remove trailing semicolon
    }
    
    return result;
}

std::string PrivateKeyRange::to_json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    oss << "{\n";
    oss << "  \"range_id\": \"" << range_id_ << "\",\n";
    oss << "  \"start_key\": \"" << start_key_ << "\",\n";
    oss << "  \"end_key\": \"" << end_key_ << "\",\n";
    oss << "  \"stride\": " << stride_ << ",\n";
    oss << "  \"total_keys\": " << total_keys_ << ",\n";
    oss << "  \"status\": \"";
    
    // Convert status enum to string
    switch (status_) {
        case Status::CONFIGURED: oss << "configured"; break;
        case Status::SCANNING: oss << "scanning"; break;
        case Status::PAUSED: oss << "paused"; break;
        case Status::COMPLETED: oss << "completed"; break;
        case Status::FAILED: oss << "failed"; break;
        case Status::CANCELLED: oss << "cancelled"; break;
    }
    
    oss << "\",\n";
    oss << "  \"estimated_time\": " << estimated_time_ << ",\n";
    oss << "  \"keys_processed\": " << keys_processed_ << ",\n";
    oss << "  \"progress_percentage\": " << get_progress_percentage() << ",\n";
    oss << "  \"keys_per_second\": " << get_keys_per_second() << ",\n";
    oss << "  \"gpu_devices\": [";
    
    for (size_t i = 0; i < gpu_devices_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << gpu_devices_[i];
    }
    
    oss << "]\n";
    oss << "}";
    
    return oss.str();
}

PrivateKeyRange PrivateKeyRange::from_json(const std::string& json) {
    // Basic JSON parsing implementation
    // In production, would use a proper JSON library like nlohmann/json
    
    auto extract_string = [&json](const std::string& key) -> std::string {
        std::string search_key = "\"" + key + "\": \"";
        size_t pos = json.find(search_key);
        if (pos == std::string::npos) {
            throw std::invalid_argument("Missing key: " + key);
        }
        pos += search_key.length();
        size_t end_pos = json.find("\"", pos);
        if (end_pos == std::string::npos) {
            throw std::invalid_argument("Invalid JSON format for key: " + key);
        }
        return json.substr(pos, end_pos - pos);
    };
    
    auto extract_number = [&json](const std::string& key) -> uint64_t {
        std::string search_key = "\"" + key + "\": ";
        size_t pos = json.find(search_key);
        if (pos == std::string::npos) {
            throw std::invalid_argument("Missing key: " + key);
        }
        pos += search_key.length();
        size_t end_pos = json.find_first_of(",\n}", pos);
        if (end_pos == std::string::npos) {
            throw std::invalid_argument("Invalid JSON format for key: " + key);
        }
        std::string value_str = json.substr(pos, end_pos - pos);
        return std::stoull(value_str);
    };
    
    try {
        std::string start_key = extract_string("start_key");
        std::string end_key = extract_string("end_key");
        uint64_t stride = extract_number("stride");
        
        return PrivateKeyRange(start_key, end_key, stride);
        
    } catch (const std::exception& e) {
        throw std::invalid_argument("Failed to parse JSON: " + std::string(e.what()));
    }
}

} // namespace models
} // namespace keyhunt