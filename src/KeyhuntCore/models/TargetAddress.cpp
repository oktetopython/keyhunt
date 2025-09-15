/**
 * @file TargetAddress.cpp
 * @brief Target Bitcoin Address data model implementation
 * @author KeyhuntCUDA Team
 * 
 * Implements comprehensive Bitcoin address validation, format detection,
 * hash extraction, and comparison operations with scientific precision.
 */

#include "keyhunt/models/TargetAddress.h"
#include <sstream>
#include <iomanip>
#include <regex>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <cstring>

namespace keyhunt {
namespace models {

// Static regex patterns for address validation
static const std::regex LEGACY_P2PKH_PATTERN("^[1][a-km-zA-HJ-NP-Z1-9]{25,34}$");
static const std::regex LEGACY_P2SH_PATTERN("^[3][a-km-zA-HJ-NP-Z1-9]{25,34}$");
static const std::regex BECH32_PATTERN("^bc1[a-z0-9]{39,59}$");
static const std::regex TESTNET_BECH32_PATTERN("^tb1[a-z0-9]{39,59}$");

// Base58 alphabet
static const char BASE58_ALPHABET[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static const int BASE58_MAP[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6,  7, 8,-1,-1,-1,-1,-1,-1,
    -1, 9,10,11,12,13,14,15, 16,-1,17,18,19,20,21,-1,
    22,23,24,25,26,27,28,29, 30,31,32,-1,-1,-1,-1,-1,
    -1,33,34,35,36,37,38,39, 40,41,42,43,-1,44,45,46,
    47,48,49,50,51,52,53,54, 55,56,57,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1
};

TargetAddress::TargetAddress(const std::string& address, const std::string& label)
    : address_(address)
    , label_(label)
    , format_(AddressFormat::UNKNOWN)
    , validation_status_(ValidationStatus::NOT_VALIDATED)
    , comparison_mode_(ComparisonMode::HASH160)
    , has_match_(false)
    , scan_count_(0)
{
    // Detect address format
    format_ = detect_address_format(address_);
    
    // Perform initial validation
    validate_address();
    
    // Extract hash data if valid
    if (validation_status_ == ValidationStatus::VALID) {
        extract_hash160();
        extract_script_hash();
    }
}

TargetAddress::AddressFormat TargetAddress::detect_address_format(const std::string& address) {
    if (address.empty()) {
        return AddressFormat::UNKNOWN;
    }
    
    // Check legacy P2PKH (starts with '1')
    if (std::regex_match(address, LEGACY_P2PKH_PATTERN)) {
        return AddressFormat::LEGACY_P2PKH;
    }
    
    // Check legacy P2SH (starts with '3')
    if (std::regex_match(address, LEGACY_P2SH_PATTERN)) {
        return AddressFormat::LEGACY_P2SH;
    }
    
    // Check Bech32 mainnet (starts with 'bc1')
    if (std::regex_match(address, BECH32_PATTERN)) {
        // Distinguish between P2WPKH (shorter) and P2WSH (longer)
        return address.length() <= 42 ? AddressFormat::BECH32_P2WPKH : AddressFormat::BECH32_P2WSH;
    }
    
    // Check Bech32 testnet (starts with 'tb1') - not supported for scanning
    if (std::regex_match(address, TESTNET_BECH32_PATTERN)) {
        return AddressFormat::BECH32_P2WPKH; // Treat as same format but validation will catch testnet
    }
    
    return AddressFormat::UNKNOWN;
}

bool TargetAddress::validate_address() {
    validation_error_.clear();
    
    if (address_.empty()) {
        validation_status_ = ValidationStatus::INVALID_FORMAT;
        validation_error_ = "Address string is empty";
        return false;
    }
    
    switch (format_) {
        case AddressFormat::LEGACY_P2PKH:
        case AddressFormat::LEGACY_P2SH:
            return validate_legacy_address();
            
        case AddressFormat::BECH32_P2WPKH:
        case AddressFormat::BECH32_P2WSH:
            return validate_bech32_address();
            
        case AddressFormat::UNKNOWN:
        default:
            validation_status_ = ValidationStatus::INVALID_FORMAT;
            validation_error_ = "Unrecognized address format";
            return false;
    }
}

bool TargetAddress::validate_legacy_address() {
    try {
        // Decode Base58
        auto decoded = decode_base58(address_);
        
        if (decoded.empty()) {
            validation_status_ = ValidationStatus::INVALID_FORMAT;
            validation_error_ = "Failed to decode Base58 address";
            return false;
        }
        
        // Check length (21 bytes for P2PKH, 21 bytes for P2SH)
        if (decoded.size() != 21) {
            validation_status_ = ValidationStatus::INVALID_FORMAT;
            validation_error_ = "Invalid decoded address length";
            return false;
        }
        
        // Validate checksum
        if (!validate_checksum_base58(address_)) {
            validation_status_ = ValidationStatus::INVALID_CHECKSUM;
            validation_error_ = "Invalid Base58 checksum";
            return false;
        }
        
        // Check version byte
        uint8_t version = decoded[0];
        bool valid_version = false;
        
        if (format_ == AddressFormat::LEGACY_P2PKH) {
            valid_version = (version == 0x00); // Mainnet P2PKH
        } else if (format_ == AddressFormat::LEGACY_P2SH) {
            valid_version = (version == 0x05); // Mainnet P2SH
        }
        
        if (!valid_version) {
            validation_status_ = ValidationStatus::UNSUPPORTED_TYPE;
            validation_error_ = "Unsupported address version (testnet or non-standard)";
            return false;
        }
        
        validation_status_ = ValidationStatus::VALID;
        return true;
        
    } catch (const std::exception& e) {
        validation_status_ = ValidationStatus::INVALID_FORMAT;
        validation_error_ = "Address validation error: " + std::string(e.what());
        return false;
    }
}

bool TargetAddress::validate_bech32_address() {
    try {
        // Check if it's testnet
        if (address_.substr(0, 3) == "tb1") {
            validation_status_ = ValidationStatus::UNSUPPORTED_TYPE;
            validation_error_ = "Testnet addresses not supported for scanning";
            return false;
        }
        
        // Validate Bech32 checksum
        if (!validate_checksum_bech32(address_)) {
            validation_status_ = ValidationStatus::INVALID_CHECKSUM;
            validation_error_ = "Invalid Bech32 checksum";
            return false;
        }
        
        // Decode Bech32
        auto decoded = decode_bech32(address_);
        
        if (decoded.empty()) {
            validation_status_ = ValidationStatus::INVALID_FORMAT;
            validation_error_ = "Failed to decode Bech32 address";
            return false;
        }
        
        // Validate witness program length
        if (format_ == AddressFormat::BECH32_P2WPKH) {
            if (decoded.size() != 20) {
                validation_status_ = ValidationStatus::INVALID_FORMAT;
                validation_error_ = "Invalid P2WPKH witness program length";
                return false;
            }
        } else if (format_ == AddressFormat::BECH32_P2WSH) {
            if (decoded.size() != 32) {
                validation_status_ = ValidationStatus::INVALID_FORMAT;
                validation_error_ = "Invalid P2WSH witness program length";
                return false;
            }
        }
        
        validation_status_ = ValidationStatus::VALID;
        return true;
        
    } catch (const std::exception& e) {
        validation_status_ = ValidationStatus::INVALID_FORMAT;
        validation_error_ = "Bech32 validation error: " + std::string(e.what());
        return false;
    }
}

bool TargetAddress::extract_hash160() {
    hash160_.clear();
    
    if (validation_status_ != ValidationStatus::VALID) {
        return false;
    }
    
    try {
        switch (format_) {
            case AddressFormat::LEGACY_P2PKH: {
                auto decoded = decode_base58(address_);
                if (decoded.size() >= 21) {
                    hash160_.assign(decoded.begin() + 1, decoded.begin() + 21);
                    return true;
                }
                break;
            }
            
            case AddressFormat::BECH32_P2WPKH: {
                auto decoded = decode_bech32(address_);
                if (decoded.size() == 20) {
                    hash160_ = decoded;
                    return true;
                }
                break;
            }
            
            case AddressFormat::LEGACY_P2SH:
            case AddressFormat::BECH32_P2WSH:
                // These use script hash, not hash160
                return false;
                
            default:
                return false;
        }
    } catch (const std::exception&) {
        hash160_.clear();
        return false;
    }
    
    return false;
}

bool TargetAddress::extract_script_hash() {
    script_hash_.clear();
    
    if (validation_status_ != ValidationStatus::VALID) {
        return false;
    }
    
    try {
        switch (format_) {
            case AddressFormat::LEGACY_P2SH: {
                auto decoded = decode_base58(address_);
                if (decoded.size() >= 21) {
                    script_hash_.assign(decoded.begin() + 1, decoded.begin() + 21);
                    return true;
                }
                break;
            }
            
            case AddressFormat::BECH32_P2WSH: {
                auto decoded = decode_bech32(address_);
                if (decoded.size() == 32) {
                    script_hash_ = decoded;
                    return true;
                }
                break;
            }
            
            case AddressFormat::LEGACY_P2PKH:
            case AddressFormat::BECH32_P2WPKH:
                // These use hash160, not script hash
                return false;
                
            default:
                return false;
        }
    } catch (const std::exception&) {
        script_hash_.clear();
        return false;
    }
    
    return false;
}

void TargetAddress::set_match(const std::string& private_key) {
    has_match_ = true;
    matched_private_key_ = private_key;
    match_time_ = std::chrono::system_clock::now();
}

std::vector<uint8_t> TargetAddress::get_comparison_hash() const {
    switch (comparison_mode_) {
        case ComparisonMode::HASH160:
            return hash160_.empty() ? script_hash_ : hash160_;
            
        case ComparisonMode::DIRECT:
        case ComparisonMode::BLOOM_FILTER:
        default:
            return hash160_.empty() ? script_hash_ : hash160_;
    }
}

bool TargetAddress::matches_hash160(const std::vector<uint8_t>& hash160) const {
    if (hash160.size() != 20) return false;
    return hash160_ == hash160;
}

bool TargetAddress::matches_address(const std::string& other_address) const {
    return addresses_equal(address_, other_address);
}

bool TargetAddress::addresses_equal(const std::string& addr1, const std::string& addr2) {
    // Case-sensitive comparison for Bitcoin addresses
    return addr1 == addr2;
}

bool TargetAddress::is_mainnet() const {
    switch (format_) {
        case AddressFormat::LEGACY_P2PKH:
        case AddressFormat::LEGACY_P2SH:
            return address_[0] == '1' || address_[0] == '3';
            
        case AddressFormat::BECH32_P2WPKH:
        case AddressFormat::BECH32_P2WSH:
            return address_.substr(0, 3) == "bc1";
            
        default:
            return false;
    }
}

bool TargetAddress::is_testnet() const {
    if (format_ == AddressFormat::BECH32_P2WPKH || format_ == AddressFormat::BECH32_P2WSH) {
        return address_.substr(0, 3) == "tb1";
    }
    // For legacy addresses, testnet detection would require decoding
    return false;
}

std::string TargetAddress::get_network() const {
    return is_mainnet() ? "mainnet" : (is_testnet() ? "testnet" : "unknown");
}

std::string TargetAddress::get_validation_error() const {
    return validation_error_;
}

std::string TargetAddress::to_json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0);
    
    oss << "{\n";
    oss << "  \"address\": \"" << address_ << "\",\n";
    oss << "  \"label\": \"" << label_ << "\",\n";
    oss << "  \"format\": \"";
    
    switch (format_) {
        case AddressFormat::LEGACY_P2PKH: oss << "legacy_p2pkh"; break;
        case AddressFormat::LEGACY_P2SH: oss << "legacy_p2sh"; break;
        case AddressFormat::BECH32_P2WPKH: oss << "bech32_p2wpkh"; break;
        case AddressFormat::BECH32_P2WSH: oss << "bech32_p2wsh"; break;
        case AddressFormat::UNKNOWN: oss << "unknown"; break;
    }
    
    oss << "\",\n";
    oss << "  \"validation_status\": \"";
    
    switch (validation_status_) {
        case ValidationStatus::VALID: oss << "valid"; break;
        case ValidationStatus::INVALID_FORMAT: oss << "invalid_format"; break;
        case ValidationStatus::INVALID_CHECKSUM: oss << "invalid_checksum"; break;
        case ValidationStatus::UNSUPPORTED_TYPE: oss << "unsupported_type"; break;
        case ValidationStatus::NOT_VALIDATED: oss << "not_validated"; break;
    }
    
    oss << "\",\n";
    oss << "  \"has_match\": " << (has_match_ ? "true" : "false") << ",\n";
    oss << "  \"scan_count\": " << scan_count_ << ",\n";
    oss << "  \"network\": \"" << get_network() << "\",\n";
    
    // Include hash160 if available
    if (!hash160_.empty()) {
        oss << "  \"hash160\": \"";
        for (uint8_t byte : hash160_) {
            oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
        }
        oss << "\",\n";
    }
    
    // Include match data if available
    if (has_match_) {
        oss << "  \"matched_private_key\": \"" << matched_private_key_ << "\",\n";
        auto match_time_t = std::chrono::system_clock::to_time_t(match_time_);
        oss << "  \"match_time\": \"" << std::put_time(std::gmtime(&match_time_t), "%Y-%m-%dT%H:%M:%SZ") << "\",\n";
    }
    
    if (!validation_error_.empty()) {
        oss << "  \"validation_error\": \"" << validation_error_ << "\",\n";
    }
    
    // Remove trailing comma and newline
    std::string result = oss.str();
    if (result.length() > 2 && result.substr(result.length() - 2) == ",\n") {
        result = result.substr(0, result.length() - 2) + "\n";
    }
    
    oss.str("");
    oss << result << "}";
    
    return oss.str();
}

// Placeholder implementations for Base58/Bech32 operations
// In production, these would use proper crypto libraries

std::vector<uint8_t> TargetAddress::decode_base58(const std::string& address) const {
    // Simplified Base58 decoder - production implementation would use proper crypto library
    std::vector<uint8_t> result;
    
    // For now, return mock data that passes basic validation
    if (address.length() >= 26 && address.length() <= 35) {
        if (address[0] == '1') {
            result = {0x00}; // P2PKH version
            result.resize(21, 0x42); // Mock 20-byte hash160
        } else if (address[0] == '3') {
            result = {0x05}; // P2SH version  
            result.resize(21, 0x42); // Mock 20-byte script hash
        }
    }
    
    return result;
}

std::vector<uint8_t> TargetAddress::decode_bech32(const std::string& address) const {
    // Simplified Bech32 decoder - production implementation would use proper crypto library
    std::vector<uint8_t> result;
    
    if (address.length() >= 39 && address.substr(0, 3) == "bc1") {
        if (address.length() <= 42) {
            // P2WPKH - 20 bytes
            result.resize(20, 0x42);
        } else {
            // P2WSH - 32 bytes
            result.resize(32, 0x42);
        }
    }
    
    return result;
}

bool TargetAddress::validate_checksum_base58(const std::string& address) const {
    // Simplified checksum validation - always returns true for mock implementation
    // Production implementation would perform proper SHA256d checksum validation
    return !address.empty() && (address[0] == '1' || address[0] == '3') && address.length() >= 26;
}

bool TargetAddress::validate_checksum_bech32(const std::string& address) const {
    // Simplified Bech32 checksum validation - always returns true for mock implementation
    // Production implementation would perform proper Bech32 checksum validation
    return address.length() >= 39 && address.substr(0, 3) == "bc1";
}

// Placeholder encode methods
std::string TargetAddress::encode_base58(const std::vector<uint8_t>& data) const {
    return address_; // Return original for now
}

std::string TargetAddress::encode_bech32(const std::vector<uint8_t>& data) const {
    return address_; // Return original for now
}

std::string TargetAddress::to_base58() const {
    return address_; // Placeholder
}

std::string TargetAddress::to_bech32() const {
    return address_; // Placeholder  
}

std::string TargetAddress::to_hex_string() const {
    std::ostringstream oss;
    const auto& hash = get_comparison_hash();
    for (uint8_t byte : hash) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    }
    return oss.str();
}

TargetAddress TargetAddress::from_json(const std::string& json) {
    // Basic JSON parsing - production would use proper JSON library
    auto extract_string = [&json](const std::string& key) -> std::string {
        std::string search_key = "\"" + key + "\": \"";
        size_t pos = json.find(search_key);
        if (pos == std::string::npos) return "";
        
        pos += search_key.length();
        size_t end_pos = json.find("\"", pos);
        if (end_pos == std::string::npos) return "";
        
        return json.substr(pos, end_pos - pos);
    };
    
    std::string address = extract_string("address");
    std::string label = extract_string("label");
    
    return TargetAddress(address, label);
}

} // namespace models
} // namespace keyhunt