/**
 * @file checkpoint_manager.h
 * @brief Advanced checkpoint management system with progress persistence and recovery capabilities
 * @author KeyhuntCUDA Team
 * 
 * T042: Develop checkpoint management system with progress persistence and recovery capabilities
 * 
 * Provides comprehensive checkpoint management for long-running private key scanning operations
 * with atomic operations, data integrity verification, compression, versioning, and recovery.
 */

#pragma once

#include "../models/CheckpointData.h"
#include "../models/PrivateKeyRange.h"
#include "../models/GPUConfiguration.h"
#include "../ecc/secp256k1.h"
#include "private_key_scanner.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <mutex>
#include <atomic>
#include <map>
#include <fstream>
#include <functional>

namespace keyhunt {
namespace scan {
namespace checkpoint {

/**
 * @brief Checkpoint file format version for compatibility
 */
constexpr uint32_t CHECKPOINT_FORMAT_VERSION = 0x00020001;  // v2.1
constexpr uint32_t CHECKPOINT_MAGIC_HEADER = 0x4B484350;   // "KHCP"

/**
 * @brief Checkpoint metadata information
 */
struct CheckpointMetadata {
    uint32_t format_version;                        // File format version
    uint32_t magic_header;                          // Magic header for validation
    std::chrono::system_clock::time_point created_time;  // Checkpoint creation time
    std::chrono::system_clock::time_point modified_time; // Last modification time
    std::string keyhunt_version;                    // KeyhuntCUDA version
    std::string system_info;                        // System information
    uint64_t file_size;                            // Total file size
    uint64_t data_checksum;                        // Data integrity checksum
    uint32_t compression_type;                      // Compression algorithm used
    bool is_incremental;                           // Whether this is incremental checkpoint
    std::string previous_checkpoint;                // Previous checkpoint file (if incremental)
    
    CheckpointMetadata() 
        : format_version(CHECKPOINT_FORMAT_VERSION)
        , magic_header(CHECKPOINT_MAGIC_HEADER)
        , created_time(std::chrono::system_clock::now())
        , modified_time(std::chrono::system_clock::now())
        , keyhunt_version("2.1.0")
        , file_size(0)
        , data_checksum(0)
        , compression_type(0)
        , is_incremental(false)
    {}
};

/**
 * @brief Extended checkpoint data with additional recovery information
 */
struct ExtendedCheckpointData : public models::CheckpointData {
    // Scanning state information
    ScanningConfiguration scanning_config;          // Complete scanning configuration
    std::vector<std::string> target_addresses;      // Target address list
    
    // Progress tracking
    ecc::BigInt256 original_start_key;              // Original range start
    ecc::BigInt256 original_end_key;                // Original range end
    ecc::BigInt256 keys_processed_successfully;     // Successfully processed keys
    ecc::BigInt256 keys_with_errors;                // Keys that had processing errors
    
    // Performance metrics history
    struct MetricsSnapshot {
        std::chrono::system_clock::time_point timestamp;
        double keys_per_second;
        double gpu_utilization;
        double memory_utilization;
        size_t batch_count;
        size_t error_count;
    };
    std::vector<MetricsSnapshot> metrics_history;
    
    // GPU state information
    models::GPUConfiguration gpu_config;            // GPU configuration used
    std::vector<int> active_devices;                // Active GPU devices
    
    // Batch processing state
    struct BatchCheckpoint {
        size_t batch_id;
        ecc::BigInt256 batch_start_key;
        ecc::BigInt256 batch_end_key;
        enum class BatchState {
            PENDING, PROCESSING, COMPLETED, FAILED
        } state;
        std::chrono::milliseconds processing_time;
        std::vector<PrivateKeyScanner::ScanMatch> matches_in_batch;
    };
    std::vector<BatchCheckpoint> batch_history;
    
    // Error tracking and recovery information
    struct ErrorRecord {
        std::chrono::system_clock::time_point timestamp;
        std::string error_type;
        std::string error_message;
        ecc::BigInt256 error_key_range_start;
        ecc::BigInt256 error_key_range_end;
        int device_id;
        size_t retry_count;
    };
    std::vector<ErrorRecord> error_records;
    
    // Validation information
    struct ValidationCheckpoint {
        size_t keys_validated;
        size_t validation_passed;
        size_t validation_failed;
        double average_validation_time;
        std::chrono::system_clock::time_point last_validation;
    };
    ValidationCheckpoint validation_info;
    
    // Recovery hints
    std::vector<ecc::BigInt256> skip_ranges_start;  // Ranges to skip on recovery
    std::vector<ecc::BigInt256> skip_ranges_end;    // (due to repeated errors)
    std::vector<ecc::BigInt256> priority_ranges_start; // High-priority ranges
    std::vector<ecc::BigInt256> priority_ranges_end;   // (near known patterns)
    
    ExtendedCheckpointData() : models::CheckpointData() {
        validation_info.keys_validated = 0;
        validation_info.validation_passed = 0;
        validation_info.validation_failed = 0;
        validation_info.average_validation_time = 0.0;
        validation_info.last_validation = std::chrono::system_clock::now();
    }
};

/**
 * @brief Checkpoint compression algorithms
 */
enum class CompressionType : uint32_t {
    NONE = 0,           // No compression
    ZLIB = 1,           // zlib compression
    LZ4 = 2,            // LZ4 fast compression  
    ZSTD = 3,           // Zstandard compression
    BROTLI = 4          // Brotli compression
};

/**
 * @brief Checkpoint file integrity verification
 */
struct CheckpointIntegrityInfo {
    uint64_t crc32_checksum;            // CRC32 checksum
    uint64_t xxhash_checksum;           // xxHash checksum
    std::vector<uint8_t> sha256_hash;   // SHA256 hash
    bool integrity_verified;            // Verification status
    std::string verification_error;     // Error message if verification failed
};

/**
 * @brief Main checkpoint manager class
 */
class CheckpointManager {
public:
    CheckpointManager();
    ~CheckpointManager();
    
    // Initialization and configuration
    bool initialize(const std::string& base_directory = "checkpoints");
    void cleanup();
    
    // Checkpoint creation and management
    bool create_checkpoint(
        const ExtendedCheckpointData& checkpoint_data,
        const std::string& filename = "",
        CompressionType compression = CompressionType::ZSTD
    );
    
    bool create_incremental_checkpoint(
        const ExtendedCheckpointData& checkpoint_data,
        const std::string& base_checkpoint_filename,
        const std::string& incremental_filename = ""
    );
    
    bool create_automatic_checkpoint(
        const ExtendedCheckpointData& checkpoint_data,
        const std::string& prefix = "auto"
    );
    
    // Checkpoint loading and recovery
    bool load_checkpoint(
        const std::string& filename,
        ExtendedCheckpointData& checkpoint_data
    );
    
    bool load_latest_checkpoint(
        const std::string& prefix,
        ExtendedCheckpointData& checkpoint_data,
        std::string& loaded_filename
    );
    
    bool merge_incremental_checkpoints(
        const std::string& base_filename,
        const std::vector<std::string>& incremental_filenames,
        ExtendedCheckpointData& merged_data
    );
    
    // Checkpoint validation and integrity
    bool verify_checkpoint_integrity(const std::string& filename);
    CheckpointIntegrityInfo get_checkpoint_integrity_info(const std::string& filename);
    bool repair_checkpoint(const std::string& filename, const std::string& repaired_filename = "");
    
    // Checkpoint management operations
    std::vector<std::string> list_checkpoints(const std::string& prefix = "") const;
    std::vector<std::string> list_checkpoint_chain(const std::string& latest_filename) const;
    bool delete_checkpoint(const std::string& filename);
    bool archive_old_checkpoints(std::chrono::hours age_threshold = std::chrono::hours(24 * 7));
    
    // Checkpoint analysis and reporting
    struct CheckpointAnalysisReport {
        std::string filename;
        CheckpointMetadata metadata;
        size_t total_keys_scanned;
        size_t total_matches_found;
        double average_scanning_speed;
        double peak_scanning_speed;
        std::chrono::milliseconds total_scanning_time;
        std::chrono::milliseconds estimated_completion_time;
        double progress_percentage;
        size_t error_count;
        size_t validation_errors;
        std::string system_info;
        std::vector<std::string> warnings;
    };
    
    CheckpointAnalysisReport analyze_checkpoint(const std::string& filename);
    std::vector<CheckpointAnalysisReport> analyze_checkpoint_chain(const std::string& latest_filename);
    
    // Configuration and settings
    struct CheckpointSettings {
        std::string base_directory;                 // Checkpoint storage directory
        size_t max_checkpoint_files;               // Maximum checkpoint files to keep
        std::chrono::seconds auto_checkpoint_interval; // Automatic checkpoint interval
        CompressionType default_compression;       // Default compression algorithm
        bool enable_integrity_checks;              // Enable integrity verification
        bool enable_incremental_checkpoints;      // Enable incremental checkpoints
        bool enable_automatic_cleanup;             // Automatic old file cleanup
        size_t max_error_records;                  // Maximum error records to keep
        size_t max_metrics_history;               // Maximum metrics snapshots
        bool encrypt_checkpoints;                  // Enable checkpoint encryption
        std::string encryption_key;                // Encryption key (if enabled)
        
        CheckpointSettings() 
            : base_directory("checkpoints")
            , max_checkpoint_files(50)
            , auto_checkpoint_interval(std::chrono::minutes(5))
            , default_compression(CompressionType::ZSTD)
            , enable_integrity_checks(true)
            , enable_incremental_checkpoints(true)
            , enable_automatic_cleanup(true)
            , max_error_records(1000)
            , max_metrics_history(100)
            , encrypt_checkpoints(false)
        {}
    };
    
    void configure(const CheckpointSettings& settings);
    CheckpointSettings get_current_settings() const;
    
    // Automatic checkpoint management
    bool enable_automatic_checkpointing(
        std::function<ExtendedCheckpointData()> data_provider,
        const std::string& prefix = "auto"
    );
    
    bool disable_automatic_checkpointing();
    bool is_automatic_checkpointing_enabled() const;
    
    // Recovery assistance
    struct RecoveryRecommendations {
        std::string recommended_checkpoint;         // Best checkpoint to resume from
        std::vector<std::string> alternative_checkpoints; // Alternative options
        std::vector<ecc::BigInt256> suggested_skip_ranges_start; // Ranges to skip
        std::vector<ecc::BigInt256> suggested_skip_ranges_end;
        bool requires_validation;                   // Whether validation is needed
        std::string recovery_strategy;              // Recommended recovery strategy
        std::vector<std::string> warnings;         // Recovery warnings
    };
    
    RecoveryRecommendations get_recovery_recommendations(
        const std::string& scan_prefix = "",
        const ecc::BigInt256& target_range_start = ecc::BigInt256(),
        const ecc::BigInt256& target_range_end = ecc::BigInt256()
    );
    
    bool apply_recovery_recommendations(
        const RecoveryRecommendations& recommendations,
        ExtendedCheckpointData& checkpoint_data
    );
    
    // Performance monitoring for checkpoint operations
    struct CheckpointPerformanceMetrics {
        std::chrono::milliseconds save_time;       // Time to save checkpoint
        std::chrono::milliseconds load_time;       // Time to load checkpoint
        size_t uncompressed_size;                  // Uncompressed data size
        size_t compressed_size;                    // Compressed file size
        double compression_ratio;                  // Compression efficiency
        size_t disk_io_bytes;                      // Total disk I/O
        double save_throughput_mbps;               // Save throughput MB/s
        double load_throughput_mbps;               // Load throughput MB/s
    };
    
    CheckpointPerformanceMetrics get_last_operation_metrics() const;
    std::vector<CheckpointPerformanceMetrics> get_performance_history() const;
    
    // Integration with scanning framework
    bool register_scanner(PrivateKeyScanner* scanner);
    bool unregister_scanner();
    bool create_scanner_checkpoint(const std::string& filename = "");
    bool restore_scanner_from_checkpoint(const std::string& filename);
    
    // Error handling and diagnostics
    enum class CheckpointError {
        SUCCESS = 0,
        FILE_NOT_FOUND,
        INVALID_FORMAT,
        CORRUPTED_DATA,
        COMPRESSION_ERROR,
        INTEGRITY_FAILURE,
        PERMISSION_DENIED,
        DISK_FULL,
        MEMORY_ERROR,
        ENCRYPTION_ERROR
    };
    
    CheckpointError get_last_error() const;
    std::string get_error_description(CheckpointError error) const;
    std::vector<std::string> get_error_log() const;
    void clear_error_log();
    
    // Utility functions
    static std::string generate_checkpoint_filename(
        const std::string& prefix = "checkpoint",
        const std::string& suffix = ".khcp"
    );
    
    static bool is_valid_checkpoint_file(const std::string& filename);
    static CheckpointMetadata extract_metadata(const std::string& filename);
    static std::string format_checkpoint_info(const CheckpointMetadata& metadata);

private:
    // Internal state
    std::string base_directory_;
    CheckpointSettings settings_;
    mutable std::mutex checkpoint_mutex_;
    
    // Automatic checkpointing
    std::atomic<bool> auto_checkpointing_enabled_;
    std::function<ExtendedCheckpointData()> auto_data_provider_;
    std::string auto_checkpoint_prefix_;
    std::thread auto_checkpoint_thread_;
    std::atomic<bool> should_stop_auto_checkpointing_;
    
    // Scanner integration
    PrivateKeyScanner* registered_scanner_;
    
    // Performance tracking
    CheckpointPerformanceMetrics last_metrics_;
    std::vector<CheckpointPerformanceMetrics> performance_history_;
    
    // Error handling
    mutable std::mutex error_mutex_;
    CheckpointError last_error_;
    std::vector<std::string> error_log_;
    
    // Internal helper methods
    
    // File I/O operations
    bool write_checkpoint_file(
        const std::string& filename,
        const ExtendedCheckpointData& data,
        const CheckpointMetadata& metadata,
        CompressionType compression
    );
    
    bool read_checkpoint_file(
        const std::string& filename,
        ExtendedCheckpointData& data,
        CheckpointMetadata& metadata
    );
    
    // Compression and decompression
    std::vector<uint8_t> compress_data(
        const std::vector<uint8_t>& input_data,
        CompressionType compression_type
    );
    
    std::vector<uint8_t> decompress_data(
        const std::vector<uint8_t>& compressed_data,
        CompressionType compression_type
    );
    
    // Serialization
    std::vector<uint8_t> serialize_checkpoint_data(const ExtendedCheckpointData& data);
    bool deserialize_checkpoint_data(
        const std::vector<uint8_t>& serialized_data,
        ExtendedCheckpointData& data
    );
    
    // Integrity verification
    uint64_t calculate_crc32(const std::vector<uint8_t>& data);
    uint64_t calculate_xxhash(const std::vector<uint8_t>& data);
    std::vector<uint8_t> calculate_sha256(const std::vector<uint8_t>& data);
    
    // File management
    bool ensure_directory_exists(const std::string& directory);
    std::string get_full_path(const std::string& filename) const;
    bool is_filename_valid(const std::string& filename) const;
    
    // Automatic checkpointing thread
    void auto_checkpoint_worker();
    
    // Error logging
    void log_error(CheckpointError error, const std::string& details);
    void set_last_error(CheckpointError error);
    
    // Recovery analysis
    double calculate_checkpoint_quality_score(const ExtendedCheckpointData& data) const;
    bool analyze_error_patterns(const ExtendedCheckpointData& data, RecoveryRecommendations& recommendations) const;
    
    // Performance measurement
    void start_performance_measurement();
    void end_performance_measurement(bool is_save_operation, size_t data_size);
};

/**
 * @brief Checkpoint manager factory for different storage backends
 */
class CheckpointManagerFactory {
public:
    enum class StorageBackend {
        LOCAL_FILESYSTEM,   // Local file system storage
        NETWORK_FILESYSTEM, // Network file system (NFS, etc.)
        CLOUD_STORAGE,      // Cloud storage (S3, etc.)
        DATABASE_STORAGE,   // Database storage
        MEMORY_STORAGE      // In-memory storage (testing)
    };
    
    static std::unique_ptr<CheckpointManager> create_manager(
        StorageBackend backend,
        const std::map<std::string, std::string>& backend_config = {}
    );
    
    static std::vector<StorageBackend> get_available_backends();
    static std::string get_backend_description(StorageBackend backend);
};

/**
 * @brief Utility classes for checkpoint operations
 */
namespace checkpoint_utils {
    
    /**
     * @brief Checkpoint file analyzer
     */
    class CheckpointAnalyzer {
    public:
        static bool validate_checkpoint_chain(const std::vector<std::string>& checkpoint_files);
        static std::vector<std::string> find_missing_checkpoints(const std::vector<std::string>& checkpoint_files);
        static bool reconstruct_checkpoint_from_fragments(
            const std::vector<std::string>& fragment_files,
            const std::string& output_filename
        );
    };
    
    /**
     * @brief Checkpoint conversion utilities
     */
    class CheckpointConverter {
    public:
        static bool convert_format_version(
            const std::string& input_filename,
            const std::string& output_filename,
            uint32_t target_version
        );
        
        static bool export_to_json(
            const std::string& checkpoint_filename,
            const std::string& json_filename
        );
        
        static bool import_from_json(
            const std::string& json_filename,
            const std::string& checkpoint_filename
        );
    };
    
    /**
     * @brief Performance benchmarking for checkpoint operations
     */
    class CheckpointBenchmarker {
    public:
        struct BenchmarkResults {
            std::map<CompressionType, double> compression_speeds;   // MB/s
            std::map<CompressionType, double> decompression_speeds; // MB/s
            std::map<CompressionType, double> compression_ratios;   // ratio
            double baseline_io_speed;                              // MB/s
            std::string fastest_compression;
            std::string best_ratio_compression;
        };
        
        static BenchmarkResults benchmark_compression_methods(size_t test_data_size = 100*1024*1024);
        static double benchmark_disk_io_speed(const std::string& directory);
    };
}

} // namespace checkpoint
} // namespace scan
} // namespace keyhunt