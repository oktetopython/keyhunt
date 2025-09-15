/**
 * @file secp256k1.h
 * @brief Unified CPU/GPU secp256k1 elliptic curve operations interface
 * @author KeyhuntCUDA Team
 * 
 * This provides the main interface for secp256k1 elliptic curve operations
 * extracted and fused from CudaBrainSecp and BitCrack implementations.
 * Supports both CPU reference validation and GPU-accelerated computation.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

namespace keyhunt {
namespace ecc {

// Forward declarations
struct Point;
struct PrivateKey;
struct PublicKey;

/**
 * @brief 256-bit integer representation for secp256k1 operations
 */
struct BigInt256 {
    uint64_t d[4];  // 4 x 64-bit limbs (little-endian)
    
    // Constructors
    BigInt256();
    BigInt256(uint64_t val);
    BigInt256(const uint64_t data[4]);
    BigInt256(const std::string& hex);
    
    // Basic operations
    void set_zero();
    void set_one();
    bool is_zero() const;
    bool is_one() const;
    
    // String conversion
    std::string to_hex() const;
    void from_hex(const std::string& hex);
    
    // Comparison
    bool operator==(const BigInt256& other) const;
    bool operator!=(const BigInt256& other) const;
    bool operator<(const BigInt256& other) const;
    bool operator>(const BigInt256& other) const;
};

/**
 * @brief Point on the secp256k1 elliptic curve
 */
struct Point {
    BigInt256 x;     // X coordinate
    BigInt256 y;     // Y coordinate
    BigInt256 z;     // Z coordinate (for projective coordinates)
    bool is_infinity; // Point at infinity flag
    
    // Constructors
    Point();
    Point(const BigInt256& x, const BigInt256& y);
    Point(const BigInt256& x, const BigInt256& y, const BigInt256& z);
    
    // Conversion between affine and projective coordinates
    void to_affine();
    void to_projective();
    
    // Validation
    bool is_valid() const;
    bool is_on_curve() const;
    
    // Serialization
    std::vector<uint8_t> serialize_compressed() const;
    std::vector<uint8_t> serialize_uncompressed() const;
    bool deserialize(const std::vector<uint8_t>& data);
    
    // String representation
    std::string to_hex() const;
};

/**
 * @brief Private key wrapper for secp256k1
 */
struct PrivateKey {
    BigInt256 key;   // Private key value
    
    // Constructors
    PrivateKey();
    PrivateKey(const BigInt256& k);
    PrivateKey(const std::string& hex);
    
    // Validation
    bool is_valid() const;
    
    // Conversion
    std::string to_hex() const;
    std::vector<uint8_t> to_bytes() const;
};

/**
 * @brief Public key wrapper for secp256k1
 */
struct PublicKey {
    Point point;     // Public key point
    
    // Constructors
    PublicKey();
    PublicKey(const Point& p);
    
    // Validation
    bool is_valid() const;
    
    // Serialization
    std::vector<uint8_t> serialize_compressed() const;
    std::vector<uint8_t> serialize_uncompressed() const;
    
    // Address generation
    std::vector<uint8_t> get_hash160_compressed() const;
    std::vector<uint8_t> get_hash160_uncompressed() const;
};

/**
 * @brief secp256k1 curve constants and parameters
 */
namespace constants {
    extern const BigInt256 FIELD_PRIME;      // p = 2^256 - 2^32 - 977
    extern const BigInt256 GROUP_ORDER;      // n = curve order
    extern const BigInt256 CURVE_B;          // b = 7 (y^2 = x^3 + 7)
    extern const Point GENERATOR;            // G = generator point
    extern const BigInt256 HALF_ORDER;       // n/2 for signature canonicalization
    
    // Montgomery constants for modular arithmetic
    extern const uint64_t MM64;              // -p^(-1) mod 2^64
    extern const BigInt256 R;                // 2^256 mod p
    extern const BigInt256 R2;               // 2^512 mod p
}

// CPU Reference Implementation Namespace
namespace cpu {
    
    /**
     * @brief CPU reference implementation for validation
     */
    class Secp256k1 {
    public:
        Secp256k1();
        ~Secp256k1();
        
        // Initialization
        bool initialize();
        void cleanup();
        
        // Core ECC operations
        Point scalar_multiply(const BigInt256& scalar, const Point& point);
        Point point_add(const Point& p1, const Point& p2);
        Point point_double(const Point& p);
        Point point_negate(const Point& p);
        
        // Key operations
        PublicKey compute_public_key(const PrivateKey& private_key);
        bool verify_key_pair(const PrivateKey& priv_key, const PublicKey& pub_key);
        
        // Validation functions
        bool validate_point(const Point& point);
        bool validate_scalar(const BigInt256& scalar);
        
        // Batch operations for validation
        std::vector<Point> batch_scalar_multiply(const std::vector<BigInt256>& scalars);
        
    private:
        bool initialized_;
        std::unique_ptr<class Secp256k1Impl> impl_;
    };
    
    // Modular arithmetic operations
    void mod_add(BigInt256& result, const BigInt256& a, const BigInt256& b);
    void mod_sub(BigInt256& result, const BigInt256& a, const BigInt256& b);
    void mod_mult(BigInt256& result, const BigInt256& a, const BigInt256& b);
    void mod_sqr(BigInt256& result, const BigInt256& a);
    void mod_inv(BigInt256& result, const BigInt256& a);
    
} // namespace cpu

// GPU Implementation Namespace
#ifdef __CUDACC__
namespace gpu {
    
    /**
     * @brief GPU accelerated implementation
     */
    class Secp256k1 {
    public:
        Secp256k1();
        ~Secp256k1();
        
        // Initialization
        bool initialize(int device_id = 0);
        void cleanup();
        
        // Memory management
        bool allocate_device_memory(size_t batch_size);
        void free_device_memory();
        
        // Batch operations for high throughput
        bool batch_scalar_multiply(const std::vector<BigInt256>& scalars,
                                 std::vector<Point>& results);
        
        bool batch_point_add(const std::vector<Point>& points1,
                            const std::vector<Point>& points2,
                            std::vector<Point>& results);
        
        // Performance optimization
        void optimize_for_device();
        void set_thread_block_size(int block_size);
        void set_grid_size(int grid_size);
        
        // Statistics
        double get_operations_per_second() const;
        size_t get_memory_usage() const;
        
    private:
        bool initialized_;
        int device_id_;
        size_t batch_size_;
        
        // Device memory pointers
        void* d_scalars_;
        void* d_points_;
        void* d_results_;
        
        // Performance metrics
        mutable double ops_per_second_;
        mutable size_t memory_usage_;
    };
    
    // Device functions (declared here, defined in .cu files)
    __device__ void mod_mult_gpu(uint64_t* result, const uint64_t* a, const uint64_t* b);
    __device__ void point_add_gpu(uint64_t* p1x, uint64_t* p1y, uint64_t* p1z,
                                 const uint64_t* p2x, const uint64_t* p2y);
    __device__ void scalar_mult_gpu(uint64_t* result_x, uint64_t* result_y,
                                   const uint64_t* scalar, const uint64_t* point_x,
                                   const uint64_t* point_y);
    
} // namespace gpu
#endif // __CUDACC__

/**
 * @brief Validation utilities for CPU/GPU consistency
 */
namespace validation {
    
    /**
     * @brief CPU/GPU consistency validator
     */
    class ConsistencyValidator {
    public:
        ConsistencyValidator();
        ~ConsistencyValidator();
        
        // Validation operations
        bool validate_scalar_multiplication(size_t test_count = 100000);
        bool validate_point_addition(size_t test_count = 100000);
        bool validate_modular_arithmetic(size_t test_count = 100000);
        
        // Precision testing
        bool test_precision_threshold(double threshold = 1e-10);
        
        // Performance comparison
        struct ValidationResults {
            bool passed;
            size_t tests_run;
            size_t tests_passed;
            double max_error;
            double average_error;
            std::string error_message;
        };
        
        ValidationResults run_comprehensive_validation();
        
        // Scientific validation report
        std::string generate_validation_report() const;
        
    private:
        std::unique_ptr<cpu::Secp256k1> cpu_impl_;
#ifdef __CUDACC__
        std::unique_ptr<gpu::Secp256k1> gpu_impl_;
#endif
        mutable ValidationResults last_results_;
    };
    
} // namespace validation

/**
 * @brief High-level secp256k1 interface combining CPU and GPU implementations
 */
class Secp256k1Interface {
public:
    enum class Mode {
        CPU_ONLY,       // Use only CPU implementation
        GPU_ONLY,       // Use only GPU implementation (if available)
        AUTO,           // Automatically choose best implementation
        HYBRID          // Use both for validation
    };
    
    Secp256k1Interface(Mode mode = Mode::AUTO);
    ~Secp256k1Interface();
    
    // Initialization
    bool initialize();
    void cleanup();
    
    // Key operations
    PublicKey compute_public_key(const PrivateKey& private_key);
    bool verify_key_pair(const PrivateKey& priv_key, const PublicKey& pub_key);
    
    // Batch operations
    std::vector<PublicKey> batch_compute_public_keys(const std::vector<PrivateKey>& private_keys);
    
    // Configuration
    void set_mode(Mode mode);
    Mode get_mode() const { return current_mode_; }
    bool is_gpu_available() const;
    
    // Performance and validation
    double get_performance_keys_per_second() const;
    bool run_self_test();
    
private:
    Mode current_mode_;
    std::unique_ptr<cpu::Secp256k1> cpu_impl_;
#ifdef __CUDACC__
    std::unique_ptr<gpu::Secp256k1> gpu_impl_;
#endif
    std::unique_ptr<validation::ConsistencyValidator> validator_;
};

} // namespace ecc
} // namespace keyhunt