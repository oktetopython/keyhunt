/**
 * @file secp256k1_point_optimized.h
 * @brief Header for optimized elliptic curve point operations
 * @author KeyhuntCUDA Team
 * 
 * T037: Implement point operations with projective coordinates and optimized addition chains
 * 
 * Defines interfaces for highly optimized secp256k1 point operations using:
 * - Projective coordinates (X, Y, Z) to avoid expensive modular inversions
 * - Optimized addition chains for scalar multiplication
 * - Precomputation tables for windowed methods
 * - Assembly-level optimizations building on T036
 */

#pragma once

#include "secp256k1.h"
#include "secp256k1_math_optimized.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <memory>

namespace keyhunt {
namespace ecc {
namespace gpu {
namespace optimized {

/**
 * @brief Projective coordinate point representation (X, Y, Z)
 * Point (X, Y, Z) represents affine point (X/Z, Y/Z)
 * Infinity point represented as (1, 1, 0)
 */
struct ProjectivePoint {
    BigInt256 x;  // X coordinate
    BigInt256 y;  // Y coordinate  
    BigInt256 z;  // Z coordinate
    
    ProjectivePoint() = default;
    ProjectivePoint(const BigInt256& x_, const BigInt256& y_, const BigInt256& z_)
        : x(x_), y(y_), z(z_) {}
    
    // Convert from affine coordinates
    explicit ProjectivePoint(const Point& affine) 
        : x(affine.x), y(affine.y), z(BigInt256(1)) {}
    
    // Convert to affine coordinates (requires modular inverse)
    Point to_affine() const;
    
    // Check if point is at infinity
    bool is_infinity() const;
    
    // Normalize to Z=1 (convert to affine in projective form)
    ProjectivePoint normalize() const;
};

/**
 * @brief Precomputed point table for windowed scalar multiplication
 */
struct PrecomputedTable {
    std::vector<ProjectivePoint> points;  // Precomputed multiples
    int window_size;                      // Window size (typically 4-6)
    size_t table_size;                   // Number of precomputed points
    
    PrecomputedTable(int w) : window_size(w), table_size(1ULL << w) {
        points.resize(table_size);
    }
};

/**
 * @brief Configuration for point operation optimizations
 */
struct PointOptimizationConfig {
    int window_size;              // Window size for scalar multiplication (4-6)
    bool use_endomorphism;        // Use GLV endomorphism optimization
    bool use_precomputation;      // Use precomputed tables
    bool use_mixed_coordinates;   // Mix projective and affine coordinates
    bool use_montgomery_ladder;   // Use Montgomery ladder for scalar mult
    
    PointOptimizationConfig() 
        : window_size(5), use_endomorphism(true), use_precomputation(true),
          use_mixed_coordinates(true), use_montgomery_ladder(false) {}
};

// Device function declarations for optimized point operations
__device__ void point_add_projective(ProjectivePoint* result, 
                                    const ProjectivePoint* p1, 
                                    const ProjectivePoint* p2);

__device__ void point_double_projective(ProjectivePoint* result, 
                                       const ProjectivePoint* p);

__device__ void point_triple_projective(ProjectivePoint* result, 
                                       const ProjectivePoint* p);

__device__ void point_multiply_scalar(ProjectivePoint* result, 
                                     const BigInt256* scalar, 
                                     const ProjectivePoint* point);

__device__ void point_multiply_precomputed(ProjectivePoint* result,
                                          const BigInt256* scalar,
                                          const ProjectivePoint* table,
                                          int window_size);

// Batch point operation kernels
__global__ void batch_point_add(ProjectivePoint* results,
                               const ProjectivePoint* p1_array,
                               const ProjectivePoint* p2_array,
                               size_t count);

__global__ void batch_point_double(ProjectivePoint* results,
                                 const ProjectivePoint* points,
                                 size_t count);

__global__ void batch_scalar_multiply(ProjectivePoint* results,
                                    const BigInt256* scalars,
                                    const ProjectivePoint* points,
                                    size_t count);

__global__ void batch_scalar_multiply_precomputed(ProjectivePoint* results,
                                                 const BigInt256* scalars,
                                                 const ProjectivePoint* table,
                                                 int window_size,
                                                 size_t count);

/**
 * @brief High-level C++ interface for optimized point operations
 */
class OptimizedPointOperations {
public:
    OptimizedPointOperations(const PointOptimizationConfig& config = PointOptimizationConfig());
    ~OptimizedPointOperations();
    
    // Initialization
    bool initialize(int device_id = 0);
    void cleanup();
    
    // Single point operations
    cudaError_t point_add(const ProjectivePoint& p1, const ProjectivePoint& p2, 
                         ProjectivePoint& result);
    cudaError_t point_double(const ProjectivePoint& point, ProjectivePoint& result);
    cudaError_t scalar_multiply(const BigInt256& scalar, const ProjectivePoint& point,
                               ProjectivePoint& result);
    
    // Batch point operations
    cudaError_t batch_point_add(const std::vector<ProjectivePoint>& p1_array,
                               const std::vector<ProjectivePoint>& p2_array,
                               std::vector<ProjectivePoint>& results);
    
    cudaError_t batch_scalar_multiply(const std::vector<BigInt256>& scalars,
                                     const std::vector<ProjectivePoint>& points,
                                     std::vector<ProjectivePoint>& results);
    
    cudaError_t batch_scalar_multiply(const std::vector<BigInt256>& scalars,
                                     const ProjectivePoint& base_point,
                                     std::vector<ProjectivePoint>& results);
    
    // Precomputation management
    cudaError_t generate_precomputed_table(const ProjectivePoint& base_point,
                                          int window_size,
                                          std::unique_ptr<PrecomputedTable>& table);
    
    cudaError_t scalar_multiply_with_precomputed(const BigInt256& scalar,
                                                const PrecomputedTable& table,
                                                ProjectivePoint& result);
    
    // Performance monitoring
    struct PointOperationMetrics {
        double avg_point_add_cycles;
        double avg_point_double_cycles;
        double avg_scalar_mult_cycles;
        size_t total_point_operations;
        double operations_per_second;
        double memory_bandwidth_utilization;
    };
    
    PointOperationMetrics get_performance_metrics() const;
    void reset_performance_counters();
    
    // Configuration
    PointOptimizationConfig get_current_config() const { return config_; }
    void update_config(const PointOptimizationConfig& new_config);
    
    // Benchmarking
    struct BenchmarkResults {
        double point_adds_per_second;
        double point_doubles_per_second;
        double scalar_mults_per_second;
        double memory_throughput_gbps;
        std::chrono::milliseconds execution_time;
    };
    
    BenchmarkResults benchmark_point_operations(size_t operation_count = 50000);
    BenchmarkResults benchmark_scalar_multiplication(size_t operation_count = 10000);

private:
    PointOptimizationConfig config_;
    bool initialized_;
    int device_id_;
    
    // Integration with optimized arithmetic
    std::unique_ptr<OptimizedModularArithmetic> arithmetic_;
    
    // GPU memory management
    ProjectivePoint* d_temp_points_1_;
    ProjectivePoint* d_temp_points_2_;
    ProjectivePoint* d_temp_results_;
    BigInt256* d_temp_scalars_;
    ProjectivePoint* d_precomputed_table_;
    size_t allocated_point_count_;
    size_t allocated_table_size_;
    
    // CUDA streams for overlapped execution
    cudaStream_t point_stream_;
    cudaStream_t memory_stream_;
    cudaStream_t precompute_stream_;
    
    // Performance tracking
    mutable PointOperationMetrics metrics_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    
    // Internal memory management
    bool allocate_device_memory(size_t max_points, size_t max_table_size = 0);
    void free_device_memory();
    cudaError_t copy_points_to_device(const std::vector<ProjectivePoint>& host_points,
                                     ProjectivePoint* device_ptr);
    cudaError_t copy_points_from_device(ProjectivePoint* device_ptr,
                                       std::vector<ProjectivePoint>& host_points,
                                       size_t count);
    
    // Launch parameter optimization
    dim3 calculate_grid_size_for_points(size_t point_count);
    dim3 calculate_block_size_for_points();
    
    // Precomputation utilities
    void generate_precomputed_table_cpu(const ProjectivePoint& base_point,
                                       int window_size,
                                       std::vector<ProjectivePoint>& table);
    void generate_precomputed_table_gpu(const ProjectivePoint& base_point,
                                       int window_size,
                                       ProjectivePoint* device_table);
};

/**
 * @brief Specialized algorithms for common secp256k1 operations
 */
namespace specialized {
    
    /**
     * @brief GLV endomorphism optimization for secp256k1
     * Exploits the special structure of secp256k1 to split scalar multiplication
     * into two smaller operations, approximately halving the computation time.
     */
    class GLVEndomorphism {
    public:
        // GLV decomposition: k = k1 + k2*lambda (mod n)
        struct GLVDecomposition {
            BigInt256 k1;
            BigInt256 k2;
            bool k1_negative;
            bool k2_negative;
        };
        
        static GLVDecomposition decompose_scalar(const BigInt256& scalar);
        
        __device__ static void point_multiply_glv(ProjectivePoint* result,
                                                 const GLVDecomposition* decomp,
                                                 const ProjectivePoint* point);
        
        __global__ static void batch_scalar_multiply_glv(ProjectivePoint* results,
                                                        const BigInt256* scalars,
                                                        const ProjectivePoint* points,
                                                        size_t count);
    private:
        static const BigInt256 lambda_;       // GLV lambda constant
        static const BigInt256 beta_;         // GLV beta constant  
        static const BigInt256 a1_, a2_, b1_, b2_;  // GLV basis vectors
    };
    
    /**
     * @brief Montgomery ladder for scalar multiplication
     * Provides resistance against side-channel attacks and uniform operation count
     */
    class MontgomeryLadder {
    public:
        __device__ static void scalar_multiply_ladder(ProjectivePoint* result,
                                                     const BigInt256* scalar,
                                                     const ProjectivePoint* point);
        
        __global__ static void batch_scalar_multiply_ladder(ProjectivePoint* results,
                                                           const BigInt256* scalars,
                                                           const ProjectivePoint* points,
                                                           size_t count);
    };
    
    /**
     * @brief Windowed Non-Adjacent Form (wNAF) for scalar multiplication
     * Reduces the average Hamming weight of the scalar representation
     */
    class WindowedNAF {
    public:
        struct NAFForm {
            std::vector<int8_t> digits;  // Non-adjacent form digits
            size_t length;
        };
        
        static NAFForm compute_wnaf(const BigInt256& scalar, int window_size);
        
        __device__ static void scalar_multiply_wnaf(ProjectivePoint* result,
                                                   const int8_t* naf_digits,
                                                   size_t naf_length,
                                                   const ProjectivePoint* precomputed_table,
                                                   int window_size);
        
        __global__ static void batch_scalar_multiply_wnaf(ProjectivePoint* results,
                                                         const BigInt256* scalars,
                                                         const ProjectivePoint* base_point,
                                                         int window_size,
                                                         size_t count);
    };
    
    /**
     * @brief Simultaneous multiple point multiplication (Strauss-Shamir)
     * Efficiently compute k1*P1 + k2*P2 + ... + kn*Pn
     */
    class MultiPointMultiplication {
    public:
        __device__ static void multi_scalar_multiply(ProjectivePoint* result,
                                                    const BigInt256* scalars,
                                                    const ProjectivePoint* points,
                                                    size_t num_points);
        
        __global__ static void batch_multi_scalar_multiply(ProjectivePoint* results,
                                                          const BigInt256* scalar_arrays,
                                                          const ProjectivePoint* point_arrays,
                                                          size_t points_per_mult,
                                                          size_t batch_size);
    };
}

/**
 * @brief Point validation and testing utilities
 */
class PointOperationValidator {
public:
    PointOperationValidator();
    ~PointOperationValidator();
    
    // Correctness validation
    bool validate_point_operations(size_t test_count = 1000);
    bool validate_scalar_multiplication(size_t test_count = 1000);
    bool validate_precomputed_tables(size_t test_count = 100);
    
    // Compare with CPU reference implementation
    bool compare_with_cpu_reference(const OptimizedPointOperations& gpu_impl,
                                   size_t test_count = 500);
    
    // Mathematical property validation
    bool validate_group_properties(size_t test_count = 1000);
    bool validate_distributive_property(size_t test_count = 500);
    bool validate_associative_property(size_t test_count = 500);
    
    // Performance regression testing
    bool run_performance_regression_tests();
    void save_performance_baseline(const std::string& filename);
    
    struct ValidationResults {
        bool correctness_passed;
        bool performance_passed;
        double gpu_speedup_factor;
        size_t tests_performed;
        std::chrono::milliseconds validation_time;
    };
    
    ValidationResults run_comprehensive_validation();

private:
    std::unique_ptr<OptimizedPointOperations> gpu_impl_;
    std::unique_ptr<cpu::Secp256k1> cpu_reference_;
    
    // Helper methods
    std::vector<ProjectivePoint> generate_random_points(size_t count);
    std::vector<BigInt256> generate_random_scalars(size_t count);
    bool points_equal(const ProjectivePoint& p1, const ProjectivePoint& p2, double tolerance = 1e-10);
};

/**
 * @brief Global registry for optimized point operations
 */
class PointOperationRegistry {
public:
    static OptimizedPointOperations* get_instance(int device_id = 0);
    static void set_global_optimization_config(const PointOptimizationConfig& config);
    static void cleanup_all_instances();
    
    // Performance monitoring across all instances
    static std::vector<OptimizedPointOperations::PointOperationMetrics> get_all_metrics();
    static void reset_all_performance_counters();

private:
    static std::unordered_map<int, std::unique_ptr<OptimizedPointOperations>> instances_;
    static PointOptimizationConfig global_config_;
    static std::mutex registry_mutex_;
};

} // namespace optimized
} // namespace gpu
} // namespace ecc
} // namespace keyhunt