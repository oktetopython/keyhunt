# Data Model: Keyhunt-CUDA Scientific Research System

**Date**: 2025-09-13  
**Phase**: 1 - Design & Contracts  
**Status**: Complete

## Overview
Data model design for GPU-accelerated Bitcoin private key scanning system, extracted from feature specification requirements and entities.

## Core Entities

### Private Key Range
**Purpose**: Defines search space for systematic Bitcoin private key scanning  
**Source**: FR-001, Key Entities specification

**Attributes**:
- `range_id`: string - Unique identifier for tracking and resumption
- `start_key`: string(64) - Starting private key in hexadecimal format
- `end_key`: string(64) - Ending private key in hexadecimal format
- `stride`: uint64 - Step size between keys (default: 1, minimum: 1)
- `total_keys`: uint64 - Calculated total keys in range
- `status`: enum - Current state (CONFIGURED, SCANNING, PAUSED, COMPLETED, FAILED, CANCELLED)
- `gpu_devices`: int[] - Assigned GPU device IDs for processing
- `estimated_time`: double - Expected completion time in seconds
- `keys_processed`: uint64 - Progress tracking counter
- `progress_percentage`: double - Calculated progress (0-100%)
- `keys_per_second`: double - Current processing rate
- `created_at`: timestamp - Range creation time
- `updated_at`: timestamp - Last modification time

**Validation Rules**:
- start_key and end_key must match regex: `^[0-9a-fA-F]{64}$`
- end_key must be numerically greater than start_key
- stride must be ≥ 1
- gpu_devices must contain valid device IDs (≥ 0)
- progress_percentage must be 0-100%

**State Transitions**:
```
CONFIGURED → SCANNING → {COMPLETED, FAILED, CANCELLED}
SCANNING → PAUSED → SCANNING
PAUSED → {CANCELLED, SCANNING}
```

### Target Address
**Purpose**: Bitcoin addresses serving as match criteria for search operations  
**Source**: FR-003, FR-011, Key Entities specification

**Attributes**:
- `address_id`: string - Unique identifier for address
- `address`: string - Bitcoin address in Base58 format
- `address_type`: enum - Address format (P2PKH, P2SH, P2WPKH, P2WSH, P2TR)
- `hash160`: bytes(20) - RIPEMD160 hash for comparison optimization
- `description`: string - Optional annotation for research purposes
- `priority`: enum - Search priority (HIGH, MEDIUM, LOW)
- `status`: enum - Match status (PENDING, FOUND, EXCLUDED)
- `found_private_key`: string(64) - Discovered private key (if found)
- `found_at`: timestamp - Discovery timestamp
- `verification_status`: enum - CPU validation status (UNVERIFIED, VERIFIED, FAILED)

**Validation Rules**:
- address must be valid Base58Check format
- hash160 must be exactly 20 bytes
- found_private_key must match hexadecimal format when present
- verification_status required when found_private_key present

### Checkpoint Data
**Purpose**: Progress state information enabling resumption of interrupted scans  
**Source**: FR-006, Key Entities specification

**Attributes**:
- `checkpoint_id`: string - Unique checkpoint identifier
- `range_id`: string - Associated private key range
- `current_position`: string(64) - Last processed private key
- `keys_processed`: uint64 - Total keys processed at checkpoint
- `gpu_states`: object[] - Per-GPU state information
  - `device_id`: int - GPU device identifier
  - `current_key`: string(64) - GPU-specific current position
  - `processed_count`: uint64 - GPU-specific processed keys
  - `memory_state`: bytes - GPU memory dump (if required)
- `performance_metrics`: object - Performance data at checkpoint
  - `keys_per_second`: double - Processing rate
  - `gpu_utilization`: double[] - Per-GPU utilization percentages
  - `memory_usage`: double[] - Per-GPU memory usage (MB)
- `created_at`: timestamp - Checkpoint creation time
- `resume_count`: int - Number of times resumed from this checkpoint
- `validation_hash`: string - Data integrity verification

**Validation Rules**:
- current_position must be within range bounds (start_key ≤ current_position ≤ end_key)
- keys_processed must be consistent with current_position and stride
- gpu_states array must match configured GPU devices
- validation_hash must verify data integrity

### Experimental Results
**Purpose**: Comprehensive data collection for scientific analysis and validation  
**Source**: FR-010, Key Entities specification

**Attributes**:
- `experiment_id`: string - Unique experiment identifier
- `range_id`: string - Associated private key range
- `experiment_type`: enum - Type of experiment (SEARCH, VALIDATION, BENCHMARK)
- `start_time`: timestamp - Experiment start time
- `end_time`: timestamp - Experiment completion time
- `duration_seconds`: double - Total execution time
- `total_keys_processed`: uint64 - Keys processed during experiment
- `matches_found`: int - Number of successful matches
- `performance_metrics`: object - Detailed performance measurements
  - `peak_keys_per_second`: double - Maximum processing rate
  - `average_keys_per_second`: double - Mean processing rate
  - `gpu_utilization_stats`: object[] - Per-GPU utilization statistics
  - `memory_usage_stats`: object[] - Per-GPU memory usage statistics
  - `error_count`: int - Number of errors encountered
- `validation_results`: object - CPU/GPU consistency validation
  - `samples_validated`: int - Number of validation samples
  - `max_error`: double - Maximum observed error
  - `precision_achieved`: double - Achieved precision level
- `configuration_snapshot`: object - System configuration at experiment time
- `raw_data_location`: string - Path to detailed raw data files

**Validation Rules**:
- duration_seconds must match end_time - start_time
- performance_metrics must be non-negative
- validation_results.precision_achieved must be ≤ 1e-10 for scientific compliance
- matches_found must be ≤ total_keys_processed

### GPU Configuration
**Purpose**: Device specifications and performance optimization settings  
**Source**: FR-008, Key Entities specification

**Attributes**:
- `config_id`: string - Unique configuration identifier
- `device_id`: int - CUDA device identifier
- `device_name`: string - GPU model name (e.g., "GeForce RTX 3090")
- `compute_capability`: string - CUDA compute capability (e.g., "8.6")
- `total_memory_mb`: int - Total device memory in megabytes
- `available_memory_mb`: int - Available memory for processing
- `multiprocessor_count`: int - Number of streaming multiprocessors
- `max_threads_per_block`: int - Maximum threads per block
- `max_blocks_per_grid`: int - Maximum blocks per grid
- `optimization_settings`: object - Performance tuning parameters
  - `threads_per_block`: int - Configured threads per block
  - `blocks_per_grid`: int - Configured blocks per grid
  - `memory_allocation_mb`: int - Allocated memory for processing
  - `stream_count`: int - Number of CUDA streams
- `performance_profile`: object - Benchmarked performance characteristics
  - `keys_per_second_baseline`: double - Baseline performance measurement
  - `memory_bandwidth_gbps`: double - Measured memory bandwidth
  - `utilization_target`: double - Target utilization percentage
- `status`: enum - Device status (AVAILABLE, IN_USE, ERROR, DISABLED)
- `last_updated`: timestamp - Configuration update time

**Validation Rules**:
- device_id must be ≥ 0 and correspond to valid CUDA device
- memory allocations must not exceed available_memory_mb
- optimization_settings must respect device limits
- performance_profile values must be positive

### Validation Report
**Purpose**: Scientific accuracy verification comparing GPU vs CPU computations  
**Source**: FR-007, Key Entities specification

**Attributes**:
- `report_id`: string - Unique validation report identifier
- `validation_type`: enum - Type of validation (ECC_OPERATIONS, ADDRESS_GENERATION, FULL_PIPELINE)
- `test_timestamp`: timestamp - When validation was performed
- `sample_count`: int - Number of test samples processed
- `gpu_device_ids`: int[] - GPU devices included in validation
- `test_parameters`: object - Validation test configuration
  - `precision_threshold`: double - Required precision (default: 1e-10)
  - `timeout_seconds`: int - Maximum validation time
  - `sample_distribution`: enum - How samples were selected (RANDOM, SYSTEMATIC, TARGETED)
- `results`: object - Validation outcomes
  - `tests_passed`: int - Number of successful validations
  - `tests_failed`: int - Number of failed validations
  - `max_observed_error`: double - Maximum error magnitude
  - `mean_error`: double - Average error across all samples
  - `standard_deviation`: double - Error distribution statistics
- `failure_details`: object[] - Details of any validation failures
  - `sample_input`: string - Input that caused failure
  - `gpu_result`: string - GPU computation result
  - `cpu_result`: string - CPU reference result
  - `error_magnitude`: double - Specific error value
- `performance_impact`: object - Validation overhead measurements
  - `validation_time_seconds`: double - Time spent on validation
  - `performance_degradation_percent`: double - Impact on processing speed
- `certification_status`: enum - Overall validation result (PASSED, FAILED, MARGINAL)

**Validation Rules**:
- sample_count must be > 0
- precision_threshold must be ≤ 1e-10 for scientific compliance
- tests_passed + tests_failed must equal sample_count
- max_observed_error must be ≤ precision_threshold for PASSED certification
- certification_status must be PASSED for production use

## Entity Relationships

### Primary Relationships
```
Private Key Range 1:N Checkpoint Data (range_id)
Private Key Range 1:N Experimental Results (range_id)
GPU Configuration 1:N Experimental Results (through device assignments)
Experimental Results 1:N Validation Report (experiment validation)
Target Address 1:N Experimental Results (through match results)
```

### Data Flow Relationships
```
Configuration → Private Key Range → Checkpoint Data → Resumption
Private Key Range → GPU Configuration → Processing → Experimental Results
Processing Results → Target Address Matching → Validation Report
Validation Report → Scientific Certification → Publication
```

## Data Storage Requirements

### Performance Requirements
- Checkpoint creation: < 1 second
- Checkpoint recovery: < 10 seconds  
- Real-time metrics update: < 100ms
- Validation report generation: < 5 minutes

### Persistence Requirements
- Checkpoint data: Binary format for performance
- Configuration data: JSON format for human readability
- Experimental results: Structured format for analysis tools
- Validation reports: Archival format for scientific record

### Scalability Requirements
- Support ranges up to 2^256 private keys
- Multi-GPU configurations (8+ devices)
- Experimental data retention for 1+ year
- Concurrent validation across multiple experiments

## Validation and Integrity

### Data Integrity
- Checksum validation for all persisted data
- Atomic updates for critical state changes
- Transaction-like behavior for checkpoint operations
- Backup and recovery procedures for research data

### Scientific Validation
- All computations validated against libsecp256k1 CPU reference
- Audit trail for all data modifications
- Reproducible results through deterministic algorithms
- Peer review compatibility for research publication

## Implementation Notes

### Already Implemented
- ✅ Private Key Range: Complete implementation with 12 passing unit tests
- ✅ Basic validation framework established
- ✅ JSON serialization support
- ✅ Progress tracking mechanisms

### Next Implementation Priority
1. Target Address entity with Bitcoin address validation
2. Checkpoint Data with binary serialization
3. GPU Configuration with device detection
4. Experimental Results with performance metrics
5. Validation Report with scientific compliance

This data model provides the foundation for the scientific research system while maintaining performance, accuracy, and scalability requirements.