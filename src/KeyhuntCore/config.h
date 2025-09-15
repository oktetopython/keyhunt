// Keyhunt-CUDA Project Configuration
// Scientific Research System for Bitcoin Puzzle Challenges

#pragma once

#ifndef KEYHUNT_CONFIG_H
#define KEYHUNT_CONFIG_H

// Version Information
#define KEYHUNT_VERSION_MAJOR 0
#define KEYHUNT_VERSION_MINOR 1
#define KEYHUNT_VERSION_PATCH 0
#define KEYHUNT_VERSION "0.1.0"

// CUDA Configuration
#define MIN_COMPUTE_CAPABILITY 75  // Turing architecture minimum
#define CUDA_VERSION_REQUIRED 11000  // CUDA 11.0+

// Performance Targets
#define TARGET_KEYS_PER_SECOND_TURING 1000000000ULL   // 1000M keys/s
#define TARGET_KEYS_PER_SECOND_HOPPER 4000000000ULL   // 4000M keys/s
#define TARGET_GPU_UTILIZATION 90                     // >90% utilization

// Scientific Validation
#define PRECISION_THRESHOLD 1e-10  // <1e-10 relative error
#define VALIDATION_SAMPLE_SIZE 100000  // Default validation samples

// Memory Configuration
#define DEFAULT_BATCH_SIZE 1000000
#define MAX_GPU_MEMORY_USAGE 0.9f  // Use up to 90% of GPU memory

// Paths
#define CONFIG_DIR "data/config"
#define CHECKPOINT_DIR "data/checkpoint"  
#define LOGS_DIR "data/logs"
#define RESULTS_DIR "data/results"

// File Names
#define CONFIG_FILE "config.txt"
#define PRIVATE_RANGES_FILE "private_ranges.txt"
#define TARGET_ADDRESSES_FILE "target_addresses.txt"
#define CHECKPOINT_FILE "checkpoint.dat"

#endif // KEYHUNT_CONFIG_H