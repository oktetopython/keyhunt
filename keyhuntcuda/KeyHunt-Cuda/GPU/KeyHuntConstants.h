/**
 * KeyHuntConstants.h
 * Critical constants that are missing from the codebase
 * Required for PUZZLE71 implementation to compile
 */

#ifndef KEYHUNT_CONSTANTS_H
#define KEYHUNT_CONSTANTS_H

// Elliptic curve group size for batch processing
// This defines how many points we process in each group
#define ELLIPTIC_CURVE_GROUP_SIZE_KEYHUNT 1024
#define ELLIPTIC_CURVE_HALF_GROUP_SIZE_KEYHUNT 512  // 1024 / 2

// Item sizes for result storage (fixed for PUZZLE71 compatibility)
// Use different names to avoid conflicts with existing definitions
#define KEYHUNT_ITEM_SIZE_A 256      // Size in bytes (larger buffer for safety)
#define KEYHUNT_ITEM_SIZE_A32 20     // Size in 32-bit words (80 bytes / 4)

// PUZZLE71 specific constants
#define PUZZLE71_RANGE_START 0x40000000000000ULL  // 2^70
#define PUZZLE71_RANGE_END   0x7FFFFFFFFFFFFFFFULL   // 2^71 - 1

// Namespace removed to avoid CUDA compilation issues
// All constants are now preprocessor defines above

// Global device variable for found flag
// Note: This is declared elsewhere, so we just provide a declaration here
#ifdef __CUDA_ARCH__
extern __device__ int found_flag;
#endif

// Missing function implementations that need to be added
#define _ModInvGrouped(dx) ModInvGrouped(dx, ELLIPTIC_CURVE_HALF_GROUP_SIZE_KEYHUNT + 2)

#endif // KEYHUNT_CONSTANTS_H