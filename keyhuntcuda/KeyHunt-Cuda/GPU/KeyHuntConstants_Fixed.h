/**
 * KeyHuntConstants.h
 * Critical constants that are missing from the codebase
 * Required for PUZZLE71 implementation to compile
 */

#ifndef KEYHUNT_CONSTANTS_H
#define KEYHUNT_CONSTANTS_H

namespace KeyHuntConstants {
    // Elliptic curve group size for batch processing
    // This defines how many points we process in each group
    static const int ELLIPTIC_CURVE_GROUP_SIZE_KEYHUNT = 1024;
    static const int ELLIPTIC_CURVE_HALF_GROUP_SIZE_KEYHUNT = 512;  // 1024 / 2
    
    // Item sizes for result storage (fixed for PUZZLE71 compatibility)
    static const int ITEM_SIZE_A = 256;      // Size in bytes (larger buffer for safety)
    static const int ITEM_SIZE_A32 = 20;     // Size in 32-bit words (80 bytes / 4)
    
    // PUZZLE71 specific constants
    static const uint64_t PUZZLE71_RANGE_START = 0x40000000000000ULL;  // 2^70
    static const uint64_t PUZZLE71_RANGE_END   = 0x7FFFFFFFFFFFFFFFULL;   // 2^71 - 1
}

// Global device variable for found flag
#ifdef __CUDA_ARCH__
__device__ uint32_t found_flag = 0;
#endif

// Missing function implementations that need to be added
#define _ModInvGrouped(dx) ModInvGrouped(dx, KeyHuntConstants::ELLIPTIC_CURVE_HALF_GROUP_SIZE_KEYHUNT + 2)

#endif // KEYHUNT_CONSTANTS_H