/*
 * This file is part of the KeyHunt-Cuda distribution.
 * Copyright (c) 2025 Your Name.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "../cuda_fix.h"
#include <cuda_runtime.h>
#include <stdint.h>

// SHA256 K constants (first 32 bits of fractional parts of cube roots of first 64 primes)
__device__ __constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA256 Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
__device__ __constant__ uint32_t I[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// RIPEMD-160 constants
__device__ __constant__ uint64_t ripemd160_sizedesc_32 = 0x0000008000000000ULL;

// Keccak-f round constants
__device__ __constant__ uint64_t _KECCAKF_RNDC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000080008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Global device variables definitions
__device__ uint64_t* _2Gnx = NULL;
__device__ uint64_t* _2Gny = NULL;
__device__ uint64_t* Gx = NULL;
__device__ uint64_t* Gy = NULL;
__device__ int found_flag = 0;

// PUZZLE71 target hash definition
__device__ __constant__ uint32_t PUZZLE71_TARGET_HASH[5] = {
    0x225b45f8,  // Bytes 0-3 (little-endian)
    0x409a46fa,  // Bytes 4-7
    0xd3504465,  // Bytes 8-11
    0x3b9a9563,  // Bytes 12-15
    0xb4242993   // Bytes 16-19
};

// Endomorphism constants definitions
namespace EndomorphismConstants {
    __device__ __constant__ uint64_t LAMBDA[4] = {
        0x1b23bd7200000000ULL,
        0xdf02967c00000000ULL,
        0x20816678ea000000ULL,
        0x5363ad4cc05c30e0ULL
    };
    
    __device__ __constant__ uint64_t BETA[4] = {
        0x719501ee00000000ULL,
        0xc1396c2800000000ULL,
        0x12f5899500000000ULL,
        0x7ae96a2b657c0710ULL
    };
    
    __device__ __constant__ uint64_t A1[2] = {
        0x9284eb1500000000ULL,
        0x3086d221a7d46bcdULL
    };
    
    __device__ __constant__ uint64_t B1[2] = {
        0xf5480256df6ff74fULL,
        0x1bbc81296f922df7ULL
    };
    
    __device__ __constant__ uint64_t A2[2] = {
        0x9d44cfd800000000ULL,
        0x114ca50f7a8e2f3fULL
    };
    
    __device__ __constant__ uint64_t B2[2] = {
        0x9284eb1500000000ULL,
        0x3086d221a7d46bcdULL
    };
    
    __device__ __constant__ uint64_t N_HALF[4] = {
        0xD0364140F8A0BB68ULL,
        0xBFD25E8C00000000ULL,
        0xAAAAAAA900000000ULL,
        0x7FFFFFFFFFFFFFFFULL
    };
}
