/*
 * This file is part of the KeyHunt-Cuda distribution (https://github.com/your-repo/keyhunt-cuda).
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

#ifndef SEARCH_MODE_H
#define SEARCH_MODE_H

// Search mode enumeration for unified GPU kernel interface
enum class SearchMode : uint32_t {
    MODE_MA = 1,      // Multiple addresses
    MODE_SA = 2,      // Single address
    MODE_MX = 3,      // Multiple X-points
    MODE_SX = 4,      // Single X-point
    MODE_ETH_MA = 5,  // Ethereum multiple addresses
    MODE_ETH_SA = 6,  // Ethereum single address
    PUZZLE71 = 7      // Specialized mode for Bitcoin Puzzle #71
};

// Compression mode enumeration
enum class CompressionMode : uint32_t {
    COMPRESSED = 0,   // Compressed public key (33 bytes)
    UNCOMPRESSED = 1, // Uncompressed public key (65 bytes)
    BOTH = 2          // Both compressed and uncompressed
};

// Coin type enumeration
enum class CoinType : uint32_t {
    BITCOIN = 0,      // Bitcoin
    ETHEREUM = 1      // Ethereum
};

#endif // SEARCH_MODE_H