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

// Search mode enumeration
enum class SearchMode {
    MODE_MA,      // Multi-address mode
    MODE_SA,      // Single address mode
    MODE_MX,      // Multi-xpoint mode
    MODE_SX,      // Single xpoint mode
    MODE_ETH_MA,  // Ethereum multi-address mode
    MODE_ETH_SA   // Ethereum single address mode
};

// Compression mode enumeration
enum class CompressionMode {
    COMPRESSED,
    UNCOMPRESSED,
    BOTH
};

// Coin type enumeration
enum class CoinType {
    BITCOIN,
    ETHEREUM
};

#endif // SEARCH_MODE_H