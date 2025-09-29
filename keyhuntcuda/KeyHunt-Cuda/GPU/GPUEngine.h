/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
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

#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include "../SECP256k1.h"
#include "SearchMode.h"

#define SEARCH_COMPRESSED 0
#define SEARCH_UNCOMPRESSED 1
#define SEARCH_BOTH 2

// operating mode
#define SEARCH_MODE_MA 1	// multiple addresses
#define SEARCH_MODE_SA 2	// single address
#define SEARCH_MODE_MX 3	// multiple xpoints
#define SEARCH_MODE_SX 4	// single xpoint
#define SEARCH_MODE_PUZZLE71 7	// Bitcoin Puzzle #71 specialized mode

#define COIN_BTC 1
#define COIN_ETH 2

// Number of key per thread (must be a multiple of GRP_SIZE) per kernel call
#define STEP_SIZE (1024*2)

// Number of thread per block
#define ITEM_SIZE_A 28
#define ITEM_SIZE_A32 (ITEM_SIZE_A/4)

#define ITEM_SIZE_X 40
#define ITEM_SIZE_X32 (ITEM_SIZE_X/4)

typedef struct {
	uint32_t thId;
	int16_t  incr;
	uint8_t* hash;
	bool mode;
} ITEM;

class GPUEngine
{

public:

    GPUEngine(Secp256K1* secp, int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
        int searchMode, int compMode, int coinType, int64_t BLOOM_SIZE, uint64_t BLOOM_BITS,
        uint8_t BLOOM_HASHES, const uint8_t* BLOOM_DATA, bool rKey);

	GPUEngine(Secp256K1* secp, int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound, 
		int searchMode, int compMode, int coinType, const uint32_t* hashORxpoint, bool rKey);

	~GPUEngine();

	bool SetKeys(Point* p);

	bool LaunchSEARCH_MODE_MA(std::vector<ITEM>& dataFound, bool spinWait = false);
	bool LaunchSEARCH_MODE_SA(std::vector<ITEM>& dataFound, bool spinWait = false);
	bool LaunchSEARCH_MODE_MX(std::vector<ITEM>& dataFound, bool spinWait = false);
	bool LaunchSEARCH_MODE_SX(std::vector<ITEM>& dataFound, bool spinWait = false);
	bool LaunchPUZZLE71(std::vector<ITEM>& dataFound, bool spinWait = false);

	int GetNbThread();
	int GetGroupSize();

	// Template function to encapsulate common kernel call pattern
	template<typename KernelFunc>
	bool callKernelWithErrorCheck(KernelFunc kernelFunc, bool resetFoundFlag = false);

	// Getter methods for unified kernel interface
	int getNbThread() const { return nbThread; }
	int getNbThreadPerGroup() const { return nbThreadPerGroup; }
	uint32_t getCompMode() const { return compMode; }
	uint32_t getSearchMode() const { return searchMode; }
	uint32_t getCoinType() const { return coinType; }
	uint32_t getMaxFound() const { return maxFound; }
	uint64_t getBloomBits() const { return BLOOM_BITS; }
	uint8_t getBloomHashes() const { return BLOOM_HASHES; }
	uint32_t* getInputHashORxpoint() const { return inputHashORxpoint; }
	uint8_t* getInputBloomLookUp() const { return inputBloomLookUp; }
	uint64_t* getInputKey() const { return inputKey; }
	uint32_t* getOutputBuffer() const { return outputBuffer; }

	//bool Check(Secp256K1 *secp);
	std::string deviceName;

	static void PrintCudaInfo();
	static void GenerateCode(Secp256K1* secp, int size);

protected:
	void InitGenratorTable(Secp256K1* secp);

	bool callKernelSEARCH_MODE_MA();
	bool callKernelSEARCH_MODE_SA();
	bool callKernelSEARCH_MODE_MX();
	bool callKernelSEARCH_MODE_SX();
	bool callKernelPUZZLE71();

	// Template function to encapsulate common launch pattern
    template<typename KernelFunc>
    bool launchUnified(std::vector<ITEM>& dataFound, bool spinWait, KernelFunc kernelFunc,
                       int itemSize, int itemSize32);

public:
	int nbThread;
	int nbThreadPerGroup;

	uint32_t* inputHashORxpoint;
	uint32_t* inputHashORxpointPinned;

	//uint8_t *bloomLookUp;
	uint8_t* inputBloomLookUp;
	uint8_t* inputBloomLookUpPinned;

	uint64_t* inputKey;
	uint64_t* inputKeyPinned;

	uint32_t* outputBuffer;
	uint32_t* outputBufferPinned;

	uint64_t* __2Gnx;
	uint64_t* __2Gny;

	uint64_t* _Gx;
	uint64_t* _Gy;

	bool initialised;
	uint32_t compMode;
	uint32_t searchMode;
	uint32_t coinType;
	bool littleEndian;

	bool rKey;
	uint32_t maxFound;
	uint32_t outputSize;

	int64_t BLOOM_SIZE;
	uint64_t BLOOM_BITS;
	uint8_t BLOOM_HASHES;

};

#endif // GPUENGINEH
