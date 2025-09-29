#include "KeyHunt.h"
#include "Constants.h"
#include "GmpUtil.h"
#include "Base58.h"
#include "hash/sha256.h"
#include "hash/keccak160.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <inttypes.h>
#include <memory>
#include <stdexcept>
#include <vector>
#ifndef WIN64
#include <pthread.h>
#endif

//using namespace std;

#define MIN(a,b) ((a)<(b)?(a):(b))

// ----------------------------------------------------------------------------

KeyHunt::KeyHunt(const std::string& inputFile,
                 int compMode,
                 int searchMode,
                 int coinType,
                 bool useGpu,
                 const std::string& outputFile,
                 uint32_t maxFound,
                 uint64_t rKey,
                 const std::string& rangeStart,
                 const std::string& rangeEnd,
                 bool& should_exit)
{
    secp = nullptr;
    bloom = nullptr;

    this->compMode = compMode;
    this->useGpu = useGpu;
    this->outputFile = outputFile;
    this->nbGPUThread = 0;
    this->inputFile = inputFile;
    this->maxFound = maxFound;
    this->rKey = rKey;
    this->searchMode = searchMode;
    this->coinType = coinType;
    this->rangeStart.SetBase16(rangeStart.c_str());
    this->rangeEnd.SetBase16(rangeEnd.c_str());
    this->rangeDiff2.Set(&this->rangeEnd);
    this->rangeDiff2.Sub(&this->rangeStart);
    this->lastrKey = 0;

    if (!this->useGpu) {
        throw std::runtime_error("GPU execution is mandatory in GPU-only build");
    }

	secp = new Secp256K1();
	
	// Enable fast initialization for PUZZLE71 mode
	if (this->searchMode == SEARCH_MODE_PUZZLE71) {
		printf("[KeyHunt] Enabling fast init for PUZZLE71 mode (file constructor)\n"); fflush(stdout);
		secp->SetFastInit(true);
	}
	
	secp->Init();

	// load file
	uint64_t N = 0;

	// 使用RAII文件处理，确保文件正确关闭
	FILE* wfd = fopen(this->inputFile.c_str(), "rb");
	FileGuard fileGuard(wfd);
	
	if (!fileGuard.get()) {
		printf("%s can not open\n", this->inputFile.c_str());
		exit(1);
	}

#ifdef WIN64
	_fseeki64(fileGuard.get(), 0, SEEK_END);
	N = _ftelli64(fileGuard.get());
#else
	fseek(fileGuard.get(), 0, SEEK_END);
	N = ftell(fileGuard.get());
#endif

	int K_LENGTH = 20;
	if (this->searchMode == (int)SEARCH_MODE_MX)
		K_LENGTH = 32;

	N = N / K_LENGTH;
	rewind(fileGuard.get());

	auto buf = std::make_unique<uint8_t[]>(K_LENGTH);

	bloom = new Bloom(2 * N, 0.000001);

	uint64_t percent = (N - 1) / 100;
	uint64_t i = 0;
	printf("\n");
	while (i < N && !should_exit) {
		memset(buf.get(), 0, K_LENGTH);
		if (fread(buf.get(), 1, K_LENGTH, fileGuard.get()) == K_LENGTH) {
			bloom->add(buf.get(), K_LENGTH);
			if ((percent != 0) && i % percent == 0) {
				printf("\rLoading      : %" PRIu64 " %%", (i / percent));
				fflush(stdout);
			}
		}
		i++;
	}
	// fileGuard会在析构时自动关闭文件

	if (should_exit) {
		delete secp;
		delete bloom;
		exit(0);
	}

	targetCounter = i;
	if (coinType == COIN_BTC) {
		if (searchMode == (int)SEARCH_MODE_MA)
			printf("Loaded       : %s Bitcoin addresses\n", formatThousands(i).c_str());
		else if (searchMode == (int)SEARCH_MODE_MX)
			printf("Loaded       : %s Bitcoin xpoints\n", formatThousands(i).c_str());
	}
	else {
		printf("Loaded       : %s Ethereum addresses\n", formatThousands(i).c_str());
	}

    printf("\n");

    bloom->print();
    printf("\n");

}

// ----------------------------------------------------------------------------

KeyHunt::KeyHunt(const std::vector<unsigned char>& hashORxpoint,
                 int compMode,
                 int searchMode,
                 int coinType,
                 bool useGpu,
                 const std::string& outputFile,
                 uint32_t maxFound,
                 uint64_t rKey,
                 const std::string& rangeStart,
                 const std::string& rangeEnd,
                 bool& should_exit)
{
	printf("[KeyHunt] Constructor called, searchMode=%d\n", searchMode);
	fflush(stdout);
	// 初始化所有指针为nullptr，防止空指针解引用
	secp = nullptr;
	bloom = nullptr;
	
	this->compMode = compMode;
	this->useGpu = useGpu;
	this->outputFile = outputFile;
	this->nbGPUThread = 0;
	this->maxFound = maxFound;
	this->rKey = rKey;
	this->searchMode = searchMode;
	this->coinType = coinType;
	this->rangeStart.SetBase16(rangeStart.c_str());
	this->rangeEnd.SetBase16(rangeEnd.c_str());
	this->rangeDiff2.Set(&this->rangeEnd);
	this->rangeDiff2.Sub(&this->rangeStart);
	this->targetCounter = 1;

 if (!this->useGpu) {
     throw std::runtime_error("GPU execution is mandatory in GPU-only build");
 }

	printf("[KeyHunt] Creating Secp256K1...\n"); fflush(stdout);
	secp = new Secp256K1();
	
	// Enable fast initialization for PUZZLE71 mode
	if (this->searchMode == SEARCH_MODE_PUZZLE71) {
		printf("[KeyHunt] Enabling fast init for PUZZLE71 mode\n"); fflush(stdout);
		secp->SetFastInit(true);
	}
	
	printf("[KeyHunt] Calling secp->Init()...\n"); fflush(stdout);
	secp->Init();
	printf("[KeyHunt] Secp256K1 initialized\n"); fflush(stdout);

	if (this->searchMode == (int)SEARCH_MODE_SA) {
		assert(hashORxpoint.size() == 20);
		for (size_t i = 0; i < hashORxpoint.size(); i++) {
			((uint8_t*)hash160Keccak)[i] = hashORxpoint.at(i);
		}
	}
	else if (this->searchMode == (int)SEARCH_MODE_SX) {
		assert(hashORxpoint.size() == 32);
		for (size_t i = 0; i < hashORxpoint.size(); i++) {
			((uint8_t*)xpoint)[i] = hashORxpoint.at(i);
		}
	}
	else if (this->searchMode == SEARCH_MODE_PUZZLE71) {
		printf("[KeyHunt] PUZZLE71 mode constructor\n");
		// For PUZZLE71, we might get a dummy hash, but we'll use hardcoded target in GPU
		if (hashORxpoint.size() >= 20) {
			for (size_t i = 0; i < 20; i++) {
				((uint8_t*)hash160Keccak)[i] = hashORxpoint.at(i);
			}
		}
	}
	printf("\n");
    printf("[KeyHunt] Initialisation sequence completed\n");
}

KeyHunt::~KeyHunt()
{
	if (secp) {
		delete secp;
		secp = nullptr;
	}
	if (bloom) {
		delete bloom;
		bloom = nullptr;
	}
}

void KeyHunt::output(std::string addr, std::string pAddr, std::string pAddrHex, std::string pubKey)
{
    LockGuard lock(
#ifdef WIN64
        ghMutex
#else
        ghMutex
#endif
    );

    FILE* f = stdout;
    bool needToClose = false;

    if (outputFile.length() > 0) {
        f = fopen(outputFile.c_str(), "a");
        if (f == nullptr) {
            printf("Cannot open %s for writing\n", outputFile.c_str());
            f = stdout;
        } else {
            needToClose = true;
        }
    }

    FileGuard fileGuard(needToClose ? f : nullptr);
    FILE* outputFilePtr = needToClose ? fileGuard.get() : f;

    if (!needToClose) {
        printf("\n");
    }

    fprintf(outputFilePtr, "PubAddress: %s\n", addr.c_str());
    fprintf(stdout, "\n=================================================================================\n");
    fprintf(stdout, "PubAddress: %s\n", addr.c_str());

    if (coinType == COIN_BTC) {
        fprintf(outputFilePtr, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
        fprintf(stdout, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
    }

    fprintf(outputFilePtr, "Priv (HEX): %s\n", pAddrHex.c_str());
    fprintf(stdout, "Priv (HEX): %s\n", pAddrHex.c_str());

    fprintf(outputFilePtr, "PubK (HEX): %s\n", pubKey.c_str());
    fprintf(stdout, "PubK (HEX): %s\n", pubKey.c_str());

    fprintf(outputFilePtr, "=================================================================================\n");
    fprintf(stdout, "=================================================================================\n");
}

// ----------------------------------------------------------------------------

bool KeyHunt::checkPrivKey(std::string addr, Int& key, int32_t incr, bool mode)
{
	Int k(&key), k2(&key);
	k.Add((uint64_t)incr);
	k2.Add((uint64_t)incr);
	// Check addresses
	Point p = secp->ComputePublicKey(&k);
	std::string px = p.x.GetBase16();
	std::string chkAddr = secp->GetAddress(mode, p);
	if (chkAddr != addr) {
		//Key may be the opposite one (negative zero or compressed key)
		k.Neg();
		k.Add(&secp->order);
		p = secp->ComputePublicKey(&k);
		std::string chkAddr = secp->GetAddress(mode, p);
		if (chkAddr != addr) {
			printf("\n=================================================================================\n");
			printf("Warning, wrong private key generated !\n");
			printf("  PivK :%s\n", k2.GetBase16().c_str());
			printf("  Addr :%s\n", addr.c_str());
			printf("  PubX :%s\n", px.c_str());
			printf("  PivK :%s\n", k.GetBase16().c_str());
			printf("  Check:%s\n", chkAddr.c_str());
			printf("  PubX :%s\n", p.x.GetBase16().c_str());
			printf("=================================================================================\n");
			return false;
		}
	}
	output(addr, secp->GetPrivAddress(mode, k), k.GetBase16(), secp->GetPublicKeyHex(mode, p));
	return true;
}

bool KeyHunt::checkPrivKeyETH(std::string addr, Int& key, int32_t incr)
{
	Int k(&key), k2(&key);
	k.Add((uint64_t)incr);
	k2.Add((uint64_t)incr);
	// Check addresses
	Point p = secp->ComputePublicKey(&k);
	std::string px = p.x.GetBase16();
	std::string chkAddr = secp->GetAddressETH(p);
	if (chkAddr != addr) {
		//Key may be the opposite one (negative zero or compressed key)
		k.Neg();
		k.Add(&secp->order);
		p = secp->ComputePublicKey(&k);
		std::string chkAddr = secp->GetAddressETH(p);
		if (chkAddr != addr) {
			printf("\n=================================================================================\n");
			printf("Warning, wrong private key generated !\n");
			printf("  PivK :%s\n", k2.GetBase16().c_str());
			printf("  Addr :%s\n", addr.c_str());
			printf("  PubX :%s\n", px.c_str());
			printf("  PivK :%s\n", k.GetBase16().c_str());
			printf("  Check:%s\n", chkAddr.c_str());
			printf("  PubX :%s\n", p.x.GetBase16().c_str());
			printf("=================================================================================\n");
			return false;
		}
	}
	output(addr, k.GetBase16()/*secp->GetPrivAddressETH(k)*/, k.GetBase16(), secp->GetPublicKeyHexETH(p));
	return true;
}

bool KeyHunt::checkPrivKeyX(Int& key, int32_t incr, bool mode)
{
	Int k(&key);
	k.Add((uint64_t)incr);
	Point p = secp->ComputePublicKey(&k);
	std::string addr = secp->GetAddress(mode, p);
	output(addr, secp->GetPrivAddress(mode, k), k.GetBase16(), secp->GetPublicKeyHex(mode, p));
	return true;
}

// ----------------------------------------------------------------------------

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam)
{
#else
void* _FindKeyGPU(void* lpParam)
{
#endif
    GPUThreadParam* p = (GPUThreadParam*)lpParam;
	p->obj->FindKeyGPU(p);
	return 0;
}

// ----------------------------------------------------------------------------


void KeyHunt::getGPUStartingKeys(Int & tRangeStart, Int & tRangeEnd, int groupSize, int nbThread, Int * keys, Point * p){

	Int tRangeDiff(tRangeEnd);
	Int tRangeStart2(tRangeStart);
	Int tRangeEnd2(tRangeStart);

	Int tThreads;
	tThreads.SetInt32(nbThread);
	tRangeDiff.Set(&tRangeEnd);
	tRangeDiff.Sub(&tRangeStart);
	tRangeDiff.Div(&tThreads);

	int rangeShowThreasold = 3;
	int rangeShowCounter = 0;

	for (int i = 0; i < nbThread; i++) {

		tRangeEnd2.Set(&tRangeStart2);
		tRangeEnd2.Add(&tRangeDiff);

		if (rKey <= 0)
			keys[i].Set(&tRangeStart2);
		else
			keys[i].Rand(&tRangeEnd2);

		tRangeStart2.Add(&tRangeDiff);

		Int k(keys + i);
		k.Add((uint64_t)(groupSize / 2));	// Starting key is at the middle of the group
		
		// Always compute public keys even for PUZZLE71 (secp is minimally initialized)
		p[i] = secp->ComputePublicKey(&k);
	}

}

void KeyHunt::FindKeyGPU(GPUThreadParam* ph)
{
	printf("[GPU Thread %d] Starting GPU thread...\n", ph->threadId);
	bool ok = true;

#ifdef WITHGPU

	// Global init
	int thId = ph->threadId;
	Int tRangeStart = ph->rangeStart;
	Int tRangeEnd = ph->rangeEnd;
	printf("[GPU Thread %d] Initialized range variables\n", thId);

	printf("[GPU Thread %d] Creating GPUEngine for searchMode=%d\n", thId, searchMode);
	fflush(stdout);
	GPUEngine* g;
	switch (searchMode) {
	case (int)SEARCH_MODE_MA:
	case (int)SEARCH_MODE_MX:
		printf("[GPU Thread %d] Creating MA/MX GPUEngine...\n", thId);
		fflush(stdout);
		g = new GPUEngine(secp, ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, searchMode, compMode, coinType,
			bloom->get_bytes(), bloom->get_bits(), bloom->get_hashes(), bloom->get_bf(), (rKey != 0));
		printf("[GPU Thread %d] MA/MX GPUEngine created\n", thId);
		fflush(stdout);
		break;
	case (int)SEARCH_MODE_SA:
		printf("[GPU Thread %d] Creating SA GPUEngine...\n", thId);
		fflush(stdout);
		g = new GPUEngine(secp, ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, searchMode, compMode, coinType,
			hash160Keccak, (rKey != 0));
		printf("[GPU Thread %d] SA GPUEngine created\n", thId);
		fflush(stdout);
		break;
	case (int)SEARCH_MODE_SX:
		printf("[GPU Thread %d] Creating SX GPUEngine...\n", thId);
		fflush(stdout);
		g = new GPUEngine(secp, ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, searchMode, compMode, coinType,
			xpoint, (rKey != 0));
		printf("[GPU Thread %d] SX GPUEngine created\n", thId);
		fflush(stdout);
		break;
	case (int)SEARCH_MODE_PUZZLE71:
		printf("[GPU Thread %d] Creating PUZZLE71 GPUEngine with params: gridX=%d, gridY=%d, gpuId=%d\n", 
			thId, ph->gridSizeX, ph->gridSizeY, ph->gpuId);
		fflush(stdout);
		// PUZZLE71 mode uses the same constructor as SA mode
		g = new GPUEngine(secp, ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, searchMode, compMode, coinType,
			hash160Keccak, (rKey != 0));
		printf("[GPU Thread %d] PUZZLE71 GPUEngine created successfully\n", thId);
		fflush(stdout);
		break;
	default:
		printf("Invalid search mode format: %d\n", searchMode);
		fflush(stdout);
		return;
		break;
	}


	printf("[GPU Thread %d] Getting number of threads...\n", thId);
	fflush(stdout);
	int nbThread = g->GetNbThread();
	printf("[GPU Thread %d] Number of threads: %d\n", thId, nbThread);
	fflush(stdout);
	
	// 使用智能指针管理GPU内存，防止内存泄漏
	auto p = std::make_unique<Point[]>(nbThread);
	auto keys = std::make_unique<Int[]>(nbThread);
	std::vector<ITEM> found;

	printf("[GPU Thread %d] GPU Device: %s\n", thId, g->deviceName.c_str());
	printf("GPU          : %s\n\n", g->deviceName.c_str());
	fflush(stdout);

	counters[thId] = 0;

	printf("[GPU Thread %d] Getting GPU starting keys...\n", thId);
	fflush(stdout);
	getGPUStartingKeys(tRangeStart, tRangeEnd, g->GetGroupSize(), nbThread, keys.get(), p.get());
	
	printf("[GPU Thread %d] Setting keys in GPU...\n", thId);
	fflush(stdout);
	ok = g->SetKeys(p.get());
	printf("[GPU Thread %d] SetKeys returned: %s\n", thId, ok ? "true" : "false");
	fflush(stdout);

	ph->hasStarted = true;
	ph->rKeyRequest = false;
	printf("[GPU Thread %d] GPU thread started, entering main loop\n", thId);
	fflush(stdout);

	// GPU Thread
	while (ok && !endOfSearch) {

		if (ph->rKeyRequest) {
			getGPUStartingKeys(tRangeStart, tRangeEnd, g->GetGroupSize(), nbThread, keys.get(), p.get());
			ok = g->SetKeys(p.get());
			ph->rKeyRequest = false;
		}

		// Call kernel
		switch (searchMode) {
		case (int)SEARCH_MODE_MA:
			ok = g->LaunchSEARCH_MODE_MA(found, false);
			for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
				ITEM it = found[i];
				if (coinType == COIN_BTC) {
					std::string addr = secp->GetAddress(it.mode, it.hash);
					if (checkPrivKey(addr, keys[it.thId], it.incr, it.mode)) {
						nbFoundKey++;
					}
				}
				else {
					std::string addr = secp->GetAddressETH(it.hash);
					if (checkPrivKeyETH(addr, keys[it.thId], it.incr)) {
						nbFoundKey++;
					}
				}
			}
			break;
		case (int)SEARCH_MODE_MX:
			ok = g->LaunchSEARCH_MODE_MX(found, false);
			for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
				ITEM it = found[i];
				//Point pk;
				//memcpy((uint32_t*)pk.x.bits, (uint32_t*)it.hash, 8);
				//string addr = secp->GetAddress(it.mode, pk);
				if (checkPrivKeyX(/*addr,*/ keys[it.thId], it.incr, it.mode)) {
					nbFoundKey++;
				}
			}
			break;
		case (int)SEARCH_MODE_SA:
			ok = g->LaunchSEARCH_MODE_SA(found, false);
			for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
				ITEM it = found[i];
				if (coinType == COIN_BTC) {
					std::string addr = secp->GetAddress(it.mode, it.hash);
					if (checkPrivKey(addr, keys[it.thId], it.incr, it.mode)) {
						nbFoundKey++;
					}
				}
				else {
					std::string addr = secp->GetAddressETH(it.hash);
					if (checkPrivKeyETH(addr, keys[it.thId], it.incr)) {
						nbFoundKey++;
					}
				}
			}
			break;
		case (int)SEARCH_MODE_SX:
			ok = g->LaunchSEARCH_MODE_SX(found, false);
			for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
				ITEM it = found[i];
				//Point pk;
				//memcpy((uint32_t*)pk.x.bits, (uint32_t*)it.hash, 8);
				//string addr = secp->GetAddress(it.mode, pk);
				if (checkPrivKeyX(/*addr,*/ keys[it.thId], it.incr, it.mode)) {
					nbFoundKey++;
				}
			}
			break;
		case (int)SEARCH_MODE_PUZZLE71:
			// PUZZLE71 mode behaves like SA mode for now
			ok = g->LaunchPUZZLE71(found, false);
			for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
				ITEM it = found[i];
				std::string addr = secp->GetAddress(it.mode, it.hash);
				if (checkPrivKey(addr, keys[it.thId], it.incr, it.mode)) {
					nbFoundKey++;
				}
			}
			break;
		default:
			break;
		}

		if (ok) {
			for (int i = 0; i < nbThread; i++) {
				keys[i].Add((uint64_t)STEP_SIZE);
			}
			counters[thId] += (uint64_t)(STEP_SIZE)*nbThread; // Point
		}

	}

	// 智能指针会自动释放内存，无需手动delete
	// delete[] keys;  // 已由unique_ptr管理
	// delete[] p;     // 已由unique_ptr管理
	delete g;

#else
	ph->hasStarted = true;
	printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif

	ph->isRunning = false;

}

// ----------------------------------------------------------------------------

bool KeyHunt::isAlive(GPUThreadParam* p, int total)
{
    for (int i = 0; i < total; i++) {
        if (!p[i].isRunning) {
            return false;
        }
    }
    return true;
}

// ----------------------------------------------------------------------------

bool KeyHunt::hasStarted(GPUThreadParam* p, int total)
{
    for (int i = 0; i < total; i++) {
        if (!p[i].hasStarted) {
            return false;
        }
    }
    return true;
}

// ----------------------------------------------------------------------------

uint64_t KeyHunt::getGPUCount(int gpuThreadBase, int totalGpuThreads)
{
    uint64_t count = 0;
    for (int i = 0; i < totalGpuThreads; i++) {
        count += counters[gpuThreadBase + i];
    }
    return count;
}

// ----------------------------------------------------------------------------

void KeyHunt::rKeyRequest(GPUThreadParam* p, int total)
{
    for (int i = 0; i < total; i++) {
        p[i].rKeyRequest = true;
    }
}
// ----------------------------------------------------------------------------

void KeyHunt::SetupRanges(uint32_t totalThreads)
{
	// 添加溢出检查
	if (totalThreads == 0) {
		printf("Error: totalThreads cannot be zero\n");
		return;
	}
	
	Int threads;
	threads.SetInt32(totalThreads);
	
	// 检查范围是否有效
	if (rangeStart.IsGreaterOrEqual(&rangeEnd)) {
		printf("Error: Invalid range - start must be less than end\n");
		return;
	}
	
	rangeDiff.Set(&rangeEnd);
	rangeDiff.Sub(&rangeStart);
	
	// 检查减法是否下溢
	if (rangeDiff.IsNegative()) {
		printf("Error: Range calculation underflow\n");
		return;
	}
	
	rangeDiff.Div(&threads);
	
	// 检查除法结果是否合理
	if (rangeDiff.IsZero() && !rangeStart.IsEqual(&rangeEnd)) {
		printf("Warning: Range per thread is zero, consider reducing thread count\n");
	}
}

// ----------------------------------------------------------------------------

void KeyHunt::Search(std::vector<int> gpuId, std::vector<int> gridSize, bool& should_exit)
{
    if (!useGpu) {
        throw std::runtime_error("GPU execution not enabled");
    }
    if (gpuId.empty()) {
        throw std::runtime_error("At least one GPU must be specified");
    }

    double t0;
    double t1;
    endOfSearch = false;
    nbGPUThread = static_cast<int>(gpuId.size());
    nbFoundKey = 0;

    SetupRanges(nbGPUThread);

    memset(counters, 0, sizeof(counters));

    std::unique_ptr<GPUThreadParam[]> params(new GPUThreadParam[nbGPUThread]);

#ifdef WIN64
    std::vector<HANDLE> threadHandles(nbGPUThread);
    ghMutex = CreateMutex(NULL, FALSE, NULL);
#else
    std::vector<pthread_t> threadHandles(nbGPUThread);
    pthread_mutex_init(&ghMutex, nullptr);
#endif

    Int nextStart;
    nextStart.Set(&rangeStart);

    for (int i = 0; i < nbGPUThread; i++) {
        params[i].obj = this;
        params[i].threadId = i;
        params[i].isRunning = true;
        params[i].hasStarted = false;
        params[i].gpuId = gpuId[i];
        if (gridSize.size() > static_cast<size_t>(2 * i + 1)) {
            params[i].gridSizeX = gridSize[2 * i];
            params[i].gridSizeY = gridSize[2 * i + 1];
        } else {
            params[i].gridSizeX = -1;
            params[i].gridSizeY = KeyHuntConstants::DEFAULT_GPU_THREADS_PER_BLOCK;
        }

        params[i].rangeStart.Set(&nextStart);
        Int nextEnd(nextStart);
        nextEnd.Add(&rangeDiff);
        params[i].rangeEnd.Set(&nextEnd);
        nextStart.Set(&nextEnd);
        params[i].rKeyRequest = false;

#ifdef WIN64
        threadHandles[i] = CreateThread(NULL, 0, _FindKeyGPU, (void*)(params.get() + i), 0, nullptr);
#else
        pthread_create(&threadHandles[i], NULL, &_FindKeyGPU, (void*)(params.get() + i));
#endif
    }

#ifndef WIN64
    setvbuf(stdout, NULL, _IONBF, 0);
#endif
    printf("\nStarting GPU search loop...\n");

    uint64_t lastCount = 0;
    uint64_t gpuCount = 0;
    uint64_t lastGPUCount = 0;

#define FILTER_SIZE 8
    double lastGpuKeyRate[FILTER_SIZE];
    uint32_t filterPos = 0;
    double gpuKeyRate = 0.0;
    char timeStr[256];

    memset(lastGpuKeyRate, 0, sizeof(lastGpuKeyRate));

    printf("Waiting for GPU threads to start...\n");
    int wait_count = 0;
    while (!hasStarted(params.get(), nbGPUThread)) {
        Timer::SleepMillis(500);
        wait_count++;
        if (wait_count % 4 == 0) {
            printf("Still waiting for GPU threads (waited %d seconds)...\n", wait_count / 2);
        }
        if (wait_count > 20) {
            printf("ERROR: GPU threads failed to start after 10 seconds\n");
            break;
        }
    }

    Timer::Init();
    t0 = Timer::get_tick();
    startTime = t0;
    Int ICount;
    double completedPerc = 0;
    uint64_t rKeyCount = 0;

    while (isAlive(params.get(), nbGPUThread)) {
        int delay = 2000;
        while (isAlive(params.get(), nbGPUThread) && delay > 0) {
            Timer::SleepMillis(500);
            delay -= 500;
        }

        gpuCount = getGPUCount(0, nbGPUThread);
        uint64_t count = gpuCount;
        ICount.SetInt64(count);
        int completedBits = ICount.GetBitLength();
        if (rKey <= 0) {
            completedPerc = CalcPercantage(ICount, rangeStart, rangeDiff2);
        }

        t1 = Timer::get_tick();
        gpuKeyRate = (double)(gpuCount - lastGPUCount) / (t1 - t0);
        lastGpuKeyRate[filterPos % FILTER_SIZE] = gpuKeyRate;
        filterPos++;

        double avgGpuKeyRate = 0.0;
        uint32_t nbSample;
        for (nbSample = 0; (nbSample < FILTER_SIZE) && (nbSample < filterPos); nbSample++) {
            avgGpuKeyRate += lastGpuKeyRate[nbSample];
        }
        avgGpuKeyRate /= (double)(nbSample);

        if (isAlive(params.get(), nbGPUThread)) {
            memset(timeStr, '\0', 256);
            printf("\r[%s] [GPU: %.2f Mk/s] [C: %.3f %%] [R: %" PRIu64 "] [T: %s (%d bit)] [F: %d]  ",
                   toTimeStr(t1, timeStr),
                   avgGpuKeyRate / 1000000.0,
                   completedPerc,
                   rKeyCount,
                   formatThousands(count).c_str(),
                   completedBits,
                   nbFoundKey);
        }
        if (rKey > 0) {
            if ((count - lastrKey) > (1000000 * rKey)) {
                rKeyRequest(params.get(), nbGPUThread);
                lastrKey = count;
                rKeyCount++;
            }
        }

        lastCount = count;
        lastGPUCount = gpuCount;
        t0 = t1;
        if (should_exit || nbFoundKey >= targetCounter || completedPerc > 100.5) {
            endOfSearch = true;
        }
    }

#ifndef WIN64
    for (auto& handle : threadHandles) {
        pthread_join(handle, nullptr);
    }
    pthread_mutex_destroy(&ghMutex);
#else
    for (auto& handle : threadHandles) {
        WaitForSingleObject(handle, INFINITE);
        CloseHandle(handle);
    }
    CloseHandle(ghMutex);
#endif

    }

// ----------------------------------------------------------------------------

std::string KeyHunt::GetHex(std::vector<unsigned char> &buffer)
{
	std::string ret;

	char tmp[128];
	for (int i = 0; i < (int)buffer.size(); i++) {
		sprintf(tmp, "%02X", buffer[i]);
		ret.append(tmp);
	}
	return ret;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

std::string KeyHunt::formatThousands(uint64_t x)
{
	char buf[32] = "";

	sprintf(buf, "%" PRIu64, x);

	std::string s(buf);

	int len = (int)s.length();

	int numCommas = (len - 1) / 3;

	if (numCommas == 0) {
		return s;
	}

	std::string result = "";

	int count = ((len % 3) == 0) ? 0 : (3 - (len % 3));

	for (int i = 0; i < len; i++) {
		result += s[i];

		if (count++ == 2 && i < len - 1) {
			result += ",";
			count = 0;
		}
	}
	return result;
}

// ----------------------------------------------------------------------------

char* KeyHunt::toTimeStr(int sec, char* timeStr)
{
	int h, m, s;
	h = (sec / 3600);
	m = (sec - (3600 * h)) / 60;
	s = (sec - (3600 * h) - (m * 60));
	sprintf(timeStr, "%0*d:%0*d:%0*d", 2, h, 2, m, 2, s);
	return (char*)timeStr;
}

// ----------------------------------------------------------------------------

//#include <gmp.h>
//#include <gmpxx.h>
// ((input - min) * 100) / (max - min)
//double KeyHunt::GetPercantage(uint64_t v)
//{
//	//Int val(v);
//	//mpz_class x(val.GetBase16().c_str(), 16);
//	//mpz_class r(rangeStart.GetBase16().c_str(), 16);
//	//x = x - mpz_class(rangeEnd.GetBase16().c_str(), 16);
//	//x = x * 100;
//	//mpf_class y(x);
//	//y = y / mpf_class(r);
//	return 0;// y.get_d();
//}




