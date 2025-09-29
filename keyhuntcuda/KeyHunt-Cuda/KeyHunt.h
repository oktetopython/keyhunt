#ifndef KEYHUNTH
#define KEYHUNTH

// KeyHunt-Cuda Version Information
#define KEYHUNT_VERSION_MAJOR 1
#define KEYHUNT_VERSION_MINOR 7
#define KEYHUNT_VERSION_PATCH 1
#define KEYHUNT_VERSION_STRING "v1.7.1"

// Version with build date
#define KEYHUNT_VERSION KEYHUNT_VERSION_STRING " (2025-09-06)"
#define KEYHUNT_BUILD_DATE "2025-09-06"

#include <string>
#include <vector>
#include <memory>
#include "SECP256k1.h"
#include "Bloom.h"
#include "GPU/GPUEngine.h"
#ifdef WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif

class KeyHunt;

struct GPUThreadParam {
    KeyHunt* obj;
    int threadId;
    bool isRunning;
    bool hasStarted;
    int gridSizeX;
    int gridSizeY;
    int gpuId;
    Int rangeStart;
    Int rangeEnd;
    bool rKeyRequest;
};

class LockGuard {
private:
#ifdef WIN64
    HANDLE& mutex;
#else
    pthread_mutex_t& mutex;
#endif
public:
    explicit LockGuard(
#ifdef WIN64
        HANDLE& m
#else
        pthread_mutex_t& m
#endif
    ) : mutex(m) {
#ifdef WIN64
        WaitForSingleObject(mutex, INFINITE);
#else
        pthread_mutex_lock(&mutex);
#endif
    }
    ~LockGuard() {
#ifdef WIN64
        ReleaseMutex(mutex);
#else
        pthread_mutex_unlock(&mutex);
#endif
    }
};

class FileGuard {
private:
    FILE* file;
public:
    explicit FileGuard(FILE* f) : file(f) {}
    ~FileGuard() {
        if (file) {
            fclose(file);
        }
    }
    FILE* get() { return file; }
    FileGuard(const FileGuard&) = delete;
    FileGuard& operator=(const FileGuard&) = delete;
};

class KeyHunt {
public:
    KeyHunt(const std::string& inputFile,
            int compMode,
            int searchMode,
            int coinType,
            bool useGpu,
            const std::string& outputFile,
            uint32_t maxFound,
            uint64_t rKey,
            const std::string& rangeStart,
            const std::string& rangeEnd,
            bool& should_exit);

    KeyHunt(const std::vector<unsigned char>& hashORxpoint,
            int compMode,
            int searchMode,
            int coinType,
            bool useGpu,
            const std::string& outputFile,
            uint32_t maxFound,
            uint64_t rKey,
            const std::string& rangeStart,
            const std::string& rangeEnd,
            bool& should_exit);

    ~KeyHunt();

    void Search(std::vector<int> gpuId, std::vector<int> gridSize, bool& should_exit);
    void FindKeyGPU(GPUThreadParam* p);

private:
    std::string GetHex(std::vector<unsigned char>& buffer);
    bool checkPrivKey(std::string addr, Int& key, int32_t incr, bool mode);
    bool checkPrivKeyETH(std::string addr, Int& key, int32_t incr);
    bool checkPrivKeyX(Int& key, int32_t incr, bool mode);

    void output(std::string addr, std::string pAddr, std::string pAddrHex, std::string pubKey);
    bool isAlive(GPUThreadParam* p, int total);
    bool hasStarted(GPUThreadParam* p, int total);
    uint64_t getGPUCount(int gpuThreadBase, int totalGpuThreads);
    void rKeyRequest(GPUThreadParam* p, int total);
    void SetupRanges(uint32_t totalThreads);
    void getGPUStartingKeys(Int& tRangeStart, Int& tRangeEnd, int groupSize, int nbThread, Int* keys, Point* p);

    std::string formatThousands(uint64_t x);
    char* toTimeStr(int sec, char* timeStr);

    Secp256K1* secp;
    Bloom* bloom;

    uint64_t counters[256];
    double startTime;
    int compMode;
    int searchMode;
    int coinType;

    bool useGpu;
    bool endOfSearch;
    int nbGPUThread;
    int nbFoundKey;
    uint64_t targetCounter;

    std::string outputFile;
    std::string inputFile;
    uint32_t hash160Keccak[5];
    uint32_t xpoint[8];

    Int rangeStart;
    Int rangeEnd;
    Int rangeDiff;
    Int rangeDiff2;

    uint32_t maxFound;
    uint64_t rKey;
    uint64_t lastrKey;

#ifdef WIN64
    HANDLE ghMutex;
#else
    pthread_mutex_t ghMutex;
#endif
};

#endif // KEYHUNTH