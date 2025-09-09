#ifndef KEYHUNT_H
#define KEYHUNT_H

#include <string>
#include <vector>
#include <cstdint>

// 搜索模式枚举
enum SearchMode {
    SEARCH_MODE_MA = 0,  // Multi Address
    SEARCH_MODE_SA = 1,  // Single Address
    SEARCH_MODE_MX = 2,  // Multi X Points
    SEARCH_MODE_SX = 3   // Single X Point
};

// 压缩模式枚举
enum CompMode {
    COMPRESSED = 0,
    UNCOMPRESSED = 1,
    BOTH = 2
};

// 币种类型枚举
enum CoinType {
    COIN_BTC = 0,
    COIN_ETH = 1
};

class KeyHunt {
public:
    // 构造函数 - 文件输入版本
    KeyHunt(const std::string& inputFile, CompMode compMode, int searchMode, CoinType coinType,
            bool gpuEnable, const std::string& outputFile, bool useSSE,
            int maxFound, uint64_t rKey, const std::string& rangeStart,
            const std::string& rangeEnd, bool& should_exit);

    // 构造函数 - 哈希/X点输入版本
    KeyHunt(const std::vector<uint8_t>& hashORxpoint, CompMode compMode, int searchMode, CoinType coinType,
            bool gpuEnable, const std::string& outputFile, bool useSSE,
            int maxFound, uint64_t rKey, const std::string& rangeStart,
            const std::string& rangeEnd, bool& should_exit);

    // 析构函数
    ~KeyHunt();

    // 主搜索方法
    void Search(int nbCPUThread, const std::vector<int>& gpuId, 
               const std::vector<int>& gridSize, bool& should_exit);

private:
    // 成员变量
    std::string inputFile_;
    std::vector<uint8_t> hashORxpoint_;
    CompMode compMode_;
    int searchMode_;
    CoinType coinType_;
    bool gpuEnable_;
    std::string outputFile_;
    bool useSSE_;
    int maxFound_;
    uint64_t rKey_;
    std::string rangeStart_;
    std::string rangeEnd_;
    bool& should_exit_;

    // 私有方法
    void initialize();
    void cleanup();
    void searchCPU(int nbThreads, bool& should_exit);
    void searchGPU(const std::vector<int>& gpuIds, const std::vector<int>& gridSizes, bool& should_exit);
    void processResults();
    void saveResult(const std::string& result);
};

#endif // KEYHUNT_H