#include "KeyHunt.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

// 构造函数 - 文件输入版本
KeyHunt::KeyHunt(const std::string& inputFile, CompMode compMode, int searchMode, CoinType coinType,
                bool gpuEnable, const std::string& outputFile, bool useSSE,
                int maxFound, uint64_t rKey, const std::string& rangeStart,
                const std::string& rangeEnd, bool& should_exit)
    : inputFile_(inputFile), compMode_(compMode), searchMode_(searchMode), coinType_(coinType),
      gpuEnable_(gpuEnable), outputFile_(outputFile), useSSE_(useSSE),
      maxFound_(maxFound), rKey_(rKey), rangeStart_(rangeStart), rangeEnd_(rangeEnd),
      should_exit_(should_exit)
{
    initialize();
}

// 构造函数 - 哈希/X点输入版本
KeyHunt::KeyHunt(const std::vector<uint8_t>& hashORxpoint, CompMode compMode, int searchMode, CoinType coinType,
                bool gpuEnable, const std::string& outputFile, bool useSSE,
                int maxFound, uint64_t rKey, const std::string& rangeStart,
                const std::string& rangeEnd, bool& should_exit)
    : hashORxpoint_(hashORxpoint), compMode_(compMode), searchMode_(searchMode), coinType_(coinType),
      gpuEnable_(gpuEnable), outputFile_(outputFile), useSSE_(useSSE),
      maxFound_(maxFound), rKey_(rKey), rangeStart_(rangeStart), rangeEnd_(rangeEnd),
      should_exit_(should_exit)
{
    initialize();
}

// 析构函数
KeyHunt::~KeyHunt()
{
    cleanup();
}

// 初始化方法
void KeyHunt::initialize()
{
    // 检查输入文件是否存在（如果是文件输入模式）
    if (!inputFile_.empty()) {
        std::ifstream file(inputFile_);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open input file: " + inputFile_);
        }
        file.close();
    }
    
    // 检查输出文件目录是否可写
    if (!outputFile_.empty()) {
        std::ofstream file(outputFile_, std::ios::app);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot write to output file: " + outputFile_);
        }
        file.close();
    }
    
    std::cout << "KeyHunt initialized successfully" << std::endl;
}

// 清理方法
void KeyHunt::cleanup()
{
    std::cout << "KeyHunt cleanup completed" << std::endl;
}

// 主搜索方法
void KeyHunt::Search(int nbCPUThread, const std::vector<int>& gpuId, 
                   const std::vector<int>& gridSize, bool& should_exit)
{
    std::cout << "Starting search with " << nbCPUThread << " CPU threads" << std::endl;
    
    if (gpuEnable_) {
        std::cout << "GPU enabled with " << gpuId.size() << " GPU(s)" << std::endl;
        searchGPU(gpuId, gridSize, should_exit);
    } else {
        searchCPU(nbCPUThread, should_exit);
    }
    
    processResults();
}

// CPU搜索方法
void KeyHunt::searchCPU(int nbThreads, bool& should_exit)
{
    std::cout << "CPU search started with " << nbThreads << " threads" << std::endl;
    // 这里实现具体的CPU搜索逻辑
    
    // 模拟搜索过程
    for (int i = 0; i < 10 && !should_exit; i++) {
        std::cout << "CPU search iteration: " << i << std::endl;
        // 模拟工作
    }
}

// GPU搜索方法
void KeyHunt::searchGPU(const std::vector<int>& gpuIds, const std::vector<int>& gridSizes, bool& should_exit)
{
    std::cout << "GPU search started with " << gpuIds.size() << " GPU(s)" << std::endl;
    // 这里实现具体的GPU搜索逻辑
    
    // 模拟搜索过程
    for (int i = 0; i < 5 && !should_exit; i++) {
        std::cout << "GPU search iteration: " << i << std::endl;
        // 模拟工作
    }
}

// 处理结果方法
void KeyHunt::processResults()
{
    std::cout << "Processing search results" << std::endl;
    // 这里实现结果处理逻辑
}

// 保存结果方法
void KeyHunt::saveResult(const std::string& result)
{
    if (!outputFile_.empty()) {
        std::ofstream file(outputFile_, std::ios::app);
        file << result << std::endl;
        file.close();
    }
    std::cout << "Result: " << result << std::endl;
}