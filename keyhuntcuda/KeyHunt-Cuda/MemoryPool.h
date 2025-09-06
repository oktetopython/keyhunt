#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <memory>
#include <vector>
#include <queue>

/**
 * @brief 简化的内存池管理器
 * @details 用于优化IntGroup对象的分配和释放
 */
class SimpleMemoryPool {
private:
    std::vector<std::unique_ptr<char[]>> memory_blocks;
    std::queue<char*> available_blocks;
    size_t block_size;
    size_t max_blocks;

public:
    /**
     * @brief 构造函数
     * @param block_size 每个内存块的大小
     * @param max_blocks 最大内存块数量
     */
    SimpleMemoryPool(size_t block_size, size_t max_blocks) 
        : block_size(block_size), max_blocks(max_blocks) {
        
        // 预分配初始内存块
        for (size_t i = 0; i < 16 && i < max_blocks; ++i) {
            auto block = std::make_unique<char[]>(block_size);
            available_blocks.push(block.get());
            memory_blocks.push_back(std::move(block));
        }
    }

    /**
     * @brief 获取一个内存块
     * @return char* 内存块指针，如果池已满则返回nullptr
     */
    char* acquire() {
        if (!available_blocks.empty()) {
            char* ptr = available_blocks.front();
            available_blocks.pop();
            return ptr;
        }
        
        // 如果池未满，创建新内存块
        if (memory_blocks.size() < max_blocks) {
            auto block = std::make_unique<char[]>(block_size);
            char* ptr = block.get();
            memory_blocks.push_back(std::move(block));
            return ptr;
        }
        
        return nullptr; // 池已满
    }

    /**
     * @brief 释放一个内存块回池中
     * @param ptr 要释放的内存块指针
     */
    void release(char* ptr) {
        if (ptr && available_blocks.size() < max_blocks) {
            available_blocks.push(ptr);
        }
    }

    /**
     * @brief 获取当前可用内存块数量
     * @return size_t 可用内存块数量
     */
    size_t available_count() {
        return available_blocks.size();
    }

    /**
     * @brief 获取池中总内存块数量
     * @return size_t 总内存块数量
     */
    size_t total_count() {
        return memory_blocks.size();
    }
};

#endif // MEMORY_POOL_H