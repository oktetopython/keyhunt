#!/bin/bash

# KeyHunt-Cuda 修复验证脚本
# 用于验证技术债务修复是否成功

echo "=========================================="
echo "KeyHunt-Cuda 技术债务修复验证脚本"
echo "=========================================="

# 检查是否在WSL环境中
if ! grep -q microsoft /proc/version; then
    echo "警告: 此脚本应在WSL环境中运行"
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 设置环境变量
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 检查CUDA环境
echo "检查CUDA环境..."
nvcc --version
if [ $? -ne 0 ]; then
    echo "错误: CUDA环境未正确配置"
    exit 1
fi

# 进入项目目录
cd "$(dirname "$0")"

# 清理之前的构建
echo "清理之前的构建..."
make clean

# 编译项目
echo "编译项目..."
make gpu=1 CCAP=75 all
if [ $? -ne 0 ]; then
    echo "错误: 编译失败"
    exit 1
fi

echo "编译成功!"

# 检查统一内核接口是否启用
echo "检查统一内核接口是否启用..."
grep -q "use_unified_kernels = true" GPU/GPUEngine.cu
if [ $? -eq 0 ]; then
    echo "✅ 统一内核接口已启用"
else
    echo "❌ 统一内核接口未启用"
fi

# 检查缓存优化是否启用
echo "检查缓存优化是否启用..."
grep -q "KEYHUNT_CACHE_OPTIMIZED" Makefile
if [ $? -eq 0 ]; then
    echo "✅ 缓存优化已启用"
else
    echo "❌ 缓存优化未启用"
fi

# 检查是否移除了重复的代码
echo "检查是否移除了重复的代码..."
grep -c "CheckPointSEARCH_MODE_MA" GPU/GPUCompute.h | grep -q "0"
if [ $? -eq 0 ]; then
    echo "✅ 重复的检查函数已移除"
else
    echo "❌ 重复的检查函数未完全移除"
fi

# 检查是否添加了语义化常量
echo "检查是否添加了语义化常量..."
grep -q "ELLIPTIC_CURVE_GROUP_SIZE" Constants.h
if [ $? -eq 0 ]; then
    echo "✅ 语义化常量已添加"
else
    echo "❌ 语义化常量未添加"
fi

echo "=========================================="
echo "验证完成"
echo "=========================================="

# 运行简单的测试（如果提供了测试数据）
if [ -f "test_data.txt" ]; then
    echo "运行简单测试..."
    ./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range 1:100 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa 2>&1 | head -20
    echo "测试完成"
else
    echo "提示: 如需运行功能测试，请提供测试数据文件 test_data.txt"
fi