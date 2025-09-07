#!/bin/bash

# KeyHunt-Cuda NVIDIA H20 专用构建脚本
# 针对 NVIDIA H20 (CC 9.0) 优化

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    KeyHunt-Cuda NVIDIA H20 构建脚本${NC}"
echo -e "${BLUE}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}脚本目录: $SCRIPT_DIR${NC}"
echo -e "${BLUE}项目目录: $PROJECT_DIR${NC}"

# 切换到项目目录
cd "$PROJECT_DIR" || {
    echo -e "${RED}无法切换到项目目录: $PROJECT_DIR${NC}"
    exit 1
}

# 检查 Makefile
if [ ! -f "Makefile" ]; then
    echo -e "${RED}错误: 在 $PROJECT_DIR 目录中未找到 Makefile${NC}"
    echo -e "${YELLOW}当前目录文件:${NC}"
    ls -la
    exit 1
fi

echo -e "${GREEN}找到 Makefile${NC}"

# 检查 CUDA 编译器
echo -e "${BLUE}检查 CUDA 工具链...${NC}"
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}错误: 未找到 nvcc 编译器${NC}"
    echo -e "${YELLOW}请安装 CUDA Toolkit${NC}"
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo -e "  ${GREEN}NVCC 版本:${NC} $NVCC_VERSION"

if ! command -v g++ &> /dev/null; then
    echo -e "${RED}错误: 未找到 g++ 编译器${NC}"
    echo -e "${YELLOW}请安装 build-essential 包${NC}"
    exit 1
fi

GPP_VERSION=$(g++ --version | head -1 | awk '{print $NF}')
echo -e "  ${GREEN}G++ 版本:${NC} $GPP_VERSION"

# 检查 GPU
echo -e "${BLUE}检查 GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "  ${GREEN}检测到 NVIDIA GPU${NC}"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits | head -1 | while IFS=',' read -r name compute_cap; do
        name=$(echo $name | xargs)
        compute_cap=$(echo $compute_cap | xargs)
        echo -e "  ${YELLOW}GPU 型号:${NC} $name"
        echo -e "  ${YELLOW}计算能力:${NC} $compute_cap"
    done
else
    echo -e "  ${YELLOW}警告: 未找到 nvidia-smi，无法检测 GPU${NC}"
fi

# 清理构建目录
echo -e "${YELLOW}清理构建目录...${NC}"
make clean
if [ $? -ne 0 ]; then
    echo -e "${RED}清理失败${NC}"
    exit 1
fi
echo -e "${GREEN}清理完成${NC}"

# NVIDIA H20 专用构建参数
# H20 基于 Hopper 架构，计算能力 9.0
echo -e "${GREEN}配置 NVIDIA H20 构建参数${NC}"
MAKE_ARGS="gpu=1 CCAP=90"

# 执行构建
echo -e "${BLUE}执行构建命令: make $MAKE_ARGS all${NC}"
make $MAKE_ARGS all

if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}NVIDIA H20 优化构建成功完成!${NC}"
    echo -e "${GREEN}可执行文件位置: ./KeyHunt${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # 显示构建的文件信息
    if [ -f "./KeyHunt" ]; then
        echo -e "${BLUE}构建文件信息:${NC}"
        ls -lh ./KeyHunt
    fi
    
    # 显示使用说明
    echo -e "${BLUE}NVIDIA H20 使用说明:${NC}"
    echo -e "  ${YELLOW}推荐运行参数:${NC}"
    echo -e "    ./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range 1:FFFFFFFF 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
    echo -e ""
    echo -e "  ${YELLOW}性能调优建议:${NC}"
    echo -e "    1. 使用较大的网格大小以充分利用 H20 的高并行计算能力"
    echo -e "    2. 调整 --gpugridsize 参数以获得最佳性能"
    echo -e "    3. H20 具有 96GB HBM3 内存，可以处理更大的数据集"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}NVIDIA H20 构建失败!${NC}"
    echo -e "${RED}请检查错误信息并重试${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi