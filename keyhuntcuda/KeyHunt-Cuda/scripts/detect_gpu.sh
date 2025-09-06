#!/bin/bash

# KeyHunt-Cuda GPU 检测脚本
# 自动检测系统中的 NVIDIA GPU 并推荐最佳编译选项

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    KeyHunt-Cuda GPU 检测脚本${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查 nvidia-smi 是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}警告: 未找到 nvidia-smi 命令${NC}"
    echo -e "${YELLOW}请确保已安装 NVIDIA 驱动程序${NC}"
    exit 1
fi

# 获取 GPU 信息
echo -e "${GREEN}检测系统中的 NVIDIA GPU...${NC}"
echo ""

# 获取 GPU 数量
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$GPU_COUNT" ]; then
    echo -e "${RED}错误: 无法获取 GPU 信息${NC}"
    exit 1
fi

echo -e "${BLUE}检测到 $GPU_COUNT 个 GPU:${NC}"

# 获取详细的 GPU 信息
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | tail -n +2 | while IFS=',' read -r name compute_cap memory; do
    # 清理数据
    name=$(echo $name | xargs)  # 去除前后空格
    compute_cap=$(echo $compute_cap | xargs)
    memory=$(echo $memory | xargs | sed 's/ MiB//')
    
    echo -e "  ${YELLOW}GPU 型号:${NC} $name"
    echo -e "  ${YELLOW}计算能力:${NC} $compute_cap"
    echo -e "  ${YELLOW}显存大小:${NC} $memory MiB"
    
    # 根据计算能力推荐编译选项
    ccap_int=$(echo $compute_cap | tr -d '.' | sed 's/^0*//')
    if [ -z "$ccap_int" ]; then
        ccap_int=0
    fi
    
    echo -e "  ${YELLOW}推荐编译命令:${NC}"
    if [ $ccap_int -ge 75 ]; then
        echo -e "    ${GREEN}make clean && make gpu=1 CCAP=$ccap_int all${NC}"
    else
        echo -e "    ${GREEN}make clean && make gpu=1 CCAP=75 all${NC} ${YELLOW}(最低支持版本)${NC}"
    fi
    echo ""
done

# 检查 CUDA 编译器
echo -e "${BLUE}检查 CUDA 工具链...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo -e "  ${GREEN}NVCC 版本:${NC} $CUDA_VERSION"
else
    echo -e "  ${RED}未找到 NVCC 编译器${NC}"
    echo -e "  ${YELLOW}请安装 CUDA Toolkit${NC}"
fi

if command -v g++ &> /dev/null; then
    GPP_VERSION=$(g++ --version | head -1 | awk '{print $NF}')
    echo -e "  ${GREEN}G++ 版本:${NC} $GPP_VERSION"
else
    echo -e "  ${RED}未找到 G++ 编译器${NC}"
    echo -e "  ${YELLOW}请安装 build-essential 包${NC}"
fi

echo ""
echo -e "${BLUE}构建选项说明:${NC}"
echo -e "  ${YELLOW}单 GPU 构建:${NC} 针对特定 GPU 架构优化，性能最佳"
echo -e "  ${YELLOW}多 GPU 构建:${NC} 支持多种 GPU 架构，兼容性更好"
echo -e "  ${YELLOW}仅 CPU 构建:${NC} 不需要 GPU 支持，适用于所有系统"
echo ""
echo -e "${BLUE}推荐构建命令:${NC}"
echo -e "  ${GREEN}自动检测构建:${NC} ./scripts/build.sh"
echo -e "  ${GREEN}单 GPU 构建:${NC} ./scripts/build.sh --ccap [计算能力]"
echo -e "  ${GREEN}多 GPU 构建:${NC} ./scripts/build.sh -m"
echo -e "  ${GREEN}仅 CPU 构建:${NC} ./scripts/build.sh --cpu-only"
echo ""
echo -e "${BLUE}========================================${NC}"