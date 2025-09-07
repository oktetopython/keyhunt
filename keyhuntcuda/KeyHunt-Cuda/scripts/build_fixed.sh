#!/bin/bash

# KeyHunt-Cuda 通用构建脚本
# 支持 Linux 和 WSL 环境

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    KeyHunt-Cuda 构建脚本${NC}"
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

# 检查操作系统
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo -e "${YELLOW}检测到 Windows 环境${NC}"
    IS_WINDOWS=1
    # Windows环境下设置CUDA_PATH环境变量
    if [ -z "$CUDA_PATH" ]; then
        # 尝试常见的CUDA安装路径
        for cuda_path in "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"; do
            if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc.exe" ]; then
                export CUDA_PATH="$cuda_path"
                echo -e "  ${GREEN}设置 CUDA_PATH 为: $CUDA_PATH${NC}"
                break
            fi
        done
    fi
else
    echo -e "${GREEN}检测到 Linux/WSL 环境${NC}"
    IS_WINDOWS=0
    # Linux环境下设置CUDA_HOME环境变量
    if [ -z "$CUDA_HOME" ]; then
        # 检查系统路径中的nvcc
        CUDA_BIN=$(which nvcc 2>/dev/null)
        if [ -n "$CUDA_BIN" ]; then
            # 从nvcc路径推断CUDA安装目录
            export CUDA_HOME=$(dirname $(dirname $CUDA_BIN))
            echo -e "  ${GREEN}设置 CUDA_HOME 为: $CUDA_HOME${NC}"
        fi
    fi
fi

# 默认参数
BUILD_TYPE="release"
GPU_SUPPORT=1
MULTI_GPU=0
CCAP=75
DEBUG=0
CLEAN_BUILD=0

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -d, --debug             构建调试版本"
    echo "  -c, --clean             清理构建目录"
    echo "  -n, --no-gpu            构建无GPU版本"
    echo "  -m, --multi-gpu         构建多GPU支持版本"
    echo "  --ccap VALUE            设置GPU计算能力 (默认: 75)"
    echo "  --cpu-only              同 --no-gpu"
    echo ""
    echo "示例:"
    echo "  $0                      # 构建默认GPU版本"
    echo "  $0 -d                   # 构建调试版本"
    echo "  $0 -c -m --ccap 86      # 清理并构建多GPU版本，计算能力8.6"
    echo "  $0 --cpu-only           # 构建仅CPU版本"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--debug)
            DEBUG=1
            BUILD_TYPE="debug"
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=1
            shift
            ;;
        -n|--no-gpu|--cpu-only)
            GPU_SUPPORT=0
            shift
            ;;
        -m|--multi-gpu)
            MULTI_GPU=1
            shift
            ;;
        --ccap)
            CCAP="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 清理构建目录
if [ $CLEAN_BUILD -eq 1 ]; then
    echo -e "${YELLOW}清理构建目录...${NC}"
    make clean
    if [ $? -ne 0 ]; then
        echo -e "${RED}清理失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}清理完成${NC}"
fi

# 构建参数
MAKE_ARGS=""

if [ $GPU_SUPPORT -eq 1 ]; then
    MAKE_ARGS="$MAKE_ARGS gpu=1"
    echo -e "${GREEN}启用 GPU 支持${NC}"
else
    echo -e "${YELLOW}构建仅 CPU 版本${NC}"
fi

if [ $MULTI_GPU -eq 1 ]; then
    MAKE_ARGS="$MAKE_ARGS MULTI_GPU=1"
    echo -e "${GREEN}启用多 GPU 支持${NC}"
else
    MAKE_ARGS="$MAKE_ARGS CCAP=$CCAP"
    echo -e "${GREEN}设置 GPU 计算能力为 $CCAP${NC}"
fi

if [ $DEBUG -eq 1 ]; then
    MAKE_ARGS="$MAKE_ARGS debug=1"
    echo -e "${GREEN}构建调试版本${NC}"
else
    echo -e "${GREEN}构建发布版本${NC}"
fi

# 执行构建
echo -e "${BLUE}执行构建命令: make $MAKE_ARGS all${NC}"
make $MAKE_ARGS all

if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}构建成功完成!${NC}"
    echo -e "${GREEN}可执行文件位置: ./KeyHunt${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # 显示构建的文件信息
    if [ -f "./KeyHunt" ]; then
        echo -e "${BLUE}构建文件信息:${NC}"
        ls -lh ./KeyHunt
    elif [ -f "./KeyHunt.exe" ]; then
        echo -e "${BLUE}构建文件信息:${NC}"
        ls -lh ./KeyHunt.exe
    fi
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}构建失败!${NC}"
    echo -e "${RED}请检查错误信息并重试${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi