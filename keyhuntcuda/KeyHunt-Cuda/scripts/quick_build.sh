#!/bin/bash

# KeyHunt-Cuda 快速构建脚本
# 一键构建最常见的配置

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    KeyHunt-Cuda 快速构建脚本${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查参数
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  cpu            构建仅 CPU 版本"
    echo "  debug          构建调试版本"
    echo "  clean          清理后构建"
    echo ""
    echo "如果没有指定选项，默认构建发布版 GPU 版本"
    exit 0
fi

# 默认构建选项
BUILD_TARGET="default"
CLEAN_FIRST=0

# 解析参数
for arg in "$@"; do
    case $arg in
        cpu)
            BUILD_TARGET="cpu"
            ;;
        debug)
            BUILD_TARGET="debug"
            ;;
        clean)
            CLEAN_FIRST=1
            ;;
        *)
            echo -e "${RED}未知选项: $arg${NC}"
            echo "使用 $0 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 清理构建目录（如果需要）
if [ $CLEAN_FIRST -eq 1 ]; then
    echo -e "${YELLOW}清理构建目录...${NC}"
    make clean
    if [ $? -ne 0 ]; then
        echo -e "${RED}清理失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}清理完成${NC}"
fi

# 根据选项执行构建
case $BUILD_TARGET in
    cpu)
        echo -e "${YELLOW}构建仅 CPU 版本...${NC}"
        make all
        ;;
    debug)
        echo -e "${YELLOW}构建调试版本...${NC}"
        make gpu=1 debug=1 CCAP=75 all
        ;;
    *)
        echo -e "${YELLOW}构建默认发布版 GPU 版本...${NC}"
        make gpu=1 CCAP=75 all
        ;;
esac

# 检查构建结果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}构建成功完成!${NC}"
    
    # 显示可执行文件信息
    if [ -f "./KeyHunt" ]; then
        echo -e "${GREEN}可执行文件位置: ./KeyHunt${NC}"
        ls -lh ./KeyHunt
    elif [ -f "./KeyHunt.exe" ]; then
        echo -e "${GREEN}可执行文件位置: ./KeyHunt.exe${NC}"
        ls -lh ./KeyHunt.exe
    fi
    
    echo -e "${GREEN}========================================${NC}"
    
    # 显示快速测试命令
    echo -e "${BLUE}快速测试命令:${NC}"
    echo -e "${YELLOW}./KeyHunt -h${NC}                    # 显示帮助信息"
    echo -e "${YELLOW}./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range e9ae493300:e9ae493400 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv${NC}"
    echo -e "                                   # 测试比特币谜题40"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}构建失败!${NC}"
    echo -e "${RED}请检查错误信息并重试${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi