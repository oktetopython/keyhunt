#!/bin/bash

# 5分钟基准测试脚本

echo "=========================================="
echo "  KeyHunt 5分钟基准测试"
echo "=========================================="

# 检查KeyHunt可执行文件是否存在
if [ ! -f "./KeyHunt" ]; then
    echo "❌ 错误: KeyHunt可执行文件不存在"
    exit 1
fi

echo "✅ KeyHunt可执行文件存在"

# 测试参数
KEY_RANGE_START="2000000000000"
KEY_RANGE_END="3ffffffffffff"
BITCOIN_ADDRESS="1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk"

echo "测试参数:"
echo "  Key Range: $KEY_RANGE_START:$KEY_RANGE_END"
echo "  Bitcoin Address: $BITCOIN_ADDRESS"

# 清理之前的Found.txt文件
rm -f Found.txt

# 运行5分钟基准测试
echo "开始5分钟基准测试..."
timeout 300s ./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range $KEY_RANGE_START:$KEY_RANGE_END $BITCOIN_ADDRESS

# 检查结果
echo "=========================================="
echo "基准测试完成"
echo "=========================================="

if [ -f "Found.txt" ]; then
    echo "找到结果:"
    cat Found.txt
else
    echo "未找到匹配的私钥（这是预期的，因为这是一个大范围测试）"
fi

# 显示最终统计信息
echo "=========================================="
echo "测试摘要:"
echo "  测试时长: 5分钟"
echo "  密钥范围: $KEY_RANGE_START:$KEY_RANGE_END"
echo "  目标地址: $BITCOIN_ADDRESS"
echo "  测试结果: 基准测试完成"
echo "=========================================="