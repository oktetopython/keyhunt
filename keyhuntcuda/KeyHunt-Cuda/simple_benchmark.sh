#!/bin/bash

# 简单基准测试脚本

echo "开始简单基准测试..."

# 检查KeyHunt可执行文件是否存在
if [ ! -f "./KeyHunt" ]; then
    echo "错误: KeyHunt可执行文件不存在"
    exit 1
fi

# 使用一个已知的小范围进行测试
echo "运行测试用例: Bitcoin puzzle 40 (范围: e9ae493300:e9ae493400)"

# 清理之前的Found.txt文件
rm -f Found.txt

# 运行一次测试并显示性能数据
echo "运行测试..."
timeout 30s ./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range e9ae493300:e9ae493400 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv 2>&1 | grep "PROFILE"

echo "基准测试完成"