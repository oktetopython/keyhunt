#!/bin/bash

# 测试统一内核接口的脚本

echo "开始测试统一内核接口..."

# 检查KeyHunt可执行文件是否存在
if [ ! -f "./KeyHunt" ]; then
    echo "错误: KeyHunt可执行文件不存在"
    exit 1
fi

# 使用一个已知的小范围进行测试，以验证统一内核接口是否正常工作
# 这里使用Bitcoin puzzle 40的已知解作为测试用例
echo "运行测试用例: Bitcoin puzzle 40 (范围: e9ae493300:e9ae493400)"
echo "预期结果: 应该找到私钥 E9AE4933D6"

# 运行测试
timeout 30s ./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range e9ae493300:e9ae493400 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv

# 检查结果
if [ -f "Found.txt" ]; then
    echo "检查Found.txt文件..."
    cat Found.txt
else
    echo "未找到Found.txt文件"
fi

echo "测试完成"