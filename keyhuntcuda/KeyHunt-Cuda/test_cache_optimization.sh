#!/bin/bash

# 测试缓存优化的脚本

echo "开始测试缓存优化..."

# 检查KeyHunt可执行文件是否存在
if [ ! -f "./KeyHunt" ]; then
    echo "错误: KeyHunt可执行文件不存在"
    exit 1
fi

# 使用一个已知的小范围进行测试，以验证缓存优化是否正常工作
echo "运行测试用例: Bitcoin puzzle 40 (范围: e9ae493300:e9ae493400)"
echo "预期结果: 应该找到私钥 E9AE4933D6，并且有性能提升"

# 清理之前的Found.txt文件
rm -f Found.txt

# 运行测试，启用性能分析
timeout 30s ./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range e9ae493300:e9ae493400 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv 2>&1 | grep -E "(PROFILE|Kernel execution time)"

# 检查结果
if [ -f "Found.txt" ]; then
    echo "检查Found.txt文件..."
    cat Found.txt
else
    echo "未找到Found.txt文件"
fi

echo "缓存优化测试完成"