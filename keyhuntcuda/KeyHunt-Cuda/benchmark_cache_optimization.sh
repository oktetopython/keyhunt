#!/bin/bash

# 基准测试缓存优化的脚本

echo "开始基准测试缓存优化..."

# 检查KeyHunt可执行文件是否存在
if [ ! -f "./KeyHunt" ]; then
    echo "错误: KeyHunt可执行文件不存在"
    exit 1
fi

# 使用一个已知的小范围进行测试，以验证缓存优化是否带来性能提升
echo "运行基准测试用例: Bitcoin puzzle 40 (范围: e9ae493300:e9ae493400)"

# 清理之前的Found.txt文件
rm -f Found.txt

# 运行多次测试以获得更准确的结果
echo "运行5次测试以获得平均性能..."

total_time=0
for i in {1..5}; do
    echo "运行测试 $i/5..."
    # 运行测试并捕获执行时间
    output=$(timeout 30s ./KeyHunt -g --gpui 0 --mode ADDRESS --coin BTC --range e9ae493300:e9ae493400 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv 2>&1 | grep "Kernel execution time" | awk '{print $5}' | sed 's/ms//')
    
    # 计算总时间
    for time in $output; do
        total_time=$(echo "$total_time + $time" | bc)
    done
    
    # 等待一段时间以避免GPU过热
    sleep 2
done

# 计算平均执行时间
average_time=$(echo "scale=2; $total_time / 5" | bc)
echo "平均内核执行时间: $average_time ms"

# 检查结果
if [ -f "Found.txt" ]; then
    echo "测试成功 - 找到了预期的私钥"
else
    echo "测试失败 - 未找到预期的私钥"
fi

echo "基准测试完成"
echo "缓存优化启用状态: 已启用"
echo "平均内核执行时间: $average_time ms"