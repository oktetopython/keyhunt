#!/bin/bash

echo "=== gECC核心算法提取工具 ==="

# 1. 分析gECC目录结构
echo "1. 分析gECC目录结构..."
if [ -d "gECC-main" ]; then
    cd gECC-main
    echo "gECC-main目录内容："
    find . -name "*.h" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.cu" | grep -E "(arith|ec|mod)" | head -20
    echo
    cd ..
fi

# 2. 识别核心算法文件
echo "2. 识别核心算法文件..."
if [ -d "gECC-main" ]; then
    cd gECC-main
    echo "椭圆曲线运算文件："
    find . -name "*" | grep -i "ec" | grep -v ".git" | grep -v "build"
    echo
    echo "模运算文件："
    find . -name "*" | grep -i "mod" | grep -v ".git" | grep -v "build"
    echo
    echo "算术运算文件："
    find . -name "*" | grep -i "arith" | grep -v ".git" | grep -v "build"
    echo
    cd ..
fi

# 3. 分析关键函数
echo "3. 分析关键函数..."
if [ -d "gECC-main/include/gecc/arith" ]; then
    cd gECC-main
    echo "在arith目录中查找关键函数："
    grep -r "scalar_mul\|point_add\|point_double\|ec_add\|ec_mul" include/gecc/arith/ | head -10
    echo
    echo "查找模运算函数："
    grep -r "mod_add\|mod_mul\|mod_inv\|fp_" include/gecc/arith/ | head -10
    echo
    cd ..
fi

# 4. 检查CUDA内核
echo "4. 检查CUDA内核..."
if [ -d "gECC-main" ]; then
    cd gECC-main
    echo "CUDA内核文件："
    find . -name "*.cu" -exec grep -l "__global__" {} \; | head -5
    echo
    echo "CUDA内核函数："
    find . -name "*.cu" -exec grep -H "__global__" {} \; | head -5
    echo
    cd ..
fi

# 5. 分析头文件结构
echo "5. 分析头文件结构..."
if [ -d "gECC-main/include" ]; then
    cd gECC-main
    echo "主要头文件："
    find include/ -name "*.h" | head -15
    echo
    echo "核心数据结构："
    grep -r "struct\|class" include/gecc/ | grep -v "namespace" | head -10
    echo
    cd ..
fi

# 6. 检查常量和参数
echo "6. 检查常量和参数..."
if [ -d "gECC-main/scripts" ]; then
    cd gECC-main
    echo "常量生成脚本："
    ls -la scripts/*.py
    echo
    echo "secp256k1相关常量："
    grep -r "secp256k1\|0xFFFFFFFFFFFFFFFE\|0x79BE667EF9DCBBAC" . | head -5
    echo
    cd ..
fi

# 7. 分析测试文件了解API
echo "7. 分析测试文件了解API..."
if [ -d "gECC-main/test" ]; then
    cd gECC-main
    echo "测试文件："
    ls -la test/*.cu
    echo
    echo "API使用示例："
    grep -r "ec_\|fp_\|batch_" test/ | head -10
    echo
    cd ..
fi

echo "=== 分析完成 ==="
