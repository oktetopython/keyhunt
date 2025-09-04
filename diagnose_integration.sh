#!/bin/bash

echo "=== gECC集成诊断工具 ==="
echo

# 1. 检查gECC是否存在
echo "1. 检查gECC项目..."
if [ -d "gECC-main" ]; then
    echo "✓ gECC-main目录存在"
    cd gECC-main
    if [ -f "CMakeLists.txt" ]; then
        echo "✓ gECC有CMakeLists.txt"
    else
        echo "✗ gECC缺少CMakeLists.txt"
    fi
    if [ -d "include" ]; then
        echo "✓ gECC有include目录"
        ls -la include/
    else
        echo "✗ gECC缺少include目录"
    fi
    if [ -d "src" ]; then
        echo "✓ gECC有src目录"
        ls -la src/
    else
        echo "✗ gECC缺少src目录"
    fi
    cd ..
elif [ -d "gECC" ]; then
    echo "✓ gECC目录存在"
    cd gECC
    if [ -f "CMakeLists.txt" ]; then
        echo "✓ gECC有CMakeLists.txt"
    else
        echo "✗ gECC缺少CMakeLists.txt"
    fi
    if [ -d "include" ]; then
        echo "✓ gECC有include目录"
        ls -la include/
    else
        echo "✗ gECC缺少include目录"
    fi
    cd ..
else
    echo "✗ gECC目录不存在"
fi

echo

# 2. 检查KeyHunt-Cuda
echo "2. 检查KeyHunt-Cuda项目..."
if [ -d "keyhuntcuda/KeyHunt-Cuda" ]; then
    echo "✓ KeyHunt-Cuda目录存在"
    cd keyhuntcuda/KeyHunt-Cuda
    if [ -f "Makefile" ]; then
        echo "✓ KeyHunt-Cuda有Makefile"
    else
        echo "✗ KeyHunt-Cuda缺少Makefile"
    fi
    if [ -d "GPU" ]; then
        echo "✓ KeyHunt-Cuda有GPU目录"
        ls -la GPU/
    else
        echo "✗ KeyHunt-Cuda缺少GPU目录"
    fi
    cd ../..
else
    echo "✗ KeyHunt-Cuda目录不存在"
fi

echo

# 3. 检查编译环境
echo "3. 检查编译环境..."
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA编译器(nvcc)可用"
    nvcc --version | head -n 1
else
    echo "✗ CUDA编译器(nvcc)不可用"
fi

if command -v g++ &> /dev/null; then
    echo "✓ g++编译器可用"
    g++ --version | head -n 1
else
    echo "✗ g++编译器不可用"
fi

echo

# 4. 检查能否编译gECC
echo "4. 尝试编译gECC..."
if [ -d "gECC-main" ]; then
    cd gECC-main
    mkdir -p build
    cd build
    if cmake .. 2>/dev/null; then
        echo "✓ gECC CMake配置成功"
        if make -j$(nproc) 2>/dev/null; then
            echo "✓ gECC编译成功"
            ls -la lib/ 2>/dev/null || ls -la .
        else
            echo "✗ gECC编译失败"
        fi
    else
        echo "✗ gECC CMake配置失败"
    fi
    cd ../..
elif [ -d "gECC" ]; then
    cd gECC
    mkdir -p build
    cd build
    if cmake .. 2>/dev/null; then
        echo "✓ gECC CMake配置成功"
        if make -j$(nproc) 2>/dev/null; then
            echo "✓ gECC编译成功"
            ls -la lib/ 2>/dev/null || ls -la .
        else
            echo "✗ gECC编译失败"
        fi
    else
        echo "✗ gECC CMake配置失败"
    fi
    cd ../..
else
    echo "✗ gECC目录不存在，跳过编译测试"
fi

echo

# 5. 检查能否编译KeyHunt-Cuda
echo "5. 尝试编译KeyHunt-Cuda..."
if [ -d "keyhuntcuda/KeyHunt-Cuda" ]; then
    cd keyhuntcuda/KeyHunt-Cuda
    if make -j$(nproc) 2>/dev/null; then
        echo "✓ KeyHunt-Cuda编译成功"
        ls -la KeyHunt*
    else
        echo "✗ KeyHunt-Cuda编译失败"
    fi
    cd ../..
else
    echo "✗ KeyHunt-Cuda目录不存在，跳过编译测试"
fi

echo
echo "=== 诊断完成 ==="
