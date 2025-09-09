#!/bin/bash

# Manual build script to bypass make timestamp issues
cd /mnt/d/mybitcoin/2/keyhunt-backup-20250115/KeyHunt-Cuda

# Create obj directories if they don't exist
mkdir -p obj obj/GPU obj/hash

# Compile all source files manually
echo "Compiling source files..."

g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/Base58.o -c Base58.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/IntGroup.o -c IntGroup.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/Main.o -c Main.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/Bloom.o -c Bloom.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/Random.o -c Random.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/Timer.o -c Timer.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/Int.o -c Int.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/IntMod.o -c IntMod.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/Point.o -c Point.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/SECP256K1.o -c SECP256K1.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/KeyHunt.o -c KeyHunt.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/GPU/GPUGenerate.o -c GPU/GPUGenerate.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/hash/ripemd160.o -c hash/ripemd160.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/hash/sha256.o -c hash/sha256.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/hash/sha512.o -c hash/sha512.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/hash/ripemd160_sse.o -c hash/ripemd160_sse.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/hash/sha256_sse.o -c hash/sha256_sse.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/hash/keccak160.o -c hash/keccak160.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/GmpUtil.o -c GmpUtil.cpp
g++ -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I/usr/local/cuda-12.6/include -o obj/CmdParse.o -c CmdParse.cpp

# Link all object files
echo "Linking object files..."
g++ obj/*.o obj/GPU/*.o obj/hash/*.o -lgmp -lpthread -L/usr/local/cuda-12.6/lib64 -lcudart -o keyhunt-cuda75

echo "Build complete! Checking binary..."
ls -la keyhunt-cuda75 2>/dev/null || echo "Binary not found"

# Test the binary
if [ -f "keyhunt-cuda75" ]; then
    ./keyhunt-cuda75 --version
else
    echo "Binary creation failed"
fi