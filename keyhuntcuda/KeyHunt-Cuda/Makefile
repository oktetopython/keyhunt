#---------------------------------------------------------------------
# Makefile for KeyHunt
#
# Author : Jean-Luc PONS

CXX = g++

SRC = Base58.cpp IntGroup.cpp Main.cpp Bloom.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      KeyHunt.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp hash/keccak160.cpp GmpUtil.cpp CmdParse.cpp

OBJDIR = obj

# Regardless of 'gpu' flag, GPUEngine.o is now always included
OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o Main.o Bloom.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o KeyHunt.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o hash/keccak160.o \
        GPU/GPUEngine.o GPU/GPUCompute.o GPU/GPUEngine_Unified.o GPU/GPUGlobals.o \
        GPU/GPUKernelsPuzzle71.o GPU/GPUKernelsPuzzle71_Phase3.o GmpUtil.o CmdParse.o)

# CUDA路径检测和设置
# 检测系统中的CUDA安装路径
ifeq ($(OS),Windows_NT)
    # Windows系统
    ifeq ($(CUDA_PATH),)
        # 如果未设置CUDA_PATH环境变量，使用默认路径
        CUDA       = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
    else
        CUDA       = $(CUDA_PATH)
    endif
    CXXCUDA    = g++
else
    # Linux/Unix系统
    # 检查CUDA_HOME环境变量
    ifdef CUDA_HOME
        CUDA       = $(CUDA_HOME)
    else
        # 检查系统路径中的nvcc
        CUDA_BIN := $(shell which nvcc 2>/dev/null)
        ifneq ($(CUDA_BIN),)
            # 从nvcc路径推断CUDA安装目录
            CUDA       = $(shell dirname $(shell dirname $(CUDA_BIN)))
        else
            # 默认路径
            CUDA       = /usr/local/cuda
        endif
    endif
    CXXCUDA    = /usr/bin/g++-12
endif

# 如果上述方法都无法确定CUDA路径，使用默认值
ifeq ($(CUDA),)
    CUDA       = /usr/local/cuda
endif

NVCC       = $(CUDA)/bin/nvcc

# GPU Architecture Support
# Default to compute capability 75 if not specified
CCAP = 75

# Support multiple GPU architectures for broader compatibility
# Compute Capability 7.5: RTX 20xx series, GTX 16xx series, Tesla T4
# Compute Capability 8.0: RTX 30xx series (A100, RTX 3080, etc.)
# Compute Capability 8.6: RTX 30xx series (RTX 3050, 3060, 3070, 3080 Ti, 3090)
# Compute Capability 8.9: RTX 40xx series (RTX 4060, 4070)
# Compute Capability 9.0: RTX 40xx series (RTX 4080, 4090, H100, H20)

ifdef MULTI_GPU
# Multi-GPU support: compile for multiple architectures
GPU_ARCHS = -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89 \
           -gencode arch=compute_90,code=sm_90
else
# Single GPU support: use specified CCAP = 75
ccap = $(CCAP)
GPU_ARCHS  = -gencode=arch=compute_$(ccap),code=sm_$(ccap)
endif

# Always compile with GPU support now
ifdef debug
CXXFLAGS   = -DWITHGPU -m64 -msse2 -mssse3 -msse4.1 -msse4.2 -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS   = -DWITHGPU -m64 -msse2 -mssse3 -msse4.1 -msse4.2 -Wno-write-strings -O2 -I. -I$(CUDA)/include
endif
LFLAGS     = -lgmp -lpthread -L$(CUDA)/lib64 -lcudart -lcudadevrt
# 添加允许不支持的编译器标志以解决GCC版本问题，以及标准库相关标志
# 添加额外的标志来解决浮点类型问题
NVCCFLAGS  = -DKEYHUNT_PROFILE_EVENTS -DKEYHUNT_CACHE_LDG_OPTIMIZED -allow-unsupported-compiler --std=c++14 -extended-lambda -Wno-deprecated-gpu-targets -rdc=true --compiler-options "-D_GLIBCXX_USE_CXX11_ABI=0"

# PUZZLE71 mode support
ifdef PUZZLE71
NVCCFLAGS += -DPUZZLE71_MODE -DUSE_ENDOMORPHISM
CXXFLAGS += -DPUZZLE71_MODE -DUSE_ENDOMORPHISM
endif

#--------------------------------------------------------------------

# GPUEngine.o compilation rule is now unconditional
ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(CUDA)/include $(GPU_ARCHS) $(NVCCFLAGS) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
else
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I. -I$(CUDA)/include $(GPU_ARCHS) $(NVCCFLAGS) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif

$(OBJDIR)/GPU/GPUCompute.o: GPU/GPUCompute.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I. -I$(CUDA)/include $(GPU_ARCHS) $(NVCCFLAGS) -o $(OBJDIR)/GPU/GPUCompute.o -c GPU/GPUCompute.cu

$(OBJDIR)/GPU/GPUEngine_Unified.o: GPU/GPUEngine_Unified.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I. -I$(CUDA)/include $(GPU_ARCHS) $(NVCCFLAGS) -o $(OBJDIR)/GPU/GPUEngine_Unified.o -c GPU/GPUEngine_Unified.cu

$(OBJDIR)/GPU/GPUGlobals.o: GPU/GPUGlobals.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I. -I$(CUDA)/include $(GPU_ARCHS) $(NVCCFLAGS) -o $(OBJDIR)/GPU/GPUGlobals.o -c GPU/GPUGlobals.cu

$(OBJDIR)/GPU/GPUKernelsPuzzle71.o: GPU/GPUKernelsPuzzle71.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I. -I$(CUDA)/include $(GPU_ARCHS) $(NVCCFLAGS) -o $(OBJDIR)/GPU/GPUKernelsPuzzle71.o -c GPU/GPUKernelsPuzzle71.cu

$(OBJDIR)/GPU/GPUKernelsPuzzle71_Phase3.o: GPU/GPUKernelsPuzzle71_Phase3.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I. -I$(CUDA)/include $(GPU_ARCHS) $(NVCCFLAGS) -o $(OBJDIR)/GPU/GPUKernelsPuzzle71_Phase3.o -c GPU/GPUKernelsPuzzle71_Phase3.cu

# Special rule for SSE-optimized files
$(OBJDIR)/hash/sha256_sse.o: hash/sha256_sse.cpp
	$(CXX) $(CXXFLAGS) -march=native -o $@ -c $<

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: KeyHunt

KeyHunt: $(OBJET)
	@echo Making KeyHunt...
	$(NVCC) -rdc=true $(GPU_ARCHS) $(OBJET) $(LFLAGS) -o KeyHunt

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p GPU

$(OBJDIR)/hash: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p hash

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/hash/*.o