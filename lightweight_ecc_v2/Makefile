# Makefile for lightweight ECC library
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra -march=native
INCLUDES = -Iinclude
LDFLAGS = 

# 源文件
SOURCES = src/uint256.cpp src/point.cpp src/curve_params.cpp src/mod_op.cpp src/ec_op.cpp src/secp256k1.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# 库文件
LIBRARY = liblightweight_ecc.a

# 测试文件
TEST_SOURCES = test/test_basic_structures.cpp
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)
TEST_TARGET = test_basic_structures

# 核心算法测试文件
CORE_TEST_SOURCES = test/test_core_algorithms.cpp
CORE_TEST_OBJECTS = $(CORE_TEST_SOURCES:.cpp=.o)
CORE_TEST_TARGET = test_core_algorithms

# 修复验证测试文件
FIX_TEST_SOURCES = test/fix_verification.cpp
FIX_TEST_OBJECTS = $(FIX_TEST_SOURCES:.cpp=.o)
FIX_TEST_TARGET = fix_verification

# 调试测试文件
DEBUG_TEST_SOURCES = test/debug_ec_operations.cpp
DEBUG_TEST_OBJECTS = $(DEBUG_TEST_SOURCES:.cpp=.o)
DEBUG_TEST_TARGET = debug_ec_operations

# 交换律测试文件
COMM_TEST_SOURCES = test/commutativity_test.cpp
COMM_TEST_OBJECTS = $(COMM_TEST_SOURCES:.cpp=.o)
COMM_TEST_TARGET = commutativity_test

# 默认目标
all: $(LIBRARY) $(TEST_TARGET) $(CORE_TEST_TARGET) $(FIX_TEST_TARGET) $(DEBUG_TEST_TARGET) $(COMM_TEST_TARGET)

# 创建静态库
$(LIBRARY): $(OBJECTS)
	ar rcs $(LIBRARY) $(OBJECTS)

# 编译基础测试
$(TEST_TARGET): $(TEST_OBJECTS) $(LIBRARY)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TEST_TARGET) $(TEST_OBJECTS) $(LIBRARY) $(LDFLAGS)

# 编译核心算法测试
$(CORE_TEST_TARGET): $(CORE_TEST_OBJECTS) $(LIBRARY)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(CORE_TEST_TARGET) $(CORE_TEST_OBJECTS) $(LIBRARY) $(LDFLAGS)

# 编译修复验证测试
$(FIX_TEST_TARGET): $(FIX_TEST_OBJECTS) $(LIBRARY)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(FIX_TEST_TARGET) $(FIX_TEST_OBJECTS) $(LIBRARY) $(LDFLAGS)

# 编译调试测试
$(DEBUG_TEST_TARGET): $(DEBUG_TEST_OBJECTS) $(LIBRARY)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(DEBUG_TEST_TARGET) $(DEBUG_TEST_OBJECTS) $(LIBRARY) $(LDFLAGS)

# 编译交换律测试
$(COMM_TEST_TARGET): $(COMM_TEST_OBJECTS) $(LIBRARY)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(COMM_TEST_TARGET) $(COMM_TEST_OBJECTS) $(LIBRARY) $(LDFLAGS)

# 编译规则
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 清理
clean:
	rm -f $(OBJECTS) $(TEST_OBJECTS) $(CORE_TEST_OBJECTS) $(FIX_TEST_OBJECTS) $(DEBUG_TEST_OBJECTS) $(COMM_TEST_OBJECTS) $(LIBRARY) $(TEST_TARGET) $(CORE_TEST_TARGET) $(FIX_TEST_TARGET) $(DEBUG_TEST_TARGET) $(COMM_TEST_TARGET)

# 测试
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# 核心算法测试
test_core: $(CORE_TEST_TARGET)
	./$(CORE_TEST_TARGET)

# 修复验证测试
fix: $(FIX_TEST_TARGET)
	./$(FIX_TEST_TARGET)

# 调试测试
debug: $(DEBUG_TEST_TARGET)
	./$(DEBUG_TEST_TARGET)

# 交换律测试
comm: $(COMM_TEST_TARGET)
	./$(COMM_TEST_TARGET)

# 运行所有测试
test_all: $(TEST_TARGET) $(CORE_TEST_TARGET) $(FIX_TEST_TARGET) $(COMM_TEST_TARGET)
	./$(TEST_TARGET)
	./$(FIX_TEST_TARGET)
	./$(COMM_TEST_TARGET)
	./$(CORE_TEST_TARGET)

# 安装
install: $(LIBRARY)
	cp $(LIBRARY) /usr/local/lib/
	cp -r include /usr/local/include/lightweight_ecc

.PHONY: all clean test test_core fix debug comm test_all install
