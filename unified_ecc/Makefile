# Makefile for Unified ECC Interface
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude
LDFLAGS =

# 条件编译标志
CXXFLAGS += -DWITH_LIGHTWEIGHT_ECC

# 包含路径
INCLUDES = -Iinclude -I../lightweight_ecc_v2/include

# 源文件
SOURCES = src/ecc_factory.cpp src/config_manager.cpp
TEST_SOURCES = test/test_unified_ecc.cpp
MATH_TEST_SOURCES = test/test_math_common.cpp
CONFIG_TEST_SOURCES = test/test_config.cpp

# 对象文件
OBJECTS = $(SOURCES:.cpp=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)
MATH_TEST_OBJECTS = $(MATH_TEST_SOURCES:.cpp=.o)
CONFIG_TEST_OBJECTS = $(CONFIG_TEST_SOURCES:.cpp=.o)

# 可执行文件
TEST_TARGET = test_unified_ecc
MATH_TEST_TARGET = test_math_common
CONFIG_TEST_TARGET = test_config

# 库路径
LIB_PATH = ../lightweight_ecc_v2
LDFLAGS += -L$(LIB_PATH) -llightweight_ecc

# 默认目标
all: $(TEST_TARGET) $(MATH_TEST_TARGET) $(CONFIG_TEST_TARGET)

# 编译规则
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 测试可执行文件
$(TEST_TARGET): $(OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(TEST_OBJECTS) -o $(TEST_TARGET) $(LDFLAGS)

# 数学库测试可执行文件
$(MATH_TEST_TARGET): $(MATH_TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $(MATH_TEST_OBJECTS) -o $(MATH_TEST_TARGET) $(LDFLAGS)

# 配置测试可执行文件
$(CONFIG_TEST_TARGET): src/config_manager.o $(CONFIG_TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) src/config_manager.o $(CONFIG_TEST_OBJECTS) -o $(CONFIG_TEST_TARGET) $(LDFLAGS)

# 运行测试
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# 运行数学库测试
test_math: $(MATH_TEST_TARGET)
	./$(MATH_TEST_TARGET)

# 运行配置测试
run_config_test: $(CONFIG_TEST_TARGET)
	./$(CONFIG_TEST_TARGET)

# 清理
clean:
	rm -f $(OBJECTS) $(TEST_OBJECTS) $(MATH_TEST_OBJECTS) $(CONFIG_TEST_OBJECTS) $(TEST_TARGET) $(MATH_TEST_TARGET) $(CONFIG_TEST_TARGET)

# 调试版本
debug: CXXFLAGS += -g -O0
debug: clean all

# 发布版本
release: CXXFLAGS += -O3 -DNDEBUG
release: clean all

.PHONY: all test clean debug release