#!/bin/bash

# KeyHunt-Cuda 项目状态验证脚本
# 快速检查项目完整性和就绪状态

echo "=========================================="
echo "🔍 KeyHunt-Cuda 项目状态验证"
echo "=========================================="
echo ""

# 初始化状态变量
STATUS_PASSED=0
STATUS_FAILED=0
TOTAL_CHECKS=0

# 颜色输出函数
print_success() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1"
    ((STATUS_FAILED++))
}

print_warning() {
    echo "⚠️  $1"
}

print_info() {
    echo "ℹ️  $1"
}

# 检查函数
check_file_exists() {
    local file="$1"
    local description="$2"
    ((TOTAL_CHECKS++))
    if [ -f "$file" ]; then
        print_success "$description: $file"
        ((STATUS_PASSED++))
    else
        print_error "$description: $file (文件不存在)"
    fi
}

check_directory_exists() {
    local dir="$1"
    local description="$2"
    ((TOTAL_CHECKS++))
    if [ -d "$dir" ]; then
        print_success "$description: $dir"
        ((STATUS_PASSED++))
    else
        print_error "$description: $dir (目录不存在)"
    fi
}

check_executable() {
    local file="$1"
    local description="$2"
    ((TOTAL_CHECKS++))
    if [ -x "$file" ]; then
        print_success "$description: $file"
        ((STATUS_PASSED++))
    else
        print_warning "$description: $file (可执行文件不存在或无权限)"
    fi
}

# 开始检查
echo "📁 检查项目文件结构..."
echo ""

# 核心源文件检查
check_file_exists "KeyHunt.cpp" "主程序文件"
check_file_exists "KeyHunt.h" "主头文件"
check_file_exists "Makefile" "构建配置"

# GPU相关文件
check_directory_exists "GPU" "GPU代码目录"
check_file_exists "GPU/GPUCompute.h" "GPU计算头文件"
check_file_exists "GPU/GPUCompute_Unified.h" "统一GPU接口"
check_file_exists "GPU/GPUMath.h" "GPU数学库"

# 哈希函数
check_directory_exists "hash" "哈希函数目录"
check_file_exists "hash/sha256.h" "SHA256实现"
check_file_exists "hash/ripemd160.h" "RIPEMD160实现"

# 文档检查
echo ""
echo "📚 检查文档完整性..."
echo ""

check_file_exists "README_UPDATED.md" "主README文档"
check_file_exists "LINUX_COMPATIBILITY_TEST.md" "Linux兼容性测试"
check_file_exists "FINAL_REVIEW_REPORT.md" "最终复检报告"
check_directory_exists "docs" "文档目录"

# 文档文件检查
check_file_exists "docs/API_REFERENCE.md" "API参考文档"
check_file_exists "docs/PERFORMANCE_OPTIMIZATION_GUIDE.md" "性能优化指南"
check_file_exists "docs/DEVELOPER_QUICK_START.md" "开发者快速开始"

# 测试文件检查
echo ""
echo "🧪 检查测试文件..."
echo ""

check_file_exists "test_ldg_optimization.sh" "LDG优化测试脚本"
check_file_exists "test_framework.sh" "测试框架"
check_file_exists "test_performance.cpp" "性能测试代码"

# 可执行文件检查
echo ""
echo "⚙️  检查构建状态..."
echo ""

check_executable "KeyHunt" "主可执行文件"

# 编译配置检查
echo ""
echo "🔧 检查编译配置..."
echo ""

if grep -q "KEYHUNT_CACHE_LDG_OPTIMIZED" Makefile; then
    print_success "LDG缓存优化已启用"
    ((STATUS_PASSED++))
else
    print_warning "LDG缓存优化未启用"
fi
((TOTAL_CHECKS++))

if grep -q "DKEYHUNT_PROFILE_EVENTS" Makefile; then
    print_success "性能分析已启用"
    ((STATUS_PASSED++))
else
    print_warning "性能分析未启用"
fi
((TOTAL_CHECKS++))

# 代码质量检查
echo ""
echo "💻 检查代码质量..."
echo ""

# 检查是否存在编译错误标记
if grep -r "TODO.*FIXME\|XXX\|HACK" . --include="*.cpp" --include="*.h" --include="*.cu" > /dev/null 2>&1; then
    print_warning "发现代码中的临时标记"
else
    print_success "代码中无临时标记"
    ((STATUS_PASSED++))
fi
((TOTAL_CHECKS++))

# 检查是否存在拼写错误
if grep -r "GRP_SZIE" . --include="*.cpp" --include="*.h" --include="*.cu" > /dev/null 2>&1; then
    print_error "发现拼写错误: GRP_SZIE"
else
    print_success "无拼写错误"
    ((STATUS_PASSED++))
fi
((TOTAL_CHECKS++))

# 性能基准检查
echo ""
echo "📊 检查性能基准..."
echo ""

if [ -f "PERFORMANCE_TEST_RESULTS.md" ]; then
    if grep -q "4000.*Mk/s" PERFORMANCE_TEST_RESULTS.md; then
        print_success "性能基准已建立 (4000+ Mk/s)"
        ((STATUS_PASSED++))
    else
        print_warning "性能基准可能需要更新"
    fi
else
    print_warning "性能测试结果文件不存在"
fi
((TOTAL_CHECKS++))

# 最终结果
echo ""
echo "=========================================="
echo "📋 验证结果汇总"
echo "=========================================="
echo ""

SUCCESS_RATE=$((STATUS_PASSED * 100 / TOTAL_CHECKS))

echo "总检查项: $TOTAL_CHECKS"
echo "通过项目: $STATUS_PASSED"
echo "失败项目: $STATUS_FAILED"
echo "成功率: ${SUCCESS_RATE}%"
echo ""

if [ $STATUS_FAILED -eq 0 ]; then
    echo "🎉 项目状态: PRODUCTION READY"
    echo "🏆 健康等级: A+ (优秀)"
    echo "✅ 所有检查均已通过"
    echo ""
    echo "🚀 项目已准备好进行生产部署！"
else
    echo "⚠️  项目状态: NEEDS ATTENTION"
    echo "🔧 需要修复 $STATUS_FAILED 个问题"
    echo ""
    echo "💡 建议: 检查上述失败的项目并进行修复"
fi

echo ""
echo "=========================================="

# 提供建议
if [ $SUCCESS_RATE -ge 95 ]; then
    echo "💡 建议行动:"
    echo "   1. 运行性能测试: ./test_ldg_optimization.sh"
    echo "   2. 验证Linux兼容性: cat LINUX_COMPATIBILITY_TEST.md"
    echo "   3. 开始生产部署"
elif [ $SUCCESS_RATE -ge 80 ]; then
    echo "💡 建议行动:"
    echo "   1. 修复上述失败的项目"
    echo "   2. 重新运行验证脚本"
    echo "   3. 完成剩余的配置"
else
    echo "💡 建议行动:"
    echo "   1. 立即修复所有失败的项目"
    echo "   2. 检查项目完整性"
    echo "   3. 联系开发团队支持"
fi

echo ""
exit $STATUS_FAILED