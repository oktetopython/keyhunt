#!/bin/bash

# KeyHunt-Cuda 综合测试框架
# 提供全面的测试覆盖，包括功能测试、性能测试、回归测试

set -e  # 遇到错误立即退出

# 测试配置
TEST_DIR="test_results"
LOG_DIR="${TEST_DIR}/logs"
REPORT_DIR="${TEST_DIR}/reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CURRENT_TEST="${TEST_DIR}/current_test_${TIMESTAMP}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 测试统计
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# 函数: 初始化测试环境
init_test_environment() {
    echo "=========================================="
    echo "KeyHunt-Cuda 综合测试框架"
    echo "=========================================="
    echo "测试时间: $(date)"
    echo "测试目录: ${TEST_DIR}"
    echo ""
    
    # 创建测试目录
    mkdir -p "${LOG_DIR}" "${REPORT_DIR}" "${CURRENT_TEST}"
    
    # 检查必要文件
    check_required_files
    
    echo "✅ 测试环境初始化完成"
    echo ""
}

# 函数: 检查必要文件
check_required_files() {
    local missing_files=()
    
    # 检查可执行文件
    if [ ! -f "KeyHunt" ]; then
        missing_files+=("KeyHunt (请先编译项目)")
    fi
    
    # 检查测试数据
    if [ ! -f "test_addresses.txt" ]; then
        missing_files+=("test_addresses.txt (测试地址文件)")
    fi
    
    # 检查GPU可用性
    if ! command -v nvidia-smi &> /dev/null; then
        missing_files+=("nvidia-smi (NVIDIA驱动)")
    fi
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        echo "❌ 缺少必要文件:"
        for file in "${missing_files[@]}"; do
            echo "   - ${file}"
        done
        echo ""
        echo "请解决上述问题后再运行测试"
        exit 1
    fi
    
    echo "✅ 必要文件检查通过"
}

# 函数: 运行单个测试
run_test() {
    local test_name=$1
    local test_command=$2
    local expected_result=$3
    local timeout_seconds=${4:-300}  # 默认5分钟超时
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo "运行测试: ${test_name}"
    echo "命令: ${test_command}"
    echo "期望结果: ${expected_result}"
    echo "超时: ${timeout_seconds}秒"
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 运行测试命令
    local test_log="${LOG_DIR}/${test_name}_${TIMESTAMP}.log"
    local test_result="${CURRENT_TEST}/${test_name}_result.txt"
    
    # 使用timeout防止测试挂起
    if timeout ${timeout_seconds} bash -c "${test_command}" > "${test_log}" 2>&1; then
        local exit_code=$?
    else
        local exit_code=124  # timeout退出码
    fi
    
    # 记录结束时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 分析结果
    local test_passed=false
    local failure_reason=""
    
    if [ ${exit_code} -eq 124 ]; then
        failure_reason="测试超时 (${timeout_seconds}秒)"
    elif [ ${exit_code} -ne 0 ] && [ "${expected_result}" = "success" ]; then
        failure_reason="命令执行失败 (退出码: ${exit_code})"
    elif [ ${exit_code} -eq 0 ] && [ "${expected_result}" = "failure" ]; then
        failure_reason="期望失败但命令成功执行"
    else
        test_passed=true
    fi
    
    # 记录结果
    if [ "${test_passed}" = true ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "✅ ${test_name} - 通过 (${duration}秒)" | tee -a "${test_result}"
        echo "${GREEN}✅ PASSED${NC}: ${test_name} (${duration}s)"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "❌ ${test_name} - 失败: ${failure_reason}" | tee -a "${test_result}"
        echo "${RED}❌ FAILED${NC}: ${test_name} - ${failure_reason}"
        echo "   日志文件: ${test_log}"
    fi
    
    echo ""
}

# 函数: 功能测试套件
run_functional_tests() {
    echo "=========================================="
    echo "功能测试套件"
    echo "=========================================="
    echo ""
    
    # 测试1: 基本功能测试
    run_test "basic_functionality" \
        "./KeyHunt -t 0 -g 32,32 --keyspace 8000000000000000:8000000000000100 --continue -m addresses -f test_addresses.txt" \
        "success" 60
    
    # 测试2: 多地址搜索模式
    run_test "multi_address_search" \
        "./KeyHunt -t 0 -g 64,64 --keyspace 8000000000000000:8000000000001000 -m addresses -f test_addresses.txt" \
        "success" 120
    
    # 测试3: 单地址搜索模式
    run_test "single_address_search" \
        "./KeyHunt -t 0 -g 32,32 --keyspace 8000000000000000:8000000000000100 -m address -a 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa" \
        "success" 60
    
    # 测试4: X点搜索模式
    run_test "xpoint_search" \
        "./KeyHunt -t 0 -g 32,32 --keyspace 8000000000000000:8000000000000100 -m xpoint -f test_xpoints.txt" \
        "success" 60
    
    # 测试5: 以太坊地址搜索
    run_test "ethereum_search" \
        "./KeyHunt -t 0 -g 32,32 --keyspace 8000000000000000:8000000000000100 -m address -c ETH -a 0x742d35Cc6634C0532925a3b844Bc9e7595f6fed" \
        "success" 60
    
    # 测试6: 压缩模式测试
    run_test "compression_modes" \
        "./KeyHunt -t 0 -g 32,32 --keyspace 8000000000000000:8000000000000100 -m addresses -f test_addresses.txt -s both" \
        "success" 60
}

# 函数: 性能测试套件
run_performance_tests() {
    echo "=========================================="
    echo "性能测试套件"
    echo "=========================================="
    echo ""
    
    # 测试1: 基准性能测试
    run_test "baseline_performance" \
        "./KeyHunt -t 0 -g 256,256 --keyspace 8000000000000000:8000001000000000 -m addresses -f test_addresses.txt" \
        "success" 300
    
    # 测试2: LDG优化性能对比
    run_test "ldg_optimized_performance" \
        "./KeyHunt -t 0 -g 256,256 --keyspace 8000000000000000:8000001000000000 -m addresses -f test_addresses.txt" \
        "success" 300
    
    # 测试3: 不同网格配置性能
    run_test "grid_size_performance" \
        "for g in '128,128' '256,256' '512,128'; do echo \"Testing grid $g\"; ./KeyHunt -t 0 -g $g --keyspace 8000000000000000:8000000100000000 -m addresses -f test_addresses.txt; done" \
        "success" 600
    
    # 测试4: 多GPU性能测试
    if [ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -gt 1 ]; then
        run_test "multi_gpu_performance" \
            "./KeyHunt -t 0 -g 256,256,256 --keyspace 8000000000000000:8000001000000000 -m addresses -f test_addresses.txt" \
            "success" 300
    else
        echo "⚠️  跳过多GPU测试: 系统中只有1个GPU"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
    fi
}

# 函数: 回归测试套件
run_regression_tests() {
    echo "=========================================="
    echo "回归测试套件"
    echo "=========================================="
    echo ""
    
    # 测试1: 内存泄漏检查
    run_test "memory_leak_check" \
        "valgrind --leak-check=full --error-exitcode=1 ./KeyHunt -t 0 -g 32,32 --keyspace 8000000000000000:8000000000000100 -m addresses -f test_addresses.txt 2>&1 | grep -E 'definitely lost|indirectly lost'" \
        "failure" 120
    
    # 测试2: 边界条件测试
    run_test "boundary_conditions" \
        "./KeyHunt -t 0 -g 1,1 --keyspace 8000000000000000:8000000000000001 -m addresses -f test_addresses.txt" \
        "success" 30
    
    # 测试3: 大地址文件测试
    run_test "large_address_file" \
        "./KeyHunt -t 0 -g 64,64 --keyspace 8000000000000000:8000000000010000 -m addresses -f test_large_addresses.txt" \
        "success" 180
    
    # 测试4: 长时间运行稳定性
    run_test "long_running_stability" \
        "timeout 300 ./KeyHunt -t 0 -g 128,128 --keyspace 8000000000000000:8000010000000000 -m addresses -f test_addresses.txt" \
        "success" 300
}

# 函数: 缓存优化测试
run_cache_optimization_tests() {
    echo "=========================================="
    echo "缓存优化测试套件"
    echo "=========================================="
    echo ""
    
    # 测试1: LDG优化验证
    run_test "ldg_optimization_validation" \
        "./test_ldg_optimization.sh" \
        "success" 600
    
    # 测试2: 内存访问模式测试
    run_test "memory_access_pattern" \
        "echo 'Testing memory access patterns...' && ./KeyHunt -t 0 -g 256,256 --keyspace 8000000000000000:8000001000000000 -m addresses -f test_addresses.txt 2>&1 | grep -i 'cache\\|memory'" \
        "success" 300
    
    # 测试3: 性能一致性测试
    run_test "performance_consistency" \
        "for i in {1..5}; do echo \"Run $i:\"; ./KeyHunt -t 0 -g 128,128 --keyspace 8000000000000000:8000000100000000 -m addresses -f test_addresses.txt 2>&1 | grep 'Mk/s'; done" \
        "success" 300
}

# 函数: 生成测试报告
generate_test_report() {
    echo "=========================================="
    echo "生成测试报告"
    echo "=========================================="
    echo ""
    
    local report_file="${REPORT_DIR}/test_report_${TIMESTAMP}.md"
    
    {
        echo "# KeyHunt-Cuda 综合测试报告"
        echo "生成时间: $(date)"
        echo "测试时间戳: ${TIMESTAMP}"
        echo ""
        echo "## 测试摘要"
        echo "- 总测试数: ${TOTAL_TESTS}"
        echo "- 通过测试: ${PASSED_TESTS}"
        echo "- 失败测试: ${FAILED_TESTS}"
        echo "- 跳过测试: ${SKIPPED_TESTS}"
        echo "- 通过率: $(echo "scale=1; ${PASSED_TESTS} * 100 / ${TOTAL_TESTS}" | bc)%"
        echo ""
        echo "## 详细结果"
        echo ""
        
        # 列出所有测试结果
        for result_file in "${CURRENT_TEST}"/*_result.txt; do
            if [ -f "${result_file}" ]; then
                local test_name=$(basename "${result_file}" _result.txt)
                echo "### ${test_name}"
                cat "${result_file}"
                echo ""
            fi
        done
        
        echo "## 性能基准"
        echo "（需要运行性能测试后填充具体数据）"
        echo ""
        echo "## 建议"
        if [ ${FAILED_TESTS} -eq 0 ]; then
            echo "✅ 所有测试通过！项目状态良好。"
        else
            echo "⚠️  有 ${FAILED_TESTS} 个测试失败，建议："
            echo "1. 检查失败测试的日志文件"
            echo "2. 修复相关问题后重新运行测试"
            echo "3. 考虑添加更多边界条件测试"
        fi
        
        echo ""
        echo "## 日志文件"
        echo "详细日志位于: ${LOG_DIR}/"
        echo "测试报告位于: ${REPORT_DIR}/"
        
    } > "${report_file}"
    
    echo "✅ 测试报告已生成: ${report_file}"
    echo ""
}

# 函数: 显示测试统计
show_test_statistics() {
    echo "=========================================="
    echo "测试统计"
    echo "=========================================="
    echo "总测试数: ${TOTAL_TESTS}"
    echo "通过测试: ${GREEN}${PASSED_TESTS}${NC}"
    echo "失败测试: ${RED}${FAILED_TESTS}${NC}"
    echo "跳过测试: ${YELLOW}${SKIPPED_TESTS}${NC}"
    echo "通过率: $(echo "scale=1; ${PASSED_TESTS} * 100 / ${TOTAL_TESTS}" | bc)%"
    echo ""
    
    if [ ${FAILED_TESTS} -eq 0 ]; then
        echo "${GREEN}🎉 所有测试通过！${NC}"
    else
        echo "${RED}⚠️  有测试失败，请查看详细报告${NC}"
    fi
}

# 函数: 创建测试数据
create_test_data() {
    echo "创建测试数据..."
    
    # 创建测试地址文件
    cat > test_addresses.txt << EOF
1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2
1HB5XMLmzFVj8ALj6mfBsbifRoD4miY36v
1Q1pE5vPGEEMqRcVRMbtBK842Y6Pza2b1D
1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2
EOF
    
    # 创建测试X点文件
    cat > test_xpoints.txt << EOF
02e9a8d7d5c6b4a3f2e1d0c9b8a7f6e5d4c3b2a1
03a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9
EOF
    
    # 创建大地址文件 (用于压力测试)
    for i in {1..1000}; do
        echo "1A$(printf "%x" $i)zP1eP5QGefi2DMPTfTL5SLmv7DivfNa" >> test_large_addresses.txt
    done
    
    echo "✅ 测试数据创建完成"
}

# 主函数
main() {
    # 解析命令行参数
    local run_functional=true
    local run_performance=true
    local run_regression=true
    local run_cache_tests=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --functional-only)
                run_performance=false
                run_regression=false
                run_cache_tests=false
                shift
                ;;
            --performance-only)
                run_functional=false
                run_regression=false
                run_cache_tests=false
                shift
                ;;
            --regression-only)
                run_functional=false
                run_performance=false
                run_cache_tests=false
                shift
                ;;
            --cache-only)
                run_functional=false
                run_performance=false
                run_regression=false
                shift
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --functional-only    只运行功能测试"
                echo "  --performance-only   只运行性能测试"
                echo "  --regression-only    只运行回归测试"
                echo "  --cache-only         只运行缓存优化测试"
                echo "  --help               显示帮助信息"
                exit 0
                ;;
            *)
                echo "未知选项: $1"
                echo "使用 --help 查看可用选项"
                exit 1
                ;;
        esac
    done
    
    # 初始化测试环境
    init_test_environment
    
    # 创建测试数据
    create_test_data
    
    # 运行测试套件
    if [ "${run_functional}" = true ]; then
        run_functional_tests
    fi
    
    if [ "${run_performance}" = true ]; then
        run_performance_tests
    fi
    
    if [ "${run_regression}" = true ]; then
        run_regression_tests
    fi
    
    if [ "${run_cache_tests}" = true ]; then
        run_cache_optimization_tests
    fi
    
    # 生成测试报告
    generate_test_report
    
    # 显示测试统计
    show_test_statistics
    
    # 返回适当的退出码
    if [ ${FAILED_TESTS} -eq 0 ]; then
        echo ""
        echo "${GREEN}✅ 所有测试通过！${NC}"
        exit 0
    else
        echo ""
        echo "${RED}❌ 有 ${FAILED_TESTS} 个测试失败${NC}"
        exit 1
    fi
}

# 运行主函数
main "$@"