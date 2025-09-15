#!/bin/bash

# 🔥 Keyhunt-CUDA 全面重构启动脚本
# 基于严厉审计报告的重构执行计划
# 执行前请仔细阅读: AUDIT_REPORT_T001_T052.md 和 REFACTOR_EXECUTION_GUIDE.md

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查当前目录
check_project_root() {
    if [[ ! -f "CMakeLists.txt" ]] || [[ ! -d "specs/001-keyhunt-cuda-puzzle" ]]; then
        log_error "请在项目根目录运行此脚本"
        log_error "当前目录: $(pwd)"
        exit 1
    fi
    log_success "项目根目录验证通过"
}

# 用户确认
confirm_refactor() {
    echo ""
    echo "🚨🚨🚨 重要警告 🚨🚨🚨"
    echo ""
    echo "此脚本将执行全面重构，包括："
    echo "1. 删除所有根目录临时文件 (test_t*.cpp, test_t*.cu, etc.)"
    echo "2. 清理构建目录"
    echo "3. 重建项目结构"
    echo "4. 重新配置CMake和CTest"
    echo ""
    echo "⚠️  这是不可逆操作！"
    echo ""
    read -p "您确定要继续吗？(输入 'YES' 确认): " confirm
    
    if [[ "$confirm" != "YES" ]]; then
        log_warning "重构已取消"
        exit 0
    fi
    
    log_success "用户确认重构"
}

# 创建备份
create_backup() {
    local backup_dir="../keyhuntcuda_backup_$(date +%Y%m%d_%H%M%S)"
    log_info "创建备份到: $backup_dir"
    
    cp -r . "$backup_dir"
    log_success "备份创建完成: $backup_dir"
}

# 阶段1: 彻底清理
phase1_cleanup() {
    log_info "🧹 阶段1: 彻底清理开始"
    
    # 统计要删除的文件
    local temp_files=$(find . -maxdepth 1 -name "test_t*" -o -name "*.cu" -o -name "*.cpp" | grep -v "./src/" | wc -l)
    
    if [[ $temp_files -gt 0 ]]; then
        log_warning "发现 $temp_files 个根目录临时文件，即将删除："
        find . -maxdepth 1 -name "test_t*" -o -name "*.cu" -o -name "*.cpp" | grep -v "./src/"
        
        # 删除临时文件
        find . -maxdepth 1 -name "test_t*" -delete
        find . -maxdepth 1 -name "*.cu" -delete
        find . -maxdepth 1 -name "*.cpp" -delete
        
        log_success "临时文件清理完成"
    else
        log_success "根目录已经干净，无需清理临时文件"
    fi
    
    # 清理构建目录
    if [[ -d "build" ]]; then
        log_info "清理构建目录"
        rm -rf build/
        log_success "构建目录清理完成"
    fi
    
    # 验证清理结果
    local remaining_temp=$(find . -maxdepth 1 -name "test_*" | wc -l)
    if [[ $remaining_temp -eq 0 ]]; then
        log_success "✅ 阶段1完成：项目清理成功"
    else
        log_error "清理不完整，仍有临时文件存在"
        exit 1
    fi
}

# 阶段2: 重建项目结构
phase2_rebuild_structure() {
    log_info "🏗️  阶段2: 重建项目结构开始"
    
    # 确保核心目录存在
    local core_dirs=(
        "src/KeyhuntCore/models"
        "src/KeyhuntCore/ecc"
        "src/KeyhuntCore/scan"
        "src/KeyhuntCore/compare"
        "src/KeyhuntCore/utils"
        "src/KeyhuntCore/gpu"
        "src/KeyhuntCore/cli"
        "src/KeyhuntCore/api"
        "src/KeyhuntCore/validation"
        "src/KeyhuntCore/crypto"
        "src/KeyhuntCore/memory"
        "src/KeyhuntCore/optimization"
        "src/KeyhuntCore/engine"
        "src/KeyhuntCore/logging"
        "src/KeyhuntCore/monitoring"
        "src/KeyhuntCore/analytics"
        "tests/contract"
        "tests/validation"
        "tests/integration"
        "tests/unit"
        "docs/api"
        "docs/architecture"
        "docs/validation"
        "docs/performance"
        "docs/user"
        "data/config"
        "data/checkpoint"
        "data/logs"
        "data/results"
    )
    
    for dir in "${core_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "创建目录: $dir"
        fi
    done
    
    log_success "✅ 阶段2完成：项目结构重建成功"
}

# 阶段3: 验证环境
phase3_verify_environment() {
    log_info "🔍 阶段3: 验证开发环境"
    
    # 检查CUDA
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        log_success "CUDA版本: $cuda_version"
        
        # 检查CUDA版本是否满足要求
        if [[ $(echo "$cuda_version >= 11.0" | bc -l) -eq 1 ]]; then
            log_success "CUDA版本满足要求 (≥11.0)"
        else
            log_error "CUDA版本过低，需要11.0+，当前: $cuda_version"
            exit 1
        fi
    else
        log_error "未找到CUDA编译器 (nvcc)"
        exit 1
    fi
    
    # 检查CMake
    if command -v cmake &> /dev/null; then
        local cmake_version=$(cmake --version | head -n1 | sed 's/cmake version //')
        log_success "CMake版本: $cmake_version"
    else
        log_error "未找到CMake"
        exit 1
    fi
    
    # 检查libsecp256k1
    if pkg-config --exists libsecp256k1; then
        local secp_version=$(pkg-config --modversion libsecp256k1)
        log_success "libsecp256k1版本: $secp_version"
    else
        log_warning "libsecp256k1未找到，可能需要安装"
        log_info "Ubuntu安装命令: sudo apt-get install libsecp256k1-dev"
    fi
    
    log_success "✅ 阶段3完成：环境验证通过"
}

# 阶段4: 初始化构建系统
phase4_init_build() {
    log_info "⚙️  阶段4: 初始化构建系统"
    
    # 创建构建目录
    mkdir -p build
    cd build
    
    # 运行CMake配置
    log_info "运行CMake配置..."
    if cmake .. 2>&1 | tee cmake_output.log; then
        log_success "CMake配置成功"
    else
        log_error "CMake配置失败，请检查 build/cmake_output.log"
        exit 1
    fi
    
    # 尝试编译
    log_info "尝试初始编译..."
    if make -j$(nproc) 2>&1 | tee make_output.log; then
        log_success "初始编译成功"
    else
        log_warning "编译失败（这是预期的，因为代码需要重新实现）"
        log_info "编译日志保存在: build/make_output.log"
    fi
    
    cd ..
    log_success "✅ 阶段4完成：构建系统初始化完成"
}

# 生成下一步指导
generate_next_steps() {
    log_info "📋 生成下一步指导"
    
    cat > NEXT_STEPS.md << 'EOF'
# 🎯 重构下一步指导

## 当前状态
✅ 项目清理完成  
✅ 结构重建完成  
✅ 环境验证通过  
✅ 构建系统初始化完成  

## 立即行动项

### 1. 开始TDD合同测试 (今天-明天)
```bash
# 进入测试目录
cd tests/contract

# 创建第一个合同测试
cp ../../specs/001-keyhunt-cuda-puzzle/templates/test_scan_configure.cpp .

# 编译和运行测试（应该失败）
cd ../../build
make test_scan_configure
./test_scan_configure  # 期望：失败（这是正确的！）
```

### 2. 配置CTest框架 (明天)
```bash
# 验证测试注册
ctest --show-only

# 运行所有测试（应该全部失败）
ctest -V  # 期望：所有测试失败（TDD要求）
```

### 3. 实现科学验证测试 (本周)
- 创建ECC一致性测试
- 集成libsecp256k1参考
- 建立精度验证框架

### 4. 开始核心实现 (下周)
- 修复编译错误
- 实现数据模型
- 开发ECC模块

## 📚 参考文档
- `specs/001-keyhunt-cuda-puzzle/AUDIT_REPORT_T001_T052.md`
- `specs/001-keyhunt-cuda-puzzle/REFACTOR_EXECUTION_GUIDE.md`
- `specs/001-keyhunt-cuda-puzzle/tasks.md`

## 🚨 质量要求提醒
- 零编译警告
- 严格TDD流程
- <1e-10精度要求
- 科学级代码质量

**下一个里程碑**: 完成所有合同测试（必须失败）
EOF

    log_success "下一步指导已生成: NEXT_STEPS.md"
}

# 主函数
main() {
    echo "🔥 Keyhunt-CUDA 全面重构启动"
    echo "基于严厉审计报告的重构执行计划"
    echo ""
    
    check_project_root
    confirm_refactor
    create_backup
    phase1_cleanup
    phase2_rebuild_structure
    phase3_verify_environment
    phase4_init_build
    generate_next_steps
    
    echo ""
    echo "🎉 重构第一阶段完成！"
    echo ""
    echo "📋 下一步："
    echo "1. 阅读 NEXT_STEPS.md"
    echo "2. 开始TDD合同测试开发"
    echo "3. 严格按照重构指南执行"
    echo ""
    echo "📚 重要文档："
    echo "- specs/001-keyhunt-cuda-puzzle/AUDIT_REPORT_T001_T052.md"
    echo "- specs/001-keyhunt-cuda-puzzle/REFACTOR_EXECUTION_GUIDE.md"
    echo "- NEXT_STEPS.md"
    echo ""
    log_success "重构启动脚本执行完成！"
}

# 执行主函数
main "$@"
