#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KeyHunt-Cuda 跨平台构建脚本
支持 Windows、Linux 和 macOS 系统
"""

import argparse
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def colorize(text, color):
    """为终端输出添加颜色"""
    colors = {
        'red': '\033[0;31m',
        'green': '\033[0;32m',
        'yellow': '\033[1;33m',
        'blue': '\033[0;34m',
        'purple': '\033[0;35m',
        'cyan': '\033[0;36m',
        'white': '\033[1;37m',
        'reset': '\033[0m'
    }
    
    if platform.system() == 'Windows':
        # Windows 命令行不支持 ANSI 颜色代码
        return text
    
    return f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"

def print_header():
    """打印脚本标题"""
    print(colorize("========================================", 'blue'))
    print(colorize("    KeyHunt-Cuda 跨平台构建脚本", 'blue'))
    print(colorize("========================================", 'blue'))

def detect_os():
    """检测操作系统类型"""
    system = platform.system().lower()
    if system == 'windows':
        return 'windows'
    elif system == 'linux':
        return 'linux'
    elif system == 'darwin':
        return 'macos'
    else:
        return 'unknown'

def check_prerequisites():
    """检查构建前提条件"""
    print(colorize("检查构建前提条件...", 'yellow'))
    
    # 检查 make
    if not shutil.which('make'):
        print(colorize("错误: 未找到 make 命令", 'red'))
        print(colorize("请安装构建工具:", 'yellow'))
        os_type = detect_os()
        if os_type == 'windows':
            print(colorize("  Windows: 安装 Visual Studio 或 MinGW-w64", 'white'))
        elif os_type == 'linux':
            print(colorize("  Linux: sudo apt install build-essential", 'white'))
        elif os_type == 'macos':
            print(colorize("  macOS: 安装 Xcode 命令行工具", 'white'))
        return False
    
    # 检查 g++
    if not shutil.which('g++'):
        print(colorize("警告: 未找到 g++ 命令", 'yellow'))
        print(colorize("某些功能可能无法正常工作", 'yellow'))
    
    # 检查 CUDA (如果启用 GPU)
    if not args.cpu_only:
        nvcc_path = shutil.which('nvcc')
        if nvcc_path:
            try:
                result = subprocess.run(['nvcc', '--version'], 
                                      capture_output=True, text=True, check=True)
                print(colorize(f"找到 NVCC: {nvcc_path}", 'green'))
            except subprocess.CalledProcessError:
                print(colorize("警告: NVCC 版本检查失败", 'yellow'))
        else:
            print(colorize("警告: 未找到 NVCC 命令", 'yellow'))
            print(colorize("如果需要 GPU 支持，请安装 CUDA Toolkit", 'yellow'))
    
    return True

def run_make_command(args_list):
    """运行 make 命令"""
    try:
        print(colorize(f"执行命令: {' '.join(args_list)}", 'blue'))
        result = subprocess.run(args_list, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(colorize(f"命令执行失败: {e}", 'red'))
        return False
    except FileNotFoundError:
        print(colorize("错误: 未找到 make 命令", 'red'))
        return False

def main():
    """主函数"""
    print_header()
    
    # 检查前提条件
    if not check_prerequisites():
        sys.exit(1)
    
    # 构建参数
    make_args = []
    
    # 清理构建目录
    if args.clean:
        print(colorize("清理构建目录...", 'yellow'))
        if not run_make_command(['make', 'clean']):
            print(colorize("清理失败", 'red'))
            sys.exit(1)
        print(colorize("清理完成", 'green'))
    
    # GPU 支持
    if not args.cpu_only:
        make_args.append('gpu=1')
        print(colorize("启用 GPU 支持", 'green'))
    else:
        print(colorize("构建仅 CPU 版本", 'yellow'))
    
    # 多 GPU 支持
    if args.multi_gpu:
        make_args.append('MULTI_GPU=1')
        print(colorize("启用多 GPU 支持", 'green'))
    else:
        make_args.append(f'CCAP={args.ccap}')
        print(colorize(f"设置 GPU 计算能力为 {args.ccap}", 'green'))
    
    # 调试版本
    if args.debug:
        make_args.append('debug=1')
        print(colorize("构建调试版本", 'green'))
    else:
        print(colorize("构建发布版本", 'green'))
    
    # 执行构建
    build_args = ['make'] + make_args + ['all']
    if run_make_command(build_args):
        print(colorize("========================================", 'green'))
        print(colorize("构建成功完成!", 'green'))
        
        # 检查可执行文件
        exe_name = 'KeyHunt.exe' if detect_os() == 'windows' else 'KeyHunt'
        if os.path.exists(exe_name):
            print(colorize(f"可执行文件位置: ./{exe_name}", 'green'))
            # 显示文件信息
            try:
                file_stat = os.stat(exe_name)
                print(colorize(f"文件大小: {file_stat.st_size} 字节", 'blue'))
                print(colorize(f"修改时间: {file_stat.st_mtime}", 'blue'))
            except Exception as e:
                print(colorize(f"无法获取文件信息: {e}", 'yellow'))
        else:
            print(colorize("警告: 未找到可执行文件", 'yellow'))
            
        print(colorize("========================================", 'green'))
    else:
        print(colorize("========================================", 'red'))
        print(colorize("构建失败!", 'red'))
        print(colorize("请检查错误信息并重试", 'red'))
        print(colorize("========================================", 'red'))
        sys.exit(1)

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='KeyHunt-Cuda 跨平台构建脚本')
    parser.add_argument('-d', '--debug', action='store_true', 
                       help='构建调试版本')
    parser.add_argument('-c', '--clean', action='store_true', 
                       help='清理构建目录')
    parser.add_argument('-n', '--cpu-only', '--no-gpu', action='store_true', 
                       help='构建仅 CPU 版本')
    parser.add_argument('-m', '--multi-gpu', action='store_true', 
                       help='构建多 GPU 支持版本')
    parser.add_argument('--ccap', type=int, default=75, 
                       help='设置 GPU 计算能力 (默认: 75)')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='显示详细输出')
    
    args = parser.parse_args()
    
    # 运行主函数
    main()