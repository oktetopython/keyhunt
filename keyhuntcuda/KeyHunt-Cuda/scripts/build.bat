@echo off
setlocal enabledelayedexpansion

REM KeyHunt-Cuda Windows 构建脚本
REM 支持 Windows CMD 和 PowerShell 环境

echo ========================================
echo    KeyHunt-Cuda Windows 构建脚本
echo ========================================

REM 默认参数
set BUILD_TYPE=release
set GPU_SUPPORT=1
set MULTI_GPU=0
set CCAP=75
set DEBUG=0
set CLEAN_BUILD=0

REM 解析命令行参数
:parse_args
if "%1"=="" goto start_build
if "%1"=="-h" goto show_help
if "%1"=="--help" goto show_help
if "%1"=="-d" goto set_debug
if "%1"=="--debug" goto set_debug
if "%1"=="-c" goto set_clean
if "%1"=="--clean" goto set_clean
if "%1"=="-n" goto set_no_gpu
if "%1"=="--no-gpu" goto set_no_gpu
if "%1"=="--cpu-only" goto set_no_gpu
if "%1"=="-m" goto set_multi_gpu
if "%1"=="--multi-gpu" goto set_multi_gpu
if "%1"=="--ccap" (
    set CCAP=%2
    shift
    shift
    goto parse_args
)
echo 未知选项: %1
goto show_help

:set_debug
set DEBUG=1
set BUILD_TYPE=debug
shift
goto parse_args

:set_clean
set CLEAN_BUILD=1
shift
goto parse_args

:set_no_gpu
set GPU_SUPPORT=0
shift
goto parse_args

:set_multi_gpu
set MULTI_GPU=1
shift
goto parse_args

:show_help
echo 用法: %0 [选项]
echo.
echo 选项:
echo   -h, --help              显示此帮助信息
echo   -d, --debug             构建调试版本
echo   -c, --clean             清理构建目录
echo   -n, --no-gpu            构建无GPU版本
echo   -m, --multi-gpu         构建多GPU支持版本
echo   --ccap VALUE            设置GPU计算能力 (默认: 75)
echo   --cpu-only              同 --no-gpu
echo.
echo 示例:
echo   %0                      # 构建默认GPU版本
echo   %0 -d                   # 构建调试版本
echo   %0 -c -m --ccap 86      # 清理并构建多GPU版本，计算能力8.6
echo   %0 --cpu-only           # 构建仅CPU版本
exit /b 0

:start_build
REM 清理构建目录
if %CLEAN_BUILD%==1 (
    echo 清理构建目录...
    make clean
    if errorlevel 1 (
        echo 清理失败
        exit /b 1
    )
    echo 清理完成
)

REM 构建参数
set MAKE_ARGS=

if %GPU_SUPPORT%==1 (
    set MAKE_ARGS=%MAKE_ARGS% gpu=1
    echo 启用 GPU 支持
) else (
    echo 构建仅 CPU 版本
)

if %MULTI_GPU%==1 (
    set MAKE_ARGS=%MAKE_ARGS% MULTI_GPU=1
    echo 启用多 GPU 支持
) else (
    set MAKE_ARGS=%MAKE_ARGS% CCAP=%CCAP%
    echo 设置 GPU 计算能力为 %CCAP%
)

if %DEBUG%==1 (
    set MAKE_ARGS=%MAKE_ARGS% debug=1
    echo 构建调试版本
) else (
    echo 构建发布版本
)

REM 执行构建
echo 执行构建命令: make %MAKE_ARGS% all
make %MAKE_ARGS% all

if errorlevel 1 (
    echo ========================================
    echo 构建失败!
    echo 请检查错误信息并重试
    echo ========================================
    exit /b 1
) else (
    echo ========================================
    echo 构建成功完成!
    echo 可执行文件位置: .\KeyHunt.exe
    echo ========================================
    
    REM 显示构建的文件信息
    if exist ".\KeyHunt.exe" (
        echo 构建文件信息:
        dir ".\KeyHunt.exe"
    )
)