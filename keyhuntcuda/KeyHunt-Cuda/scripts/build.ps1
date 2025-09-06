# KeyHunt-Cuda PowerShell 构建脚本
# 支持 Windows PowerShell 和 PowerShell Core

Write-Host "========================================" -ForegroundColor Blue
Write-Host "    KeyHunt-Cuda PowerShell 构建脚本" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue

# 默认参数
$BuildType = "release"
$GpuSupport = $true
$MultiGpu = $false
$Ccap = 75
$Debug = $false
$CleanBuild = $false

# 解析命令行参数
for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        {($_ -eq "-h") -or ($_ -eq "--help")} {
            Write-Host "用法: .\build.ps1 [选项]"
            Write-Host ""
            Write-Host "选项:"
            Write-Host "  -h, --help              显示此帮助信息"
            Write-Host "  -d, --debug             构建调试版本"
            Write-Host "  -c, --clean             清理构建目录"
            Write-Host "  -n, --no-gpu            构建无GPU版本"
            Write-Host "  -m, --multi-gpu         构建多GPU支持版本"
            Write-Host "  --ccap VALUE            设置GPU计算能力 (默认: 75)"
            Write-Host "  --cpu-only              同 --no-gpu"
            Write-Host ""
            Write-Host "示例:"
            Write-Host "  .\build.ps1                      # 构建默认GPU版本"
            Write-Host "  .\build.ps1 -d                   # 构建调试版本"
            Write-Host "  .\build.ps1 -c -m --ccap 86      # 清理并构建多GPU版本，计算能力8.6"
            Write-Host "  .\build.ps1 --cpu-only           # 构建仅CPU版本"
            exit 0
        }
        {($_ -eq "-d") -or ($_ -eq "--debug")} {
            $Debug = $true
            $BuildType = "debug"
        }
        {($_ -eq "-c") -or ($_ -eq "--clean")} {
            $CleanBuild = $true
        }
        {($_ -eq "-n") -or ($_ -eq "--no-gpu") -or ($_ -eq "--cpu-only")} {
            $GpuSupport = $false
        }
        {($_ -eq "-m") -or ($_ -eq "--multi-gpu")} {
            $MultiGpu = $true
        }
        "--ccap" {
            $i++
            if ($i -lt $args.Count) {
                $Ccap = $args[$i]
            } else {
                Write-Host "错误: --ccap 需要一个值" -ForegroundColor Red
                exit 1
            }
        }
        default {
            Write-Host "未知选项: $($args[$i])" -ForegroundColor Red
            Write-Host "使用 -h 或 --help 查看帮助信息" -ForegroundColor Yellow
            exit 1
        }
    }
}

# 清理构建目录
if ($CleanBuild) {
    Write-Host "清理构建目录..." -ForegroundColor Yellow
    & make clean
    if ($LASTEXITCODE -ne 0) {
        Write-Host "清理失败" -ForegroundColor Red
        exit 1
    }
    Write-Host "清理完成" -ForegroundColor Green
}

# 构建参数
$MakeArgs = @()

if ($GpuSupport) {
    $MakeArgs += "gpu=1"
    Write-Host "启用 GPU 支持" -ForegroundColor Green
} else {
    Write-Host "构建仅 CPU 版本" -ForegroundColor Yellow
}

if ($MultiGpu) {
    $MakeArgs += "MULTI_GPU=1"
    Write-Host "启用多 GPU 支持" -ForegroundColor Green
} else {
    $MakeArgs += "CCAP=$Ccap"
    Write-Host "设置 GPU 计算能力为 $Ccap" -ForegroundColor Green
}

if ($Debug) {
    $MakeArgs += "debug=1"
    Write-Host "构建调试版本" -ForegroundColor Green
} else {
    Write-Host "构建发布版本" -ForegroundColor Green
}

# 执行构建
$MakeCommand = "make"
$AllArgs = $MakeArgs + @("all")
Write-Host "执行构建命令: $MakeCommand $($AllArgs -join ' ')" -ForegroundColor Blue
& $MakeCommand $AllArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "构建成功完成!" -ForegroundColor Green
    Write-Host "可执行文件位置: .\KeyHunt.exe" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    # 显示构建的文件信息
    if (Test-Path ".\KeyHunt.exe") {
        Write-Host "构建文件信息:" -ForegroundColor Blue
        Get-ChildItem ".\KeyHunt.exe" | Format-Table Name, Length, LastWriteTime
    }
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "构建失败!" -ForegroundColor Red
    Write-Host "请检查错误信息并重试" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    exit 1
}