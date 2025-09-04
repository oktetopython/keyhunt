param(
  [string]$Params = "",
  [int]$DurationSec = 60
)

# Ensure reports directory exists
New-Item -ItemType Directory -Force -Path "reports" | Out-Null

# Check for Nsight Systems CLI
if (-not (Get-Command nsys -ErrorAction SilentlyContinue)) {
  Write-Error "Nsight Systems (nsys) CLI not found in PATH. Please install CUDA Nsight Systems or add it to PATH."
  exit 1
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$base = "reports/keyhunt_$ts"

# Timeline profile
$rep = "$base.timeline"
$cmd = "nsys profile -o $rep --force-overwrite=true -t cuda,osrt,nvtx --cuda-memory-usage=true --duration=$DurationSec .\KeyHunt.exe $Params"
Write-Host "[RUN] $cmd" -ForegroundColor Cyan
Invoke-Expression $cmd

# Export text stats for quick sharing
$statsTxt = "$rep.stats.txt"
$statsCmd = "nsys stats $rep.nsys-rep > $statsTxt"
Write-Host "[RUN] $statsCmd" -ForegroundColor Cyan
cmd /c $statsCmd | Out-Null

Write-Host "[DONE] Nsight Systems run complete" -ForegroundColor Green
Write-Host "Artifacts:" -ForegroundColor Green
Write-Host " - $rep.nsys-rep (open with Nsight Systems GUI)"
Write-Host " - $statsTxt (text summary)"

