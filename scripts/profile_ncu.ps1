param(
  [string]$Params = "",
  [int]$LaunchCount = 0
)

# Ensure reports directory exists
New-Item -ItemType Directory -Force -Path "reports" | Out-Null

# Check for Nsight Compute CLI
if (-not (Get-Command ncu -ErrorAction SilentlyContinue)) {
  Write-Error "Nsight Compute (ncu) CLI not found in PATH. Please install CUDA Nsight Compute or add it to PATH."
  exit 1
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$base = "reports/keyhunt_$ts"

$kernelPattern = 'compute_keys*'
$sections = 'LaunchStats,Occupancy,SpeedOfLight,MemoryWorkloadAnalysis,RooflineChart'

if ($LaunchCount -gt 0) {
  $rep = "$base.ncu_once"
  $cmd = "ncu --kernel-name-pattern \"$kernelPattern\" --launch-count $LaunchCount --target-processes all --set=full --section $sections -o $rep .\\KeyHunt.exe $Params"
} else {
  $rep = "$base.ncu_full"
  $cmd = "ncu --kernel-name-pattern \"$kernelPattern\" --target-processes all --set=full --section $sections -o $rep .\\KeyHunt.exe $Params"
}

Write-Host "[RUN] $cmd" -ForegroundColor Cyan
Invoke-Expression $cmd

Write-Host "[DONE] Nsight Compute run complete" -ForegroundColor Green
Write-Host "Artifacts:" -ForegroundColor Green
Write-Host " - $rep.ncu-rep (open with Nsight Compute GUI)"

