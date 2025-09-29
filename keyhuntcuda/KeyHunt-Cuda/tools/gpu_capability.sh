#!/usr/bin/env bash
# GPU capability helper. Emits JSON describing detected NVIDIA GPUs. Gracefully
# handles systems without nvidia-smi.

set -euo pipefail

timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t rows < <(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null || true)
  if [[ ${#rows[@]} -eq 0 ]]; then
    echo "{\"generated_at\":\"$timestamp\",\"status\":\"nvidia-smi returned no data\",\"gpus\":[]}"
    exit 0
  fi
  echo -n "{\"generated_at\":\"$timestamp\",\"status\":\"ok\",\"gpus\":["
  first=true
  for row in "${rows[@]}"; do
    IFS="," read -r name driver cap <<<"$row"
    name=$(echo "$name" | sed 's/^ *//;s/ *$//')
    driver=$(echo "$driver" | sed 's/^ *//;s/ *$//')
    cap=$(echo "$cap" | sed 's/^ *//;s/ *$//')
    if $first; then
      first=false
    else
      echo -n ","
    fi
    printf '{"name":"%s","driver":"%s","compute_capability":"%s"}' "$name" "$driver" "$cap"
  done
  echo "]}"
else
  echo "{\"generated_at\":\"$timestamp\",\"status\":\"nvidia-smi not available\",\"gpus\":[]}"
fi
