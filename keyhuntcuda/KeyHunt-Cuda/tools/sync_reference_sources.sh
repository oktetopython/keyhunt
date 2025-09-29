#!/usr/bin/env bash
#
# Synchronise reference sources and store reproducible snapshots.
#
# Usage:
#   tools/sync_reference_sources.sh --apply [--config path] [--output-dir path] [--report path]
#
# The script archives the configured directories at the current HEAD, writes
# SHA256 manifests, and appends a summary line to the source fusion report.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sync_reference_sources.sh --apply [options]

Options:
  --config <path>       Path to reference list (default: tools/reference_sources.yml)
  --output-dir <path>   Root directory to store snapshots (default: src/reference_snapshots)
  --report <path>       Markdown report file to append snapshot info
                        (default: docs/source_fusion_report.md)
  -h, --help            Show this help message and exit
EOF
}

CONFIG=""
OUTPUT_DIR=""
REPORT_FILE=""

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      # no-op flag for explicit intent
      shift
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --report)
      REPORT_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ROOT=$(git rev-parse --show-toplevel)
CONFIG=${CONFIG:-"$ROOT/tools/reference_sources.yml"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/src/reference_snapshots"}
REPORT_FILE=${REPORT_FILE:-"$ROOT/docs/source_fusion_report.md"}

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
snapshot_dir="$OUTPUT_DIR/$timestamp"

mkdir -p "$snapshot_dir"

if [[ -f "$CONFIG" ]]; then
  while IFS= read -r path || [[ -n "$path" ]]; do
    [[ -z "$path" || "$path" =~ ^# ]] && continue
    # ensure destination directory exists
    dest_dir="$snapshot_dir/$(dirname "$path")"
    mkdir -p "$dest_dir"
    if git rev-parse --verify "HEAD:$path" >/dev/null 2>&1; then
      git archive --format=tar HEAD "$path" | tar -xC "$snapshot_dir"
    else
      echo "warning: path '$path' not found at HEAD, skipping" >&2
    fi
  done <"$CONFIG"
else
  echo "warning: reference config '$CONFIG' not found; creating empty snapshot" >&2
fi

pushd "$snapshot_dir" >/dev/null
if find . -type f | read -r; then
  find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
else
  : > SHA256SUMS.txt
fi
popd >/dev/null

snapshot_hash=$(sha256sum "$snapshot_dir/SHA256SUMS.txt" | awk '{print $1}')
commit_hash=$(git rev-parse HEAD)

mkdir -p "$(dirname "$REPORT_FILE")"
if [[ ! -f "$REPORT_FILE" ]]; then
  cat <<'EOF' >"$REPORT_FILE"
| Snapshot | Digest | Commit |
|----------|--------|--------|
EOF
fi

printf '| %s | %s | %s |%s' "$timestamp" "$snapshot_hash" "$commit_hash" $'\n' >>"$REPORT_FILE"

cat <<EOF
Snapshot stored at: $snapshot_dir
Snapshot digest    : $snapshot_hash
Referenced commit  : $commit_hash
EOF
