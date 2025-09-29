# SPDX-License-Identifier: GPL-3.0-or-later
# Source-Fusion Provenance: Original performance harness for KeyHunt-CUDA CI (2024-09-29).

"""Collect deterministic performance signals for regression comparison."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict


DEFAULT_OUTPUT = Path(os.getenv("PERF_OUTPUT", "perf/latest.json"))
SIMPLE_BENCHMARK = Path("build/simple_benchmark")


def ensure_build_directory() -> None:
    SIMPLE_BENCHMARK.parent.mkdir(parents=True, exist_ok=True)


def run_simple_benchmark(num_keys: int) -> Dict[str, float]:
    if not SIMPLE_BENCHMARK.exists():
        compile_cmd = [
            "g++",
            "-std=c++17",
            "-O2",
            "-pthread",
            "simple_benchmark.cpp",
            "-o",
            str(SIMPLE_BENCHMARK),
        ]
        subprocess.run(compile_cmd, check=True)

    run_cmd = [str(SIMPLE_BENCHMARK), str(num_keys)]
    proc = subprocess.run(run_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    metrics: Dict[str, float] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("Hardcoded target:"):
            metrics["hardcoded_speedup"] = float(line.split()[-1].rstrip("x"))
        elif line.startswith("Batch (16):"):
            metrics["batch16_speedup"] = float(line.split()[-1].rstrip("x"))
        elif line.startswith("Batch (32):"):
            metrics["batch32_speedup"] = float(line.split()[-1].rstrip("x"))
        elif line.startswith("Multi-thread:"):
            metrics["multithread_speedup"] = float(line.split()[-1].rstrip("x"))
        elif line.startswith("Combined (all):"):
            metrics["combined_speedup"] = float(line.split()[-1].rstrip("x"))

    if not metrics:
        raise RuntimeError("Failed to parse benchmark output for performance metrics")
    return metrics


def emit_result(payload: Dict[str, object]) -> None:
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"perf_collect: wrote {DEFAULT_OUTPUT}")


def main() -> int:
    ensure_build_directory()

    if os.getenv("ENABLE_PERF_SMOKE") != "1":
        emit_result({
            "status": "skipped",
            "reason": "ENABLE_PERF_SMOKE not set",
        })
        return 0

    num_keys = int(os.getenv("PERF_NUM_KEYS", "100000"))
    metrics = run_simple_benchmark(num_keys)
    emit_result({
        "status": "ok",
        "num_keys": num_keys,
        "metrics": metrics,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
