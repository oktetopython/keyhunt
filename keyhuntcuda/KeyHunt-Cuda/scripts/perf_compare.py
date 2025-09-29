# SPDX-License-Identifier: GPL-3.0-or-later
# Source-Fusion Provenance: Original performance regression comparator for KeyHunt-CUDA CI (2024-09-29).

"""Compare collected performance metrics against baseline expectations."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict


BASELINE_PATH = Path(os.getenv("PERF_BASELINE", "tests/perf/baseline.json"))
MEASUREMENT_PATH = Path(os.getenv("PERF_OUTPUT", "perf/latest.json"))


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    baseline = load_json(BASELINE_PATH)
    measurement = load_json(MEASUREMENT_PATH)

    if measurement.get("status") != "ok":
        print(f"perf_compare: measurement status='{measurement.get('status')}', skipping strict comparison.")
        return 0

    metrics_expected: Dict[str, float] = baseline.get("metrics", {})
    tolerance = float(os.getenv("PERF_TOLERANCE", baseline.get("tolerance", 0.02)))
    metrics_observed: Dict[str, float] = measurement.get("metrics", {})

    failures = []
    for key, expected in metrics_expected.items():
        observed = metrics_observed.get(key)
        if observed is None:
            failures.append(f"Missing metric '{key}' in measurement output")
            continue
        if expected == 0:
            continue
        delta = abs(observed - expected) / expected
        if delta > tolerance:
            failures.append(f"Metric {key}: observed={observed:.4f}, expected={expected:.4f}, drift={delta:.4%} > {tolerance:.2%}")

    if failures:
        print("Performance regression detected:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("perf_compare: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
