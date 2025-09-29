# SPDX-License-Identifier: GPL-3.0-or-later
# Source-Fusion Provenance: License audit script created for KeyHunt-CUDA CI (2024-09-29).

"""Validate third-party Python dependencies against license matrix."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set


LICENSE_MATRIX = Path("docs/license_matrix.md")
TOOLS_WHITELIST = {"pip", "setuptools", "wheel", "pip-licenses", "PyYAML"}


def load_permitted_spdx() -> Set[str]:
    if not LICENSE_MATRIX.exists():
        return {"MIT", "GPL-3.0", "GPL-3.0-only", "GPL-3.0-or-later"}

    permitted: Set[str] = set()
    for line in LICENSE_MATRIX.read_text(encoding="utf-8").splitlines():
        if "|" not in line or "SPDX" in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3:
            continue
        spdx = parts[2]
        if spdx and spdx != "SPDX License":
            permitted.add(spdx)
    if not permitted:
        permitted.update({"MIT", "GPL-3.0"})
    return permitted


def collect_pip_licenses() -> List[Dict[str, str]]:
    cmd = ["pip-licenses", "--format=json"]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def main() -> int:
    permitted = load_permitted_spdx()
    packages = collect_pip_licenses()
    failures: List[str] = []

    for pkg in packages:
        name = pkg.get("Name")
        license_name = pkg.get("License")
        if name in TOOLS_WHITELIST:
            continue
        if license_name not in permitted:
            failures.append(f"Package {name} uses disallowed license '{license_name}'")

    if failures:
        print("License compliance failures detected:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("check_licenses: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
