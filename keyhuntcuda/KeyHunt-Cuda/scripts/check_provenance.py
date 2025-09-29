# SPDX-License-Identifier: GPL-3.0-or-later
# Source-Fusion Provenance: Original compliance script authored for KeyHunt-CUDA (2024-09-29).

"""Ensure modified source files retain provenance headers."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


HEADER_KEYWORDS = ("Provenance", "Source-Fusion", "SPDX-License-Identifier")
CHECK_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".py", ".sh"}
MAX_SCAN_LINES = 60


@dataclass
class DiffEntry:
    status: str
    path: Path


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _compute_diff() -> List[DiffEntry]:
    diff_range = os.getenv("CI_DIFF_RANGE")
    if diff_range:
        diff = _run(["git", "diff", "--name-status", diff_range])
        return _parse(diff.stdout)

    base_sha = os.getenv("CI_BASE_SHA")
    if base_sha:
        diff = _run(["git", "diff", "--name-status", base_sha, "HEAD"])
        return _parse(diff.stdout)

    base_branch = os.getenv("GITHUB_BASE_REF")
    if base_branch:
        _run(["git", "fetch", "origin", base_branch, "--depth", "1"])
        merge_base = _run(["git", "merge-base", "HEAD", f"origin/{base_branch}"]).stdout.strip()
    else:
        _run(["git", "fetch", "origin", "main", "--depth", "1"])
        merge_base = _run(["git", "merge-base", "HEAD", "origin/main"]).stdout.strip()

    diff = _run(["git", "diff", "--name-status", merge_base, "HEAD"])
    return _parse(diff.stdout)


def _parse(payload: str) -> List[DiffEntry]:
    entries: List[DiffEntry] = []
    for line in payload.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        status, path = parts
        entries.append(DiffEntry(status=status.strip(), path=Path(path.strip())))
    return entries


def needs_check(entry: DiffEntry) -> bool:
    if not entry.status or entry.status[0] not in {"A", "M", "R"}:
        return False
    ext = entry.path.suffix.lower()
    return ext in CHECK_EXTENSIONS


def has_provenance_header(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(MAX_SCAN_LINES):
                line = fh.readline()
                if not line:
                    break
                if any(keyword in line for keyword in HEADER_KEYWORDS):
                    return True
    except FileNotFoundError:
        return False
    return False


def main() -> int:
    failures: List[str] = []
    for entry in _compute_diff():
        if not needs_check(entry):
            continue
        if not entry.path.exists():
            # Deleted or renamed files handled elsewhere.
            continue
        if not has_provenance_header(entry.path):
            failures.append(f"Missing provenance header: {entry.path}")

    if failures:
        print("Provenance check failed:\n" + "\n".join(f"  - {msg}" for msg in failures), file=sys.stderr)
        return 1

    print("check_provenance: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
