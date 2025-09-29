# SPDX-License-Identifier: GPL-3.0-or-later
# Source-Fusion Provenance: Original CI guardrail script authored for KeyHunt-CUDA (2024-09-29).

"""Incremental diff checks for AI-governed CI pipelines.

This script enforces repository rules for newly added or modified files:
  * Rejects filenames that look like anonymous copies (`*_new`, `*_copy`).
  * Flags mock artefacts accidentally committed (paths containing `mock/` etc.).

The diff base can be controlled via environment variables:
  * GITHUB_BASE_REF (branch name in pull requests)
  * CI_BASE_SHA (explicit commit SHA)
  * CI_DIFF_RANGE (explicit git diff range such as `origin/main...HEAD`)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List


FORBIDDEN_SUFFIXES = ("_new", "_copy", "_backup")
FORBIDDEN_DIR_PATTERNS = (
    re.compile(r"(^|/)mocks?(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)__mocks?__(/|$)", re.IGNORECASE),
)


@dataclass
class DiffEntry:
    status: str
    path: str


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def compute_diff_entries() -> List[DiffEntry]:
    diff_range = os.getenv("CI_DIFF_RANGE")
    if diff_range:
        cmd = ["git", "diff", "--name-status", diff_range]
        result = _run(cmd)
        return _parse_diff_output(result.stdout)

    base_sha = os.getenv("CI_BASE_SHA")
    if base_sha:
        cmd = ["git", "diff", "--name-status", base_sha, "HEAD"]
        result = _run(cmd)
        return _parse_diff_output(result.stdout)

    base_branch = os.getenv("GITHUB_BASE_REF")
    if base_branch:
        _run(["git", "fetch", "origin", base_branch, "--depth", "1"])
        merge_base = _run(["git", "merge-base", "HEAD", f"origin/{base_branch}"]).stdout.strip()
    else:
        _run(["git", "fetch", "origin", "main", "--depth", "1"])
        merge_base = _run(["git", "merge-base", "HEAD", "origin/main"]).stdout.strip()

    diff = _run(["git", "diff", "--name-status", merge_base, "HEAD"])
    return _parse_diff_output(diff.stdout)


def _parse_diff_output(output: str) -> List[DiffEntry]:
    entries: List[DiffEntry] = []
    for line in output.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        status, path = parts
        entries.append(DiffEntry(status=status.strip(), path=path.strip()))
    return entries


def _violates_suffix(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(suffix) or lower.endswith(suffix + os.path.splitext(lower)[1]) for suffix in FORBIDDEN_SUFFIXES)


def _violates_mock(path: str) -> bool:
    return any(pattern.search(path) for pattern in FORBIDDEN_DIR_PATTERNS)


def main() -> int:
    entries = compute_diff_entries()
    failures: List[str] = []

    for entry in entries:
        if entry.status and entry.status[0] not in {"A", "M", "R"}:
            continue

        normalized = entry.path.strip()
        if entry.status.startswith("A") or entry.status.startswith("R"):
            if _violates_suffix(normalized):
                failures.append(f"Forbidden suffix detected in new file: {normalized}")

        if _violates_mock(normalized):
            failures.append(f"Mock artefact detected: {normalized}")

    if failures:
        print("Incremental integrity check failed:\n" + "\n".join(f"  - {msg}" for msg in failures), file=sys.stderr)
        return 1

    print("check_incremental: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
