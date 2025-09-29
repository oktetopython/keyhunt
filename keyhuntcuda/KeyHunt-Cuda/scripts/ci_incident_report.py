# SPDX-License-Identifier: GPL-3.0-or-later
# Source-Fusion Provenance: CI incident logging helper created for KeyHunt-CUDA (2024-09-29).

"""Emit a lightweight incident report after failed CI runs."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_PATH = Path(os.getenv("CI_INCIDENT_PATH", "artifacts/ci_incident.json"))


def gather_context(reason: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "generated_at": now,
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "run_number": os.getenv("GITHUB_RUN_NUMBER"),
        "workflow": os.getenv("GITHUB_WORKFLOW"),
        "job": os.getenv("GITHUB_JOB"),
        "repository": os.getenv("GITHUB_REPOSITORY"),
        "ref": os.getenv("GITHUB_REF"),
        "sha": os.getenv("GITHUB_SHA"),
        "actor": os.getenv("GITHUB_ACTOR"),
        "reason": reason,
    }


def main() -> int:
    reason = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "CI job failed"
    payload = gather_context(reason)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ci_incident_report: wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
