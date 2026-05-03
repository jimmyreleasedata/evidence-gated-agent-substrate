#!/usr/bin/env python3
from pathlib import Path

required = [
    "README_FIRST.md",
    "metadata/croissant.json",
    "checksums/SHA256SUMS",
    "validation/final_submission_readiness_report.md",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))
print("submission package smoke validation passed")
