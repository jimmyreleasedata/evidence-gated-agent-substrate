#!/usr/bin/env python3
"""Validate the final NeurIPS evidence package surface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_FILES = [
    "collection_manifest.json",
    "evidence_gate_report.json",
    "final_acceptance_status.json",
    "inventories/all_runs_inventory.csv",
    "inventories/all_artifacts_inventory.csv",
    "inventories/claim_support_matrix.csv",
    "inventories/figure_input_inventory.csv",
    "release_metadata/sensitive_artifact_redaction_manifest.csv",
    "reports/final_evidence_summary.md",
]


def _csv_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) > 1


def validate_final_evidence(final_root: Path) -> dict[str, object]:
    missing = [relative for relative in REQUIRED_FILES if not (final_root / relative).exists()]
    empty_required = [
        relative
        for relative in [
            "inventories/all_runs_inventory.csv",
            "inventories/all_artifacts_inventory.csv",
            "inventories/claim_support_matrix.csv",
            "inventories/figure_input_inventory.csv",
        ]
        if (final_root / relative).exists() and not _csv_has_rows(final_root / relative)
    ]
    gate_path = final_root / "evidence_gate_report.json"
    gate_allowed = False
    if gate_path.exists():
        try:
            gate_allowed = bool(json.loads(gate_path.read_text(encoding="utf-8")).get("allowed"))
        except json.JSONDecodeError:
            gate_allowed = False
    result = {
        "allowed": not missing and not empty_required and gate_allowed,
        "final_root": str(final_root),
        "missing": missing,
        "empty_required": empty_required,
        "evidence_gate_allowed": gate_allowed,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    args = parser.parse_args()
    result = validate_final_evidence(args.final_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
