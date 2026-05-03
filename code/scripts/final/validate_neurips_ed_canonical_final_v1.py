#!/usr/bin/env python3
"""Validate canonical final rerun artifacts and paper-facing constraints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.neurips_ed_canonical_common import (  # noqa: E402
    COMPONENT_FIELDS,
    LLM_REQUIRED_FIELDS,
    blank_or_na,
    canonical_gate_reason,
    default_root,
    is_under,
    read_csv,
    truthy,
    write_json,
)


def validate_canonical_final(root: Path) -> dict[str, object]:
    root = root.resolve()
    failures: list[dict[str, str]] = []
    surface_path = root / "global" / "paper_facing_surface.csv"
    gate_path = root / "global" / "evidence_gate_report.json"
    for required in [
        surface_path,
        gate_path,
        root / "global" / "final_claim_support_matrix.csv",
        root / "global" / "figure_input_inventory.csv",
    ]:
        if not required.exists():
            failures.append({"check": "required_file_exists", "path": str(required), "reason": "missing"})
    rows = read_csv(surface_path)
    for index, row in enumerate(rows):
        reason = canonical_gate_reason(row, root)
        if reason:
            failures.append({"check": "paper_facing_gate", "row": str(index), "reason": reason})
        source_root = str(row.get("source_root", ""))
        if source_root and not is_under(Path(source_root), root):
            failures.append({"check": "canonical_source_root", "row": str(index), "reason": "source_outside_canonical_root"})
        if str(row.get("driver_type", "")) == "llm_agent":
            missing = [field for field in LLM_REQUIRED_FIELDS if blank_or_na(row.get(field, ""))]
            if missing:
                failures.append({"check": "llm_metadata", "row": str(index), "reason": ",".join(missing)})
        if truthy(row.get("p99_increase_supported")) and blank_or_na(row.get("end_to_end_p99_ms", "")):
            failures.append({"check": "webarena_p99_claim", "row": str(index), "reason": "p99_supported_with_na_field"})
        if truthy(row.get("component_shift_supported")) and all(blank_or_na(row.get(field, "")) for field in COMPONENT_FIELDS):
            failures.append({"check": "webarena_component_claim", "row": str(index), "reason": "component_supported_without_component_fields"})
        if truthy(row.get("natural_failure_diversity_supported")):
            share = str(row.get("dominant_failure_share", "")).strip()
            if share in {"1", "1.0", "100%"}:
                failures.append({"check": "swe_failure_diversity", "row": str(index), "reason": "single_category_share_1"})

    claim_text = (root / "global" / "final_claim_support_matrix.md").read_text(encoding="utf-8", errors="replace") if (root / "global" / "final_claim_support_matrix.md").exists() else ""
    if "full rl training" in claim_text.lower() and "not full rl training" not in claim_text.lower():
        failures.append({"check": "framework_adapter_claim", "reason": "framework_adapter_described_as_full_training"})

    report = {
        "allowed": not failures,
        "root": str(root),
        "paper_facing_rows": len(rows),
        "failures": failures,
    }
    write_json(root / "global" / "validation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    args = parser.parse_args()
    report = validate_canonical_final(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
