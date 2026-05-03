#!/usr/bin/env python3
"""Validate the full NeurIPS evidence v2 package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.full_evidence_common import PACKAGE_NAME, read_csv, resolve_final_root, truthy


REQUIRED_FILES = [
    "README.md",
    "collection_manifest.yaml",
    "collection_manifest.json",
    "artifact_hashes.sha256",
    "sensitive_artifact_redaction_manifest.csv",
    "inventories/expected_roots_manifest.yaml",
    "inventories/discovered_roots.csv",
    "inventories/missing_expected_roots.csv",
    "inventories/all_artifacts_inventory.csv",
    "inventories/all_run_roots_inventory.csv",
    "inventories/all_runs_long.csv",
    "inventories/all_metrics_long.csv",
    "inventories/all_evidence_rows.csv",
    "inventories/paper_facing_surface.csv",
    "inventories/excluded_rows.csv",
    "inventories/conflict_report.csv",
    "inventories/duplicate_report.csv",
    "inventories/figure_input_inventory.csv",
    "inventories/claim_support_matrix.csv",
    "reports/full_evidence_summary.md",
    "reports/collection_coverage_report.md",
    "reports/missing_data_report.md",
    "reports/conflict_resolution_report.md",
    "reports/paper_facing_surface_report.md",
    "reports/claim_support_matrix.md",
    "reports/figure_input_inventory.md",
    "reports/final_numbers_comparison.md",
    "reports/final_limitations_and_blockers.md",
]


def validate_full_evidence_collection(*, final_root: Path) -> dict[str, object]:
    final_root = final_root.resolve()
    failures: list[str] = []
    for relative in REQUIRED_FILES:
        if not (final_root / relative).exists():
            failures.append(f"missing_required_file:{relative}")

    manifest_path = final_root / "collection_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("package_name") != PACKAGE_NAME:
            failures.append("wrong_package_name")

    paper_rows = read_csv(final_root / "inventories" / "paper_facing_surface.csv")
    for index, row in enumerate(paper_rows, start=2):
        impl = row.get("implementation_source", "")
        if impl == "mock_fixture":
            failures.append(f"paper_surface_mock_fixture_row:{index}")
        if impl == "synthetic_executable":
            failures.append(f"paper_surface_synthetic_row:{index}")
        if row.get("driver_type") == "llm_agent":
            for key in ("driver_id", "model_id", "backend_engine", "policy_version"):
                if not row.get(key, ""):
                    failures.append(f"paper_surface_missing_llm_{key}:row{index}")
        if row.get("family") == "webarena_verified":
            if truthy(row.get("p99_increase_supported")) and not row.get("end_to_end_p99_ms", ""):
                failures.append(f"webarena_p99_claim_without_p99:row{index}")
            if truthy(row.get("component_shift_supported")) and not any(row.get(key, "") for key in ("browser_env_latency_ms", "evaluator_latency_ms")):
                failures.append(f"webarena_component_claim_without_component:row{index}")
        if row.get("family") == "swe_gym":
            if row.get("dominant_failure_category") and row.get("dominant_failure_category") in {"patch_apply_failure", "controlled_timeout"}:
                share = row.get("natural_failure_share") or row.get("controlled_failure_share")
                if share in {"1", "1.0"} and truthy(row.get("natural_failure_diversity_supported")):
                    failures.append(f"swe_failure_diversity_overclaimed:row{index}")

    claim_rows = read_csv(final_root / "inventories" / "claim_support_matrix.csv")
    by_claim = {row.get("claim_id"): row for row in claim_rows}
    c9 = by_claim.get("C9")
    if c9 and c9.get("support_status") == "supported":
        for row in paper_rows:
            if row.get("dominant_failure_category") in {"patch_apply_failure", "controlled_timeout"}:
                share = row.get("natural_failure_share") or row.get("controlled_failure_share")
                if share in {"1", "1.0"}:
                    failures.append("claim_c9_supported_despite_single_failure_category")
                    break
    c10 = by_claim.get("C10")
    if c10 and c10.get("support_status") == "supported":
        if not any(truthy(row.get("p99_increase_supported")) or truthy(row.get("component_shift_supported")) for row in paper_rows):
            failures.append("claim_c10_supported_without_webarena_decision_labels")

    figure_rows = read_csv(final_root / "inventories" / "figure_input_inventory.csv")
    if len(figure_rows) < 14:
        failures.append("figure_input_inventory_incomplete")

    report = {"allowed": not failures, "final_root": str(final_root), "failures": failures}
    (final_root / "logs").mkdir(parents=True, exist_ok=True)
    (final_root / "logs" / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    args = parser.parse_args()
    report = validate_full_evidence_collection(final_root=resolve_final_root(Path.cwd(), args.final_root))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
