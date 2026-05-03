#!/usr/bin/env python3
"""Run the full evidence v2 gate over all normalized evidence rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.full_evidence_common import read_csv, resolve_final_root, truthy, write_csv, write_json, write_parquet_if_possible


CONTROL_TYPES = {"oracle_control", "negative_control"}


def _has_any(row: dict[str, str], keys: list[str]) -> bool:
    return any(str(row.get(key, "")).strip() for key in keys)


def _reason(row: dict[str, str]) -> str:
    implementation = str(row.get("implementation_source", "")).strip().lower()
    evidence_class = str(row.get("evidence_class", "")).strip().lower()
    source_category = str(row.get("source_category", "")).strip().lower()
    driver_type = str(row.get("driver_type", "")).strip()
    family = str(row.get("family", "")).strip().lower()

    if implementation == "mock_fixture" or truthy(row.get("fixture_only")):
        return "mock_fixture_paper_facing"
    if implementation == "synthetic_executable":
        return "synthetic_executable_paper_facing"
    if source_category == "11_negative_blocker_evidence" or evidence_class == "negative_blocker_evidence":
        return "negative_blocker_not_paper_facing"
    if truthy(row.get("unsupported_backend_pair")):
        return "unsupported_backend_pair_not_success"
    if driver_type == "llm_agent":
        if not all(str(row.get(key, "")).strip() for key in ["driver_id", "driver_type", "model_id", "policy_version"]):
            return "missing_llm_driver_metadata"
        if not _has_any(row, ["backend_engine", "backend", "model_backend"]):
            return "missing_llm_driver_metadata"
    if driver_type in CONTROL_TYPES and evidence_class != "calibration_control":
        return "control_not_marked_calibration"
    if row.get("evidence_validation_pass") and not truthy(row.get("evidence_validation_pass")):
        return "evidence_validation_failed"
    if family == "webarena_verified":
        if "real_upstream_live" not in implementation and "real_upstream_replay" not in implementation and implementation not in {"real_upstream", "real_live_capture"}:
            return "missing_webarena_real_upstream_evidence"
        if not _has_any(row, ["trace_hash", "artifact_hash", "eval_hash", "evaluator_output_hash"]) and not _has_any(
            row, ["evaluator_version", "summary_path", "trace_path"]
        ):
            return "missing_webarena_trace_or_eval_hash"
    if family == "swe_gym" and evidence_class in {"llm_agent_core", "model_backend_contrast", "real_task_cutover"}:
        required = ["instance_id", "repo", "base_commit", "harness_version", "image_or_sif_hash"]
        if any(not str(row.get(key, "")).strip() for key in required):
            # Historical summaries sometimes only carry aggregate SWE rows. Keep
            # them appendix-only instead of silently passing them as paper data.
            return "missing_swe_real_metadata"
    if family == "miniwob":
        if "browser" not in implementation and not _has_any(row, ["browser_version", "browser_backend"]):
            return "missing_miniwob_browser_backed_evidence"
    stress_marker = evidence_class == "reviewer_risk_fix" or "stress" in str(row.get("regime", "")).lower()
    if stress_marker and not _has_any(
        row,
        [
            "decision_label",
            "decision_labels",
            "support_label",
            "measurable_queueing",
            "significant_queueing",
            "dominant_queueing",
            "natural_failure_diversity_supported",
            "p99_increase_supported",
            "component_shift_supported",
            "behavior_shift_supported",
            "controller_reversal_supported",
            "operating_point_shift_supported",
        ],
    ):
        return "stress_claim_missing_decision_label"
    return ""


def run_full_evidence_gate(*, final_root: Path) -> dict[str, object]:
    final_root = final_root.resolve()
    rows = read_csv(final_root / "inventories" / "all_evidence_rows.csv")
    accepted: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("paper_facing_allowed", "")).lower() not in {"true", "1", "yes"}:
            row = dict(row)
            row["exclusion_reason"] = row.get("exclusion_reason") or "not_marked_paper_facing"
            excluded.append(row)
            continue
        reason = _reason(row)
        if reason:
            row = dict(row)
            row["exclusion_reason"] = reason
            excluded.append(row)
        else:
            accepted.append(row)

    write_csv(final_root / "inventories" / "paper_facing_surface.csv", accepted)
    write_parquet_if_possible(final_root / "inventories" / "paper_facing_surface.csv", accepted)
    write_csv(final_root / "inventories" / "excluded_rows.csv", excluded)
    write_parquet_if_possible(final_root / "inventories" / "excluded_rows.csv", excluded)
    write_csv(final_root / "inventories" / "rejected_rows_inventory.csv", excluded)
    report = {
        "allowed": True,
        "input_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(excluded),
        "paper_facing_surface": str(final_root / "inventories" / "paper_facing_surface.csv"),
        "excluded_rows": str(final_root / "inventories" / "excluded_rows.csv"),
        "rejection_reasons": sorted({row.get("exclusion_reason", "") for row in excluded if row.get("exclusion_reason", "")}),
    }
    write_json(final_root / "evidence_gate_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_full_evidence_gate(final_root=resolve_final_root(Path.cwd(), args.final_root)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
