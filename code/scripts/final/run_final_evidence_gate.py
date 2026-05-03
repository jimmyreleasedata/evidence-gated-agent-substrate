#!/usr/bin/env python3
"""Run the final paper-facing evidence gate over row-level evidence CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


TRUE_VALUES = {"1", "true", "yes", "y", "pass", "passed"}
CONTROL_TYPES = {"oracle_control", "negative_control"}
ROW_LEVEL_HINTS = (
    "paper_rows",
    "agent_rows",
    "backend_rows",
    "external_model_backend_rows",
    "qwen_",
    "live_rows",
    "patch_rows",
)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUE_VALUES


def _has_any(row: dict[str, str], keys: Iterable[str]) -> bool:
    return any(str(row.get(key, "")).strip() for key in keys)


def _csv_candidates(inputs_root: Path) -> list[Path]:
    candidates: list[Path] = []
    if not inputs_root.exists():
        return candidates
    for path in sorted(inputs_root.rglob("*.csv")):
        lowered = path.name.lower()
        if any(hint in lowered for hint in ROW_LEVEL_HINTS):
            candidates.append(path)
            continue
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
        except UnicodeDecodeError:
            continue
        if {"driver_id", "driver_type", "implementation_source"} <= fieldnames and (
            {"evidence_validation_pass", "trace_hash", "summary_path", "instance_id"} & fieldnames
        ):
            candidates.append(path)
    return candidates


def _failure(path: Path, row_number: int, reason: str, row: dict[str, str]) -> dict[str, object]:
    return {
        "csv_path": str(path),
        "row_number": row_number,
        "reason": reason,
        "family": row.get("family") or row.get("workload_family") or "",
        "driver_id": row.get("driver_id", ""),
        "driver_type": row.get("driver_type", ""),
        "model_id": row.get("model_id", ""),
    }


def _validate_row(path: Path, row_number: int, row: dict[str, str]) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    family = (row.get("family") or row.get("workload_family") or "").strip().lower()
    driver_type = (row.get("driver_type") or "").strip()
    implementation_source = (row.get("implementation_source") or "").strip().lower()
    evidence_category = (row.get("evidence_category") or row.get("category") or "").strip().lower()

    if implementation_source == "mock_fixture":
        failures.append(_failure(path, row_number, "mock_fixture_paper_facing", row))
    if implementation_source == "synthetic_executable":
        failures.append(_failure(path, row_number, "synthetic_executable_paper_facing", row))

    if "evidence_validation_pass" in row and not _truthy(row.get("evidence_validation_pass")):
        failures.append(_failure(path, row_number, "evidence_validation_failed", row))

    if driver_type == "llm_agent":
        required = ["driver_id", "driver_type", "model_id", "policy_version"]
        if not all(str(row.get(key, "")).strip() for key in required) or not _has_any(
            row, ["backend_engine", "backend", "model_backend"]
        ):
            failures.append(_failure(path, row_number, "missing_llm_driver_metadata", row))

    if driver_type in CONTROL_TYPES and evidence_category != "calibration_control":
        failures.append(_failure(path, row_number, "control_not_marked_calibration", row))

    if (row.get("status") or row.get("load_status") or "").strip() == "unsupported_backend_pair":
        if evidence_category not in {"model_backend_contrast", "legacy_negative_or_blocked"}:
            failures.append(_failure(path, row_number, "unsupported_backend_pair_as_success", row))

    if family in {"webarena", "webarena_verified"}:
        if "real_upstream_live" not in implementation_source and "real_upstream_replay" not in implementation_source:
            if implementation_source not in {"real_upstream", "real_live_capture"}:
                failures.append(_failure(path, row_number, "missing_webarena_real_upstream_evidence", row))
        if not _has_any(row, ["trace_hash", "artifact_hash", "eval_hash", "evaluator_output_hash"]):
            if _has_any(row, ["evaluator_version", "trace_path", "summary_path"]):
                pass
            else:
                failures.append(_failure(path, row_number, "missing_webarena_trace_or_eval_hash", row))
    elif family in {"swe", "swe_gym"}:
        required = ["instance_id", "repo", "base_commit", "harness_version", "image_or_sif_hash"]
        if not all(str(row.get(key, "")).strip() for key in required):
            failures.append(_failure(path, row_number, "missing_swe_real_metadata", row))
    elif family == "miniwob":
        if not (
            "browser" in implementation_source
            or implementation_source in {"real_upstream", "real_browser_backed"}
            or _has_any(row, ["browser_version", "browser_backend"])
        ):
            failures.append(_failure(path, row_number, "missing_miniwob_browser_backed_evidence", row))

    stress_marker = "stress" in evidence_category or "stress" in (row.get("regime") or "").lower()
    if stress_marker and not _has_any(row, ["decision_label", "decision_labels", "support_label"]):
        failures.append(_failure(path, row_number, "stress_claim_missing_decision_label", row))

    return failures


def run_final_gate(inputs_root: Path, output_path: Path | None = None) -> dict[str, object]:
    candidates = _csv_candidates(inputs_root)
    checked_rows = 0
    failures: list[dict[str, object]] = []
    for path in candidates:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                checked_rows += 1
                failures.extend(_validate_row(path, row_number, row))
    report = {
        "allowed": not failures,
        "inputs_root": str(inputs_root),
        "candidate_csvs": [str(path) for path in candidates],
        "checked_rows": checked_rows,
        "failures": failures,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", "--phase7-inputs-root", dest="inputs_root", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_final_gate(args.inputs_root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
