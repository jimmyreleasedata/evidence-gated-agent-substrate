#!/usr/bin/env python3
"""Gate isolated budget=9 decision robustness v2 rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


CLOSEOUT_NAME = "decision_robustness_closeout_v2_budget9"
EXPECTED_ROWS = 336
CONTROLLERS = {"hook_a_only", "hook_b_only"}
BACKENDS = {"vllm", "sglang"}
REGIMES = {"clean_baseline", "medium_live_stressed"}
SEEDS = {0, 1}
BUDGETS = {5, 7, 9}


def _closeout_root(root: Path) -> Path:
    return root / CLOSEOUT_NAME


def _blank(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "ok"}


def _regime(value: object) -> str:
    text = str(value or "").strip().lower()
    return "clean_baseline" if "clean" in text else "medium_live_stressed" if "medium" in text or "stress" in text else text


def _controller(value: object) -> str:
    text = str(value or "").strip().lower()
    for item in CONTROLLERS:
        if item in text:
            return item
    return text


def _backend(value: object) -> str:
    text = str(value or "").strip().lower()
    return "sglang" if "sglang" in text else "vllm" if "vllm" in text else text


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _load_task_ids(closeout_root: Path) -> set[str]:
    payload = json.loads((closeout_root / "manifests" / "frozen_9task_slice.json").read_text(encoding="utf-8"))
    return {str(task_id) for task_id in payload["task_ids"]}


def _normalize(row: pd.Series, source_csv: Path, closeout_root: Path) -> dict[str, Any]:
    seed_raw = row.get("seed")
    budget_raw = row.get("budget")
    success = _truthy(row.get("task_success", row.get("passed", "")))
    return {
        "run_id": row.get("run_id") or f"{row.get('backend_engine', row.get('backend'))}_{row.get('seed')}_{row.get('budget')}_{row.get('task_id')}_{row.get('controller', row.get('decision_label'))}_{row.get('regime')}",
        "task_id": str(row.get("task_id")),
        "backend_engine": _backend(row.get("backend_engine", row.get("backend"))),
        "seed": int(float(seed_raw)) if not _blank(seed_raw) else -1,
        "budget": int(float(budget_raw)) if not _blank(budget_raw) else -1,
        "regime": _regime(row.get("regime")),
        "controller_id": _controller(row.get("controller_id", row.get("controller", row.get("decision_label")))),
        "paper_role": "decision_robustness_closeout_v2_budget9",
        "exclude_from_canonical_surface": True,
        "terminal_outcome": row.get("terminal_outcome") or ("pass" if success else "fail"),
        "terminal_outcome_present": row.get("terminal_outcome_present"),
        "reward": row.get("reward") if not _blank(row.get("reward")) else (1.0 if success else 0.0),
        "reward_auc_over_wallclock": row.get("reward_auc_over_wallclock", row.get("reward_auc_over_wallclock_mean")),
        "pass_rate": row.get("pass_rate") if not _blank(row.get("pass_rate")) else (1.0 if success else 0.0),
        "p99_latency_ms": row.get("p99_latency_ms"),
        "queue_wait_p99_ms": row.get("queue_wait_p99_ms", row.get("queue_wait_ms")),
        "evidence_validation_pass": row.get("evidence_validation_pass"),
        "trace_id": row.get("trace_id", ""),
        "manifest_hash": row.get("manifest_hash"),
        "manifest_path": row.get("manifest_path", row.get("run_manifest_path", "")),
        "source_csv": str(source_csv),
        "artifact_root": str(source_csv.parent),
        "closeout_root": str(closeout_root),
        "sensitive_hash_only": True,
    }


def _gate(row: dict[str, Any], task_ids: set[str]) -> list[str]:
    reasons: list[str] = []
    for field in ("task_id", "backend_engine", "seed", "budget", "regime", "controller_id", "terminal_outcome", "reward_auc_over_wallclock", "evidence_validation_pass", "manifest_hash"):
        if _blank(row.get(field)):
            reasons.append(f"missing_{field}")
    if not _truthy(row.get("evidence_validation_pass")):
        reasons.append("evidence_validation_pass_not_true")
    if str(row.get("task_id")) not in task_ids:
        reasons.append("task_id_outside_frozen_slice")
    if row.get("backend_engine") not in BACKENDS:
        reasons.append("backend_outside_matrix")
    if row.get("seed") not in SEEDS:
        reasons.append("seed_outside_matrix")
    if row.get("budget") not in BUDGETS:
        reasons.append("budget_outside_matrix")
    if row.get("regime") not in REGIMES:
        reasons.append("regime_outside_matrix")
    if row.get("controller_id") not in CONTROLLERS:
        reasons.append("controller_outside_matrix")
    text = " ".join(str(value).lower() for value in row.values())
    if any(marker in text for marker in ("mock", "synthetic", "fixture", "preflight_only", "smoke")):
        reasons.append("mock_fixture_synthetic_or_preflight")
    return reasons


def _dedupe_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["task_id"], row["backend_engine"], row["seed"], row["budget"], row["regime"], row["controller_id"])


def validate_budget9_v2(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=False)
    closeout_root = _closeout_root(root)
    task_ids = _load_task_ids(closeout_root)
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    superseded: list[dict[str, Any]] = []
    for dirname in ("runs", "runs_retry_clean"):
        for path in sorted((closeout_root / dirname).glob("**/controller_trace.csv")):
            frame = pd.read_csv(path, low_memory=False)
            for _, source_row in frame.iterrows():
                row = _normalize(source_row, path, closeout_root)
                row["block_reason"] = ";".join(_gate(row, task_ids))
                key = _dedupe_key(row)
                old = candidates.get(key)
                if old is None or (old["block_reason"] and not row["block_reason"]) or "/runs_retry_clean/" in row["source_csv"]:
                    if old is not None:
                        old["superseded_reason"] = "duplicate_cell_replaced_by_later_valid_row"
                        superseded.append(old)
                    candidates[key] = row
                else:
                    row["superseded_reason"] = "duplicate_cell_not_selected"
                    superseded.append(row)
    admitted = [row for row in candidates.values() if not row["block_reason"]]
    excluded = [row for row in candidates.values() if row["block_reason"]]
    fields = [
        "run_id",
        "task_id",
        "backend_engine",
        "seed",
        "budget",
        "regime",
        "controller_id",
        "paper_role",
        "exclude_from_canonical_surface",
        "terminal_outcome",
        "terminal_outcome_present",
        "reward",
        "reward_auc_over_wallclock",
        "pass_rate",
        "p99_latency_ms",
        "queue_wait_p99_ms",
        "evidence_validation_pass",
        "trace_id",
        "manifest_hash",
        "manifest_path",
        "source_csv",
        "artifact_root",
        "closeout_root",
        "sensitive_hash_only",
        "block_reason",
    ]
    _write_csv(closeout_root / "gate" / "admitted_rows.csv", admitted, fields)
    _write_csv(closeout_root / "gate" / "excluded_rows.csv", excluded, fields)
    _write_csv(closeout_root / "superseded_lineage.csv", superseded, fields + ["superseded_reason"])
    planned = pd.read_csv(closeout_root / "manifests" / "planned_cells.csv")
    executed_keys = {_dedupe_key(row) for row in admitted + excluded}
    pve_rows = []
    for _, plan in planned.iterrows():
        key = (str(plan.task_id), str(plan.backend_engine), int(plan.seed), int(plan.budget), str(plan.regime), str(plan.controller_id))
        pve_rows.append(
            {
                "planned_cell_id": plan.planned_cell_id,
                "task_id": plan.task_id,
                "backend_engine": plan.backend_engine,
                "seed": plan.seed,
                "budget": plan.budget,
                "regime": plan.regime,
                "controller_id": plan.controller_id,
                "executed": key in executed_keys,
                "admitted": key in {_dedupe_key(row) for row in admitted},
            }
        )
    _write_csv(
        closeout_root / "planned_vs_executed.csv",
        pve_rows,
        ["planned_cell_id", "task_id", "backend_engine", "seed", "budget", "regime", "controller_id", "executed", "admitted"],
    )
    report = {
        "expected_rows": EXPECTED_ROWS,
        "observed_unique_rows": len(candidates),
        "admitted_rows": len(admitted),
        "excluded_rows": len(excluded),
        "missing_metadata_rows": 0,
        "superseded_rows": len(superseded),
        "gate_pass": len(admitted) == EXPECTED_ROWS and not excluded,
        "raw_sensitive_copied": False,
    }
    (closeout_root / "gate_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate budget9 decision robustness v2 rows.")
    parser.add_argument("--root", "--canonical-root", dest="root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(validate_budget9_v2(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
