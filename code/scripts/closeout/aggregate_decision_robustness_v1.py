#!/usr/bin/env python3
"""Aggregate decision-robustness closeout rows into the claim ladder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


CLOSEOUT_NAME = "decision_robustness_closeout_v1"
BACKENDS = {"vllm", "sglang"}
SEEDS = {0, 1}
BUDGETS = {5, 7, 9}


def _closeout_root(root: Path) -> Path:
    return root / CLOSEOUT_NAME


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 6)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _cell_metrics(admitted: pd.DataFrame, excluded: pd.DataFrame, missing: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if admitted.empty:
        return rows
    for (backend, seed, budget), group in admitted.groupby(["backend_engine", "seed", "budget"], dropna=False):
        clean = group[group["regime"].astype(str).eq("clean_baseline")]
        stress = group[group["regime"].astype(str).eq("medium_live_stressed")]
        clean_a = clean[clean["controller_id"].astype(str).eq("hook_a_only")]
        clean_b = clean[clean["controller_id"].astype(str).eq("hook_b_only")]
        stress_a = stress[stress["controller_id"].astype(str).eq("hook_a_only")]
        stress_b = stress[stress["controller_id"].astype(str).eq("hook_b_only")]
        auc_clean_a = _mean(clean_a, "reward_auc_over_wallclock")
        auc_clean_b = _mean(clean_b, "reward_auc_over_wallclock")
        auc_stress_a = _mean(stress_a, "reward_auc_over_wallclock")
        auc_stress_b = _mean(stress_b, "reward_auc_over_wallclock")
        comparable = all(value is not None for value in (auc_clean_a, auc_clean_b, auc_stress_a, auc_stress_b))
        clean_prefers = bool(comparable and auc_clean_a > auc_clean_b)
        stress_prefers = bool(comparable and auc_stress_b > auc_stress_a)
        ex_count = 0
        miss_count = 0
        if not excluded.empty:
            ex_mask = (
                excluded.get("backend_engine", pd.Series(dtype=str)).astype(str).eq(str(backend))
                & pd.to_numeric(excluded.get("seed", pd.Series(dtype=float)), errors="coerce").eq(float(seed))
                & pd.to_numeric(excluded.get("budget", pd.Series(dtype=float)), errors="coerce").eq(float(budget))
            )
            ex_count = int(ex_mask.sum())
        if not missing.empty:
            miss_mask = (
                missing.get("backend_engine", pd.Series(dtype=str)).astype(str).eq(str(backend))
                & pd.to_numeric(missing.get("seed", pd.Series(dtype=float)), errors="coerce").eq(float(seed))
                & pd.to_numeric(missing.get("budget", pd.Series(dtype=float)), errors="coerce").eq(float(budget))
            )
            miss_count = int(miss_mask.sum())
        rows.append(
            {
                "backend_engine": backend,
                "seed": int(seed),
                "budget": int(budget),
                "auc_hook_a_clean": auc_clean_a,
                "auc_hook_b_clean": auc_clean_b,
                "auc_hook_a_stress": auc_stress_a,
                "auc_hook_b_stress": auc_stress_b,
                "delta_clean": round(float(auc_clean_a - auc_clean_b), 6) if comparable else "",
                "delta_stress": round(float(auc_stress_b - auc_stress_a), 6) if comparable else "",
                "clean_prefers_hook_a": _bool_text(clean_prefers),
                "stress_prefers_hook_b": _bool_text(stress_prefers),
                "reversal_cell": _bool_text(clean_prefers and stress_prefers),
                "admitted_row_count": len(group),
                "excluded_row_count": ex_count,
                "missing_metadata_count": miss_count,
                "pass_rate_hook_a": _mean(group[group["controller_id"].astype(str).eq("hook_a_only")], "pass_rate"),
                "pass_rate_hook_b": _mean(group[group["controller_id"].astype(str).eq("hook_b_only")], "pass_rate"),
                "p99_latency_hook_a": _mean(group[group["controller_id"].astype(str).eq("hook_a_only")], "p99_latency_ms"),
                "p99_latency_hook_b": _mean(group[group["controller_id"].astype(str).eq("hook_b_only")], "p99_latency_ms"),
                "queue_wait_hook_a": _mean(group[group["controller_id"].astype(str).eq("hook_a_only")], "queue_wait_p99_ms"),
                "queue_wait_hook_b": _mean(group[group["controller_id"].astype(str).eq("hook_b_only")], "queue_wait_p99_ms"),
            }
        )
    return rows


def _summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_key: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault(row[key], []).append(row)
    for value, group in sorted(by_key.items(), key=lambda item: str(item[0])):
        comparable = [row for row in group if row.get("delta_clean") != "" and row.get("delta_stress") != ""]
        reversals = [row for row in comparable if str(row.get("reversal_cell")).lower() == "true"]
        result.append(
            {
                key: value,
                "comparable_cells": len(comparable),
                "reversal_cells": len(reversals),
                "reversal_rate": round(len(reversals) / len(comparable), 6) if comparable else 0.0,
            }
        )
    return result


def _claim_level(rows: list[dict[str, Any]], excluded: pd.DataFrame, missing: pd.DataFrame) -> tuple[str, str]:
    comparable = [row for row in rows if row.get("delta_clean") != "" and row.get("delta_stress") != ""]
    if not comparable:
        return "FAIL", "No comparable admitted cells were available."
    if len(excluded) > 0 or len(missing) > 0:
        return "FAIL", "Gate failures or missing metadata were present."
    reversals = [row for row in comparable if str(row.get("reversal_cell")).lower() == "true"]
    reversal_rate = len(reversals) / len(comparable)
    backends = {str(row["backend_engine"]) for row in comparable}
    seeds = {int(row["seed"]) for row in comparable}
    budgets = {int(row["budget"]) for row in comparable}
    backend_summary = _summary(comparable, "backend_engine")
    backend_floor_ok = all(float(row["reversal_rate"]) > 0.5 for row in backend_summary)
    if (
        len(reversals) == len(comparable)
        and backends == BACKENDS
        and seeds == SEEDS
        and budgets == BUDGETS
    ):
        return (
            "STRONG",
            "Across the tested WebArena Verified controller slice, both vLLM and SGLang show hook_a_only preferred under clean_baseline and hook_b_only preferred under medium_live_stressed for all tested seeds and budgets.",
        )
    if reversal_rate >= 0.75 and backends == BACKENDS and seeds == SEEDS and budgets == BUDGETS and backend_floor_ok:
        return (
            "DIRECTIONAL",
            "Across the tested WebArena Verified controller slice, the clean-to-stressed controller-ordering reversal is directionally stable over the tested seeds and budgets.",
        )
    if reversals:
        if len(reversals) == len(comparable):
            missing_budgets = sorted(BUDGETS - budgets)
            if missing_budgets:
                return (
                    "NARROW",
                    "All admitted comparable WebArena Verified controller cells reversed for the tested budgets, but the requested budget range is incomplete because budget(s) "
                    + ",".join(str(item) for item in missing_budgets)
                    + " were unavailable under the discovered real controller slice.",
                )
        return (
            "NARROW",
            "Some tested WebArena Verified controller cells show the clean-to-stressed ordering reversal, but the result is not stable enough for a robustness claim.",
        )
    return "FAIL", "The tested WebArena Verified controller cells do not support the reversal."


def aggregate_decision_robustness(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=False)
    closeout_root = _closeout_root(root)
    gate_dir = closeout_root / "gate"
    aggregate_dir = closeout_root / "aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    admitted = _load_csv(gate_dir / "decision_robustness_admitted_rows.csv")
    excluded = _load_csv(gate_dir / "decision_robustness_excluded_rows.csv")
    missing = _load_csv(gate_dir / "decision_robustness_missing_metadata.csv")
    rows = _cell_metrics(admitted, excluded, missing)
    cell_fields = [
        "backend_engine",
        "seed",
        "budget",
        "auc_hook_a_clean",
        "auc_hook_b_clean",
        "auc_hook_a_stress",
        "auc_hook_b_stress",
        "delta_clean",
        "delta_stress",
        "clean_prefers_hook_a",
        "stress_prefers_hook_b",
        "reversal_cell",
        "admitted_row_count",
        "excluded_row_count",
        "missing_metadata_count",
        "pass_rate_hook_a",
        "pass_rate_hook_b",
        "p99_latency_hook_a",
        "p99_latency_hook_b",
        "queue_wait_hook_a",
        "queue_wait_hook_b",
    ]
    _write_csv(aggregate_dir / "decision_robustness_cell_metrics.csv", rows, cell_fields)
    backend_summary = _summary(rows, "backend_engine")
    budget_summary = _summary(rows, "budget")
    seed_summary = _summary(rows, "seed")
    _write_csv(aggregate_dir / "decision_robustness_backend_summary.csv", backend_summary, ["backend_engine", "comparable_cells", "reversal_cells", "reversal_rate"])
    _write_csv(aggregate_dir / "decision_robustness_budget_summary.csv", budget_summary, ["budget", "comparable_cells", "reversal_cells", "reversal_rate"])
    _write_csv(aggregate_dir / "decision_robustness_seed_summary.csv", seed_summary, ["seed", "comparable_cells", "reversal_cells", "reversal_rate"])
    claim_level, sentence = _claim_level(rows, excluded, missing)
    claim_row = {
        "claim_level": claim_level,
        "recommended_claim_sentence": sentence,
        "comparable_cells": len([row for row in rows if row.get("delta_clean") != "" and row.get("delta_stress") != ""]),
        "reversal_cells": len([row for row in rows if str(row.get("reversal_cell")).lower() == "true"]),
        "admitted_rows": len(admitted),
        "excluded_rows": len(excluded),
        "missing_metadata_rows": len(missing),
    }
    _write_csv(
        aggregate_dir / "decision_robustness_claim_matrix.csv",
        [claim_row],
        ["claim_level", "recommended_claim_sentence", "comparable_cells", "reversal_cells", "admitted_rows", "excluded_rows", "missing_metadata_rows"],
    )
    (aggregate_dir / "decision_robustness_aggregate_report.json").write_text(json.dumps(claim_row, indent=2, sort_keys=True), encoding="utf-8")
    return claim_row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate decision robustness closeout rows.")
    parser.add_argument("--root", "--canonical-root", dest="root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate_decision_robustness(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
