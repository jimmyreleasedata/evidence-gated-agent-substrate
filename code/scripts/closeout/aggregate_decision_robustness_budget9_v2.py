#!/usr/bin/env python3
"""Aggregate isolated budget=9 decision robustness v2 rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


CLOSEOUT_NAME = "decision_robustness_closeout_v2_budget9"
EXPECTED_ROWS = 336
BACKENDS = {"vllm", "sglang"}
SEEDS = {0, 1}
BUDGETS = {5, 7, 9}


def _closeout_root(root: Path) -> Path:
    return root / CLOSEOUT_NAME


def _mean(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return round(float(values.mean()), 6) if not values.empty else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def aggregate_budget9_v2(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=False)
    closeout_root = _closeout_root(root)
    admitted_path = closeout_root / "gate" / "admitted_rows.csv"
    excluded_path = closeout_root / "gate" / "excluded_rows.csv"
    admitted = pd.read_csv(admitted_path, low_memory=False) if admitted_path.exists() else pd.DataFrame()
    excluded = pd.read_csv(excluded_path, low_memory=False) if excluded_path.exists() else pd.DataFrame()
    cells: list[dict[str, Any]] = []
    if not admitted.empty:
        for (backend, seed, budget), group in admitted.groupby(["backend_engine", "seed", "budget"], dropna=False):
            clean = group[group["regime"].astype(str).eq("clean_baseline")]
            stress = group[group["regime"].astype(str).eq("medium_live_stressed")]
            ca = _mean(clean[clean["controller_id"].astype(str).eq("hook_a_only")], "reward_auc_over_wallclock")
            cb = _mean(clean[clean["controller_id"].astype(str).eq("hook_b_only")], "reward_auc_over_wallclock")
            sa = _mean(stress[stress["controller_id"].astype(str).eq("hook_a_only")], "reward_auc_over_wallclock")
            sb = _mean(stress[stress["controller_id"].astype(str).eq("hook_b_only")], "reward_auc_over_wallclock")
            comparable = all(value is not None for value in (ca, cb, sa, sb))
            reversal = bool(comparable and ca > cb and sb > sa)
            cells.append(
                {
                    "backend_engine": backend,
                    "seed": int(seed),
                    "budget": int(budget),
                    "auc_hook_a_clean": ca,
                    "auc_hook_b_clean": cb,
                    "auc_hook_a_stress": sa,
                    "auc_hook_b_stress": sb,
                    "delta_clean": round(ca - cb, 6) if comparable else "",
                    "delta_stress": round(sb - sa, 6) if comparable else "",
                    "reversal_cell": reversal,
                    "admitted_row_count": len(group),
                }
            )
    _write_csv(
        closeout_root / "aggregates" / "cell_metrics.csv",
        cells,
        [
            "backend_engine",
            "seed",
            "budget",
            "auc_hook_a_clean",
            "auc_hook_b_clean",
            "auc_hook_a_stress",
            "auc_hook_b_stress",
            "delta_clean",
            "delta_stress",
            "reversal_cell",
            "admitted_row_count",
        ],
    )
    comparable = [row for row in cells if row["delta_clean"] != "" and row["delta_stress"] != ""]
    reversal_count = sum(1 for row in comparable if bool(row["reversal_cell"]))
    backends = {str(row["backend_engine"]) for row in comparable}
    seeds = {int(row["seed"]) for row in comparable}
    budgets = {int(row["budget"]) for row in comparable}
    if len(admitted) == EXPECTED_ROWS and excluded.empty and reversal_count == len(comparable) and backends == BACKENDS and seeds == SEEDS and budgets == BUDGETS:
        level = "STRONG"
        sentence = "Across the isolated admitted WebArena Verified budget9 controller closeout, all tested backend, seed, and budget cells reverse ordering between clean and medium-live-stressed regimes."
    elif comparable and reversal_count / len(comparable) >= 0.75:
        level = "DIRECTIONAL"
        sentence = "The isolated WebArena Verified budget9 controller closeout directionally supports the clean-to-stressed reversal, but not all STRONG conditions hold."
    elif reversal_count:
        level = "NARROW"
        sentence = "Some isolated WebArena Verified budget9 controller cells reverse, but the result is not stable enough for a robustness claim."
    else:
        level = "FAIL"
        sentence = "The isolated WebArena Verified budget9 controller closeout does not support the reversal."
    claim = {
        "claim_level": level,
        "recommended_claim_sentence": sentence,
        "expected_rows": EXPECTED_ROWS,
        "admitted_rows": len(admitted),
        "excluded_rows": len(excluded),
        "comparable_cells": len(comparable),
        "reversal_cells": reversal_count,
    }
    _write_csv(
        closeout_root / "claim_matrix.csv",
        [claim],
        ["claim_level", "recommended_claim_sentence", "expected_rows", "admitted_rows", "excluded_rows", "comparable_cells", "reversal_cells"],
    )
    summary = "\n".join(
        [
            "# Decision Robustness Budget9 V2 Summary",
            "",
            f"- closeout_root: `{closeout_root}`",
            f"- expected_rows: {EXPECTED_ROWS}",
            f"- admitted_rows: {len(admitted)}",
            f"- excluded_rows: {len(excluded)}",
            f"- comparable_cells: {len(comparable)}",
            f"- reversal_cells: {reversal_count}",
            f"- claim_level: {level}",
            f"- recommended_claim_sentence: {sentence}",
            "- canonical_surface_mutated: no",
            "- paper_text_mutated: no",
            "- figures_regenerated: no",
        ]
    )
    (closeout_root / "summary.md").write_text(summary + "\n", encoding="utf-8")
    (closeout_root / "aggregates" / "aggregate_report.json").write_text(json.dumps(claim, indent=2, sort_keys=True), encoding="utf-8")
    return claim


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate budget9 decision robustness v2 rows.")
    parser.add_argument("--root", "--canonical-root", dest="root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(aggregate_budget9_v2(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
