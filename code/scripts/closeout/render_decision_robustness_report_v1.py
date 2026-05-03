#!/usr/bin/env python3
"""Render decision-robustness closeout reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


CLOSEOUT_NAME = "decision_robustness_closeout_v1"


def _closeout_root(root: Path) -> Path:
    return root / CLOSEOUT_NAME


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _first(df: pd.DataFrame, column: str, default: Any = "") -> Any:
    if df.empty or column not in df.columns:
        return default
    return df[column].iloc[0]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in df.columns) + " |")
    return "\n".join(lines)


def render_decision_robustness_report(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=False)
    closeout_root = _closeout_root(root)
    reports = closeout_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    planned = _load_csv(closeout_root / "manifests" / "decision_robustness_planned_cells.csv")
    reuse = _load_csv(closeout_root / "manifests" / "reuse_decisions.csv")
    jobs = _load_csv(closeout_root / "logs" / "planned_execution_jobs.csv")
    trace_groups = list((closeout_root / "runs").glob("**/controller_trace.csv")) + list(
        (closeout_root / "runs_retry_clean").glob("**/controller_trace.csv")
    )
    admitted = _load_csv(closeout_root / "gate" / "decision_robustness_admitted_rows.csv")
    excluded = _load_csv(closeout_root / "gate" / "decision_robustness_excluded_rows.csv")
    missing = _load_csv(closeout_root / "gate" / "decision_robustness_missing_metadata.csv")
    claim = _load_csv(closeout_root / "aggregates" / "decision_robustness_claim_matrix.csv")
    backend_summary = _load_csv(closeout_root / "aggregates" / "decision_robustness_backend_summary.csv")
    seed_summary = _load_csv(closeout_root / "aggregates" / "decision_robustness_seed_summary.csv")
    budget_summary = _load_csv(closeout_root / "aggregates" / "decision_robustness_budget_summary.csv")
    claim_level = str(_first(claim, "claim_level", "FAIL"))
    sentence = str(_first(claim, "recommended_claim_sentence", "No unconditional robustness claim is recommended."))
    reused = int(reuse.get("reuse_decision", pd.Series(dtype=str)).eq("reuse").sum()) if not reuse.empty else 0
    executed_jobs = len(trace_groups) if trace_groups else (len(jobs) if not jobs.empty else 0)

    summary_lines = [
        "# Decision Robustness Closeout Summary",
        "",
        f"- closeout_root: `{closeout_root}`",
        f"- planned cells: {len(planned)}",
        f"- executed trace groups recorded: {executed_jobs}",
        f"- reused canonical cells: {reused}",
        f"- admitted rows: {len(admitted)}",
        f"- excluded rows: {len(excluded)}",
        f"- missing metadata rows: {len(missing)}",
        f"- recommended claim level: {claim_level}",
        "",
        "## Per-Backend Reversal Summary",
        _markdown_table(backend_summary) if not backend_summary.empty else "No backend summary available.",
        "",
        "## Per-Seed Reversal Summary",
        _markdown_table(seed_summary) if not seed_summary.empty else "No seed summary available.",
        "",
        "## Per-Budget Reversal Summary",
        _markdown_table(budget_summary) if not budget_summary.empty else "No budget summary available.",
        "",
        "## Recommended Claim Sentence",
        sentence,
        "",
        "This is a separate closeout package and is not part of the canonical 930-row paper-facing surface unless explicitly promoted later.",
    ]
    (reports / "decision_robustness_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    reproduce_lines = [
        "# Decision Robustness Reproduce Commands",
        "",
        "```bash",
        "export NIPS_CANONICAL_FINAL_ROOT=/path/to/canonical/root",
        "python scripts/closeout/build_decision_robustness_manifest_v1.py --root \"$NIPS_CANONICAL_FINAL_ROOT\"",
        "bash scripts/closeout/run_decision_robustness_webarena_v1.sh --root \"$NIPS_CANONICAL_FINAL_ROOT\" --planned-cells \"$NIPS_CANONICAL_FINAL_ROOT/decision_robustness_closeout_v1/manifests/decision_robustness_smoke_cells.csv\" --reuse-canonical true --dry-run false",
        "bash scripts/closeout/run_decision_robustness_webarena_v1.sh --root \"$NIPS_CANONICAL_FINAL_ROOT\" --planned-cells \"$NIPS_CANONICAL_FINAL_ROOT/decision_robustness_closeout_v1/manifests/decision_robustness_planned_cells.csv\" --reuse-canonical true --dry-run false",
        "python scripts/closeout/validate_decision_robustness_gate_v1.py --root \"$NIPS_CANONICAL_FINAL_ROOT\"",
        "python scripts/closeout/aggregate_decision_robustness_v1.py --root \"$NIPS_CANONICAL_FINAL_ROOT\"",
        "python scripts/closeout/render_decision_robustness_report_v1.py --root \"$NIPS_CANONICAL_FINAL_ROOT\"",
        "```",
        "",
        "Smoke outputs must remain under `preflight_only/`; real closeout rows must remain under `runs/`.",
    ]
    (reports / "decision_robustness_reproduce.md").write_text("\n".join(reproduce_lines) + "\n", encoding="utf-8")

    recommendation = [
        "# Decision Robustness Claim Recommendation",
        "",
        f"recommended_claim_level: {claim_level}",
        "",
        sentence,
        "",
        "Do not emit an unconditional robustness claim beyond this ladder result.",
    ]
    (reports / "decision_robustness_claim_recommendation.md").write_text("\n".join(recommendation) + "\n", encoding="utf-8")

    mutation = [
        "canonical paper-facing surface changed: no",
        "canonical figure inputs changed: no",
        "reviewer bundle changed: no",
        "paper text changed: no",
        "basis: this renderer writes only under decision_robustness_closeout_v1/reports and does not edit canonical global CSVs, figure inputs, reviewer bundle, or manuscript files.",
    ]
    (reports / "no_canonical_surface_mutation_report.txt").write_text("\n".join(mutation) + "\n", encoding="utf-8")
    result = {
        "closeout_root": str(closeout_root),
        "claim_level": claim_level,
        "summary_path": str(reports / "decision_robustness_summary.md"),
        "reproduce_path": str(reports / "decision_robustness_reproduce.md"),
        "no_mutation_path": str(reports / "no_canonical_surface_mutation_report.txt"),
    }
    (reports / "decision_robustness_render_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render decision robustness closeout reports.")
    parser.add_argument("--root", "--canonical-root", dest="root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = render_decision_robustness_report(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
