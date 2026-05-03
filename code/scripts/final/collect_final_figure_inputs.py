#!/usr/bin/env python3
"""Collect final figure/table input inventory rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIGURE_INPUTS: list[tuple[str, list[str]]] = [
    ("Workload family / traffic source / claim scope", ["inventories/all_runs_inventory.csv", "inputs/driver_inventory.csv"]),
    ("Model/backend inventory", ["inputs/model_backend_inventory.csv", "inputs/model_backend_matrix/model_backend_inventory.csv"]),
    ("Qwen + Llama + optional MiniMax model-backend matrix", ["inputs/model_traffic_summary.csv", "inputs/model_backend_matrix"]),
    ("vLLM vs SGLang backend contrast", ["inputs/backend_engine_contrast.csv", "inputs/phase4/backend_engine_contrast.csv"]),
    ("Cross-family latency under LLM-agent traffic", ["inputs/model_traffic_summary.csv", "inputs/bottleneck_component_summary.csv"]),
    ("Model-size/system-cost curve", ["inputs/model_size_curve.csv", "plot_inputs/model_size_curve_plot.csv"]),
    ("MiniWoB telemetry/concurrency under LLM traffic", ["inputs/miniwob_model_backend_telemetry.csv", "inputs/telemetry_overhead_summary.csv"]),
    ("SWE patch-provider + queue saturation + failure diversity", ["inputs/swe_model_backend_patch_summary.csv", "inputs/swe_queue_regime_summary.csv", "inputs/swe_failure_regime_summary.csv"]),
    ("WebArena paired stress / controller sensitivity", ["inputs/webarena_paired_stress_summary.csv", "inputs/webarena_paired_deltas.csv"]),
    ("veRL + TRL framework adapter table", ["inputs/framework_adapters/framework_adapter_summary.csv", "inputs/framework_adapters/verl_adapter_summary.csv", "inputs/framework_adapters/trl_adapter_summary.csv"]),
    ("Evidence gate and traceability coverage", ["evidence_gate_report.json", "inventories/all_artifacts_inventory.csv", "inventories/claim_support_matrix.csv"]),
]


def collect_figure_inputs(final_root: Path, output_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, candidates in FIGURE_INPUTS:
        present = [candidate for candidate in candidates if (final_root / candidate).exists()]
        rows.append(
            {
                "figure_input": label,
                "candidate_paths": ";".join(candidates),
                "present_paths": ";".join(present),
                "status": "present" if present else "missing",
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    output_csv = args.output_csv or args.final_root / "inventories" / "figure_input_inventory.csv"
    collect_figure_inputs(args.final_root, output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
