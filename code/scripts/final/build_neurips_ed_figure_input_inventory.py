#!/usr/bin/env python3
"""Build final figure/table input inventory for the canonical rerun."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.neurips_ed_canonical_common import count_by, default_root, read_csv, write_csv, write_md  # noqa: E402


OBJECTS = [
    ("T1", "Workload family / traffic source / claim scope", "main"),
    ("T2", "Model/backend inventory", "main"),
    ("T3", "Final claim support matrix", "main"),
    ("F1", "Evidence-gated declared traffic coverage by round and family", "main"),
    ("F2", "Qwen-family LLM-driver traffic cost", "main"),
    ("F3", "Cross-family bottleneck ownership", "main"),
    ("F4", "MiniWoB telemetry/concurrency under LLM traffic", "main"),
    ("F5", "SWE queue/failure diagnostics", "main"),
    ("F6", "Backend contrast vLLM vs SGLang", "appendix"),
    ("F7", "Framework adapter demo veRL/TRL", "appendix"),
    ("F8", "WebArena paired stress / bounded controller pilot", "appendix"),
]


def build_figure_input_inventory(root: Path) -> list[dict[str, str]]:
    root = root.resolve()
    surface = read_csv(root / "global" / "paper_facing_surface.csv")
    family_count = len(count_by(surface, "family"))
    model_count = len({row.get("model_id", "") for row in surface if row.get("model_id", "")})
    backend_count = len({row.get("backend_engine", "") for row in surface if row.get("backend_engine", "")})
    rows = [
        {
            "figure_id": figure_id,
            "title": title,
            "source_csv": str(root / "global" / "paper_facing_surface.csv"),
            "derivation_script": "scripts/final/build_neurips_ed_figure_input_inventory.py",
            "row_count": str(len(surface)),
            "family_count": str(family_count),
            "model_count": str(model_count),
            "backend_count": str(backend_count),
            "evidence_gate_status": "pass" if surface else "blocked",
            "claim_ids_supported": "C8_EVIDENCE_GATE_TRACEABILITY",
            "main_or_appendix": location,
        }
        for figure_id, title, location in OBJECTS
    ]
    write_csv(root / "global" / "figure_input_inventory.csv", rows)
    write_md(
        root / "reports" / "figure_input_inventory.md",
        [
            "# Figure Input Inventory",
            "",
            *[f"- {row['figure_id']}: {row['title']} -> {row['source_csv']}" for row in rows],
        ],
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    args = parser.parse_args()
    build_figure_input_inventory(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
