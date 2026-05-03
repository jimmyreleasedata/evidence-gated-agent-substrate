#!/usr/bin/env python3
"""Build the full v2 figure/table input inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.full_evidence_common import read_csv, resolve_final_root, write_csv


OBJECTS = [
    ("F1", "Evidence-gated declared traffic coverage", "main", "C1,C2,C14"),
    ("F2", "Qwen/external model traffic cost", "main", "C3,C4,C5"),
    ("F3", "Cross-family bottleneck ownership", "main", "C3"),
    ("F4", "Model/backend compatibility matrix", "main", "C4,C12"),
    ("F5", "MiniWoB telemetry/concurrency under LLM traffic", "main", "C6"),
    ("F6", "SWE queue and failure diagnostics", "main", "C7,C8,C9"),
    ("F7", "WebArena paired stress / bounded controller pilot", "appendix", "C10,C11"),
    ("F8", "vLLM/SGLang backend contrast", "appendix", "C12"),
    ("F9", "veRL/TRL adapter demo", "appendix", "C13"),
    ("F10", "Evidence gate / traceability map", "main", "C2,C14"),
    ("T1", "Workload family / traffic source / claim scope", "main", "C1,C2"),
    ("T2", "Model/backend inventory", "main", "C4"),
    ("T3", "Claim support matrix", "main", "C14"),
    ("T4", "Accepted/rejected evidence rows", "main", "C2,C14"),
]


def build_full_figure_input_inventory(*, final_root: Path) -> list[dict[str, str]]:
    final_root = final_root.resolve()
    surface_path = final_root / "inventories" / "paper_facing_surface.csv"
    rows = read_csv(surface_path)
    source_roots = sorted({row.get("source_root", "") for row in rows if row.get("source_root", "")})
    families = {row.get("family", "") for row in rows if row.get("family", "")}
    models = {row.get("model_id", "") for row in rows if row.get("model_id", "")}
    backends = {row.get("backend_engine", "") for row in rows if row.get("backend_engine", "")}
    has_sensitive = any("REDACTED_BROWSER_STATE_LABEL" in row.get("source_file", "").lower() for row in rows)

    out: list[dict[str, str]] = []
    for figure_id, title, location, claims in OBJECTS:
        out.append(
            {
                "figure_id": figure_id,
                "title": title,
                "source_csv": "inventories/paper_facing_surface.csv",
                "source_roots": ";".join(source_roots[:50]),
                "derivation_script": "scripts/final/build_full_figure_input_inventory.py",
                "evidence_category": "paper_facing_surface",
                "row_count": str(len(rows)),
                "family_count": str(len(families)),
                "model_count": str(len(models)),
                "backend_count": str(len(backends)),
                "has_sensitive_data": str(has_sensitive).lower(),
                "gate_status": "pass" if surface_path.exists() else "missing",
                "claim_ids_supported": claims,
                "paper_location_suggestion": location,
                "appendix_or_main": location,
            }
        )
    write_csv(final_root / "inventories" / "figure_input_inventory.csv", out)
    lines = [
        "# Figure Input Inventory",
        "",
        "| ID | Title | Rows | Location | Claims |",
        "|---|---|---:|---|---|",
    ]
    for row in out:
        lines.append(
            f"| {row['figure_id']} | {row['title']} | {row['row_count']} | {row['appendix_or_main']} | {row['claim_ids_supported']} |"
        )
    (final_root / "reports" / "figure_input_inventory.md").parent.mkdir(parents=True, exist_ok=True)
    (final_root / "reports" / "figure_input_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    args = parser.parse_args()
    rows = build_full_figure_input_inventory(final_root=resolve_final_root(Path.cwd(), args.final_root))
    print(json.dumps({"figure_inputs": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
