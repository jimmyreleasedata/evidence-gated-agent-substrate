#!/usr/bin/env python3
"""Build the evidence-stratified claim support matrix for the canonical rerun."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.neurips_ed_canonical_common import count_by, default_root, read_csv, write_csv, write_md  # noqa: E402


CLAIMS = [
    ("C1_REAL_TASK_SUBSTRATE", "ROUND1_REAL_TASK_CUTOVER", "real WebArena / SWE / MiniWoB anchors"),
    ("C2_QWEN_FAMILY_DRIVER_TRAFFIC", "ROUND2_QWEN_FAMILY_DRIVER", "local Qwen-family LLM driver traffic"),
    ("C3_MODEL_BACKEND_EXTENSION", "closeout_EXTENSION_RISKFIX", "external model / backend compatibility"),
    ("C4_FRAMEWORK_ADAPTER_COMPATIBILITY", "closeout_EXTENSION_RISKFIX", "veRL / TRL adapter demos only"),
    ("C5_SWE_QUEUE_DIAGNOSTIC", "closeout_EXTENSION_RISKFIX", "SWE queue diagnostic"),
    ("C6_SWE_FAILURE_DIAGNOSTIC", "closeout_EXTENSION_RISKFIX", "SWE failure diagnostic"),
    ("C7_WEBARENA_STRESS", "closeout_EXTENSION_RISKFIX", "WebArena paired stress / bounded pilot"),
    ("C8_EVIDENCE_GATE_TRACEABILITY", "ALL", "evidence-stratified evidence gate and traceability"),
]


def build_claim_support_matrix(root: Path) -> list[dict[str, str]]:
    root = root.resolve()
    rows = read_csv(root / "global" / "paper_facing_surface.csv")
    by_round = count_by(rows, "round_id")
    output: list[dict[str, str]] = []
    for claim_id, source_round, scope in CLAIMS:
        if source_round == "ALL":
            row_count = len(rows)
            status = "supported" if row_count else "blocked"
        else:
            row_count = by_round.get(source_round, 0)
            status = "supported" if row_count else "blocked"
        writing_instruction = scope
        if claim_id == "C4_FRAMEWORK_ADAPTER_COMPATIBILITY":
            writing_instruction = "Adapter compatibility demo only; do not describe as full RL training."
        if claim_id == "C7_WEBARENA_STRESS":
            writing_instruction = "Only claim p99/component effects if paired p99/component fields support them."
        output.append(
            {
                "claim_id": claim_id,
                "source_round": source_round,
                "support_status": status,
                "row_count": str(row_count),
                "claim_scope": scope,
                "writing_instruction": writing_instruction,
                "gate_status": "pass" if status == "supported" else "blocked",
            }
        )
    write_csv(root / "global" / "final_claim_support_matrix.csv", output)
    write_md(
        root / "global" / "final_claim_support_matrix.md",
        [
            "# Final Claim Support Matrix",
            "",
            *[f"- {row['claim_id']}: {row['support_status']} ({row['row_count']} rows)" for row in output],
        ],
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    args = parser.parse_args()
    build_claim_support_matrix(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
