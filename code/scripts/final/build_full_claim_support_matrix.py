#!/usr/bin/env python3
"""Build the full v2 claim-support matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.full_evidence_common import read_csv, resolve_final_root, truthy, write_csv


CLAIMS = [
    ("C1", "The suite is an evidence-gated executable benchmark substrate over real WebArena, MiniWoB, and SWE tasks."),
    ("C2", "Paper-facing rows are admitted only when traffic source, manifests, outcomes, trace boundaries, schema/replay/freeze metadata are complete."),
    ("C3", "Contract-conforming LLM drivers produce real benchmark traffic and expose model-call cost, token use, invalid actions, and environment/verifier timing."),
    ("C4", "Multiple model families/backends can be represented in the same substrate, while unsupported pairs are reported."),
    ("C5", "WebArena Qwen traffic cost is measurable; smoke-only WebArena Qwen-30B must not be paper-facing."),
    ("C6", "MiniWoB telemetry/concurrency under LLM traffic is measurable and real browser-backed."),
    ("C7", "SWE patch-provider traffic is measurable under a real upstream harness/verifier."),
    ("C8", "SWE queue story is scoped by measurable/significant/dominant queue labels."),
    ("C9", "SWE failure diversity is supported only when multiple natural failure categories appear."),
    ("C10", "WebArena stress does not claim p99 increase or component shift without direct paired support."),
    ("C11", "WebArena controller reversal is bounded pilot unless confirmatory paired backend/seed evidence exists."),
    ("C12", "vLLM/SGLang contrast is compatibility/measurement evidence, not universal backend superiority."),
    ("C13", "veRL/TRL are adapter demos, not full RL training results."),
    ("C14", "Every figure/table has a source path, derivation script, input CSV, and gate status."),
]


def _source_roots(rows: list[dict[str, str]], predicate) -> str:
    return ";".join(sorted({row.get("source_root", "") for row in rows if predicate(row) and row.get("source_root", "")}))


def _metric_values(rows: list[dict[str, str]], keys: list[str]) -> str:
    parts: list[str] = []
    for key in keys:
        vals = sorted({row.get(key, "") for row in rows if row.get(key, "")})
        if vals:
            parts.append(f"{key}={','.join(vals[:8])}")
    return "; ".join(parts)


def build_full_claim_support_matrix(*, final_root: Path) -> list[dict[str, str]]:
    final_root = final_root.resolve()
    rows = read_csv(final_root / "inventories" / "paper_facing_surface.csv")
    figures = read_csv(final_root / "inventories" / "figure_input_inventory.csv")
    out: list[dict[str, str]] = []

    families = {row.get("family", "") for row in rows}
    model_families = {row.get("model_family", "") for row in rows if row.get("model_family", "")}
    backend_engines = {row.get("backend_engine", "") for row in rows if row.get("backend_engine", "")}
    has_llm = any(row.get("driver_type") == "llm_agent" for row in rows)
    has_queue = any(truthy(row.get("dominant_queueing")) or truthy(row.get("significant_queueing")) or truthy(row.get("measurable_queueing")) for row in rows)
    has_dominant_queue = any(truthy(row.get("dominant_queueing")) for row in rows)
    single_failure_share = any(
        row.get("dominant_failure_category") in {"patch_apply_failure", "controlled_timeout"}
        and str(row.get("natural_failure_share") or row.get("controlled_failure_share") or "") in {"1", "1.0"}
        for row in rows
    )
    has_webarena_stress_support = any(
        truthy(row.get("p99_increase_supported")) or truthy(row.get("component_shift_supported")) for row in rows
    )
    has_controller_confirmatory = any(truthy(row.get("controller_reversal_supported")) for row in rows)
    has_framework = any(row.get("evidence_class") == "framework_adapter" for row in rows)
    has_backend_contrast = len(backend_engines) >= 2 or any(row.get("evidence_class") == "model_backend_contrast" for row in rows)

    for claim_id, text in CLAIMS:
        status = "blocked"
        writing = text
        predicate = lambda row: True
        metrics = ""
        if claim_id == "C1":
            status = "supported" if {"webarena_verified", "miniwob", "swe_gym"} & families else "blocked"
            predicate = lambda row: row.get("family", "") in {"webarena_verified", "miniwob", "swe_gym"}
        elif claim_id == "C2":
            status = "supported" if rows else "blocked"
            metrics = f"paper_rows={len(rows)}"
        elif claim_id == "C3":
            status = "supported" if has_llm else "blocked"
            predicate = lambda row: row.get("driver_type") == "llm_agent"
            metrics = _metric_values(rows, ["model_latency_ms", "prompt_tokens", "invalid_action_rate"])
        elif claim_id == "C4":
            status = "supported" if model_families or backend_engines else "blocked"
            metrics = f"model_families={len(model_families)}; backend_engines={len(backend_engines)}"
        elif claim_id == "C5":
            status = "supported" if any(row.get("family") == "webarena_verified" and row.get("model_family") == "qwen" for row in rows) else "blocked"
            writing = text + " Keep Qwen-30B WebArena rows smoke-only unless the gate admits non-smoke real rows."
        elif claim_id == "C6":
            status = "supported" if any(row.get("family") == "miniwob" for row in rows) else "blocked"
            predicate = lambda row: row.get("family") == "miniwob"
        elif claim_id == "C7":
            status = "supported" if any(row.get("family") == "swe_gym" for row in rows) else "blocked"
            predicate = lambda row: row.get("family") == "swe_gym"
        elif claim_id == "C8":
            status = "supported" if has_dominant_queue else ("narrowed" if has_queue else "blocked")
            writing = text + (" Dominant wording allowed only for rows labeled dominant_queueing." if has_dominant_queue else " Use measurable/significant wording only.")
            predicate = lambda row: row.get("family") == "swe_gym"
        elif claim_id == "C9":
            status = "narrowed" if single_failure_share else "supported"
            writing = "SWE ecological failure diversity must not be claimed when a single dominant natural or controlled failure category has share 1.0."
            predicate = lambda row: row.get("family") == "swe_gym"
        elif claim_id == "C10":
            status = "supported" if has_webarena_stress_support else ("narrowed" if any(row.get("family") == "webarena_verified" for row in rows) else "blocked")
            writing = text + " Prefer operating-point wording if p99/component labels are absent."
            predicate = lambda row: row.get("family") == "webarena_verified"
        elif claim_id == "C11":
            status = "supported" if has_controller_confirmatory else "narrowed"
            writing = text + " Use bounded controller-sensitivity pilot wording unless confirmatory labels exist."
        elif claim_id == "C12":
            status = "supported" if has_backend_contrast else "appendix_only"
            writing = text
        elif claim_id == "C13":
            status = "narrowed" if has_framework else "appendix_only"
            writing = "veRL/TRL can be described only as adapter demos, not full RL training or improvement."
            predicate = lambda row: row.get("evidence_class") == "framework_adapter"
        elif claim_id == "C14":
            status = "supported" if figures else "blocked"
            metrics = f"figure_inventory_rows={len(figures)}"

        sources = _source_roots(rows, predicate)
        out.append(
            {
                "claim_id": claim_id,
                "final_claim_text": text,
                "support_status": status,
                "source_files": "inventories/paper_facing_surface.csv",
                "source_roots": sources,
                "row_count": str(sum(1 for row in rows if predicate(row))),
                "exact_metric_values": metrics,
                "writing_instruction": writing,
                "gate_status": "pass" if status in {"supported", "narrowed", "appendix_only"} else "blocked",
            }
        )

    write_csv(final_root / "inventories" / "claim_support_matrix.csv", out)
    lines = [
        "# Claim Support Matrix",
        "",
        "| Claim | Status | Writing instruction | Rows |",
        "|---|---|---|---|",
    ]
    for row in out:
        lines.append(f"| {row['claim_id']} | {row['support_status']} | {row['writing_instruction']} | {row['row_count']} |")
    (final_root / "reports" / "claim_support_matrix.md").parent.mkdir(parents=True, exist_ok=True)
    (final_root / "reports" / "claim_support_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    args = parser.parse_args()
    rows = build_full_claim_support_matrix(final_root=resolve_final_root(Path.cwd(), args.final_root))
    print(json.dumps({"claims": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
