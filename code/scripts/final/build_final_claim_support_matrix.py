#!/usr/bin/env python3
"""Build the final claim-support matrix for the evidence package."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CLAIMS: list[dict[str, str]] = [
    {
        "claim_id": "C1",
        "old_claim_text": "",
        "base_text": "Contract-conforming LLM drivers can plug into the substrate and generate real benchmark traffic across WebArena, MiniWoB, and SWE.",
        "evidence_category": "llm_agent_core",
        "figure_or_table": "Workload family / traffic source / claim scope",
    },
    {
        "claim_id": "C2",
        "old_claim_text": "",
        "base_text": "Different model families and serving backends expose measurable systems-metric differences.",
        "evidence_category": "model_backend_contrast",
        "figure_or_table": "Model/backend matrix",
    },
    {
        "claim_id": "C3",
        "old_claim_text": "",
        "base_text": "Workload families expose distinct bottleneck ownership under the same agent/driver contract.",
        "evidence_category": "real_task_cutover",
        "figure_or_table": "Cross-family latency table",
    },
    {
        "claim_id": "C4",
        "old_claim_text": "",
        "base_text": "Calibration/control traffic verifies evaluator/verifier discriminativeness and stress-surface validity.",
        "evidence_category": "calibration_control",
        "figure_or_table": "Calibration/control baseline table",
    },
    {
        "claim_id": "C5",
        "old_claim_text": "SWE queueing can become dominant.",
        "base_text": "SWE queueing becomes measurable/significant/dominant only under the decision labels produced by the saturation sweep.",
        "evidence_category": "reviewer_risk_fix",
        "figure_or_table": "SWE queue saturation table",
    },
    {
        "claim_id": "C6",
        "old_claim_text": "SWE failure-stressed traffic demonstrates ecological failure diversity.",
        "base_text": "SWE failure diversity is supported only if LLM patch-provider traffic produces natural non-controlled failure categories.",
        "evidence_category": "reviewer_risk_fix",
        "figure_or_table": "SWE failure diversity table",
    },
    {
        "claim_id": "C7",
        "old_claim_text": "WebArena stress increases p99 and reverses controllers.",
        "base_text": "WebArena stress changes operating point if paired deltas support behavior/component/controller shifts; no p99 or render/network claim is made without evidence.",
        "evidence_category": "reviewer_risk_fix",
        "figure_or_table": "WebArena paired-stress table",
    },
    {
        "claim_id": "C8",
        "old_claim_text": "",
        "base_text": "vLLM/SGLang backend contrast is compatibility and systems-measurement evidence, not universal backend superiority.",
        "evidence_category": "model_backend_contrast",
        "figure_or_table": "Backend contrast table",
    },
    {
        "claim_id": "C9",
        "old_claim_text": "",
        "base_text": "veRL/TRL framework adapters consume trajectory/evidence records as adapter demos, not full RL training results.",
        "evidence_category": "framework_adapter",
        "figure_or_table": "Framework adapter table",
    },
    {
        "claim_id": "C10",
        "old_claim_text": "",
        "base_text": "Evidence gate and traceability make every paper-facing figure auditable from run roots, manifests, processed inputs, and validators.",
        "evidence_category": "release_metadata",
        "figure_or_table": "Evidence gate and traceability coverage",
    },
]


def _status_for_claim(claim_id: str, decision_labels: dict[str, str], sources: list[str]) -> tuple[str, str, str]:
    has_source = bool(sources)
    if claim_id == "C5":
        label = decision_labels.get("swe_queue", "")
        if label == "dominant_queueing":
            return "supported", "SWE queueing can become dominant under verifier saturation.", "dominant_queueing decision label present."
        if label in {"significant_queueing", "measurable_queueing"}:
            return "narrowed", f"SWE queueing becomes {label.replace('_queueing', '')} under controlled verifier saturation.", f"{label} decision label present."
        return "blocked", "SWE queueing is reported as a diagnostic without dominance wording.", "No queueing decision label found."
    if claim_id == "C6":
        label = decision_labels.get("swe_failure", "")
        if label == "natural_failure_diversity_supported":
            return "supported", CLAIMS[5]["base_text"], "Multiple natural LLM failure categories observed."
        return "narrowed" if has_source else "blocked", CLAIMS[5]["base_text"], "Controlled failures remain calibration unless natural categories are observed."
    if claim_id == "C7":
        label = decision_labels.get("webarena_stress", "")
        if label == "p99_increase_supported":
            return "supported", "WebArena paired stress supports a scoped p99 increase claim.", "p99_increase_supported decision label present."
        if label:
            return "narrowed", CLAIMS[6]["base_text"], f"{label} decision label present."
        return "blocked", CLAIMS[6]["base_text"], "No paired WebArena stress decision label found."
    if claim_id in {"C1", "C2", "C3", "C4", "C8", "C9", "C10"}:
        return ("supported" if has_source else "appendix_only", "", "Evidence source present." if has_source else "No direct source path was found.")
    return "blocked", "", "No rule."


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Final Claim Support Matrix",
        "",
        "| Claim | Status | Final claim | Evidence | Reason |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        evidence = row["evidence_source_paths"].replace("|", "\\|")
        lines.append(
            f"| {row['claim_id']} | {row['support_status']} | {row['final_claim_text']} | {evidence} | {row['decision_reason']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_claim_support_matrix(
    output_csv: Path,
    output_md: Path,
    evidence_sources: dict[str, list[str]] | None = None,
    decision_labels: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    evidence_sources = evidence_sources or {}
    decision_labels = decision_labels or {}
    rows: list[dict[str, str]] = []
    for claim in CLAIMS:
        sources = evidence_sources.get(claim["claim_id"], [])
        status, final_text, reason = _status_for_claim(claim["claim_id"], decision_labels, sources)
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "old_claim_text": claim["old_claim_text"],
                "final_claim_text": final_text or claim["base_text"],
                "evidence_category": claim["evidence_category"],
                "evidence_source_paths": ";".join(sources),
                "support_status": status,
                "decision_reason": reason,
                "figure_or_table": claim["figure_or_table"],
                "required_fields_present": str(bool(sources)).lower(),
                "gate_status": "pass" if status in {"supported", "narrowed", "appendix_only"} else "blocked",
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(output_md, rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    build_claim_support_matrix(args.output_csv, args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
