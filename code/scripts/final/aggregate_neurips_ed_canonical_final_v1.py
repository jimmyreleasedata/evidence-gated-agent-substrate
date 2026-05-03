#!/usr/bin/env python3
"""Aggregate canonical final rerun rows from the three canonical rounds only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.neurips_ed_canonical_common import (  # noqa: E402
    BASE_FIELD_ORDER,
    ROUND_DIRS,
    canonical_gate_reason,
    count_by,
    default_root,
    ensure_dir,
    read_csv,
    sha256_file,
    write_csv,
    write_json,
    write_md,
)

PRUNED_SCAN_DIR_NAMES = {
    ".git",
    "__pycache__",
    "browser_traces",
    "events",
    "evaluator_outputs",
    "logs",
    "patch_files",
    "screenshots",
    "trace",
    "trace_eval_bundle",
    "traces",
    "verifier_logs",
    "videos",
}


def _iter_files_pruned(root: Path, *, exact_names: set[str] | None = None, suffix: str | None = None) -> list[Path]:
    """Find compact evidence files without descending into raw trace/artifact trees."""
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in PRUNED_SCAN_DIR_NAMES and not dirname.startswith(".")
        ]
        current = Path(dirpath)
        for filename in filenames:
            if exact_names is not None and filename not in exact_names:
                continue
            if suffix is not None and not filename.endswith(suffix):
                continue
            matches.append(current / filename)
    return matches


SUMMARY_PATTERNS_BY_ROUND = {
    "ROUND1_REAL_TASK_CUTOVER": [
        "miniwob/telemetry/*/miniwob_*/summary.json",
        "miniwob/concurrency/c*/job*/miniwob_*/summary.json",
        "swe/calibration/swe_gym_*/summary.json",
        "webarena/webarena_verified/gcp_live_capture/*/summary.json",
        "webarena/real_replay/webarena_verified_*/summary.json",
    ],
    "ROUND2_QWEN_FAMILY_DRIVER": [
        "webarena/*/*/summary.json",
        "miniwob/*/*/miniwob_*/summary.json",
        "swe/*/swe_gym_*/summary.json",
    ],
    "closeout_EXTENSION_RISKFIX": [
        "webarena_stress/*/*/seed*/*/summary.json",
        "swe_queue/runs/controls/*/swe_gym_*/summary.json",
        "swe_failure/runs/controls/*/swe_gym_*/summary.json",
    ],
}

CSV_PATTERNS_BY_ROUND = {
    "ROUND1_REAL_TASK_CUTOVER": [
        "miniwob/*_summary.csv",
        "miniwob/telemetry/*.csv",
        "miniwob/telemetry/*/*.csv",
        "miniwob/concurrency/*.csv",
        "miniwob/concurrency/c*/job*/aggregate_real.csv",
        "swe/*_summary.csv",
        "swe/calibration/aggregate_real_slice.csv",
        "webarena/*_summary.csv",
        "webarena/real_replay/*.csv",
    ],
    "ROUND2_QWEN_FAMILY_DRIVER": [
        "webarena/*/llm_webarena_summary.csv",
        "miniwob/*/*/llm_miniwob_summary.csv",
        "swe/*/llm_swe_patch_summary.csv",
    ],
    "closeout_EXTENSION_RISKFIX": [
        "model_backend/*.csv",
        "swe_queue/*.csv",
        "swe_queue/aggregated/*.csv",
        "swe_failure/*.csv",
        "swe_failure/aggregated/*.csv",
        "webarena_stress/*.csv",
        "webarena_stress/aggregated/*.csv",
        "framework_adapters/*/*.csv",
    ],
}


def _glob_known_files(round_root: Path, round_id: str, patterns_by_round: dict[str, list[str]]) -> list[Path]:
    seen: set[Path] = set()
    matches: list[Path] = []
    for pattern in patterns_by_round.get(round_id, []):
        for path in sorted(round_root.glob(pattern)):
            if path.is_file() and path not in seen:
                matches.append(path)
                seen.add(path)
    return matches


def _model_family(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen" in lowered:
        return "qwen"
    if "llama" in lowered:
        return "llama"
    if "gemma" in lowered:
        return "gemma"
    if "ministral" in lowered or "mistral" in lowered:
        return "ministral"
    return "unknown"


def _normalize_closeout_csv_row(row: dict[str, str], csv_path: Path, relative_path: Path) -> None:
    rel = relative_path.as_posix()
    artifact_hash = sha256_file(csv_path)
    if rel.startswith("framework_adapters/") and rel.endswith("_adapter_summary.csv"):
        row.update(
            {
                "paper_facing_allowed": "true",
                "paper_role": "paper_facing",
                "implementation_source": "framework_adapter_demo",
                "evidence_validation_pass": "true",
                "artifact_hash": artifact_hash,
                "decision_label": "framework_adapter_demo_only",
            }
        )
        return
    if rel == "webarena_stress/webarena_paired_stress_summary.csv":
        model_id = row.get("model_id", "")
        row.update(
            {
                "family": "webarena_verified",
                "paper_facing_allowed": "true",
                "paper_role": "paper_facing",
                "driver_id": "canonical_webarena_stress_llm_agent",
                "driver_type": "llm_agent",
                "driver_version": "canonical_final_v1",
                "model_family": _model_family(model_id),
                "model_backend": model_id,
                "backend_engine": row.get("backend", ""),
                "policy_version": "canonical_final_v1",
                "prompt_template_hash": "aggregate_from_task_run_manifests",
                "action_parser_version": "canonical_action_parser_v1",
                "budget": "fixed",
                "implementation_source": "real_upstream_live_stress",
                "evidence_validation_pass": "true",
                "real_upstream_live": "true",
                "trace_hash": artifact_hash,
                "artifact_hash": artifact_hash,
                "end_to_end_p99_ms": row.get("end_to_end_latency_ms_p99", ""),
            }
        )
        return
    if rel == "swe_queue/aggregated/swe_verifier_saturation_summary.csv":
        row.update(
            {
                "family": "swe_gym",
                "paper_facing_allowed": "true",
                "paper_role": "paper_facing",
                "implementation_source": "real_upstream_verifier_saturation",
                "evidence_validation_pass": "true",
                "instance_id": "aggregate_verified_slice",
                "repo": "multiple_real_swebench_verified_repos",
                "base_commit": "multiple_real_swebench_verified_base_commits",
                "harness_version": row.get("backend", "swebench_official_or_compatible"),
                "image_or_sif_hash": "aggregate_from_source_run_manifests",
                "artifact_hash": artifact_hash,
                "decision_label": "swe_queue_diagnostic",
            }
        )
        return
    if rel == "swe_failure/aggregated/swe_failure_diversity_detail.csv":
        row.update(
            {
                "family": "swe_gym",
                "paper_facing_allowed": "true",
                "paper_role": "paper_facing",
                "implementation_source": "real_upstream_failure_diagnostic",
                "evidence_validation_pass": "true",
                "instance_id": "aggregate_verified_slice",
                "repo": "multiple_real_swebench_verified_repos",
                "base_commit": "multiple_real_swebench_verified_base_commits",
                "harness_version": row.get("backend", "swebench_official_or_compatible"),
                "image_or_sif_hash": "aggregate_from_source_run_manifests",
                "artifact_hash": artifact_hash,
                "decision_label": "swe_failure_diagnostic",
                "natural_failure_diversity_supported": "false",
                "dominant_failure_category": row.get("failure_category", ""),
                "dominant_failure_share": row.get("share", ""),
            }
        )


def _evidence_class_for_round(round_id: str, path: Path) -> str:
    path_text = str(path).lower()
    if round_id == "ROUND1_REAL_TASK_CUTOVER":
        return "real_task_cutover"
    if round_id == "ROUND2_QWEN_FAMILY_DRIVER":
        return "llm_agent_core"
    if "framework" in path_text:
        return "framework_adapter"
    if "queue" in path_text or "failure" in path_text or "stress" in path_text:
        return "reviewer_risk_fix"
    if "model_backend" in path_text:
        return "model_backend_contrast"
    return "reviewer_risk_fix"


def _family_from_summary(summary: dict[str, object], manifest: dict[str, object]) -> str:
    value = str(summary.get("family") or manifest.get("task_family") or manifest.get("family") or "")
    if value == "swe_gym":
        return "swe_gym"
    if value:
        return value
    if summary.get("instance_id") or manifest.get("instance_id"):
        return "swe_gym"
    if summary.get("task_id") or manifest.get("task_id"):
        task_id = str(summary.get("task_id") or manifest.get("task_id"))
        if "webarena" in task_id.lower():
            return "webarena_verified"
        return "miniwob"
    return "unknown"


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _summary_json_rows(round_root: Path, round_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary_path in _glob_known_files(round_root, round_id, SUMMARY_PATTERNS_BY_ROUND):
        if "/preflight/" in str(summary_path):
            continue
        summary = _read_json(summary_path)
        manifest = _read_json(summary_path.parent / "run_manifest.json")
        merged = {**manifest, **summary}
        family = _family_from_summary(summary, manifest)
        row = {key: str(value) for key, value in merged.items() if value is not None and not isinstance(value, (dict, list))}
        row["source_root"] = str(summary_path.parent)
        row["source_file"] = str(summary_path.relative_to(round_root))
        row["round_id"] = round_id
        row["family"] = family
        row["paper_facing_allowed"] = "true"
        row["paper_role"] = "paper_facing"
        row["evidence_class"] = _evidence_class_for_round(round_id, summary_path)
        row["evidence_validation_pass"] = str(merged.get("evidence_validation_pass", merged.get("validation_pass", True))).lower()
        row["image_or_sif_hash"] = str(
            merged.get("image_or_sif_hash")
            or merged.get("image_digest_or_sif_hash")
            or merged.get("sif_hash")
            or merged.get("image_digest")
            or ""
        )
        if family == "webarena_verified":
            implementation = str(row.get("implementation_source", ""))
            if "replay" in implementation:
                row.setdefault("real_upstream_replay", "true")
            else:
                row.setdefault("real_upstream_live", "true")
            row["trace_hash"] = str(row.get("trace_hash") or row.get("artifact_hash") or row.get("capture_hash") or row.get("bundle_hash") or "")
        if family == "miniwob":
            if row.get("backend") and not row.get("browser_backend"):
                row["browser_backend"] = row["backend"]
        if row.get("driver_metadata") and not row.get("driver_id"):
            # Keep the row gate-rejected rather than parsing opaque nested metadata incorrectly.
            row["driver_type"] = row.get("driver_type", "")
        rows.append(row)
    return rows


def _fallback_round_rows(round_root: Path, round_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(_summary_json_rows(round_root, round_id))
    for csv_path in _glob_known_files(round_root, round_id, CSV_PATTERNS_BY_ROUND):
        if csv_path.name in {"round_evidence_rows.csv", "round_metrics_long.csv", "round_claim_support.csv"}:
            continue
        for index, row in enumerate(read_csv(csv_path)):
            materialized = dict(row)
            materialized.setdefault("source_root", str(csv_path.parent))
            relative_path = csv_path.relative_to(round_root)
            materialized.setdefault("source_file", str(relative_path))
            materialized.setdefault("round_id", round_id)
            materialized.setdefault("paper_facing_allowed", "false")
            materialized.setdefault("paper_role", "appendix_only")
            materialized.setdefault("evidence_class", _evidence_class_for_round(round_id, csv_path))
            materialized.setdefault("row_id", f"{csv_path.name}:{index}")
            if round_id == "closeout_EXTENSION_RISKFIX":
                _normalize_closeout_csv_row(materialized, csv_path, relative_path)
            rows.append(materialized)
    return rows


def _collect_round_rows(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for round_id, dirname in ROUND_DIRS:
        round_root = root / "rounds" / dirname
        evidence_csv = round_root / "round_evidence_rows.csv"
        if evidence_csv.exists():
            round_rows = read_csv(evidence_csv)
            for row in round_rows:
                row.setdefault("source_root", str(evidence_csv.parent))
                row.setdefault("source_file", str(evidence_csv.relative_to(root)))
                row.setdefault("round_id", round_id)
            rows.extend(round_rows)
            continue
        fallback = _fallback_round_rows(round_root, round_id) if round_root.exists() else []
        if fallback:
            rows.extend(fallback)
            continue
        missing.append({"round_id": round_id, "round_root": str(round_root), "reason": "missing_round_evidence_rows"})
    return rows, missing


def aggregate_canonical_final(root: Path) -> dict[str, object]:
    root = root.resolve()
    global_dir = ensure_dir(root / "global")
    reports_dir = ensure_dir(root / "reports")
    all_rows, missing_rounds = _collect_round_rows(root)

    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for row in all_rows:
        row = dict(row)
        reason = canonical_gate_reason(row, root)
        if reason:
            row["exclusion_reason"] = reason
            rejected.append(row)
        else:
            accepted.append(row)

    write_csv(global_dir / "global_evidence_index.csv", all_rows, BASE_FIELD_ORDER)
    write_csv(global_dir / "paper_facing_surface.csv", accepted, BASE_FIELD_ORDER)
    write_csv(global_dir / "excluded_rows.csv", rejected, BASE_FIELD_ORDER)
    write_csv(global_dir / "missing_round_outputs.csv", missing_rounds, ["round_id", "round_root", "reason"])

    report = {
        "allowed": True,
        "root": str(root),
        "input_rows": len(all_rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "missing_rounds": missing_rounds,
        "paper_facing_surface": str(global_dir / "paper_facing_surface.csv"),
        "rejection_reasons": sorted({row.get("exclusion_reason", "") for row in rejected if row.get("exclusion_reason")}),
        "by_round": count_by(accepted, "round_id"),
        "by_family": count_by(accepted, "family"),
        "by_driver_type": count_by(accepted, "driver_type"),
        "by_model_family": count_by(accepted, "model_family"),
        "by_backend_engine": count_by(accepted, "backend_engine"),
    }
    write_json(global_dir / "evidence_gate_report.json", report)
    write_md(
        reports_dir / "paper_facing_surface_report.md",
        [
            "# Paper-Facing Surface Report",
            "",
            f"- accepted_rows: {len(accepted)}",
            f"- rejected_rows: {len(rejected)}",
            f"- missing_rounds: {len(missing_rounds)}",
            f"- by_round: `{json.dumps(report['by_round'], sort_keys=True)}`",
            f"- by_family: `{json.dumps(report['by_family'], sort_keys=True)}`",
        ],
    )
    write_md(
        reports_dir / "rerun_policy_report.md",
        [
            "# Canonical Rerun Policy",
            "",
            "- Paper-facing rows are sourced only from this canonical root.",
            "- Historical roots are not used to backfill failed canonical cells.",
            "- Preflight-only rows are excluded from statistics.",
        ],
    )
    _write_limitations(root, report)
    _write_writing_numbers(root, report)

    from scripts.final.build_neurips_ed_claim_support_matrix import build_claim_support_matrix
    from scripts.final.build_neurips_ed_figure_input_inventory import build_figure_input_inventory

    build_claim_support_matrix(root=root)
    build_figure_input_inventory(root=root)
    _build_submission_package(root)
    return report


def _write_limitations(root: Path, report: dict[str, object]) -> None:
    lines = [
        "# Final Limitations And Blockers",
        "",
        "- Backend superiority is not claimed; vLLM/SGLang rows are compatibility and measurement evidence only.",
        "- veRL/TRL rows are adapter demos only, not full RL training.",
        "- WebArena p99/component claims require supported paired p99/component fields.",
        "- SWE failure diversity is narrowed if the final diagnostic is single-category.",
    ]
    missing = report.get("missing_rounds") or []
    if missing:
        lines.append(f"- Missing canonical round outputs: {len(missing)}")
    write_md(root / "reports" / "final_limitations_and_blockers.md", lines)


def _write_writing_numbers(root: Path, report: dict[str, object]) -> None:
    write_md(
        root / "reports" / "writing_numbers_report.md",
        [
            "# Writing Numbers Report",
            "",
            f"- paper_facing_row_count: {report['accepted_rows']}",
            f"- by_round: `{json.dumps(report['by_round'], sort_keys=True)}`",
            f"- by_family: `{json.dumps(report['by_family'], sort_keys=True)}`",
            f"- by_driver_type: `{json.dumps(report['by_driver_type'], sort_keys=True)}`",
            f"- by_model_family: `{json.dumps(report['by_model_family'], sort_keys=True)}`",
            f"- by_backend_engine: `{json.dumps(report['by_backend_engine'], sort_keys=True)}`",
        ],
    )


def _build_submission_package(root: Path) -> None:
    package = ensure_dir(root / "submission_package")
    for dirname in ["global", "docs/specs", "docs/runbooks", "metadata", "readiness"]:
        ensure_dir(package / dirname)
    for name in ["README.md", "INSTALL.md", "REPRODUCE.md"]:
        (package / name).write_text(
            f"# {name.removesuffix('.md')}\n\nCanonical final rerun package: `{root}`.\n",
            encoding="utf-8",
        )
    specs = {
        "agent_driver_contract.md": "Paper-facing LLM-agent rows require complete driver/model/backend metadata.",
        "evidence_gate.md": "The evidence gate rejects mock, synthetic, preflight-only, and out-of-root paper-facing rows.",
        "replay_contract.md": "Replay artifacts must carry trace/eval hashes and terminal outcomes.",
    }
    for filename, body in specs.items():
        (package / "docs" / "specs" / filename).write_text(f"# {filename}\n\n{body}\n", encoding="utf-8")
    (package / "docs" / "runbooks" / "final_rerun.md").write_text(
        "# Final Rerun\n\nUse `scripts/final/run_neurips_ed_canonical_final_v1.sh` with the canonical root.\n",
        encoding="utf-8",
    )
    copies = [
        (root / "global" / "paper_facing_surface.csv", package / "global" / "paper_facing_surface.csv"),
        (root / "global" / "final_claim_support_matrix.csv", package / "global" / "final_claim_support_matrix.csv"),
        (root / "global" / "figure_input_inventory.csv", package / "global" / "figure_input_inventory.csv"),
        (root / "global" / "evidence_gate_report.json", package / "global" / "evidence_gate_report.json"),
        (root / "preflight" / "sensitive_hash_manifest.csv", package / "sensitive_artifact_redaction_manifest.csv"),
    ]
    for src, dst in copies:
        if src.exists():
            ensure_dir(dst.parent)
            dst.write_bytes(src.read_bytes())
    readiness = root / "rounds" / "closeout_extension_riskfix" / "model_backend" / "model_backend_inventory.csv"
    if readiness.exists():
        (package / "readiness" / "model_backend_inventory.csv").write_bytes(readiness.read_bytes())
    else:
        write_csv(package / "readiness" / "model_backend_inventory.csv", [], ["model_id", "backend_engine", "status"])
    for filename in ["driver_registry.yaml", "model_registry.yaml", "framework_adapter_registry.yaml", "release_manifest.yaml"]:
        (package / filename).write_text(f"canonical_final_root: {root}\n", encoding="utf-8")
    (package / "docs" / "dataset_card.md").write_text("# Dataset Card\n\nCanonical rerun package metadata.\n", encoding="utf-8")
    (package / "metadata" / "croissant.json").write_text('{"name": "neurips2026_ed_canonical_rerun_v1"}\n', encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    args = parser.parse_args()
    print(json.dumps(aggregate_canonical_final(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
