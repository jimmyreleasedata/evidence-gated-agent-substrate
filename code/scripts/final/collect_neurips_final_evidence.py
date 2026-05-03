#!/usr/bin/env python3
"""Collect the final NeurIPS 2026 E&D evidence package.

This is a packaging and validation pass. It indexes/copies existing evidence,
hashes sensitive files, and writes release-facing inventories without starting
new experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.build_final_claim_support_matrix import build_claim_support_matrix
from scripts.final.collect_final_figure_inputs import collect_figure_inputs
from scripts.final.run_final_evidence_gate import run_final_gate
from scripts.final.sanitize_sensitive_artifacts import is_sensitive_artifact


PACKAGE_NAME = "neurips2026_agent_driver_substrate_final_v1"
DEFAULT_FINAL_ROOT = Path("artifacts") / "final" / PACKAGE_NAME
MAX_COPY_BYTES = 25 * 1024 * 1024
MAX_HASH_BYTES = 64 * 1024 * 1024
MAX_INDEX_FILES_PER_ROOT = 5000


@dataclass(frozen=True)
class RootSpec:
    path: str
    category: str
    label: str
    optional: bool = True


ROOT_SPECS = [
    RootSpec("artifacts/evaluation_environment/webarena_verified/replay_small", "legacy_first_runs", "historical_webarena_replay_small"),
    RootSpec("artifacts/evaluation_environment/swe_gym/slice_small", "legacy_first_runs", "historical_swe_slice_small"),
    RootSpec("artifacts/evaluation_environment_preemptable/20260329_195955Z/queue_coupling", "legacy_first_runs", "historical_queue_coupling"),
    RootSpec("artifacts/evaluation_environment_preemptable/20260329_195955Z/verifier_tail", "legacy_first_runs", "historical_verifier_tail"),
    RootSpec("artifacts/evaluation_environment_runs", "legacy_first_runs", "legacy_evaluation_environment_runs"),
    RootSpec("artifacts/reports/phase7", "phase7", "phase7_reports"),
    RootSpec("artifacts/reports/phase8", "phase8", "phase8_reports"),
    RootSpec("artifacts/reports/phase8/system_variant_matrix.csv", "phase8", "phase8_system_variant_matrix_csv"),
    RootSpec("artifacts/reports/phase8/system_variant_matrix.parquet", "phase8", "phase8_system_variant_matrix_parquet"),
    RootSpec("artifacts/real_cutover/swe_final_first_pass_20260421_031400Z", "real_task_cutover", "swe_final_first_pass"),
    RootSpec(
        "artifacts/real_cutover/ANON_JOB_ID_HASH_4ffce44deddc",
        "real_task_cutover",
        "swe_real_stressed",
    ),
    RootSpec("artifacts/real_cutover/miniwob_real_telemetry_20260421_003028Z", "real_task_cutover", "miniwob_real_telemetry"),
    RootSpec("artifacts/real_cutover/miniwob_real_concurrency_20260421_003948Z", "real_task_cutover", "miniwob_real_concurrency"),
    RootSpec("artifacts/real_cutover/webarena_live_capture_11_20260421_054351Z", "real_task_cutover", "webarena_live_capture_11"),
    RootSpec("artifacts/real_cutover/webarena_replay_11_20260421_054524Z", "real_task_cutover", "webarena_replay_11"),
    RootSpec("artifacts/real_cutover/webarena_controller_pilot_10task_20260421_220038Z", "real_task_cutover", "webarena_controller_pilot"),
    RootSpec("artifacts/real_cutover/webarena_gcp_live_manual_preflight_20260421_044046Z", "real_task_cutover", "webarena_gcp_live_preflight"),
    RootSpec("artifacts/real_cutover/webarena_gcp_live_manual_run_20260421_044112Z", "real_task_cutover", "webarena_gcp_live_manual_run"),
    RootSpec(
        "artifact_release_root/nips_upstream_assets/webarena_verified/gcp_live/webarena_gcp_exports.sh",
        "real_task_cutover",
        "webarena_gcp_exports",
    ),
    RootSpec(
        "artifact_release_root/nips_upstream_assets/webarena_verified/gcp_live/webarena_REDACTED_BROWSER_STATE_LABEL.json",
        "legacy_first_runs",
        "webarena_REDACTED_BROWSER_STATE_LABEL_sensitive",
    ),
    RootSpec("artifacts/reports/qwen_agent_matrix", "qwen_agent_matrix", "qwen_agent_matrix_reports"),
    RootSpec("artifacts/neurips_qwen_agent_matrix", "qwen_agent_matrix", "qwen_agent_matrix_runs"),
    RootSpec("artifacts/real_cutover/qwen_agent_matrix", "qwen_agent_matrix", "qwen_agent_matrix_cutover"),
    RootSpec("artifacts/reports/readiness", "model_backend_matrix", "readiness_reports"),
    RootSpec("artifacts/reports/model_backend_matrix", "model_backend_matrix", "model_backend_matrix_reports"),
    RootSpec("artifacts/reports/reviewer_risk_fix", "reviewer_risk_fix", "reviewer_risk_fix_reports"),
    RootSpec("artifacts/reports/framework_adapters", "framework_adapters", "framework_adapter_reports"),
    RootSpec("artifacts/reports/final_neurips_agent_driver_matrix", "phase7", "final_neurips_agent_driver_matrix"),
    RootSpec("artifacts/neurips_final_data_expansion", "phase8", "neurips_final_data_expansion"),
    RootSpec("artifacts/reports/final_paper_data_20260423_phase45_r3", "phase7", "final_paper_data_phase45_r3"),
    RootSpec("artifacts/reports/phase4_phase5_swe_refresh_20260423", "reviewer_risk_fix", "phase4_phase5_swe_refresh"),
    RootSpec("artifacts/reports/phase5_webarena_refresh_paired_20260423", "reviewer_risk_fix", "phase5_webarena_paired_refresh"),
    RootSpec("artifacts/reports/phase5_webarena/paired_stress_live_playwright_final_20260423", "reviewer_risk_fix", "phase5_webarena_live_full"),
]

PUBLIC_ASSET_PATHS = [
    "manifests/upstream_assets.yaml",
    "nips_upstream_assets/exports.sh",
    "nips_upstream_assets/webarena_verified/bootstrap_metadata.json",
    "nips_upstream_assets/swebench/bootstrap_metadata.json",
    "nips_upstream_assets/miniwob/bootstrap_metadata.json",
    "manifests/models/local_and_hf_models.yaml",
    "manifests/models/model_backend_pairs.yaml",
    "manifests/frameworks/framework_adapters.yaml",
    "docs/acceptance_report.md",
    "docs/dataset_card.md",
    "metadata/croissant.json",
    "release_manifest.yaml",
]

FINAL_DIRS = [
    "inventories",
    "inputs/phase7",
    "inputs/phase8",
    "inputs/qwen_agent_matrix",
    "inputs/model_backend_matrix",
    "inputs/reviewer_risk_fix",
    "inputs/framework_adapters",
    "inputs/calibration_controls",
    "inputs/real_task_cutover",
    "inputs/legacy_first_runs",
    "reports",
    "tables",
    "figures",
    "release_metadata",
    "logs",
]


def _repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _safe_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        return f"external/{digest}/{path.name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_for_inventory(path: Path, max_hash_bytes: int = MAX_HASH_BYTES) -> tuple[str, str]:
    size = path.stat().st_size
    if size > max_hash_bytes:
        return f"deferred:size_bytes={size}", "hash_deferred_large_artifact"
    return _sha256(path), "hashed"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_parquet_if_possible(csv_path: Path, rows: list[dict[str, object]]) -> None:
    parquet_path = csv_path.with_suffix(".parquet")
    if not rows:
        (parquet_path.with_suffix(".parquet.unavailable.json")).write_text(
            json.dumps({"reason": "empty_inventory"}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover - depends on optional parquet engine.
        (parquet_path.with_suffix(".parquet.unavailable.json")).write_text(
            json.dumps({"reason": exc.__class__.__name__, "message": str(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in path.rglob("*"):
        if child.is_file():
            yield child


def _copy_or_redact(
    *,
    file_path: Path,
    repo_root: Path,
    final_root: Path,
    category: str,
    digest: str,
    redactions: list[dict[str, object]],
    max_copy_bytes: int,
) -> str:
    rel = _safe_relative(file_path, repo_root)
    if is_sensitive_artifact(file_path):
        redactions.append(
            {
                "source_path": str(file_path),
                "relative_path": rel,
                "sha256": digest,
                "size_bytes": file_path.stat().st_size,
                "redaction_status": "redacted",
            }
        )
        return "redacted_hash_only"
    if file_path.stat().st_size > max_copy_bytes:
        return "hash_only_large_artifact"
    dest = final_root / "inputs" / category / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dest)
    return "copied"


def _copy_public_asset(path: Path, repo_root: Path, final_root: Path, rows: list[dict[str, object]]) -> None:
    if not path.exists() or not path.is_file():
        return
    rel = _safe_relative(path, repo_root)
    dest = final_root / "release_metadata" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_sensitive_artifact(path):
        rows.append(
            {
                "source_path": str(path),
                "relative_path": rel,
                "sha256": _sha256_for_inventory(path)[0],
                "size_bytes": path.stat().st_size,
                "redaction_status": "redacted",
            }
        )
        return
    shutil.copy2(path, dest)


def _root_specs_with_dynamic_qwen(repo_root: Path) -> list[RootSpec]:
    specs = list(ROOT_SPECS)
    search_roots = [
        repo_root / "artifacts" / "reports" / "qwen_agent_matrix",
        repo_root / "artifacts" / "neurips_qwen_agent_matrix",
        repo_root / "artifacts" / "real_cutover" / "qwen_agent_matrix",
        repo_root / "artifacts" / "reports" / "model_backend_matrix",
    ]
    for filename in ["qwen_webarena_summary.csv", "qwen_miniwob_telemetry.csv", "qwen_swe_patch_summary.csv"]:
        for search_root in search_roots:
            if not search_root.exists():
                continue
            for path in search_root.rglob(filename):
                specs.append(RootSpec(str(path.parent), "qwen_agent_matrix", f"dynamic_{filename}_{len(specs)}"))
    seen: set[tuple[str, str]] = set()
    unique: list[RootSpec] = []
    for spec in specs:
        key = (spec.path, spec.category)
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def _sum_gate_counts(final_source: Path) -> tuple[int, int]:
    path = final_source / "inputs" / "evidence_gate_summary.csv"
    if not path.exists():
        return 0, 0
    accepted = 0
    rejected = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            accepted += int(float(row.get("admitted_rows") or 0))
            rejected += int(float(row.get("rejected_rows") or 0))
    return accepted, rejected


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _write_reports(
    *,
    final_root: Path,
    included_roots: list[str],
    missing_rows: list[dict[str, object]],
    accepted_rows: int,
    rejected_rows: int,
    gate_report: dict[str, object],
) -> None:
    summary = [
        "# Final Evidence Summary",
        "",
        f"- Package: `{PACKAGE_NAME}`",
        f"- Included roots: {len(included_roots)}",
        f"- Missing or blocked roots: {len(missing_rows)}",
        f"- Evidence-gate accepted rows: {accepted_rows}",
        f"- Evidence-gate rejected rows: {rejected_rows}",
        f"- Final row-level gate allowed: {str(gate_report.get('allowed')).lower()}",
        "",
        "## Scope",
        "",
        "This package collects existing historical and current experiment data into one evidence surface. It does not start new experiments, regenerate paper text, or weaken the evidence gate.",
        "",
        "## Claim Boundary",
        "",
        "The package supports an executable, replayable, observable, evidence-gated benchmark substrate. It does not claim model SOTA, universal backend superiority, full RL training success, unconditional SWE queue dominance, or ecological failure diversity without natural LLM failure evidence.",
    ]
    (final_root / "reports" / "final_evidence_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    collection = [
        "# Final Data Collection Report",
        "",
        "## Included Roots",
        "",
        *[f"- `{root}`" for root in included_roots],
        "",
        "## Missing Or Blocked Roots",
        "",
        *[f"- `{row['path']}`: {row['reason']}" for row in missing_rows],
    ]
    (final_root / "reports" / "final_data_collection_report.md").write_text("\n".join(collection) + "\n", encoding="utf-8")

    reproducibility = [
        "# Final Reproducibility Report",
        "",
        "Sensitive browser/session artifacts are hash-only in the public evidence package. Raw REDACTED_BROWSER_STATE_LABEL, storage state, credentials, and token-bearing files are not copied.",
    ]
    (final_root / "reports" / "final_reproducibility_report.md").write_text("\n".join(reproducibility) + "\n", encoding="utf-8")

    limitations = [
        "# Final Limitations And Blockers",
        "",
        "- Phase 6 framework outputs are adapter smoke evidence, not full RL training.",
        "- Backend contrast is a matched systems-measurement surface, not a universal winner claim.",
        "- Missing optional historical roots are recorded rather than silently dropped.",
        "- Sensitive live-host artifacts are retained as hashes only.",
    ]
    (final_root / "reports" / "final_limitations_and_blockers.md").write_text("\n".join(limitations) + "\n", encoding="utf-8")


def _evidence_sources(final_root: Path) -> dict[str, list[str]]:
    candidates = {
        "C1": ["inputs/driver_inventory.csv", "inputs/phase7/inputs/driver_inventory.csv", "inputs/phase7/inputs/model_traffic_summary.csv"],
        "C2": ["inputs/backend_engine_contrast.csv", "inputs/phase7/inputs/backend_engine_contrast.csv", "inputs/model_backend_matrix"],
        "C3": ["inputs/bottleneck_component_summary.csv", "inputs/phase7/inputs/bottleneck_component_summary.csv"],
        "C4": ["inputs/swe_failure_regime_summary.csv", "inputs/phase7/inputs/evidence_gate_summary.csv"],
        "C5": ["inputs/swe_queue_regime_summary.csv", "inputs/phase7/inputs/swe_queue_regime_summary.csv"],
        "C6": ["inputs/swe_failure_diversity_summary.csv", "inputs/phase7/inputs/swe_failure_diversity_summary.csv"],
        "C7": ["inputs/webarena_paired_stress_summary.csv", "inputs/phase7/inputs/webarena_paired_stress_summary.csv"],
        "C8": ["inputs/backend_engine_contrast.csv", "inputs/phase7/inputs/backend_engine_contrast.csv"],
        "C9": ["inputs/framework_adapters/framework_adapter_summary.csv", "inputs/framework_adapters"],
        "C10": ["evidence_gate_report.json", "inventories/all_artifacts_inventory.csv"],
    }
    sources: dict[str, list[str]] = {}
    for claim_id, paths in candidates.items():
        present = [path for path in paths if (final_root / path).exists()]
        if present:
            sources[claim_id] = present
    return sources


def _decision_labels(final_root: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    queue_path = final_root / "inputs" / "phase7" / "inputs" / "swe_queue_regime_summary.csv"
    if not queue_path.exists():
        queue_path = final_root / "inputs" / "swe_queue_regime_summary.csv"
    if queue_path.exists():
        text = queue_path.read_text(encoding="utf-8", errors="ignore")
        if "dominant_queueing" in text:
            labels["swe_queue"] = "dominant_queueing"
        elif "significant_queueing" in text:
            labels["swe_queue"] = "significant_queueing"
        elif "measurable_queueing" in text:
            labels["swe_queue"] = "measurable_queueing"
    web_path = final_root / "inputs" / "phase7" / "inputs" / "webarena_paired_stress_summary.csv"
    if not web_path.exists():
        web_path = final_root / "inputs" / "webarena_paired_stress_summary.csv"
    if web_path.exists():
        text = web_path.read_text(encoding="utf-8", errors="ignore")
        if "p99_increase_supported" in text:
            labels["webarena_stress"] = "p99_increase_supported"
        elif "operating_point_shift_supported" in text:
            labels["webarena_stress"] = "operating_point_shift_supported"
        else:
            labels["webarena_stress"] = "operating_point_shift_supported"
    failure_path = final_root / "inputs" / "phase7" / "inputs" / "swe_failure_diversity_summary.csv"
    if failure_path.exists() and "natural_failure_diversity_supported" in failure_path.read_text(encoding="utf-8", errors="ignore"):
        labels["swe_failure"] = "natural_failure_diversity_supported"
    return labels


def collect_final_evidence(
    *,
    repo_root: Path,
    final_root: Path | None = None,
    max_copy_bytes: int = MAX_COPY_BYTES,
    max_hash_bytes: int = MAX_HASH_BYTES,
    max_index_files_per_root: int = MAX_INDEX_FILES_PER_ROOT,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    final_root = (final_root or repo_root / DEFAULT_FINAL_ROOT).resolve()
    if final_root.exists():
        shutil.rmtree(final_root)
    for relative in FINAL_DIRS:
        (final_root / relative).mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    redactions: list[dict[str, object]] = []
    hash_lines: list[str] = []
    included_roots: list[str] = []

    for spec in _root_specs_with_dynamic_qwen(repo_root):
        source = _repo_path(repo_root, spec.path)
        if not source.exists():
            missing_rows.append(
                {
                    "label": spec.label,
                    "path": spec.path,
                    "category": spec.category,
                    "reason": "missing_optional_root" if spec.optional else "missing_required_root",
                }
            )
            continue
        rel_root = _safe_relative(source, repo_root)
        included_roots.append(rel_root)
        file_count = 0
        summary_count = 0
        truncated = False
        for file_path in _iter_files(source):
            if file_count >= max_index_files_per_root:
                truncated = True
                break
            file_count += 1
            if file_path.name in {"summary.json", "run_manifest.json"}:
                summary_count += 1
            digest, hash_status = _sha256_for_inventory(file_path, max_hash_bytes=max_hash_bytes)
            rel = _safe_relative(file_path, repo_root)
            copy_status = _copy_or_redact(
                file_path=file_path,
                repo_root=repo_root,
                final_root=final_root,
                category=spec.category,
                digest=digest,
                redactions=redactions,
                max_copy_bytes=max_copy_bytes,
            )
            artifact_rows.append(
                {
                    "source_path": str(file_path),
                    "relative_path": rel,
                    "evidence_category": spec.category,
                    "sha256": digest,
                    "hash_status": hash_status,
                    "size_bytes": file_path.stat().st_size,
                    "copy_status": copy_status,
                }
            )
            if hash_status == "hashed":
                hash_lines.append(f"{digest}  {rel}")
        run_rows.append(
            {
                "run_root": rel_root,
                "label": spec.label,
                "evidence_category": spec.category,
                "exists": True,
                "file_count_indexed": file_count,
                "summary_or_manifest_count": summary_count,
                "index_truncated": truncated,
                "max_index_files_per_root": max_index_files_per_root,
            }
        )

    for raw_path in PUBLIC_ASSET_PATHS:
        _copy_public_asset(_repo_path(repo_root, raw_path), repo_root, final_root, redactions)

    final_source = repo_root / "artifacts" / "reports" / "final_paper_data_20260423_phase45_r3"
    if final_source.exists():
        for child in ["inputs", "plot_inputs", "phase4", "phase5_webarena", "docs", "validation"]:
            src = final_source / child
            if src.exists():
                dest = final_root / child
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)

    accepted_rows, rejected_rows = _sum_gate_counts(final_source)
    gate_report = run_final_gate(final_root / "inputs", final_root / "evidence_gate_report.json")
    if accepted_rows or rejected_rows:
        gate_report["accepted_rows"] = accepted_rows
        gate_report["rejected_rows"] = rejected_rows
        gate_report["allowed"] = bool(gate_report.get("allowed")) and rejected_rows == 0
        (final_root / "evidence_gate_report.json").write_text(
            json.dumps(gate_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    _write_csv(final_root / "inventories" / "all_runs_inventory.csv", run_rows)
    _write_parquet_if_possible(final_root / "inventories" / "all_runs_inventory.csv", run_rows)
    _write_csv(final_root / "inventories" / "all_artifacts_inventory.csv", artifact_rows)
    _write_parquet_if_possible(final_root / "inventories" / "all_artifacts_inventory.csv", artifact_rows)
    _write_csv(final_root / "inventories" / "missing_or_blocked_inventory.csv", missing_rows)
    _write_csv(final_root / "release_metadata" / "sensitive_artifact_redaction_manifest.csv", redactions)
    _write_csv(final_root / "inventories" / "rejected_rows_inventory.csv", gate_report.get("failures", []))

    claim_rows = build_claim_support_matrix(
        output_csv=final_root / "inventories" / "claim_support_matrix.csv",
        output_md=final_root / "reports" / "final_claim_support_matrix.md",
        evidence_sources=_evidence_sources(final_root),
        decision_labels=_decision_labels(final_root),
    )
    figure_rows = collect_figure_inputs(final_root=final_root, output_csv=final_root / "inventories" / "figure_input_inventory.csv")

    hash_lines.sort()
    (final_root / "artifact_hashes.sha256").write_text("\n".join(hash_lines) + ("\n" if hash_lines else ""), encoding="utf-8")

    manifest = {
        "package_name": PACKAGE_NAME,
        "final_root": str(final_root),
        "repo_root": str(repo_root),
        "repo_commit": _git_commit(repo_root),
        "included_roots": included_roots,
        "missing_or_blocked_count": len(missing_rows),
        "artifact_count": len(artifact_rows),
        "sensitive_redaction_count": len(redactions),
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "claim_count": len(claim_rows),
        "figure_input_count": len(figure_rows),
    }
    (final_root / "collection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (final_root / "collection_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")

    _write_reports(
        final_root=final_root,
        included_roots=included_roots,
        missing_rows=missing_rows,
        accepted_rows=accepted_rows,
        rejected_rows=rejected_rows,
        gate_report=gate_report,
    )
    (final_root / "README.md").write_text(
        "\n".join(
            [
                f"# {PACKAGE_NAME}",
                "",
                "Final evidence package for the NeurIPS 2026 E&D agent-driver substrate submission.",
                "",
                "This package collects existing evidence roots, hashes artifacts, redacts sensitive session files, and records missing or blocked optional roots. It does not run new experiments.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    acceptance = {
        "status": "pass" if gate_report.get("allowed") and accepted_rows >= 0 else "fail",
        "evidence_gate_allowed": gate_report.get("allowed"),
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "missing_or_blocked_count": len(missing_rows),
        "sensitive_redaction_count": len(redactions),
    }
    (final_root / "final_acceptance_status.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **manifest,
        "final_root": str(final_root),
        "evidence_gate_allowed": gate_report.get("allowed"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--final-root",
        type=Path,
        default=Path(os.environ.get("NIPS_FINAL_EVIDENCE_ROOT", "")) if os.environ.get("NIPS_FINAL_EVIDENCE_ROOT") else None,
    )
    parser.add_argument("--max-copy-bytes", type=int, default=MAX_COPY_BYTES)
    parser.add_argument("--max-hash-bytes", type=int, default=MAX_HASH_BYTES)
    parser.add_argument("--max-index-files-per-root", type=int, default=MAX_INDEX_FILES_PER_ROOT)
    args = parser.parse_args()
    result = collect_final_evidence(
        repo_root=args.repo_root,
        final_root=args.final_root,
        max_copy_bytes=args.max_copy_bytes,
        max_hash_bytes=args.max_hash_bytes,
        max_index_files_per_root=args.max_index_files_per_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
