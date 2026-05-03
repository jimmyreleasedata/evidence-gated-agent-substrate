"""Shared helpers for the full NeurIPS evidence v2 collection pass."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from scripts.final.sanitize_sensitive_artifacts import is_sensitive_artifact


PACKAGE_NAME = "neurips2026_agent_driver_substrate_full_evidence_v2"
DEFAULT_FINAL_ROOT = Path("artifacts") / "final" / PACKAGE_NAME

TRUE_VALUES = {"1", "true", "yes", "y", "pass", "passed", "ok"}
FALSE_VALUES = {"0", "false", "no", "n", "fail", "failed"}
MAX_HASH_BYTES = 64 * 1024 * 1024
MAX_COPY_BYTES = 25 * 1024 * 1024

CATEGORY_DIRS = [
    "00_legacy_phase_runs",
    "01_evaluation_environment_runs",
    "02_real_backend_bootstrap",
    "03_real_upstream_assets",
    "04_real_task_cutover",
    "05_qwen_reference_agent",
    "06_model_backend_matrix",
    "07_reviewer_risk_fix",
    "08_framework_adapters",
    "09_final_patch_closeout",
    "10_release_metadata",
    "11_negative_blocker_evidence",
]

OUTPUT_DIRS = [
    "inventories",
    "reports",
    "final_tables",
    "final_figures",
    "logs",
    *[f"inputs_by_category/{category}" for category in CATEGORY_DIRS],
]

SCAN_ROOTS = [
    "artifacts/reports/phase7",
    "artifacts/reports/phase8",
    "artifacts/reports/phase4_swe_backend_contrast*",
    "artifacts/reports/phase4_phase5_swe_refresh_*",
    "artifacts/reports/phase5_swe",
    "artifacts/reports/phase5_webarena",
    "artifacts/reports/phase5_webarena_refresh*",
    "artifacts/reports/final*",
    "artifacts/reports/model_backend_matrix",
    "artifacts/reports/readiness",
    "artifacts/reports/qwen_agent_matrix",
    "artifacts/reports/framework_adapters",
    "artifacts/reports/reviewer_risk_fix",
    "artifacts/reports/regime_upgrade_*",
    "artifacts/reports/reversal_robustness_*",
    "artifacts/reports/second_model*",
    "artifacts/reports/model_scan*",
    "artifacts/reports/webarena_verified_*",
    "artifacts/reports/debug_swe*",
    "artifacts/reports/_invalid*",
    "artifacts/real_cutover",
    "artifacts/evaluation_environment",
    "artifacts/evaluation_environment_runs",
    "artifacts/evaluation_environment_preemptable",
    "artifacts/evaluation_environment_data_collection_debug",
    "artifacts/evaluation_environment_live_debug",
    "artifacts/phase7",
    "artifacts/phase7_probes",
    "artifacts/neurips_qwen_agent_matrix",
    "artifacts/neurips_final_data_expansion",
    "manifests",
    "metadata",
    "docs",
    "nips_upstream_assets",
    "vendor",
]

PRUNE_DIR_NAMES = {
    ".git",
    ".venv",
    ".venvs",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "model_cache",
    "cache",
    "wandb",
    "runtime",
    "workdir",
    "repo",
    "repo_snapshot",
    "repos",
}

EXPERIMENT_FILE_NAMES = {
    "run_manifest.json",
    "summary.json",
    "events.jsonl",
    "capture_summary.json",
    "capture_status.json",
    "replay_summary.json",
    "evaluator_output.json",
    "action_log.jsonl",
    "run_index.json",
    "evaluation_environment_run_inventory.csv",
    "evidence_gate_report.json",
    "final_claim_support_matrix.csv",
    "figure_input_inventory.csv",
}

EXPERIMENT_PATTERNS = (
    re.compile(r".*_summary\.csv$"),
    re.compile(r".*_summary\.md$"),
    re.compile(r"aggregate_.*\.csv$"),
    re.compile(r".*verifier.*\.log$"),
)

EVIDENCE_SCHEMA = [
    "source_root",
    "source_file",
    "source_category",
    "run_id",
    "episode_id",
    "row_id",
    "timestamp",
    "pbs_job_id",
    "git_commit_if_available",
    "family",
    "task_id",
    "instance_id",
    "task_source",
    "upstream_repo",
    "upstream_commit",
    "dataset_version",
    "task_manifest_hash",
    "implementation_source",
    "evidence_class",
    "replay_class",
    "execution_mode",
    "regime",
    "telemetry_mode",
    "concurrency",
    "seed",
    "budget",
    "driver_id",
    "driver_type",
    "driver_version",
    "policy_version",
    "prompt_template_hash",
    "action_parser_version",
    "traffic_source_name",
    "model_family",
    "model_id",
    "model_size",
    "model_path_or_hf_id",
    "model_revision",
    "model_path_hash",
    "tokenizer_hash",
    "backend_engine",
    "backend_version",
    "tensor_parallel_size",
    "unsupported_backend_pair",
    "model_latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "invalid_action",
    "invalid_action_rate",
    "num_env_steps",
    "browser_env_latency_ms",
    "evaluator_latency_ms",
    "env_step_latency_ms",
    "throughput_episodes_per_s",
    "telemetry_overhead_pct",
    "patch_generation_latency_ms",
    "patch_apply_success",
    "build_ms",
    "test_ms",
    "verifier_service_ms",
    "verifier_queue_wait_ms",
    "end_to_end_p50_ms",
    "end_to_end_p95_ms",
    "end_to_end_p99_ms",
    "queue_wait_share_p99",
    "pass_rate",
    "failure_rate",
    "retry_rate",
    "task_success",
    "reward",
    "reward_auc_over_wallclock",
    "trace_completeness",
    "replay_validity",
    "evidence_validation_pass",
    "trace_hash",
    "artifact_hash",
    "eval_hash",
    "evaluator_output_hash",
    "evaluator_version",
    "browser_version",
    "browser_backend",
    "repo",
    "base_commit",
    "harness_version",
    "image_or_sif_hash",
    "outcome_class",
    "failure_category",
    "failure_taxonomy",
    "dominant_failure_category",
    "natural_failure_share",
    "controlled_failure_share",
    "measurable_queueing",
    "significant_queueing",
    "dominant_queueing",
    "natural_failure_diversity_supported",
    "p99_increase_supported",
    "component_shift_supported",
    "behavior_shift_supported",
    "controller_reversal_supported",
    "operating_point_shift_supported",
    "decision_label",
    "decision_labels",
    "support_label",
    "paper_facing_allowed",
    "paper_role",
    "smoke_only",
    "fixture_only",
    "exclusion_reason",
    "support_status",
    "logical_cell_key",
    "artifact_row_hash",
]


@dataclass(frozen=True)
class ExpectedRoot:
    root_id: str
    expected_path: str
    category: str
    expected_files: tuple[str, ...] = ()
    paper_role: str = "appendix_only"
    required: bool = False
    notes: str = ""


def expected_roots() -> list[ExpectedRoot]:
    roots = [
        ExpectedRoot("status_board", "docs/status_board.md", "00_legacy_phase_runs", paper_role="metadata_only"),
        ExpectedRoot("execution_log", "docs/execution_log.md", "00_legacy_phase_runs", paper_role="metadata_only"),
        ExpectedRoot("acceptance_report", "docs/acceptance_report.md", "00_legacy_phase_runs", paper_role="metadata_only"),
        ExpectedRoot("dataset_card", "docs/dataset_card.md", "00_legacy_phase_runs", paper_role="metadata_only"),
        ExpectedRoot("release_manifest", "release_manifest.yaml", "00_legacy_phase_runs", paper_role="metadata_only"),
        ExpectedRoot("croissant", "metadata/croissant.json", "00_legacy_phase_runs", paper_role="metadata_only"),
        ExpectedRoot("phase7_summary", "artifacts/reports/phase7/summary.md", "00_legacy_phase_runs", paper_role="metadata_only"),
        ExpectedRoot("phase7_inputs", "artifacts/reports/phase7/inputs", "00_legacy_phase_runs", paper_role="appendix_only"),
        ExpectedRoot("phase7_tables", "artifacts/reports/phase7/tables", "00_legacy_phase_runs", paper_role="appendix_only"),
        ExpectedRoot("phase7_plots", "artifacts/reports/phase7/plots", "00_legacy_phase_runs", paper_role="appendix_only"),
        ExpectedRoot("phase8_summary", "artifacts/reports/phase8/summary.md", "00_legacy_phase_runs", paper_role="metadata_only"),
        ExpectedRoot("phase8_acceptance_delta", "artifacts/reports/phase8/acceptance_delta.md", "00_legacy_phase_runs", paper_role="metadata_only"),
        ExpectedRoot("phase8_system_variant_csv", "artifacts/reports/phase8/system_variant_matrix.csv", "00_legacy_phase_runs", paper_role="appendix_only"),
        ExpectedRoot("phase8_system_variant_parquet", "artifacts/reports/phase8/system_variant_matrix.parquet", "00_legacy_phase_runs", paper_role="appendix_only"),
        ExpectedRoot("phase8_plots", "artifacts/reports/phase8/plots", "00_legacy_phase_runs", paper_role="appendix_only"),
        ExpectedRoot("evaluation_environment_webarena_replay_small", "artifacts/evaluation_environment/webarena_verified/replay_small", "01_evaluation_environment_runs", paper_role="appendix_only", expected_files=("run_manifest.json", "summary.json")),
        ExpectedRoot("evaluation_environment_swe_slice_small", "artifacts/evaluation_environment/swe_gym/slice_small", "01_evaluation_environment_runs", paper_role="appendix_only", expected_files=("run_manifest.json", "summary.json")),
        ExpectedRoot("evaluation_environment_queue_coupling", "artifacts/evaluation_environment_preemptable/20260329_195955Z/queue_coupling", "01_evaluation_environment_runs", paper_role="appendix_only"),
        ExpectedRoot("evaluation_environment_verifier_tail", "artifacts/evaluation_environment_preemptable/20260329_195955Z/verifier_tail", "01_evaluation_environment_runs", paper_role="appendix_only"),
        ExpectedRoot("evaluation_environment_runs_root", "artifacts/evaluation_environment_runs", "01_evaluation_environment_runs", paper_role="appendix_only"),
        ExpectedRoot("evaluation_environment_round1_manifest", "artifacts/evaluation_environment_runs/20260330_system_variant_round1_submission_manifest.json", "01_evaluation_environment_runs", paper_role="metadata_only"),
        ExpectedRoot("evaluation_environment_round1_web_sync", "artifacts/evaluation_environment_runs/20260330_system_variant_round1_web_sync", "01_evaluation_environment_runs", paper_role="appendix_only"),
        ExpectedRoot("evaluation_environment_round1_web_naive_async", "artifacts/evaluation_environment_runs/20260330_system_variant_round1_web_naive_async", "01_evaluation_environment_runs", paper_role="appendix_only"),
        ExpectedRoot("evaluation_environment_round1_web_full_system", "artifacts/evaluation_environment_runs/20260330_system_variant_round1_web_full_system", "01_evaluation_environment_runs", paper_role="appendix_only"),
        ExpectedRoot("evaluation_environment_round1_queue_coupling", "artifacts/evaluation_environment_runs/20260330_system_variant_round1_queue_coupling", "01_evaluation_environment_runs", paper_role="appendix_only"),
        ExpectedRoot("evaluation_environment_miniwob_telemetry", "artifacts/evaluation_environment_runs/20260330_neurips_evidence_miniwob/telemetry", "01_evaluation_environment_runs", paper_role="appendix_only"),
        ExpectedRoot("evaluation_environment_miniwob_concurrency", "artifacts/evaluation_environment_runs/20260330_neurips_evidence_miniwob/concurrency", "01_evaluation_environment_runs", paper_role="appendix_only"),
        ExpectedRoot("bootstrap_real_backends", "scripts/bootstrap_real_backends.sh", "02_real_backend_bootstrap", paper_role="metadata_only"),
        ExpectedRoot("validate_real_backends", "scripts/validate_real_backends.sh", "02_real_backend_bootstrap", paper_role="metadata_only"),
        ExpectedRoot("external_paths", "runtime/external_paths.py", "02_real_backend_bootstrap", paper_role="metadata_only"),
        ExpectedRoot("repo_probe", "runtime/repo_probe.py", "02_real_backend_bootstrap", paper_role="metadata_only"),
        ExpectedRoot("real_backend_validation", "runtime/real_backend_validation.py", "02_real_backend_bootstrap", paper_role="metadata_only"),
        ExpectedRoot("vendor_browsergym", "vendor/browsergym", "02_real_backend_bootstrap", paper_role="metadata_only"),
        ExpectedRoot("vendor_swe_bench", "vendor/SWE-bench", "02_real_backend_bootstrap", paper_role="metadata_only"),
        ExpectedRoot("local_paths_manifest", "manifests/local_paths.yaml", "02_real_backend_bootstrap", paper_role="metadata_only"),
        ExpectedRoot("real_backend_validation_summary", "artifacts/real_backends/validation_summary.json", "02_real_backend_bootstrap", paper_role="metadata_only"),
        ExpectedRoot("testing_real_backends_doc", "docs/testing_real_backends.md", "02_real_backend_bootstrap", paper_role="metadata_only"),
        ExpectedRoot("upstream_exports", "nips_upstream_assets/exports.sh", "03_real_upstream_assets", paper_role="metadata_only"),
        ExpectedRoot("upstream_webarena", "nips_upstream_assets/webarena_verified", "03_real_upstream_assets", paper_role="metadata_only"),
        ExpectedRoot("upstream_webarena_REDACTED_BROWSER_STATE_LABEL_hash_only", "nips_upstream_assets/webarena_verified/gcp_live/webarena_REDACTED_BROWSER_STATE_LABEL.json", "03_real_upstream_assets", paper_role="metadata_only", notes="sensitive hash-only storage state"),
        ExpectedRoot("upstream_swebench", "nips_upstream_assets/swebench", "03_real_upstream_assets", paper_role="metadata_only"),
        ExpectedRoot("upstream_miniwob", "nips_upstream_assets/miniwob", "03_real_upstream_assets", paper_role="metadata_only"),
        ExpectedRoot("upstream_assets_manifest", "manifests/upstream_assets.yaml", "03_real_upstream_assets", paper_role="metadata_only"),
        ExpectedRoot("upstream_asset_preflight", "artifacts/real_upstream_asset_preflight.json", "03_real_upstream_assets", paper_role="metadata_only"),
        ExpectedRoot("real_upstream_runbook", "docs/runbooks/real_upstream_assets.md", "03_real_upstream_assets", paper_role="metadata_only"),
        ExpectedRoot("real_cutover_swe_first_pass", "artifacts/real_cutover/swe_final_first_pass_20260421_031400Z", "04_real_task_cutover", paper_role="paper_facing"),
        ExpectedRoot("real_cutover_swe_stressed", "artifacts/real_cutover/ANON_JOB_ID_HASH_4ffce44deddc", "04_real_task_cutover", paper_role="paper_facing"),
        ExpectedRoot("real_cutover_miniwob_telemetry", "artifacts/real_cutover/miniwob_real_telemetry_20260421_003028Z", "04_real_task_cutover", paper_role="paper_facing"),
        ExpectedRoot("real_cutover_miniwob_concurrency", "artifacts/real_cutover/miniwob_real_concurrency_20260421_003948Z", "04_real_task_cutover", paper_role="paper_facing"),
        ExpectedRoot("real_cutover_webarena_live_11", "artifacts/real_cutover/webarena_live_capture_11_20260421_054351Z", "04_real_task_cutover", paper_role="paper_facing"),
        ExpectedRoot("real_cutover_webarena_replay_11", "artifacts/real_cutover/webarena_replay_11_20260421_054524Z", "04_real_task_cutover", paper_role="paper_facing"),
        ExpectedRoot("real_cutover_webarena_controller_pilot", "artifacts/real_cutover/webarena_controller_pilot_10task_20260421_220038Z", "04_real_task_cutover", paper_role="calibration_control"),
        ExpectedRoot("qwen_agent_reports", "artifacts/reports/qwen_agent_matrix", "05_qwen_reference_agent", paper_role="paper_facing"),
        ExpectedRoot("qwen_agent_runs", "artifacts/neurips_qwen_agent_matrix", "05_qwen_reference_agent", paper_role="paper_facing"),
        ExpectedRoot("qwen_agent_cutover", "artifacts/real_cutover/qwen_agent_matrix", "05_qwen_reference_agent", paper_role="paper_facing"),
        ExpectedRoot("readiness_reports", "artifacts/reports/readiness", "06_model_backend_matrix", paper_role="metadata_only"),
        ExpectedRoot("model_backend_matrix", "artifacts/reports/model_backend_matrix", "06_model_backend_matrix", paper_role="paper_facing"),
        ExpectedRoot("model_backend_inventory_csv", "artifacts/reports/model_backend_matrix/model_backend_inventory.csv", "06_model_backend_matrix", paper_role="paper_facing"),
        ExpectedRoot("model_backend_inventory_md", "artifacts/reports/model_backend_matrix/model_backend_inventory.md", "06_model_backend_matrix", paper_role="paper_facing"),
        ExpectedRoot("all_model_smoke", "artifacts/reports/model_backend_matrix/all_model_smoke", "06_model_backend_matrix", paper_role="smoke_only"),
        ExpectedRoot("webarena_model_backend", "artifacts/reports/model_backend_matrix/webarena", "06_model_backend_matrix", paper_role="paper_facing"),
        ExpectedRoot("miniwob_model_backend", "artifacts/reports/model_backend_matrix/miniwob", "06_model_backend_matrix", paper_role="paper_facing"),
        ExpectedRoot("swe_model_backend", "artifacts/reports/model_backend_matrix/swe", "06_model_backend_matrix", paper_role="paper_facing"),
        ExpectedRoot("backend_engine_contrast_csv", "artifacts/reports/model_backend_matrix/backend_engine_contrast.csv", "06_model_backend_matrix", paper_role="paper_facing"),
        ExpectedRoot("backend_engine_contrast_md", "artifacts/reports/model_backend_matrix/backend_engine_contrast.md", "06_model_backend_matrix", paper_role="paper_facing"),
        ExpectedRoot("local_and_hf_models", "manifests/models/local_and_hf_models.yaml", "06_model_backend_matrix", paper_role="metadata_only"),
        ExpectedRoot("model_backend_pairs", "manifests/models/model_backend_pairs.yaml", "06_model_backend_matrix", paper_role="metadata_only"),
        ExpectedRoot("reviewer_risk_fix", "artifacts/reports/reviewer_risk_fix", "07_reviewer_risk_fix", paper_role="paper_facing"),
        ExpectedRoot("swe_queue_saturation", "artifacts/reports/reviewer_risk_fix/swe_queue_saturation", "07_reviewer_risk_fix", paper_role="paper_facing"),
        ExpectedRoot("swe_failure_diversity", "artifacts/reports/reviewer_risk_fix/swe_failure_diversity", "07_reviewer_risk_fix", paper_role="paper_facing"),
        ExpectedRoot("webarena_paired_stress", "artifacts/reports/reviewer_risk_fix/webarena_paired_stress", "07_reviewer_risk_fix", paper_role="paper_facing"),
        ExpectedRoot("framework_adapters", "artifacts/reports/framework_adapters", "08_framework_adapters", paper_role="appendix_only"),
        ExpectedRoot("verl_adapter_summary_csv", "artifacts/reports/framework_adapters/verl_adapter_summary.csv", "08_framework_adapters", paper_role="appendix_only"),
        ExpectedRoot("trl_adapter_summary_csv", "artifacts/reports/framework_adapters/trl_adapter_summary.csv", "08_framework_adapters", paper_role="appendix_only"),
        ExpectedRoot("framework_adapter_manifest", "manifests/frameworks/framework_adapters.yaml", "08_framework_adapters", paper_role="metadata_only"),
        ExpectedRoot("final_neurips_agent_driver_matrix", "artifacts/reports/final_neurips_agent_driver_matrix", "09_final_patch_closeout", paper_role="paper_facing"),
        ExpectedRoot("final_v1_package", "artifacts/final/neurips2026_agent_driver_substrate_final_v1", "09_final_patch_closeout", paper_role="paper_facing"),
        ExpectedRoot("final_v2_package_existing", "artifacts/final/neurips2026_agent_driver_substrate_full_evidence_v2", "09_final_patch_closeout", paper_role="metadata_only"),
        ExpectedRoot("webarena_two_node_probe", "artifacts/real_cutover/webarena_two_node_probe", "11_negative_blocker_evidence", paper_role="negative_blocker"),
        ExpectedRoot("webarena_gcp_live_manual_preflight", "artifacts/real_cutover/webarena_gcp_live_manual_preflight_20260421_044046Z", "11_negative_blocker_evidence", paper_role="negative_blocker"),
        ExpectedRoot("webarena_gcp_live_manual_run", "artifacts/real_cutover/webarena_gcp_live_manual_run_20260421_044112Z", "11_negative_blocker_evidence", paper_role="negative_blocker"),
    ]
    return roots


def repo_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def resolve_final_root(repo_root: Path, final_root: Path | None = None) -> Path:
    if final_root is not None:
        return final_root.resolve()
    env = os.environ.get("NIPS_FULL_EVIDENCE_ROOT")
    if env:
        return Path(env).resolve()
    return (repo_root / DEFAULT_FINAL_ROOT).resolve()


def ensure_output_dirs(final_root: Path) -> None:
    for relative in OUTPUT_DIRS:
        (final_root / relative).mkdir(parents=True, exist_ok=True)


def safe_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        return f"external/{digest}/{path.name}"


def sha256_file(path: Path, max_bytes: int = MAX_HASH_BYTES) -> tuple[str, str]:
    size = path.stat().st_size
    if size > max_bytes:
        return f"deferred:size_bytes={size}", "hash_deferred_large_artifact"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), "hashed"


def row_hash(row: dict[str, object]) -> str:
    payload = json.dumps(row, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUE_VALUES


def falsy(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in FALSE_VALUES


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_parquet_if_possible(csv_path: Path, rows: list[dict[str, object]]) -> None:
    parquet_path = csv_path.with_suffix(".parquet")
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover - depends on optional pyarrow/fastparquet.
        write_json(
            parquet_path.with_suffix(".parquet.unavailable.json"),
            {"reason": exc.__class__.__name__, "message": str(exc), "csv_fallback": str(csv_path)},
        )


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def is_experiment_filename(name: str) -> bool:
    return name in EXPERIMENT_FILE_NAMES or any(pattern.match(name) for pattern in EXPERIMENT_PATTERNS)


def infer_category(path: Path) -> str:
    text = str(path).lower()
    if "webarena_two_node_probe" in text or "gcp_live_manual" in text or "blocker" in text:
        return "11_negative_blocker_evidence"
    if "framework_adapter" in text or "/framework_adapters" in text:
        return "08_framework_adapters"
    if "reviewer_risk_fix" in text or "phase5" in text or "paired_stress" in text or "swe_queue" in text or "failure_diversity" in text:
        return "07_reviewer_risk_fix"
    if "model_backend_matrix" in text or "readiness" in text:
        return "06_model_backend_matrix"
    if "qwen_agent" in text:
        return "05_qwen_reference_agent"
    if "real_cutover" in text:
        return "04_real_task_cutover"
    if "nips_upstream_assets" in text or "upstream_assets" in text:
        return "03_real_upstream_assets"
    if "real_backends" in text or "bootstrap_real_backends" in text or "validate_real_backends" in text:
        return "02_real_backend_bootstrap"
    if "evaluation_environment" in text:
        return "01_evaluation_environment_runs"
    if "final_neurips" in text or "final_paper_data" in text or "neurips2026_agent_driver_substrate_final" in text:
        return "09_final_patch_closeout"
    if "release_manifest" in text or "dataset_card" in text or "croissant" in text or "acceptance_report" in text:
        return "10_release_metadata"
    return "00_legacy_phase_runs"


def infer_family(path: Path, row: dict[str, object] | None = None) -> str:
    row = row or {}
    for key in ("family", "workload_family", "benchmark_family"):
        value = str(row.get(key, "")).strip()
        if value:
            return normalize_family(value)
    text = str(path).lower()
    if "webarena" in text:
        return "webarena_verified"
    if "miniwob" in text:
        return "miniwob"
    if "swe" in text:
        return "swe_gym"
    return ""


def normalize_family(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"webarena", "webarena_verified", "webarena verified"}:
        return "webarena_verified"
    if lowered in {"swe", "swe_gym", "swe-gym", "swebench", "swe-bench"}:
        return "swe_gym"
    if lowered in {"miniwob", "miniwob++", "miniwob_plus"}:
        return "miniwob"
    return value.strip()


def infer_experiment_type(path: Path) -> str:
    text = str(path).lower()
    if "queue" in text:
        return "swe_queue_saturation"
    if "failure" in text:
        return "swe_failure_diversity"
    if "paired_stress" in text or "stress" in text:
        return "webarena_paired_stress"
    if "backend_engine_contrast" in text:
        return "backend_engine_contrast"
    if "model_backend" in text:
        return "model_backend_matrix"
    if "qwen_agent" in text:
        return "qwen_reference_agent"
    if "framework_adapter" in text:
        return "framework_adapter"
    if "real_cutover" in text:
        return "real_task_cutover"
    return "experiment_root"


def infer_date(path: Path) -> str:
    match = re.search(r"(20\d{6}(?:_\d{6}Z?)?)", str(path))
    return match.group(1) if match else ""


def infer_pbs_job_id(path: Path) -> str:
    match = re.search(r"(\d{6,8}\.evaluation_environment[^/]+)", str(path))
    return match.group(1) if match else ""


def has_metadata_file(root: Path, names: Iterable[str]) -> bool:
    return any((root / name).exists() for name in names)


def classify_evidence(row: dict[str, object], source_category: str) -> str:
    implementation = str(row.get("implementation_source", "")).lower()
    driver_type = str(row.get("driver_type", "")).lower()
    paper_role = str(row.get("paper_role", "")).lower()
    text = " ".join(str(row.get(key, "")) for key in ("source_root", "source_file", "regime", "evidence_class")).lower()
    if source_category == "11_negative_blocker_evidence" or "blocker" in text or "portmap" in implementation:
        return "negative_blocker_evidence"
    if paper_role == "smoke_only" or truthy(row.get("smoke_only")) or "smoke" in text:
        return "smoke_only"
    if source_category == "08_framework_adapters":
        return "framework_adapter"
    if source_category == "07_reviewer_risk_fix" or "queue" in text or "failure_diversity" in text or "paired_stress" in text:
        return "reviewer_risk_fix"
    if source_category == "06_model_backend_matrix" and "backend" in text:
        return "model_backend_contrast"
    if driver_type == "llm_agent" or source_category in {"05_qwen_reference_agent", "06_model_backend_matrix"}:
        return "llm_agent_core"
    if driver_type in {"oracle_control", "negative_control", "scripted_reference", "controller"}:
        return "calibration_control"
    if source_category == "04_real_task_cutover":
        return "real_task_cutover"
    if source_category == "10_release_metadata":
        return "release_metadata"
    return "legacy_baseline"


def model_family_from_model_id(model_id: str) -> str:
    text = model_id.lower()
    if "qwen" in text:
        return "qwen"
    if "llama" in text:
        return "llama"
    if "gemma" in text:
        return "gemma"
    if "ministral" in text or "mistral" in text:
        return "mistral"
    if "minimax" in text:
        return "minimax"
    return ""


def model_size_from_model_id(model_id: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)[bB]", model_id)
    return f"{match.group(1)}B" if match else ""


def normalize_row(raw: dict[str, object], *, source_root: Path, source_file: Path, source_category: str, index: int) -> dict[str, object]:
    row = {key: "" for key in EVIDENCE_SCHEMA}
    raw_str = {str(key): "" if value is None else str(value) for key, value in raw.items()}
    aliases = {
        "backend": "backend_engine",
        "model_backend": "backend_engine",
        "workload_family": "family",
        "benchmark_family": "family",
        "admitted": "evidence_validation_pass",
        "success": "task_success",
        "repo_name": "repo",
        "image_hash": "image_or_sif_hash",
        "sif_hash": "image_or_sif_hash",
        "trace_artifact_hash": "trace_hash",
    }
    for key, value in raw_str.items():
        dest = aliases.get(key, key)
        if dest in row:
            row[dest] = value
    row["source_root"] = str(source_root)
    row["source_file"] = str(source_file)
    row["source_category"] = source_category
    row["family"] = normalize_family(str(row["family"] or infer_family(source_file, raw_str)))
    if not row["run_id"]:
        row["run_id"] = str(raw_str.get("run_id") or source_root.name)
    row["row_id"] = str(raw_str.get("row_id") or f"{source_file.name}:{index}")
    if not row["model_family"] and row["model_id"]:
        row["model_family"] = model_family_from_model_id(str(row["model_id"]))
    if not row["model_size"] and row["model_id"]:
        row["model_size"] = model_size_from_model_id(str(row["model_id"]))
    if not row["evidence_class"]:
        row["evidence_class"] = classify_evidence(row, source_category)
    if not row["paper_role"]:
        if source_category == "11_negative_blocker_evidence":
            row["paper_role"] = "negative_blocker"
        elif row["evidence_class"] == "smoke_only":
            row["paper_role"] = "smoke_only"
        elif row["evidence_class"] in {"release_metadata", "legacy_baseline", "framework_adapter"}:
            row["paper_role"] = "appendix_only"
        else:
            row["paper_role"] = "paper_facing"
    if not row["paper_facing_allowed"]:
        row["paper_facing_allowed"] = "true" if row["paper_role"] == "paper_facing" else "false"
    if not row["implementation_source"]:
        if row["source_category"] == "04_real_task_cutover":
            row["implementation_source"] = "real_upstream"
        elif row["source_category"] == "11_negative_blocker_evidence":
            row["implementation_source"] = "negative_blocker"
    row["artifact_row_hash"] = row_hash(raw_str)
    row["logical_cell_key"] = "|".join(
        str(row.get(key, ""))
        for key in ("family", "task_id", "instance_id", "driver_id", "model_id", "backend_engine", "regime", "seed")
    )
    return row


def copy_or_redact_artifact(
    *,
    file_path: Path,
    repo_root: Path,
    final_root: Path,
    category: str,
    digest: str,
    max_copy_bytes: int = MAX_COPY_BYTES,
) -> tuple[str, dict[str, object] | None]:
    rel = safe_relative(file_path, repo_root)
    if is_sensitive_artifact(file_path):
        return "redacted_hash_only", {
            "source_path": str(file_path),
            "relative_path": rel,
            "sha256": digest,
            "size_bytes": file_path.stat().st_size,
            "redaction_status": "redacted_required_runtime_secret",
        }
    if file_path.stat().st_size > max_copy_bytes:
        return "hash_only_large_artifact", None
    dest = final_root / "inputs_by_category" / category / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dest)
    return "copied", None


def iter_files_limited(root: Path, *, skip_root: Path | None = None, max_files: int | None = None) -> Iterable[Path]:
    count = 0
    if root.is_file():
        yield root
        return
    skip_resolved = skip_root.resolve() if skip_root else None
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        if skip_resolved and current_path.resolve() == skip_resolved:
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if name not in PRUNE_DIR_NAMES]
        for name in files:
            yield current_path / name
            count += 1
            if max_files is not None and count >= max_files:
                return
