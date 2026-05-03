"""Shared helpers for the NeurIPS E&D canonical final rerun."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


PACKAGE_NAME = "neurips2026_ed_canonical_rerun_v1"
ROUND_DIRS = [
    ("ROUND1_REAL_TASK_CUTOVER", "round1_real_task_cutover"),
    ("ROUND2_QWEN_FAMILY_DRIVER", "round2_qwen_family_driver"),
    ("closeout_EXTENSION_RISKFIX", "closeout_extension_riskfix"),
]
TRUE_VALUES = {"1", "true", "yes", "y", "pass", "passed", "ok", "supported"}
FALSE_VALUES = {"0", "false", "no", "n", "fail", "failed", "blocked", "unsupported"}
LLM_REQUIRED_FIELDS = [
    "driver_id",
    "driver_type",
    "driver_version",
    "model_family",
    "model_id",
    "model_backend",
    "backend_engine",
    "policy_version",
    "prompt_template_hash",
    "action_parser_version",
    "budget",
    "implementation_source",
    "evidence_validation_pass",
]
COMPONENT_FIELDS = [
    "dom_get_ms",
    "screenshot_ms",
    "network_or_har_ms",
    "render_or_page_wait_ms",
    "navigation_ms",
]
BASE_FIELD_ORDER = [
    "source_root",
    "source_file",
    "round_id",
    "family",
    "task_id",
    "instance_id",
    "implementation_source",
    "paper_facing_allowed",
    "paper_role",
    "evidence_class",
    "replay_class",
    "driver_id",
    "driver_type",
    "driver_version",
    "model_family",
    "model_id",
    "model_size",
    "model_backend",
    "backend_engine",
    "policy_version",
    "prompt_template_hash",
    "action_parser_version",
    "budget",
    "seed",
    "evidence_validation_pass",
    "trace_hash",
    "artifact_hash",
    "eval_hash",
    "evaluator_output_hash",
    "evaluator_version",
    "real_upstream_live",
    "real_upstream_replay",
    "browser_backend",
    "browser_version",
    "upstream_package_version",
    "repo",
    "base_commit",
    "harness_version",
    "image_or_sif_hash",
    "smoke_only",
    "fixture_only",
    "unsupported_backend_pair",
    "regime",
    "decision_label",
    "decision_labels",
    "measurable_queueing",
    "significant_queueing",
    "dominant_queueing",
    "natural_failure_diversity_supported",
    "dominant_failure_category",
    "dominant_failure_share",
    "p99_increase_supported",
    "component_shift_supported",
    "behavior_shift_supported",
    "controller_reversal_supported",
    "operating_point_shift_supported",
    "end_to_end_p99_ms",
    *COMPONENT_FIELDS,
    "exclusion_reason",
]


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason: str = ""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_root() -> Path:
    return Path("artifact_release_root/artifacts/final") / PACKAGE_NAME


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def falsy(value: object) -> bool:
    return str(value or "").strip().lower() in FALSE_VALUES


def blank_or_na(value: object) -> bool:
    return str(value or "").strip().lower() in {"", "na", "n/a", "none", "null", "nan"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def fieldnames_for(rows: Iterable[Mapping[str, object]], preferred: list[str] | None = None) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for field in preferred or []:
        if field not in seen:
            fields.append(field)
            seen.add(field)
    for row in rows:
        for field in row.keys():
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def write_csv(path: Path, rows: list[Mapping[str, object]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    names = fieldnames_for(rows, fieldnames or BASE_FIELD_ORDER)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name, "")) for name in names})


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_md(path: Path, lines: Iterable[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_command(args: list[str], timeout_s: int = 20) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
        status = "pass" if completed.returncode == 0 else "fail"
        detail = (completed.stdout or completed.stderr or "").strip().splitlines()
        return status, detail[0] if detail else f"exit_code={completed.returncode}"
    except FileNotFoundError:
        return "blocked", f"not_found: {args[0]}"
    except subprocess.TimeoutExpired:
        return "blocked", f"timeout_s={timeout_s}: {' '.join(args)}"


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def has_any(row: Mapping[str, object], keys: Iterable[str]) -> bool:
    return any(not blank_or_na(row.get(key, "")) for key in keys)


def count_by(rows: list[Mapping[str, object]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key, "") or "unknown") for row in rows)
    return dict(sorted(counter.items()))


def canonical_gate_reason(row: Mapping[str, object], root: Path) -> str:
    source_root = str(row.get("source_root", "")).strip()
    if source_root and not is_under(Path(source_root), root):
        return "source_outside_canonical_root"
    if truthy(row.get("exclude_from_paper_surface")):
        return "explicitly_excluded_from_paper_surface"
    if str(row.get("paper_role", "")).strip().lower() == "preflight_only" or str(row.get("evidence_class", "")).strip().lower() == "preflight":
        return "preflight_only_not_paper_facing"
    if not truthy(row.get("paper_facing_allowed")):
        return "not_marked_paper_facing"
    if truthy(row.get("smoke_only")):
        return "smoke_only_not_paper_facing"
    if truthy(row.get("unsupported_backend_pair")):
        return "unsupported_backend_pair_not_success"
    implementation = str(row.get("implementation_source", "")).strip().lower()
    if implementation == "mock_fixture" or truthy(row.get("fixture_only")):
        return "mock_fixture_paper_facing"
    if implementation == "synthetic_executable":
        return "synthetic_executable_paper_facing"
    if "expected_slug" in implementation or str(row.get("evaluator_version", "")).strip().lower() == "expected_slug":
        return "expected_slug_mock_evaluator"
    if str(row.get("driver_type", "")).strip() == "llm_agent":
        missing = [field for field in LLM_REQUIRED_FIELDS if blank_or_na(row.get(field, ""))]
        if missing:
            return "missing_llm_driver_metadata"
    if row.get("evidence_validation_pass") and falsy(row.get("evidence_validation_pass")):
        return "evidence_validation_failed"

    family = str(row.get("family", "")).strip().lower()
    if family == "webarena_verified":
        has_real_live = "real_upstream_live" in implementation or truthy(row.get("real_upstream_live"))
        has_real_replay = "real_upstream_replay" in implementation or truthy(row.get("real_upstream_replay"))
        if not (has_real_live or has_real_replay):
            return "missing_webarena_real_upstream_evidence"
        if not has_any(row, ["trace_hash", "artifact_hash", "eval_hash", "evaluator_output_hash"]):
            return "missing_webarena_trace_or_eval_hash"
        if blank_or_na(row.get("evaluator_version", "")):
            return "missing_webarena_evaluator_version"
    if family in {"swe", "swe_gym", "swe-bench", "swebench"}:
        required = ["instance_id", "repo", "base_commit", "harness_version", "image_or_sif_hash"]
        if any(blank_or_na(row.get(field, "")) for field in required):
            return "missing_swe_real_metadata"
    if family in {"miniwob", "miniwob++", "miniwob_plus"}:
        if "mockminiwobenv" in implementation:
            return "mock_miniwob_env_paper_facing"
        if not ("browser" in implementation or has_any(row, ["browser_backend", "browser_version", "upstream_package_version"])):
            return "missing_miniwob_browser_backed_evidence"

    stress_marker = str(row.get("evidence_class", "")).strip().lower() == "reviewer_risk_fix" or "stress" in str(
        row.get("regime", "")
    ).lower()
    if stress_marker and not has_any(
        row,
        [
            "decision_label",
            "decision_labels",
            "measurable_queueing",
            "significant_queueing",
            "dominant_queueing",
            "natural_failure_diversity_supported",
            "p99_increase_supported",
            "component_shift_supported",
            "behavior_shift_supported",
            "controller_reversal_supported",
            "operating_point_shift_supported",
        ],
    ):
        return "stress_claim_missing_decision_label"
    return ""


def import_status(module_name: str) -> tuple[str, str]:
    code = "\n".join(
        [
            "import importlib.metadata as metadata",
            "import importlib.util",
            f"module_name = {module_name!r}",
            "spec = importlib.util.find_spec(module_name)",
            "if spec is None:",
            "    raise SystemExit(1)",
            "try:",
            "    print(metadata.version(module_name))",
            "except metadata.PackageNotFoundError:",
            "    print('installed')",
        ]
    )
    return run_command([sys.executable, "-c", code], timeout_s=15)


def executable_status(binary: str, args: list[str] | None = None, timeout_s: int = 15) -> tuple[str, str]:
    resolved = shutil.which(binary)
    if not resolved:
        return "blocked", f"{binary} not found"
    return run_command([resolved, *(args or ["--version"])], timeout_s=timeout_s)
