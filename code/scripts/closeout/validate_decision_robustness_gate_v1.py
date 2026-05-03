#!/usr/bin/env python3
"""Validate decision-robustness closeout rows through a separate evidence gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


CLOSEOUT_NAME = "decision_robustness_closeout_v1"
WORKLOAD_SLICE = "webarena_verified_decision_slice"
CONTROLLERS = {"hook_a_only", "hook_b_only"}
BACKENDS = {"vllm", "sglang"}
REGIMES = {"clean_baseline", "medium_live_stressed"}
BUDGETS = {5, 7, 9}
SEEDS_DEFAULT = {0, 1}
SENSITIVE_PATTERNS = ("REDACTED_BROWSER_STATE_LABEL", "REDACTED_BROWSER_STATE_LABEL", "headers", "hf_token", "token", "live-session artifact")
REQUIRED_FIELDS = [
    "run_id",
    "release_root",
    "closeout_root",
    "workload_family",
    "workload_slice",
    "task_id",
    "regime",
    "stress_config_id",
    "backend_engine",
    "controller_id",
    "seed",
    "budget",
    "driver_id",
    "driver_type",
    "driver_version",
    "traffic_source_id",
    "traffic_source_type",
    "model_family",
    "model_id",
    "model_backend",
    "policy_version",
    "prompt_template_hash",
    "action_parser_version",
    "implementation_source",
    "terminal_outcome",
    "reward",
    "reward_auc_over_wall_clock_input_path",
    "pass_rate_input_path",
    "evidence_validation_pass",
    "trace_id",
    "manifest_hash",
    "artifact_root",
    "paper_role",
    "exclude_from_canonical_surface",
    "sensitive_hash_only",
]


def _closeout_root(root: Path) -> Path:
    return root / CLOSEOUT_NAME


def _blank(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "ok"}


def _normalize_backend(value: object) -> str:
    text = str(value or "").strip().lower()
    if "sglang" in text:
        return "sglang"
    if "vllm" in text:
        return "vllm"
    return text


def _normalize_regime(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"clean", "clean_baseline", "baseline"} or "clean" in text:
        return "clean_baseline"
    if text in {"medium", "medium_live_stressed"} or "medium" in text or "stress" in text:
        return "medium_live_stressed"
    return text


def _normalize_controller(value: object) -> str:
    text = str(value or "").strip().lower()
    for controller in CONTROLLERS:
        if controller in text:
            return controller
    return text


def _same_rootish(candidate: object, root: Path, closeout_root: Path) -> bool:
    if _blank(candidate):
        return False
    text = str(candidate)
    allowed = [root, closeout_root]
    for base in allowed:
        base_text = str(base)
        try:
            candidate_resolved = str(Path(text).resolve(strict=False))
            base_resolved = str(base.resolve(strict=False))
        except OSError:
            candidate_resolved = text
            base_resolved = base_text
        if text.startswith(base_text) or candidate_resolved.startswith(base_resolved):
            return True
        marker = "artifacts/final/"
        if marker in candidate_resolved and marker in base_resolved:
            if candidate_resolved.split(marker, 1)[1].startswith(base_resolved.split(marker, 1)[1]):
                return True
    return False


def _read_manifest_trace_id(path_value: object) -> str:
    if _blank(path_value):
        return ""
    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("trace_id") or payload.get("run_id") or "")


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_discovery_task_ids(closeout_root: Path) -> set[str]:
    path = closeout_root / "manifests" / "discovered_decision_slice.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(task_id) for task_id in payload.get("task_ids", [])}


def _load_run_tables(closeout_root: Path) -> list[tuple[Path, pd.DataFrame]]:
    tables: list[tuple[Path, pd.DataFrame]] = []
    for dirname in ("runs", "runs_retry_clean"):
        for path in sorted((closeout_root / dirname).glob("**/controller_trace.csv")):
            tables.append((path, pd.read_csv(path, low_memory=False)))
    return tables


def _load_reused_tables(root: Path, closeout_root: Path) -> list[tuple[Path, pd.DataFrame]]:
    reuse_path = closeout_root / "manifests" / "reuse_decisions.csv"
    trace_path = root / "decision_sensitive_admission" / "controller_trace.csv"
    if not reuse_path.exists() or not trace_path.exists():
        return []
    reuse = pd.read_csv(reuse_path)
    if reuse.empty or not reuse.get("reuse_decision", pd.Series(dtype=str)).eq("reuse").any():
        return []
    trace = pd.read_csv(trace_path, low_memory=False)
    trace["task_id_norm"] = trace["task_id"].astype(str)
    trace["backend_norm"] = trace.get("backend_engine", trace.get("backend", "")).map(_normalize_backend)
    trace["controller_norm"] = trace.get("controller", trace.get("decision_label", "")).map(_normalize_controller)
    trace["regime_public"] = trace.get("regime", "").map(_normalize_regime)
    trace["seed_norm"] = trace.get("seed", pd.Series(dtype=float)).map(lambda value: int(float(value)) if not _blank(value) else -1)
    trace["budget_norm"] = trace.get("budget", pd.Series(dtype=float)).map(lambda value: int(float(value)) if not _blank(value) else -1)

    reuse_keys = {
        (
            str(row.task_id),
            str(row.backend_engine),
            str(row.regime),
            int(row.seed),
            int(row.budget),
            str(row.controller_id),
        )
        for row in reuse[reuse["reuse_decision"].eq("reuse")].itertuples(index=False)
    }
    mask = trace.apply(
        lambda row: (
            str(row["task_id_norm"]),
            str(row["backend_norm"]),
            str(row["regime_public"]),
            int(row["seed_norm"]),
            int(row["budget_norm"]),
            str(row["controller_norm"]),
        )
        in reuse_keys,
        axis=1,
    )
    return [(trace_path, trace[mask].copy())]


def _row_value(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index and not _blank(row.get(name)):
            return row.get(name)
    return ""


def _normalize_row(root: Path, closeout_root: Path, row: pd.Series, source_csv: Path) -> dict[str, Any]:
    backend = _normalize_backend(_row_value(row, "backend_engine", "backend", "model_backend"))
    regime = _normalize_regime(_row_value(row, "regime"))
    controller = _normalize_controller(_row_value(row, "controller_id", "controller", "decision_label"))
    seed_raw = _row_value(row, "seed")
    budget_raw = _row_value(row, "budget")
    seed = int(float(seed_raw)) if not _blank(seed_raw) else -1
    budget = int(float(budget_raw)) if not _blank(budget_raw) else -1
    task_id = str(_row_value(row, "task_id"))
    manifest_path = _row_value(row, "run_manifest_path", "manifest_path")
    trace_id = str(_row_value(row, "trace_id") or _read_manifest_trace_id(manifest_path))
    summary_path = _row_value(row, "summary_path")
    artifact_root = _row_value(row, "artifact_root") or (str(Path(str(summary_path)).parent) if not _blank(summary_path) else str(source_csv.parent))
    success = _truthy(_row_value(row, "task_success", "passed"))
    terminal = _row_value(row, "terminal_outcome") or ("pass" if success else "fail")
    run_id = str(_row_value(row, "run_id") or f"webarena_robustness_{backend}_{regime}_{controller}_seed{seed}_budget{budget}_task{task_id}")
    normalized = {
        "run_id": run_id,
        "release_root": str(_row_value(row, "release_root", "source_root") or root),
        "closeout_root": str(closeout_root),
        "workload_family": "webarena_verified",
        "workload_slice": WORKLOAD_SLICE,
        "task_id": task_id,
        "regime": regime,
        "stress_config_id": regime,
        "backend_engine": backend,
        "controller_id": controller,
        "seed": seed,
        "budget": budget,
        "driver_id": _row_value(row, "driver_id") or f"webarena_real_controller:{controller}:{regime}:{backend}",
        "driver_type": _row_value(row, "driver_type") or "controller",
        "driver_version": _row_value(row, "driver_version") or "real_controller_study_v1",
        "traffic_source_id": _row_value(row, "traffic_source_id") or f"webarena_real_controller_{backend}",
        "traffic_source_type": _row_value(row, "traffic_source_type") or "controller_driver",
        "model_family": _row_value(row, "model_family") or "controller_policy",
        "model_id": _row_value(row, "model_id") or f"webarena_real_controller_{backend}",
        "model_backend": _row_value(row, "model_backend") or backend,
        "policy_version": _row_value(row, "policy_version"),
        "prompt_template_hash": _row_value(row, "prompt_template_hash"),
        "action_parser_version": _row_value(row, "action_parser_version"),
        "implementation_source": _row_value(row, "implementation_source"),
        "terminal_outcome": terminal,
        "terminal_outcome_present": _row_value(row, "terminal_outcome_present") or True,
        "reward": _row_value(row, "reward") if not _blank(_row_value(row, "reward")) else (1.0 if success else 0.0),
        "reward_auc_over_wallclock": _row_value(row, "reward_auc_over_wallclock", "reward_auc_over_wallclock_mean"),
        "reward_auc_over_wall_clock_input_path": str(source_csv),
        "pass_rate": _row_value(row, "pass_rate") if not _blank(_row_value(row, "pass_rate")) else (1.0 if success else 0.0),
        "pass_rate_input_path": str(source_csv),
        "p99_latency_ms": _row_value(row, "p99_latency_ms"),
        "queue_wait_p99_ms": _row_value(row, "queue_wait_p99_ms"),
        "evidence_validation_pass": _row_value(row, "evidence_validation_pass"),
        "trace_id": trace_id,
        "manifest_hash": _row_value(row, "manifest_hash"),
        "manifest_path": manifest_path,
        "artifact_root": artifact_root,
        "paper_role": "decision_robustness_closeout",
        "exclude_from_canonical_surface": True,
        "sensitive_hash_only": True,
        "source_csv": str(source_csv),
        "source_paper_role": _row_value(row, "paper_role"),
        "source_root": _row_value(row, "source_root", "release_root", "summary_path", "events_path") or str(source_csv),
    }
    return normalized


def _gate_row(row: dict[str, Any], root: Path, closeout_root: Path, task_ids: set[str], allowed_seeds: set[int]) -> tuple[bool, list[str], list[str]]:
    block: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if _blank(row.get(field))]
    if missing:
        block.append("missing_required_metadata")
    if not _truthy(row.get("evidence_validation_pass")):
        block.append("evidence_validation_pass_not_true")
    if row.get("terminal_outcome") not in {"pass", "fail"}:
        block.append("terminal_outcome_missing_or_invalid")
    if _blank(row.get("trace_id")):
        block.append("trace_id_missing")
    if _blank(row.get("manifest_hash")):
        block.append("manifest_hash_missing")
    source_role = str(row.get("source_paper_role") or "").lower()
    source_path = str(row.get("source_csv") or "").lower()
    if any(token in source_role or token in source_path for token in ("mock", "synthetic", "fixture", "preflight", "smoke")):
        block.append("preflight_or_smoke")
    if not _same_rootish(row.get("source_root") or row.get("source_csv"), root, closeout_root):
        block.append("source_outside_canonical_or_closeout_root")
    if row.get("controller_id") not in CONTROLLERS:
        block.append("controller_outside_frozen_matrix")
    if row.get("backend_engine") not in BACKENDS:
        block.append("backend_outside_frozen_matrix")
    if row.get("regime") not in REGIMES:
        block.append("regime_outside_frozen_matrix")
    if int(row.get("seed", -1)) not in allowed_seeds:
        block.append("seed_outside_frozen_matrix")
    if int(row.get("budget", -1)) not in BUDGETS:
        block.append("budget_outside_frozen_matrix")
    if task_ids and str(row.get("task_id")) not in task_ids:
        block.append("task_id_outside_discovered_slice")
    combined = " ".join(str(value).lower() for value in row.values())
    if any(pattern in combined for pattern in SENSITIVE_PATTERNS):
        block.append("sensitive_path_or_secret_marker")
    return not block, block, missing


def _dedupe_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, int]:
    return (
        str(row.get("task_id")),
        str(row.get("backend_engine")),
        str(row.get("regime")),
        str(row.get("controller_id")),
        int(row.get("seed", -1)),
        int(row.get("budget", -1)),
    )


def _candidate_rank(candidate: dict[str, Any]) -> tuple[int, int]:
    source = str(candidate["row"].get("source_csv") or "")
    retry_rank = 1 if "/runs_retry_clean/" in source else 0
    evidence_rank = 1 if candidate["ok"] else 0
    return evidence_rank, retry_rank


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def validate_decision_robustness_gate(root: Path, *, include_reused: bool = True, allow_optional_seed2: bool = False) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=False)
    closeout_root = _closeout_root(root)
    gate_dir = closeout_root / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    task_ids = _load_discovery_task_ids(closeout_root)
    allowed_seeds = set(SEEDS_DEFAULT)
    if allow_optional_seed2:
        allowed_seeds.add(2)

    tables = _load_run_tables(closeout_root)
    if include_reused:
        tables.extend(_load_reused_tables(root, closeout_root))

    candidates: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    sensitive_rows: list[dict[str, Any]] = []
    for source_csv, frame in tables:
        for index, source_row in frame.iterrows():
            row = _normalize_row(root, closeout_root, source_row, source_csv)
            row["gate_row_id"] = f"{source_csv.name}:{index}"
            ok, block, missing = _gate_row(row, root, closeout_root, task_ids, allowed_seeds)
            row["block_reason"] = ";".join(block)
            row["missing_fields"] = ";".join(missing)
            candidates.append({"row": row, "ok": ok, "block": block, "missing": missing})
            if missing:
                missing_rows.append(row)
            sensitive_rows.append(
                {
                    "gate_row_id": row["gate_row_id"],
                    "run_id": row["run_id"],
                    "sensitive_hash_only": "true",
                    "raw_sensitive_copied": "false",
                    "source_csv_hash": _hash_text(str(source_csv)),
                }
            )

    admitted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    superseded: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str, str, int, int], list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_key.setdefault(_dedupe_key(candidate["row"]), []).append(candidate)
    for group in by_key.values():
        group_sorted = sorted(group, key=_candidate_rank, reverse=True)
        chosen = group_sorted[0]
        if chosen["ok"]:
            admitted.append(chosen["row"])
            for candidate in group_sorted[1:]:
                row = dict(candidate["row"])
                row["superseded_reason"] = "duplicate_cell_replaced_by_clean_retry_or_valid_row"
                superseded.append(row)
        else:
            excluded.extend(candidate["row"] for candidate in group_sorted)

    output_fields = REQUIRED_FIELDS + [
        "terminal_outcome_present",
        "reward_auc_over_wallclock",
        "pass_rate",
        "p99_latency_ms",
        "queue_wait_p99_ms",
        "manifest_path",
        "source_csv",
        "source_root",
        "source_paper_role",
        "gate_row_id",
        "block_reason",
        "missing_fields",
    ]
    _write_csv(gate_dir / "decision_robustness_admitted_rows.csv", admitted, output_fields)
    _write_csv(gate_dir / "decision_robustness_excluded_rows.csv", excluded, output_fields)
    _write_csv(
        gate_dir / "decision_robustness_missing_metadata.csv",
        missing_rows,
        ["gate_row_id", "run_id", "task_id", "backend_engine", "regime", "controller_id", "seed", "budget", "missing_fields"],
    )
    _write_csv(
        gate_dir / "decision_robustness_superseded_rows.csv",
        superseded,
        output_fields + ["superseded_reason"],
    )
    _write_csv(gate_dir / "sensitive_hash_manifest.csv", sensitive_rows, ["gate_row_id", "run_id", "sensitive_hash_only", "raw_sensitive_copied", "source_csv_hash"])
    report = {
        "closeout_root": str(closeout_root),
        "input_tables": [str(path) for path, _ in tables],
        "total_rows_seen": len(admitted) + len(excluded),
        "admitted_rows": len(admitted),
        "excluded_rows": len(excluded),
        "missing_metadata_rows": len(missing_rows),
        "superseded_rows": len(superseded),
        "raw_sensitive_copied": False,
        "include_reused": include_reused,
    }
    (gate_dir / "decision_robustness_gate_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate decision robustness closeout rows.")
    parser.add_argument("--root", "--canonical-root", dest="root", type=Path, required=True)
    parser.add_argument("--include-reused", choices=["true", "false"], default="true")
    parser.add_argument("--allow-optional-seed2", action="store_true")
    args = parser.parse_args(argv)
    report = validate_decision_robustness_gate(
        args.root,
        include_reused=args.include_reused == "true",
        allow_optional_seed2=args.allow_optional_seed2,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
