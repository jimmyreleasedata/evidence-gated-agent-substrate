#!/usr/bin/env python3
"""Build the WebArena controller-choice robustness closeout manifest.

This script is preparation-only. It discovers the existing admitted controller
decision slice, writes the frozen closeout spec, planned cells, smoke cells, and
reuse decisions. It does not run WebArena or mutate the canonical surface.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


CLOSEOUT_NAME = "decision_robustness_closeout_v1"
WORKLOAD_SLICE = "webarena_verified_decision_slice"
CONTROLLERS = ("hook_a_only", "hook_b_only")
BACKENDS = ("vllm", "sglang")
PUBLIC_REGIMES = ("clean_baseline", "medium_live_stressed")
RUNNER_REGIME_BY_PUBLIC = {
    "clean_baseline": "clean",
    "medium_live_stressed": "medium",
}
SEEDS = (0, 1)
BUDGETS = (5, 7, 9)
RUNNER_ENTRYPOINT = "scripts/run_webarena_real_controller_pilot.sh"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _closeout_root(root: Path) -> Path:
    return root / CLOSEOUT_NAME


def _ensure_dirs(closeout_root: Path) -> None:
    for rel in (
        "manifests",
        "runs",
        "traces",
        "gate",
        "aggregates",
        "reports",
        "logs",
        "preflight_only",
    ):
        (closeout_root / rel).mkdir(parents=True, exist_ok=True)


def _read_trace(root: Path) -> pd.DataFrame:
    trace_path = root / "decision_sensitive_admission" / "controller_trace.csv"
    if not trace_path.exists():
        raise FileNotFoundError(f"missing existing decision trace: {trace_path}")
    df = pd.read_csv(trace_path, low_memory=False)
    if df.empty:
        raise ValueError(f"empty existing decision trace: {trace_path}")
    return df


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


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "ok"}


def _blank(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}


def _task_sort_key(task_id: str) -> tuple[int, str]:
    return (int(task_id), task_id) if str(task_id).isdigit() else (10**9, str(task_id))


def _same_rootish(candidate: object, root: Path) -> bool:
    if _blank(candidate):
        return False
    text = str(candidate)
    root_text = str(root)
    try:
        candidate_resolved = str(Path(text).resolve(strict=False))
        root_resolved = str(root.resolve(strict=False))
    except OSError:
        candidate_resolved = text
        root_resolved = root_text
    if candidate_resolved.startswith(root_resolved) or text.startswith(root_text):
        return True
    marker = "artifacts/final/"
    if marker in candidate_resolved and marker in root_resolved:
        return candidate_resolved.split(marker, 1)[1].startswith(root_resolved.split(marker, 1)[1])
    return False


def _discover_slice(root: Path, trace: pd.DataFrame) -> dict[str, Any]:
    family = trace.get("family", pd.Series([""] * len(trace))).astype(str).str.lower()
    df = trace[family.str.contains("webarena", na=False)].copy()
    if df.empty:
        df = trace.copy()
    df["task_id_norm"] = df["task_id"].astype(str)
    df["backend_norm"] = df.get("backend_engine", df.get("backend", "")).map(_normalize_backend)
    df["controller_norm"] = df.get("controller", df.get("decision_label", "")).map(_normalize_controller)
    df["regime_public"] = df.get("regime", "").map(_normalize_regime)
    task_ids = sorted(df["task_id_norm"].dropna().astype(str).unique().tolist(), key=_task_sort_key)
    backends = sorted(set(df["backend_norm"]) & set(BACKENDS))
    controllers = sorted(set(df["controller_norm"]) & set(CONTROLLERS))
    regimes = sorted(set(df["regime_public"]) & set(PUBLIC_REGIMES))
    budgets = sorted(int(float(value)) for value in df.get("budget", pd.Series(dtype=float)).dropna().unique())
    seeds = sorted(int(float(value)) for value in df.get("seed", pd.Series(dtype=float)).dropna().unique())
    missing: list[str] = []
    if not task_ids:
        missing.append("task_ids")
    if set(backends) != set(BACKENDS):
        missing.append("vllm_sglang_backends")
    if set(controllers) != set(CONTROLLERS):
        missing.append("hook_a_hook_b_controllers")
    if set(regimes) != set(PUBLIC_REGIMES):
        missing.append("clean_medium_regimes")
    return {
        "canonical_root": str(root),
        "existing_trace_path": str(root / "decision_sensitive_admission" / "controller_trace.csv"),
        "workload_family": "webarena_verified",
        "workload_slice": WORKLOAD_SLICE,
        "task_ids": task_ids,
        "controllers": list(CONTROLLERS),
        "backends": list(BACKENDS),
        "regimes": list(PUBLIC_REGIMES),
        "runner_regimes": RUNNER_REGIME_BY_PUBLIC,
        "existing_budgets": budgets,
        "existing_seeds": seeds,
        "requested_seeds": list(SEEDS),
        "requested_budgets": list(BUDGETS),
        "runner_entrypoint": RUNNER_ENTRYPOINT,
        "missing_discovery_fields": missing,
        "supported_task_count": len(task_ids),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_task_lists(closeout_root: Path, task_ids: list[str]) -> None:
    for budget in BUDGETS:
        path = closeout_root / "manifests" / f"task_ids_budget_{budget}.txt"
        selected = task_ids[: min(budget, len(task_ids))]
        path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")


def _planned_rows(root: Path, discovery: dict[str, Any], *, allow_optional_seed2: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_ids = list(discovery["task_ids"])
    seeds = list(SEEDS) + ([2] if allow_optional_seed2 else [])
    for budget in BUDGETS:
        selected_task_ids = task_ids[: min(budget, len(task_ids))]
        notes = []
        executable = True
        if budget > len(task_ids):
            notes.append("budget_exceeds_discovered_task_count")
            executable = False
            selected_task_ids = task_ids
        for task_id in selected_task_ids:
            for backend in BACKENDS:
                for regime in PUBLIC_REGIMES:
                    for controller in CONTROLLERS:
                        for seed in seeds:
                            planned_cell_id = f"{backend}_{regime}_{controller}_seed{seed}_budget{budget}_task{task_id}"
                            rows.append(
                                {
                                    "closeout_id": CLOSEOUT_NAME,
                                    "planned_cell_id": planned_cell_id,
                                    "workload_family": "webarena_verified",
                                    "workload_slice": WORKLOAD_SLICE,
                                    "task_id": task_id,
                                    "regime": regime,
                                    "runner_regime": RUNNER_REGIME_BY_PUBLIC[regime],
                                    "stress_config_id": regime,
                                    "controller_id": controller,
                                    "backend_engine": backend,
                                    "seed": seed,
                                    "budget": budget,
                                    "runner_entrypoint": RUNNER_ENTRYPOINT,
                                    "run_mode": "planned_real_closeout",
                                    "paper_role": "decision_robustness_closeout",
                                    "exclude_from_canonical_surface": "true",
                                    "expected_release_root": str(root),
                                    "executable": str(executable).lower(),
                                    "notes": ";".join(notes),
                                }
                            )
    return rows


def _smoke_rows(planned: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_ids = sorted({str(row["task_id"]) for row in planned}, key=_task_sort_key)
    if not task_ids:
        return []
    first_task = task_ids[0]
    return [
        row
        for row in planned
        if row["task_id"] == first_task
        and row["backend_engine"] == "vllm"
        and row["seed"] == 1
        and row["budget"] == 5
        and row["regime"] in PUBLIC_REGIMES
        and row["controller_id"] in CONTROLLERS
    ]


def _source_trace_key(df: pd.DataFrame) -> dict[tuple[str, str, str, int, int, str], dict[str, Any]]:
    keyed: dict[tuple[str, str, str, int, int, str], dict[str, Any]] = {}
    if df.empty:
        return keyed
    work = df.copy()
    work["backend_norm"] = work.get("backend_engine", work.get("backend", "")).map(_normalize_backend)
    work["controller_norm"] = work.get("controller", work.get("decision_label", "")).map(_normalize_controller)
    work["regime_public"] = work.get("regime", "").map(_normalize_regime)
    for _, row in work.iterrows():
        try:
            seed = int(float(row.get("seed")))
            budget = int(float(row.get("budget")))
        except (TypeError, ValueError):
            continue
        key = (
            str(row.get("task_id")),
            str(row.get("backend_norm")),
            str(row.get("regime_public")),
            seed,
            budget,
            str(row.get("controller_norm")),
        )
        keyed[key] = row.to_dict()
    return keyed


def _reuse_decisions(root: Path, planned: list[dict[str, Any]], trace: pd.DataFrame) -> list[dict[str, Any]]:
    source = _source_trace_key(trace)
    rows: list[dict[str, Any]] = []
    for planned_row in planned:
        decision = "not_applicable"
        reason = "reuse_only_applies_to_seed0_budget7"
        source_path = ""
        if int(planned_row["seed"]) == 0 and int(planned_row["budget"]) == 7 and planned_row.get("executable") == "true":
            key = (
                str(planned_row["task_id"]),
                str(planned_row["backend_engine"]),
                str(planned_row["regime"]),
                0,
                7,
                str(planned_row["controller_id"]),
            )
            row = source.get(key)
            checks: list[str] = []
            if row is None:
                decision = "rerun"
                reason = "matching_canonical_row_not_found"
            else:
                source_path = str(row.get("summary_path") or row.get("events_path") or "")
                checks.append("match_found")
                if not _same_rootish(row.get("source_root") or row.get("release_root") or row.get("summary_path"), root):
                    checks.append("source_root_not_canonical")
                if not _truthy(row.get("evidence_validation_pass")):
                    checks.append("evidence_validation_pass_not_true")
                if _blank(row.get("manifest_hash")):
                    checks.append("manifest_hash_missing")
                if _blank(row.get("run_manifest_path") or row.get("manifest_path")):
                    checks.append("manifest_path_missing")
                paper_role = str(row.get("paper_role") or "").lower()
                if any(token in paper_role for token in ("smoke", "fixture", "preflight", "diagnostic")):
                    checks.append("non_admitted_role")
                blocking = [check for check in checks if check != "match_found"]
                decision = "reuse" if not blocking else "rerun"
                reason = "canonical_seed0_budget7_row_gate_passes" if not blocking else ";".join(blocking)
        rows.append(
            {
                "planned_cell_id": planned_row["planned_cell_id"],
                "task_id": planned_row["task_id"],
                "backend_engine": planned_row["backend_engine"],
                "regime": planned_row["regime"],
                "controller_id": planned_row["controller_id"],
                "seed": planned_row["seed"],
                "budget": planned_row["budget"],
                "reuse_decision": decision,
                "reuse_reason": reason,
                "source_path": source_path,
            }
        )
    return rows


def _write_discovery_report(closeout_root: Path, discovery: dict[str, Any]) -> None:
    missing = discovery["missing_discovery_fields"]
    text = "\n".join(
        [
            "# Decision Robustness Discovery Report",
            "",
            f"- canonical_root: `{discovery['canonical_root']}`",
            f"- existing_trace_path: `{discovery['existing_trace_path']}`",
            f"- task_ids: {', '.join(discovery['task_ids'])}",
            f"- supported_task_count: {discovery['supported_task_count']}",
            f"- controllers: {', '.join(discovery['controllers'])}",
            f"- backends: {', '.join(discovery['backends'])}",
            f"- regimes: {', '.join(discovery['regimes'])}",
            f"- existing_budgets: {discovery['existing_budgets']}",
            f"- existing_seeds: {discovery['existing_seeds']}",
            f"- requested_budgets: {discovery['requested_budgets']}",
            f"- budget_9_status: {'blocked_until_at_least_9_supported_tasks_exist' if discovery['supported_task_count'] < 9 else 'executable'}",
            f"- missing_discovery_fields: {missing if missing else 'none'}",
            "",
            "No WebArena live traffic was run by this discovery script.",
        ]
    )
    (closeout_root / "manifests" / "discovery_report.md").write_text(text + "\n", encoding="utf-8")


def _write_spec_doc(root: Path, closeout_root: Path, discovery: dict[str, Any]) -> None:
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    spec = "\n".join(
        [
            "# WebArena Verified Controller-Choice Seed/Budget Robustness Closeout V1",
            "",
            f"- closeout_root: `{closeout_root}`",
            f"- workload_slice: `{WORKLOAD_SLICE}`",
            f"- task_ids: {', '.join(discovery['task_ids'])}",
            f"- controllers: {', '.join(CONTROLLERS)}",
            f"- backends: {', '.join(BACKENDS)}",
            f"- regimes: {', '.join(PUBLIC_REGIMES)}",
            f"- seeds: {', '.join(str(seed) for seed in SEEDS)}",
            f"- budgets: {', '.join(str(budget) for budget in BUDGETS)}",
            "- stress_config: clean_baseline maps to runner clean; medium_live_stressed maps to runner medium.",
            "- admission_predicate: separate closeout rows only; evidence_validation_pass=true, terminal outcome present, manifest hash present, trace id present, required driver/model/controller metadata present, no smoke/preflight/fixture/mock rows.",
            "- aggregation_script: `scripts/closeout/aggregate_decision_robustness_v1.py`.",
            "- claim_ladder: STRONG, DIRECTIONAL, NARROW, FAIL as specified in the closeout task.",
            "- failure_policy: no synthetic rows; missing runner/metadata/gate failures are reported and not patched around.",
            f"- artifact_paths: `{closeout_root}/manifests`, `{closeout_root}/runs`, `{closeout_root}/gate`, `{closeout_root}/aggregates`, `{closeout_root}/reports`.",
            "",
            "This closeout package is not part of the canonical paper-facing surface unless explicitly promoted later.",
        ]
    )
    (docs_dir / "decision_robustness_closeout_v1.md").write_text(spec + "\n", encoding="utf-8")


def _append_docs_entries(root: Path, closeout_root: Path, planned_n: int, reusable_n: int) -> None:
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now()
    status = (
        f"\n## {stamp} - Decision robustness closeout prepared\n"
        f"- closeout_root: `{closeout_root}`\n"
        f"- planned_rows: {planned_n}\n"
        f"- canonical_reuse_rows: {reusable_n}\n"
        "- status: preparation only; no live WebArena traffic executed.\n"
    )
    execution = (
        f"\n## {stamp} - Decision robustness preparation\n"
        "- commands: build manifest/discovery/reuse decisions via `build_decision_robustness_manifest_v1.py`.\n"
        f"- outputs: `{closeout_root}/manifests/discovered_decision_slice.json`, planned cells, smoke cells, reuse decisions.\n"
        "- live execution: not run.\n"
    )
    with (docs_dir / "status_board.md").open("a", encoding="utf-8") as handle:
        handle.write(status)
    with (docs_dir / "execution_log.md").open("a", encoding="utf-8") as handle:
        handle.write(execution)


def build_decision_robustness_manifest(
    root: Path,
    *,
    allow_optional_seed2: bool = False,
    write_repo_docs: bool = True,
) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=False)
    closeout_root = _closeout_root(root)
    _ensure_dirs(closeout_root)
    trace = _read_trace(root)
    discovery = _discover_slice(root, trace)
    _write_json(closeout_root / "manifests" / "discovered_decision_slice.json", discovery)
    _write_discovery_report(closeout_root, discovery)
    _write_task_lists(closeout_root, list(discovery["task_ids"]))

    planned = _planned_rows(root, discovery, allow_optional_seed2=allow_optional_seed2)
    smoke = _smoke_rows(planned)
    reuse = _reuse_decisions(root, planned, trace)
    fieldnames = [
        "closeout_id",
        "planned_cell_id",
        "workload_family",
        "workload_slice",
        "task_id",
        "regime",
        "runner_regime",
        "stress_config_id",
        "controller_id",
        "backend_engine",
        "seed",
        "budget",
        "runner_entrypoint",
        "run_mode",
        "paper_role",
        "exclude_from_canonical_surface",
        "expected_release_root",
        "executable",
        "notes",
    ]
    _write_csv(closeout_root / "manifests" / "decision_robustness_planned_cells.csv", planned, fieldnames)
    _write_csv(closeout_root / "manifests" / "decision_robustness_smoke_cells.csv", smoke, fieldnames)
    _write_csv(
        closeout_root / "manifests" / "reuse_decisions.csv",
        reuse,
        [
            "planned_cell_id",
            "task_id",
            "backend_engine",
            "regime",
            "controller_id",
            "seed",
            "budget",
            "reuse_decision",
            "reuse_reason",
            "source_path",
        ],
    )
    blocked_rows = [row for row in planned if "budget_exceeds_discovered_task_count" in str(row.get("notes"))]
    _write_csv(
        closeout_root / "logs" / "skipped_cells.csv",
        [
            {
                "planned_cell_id": row["planned_cell_id"],
                "task_id": row["task_id"],
                "backend_engine": row["backend_engine"],
                "regime": row["regime"],
                "controller_id": row["controller_id"],
                "seed": row["seed"],
                "budget": row["budget"],
                "skip_reason": row["notes"],
            }
            for row in blocked_rows
        ],
        ["planned_cell_id", "task_id", "backend_engine", "regime", "controller_id", "seed", "budget", "skip_reason"],
    )
    if write_repo_docs:
        _write_spec_doc(root, closeout_root, discovery)
        _append_docs_entries(root, closeout_root, len(planned), sum(1 for row in reuse if row["reuse_decision"] == "reuse"))
    return {
        "closeout_root": str(closeout_root),
        "supported_task_count": discovery["supported_task_count"],
        "planned_rows": len(planned),
        "smoke_rows": len(smoke),
        "reusable_rows": sum(1 for row in reuse if row["reuse_decision"] == "reuse"),
        "blocked_budget9_rows": len(blocked_rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build WebArena decision robustness closeout manifest.")
    parser.add_argument("--root", "--canonical-root", dest="root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-optional-seed2", action="store_true")
    parser.add_argument("--no-repo-docs", action="store_true", help="Do not write docs/status_board.md or docs/execution_log.md.")
    args = parser.parse_args(argv)
    result = build_decision_robustness_manifest(
        args.root,
        allow_optional_seed2=args.allow_optional_seed2,
        write_repo_docs=not args.no_repo_docs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
