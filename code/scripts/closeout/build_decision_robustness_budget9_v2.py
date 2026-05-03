#!/usr/bin/env python3
"""Build an isolated v2 WebArena controller robustness closeout with budget=9."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.webarena_verified.real_controller_study import SUPPORTED_TASK_IDS


CLOSEOUT_NAME = "decision_robustness_closeout_v2_budget9"
CONTROLLERS = ("hook_a_only", "hook_b_only")
BACKENDS = ("vllm", "sglang")
REGIMES = ("clean_baseline", "medium_live_stressed")
RUNNER_REGIME = {"clean_baseline": "clean", "medium_live_stressed": "medium"}
SEEDS = (0, 1)
BUDGETS = (5, 7, 9)
EXPECTED_ROWS = 2 * 2 * 2 * 2 * sum(BUDGETS)


def _closeout_root(root: Path) -> Path:
    return root / CLOSEOUT_NAME


def _ensure_dirs(closeout_root: Path) -> None:
    for rel in ("manifests", "runs", "runs_retry_clean", "gate", "aggregates", "reports", "logs", "preflight_only"):
        (closeout_root / rel).mkdir(parents=True, exist_ok=True)


def _read_task_ids(path: Path) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    duplicates: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        task_id = raw.strip()
        if not task_id or task_id.startswith("#"):
            continue
        if task_id in seen:
            duplicates.append(task_id)
            continue
        seen.add(task_id)
        ordered.append(task_id)
    return ordered


def _dataset_task_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("task_id")) for row in payload}


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


def _freeze_slice(available_tasks_path: Path, dataset_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    available = _read_task_ids(available_tasks_path)
    dataset_ids = _dataset_task_ids(dataset_path)
    frozen: list[str] = []
    unsupported: list[dict[str, str]] = []
    for task_id in available:
        if task_id not in dataset_ids:
            unsupported.append({"task_id": task_id, "reason": "not_in_dataset"})
            continue
        if task_id not in SUPPORTED_TASK_IDS:
            unsupported.append({"task_id": task_id, "reason": "not_controller_supported"})
            continue
        if task_id not in frozen:
            frozen.append(task_id)
        if len(frozen) == 9:
            break
    for task_id in available:
        if task_id not in frozen and task_id in dataset_ids and task_id not in SUPPORTED_TASK_IDS:
            if not any(row["task_id"] == task_id for row in unsupported):
                unsupported.append({"task_id": task_id, "reason": "not_controller_supported"})
    if len(set(frozen)) < 9:
        raise ValueError(f"fewer than 9 unique controller-compatible task ids discovered: {frozen}")
    return frozen, unsupported


def _planned_rows(root: Path, task_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        if budget > len(set(task_ids)):
            raise ValueError(f"requested_budget {budget} exceeds unique discovered task ids {len(set(task_ids))}")
        for task_id in task_ids[:budget]:
            for backend in BACKENDS:
                for seed in SEEDS:
                    for regime in REGIMES:
                        for controller in CONTROLLERS:
                            rows.append(
                                {
                                    "closeout_id": CLOSEOUT_NAME,
                                    "planned_cell_id": f"{backend}_seed{seed}_budget{budget}_{regime}_{controller}_task{task_id}",
                                    "workload_family": "webarena_verified",
                                    "workload_slice": "webarena_verified_controller_budget9_v2",
                                    "task_id": task_id,
                                    "regime": regime,
                                    "runner_regime": RUNNER_REGIME[regime],
                                    "controller_id": controller,
                                    "backend_engine": backend,
                                    "seed": seed,
                                    "budget": budget,
                                    "paper_role": "decision_robustness_closeout_v2_budget9",
                                    "exclude_from_canonical_surface": "true",
                                    "expected_release_root": str(root),
                                }
                            )
    return rows


def _write_task_lists(closeout_root: Path, task_ids: list[str]) -> None:
    for budget in BUDGETS:
        (closeout_root / "manifests" / f"task_ids_budget_{budget}.txt").write_text(
            "\n".join(task_ids[:budget]) + "\n",
            encoding="utf-8",
        )


def _write_job_manifest(closeout_root: Path, task_ids: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    idx = 1
    for backend in BACKENDS:
        for seed in SEEDS:
            for budget in BUDGETS:
                task_file = closeout_root / "manifests" / f"task_ids_budget_{budget}.txt"
                output_root = closeout_root / "runs" / f"backend_{backend}" / f"seed_{seed}" / f"budget_{budget}"
                jobs.append(
                    {
                        "job_index": idx,
                        "backend_engine": backend,
                        "seed": seed,
                        "budget": budget,
                        "task_count": budget,
                        "task_file": str(task_file),
                        "output_root": str(output_root),
                        "log_path": str(closeout_root / "logs" / f"run_backend_{backend}_seed{seed}_budget{budget}.log"),
                    }
                )
                idx += 1
    _write_csv(
        closeout_root / "manifests" / "execution_jobs.csv",
        jobs,
        ["job_index", "backend_engine", "seed", "budget", "task_count", "task_file", "output_root", "log_path"],
    )
    return jobs


def build_budget9_v2(root: Path, *, available_tasks_path: Path, dataset_path: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=False)
    closeout_root = _closeout_root(root)
    _ensure_dirs(closeout_root)
    task_ids, unsupported = _freeze_slice(available_tasks_path, dataset_path)
    planned = _planned_rows(root, task_ids)
    _write_task_lists(closeout_root, task_ids)
    jobs = _write_job_manifest(closeout_root, task_ids)
    _write_json(
        closeout_root / "manifests" / "frozen_9task_slice.json",
        {
            "closeout_id": CLOSEOUT_NAME,
            "task_ids": task_ids,
            "task_count": len(task_ids),
            "available_tasks_path": str(available_tasks_path),
            "dataset_path": str(dataset_path),
            "duplicate_task_ids_forbidden": True,
            "requested_budgets": list(BUDGETS),
            "expected_rows": EXPECTED_ROWS,
        },
    )
    _write_csv(
        closeout_root / "manifests" / "planned_cells.csv",
        planned,
        [
            "closeout_id",
            "planned_cell_id",
            "workload_family",
            "workload_slice",
            "task_id",
            "regime",
            "runner_regime",
            "controller_id",
            "backend_engine",
            "seed",
            "budget",
            "paper_role",
            "exclude_from_canonical_surface",
            "expected_release_root",
        ],
    )
    _write_csv(closeout_root / "unsupported_cells.csv", unsupported, ["task_id", "reason"])
    _write_csv(
        closeout_root / "planned_vs_executed.csv",
        [
            {
                "planned_rows": len(planned),
                "expected_rows": EXPECTED_ROWS,
                "executed_rows": 0,
                "admitted_rows": 0,
                "status": "planned_not_executed",
            }
        ],
        ["planned_rows", "expected_rows", "executed_rows", "admitted_rows", "status"],
    )
    _write_csv(closeout_root / "superseded_lineage.csv", [], ["source_row_id", "replacement_row_id", "reason"])
    summary = "\n".join(
        [
            "# Decision Robustness Budget9 V2",
            "",
            f"- closeout_root: `{closeout_root}`",
            f"- frozen_task_ids: {', '.join(task_ids)}",
            f"- planned_rows: {len(planned)}",
            "- canonical_surface_mutated: no",
            "- paper_text_mutated: no",
            "- figures_regenerated: no",
        ]
    )
    (closeout_root / "summary.md").write_text(summary + "\n", encoding="utf-8")
    return {"closeout_root": str(closeout_root), "frozen_task_count": len(task_ids), "planned_rows": len(planned), "jobs": len(jobs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build isolated budget=9 decision robustness v2 manifest.")
    parser.add_argument("--root", "--canonical-root", dest="root", type=Path, required=True)
    parser.add_argument("--available-tasks-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_budget9_v2(args.root, available_tasks_path=args.available_tasks_path, dataset_path=args.dataset_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
