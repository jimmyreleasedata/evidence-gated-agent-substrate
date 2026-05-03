#!/usr/bin/env python3
"""SWE-Gym mock runner with Apptainer-first backend selection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import sys

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from adapters.swe_gym.config import SweGymConfig
from adapters.swe_gym.replay import replay_from_manifest
from adapters.swe_gym.tasks import default_task_ids, get_task
from adapters.swe_gym.verifier import run_swe_task


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "replay"), default="live")
    parser.add_argument("--task-id", default="fix_answer")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--telemetry-mode", choices=("off", "basic", "full"), default="basic")
    parser.add_argument("--backend", choices=("apptainer", "docker"), default="apptainer")
    parser.add_argument(
        "--system-version",
        choices=("sync", "naive_async", "scheduler_only", "replay_only", "control_plane_only", "full_system"),
        default="full_system",
    )
    parser.add_argument(
        "--regime",
        choices=("baseline", "swe_gym_queue_stressed", "swe_gym_failure_stressed"),
        default="baseline",
    )
    parser.add_argument("--offered-load", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/swe_gym"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-host-fallback", action="store_true")
    parser.add_argument("--source-manifest", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = SweGymConfig(
        output_root=args.output_root,
        telemetry_mode=args.telemetry_mode,
        backend=args.backend,
        system_version=args.system_version,
        regime=args.regime,
        offered_load=args.offered_load,
        task_id=args.task_id,
        dry_run=args.dry_run,
        allow_host_fallback=not args.no_host_fallback,
        source_manifest=args.source_manifest,
    )

    if args.mode == "replay":
        if config.source_manifest is None:
            raise SystemExit("--source-manifest is required for replay mode")
        summary = replay_from_manifest(config.source_manifest, config.output_root, telemetry_mode=config.telemetry_mode)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    task_ids = default_task_ids() if args.all_tasks else [config.task_id]
    rows = [run_swe_task(get_task(task_id), config) for task_id in task_ids]
    aggregate_path = config.output_root / f"aggregate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    write_csv(
        aggregate_path,
        rows,
        [
            "mode",
            "run_id",
            "task_id",
            "task_instance_id",
            "backend",
            "system_version",
            "regime",
            "offered_load",
            "runtime_backend",
            "passed",
            "validation_pass",
            "duration_ms",
            "sandbox_start_ms",
            "repo_checkout_ms",
            "patch_apply_ms",
            "build_ms",
            "test_ms",
            "verifier_queue_wait_ms",
            "verifier_service_ms",
            "sandbox_cleanup_ms",
            "image_digest",
            "sif_hash",
            "repo_commit",
            "trace_path",
            "manifest_path",
            "summary_path",
        ],
    )
    print(json.dumps({"aggregate_path": str(aggregate_path), "task_count": len(rows), "backend": config.backend}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
