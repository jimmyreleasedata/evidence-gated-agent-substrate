#!/usr/bin/env python3
"""Normalize discovered evidence artifacts into the full v2 long schema."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.discover_all_experiment_roots import discover_experiment_roots
from scripts.final.full_evidence_common import (
    EVIDENCE_SCHEMA,
    infer_category,
    normalize_row,
    read_csv,
    repo_path,
    resolve_final_root,
    safe_relative,
    write_csv,
    write_parquet_if_possible,
)


CSV_HINTS = (
    "_summary.csv",
    "aggregate_",
    "inventory.csv",
    "matrix.csv",
    "contrast.csv",
    "telemetry.csv",
    "concurrency.csv",
    "surface.csv",
)


def _json_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _csv_payloads(path: Path) -> list[dict[str, object]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _root_rows(repo_root: Path, root: Path, source_category: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    manifest = _json_payload(root / "run_manifest.json") if (root / "run_manifest.json").exists() else {}
    summary = _json_payload(root / "summary.json") if (root / "summary.json").exists() else {}
    if manifest or summary:
        merged = {**manifest, **summary}
        rows.append(
            normalize_row(
                merged,
                source_root=Path(safe_relative(root, repo_root)),
                source_file=Path(safe_relative(root / ("summary.json" if summary else "run_manifest.json"), repo_root)),
                source_category=source_category,
                index=1,
            )
        )
    for child in sorted(root.iterdir()):
        if not child.is_file() or child.suffix.lower() != ".csv":
            continue
        if not (child.name.endswith(CSV_HINTS) or any(hint in child.name for hint in CSV_HINTS)):
            continue
        for index, payload in enumerate(_csv_payloads(child), start=1):
            rows.append(
                normalize_row(
                    payload,
                    source_root=Path(safe_relative(root, repo_root)),
                    source_file=Path(safe_relative(child, repo_root)),
                    source_category=source_category,
                    index=index,
                )
            )
    for json_name in ("capture_summary.json", "capture_status.json", "replay_summary.json", "evaluator_output.json", "evidence_gate_report.json"):
        path = root / json_name
        if path.exists() and not (manifest or summary):
            rows.append(
                normalize_row(
                    _json_payload(path),
                    source_root=Path(safe_relative(root, repo_root)),
                    source_file=Path(safe_relative(path, repo_root)),
                    source_category=source_category,
                    index=1,
                )
            )
    return rows


def normalize_all_evidence_rows(*, repo_root: Path, final_root: Path | None = None) -> list[dict[str, object]]:
    repo_root = repo_root.resolve()
    final_root = resolve_final_root(repo_root, final_root)
    discovered_path = final_root / "inventories" / "discovered_roots.csv"
    if not discovered_path.exists():
        discover_experiment_roots(repo_root=repo_root, final_root=final_root)
    discovered = read_csv(discovered_path)

    rows: list[dict[str, object]] = []
    for discovered_row in discovered:
        root = repo_path(repo_root, discovered_row["path"])
        if not root.exists() or not root.is_dir():
            continue
        source_category = discovered_row.get("inferred_category") or infer_category(root)
        rows.extend(_root_rows(repo_root, root, source_category))

    write_csv(final_root / "inventories" / "all_evidence_rows.csv", rows, fieldnames=EVIDENCE_SCHEMA)
    write_parquet_if_possible(final_root / "inventories" / "all_evidence_rows.csv", rows)
    # These are long-form aliases used by downstream validation/reporting.
    write_csv(final_root / "inventories" / "all_runs_long.csv", rows, fieldnames=EVIDENCE_SCHEMA)
    write_parquet_if_possible(final_root / "inventories" / "all_runs_long.csv", rows)
    write_csv(final_root / "inventories" / "all_metrics_long.csv", rows, fieldnames=EVIDENCE_SCHEMA)
    write_parquet_if_possible(final_root / "inventories" / "all_metrics_long.csv", rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--final-root", type=Path)
    args = parser.parse_args()
    rows = normalize_all_evidence_rows(repo_root=args.repo_root, final_root=args.final_root)
    print(json.dumps({"evidence_rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
