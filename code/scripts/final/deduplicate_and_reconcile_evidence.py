#!/usr/bin/env python3
"""Deduplicate and reconcile full evidence rows without dropping conflicts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.full_evidence_common import EVIDENCE_SCHEMA, read_csv, resolve_final_root, row_hash, write_csv


METRIC_FIELDS = [
    "model_latency_ms",
    "invalid_action_rate",
    "task_success",
    "reward",
    "end_to_end_p99_ms",
    "queue_wait_share_p99",
    "pass_rate",
    "failure_rate",
]


def deduplicate_and_reconcile_evidence(*, final_root: Path) -> dict[str, object]:
    final_root = final_root.resolve()
    rows = read_csv(final_root / "inventories" / "all_evidence_rows.csv")
    duplicate_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []
    seen_hashes: dict[str, str] = {}
    logical_seen: dict[str, dict[str, str]] = {}

    for row in rows:
        digest = row.get("artifact_row_hash") or row_hash(row)
        row["artifact_row_hash"] = digest
        if digest in seen_hashes:
            duplicate_rows.append(
                {
                    "duplicate_row_id": row.get("row_id", ""),
                    "canonical_row_id": seen_hashes[digest],
                    "artifact_row_hash": digest,
                    "dedup_action": "exact_duplicate_kept_in_all_rows",
                }
            )
        else:
            seen_hashes[digest] = row.get("row_id", "")

        key = row.get("logical_cell_key") or "|".join(
            row.get(field, "") for field in ("family", "task_id", "instance_id", "driver_id", "model_id", "backend_engine", "regime", "seed")
        )
        previous = logical_seen.get(key)
        if previous is None:
            logical_seen[key] = row
            continue
        changed = [
            field
            for field in METRIC_FIELDS
            if row.get(field, "") and previous.get(field, "") and row.get(field, "") != previous.get(field, "")
        ]
        if changed:
            conflict_rows.append(
                {
                    "logical_cell_key": key,
                    "first_row_id": previous.get("row_id", ""),
                    "second_row_id": row.get("row_id", ""),
                    "conflicting_fields": ";".join(changed),
                    "resolution": "kept_all_rows_paper_surface_uses_precedence",
                }
            )
        elif row.get("timestamp", "") != previous.get("timestamp", ""):
            duplicate_rows.append(
                {
                    "duplicate_row_id": row.get("row_id", ""),
                    "canonical_row_id": previous.get("row_id", ""),
                    "artifact_row_hash": digest,
                    "dedup_action": "repeated_run_kept",
                }
            )

    write_csv(final_root / "inventories" / "duplicate_report.csv", duplicate_rows)
    write_csv(final_root / "inventories" / "conflict_report.csv", conflict_rows)
    write_csv(final_root / "inventories" / "all_evidence_rows.csv", rows, fieldnames=EVIDENCE_SCHEMA)
    return {"duplicate_count": len(duplicate_rows), "conflict_count": len(conflict_rows), "row_count": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deduplicate_and_reconcile_evidence(final_root=resolve_final_root(Path.cwd(), args.final_root)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
