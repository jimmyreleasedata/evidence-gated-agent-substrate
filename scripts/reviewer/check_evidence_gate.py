#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing required artifact: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.dataset_root
    surface = require(root / "data/global/paper_facing_surface.csv")
    excluded = require(root / "data/global/excluded_rows.csv")
    report_path = require(root / "data/global/evidence_gate_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    admitted = int(report.get("accepted_rows", report.get("admitted_rows", -1)))
    rejected = int(report.get("rejected_rows", report.get("excluded_rows", -1)))
    surface_rows = count_csv_rows(surface)
    excluded_rows = count_csv_rows(excluded)
    if admitted != 930 or rejected != 1184:
        raise SystemExit(f"unexpected gate counts: admitted={admitted} excluded={rejected}")
    if surface_rows != 930:
        raise SystemExit(f"unexpected paper_facing_surface row count: {surface_rows}")
    if excluded_rows != 1184:
        raise SystemExit(f"unexpected excluded_rows row count: {excluded_rows}")
    print("evidence_gate_check=passed admitted=930 excluded=1184")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
