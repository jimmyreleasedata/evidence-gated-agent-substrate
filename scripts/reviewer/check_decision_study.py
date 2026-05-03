#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def rows(path: Path) -> int:
    if not path.exists():
        raise SystemExit(f"missing required artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.dataset_root
    fixed = root / "data/decision_study/fixed_budget_slice"
    fixed_admitted = rows(fixed / "decision_slice_admitted.csv")
    fixed_blocked = rows(fixed / "decision_slice_blocked.csv")
    if fixed_admitted != 56 or fixed_blocked != 0:
        raise SystemExit(f"unexpected fixed-budget slice counts: admitted={fixed_admitted} blocked={fixed_blocked}")
    budget = root / "data/decision_study/budget_grid_study"
    gate_path = budget / "gate_report.json"
    if not gate_path.exists():
        raise SystemExit(f"missing required artifact: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if int(gate.get("expected_rows", -1)) != 336 or int(gate.get("admitted_rows", -1)) != 336 or int(gate.get("excluded_rows", -1)) != 0:
        raise SystemExit(f"unexpected budget-grid gate values: {gate}")
    claim_path = budget / "claim_matrix.csv"
    if not claim_path.exists():
        raise SystemExit(f"missing required artifact: {claim_path}")
    with claim_path.open(newline="", encoding="utf-8") as handle:
        claim_rows = list(csv.DictReader(handle))
    if not claim_rows:
        raise SystemExit("empty budget-grid claim matrix")
    row = claim_rows[0]
    comparable = int(float(row.get("comparable_cells", -1)))
    reversal = int(float(row.get("reversal_cells", -1)))
    if comparable != 12 or reversal != 12:
        raise SystemExit(f"unexpected decision cells: comparable={comparable} reversal={reversal}")
    print("decision_study_check=passed fixed=56/56 budget_grid=336/336 comparable=12 reversal=12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
