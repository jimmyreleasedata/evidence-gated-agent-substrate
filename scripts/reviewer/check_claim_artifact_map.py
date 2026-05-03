#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


STALE = [
    "targeted_" + "attribution_audit",
    "PLACEHOLDER_ANON_" + "REMOTE_URL",
    "PLACEHOLDER_ANON_" + "DATASET_URL",
    "ANON_" + "CANONICAL_ROOT",
]


def resolve(location: str, artifact_path: str, code_root: Path, dataset_root: Path) -> Path:
    if location == "github_repo":
        return code_root / artifact_path
    if location == "hf_dataset":
        return dataset_root / artifact_path
    if location == "supplementary_zip":
        return dataset_root / artifact_path
    raise SystemExit(f"invalid location={location!r} for artifact_path={artifact_path}")


def check_map(path: Path, code_root: Path, dataset_root: Path) -> int:
    if not path.exists():
        raise SystemExit(f"missing claim map: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"paper_claim", "location", "artifact_path", "evidence_summary", "expected_value", "validation_script"}
    if not rows:
        raise SystemExit(f"empty claim map: {path}")
    if not required <= set(rows[0]):
        raise SystemExit(f"claim map missing columns: {sorted(required - set(rows[0]))}")
    unresolved = []
    stale = []
    for row in rows:
        combined = " ".join(row.values())
        for marker in STALE:
            if marker in combined:
                stale.append((row["paper_claim"], marker))
        target = resolve(row["location"], row["artifact_path"], code_root, dataset_root)
        if not target.exists():
            unresolved.append((row["paper_claim"], row["location"], row["artifact_path"]))
        script = row.get("validation_script", "")
        if script and not (code_root / script).exists():
            unresolved.append((row["paper_claim"], "validation_script", script))
    if stale:
        raise SystemExit(f"stale claim-map markers: {stale}")
    if unresolved:
        raise SystemExit(f"unresolved claim-map artifacts: {unresolved}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    args = parser.parse_args()
    code_map = args.code_root / "validation/paper_claim_to_artifact_map.csv"
    dataset_map = args.dataset_root / "validation/paper_claim_to_artifact_map.csv"
    code_rows = check_map(code_map, args.code_root, args.dataset_root)
    dataset_rows = check_map(dataset_map, args.code_root, args.dataset_root)
    print(f"claim_artifact_map_check=passed github_rows={code_rows} hf_rows={dataset_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
