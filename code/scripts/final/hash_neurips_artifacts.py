#!/usr/bin/env python3
"""Hash artifacts listed in a final evidence package inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_inventory(inventory_csv: Path, output_path: Path) -> int:
    count = 0
    lines: list[str] = []
    with inventory_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source = Path(row.get("source_path", ""))
            if source.exists() and source.is_file():
                lines.append(f"{sha256_file(source)}  {row.get('relative_path') or source}")
                count += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sorted(lines)) + ("\n" if lines else ""), encoding="utf-8")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hash_inventory(args.inventory_csv, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
