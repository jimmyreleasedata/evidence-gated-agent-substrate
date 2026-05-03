#!/usr/bin/env python3
import argparse
from pathlib import Path


def require_nonempty_dir(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"missing directory: {path}")
    if not any(child.is_file() for child in path.rglob("*")):
        raise SystemExit(f"directory has no files: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.dataset_root
    require_nonempty_dir(root / "data/figure_inputs")
    require_nonempty_dir(root / "data/table_inputs")
    required = [
        "metadata/croissant_completed_for_openreview.json",
        "supplementary_zip/neurips_ed_anonymous_supplementary.zip",
    ]
    for rel in required:
        path = root / rel
        if not path.exists():
            raise SystemExit(f"missing required artifact: {path}")
    print("table_input_check=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
