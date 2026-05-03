#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.dataset_root
    sums = root / "checksums/HF_DATASET_SHA256SUMS"
    if not sums.exists():
        sums = root / "checksums/SHA256SUMS"
    if not sums.exists():
        raise SystemExit(f"missing checksum file: {sums}")
    checked = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip()
        path = root / rel
        if not path.exists():
            raise SystemExit(f"checksum target missing: {rel}")
        observed = sha256(path)
        if observed != expected:
            raise SystemExit(f"checksum mismatch for {rel}: expected={expected} observed={observed}")
        checked += 1
    print(f"checksum_check=passed files={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
