#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
import sys

def main() -> int:
    parser = argparse.ArgumentParser(description="Download the public artifact dataset.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    repo_url = f"https://huggingface.co/datasets/{args.repo}"
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=args.out)
        print(f"downloaded dataset to {args.out}")
        return 0
    except Exception as exc:
        print(f"huggingface_hub download unavailable: {exc}")
        print(f"manual fallback: git clone {repo_url} {args.out}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
