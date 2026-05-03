#!/usr/bin/env python3
"""Batch-run entrypoint placeholder for benchmark-suite."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a batch benchmark suite.")
    parser.add_argument("--family", required=True, help="Benchmark family name.")
    parser.add_argument("--manifest", required=True, help="Task manifest path.")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrency.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print(
        "batch_run placeholder",
        {
            "family": args.family,
            "manifest": args.manifest,
            "concurrency": args.concurrency,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
