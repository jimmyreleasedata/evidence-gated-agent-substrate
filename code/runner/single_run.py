#!/usr/bin/env python3
"""Single-run entrypoint placeholder for benchmark-suite."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single benchmark task.")
    parser.add_argument("--family", required=True, help="Benchmark family name.")
    parser.add_argument("--task-id", required=True, help="Task identifier.")
    parser.add_argument("--config", help="Optional configuration file.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print(
        "single_run placeholder",
        {"family": args.family, "task_id": args.task_id, "config": args.config},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
