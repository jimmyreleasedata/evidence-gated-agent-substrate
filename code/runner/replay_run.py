#!/usr/bin/env python3
"""Replay-run entrypoint placeholder for benchmark-suite."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a captured benchmark run.")
    parser.add_argument("--family", required=True, help="Benchmark family name.")
    parser.add_argument("--trace", required=True, help="Trace JSONL path.")
    parser.add_argument(
        "--replay-class",
        required=True,
        choices=("R0", "R1", "R2"),
        help="Replay class for the run.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print(
        "replay_run placeholder",
        {
            "family": args.family,
            "trace": args.trace,
            "replay_class": args.replay_class,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
