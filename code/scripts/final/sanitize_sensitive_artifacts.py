#!/usr/bin/env python3
"""Sensitive-artifact detection for the final evidence package."""

from __future__ import annotations

import argparse
from pathlib import Path


SENSITIVE_NAME_FRAGMENTS = {
    "REDACTED_BROWSER_STATE_LABEL",
    "REDACTED_BROWSER_STATE_LABEL",
    "REDACTED_BROWSER_STATE_LABEL",
    "credential",
    "credentials",
    "credential",
    "passwd",
    "secret",
    "token",
    "hf_token",
}


def is_sensitive_artifact(path: Path) -> bool:
    lowered = str(path).lower()
    name = path.name.lower()
    if name in {".netrc", ".huggingface", "headers_with_credentials.json"}:
        return True
    return any(fragment in lowered for fragment in SENSITIVE_NAME_FRAGMENTS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(f"{path}\t{'sensitive' if is_sensitive_artifact(path) else 'public'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
