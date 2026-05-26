#!/usr/bin/env python3
"""Compatibility wrapper for manifest commands."""
# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from skulk_weights_publisher.cli import run  # noqa: E402


if __name__ == "__main__":
    args = sys.argv[1:]
    global_args: list[str] = []
    while args and args[0] in {"--config", "--manifest"}:
        if len(args) < 2:
            print(f"{args[0]} requires a path", file=sys.stderr)
            raise SystemExit(2)
        global_args.extend(args[:2])
        args = args[2:]
    raise SystemExit(run([*global_args, "catalog", *args]))
