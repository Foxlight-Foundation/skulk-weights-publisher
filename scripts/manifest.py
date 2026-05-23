#!/usr/bin/env python3
"""Compatibility wrapper for manifest commands."""
# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from skulk_vindex_publisher.cli import run  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run(["manifest", *sys.argv[1:]]))
