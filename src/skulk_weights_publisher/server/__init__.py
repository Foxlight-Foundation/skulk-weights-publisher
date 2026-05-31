"""skulk-ui entry point — local web GUI for Skulk Weights Publisher."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def _ui_root() -> Path:
    """Return the path to the ui/ directory relative to this package."""
    return Path(__file__).parent.parent.parent.parent / "ui"


def _dist_root() -> Path:
    return _ui_root() / "dist"


def _ensure_built() -> None:
    """Build the React app if ui/dist/index.html is missing."""
    dist = _dist_root()
    if (dist / "index.html").is_file():
        return

    ui = _ui_root()
    if not ui.is_dir():
        print(
            "skulk-ui: cannot find the ui/ directory.\n"
            "Run skulk-ui from the skulk-weights-publisher source tree.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not shutil.which("yarn"):
        print(
            "skulk-ui: ui/dist/ not found and yarn is not available.\n"
            "Install Node.js 18+ and Yarn, then re-run skulk-ui — "
            "it will build automatically.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("skulk-ui: building UI for the first time (this takes ~30 s)…")

    install = subprocess.run(["yarn", "install"], cwd=str(ui))
    if install.returncode != 0:
        print("skulk-ui: yarn install failed", file=sys.stderr)
        sys.exit(1)

    build = subprocess.run(["yarn", "build"], cwd=str(ui))
    if build.returncode != 0:
        print("skulk-ui: yarn build failed", file=sys.stderr)
        sys.exit(1)

    print("skulk-ui: UI built successfully.")


def main() -> None:
    """Launch the skulk-ui local web server and open the browser."""
    parser = argparse.ArgumentParser(
        prog="skulk-ui",
        description="Local web GUI for Skulk Weights Publisher.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7842,
        help="Port to listen on (default: 7842).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    args = parser.parse_args()

    try:
        import uvicorn  # type: ignore[import-untyped]
    except ImportError:
        print(
            "skulk-ui requires the [ui] extras:\n"
            "  uv sync --extra ui",
            file=sys.stderr,
        )
        sys.exit(1)

    _ensure_built()

    from skulk_weights_publisher.server.app import app

    if not args.no_open:
        def _open_browser() -> None:
            import webbrowser
            time.sleep(1.2)
            webbrowser.open(f"http://localhost:{args.port}")

        threading.Thread(target=_open_browser, daemon=True).start()

    print(f"skulk-ui: http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
