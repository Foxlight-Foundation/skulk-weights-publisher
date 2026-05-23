"""Command line interface for the Skulk vindex publisher."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from skulk_vindex_publisher.doctor import run_doctor
from skulk_vindex_publisher.manifest import (
    ManifestError,
    find_entry,
    list_entries,
    validate_manifest,
)
from skulk_vindex_publisher.publisher import (
    PublishError,
    build_publish_plan,
    default_scratch_root,
    execute_publish_plan,
)


def _cmd_manifest_validate(args: argparse.Namespace) -> int:
    entries = validate_manifest(Path(args.manifest))
    print(f"{args.manifest} valid: {len(entries)} entries")
    return 0


def _cmd_manifest_get(args: argparse.Namespace) -> int:
    entry = find_entry(args.key, Path(args.manifest))
    print(entry.to_json())
    return 0


def _cmd_manifest_list(args: argparse.Namespace) -> int:
    entries = list_entries(args.tier, Path(args.manifest))
    for entry in entries:
        print(entry.key)
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    entry = find_entry(args.model, Path(args.manifest))
    scratch_root = (
        Path(args.scratch).expanduser() if args.scratch else default_scratch_root()
    )
    plan = build_publish_plan(entry, scratch_root=scratch_root)
    for line in plan.summary_lines(force=args.force):
        print(line)
    if args.dry_run:
        print("dry run complete")
        return 0
    execute_publish_plan(plan, dry_run=False, force=args.force)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    report = run_doctor(publish=args.publish, manifest_path=Path(args.manifest))
    for check in report.checks:
        stream = sys.stdout if check.ok else sys.stderr
        print(check.message, file=stream)
    if not report.ok:
        return 1
    print("doctor checks passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""

    parser = argparse.ArgumentParser(
        prog="skulk-vindex",
        description="Build, validate, and publish Skulk LARQL vindex artifacts.",
    )
    parser.add_argument(
        "--manifest",
        default="models.yaml",
        help="Path to the vindex catalogue manifest.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Validate/query models.yaml",
    )
    manifest_subparsers = manifest_parser.add_subparsers(
        dest="manifest_command", required=True
    )
    validate_parser = manifest_subparsers.add_parser("validate")
    validate_parser.set_defaults(func=_cmd_manifest_validate)

    get_parser = manifest_subparsers.add_parser("get")
    get_parser.add_argument("--key", required=True)
    get_parser.set_defaults(func=_cmd_manifest_get)

    list_parser = manifest_subparsers.add_parser("list")
    list_parser.add_argument("--tier", choices=["all", "smoke", "moe"], default="all")
    list_parser.set_defaults(func=_cmd_manifest_list)

    publish_parser = subparsers.add_parser("publish", help="Publish one manifest entry")
    publish_parser.add_argument(
        "--model",
        required=True,
        help="Manifest key to publish.",
    )
    publish_parser.add_argument("--dry-run", action="store_true")
    publish_parser.add_argument("--force", action="store_true")
    publish_parser.add_argument("--scratch", help="Override SKULK_VINDEX_SCRATCH.")
    publish_parser.set_defaults(func=_cmd_publish)

    doctor_parser = subparsers.add_parser("doctor", help="Check local prerequisites")
    doctor_parser.add_argument("--publish", action="store_true")
    doctor_parser.set_defaults(func=_cmd_doctor)

    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ManifestError, PublishError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def main() -> None:
    """Console script entry point."""

    raise SystemExit(run())


if __name__ == "__main__":
    main()
