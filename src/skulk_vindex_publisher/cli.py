"""Command line interface for the Skulk vindex publisher."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from skulk_vindex_publisher.catalogue import (
    CatalogueView,
    filter_catalogue_entries,
    find_catalogue_entry,
    load_catalogue_view,
    write_default_config,
)
from skulk_vindex_publisher.doctor import run_doctor
from skulk_vindex_publisher.manifest import (
    ManifestError,
)
from skulk_vindex_publisher.publisher import (
    PublishError,
    build_publish_plan,
    default_scratch_root,
    execute_publish_plan,
)


def _catalogue_view_from_args(args: argparse.Namespace) -> CatalogueView:
    config_path = Path(args.config) if args.config else None
    manifest_path = Path(args.manifest) if args.manifest else None
    return load_catalogue_view(config_path=config_path, manifest_path=manifest_path)


def _cmd_catalogue_validate(args: argparse.Namespace) -> int:
    view = _catalogue_view_from_args(args)
    print(
        f"catalogue valid: {len(view.entries)} entries from "
        f"{len(view.sources)} sources"
    )
    return 0


def _cmd_catalogue_sources(args: argparse.Namespace) -> int:
    view = _catalogue_view_from_args(args)
    for source in view.sources:
        print(
            f"{source.kind} {source.name} "
            f"namespace={source.namespace or '-'} "
            f"hf_owner={source.hf_owner or '-'} "
            f"entries={len(source.entries)} "
            f"origin={source.origin}"
        )
    return 0


def _cmd_catalogue_get(args: argparse.Namespace) -> int:
    view = _catalogue_view_from_args(args)
    entry = find_catalogue_entry(args.key, view)
    print(entry.to_json())
    return 0


def _cmd_catalogue_list(args: argparse.Namespace) -> int:
    view = _catalogue_view_from_args(args)
    entries = filter_catalogue_entries(view, args.tier)
    for entry in entries:
        print(entry.key)
    return 0


def _cmd_catalogue_init(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    write_default_config(output_path, force=args.force)
    print(f"wrote {output_path}")
    return 0


def _cmd_manifest_validate(args: argparse.Namespace) -> int:
    return _cmd_catalogue_validate(args)


def _cmd_manifest_get(args: argparse.Namespace) -> int:
    return _cmd_catalogue_get(args)


def _cmd_manifest_list(args: argparse.Namespace) -> int:
    return _cmd_catalogue_list(args)


def _cmd_publish(args: argparse.Namespace) -> int:
    entry = find_catalogue_entry(args.model, _catalogue_view_from_args(args))
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
    config_path = Path(args.config) if args.config else None
    manifest_path = Path(args.manifest) if args.manifest else None
    report = run_doctor(
        publish=args.publish,
        config_path=config_path,
        manifest_path=manifest_path,
    )
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
        description="Build, validate, and publish Skulk LARQL vindexes.",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--config",
        help=(
            "Path to skulk-vindex.yaml. If omitted, skulk-vindex.yaml is used "
            "when present; otherwise the built-in Foxlight catalogue is used."
        ),
    )
    source_group.add_argument(
        "--manifest",
        help="Legacy path to one vindex manifest.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalogue_parser = subparsers.add_parser(
        "catalogue",
        help="Validate/query merged catalogue sources",
    )
    catalogue_subparsers = catalogue_parser.add_subparsers(
        dest="catalogue_command", required=True
    )
    catalogue_validate_parser = catalogue_subparsers.add_parser("validate")
    catalogue_validate_parser.set_defaults(func=_cmd_catalogue_validate)

    catalogue_sources_parser = catalogue_subparsers.add_parser("sources")
    catalogue_sources_parser.set_defaults(func=_cmd_catalogue_sources)

    catalogue_get_parser = catalogue_subparsers.add_parser("get")
    catalogue_get_parser.add_argument("--key", required=True)
    catalogue_get_parser.set_defaults(func=_cmd_catalogue_get)

    catalogue_list_parser = catalogue_subparsers.add_parser("list")
    catalogue_list_parser.add_argument(
        "--tier", choices=["all", "smoke", "moe"], default="all"
    )
    catalogue_list_parser.set_defaults(func=_cmd_catalogue_list)

    catalogue_init_parser = catalogue_subparsers.add_parser("init")
    catalogue_init_parser.add_argument("--output", default="skulk-vindex.yaml")
    catalogue_init_parser.add_argument("--force", action="store_true")
    catalogue_init_parser.set_defaults(func=_cmd_catalogue_init)

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Compatibility alias for catalogue commands",
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

    publish_parser = subparsers.add_parser(
        "publish", help="Publish one catalogue entry"
    )
    publish_parser.add_argument(
        "--model",
        required=True,
        help="Catalogue key to publish.",
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
