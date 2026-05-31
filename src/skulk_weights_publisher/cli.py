"""Command line interface for the Skulk Weights Publisher."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from skulk_weights_publisher.catalogue import (
    CatalogueView,
    filter_catalogue_entries,
    find_catalogue_entry,
    load_catalogue_view,
    write_default_config,
)
from skulk_weights_publisher.doctor import run_doctor
from skulk_weights_publisher.manifest import (
    ManifestError,
)
from skulk_weights_publisher.mtp_extractor import MtpExtractionError
from skulk_weights_publisher.publisher import (
    PublishError,
    build_publish_plan,
    default_scratch_root,
    execute_publish_plan,
    resolve_publish_collection,
)

ARTIFACT_CHOICES = ["vindex", "mtp", "vision", "all"]


def _catalogue_view_from_args(args: argparse.Namespace) -> CatalogueView:
    config_path = Path(args.config) if args.config else None
    manifest_path = Path(args.manifest) if args.manifest else None
    return load_catalogue_view(config_path=config_path, manifest_path=manifest_path)


def _cmd_catalogue_validate(args: argparse.Namespace) -> int:
    view = _catalogue_view_from_args(args)
    print(
        f"catalog valid: {len(view.entries)} entries from {len(view.sources)} sources"
    )
    return 0


def _cmd_catalogue_sources(args: argparse.Namespace) -> int:
    view = _catalogue_view_from_args(args)
    for source in view.sources:
        print(
            f"{source.kind} {source.name} "
            f"namespace={source.namespace or '-'} "
            f"hf_owner={source.hf_owner or '-'} "
            f"hf_collection={source.hf_collection or '-'} "
            f"entries={len(source.entries)} "
            f"origin={source.origin}"
        )
    return 0


def _cmd_catalogue_show(args: argparse.Namespace) -> int:
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


def _cmd_catalog_add(args: argparse.Namespace) -> int:
    import os

    from skulk_weights_publisher.catalog_adder import (
        CatalogAddError,
        build_entry_block,
        derive_key_slug,
        detect_base_model,
        detect_mtp_keys,
        detect_quant,
        detect_tier,
        fetch_hf_model_info,
        find_builtin_catalog_path,
        parse_hf_model_id,
    )

    try:
        from skulk_weights_publisher.catalogue import load_catalogue_view
        from skulk_weights_publisher.manifest import ALLOWED_QUANTS

        model_id = parse_hf_model_id(args.model)
        token = os.environ.get("HF_TOKEN")
        print(f"fetching metadata for {model_id}...")
        info = fetch_hf_model_info(model_id, token=token)
        base_model = detect_base_model(info)
        quant = detect_quant(info)
        if quant not in ALLOWED_QUANTS:
            raise CatalogAddError(
                f"detected quant '{quant}' is not supported "
                f"(allowed: {', '.join(sorted(ALLOWED_QUANTS))})"
            )
        tier = detect_tier(info)
        mtp_keys: list[str] = []
        if base_model:
            print(f"checking {base_model} for MTP keys...")
            mtp_keys = detect_mtp_keys(base_model, token=token)
        key_slug = derive_key_slug(model_id, quant)
        effective_key = f"foxlight/{key_slug}"
        hf_repo_new = f"FoxlightAI/{key_slug}-vindex"
        view = load_catalogue_view()
        existing_keys = {e.key for e in view.entries}
        if effective_key in existing_keys:
            raise CatalogAddError(
                f"'{effective_key}' already exists in the catalog"
            )
        existing_hf_repos = {e.hf_repo for e in view.entries}
        if hf_repo_new in existing_hf_repos:
            raise CatalogAddError(
                f"hf_repo '{hf_repo_new}' already exists in the catalog"
            )
        entry_block = build_entry_block(
            key_slug=key_slug,
            source_model=model_id,
            quant=quant,
            tier=tier,
            base_model=base_model,
            mtp_keys=mtp_keys,
        )
        print("\n--- entry preview ---")
        print(entry_block)
        if mtp_keys:
            print(f"detected {len(mtp_keys)} MTP tensor keys in {base_model}")
        else:
            print("no MTP keys detected — mtp fields omitted")
        if args.dry_run:
            print("--- dry run: nothing written ---")
            return 0
        catalog_path = find_builtin_catalog_path()
        if not args.yes:
            answer = input(f"\nAppend to {catalog_path}? [y/N] ").strip().lower()
            if answer != "y":
                print("aborted")
                return 0
        with open(catalog_path, "a", encoding="utf-8") as f:
            f.write(entry_block)
        print(f"added {key_slug} to {catalog_path}")
        print(
            "remember to update test entry counts in "
            "tests/test_cli.py and tests/test_catalogue.py"
        )
        return 0
    except CatalogAddError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _cmd_publish(args: argparse.Namespace) -> int:
    artifact = getattr(args, "artifact", None) or "all"
    entry = find_catalogue_entry(args.model, _catalogue_view_from_args(args))
    scratch_root = (
        Path(args.scratch).expanduser() if args.scratch else default_scratch_root()
    )
    collection_slug = resolve_publish_collection(entry)
    plan = build_publish_plan(
        entry,
        scratch_root=scratch_root,
        collection_slug=collection_slug,
        use_entry_collection=False,
    )
    for line in plan.summary_lines(force=args.force, artifact=artifact):
        print(line)
    if args.dry_run:
        print("dry run complete")
        return 0
    execute_publish_plan(plan, dry_run=False, force=args.force, artifact=artifact)
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
        prog="skulk-weights",
        description="Build and publish weight artifacts for Skulk.",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--config",
        help=(
            "Path to skulk-weights.yaml. If omitted, skulk-weights.yaml is used "
            "when present; otherwise the built-in Foxlight catalog is used."
        ),
    )
    source_group.add_argument(
        "--manifest",
        help="Legacy path to one manifest file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # catalog subcommand
    catalogue_parser = subparsers.add_parser(
        "catalog",
        help="Validate and query merged catalog sources.",
    )
    catalogue_subparsers = catalogue_parser.add_subparsers(
        dest="catalogue_command", required=True
    )

    catalogue_validate_parser = catalogue_subparsers.add_parser(
        "validate", help="Validate all catalog sources."
    )
    catalogue_validate_parser.set_defaults(func=_cmd_catalogue_validate)

    catalogue_sources_parser = catalogue_subparsers.add_parser(
        "sources", help="List catalog sources."
    )
    catalogue_sources_parser.set_defaults(func=_cmd_catalogue_sources)

    catalogue_show_parser = catalogue_subparsers.add_parser(
        "show", help="Print a single resolved catalog entry as JSON."
    )
    catalogue_show_parser.add_argument("key", help="Fully-qualified catalog key.")
    catalogue_show_parser.set_defaults(func=_cmd_catalogue_show)

    catalogue_list_parser = catalogue_subparsers.add_parser(
        "list", help="List catalog keys, optionally filtered by tier."
    )
    catalogue_list_parser.add_argument(
        "--tier", choices=["all", "smoke", "moe"], default="all"
    )
    catalogue_list_parser.set_defaults(func=_cmd_catalogue_list)

    catalogue_init_parser = catalogue_subparsers.add_parser(
        "init", help="Write a starter skulk-weights.yaml operator config."
    )
    catalogue_init_parser.add_argument(
        "--output",
        default="skulk-weights.yaml",
        help="Output path (default: skulk-weights.yaml).",
    )
    catalogue_init_parser.add_argument("--force", action="store_true")
    catalogue_init_parser.set_defaults(func=_cmd_catalogue_init)

    catalogue_add_parser = catalogue_subparsers.add_parser(
        "add", help="Auto-detect and add a HF model to the Foxlight catalog."
    )
    catalogue_add_parser.add_argument(
        "model", help="HuggingFace model URL or owner/repo."
    )
    catalogue_add_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the entry without writing to disk.",
    )
    catalogue_add_parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt."
    )
    catalogue_add_parser.set_defaults(func=_cmd_catalog_add)

    # publish subcommand
    publish_parser = subparsers.add_parser(
        "publish",
        help="Publish weight artifacts for one catalog entry.",
    )
    publish_parser.add_argument(
        "--model",
        required=True,
        help="Fully-qualified catalog key to publish.",
    )
    publish_parser.add_argument(
        "--artifact",
        choices=ARTIFACT_CHOICES,
        default=None,
        help=(
            "Artifact type to publish. Omit or pass 'all' to publish all "
            "declared artifacts for this entry. Choices: vindex, mtp, vision, all."
        ),
    )
    publish_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the publish plan without executing anything.",
    )
    publish_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output path.",
    )
    publish_parser.add_argument(
        "--scratch",
        help="Override SKULK_WEIGHTS_SCRATCH scratch directory.",
    )
    publish_parser.set_defaults(func=_cmd_publish)

    # doctor subcommand
    doctor_parser = subparsers.add_parser(
        "doctor", help="Check local prerequisites."
    )
    doctor_parser.add_argument(
        "--publish",
        action="store_true",
        help="Also check prerequisites required for non-dry-run publishing.",
    )
    doctor_parser.set_defaults(func=_cmd_doctor)

    return parser


def _normalize_legacy_argv(argv: Sequence[str]) -> list[str]:
    normalized = list(argv)
    skip_next = False
    for index, value in enumerate(normalized):
        if skip_next:
            skip_next = False
            continue
        if value in {"--config", "--manifest"}:
            skip_next = True
            continue
        if value == "catalogue":
            normalized[index] = "catalog"
            break
    return normalized


def run(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    normalized_argv = _normalize_legacy_argv(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalized_argv)
    try:
        return int(args.func(args))
    except (ManifestError, MtpExtractionError, PublishError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def main() -> None:
    """Console script entry point."""

    raise SystemExit(run())


if __name__ == "__main__":
    main()
