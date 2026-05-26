"""Publication planning and execution helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from skulk_weights_publisher.defaults import COLLECTION_ENV_VAR
from skulk_weights_publisher.manifest import HF_COLLECTION_PATTERN, ManifestEntry


class PublishError(RuntimeError):
    """Raised when a publish command cannot be executed safely."""


@dataclass(frozen=True)
class PublishPlan:
    """Concrete commands and paths for publishing one catalogue entry."""

    entry: ManifestEntry
    scratch_root: Path
    output_path: Path
    extract_command: tuple[str, ...]
    publish_command: tuple[str, ...]
    collection_slug: str | None

    def summary_lines(self, *, force: bool, artifact: str = "all") -> tuple[str, ...]:
        """Return the human-readable command summary printed before execution."""

        collection_line = (
            f"collection: https://huggingface.co/collections/{self.collection_slug}"
            if self.collection_slug
            else "collection: disabled"
        )
        lines: list[str] = [
            f"model key: {self.entry.key}",
            f"tier: {self.entry.tier}",
            f"artifact: {artifact}",
            f"source model: {self.entry.source_model}",
            f"output path: {self.output_path}",
            f"target repo: hf://{self.entry.hf_repo}",
            collection_line,
            f"publish slices: {self.entry.publish_slices}",
            f"force overwrite: {int(force)}",
        ]
        if artifact in ("all", "vindex"):
            lines += [
                f"extract command:{format_command(self.extract_command)}",
                f"publish command:{format_command(self.publish_command)}",
            ]
        if artifact in ("all", "mtp"):
            lines.append("mtp step: not yet implemented")
        if artifact in ("all", "vision"):
            lines.append("vision step: not yet implemented")
        return tuple(lines)


def format_command(command: Sequence[str]) -> str:
    """Return a shell-readable representation of a command tuple."""

    import shlex

    return "".join(f" {shlex.quote(part)}" for part in command)


def build_publish_plan(
    entry: ManifestEntry,
    *,
    scratch_root: Path | None = None,
    collection_slug: str | None = None,
    use_entry_collection: bool = True,
) -> PublishPlan:
    """Build deterministic LARQL extraction and publication commands."""

    resolved_scratch = scratch_root or Path.cwd() / ".scratch"
    resolved_collection = (
        entry.hf_collection
        if collection_slug is None and use_entry_collection
        else collection_slug
    )
    output_path = resolved_scratch / entry.output_name
    extract_command = (
        "larql",
        "extract",
        entry.source_model,
        "-o",
        str(output_path),
        "--quant",
        entry.quant,
    )
    publish_command = (
        "larql",
        "publish",
        str(output_path),
        "--repo",
        entry.hf_repo,
        "--slices",
        entry.publish_slices,
    )
    return PublishPlan(
        entry=entry,
        scratch_root=resolved_scratch,
        output_path=output_path,
        extract_command=extract_command,
        publish_command=publish_command,
        collection_slug=resolved_collection,
    )


def resolve_publish_collection(
    entry: ManifestEntry,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the collection target for a publish request."""

    env = os.environ if environ is None else environ
    override = env.get(COLLECTION_ENV_VAR)
    if override is None or not override.strip():
        return entry.hf_collection
    normalized_override = override.strip()
    if normalized_override.lower() in {"0", "false", "no", "none", "off", "disabled"}:
        return None
    if not HF_COLLECTION_PATTERN.fullmatch(normalized_override):
        raise PublishError(
            f"{COLLECTION_ENV_VAR} must look like owner/slug or be 'none'"
        )
    return normalized_override


def default_scratch_root(environ: Mapping[str, str] | None = None) -> Path:
    """Resolve the scratch root from environment or the current checkout."""

    env = os.environ if environ is None else environ
    configured = env.get("SKULK_WEIGHTS_SCRATCH")
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / ".scratch"


def execute_publish_plan(
    plan: PublishPlan,
    *,
    dry_run: bool,
    force: bool,
    artifact: str = "all",
    environ: Mapping[str, str] | None = None,
) -> None:
    """Execute or dry-run the extraction and publication plan."""

    env = os.environ if environ is None else environ
    if dry_run:
        return
    if shutil.which("larql") is None:
        raise PublishError("larql is required for non-dry-run publishing")
    if not env.get("HF_TOKEN"):
        raise PublishError("HF_TOKEN is required for non-dry-run publishing")

    if artifact in ("all", "vindex"):
        plan.scratch_root.mkdir(parents=True, exist_ok=True)
        if plan.output_path.exists():
            if not force:
                raise PublishError(
                    f"output path already exists: {plan.output_path}\n"
                    "remove it manually or rerun with --force to replace it"
                )
            shutil.rmtree(plan.output_path)
        subprocess.run(plan.extract_command, check=True)
        subprocess.run(plan.publish_command, check=True)
        if plan.collection_slug is not None:
            add_vindex_to_collection(
                plan.collection_slug,
                plan.entry.hf_repo,
                token=env.get("HF_TOKEN"),
            )


def add_vindex_to_collection(
    collection_slug: str,
    repo_id: str,
    *,
    token: str | None,
) -> None:
    """Add a published vindex model repo to its Hugging Face collection."""

    try:
        from huggingface_hub import add_collection_item

        add_collection_item(
            collection_slug,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
            token=token,
        )
    except Exception as exc:
        raise PublishError(
            f"failed to add {repo_id} to collection {collection_slug}: {exc}"
        ) from exc
