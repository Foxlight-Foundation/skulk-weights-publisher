"""Publication planning and execution helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from skulk_vindex_publisher.manifest import ManifestEntry


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

    def summary_lines(self, *, force: bool) -> tuple[str, ...]:
        """Return the human-readable command summary printed before execution."""

        return (
            f"model key: {self.entry.key}",
            f"tier: {self.entry.tier}",
            f"source model: {self.entry.source_model}",
            f"output path: {self.output_path}",
            f"target repo: hf://{self.entry.hf_repo}",
            f"publish slices: {self.entry.publish_slices}",
            f"force overwrite: {int(force)}",
            f"extract command:{format_command(self.extract_command)}",
            f"publish command:{format_command(self.publish_command)}",
        )


def format_command(command: Sequence[str]) -> str:
    """Return a shell-readable representation of a command tuple."""

    import shlex

    return "".join(f" {shlex.quote(part)}" for part in command)


def build_publish_plan(
    entry: ManifestEntry,
    *,
    scratch_root: Path | None = None,
) -> PublishPlan:
    """Build deterministic LARQL extraction and publication commands."""

    resolved_scratch = scratch_root or Path.cwd() / ".scratch"
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
    )


def default_scratch_root(environ: Mapping[str, str] | None = None) -> Path:
    """Resolve the scratch root from environment or the current checkout."""

    env = os.environ if environ is None else environ
    configured = env.get("SKULK_VINDEX_SCRATCH")
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / ".scratch"


def execute_publish_plan(
    plan: PublishPlan,
    *,
    dry_run: bool,
    force: bool,
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
