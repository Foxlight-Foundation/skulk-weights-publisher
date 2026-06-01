"""Publication planning and execution helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from skulk_weights_publisher.card_publish import publish_model_card
from skulk_weights_publisher.collection_publish import file_artifact_in_collection
from skulk_weights_publisher.defaults import COLLECTION_ENV_VAR
from skulk_weights_publisher.manifest import HF_COLLECTION_PATTERN, ManifestEntry


class PublishError(RuntimeError):
    """Raised when a publish command cannot be executed safely."""


@dataclass(frozen=True)
class MtpSidecarStep:
    """Parameters for the MTP weight extraction and sidecar publish step."""

    source_repo: str
    sidecar_repo: str
    mtp_quant: str


@dataclass(frozen=True)
class VisionSidecarStep:
    """Parameters for the vision encoder sidecar publish step."""

    source_repo: str
    sidecar_repo: str


@dataclass(frozen=True)
class PublishPlan:
    """Concrete commands and paths for publishing one catalogue entry."""

    entry: ManifestEntry
    scratch_root: Path
    output_path: Path
    extract_command: tuple[str, ...]
    publish_command: tuple[str, ...]
    collection_slug: str | None
    mtp_step: MtpSidecarStep | None = None
    vision_step: VisionSidecarStep | None = None

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
            if self.mtp_step is not None:
                mtp_output = self.scratch_root / (
                    self.mtp_step.sidecar_repo.replace("/", "--") + "-mtp.safetensors"
                )
                lines += [
                    f"mtp source repo:  hf://{self.mtp_step.source_repo}",
                    f"mtp sidecar repo: hf://{self.mtp_step.sidecar_repo}/mtp.safetensors",
                    f"mtp quant:        {self.mtp_step.mtp_quant}",
                    f"mtp output path:  {mtp_output}",
                ]
            else:
                lines.append("mtp step: not configured for this entry")
        if artifact in ("all", "vision"):
            if self.vision_step is not None:
                lines += [
                    f"vision source repo:  hf://{self.vision_step.source_repo}",
                    f"vision sidecar repo: hf://{self.vision_step.sidecar_repo}",
                    "vision step:         mirror weights + configs (no quantization)",
                ]
            else:
                lines.append("vision step: not configured for this entry")
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
    mtp_step = (
        MtpSidecarStep(
            source_repo=entry.mtp_source_repo,
            sidecar_repo=cast(str, entry.mtp_sidecar_repo),
            mtp_quant=cast(str, entry.mtp_quant),
        )
        if entry.mtp_source_repo is not None
        else None
    )
    vision_step = (
        VisionSidecarStep(
            source_repo=entry.vision_source_repo,
            sidecar_repo=cast(str, entry.vision_sidecar_repo),
        )
        if entry.vision_source_repo is not None
        else None
    )
    return PublishPlan(
        entry=entry,
        scratch_root=resolved_scratch,
        output_path=output_path,
        extract_command=extract_command,
        publish_command=publish_command,
        collection_slug=resolved_collection,
        mtp_step=mtp_step,
        vision_step=vision_step,
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
    if not env.get("HF_TOKEN"):
        raise PublishError("HF_TOKEN is required for non-dry-run publishing")

    if artifact not in ("all", "vindex", "mtp", "vision"):
        raise PublishError(
            f"artifact '{artifact}' is not yet implemented for non-dry-run publishing; "
            "use --dry-run to preview the plan"
        )

    if artifact in ("all", "vindex"):
        if shutil.which("larql") is None:
            raise PublishError("larql is required for vindex publishing")
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
            file_artifact_in_collection(
                plan.entry.hf_repo,
                "vindex",
                token=env.get("HF_TOKEN"),
            )
        publish_model_card(
            repo_id=plan.entry.hf_repo,
            artifact_type="vindex",
            source_repo=plan.entry.source_model,
            token=env.get("HF_TOKEN"),
            quant=plan.entry.quant,
            catalog_key=plan.entry.key,
            target_model=plan.entry.source_model,
        )

    if artifact in ("all", "mtp"):
        if plan.mtp_step is None:
            if artifact == "mtp":
                raise PublishError(
                    f"no MTP sidecar configured for {plan.entry.key}; "
                    "add mtp_source_repo, mtp_sidecar_repo, and mtp_quant"
                    " to the catalog entry"
                )
        else:
            from skulk_weights_publisher.mtp_extractor import extract_mtp

            extract_mtp(
                plan.mtp_step.source_repo,
                plan.mtp_step.sidecar_repo,
                plan.mtp_step.mtp_quant,
                plan.scratch_root,
                token=env.get("HF_TOKEN"),
                dry_run=False,
                catalog_key=plan.entry.key,
            )
            if plan.collection_slug is not None:
                file_artifact_in_collection(
                    plan.mtp_step.sidecar_repo,
                    "mtp-sidecar",
                    token=env.get("HF_TOKEN"),
                )

    if artifact in ("all", "vision"):
        if plan.vision_step is None:
            if artifact == "vision":
                raise PublishError(
                    f"no vision sidecar configured for {plan.entry.key}; "
                    "add vision_source_repo and vision_sidecar_repo"
                    " to the catalog entry"
                )
        else:
            from skulk_weights_publisher.vision_extractor import (
                extract_and_publish_vision,
            )

            extract_and_publish_vision(
                plan.vision_step.source_repo,
                plan.vision_step.sidecar_repo,
                plan.scratch_root,
                token=env.get("HF_TOKEN"),
                dry_run=False,
                target_model=plan.entry.source_model,
                catalog_key=plan.entry.key,
            )
            if plan.collection_slug is not None:
                file_artifact_in_collection(
                    plan.vision_step.sidecar_repo,
                    "vision-sidecar",
                    token=env.get("HF_TOKEN"),
                )


