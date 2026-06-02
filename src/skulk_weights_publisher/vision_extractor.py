"""Vision encoder sidecar publishing.

Some VLM checkpoints published by mlx-community omit the vision encoder — the
main repo carries only the language model, and the vision weights live in a
third-party repo (e.g. ``davehind/Kimi-K2.5-vision``). Skulk's
``VisionCardConfig.weights_repo`` points at that separate repo, so depending on
a third party introduces availability and versioning risk.

This module republishes a Foxlight-owned vision sidecar: it mirrors the vision
source repo's weights and configs into a Foxlight-controlled repo, byte-for-byte
(no quantization, no tensor surgery), so Foxlight owns the full artifact chain.

Deliberately faithful: unlike MTP extraction, vision republishing performs **no
quantization and no dtype conversion** — the weights are copied unchanged. The
issue signature (``extract_and_publish_vision(source_repo, sidecar_repo)``)
carries no quant parameter for exactly this reason; precision is preserved.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path

# Weight + config/tokenizer/processor files worth mirroring into the sidecar.
# Excludes the source repo's READMEs, .gitattributes, and other repo cruft.
_VISION_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.json",
    "*.txt",
    "*.model",
]


class VisionExtractionError(RuntimeError):
    """Raised when vision sidecar extraction or publishing fails."""


def _stderr_log(message: str) -> None:
    """Default log sink — writes a progress line to stderr (CLI behavior)."""
    print(message, file=sys.stderr)


def extract_and_publish_vision(
    source_repo: str,
    sidecar_repo: str,
    scratch_root: Path,
    *,
    token: str | None,
    dry_run: bool = False,
    target_model: str | None = None,
    catalog_key: str | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    """Mirror vision encoder weights from ``source_repo`` to ``sidecar_repo``.

    Downloads the weight and config files from ``source_repo`` and re-uploads
    them, unchanged, to the Foxlight-owned ``sidecar_repo``. The copy is faithful
    — no quantization or dtype conversion — so the published sidecar is
    numerically identical to the upstream vision encoder.

    Progress lines go through the ``log`` callback, which defaults to stderr (the
    CLI behavior). In dry-run mode prints the plan without downloading or
    uploading anything.
    """

    emit = log if log is not None else _stderr_log

    if dry_run:
        _print_dry_run_plan(source_repo, sidecar_repo, scratch_root)
        return

    try:
        from huggingface_hub import create_repo, snapshot_download, upload_folder
        from huggingface_hub.utils.tqdm import (
            disable_progress_bars,  # type: ignore[import-untyped]
        )

        disable_progress_bars()
    except ImportError as exc:
        raise VisionExtractionError(
            "huggingface_hub is required for vision sidecar publishing"
        ) from exc

    # Create the sidecar repo up front so a permissions problem fails fast,
    # before the (potentially large) download.
    create_repo(
        sidecar_repo, repo_type="model", private=False, exist_ok=True, token=token
    )

    # Start from a clean scratch dir: snapshot_download(local_dir=...) only writes
    # the files that match the current source snapshot, so a reused dir from an
    # earlier run (with shards/configs the source has since dropped or renamed)
    # would leave stale files behind and break the byte-for-byte mirror guarantee.
    local_dir = scratch_root / "vision" / sidecar_repo.replace("/", "--")
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    emit(f"vision: downloading weights from hf://{source_repo}")
    snapshot_download(
        repo_id=source_repo,
        repo_type="model",
        local_dir=str(local_dir),
        token=token,
        allow_patterns=_VISION_ALLOW_PATTERNS,
    )

    weight_files = sorted(local_dir.rglob("*.safetensors"))
    if not weight_files:
        raise VisionExtractionError(
            f"no .safetensors weights found in {source_repo}; "
            "confirm this repo holds the vision encoder weights"
        )
    emit(f"vision: {len(weight_files)} weight file(s) downloaded")

    emit(f"vision: uploading to hf://{sidecar_repo}")
    # delete_patterns prunes files already on the Hub that the current snapshot no
    # longer contains, so republishing a repointed/slimmed source produces a true
    # mirror instead of accumulating obsolete shards/configs/indexes alongside the
    # new ones. Scoped to the same file kinds we manage so HF repo metadata
    # (.gitattributes, auto-created README) is left untouched.
    upload_folder(
        folder_path=str(local_dir),
        repo_id=sidecar_repo,
        repo_type="model",
        token=token,
        commit_message=f"Mirror vision encoder from {source_repo}",
        delete_patterns=_VISION_ALLOW_PATTERNS,
    )
    emit(f"vision: published to hf://{sidecar_repo}")

    # Skulk owns artifact lifecycle — discard the local mirror we staged.
    shutil.rmtree(local_dir)

    # Publish a self-describing model card. README.md is not in
    # _VISION_ALLOW_PATTERNS, so a later mirror's delete_patterns never prunes it.
    from skulk_weights_publisher.card_publish import publish_model_card

    publish_model_card(
        repo_id=sidecar_repo,
        artifact_type="vision-sidecar",
        source_repo=source_repo,
        token=token,
        target_model=target_model,
        catalog_key=catalog_key,
        log=emit,
    )


def _print_dry_run_plan(
    source_repo: str,
    sidecar_repo: str,
    scratch_root: Path,
) -> None:
    local_dir = scratch_root / "vision" / sidecar_repo.replace("/", "--")
    lines = [
        f"vision source repo:  hf://{source_repo}",
        f"vision sidecar repo: hf://{sidecar_repo}",
        f"vision scratch dir:  {local_dir}",
        "vision step:         mirror weights + configs → upload (no quantization)",
    ]
    print("\n".join(lines))
