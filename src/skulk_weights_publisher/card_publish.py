"""Render and publish self-describing model cards alongside artifacts.

This is the **network-touching** companion to the pure ``model_card`` renderer.
It resolves the bits SWP doesn't already have in hand — the source repo's exact
commit SHA and its license — from the Hub, then renders the card and uploads it
as ``README.md`` to the published repo.

Kept separate from ``model_card`` so the renderer stays pure/offline and easily
tested; here the network/clock effects live behind small, mockable functions.

Provenance resolution is **best-effort**: if the SHA or license can't be
determined (network hiccup, missing card metadata, ``huggingface_hub`` absent),
the card is still published with whatever is known rather than failing the
publish. The card *upload* itself is not swallowed — if it fails, the caller
hears about it, because a weights-only repo is exactly the decontextualized state
this feature exists to prevent.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from skulk_weights_publisher.model_card import ArtifactType, CardInfo, render_model_card


@dataclass(frozen=True)
class SourceProvenance:
    """Facts resolved from the source repo for the card's provenance block."""

    revision: str | None = None
    license: str | None = None
    license_name: str | None = None
    license_link: str | None = None


def _stderr_log(message: str) -> None:
    print(message, file=sys.stderr)


def _tool_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("skulk-weights-publisher")
    except Exception:  # noqa: BLE001 - cosmetic; never fail a publish over it
        return None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_source_provenance(
    source_repo: str,
    *,
    token: str | None,
) -> SourceProvenance:
    """Resolve the source repo's commit SHA and license from the Hub.

    Best-effort: any failure (no network, unknown repo, no card metadata, missing
    ``huggingface_hub``) yields an empty :class:`SourceProvenance` rather than
    raising, so the card can still be published with partial information.
    """

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return SourceProvenance()

    try:
        info = HfApi().model_info(source_repo, token=token)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return SourceProvenance()

    sha = _as_str(getattr(info, "sha", None))
    card = getattr(info, "cardData", None)
    if not isinstance(card, dict):
        return SourceProvenance(revision=sha)
    return SourceProvenance(
        revision=sha,
        license=_as_str(card.get("license")),
        license_name=_as_str(card.get("license_name")),
        license_link=_as_str(card.get("license_link")),
    )


def publish_model_card(
    *,
    repo_id: str,
    artifact_type: ArtifactType,
    source_repo: str,
    token: str | None,
    target_model: str | None = None,
    quant: str | None = None,
    catalog_key: str | None = None,
    weight_filename: str | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    """Render a self-describing card for a published artifact and upload it.

    Resolves source provenance (SHA + inherited license) best-effort, renders the
    card, and uploads it as ``README.md`` to ``repo_id``. The target defaults to
    the source repo when not given (true for MTP/vindex, where the artifact serves
    the model it was derived from).
    """

    emit = log if log is not None else _stderr_log

    try:
        from huggingface_hub import upload_file
    except ImportError:
        emit("card: huggingface_hub unavailable; skipping model card")
        return

    provenance = resolve_source_provenance(source_repo, token=token)
    info = CardInfo(
        artifact_type=artifact_type,
        repo_id=repo_id,
        source_repo=source_repo,
        source_revision=provenance.revision,
        target_model=target_model or source_repo,
        quant=quant,
        license=provenance.license,
        license_name=provenance.license_name,
        license_link=provenance.license_link,
        catalog_key=catalog_key,
        tool_version=_tool_version(),
        generated_at=_now_iso(),
        weight_filename=weight_filename,
    )
    content = render_model_card(info)

    emit(f"card: uploading model card to hf://{repo_id}/README.md")
    upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=f"Add self-describing model card ({artifact_type})",
    )
    emit(f"card: published model card to hf://{repo_id}/README.md")
