"""File published artifacts into per-artifact-type Hugging Face collections.

So the FoxlightAI org is browsable by artifact type rather than a flat repo
list, each publish is filed into the collection for its kind — ``Vindexes``,
``MTP Sidecars``, ``Vision Sidecars`` (and ``Models`` later for fine-tunes).

Collections are resolved **by title, create-if-missing** rather than by a
hard-coded slug. That is deliberately rollout-robust: the collection can be
deleted and recreated (as the HF-org rollout plan does) without breaking
publishes, because ``ensure_collection`` recreates it on demand. Owner is derived
from the repo being filed, so the collection always lives under the same org as
its items.
"""

from __future__ import annotations

from skulk_weights_publisher.model_card import ArtifactType

# One collection per artifact type, addressed by title (HF assigns the slug).
COLLECTION_TITLES: dict[ArtifactType, str] = {
    "vindex": "Vindexes",
    "mtp-sidecar": "MTP Sidecars",
    "vision-sidecar": "Vision Sidecars",
}


class CollectionError(RuntimeError):
    """Raised when a collection cannot be ensured or an item cannot be filed."""


def ensure_collection(title: str, owner: str, *, token: str | None) -> str:
    """Return the slug of ``owner``'s collection titled ``title``, creating it
    if it does not exist. Idempotent via ``exists_ok``."""

    try:
        from huggingface_hub import create_collection
    except ImportError as exc:  # pragma: no cover - hub is a base dependency
        raise CollectionError(
            "huggingface_hub is required to manage collections"
        ) from exc

    try:
        collection = create_collection(
            title=title,
            namespace=owner,
            exists_ok=True,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001 - surface any hub failure as our error
        raise CollectionError(
            f"failed to ensure collection {owner}/{title!r}: {exc}"
        ) from exc

    return collection.slug


def file_artifact_in_collection(
    repo_id: str,
    artifact_type: ArtifactType,
    *,
    token: str | None,
    note: str | None = None,
) -> None:
    """File ``repo_id`` into the collection for its artifact type.

    The collection is ensured (create-if-missing) under the repo's own owner,
    then the repo is added as a model item (idempotent via ``exists_ok``).
    """

    title = COLLECTION_TITLES.get(artifact_type)
    if title is None:
        raise CollectionError(
            f"no collection configured for artifact type {artifact_type!r}"
        )
    if "/" not in repo_id:
        raise CollectionError(f"cannot derive collection owner from {repo_id!r}")
    owner = repo_id.split("/", maxsplit=1)[0]

    try:
        from huggingface_hub import add_collection_item
    except ImportError as exc:  # pragma: no cover - hub is a base dependency
        raise CollectionError(
            "huggingface_hub is required to manage collections"
        ) from exc

    slug = ensure_collection(title, owner, token=token)
    try:
        add_collection_item(
            slug,
            item_id=repo_id,
            item_type="model",
            note=note,
            exists_ok=True,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001 - surface any hub failure as our error
        raise CollectionError(
            f"failed to add {repo_id} to collection {slug}: {exc}"
        ) from exc
