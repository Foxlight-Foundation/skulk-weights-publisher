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
    only if none exists.

    Looks up an existing collection by title first. ``create_collection`` is not
    reliably idempotent by title — HF collection slugs carry a generated unique
    suffix, so creating blindly can spawn duplicate "MTP Sidecars" collections and
    scatter items across them on repeated publishes. Reuse-before-create keeps all
    items of a type in one collection.
    """

    try:
        from huggingface_hub import create_collection, list_collections
    except ImportError as exc:  # pragma: no cover - hub is a base dependency
        raise CollectionError(
            "huggingface_hub is required to manage collections"
        ) from exc

    try:
        for collection in list_collections(owner=owner, token=token):
            if collection.title == title:
                return collection.slug
        created = create_collection(
            title=title,
            namespace=owner,
            exists_ok=True,
            token=token,
        )
        return created.slug
    except Exception as exc:  # noqa: BLE001 - surface any hub failure as our error
        raise CollectionError(
            f"failed to ensure collection {owner}/{title!r}: {exc}"
        ) from exc


def file_artifact_in_collection(
    repo_id: str,
    artifact_type: ArtifactType,
    *,
    token: str | None,
    collection_slug: str | None = None,
    note: str | None = None,
) -> None:
    """File ``repo_id`` into a Hugging Face collection.

    When ``collection_slug`` is given (an explicitly-configured target — a catalog
    ``hf_collection`` or the ``SKULK_WEIGHTS_COLLECTION`` override), the repo is
    added to that exact collection, so operator configuration is honored and the
    dry-run plan matches what executes. Otherwise the collection is resolved by
    title for the artifact type, create-if-missing under the repo's own owner
    (the rollout-robust default for sidecars, which carry no configured slug).
    The item add is idempotent via ``exists_ok``.
    """

    try:
        from huggingface_hub import add_collection_item
    except ImportError as exc:  # pragma: no cover - hub is a base dependency
        raise CollectionError(
            "huggingface_hub is required to manage collections"
        ) from exc

    if collection_slug is not None:
        slug = collection_slug
    else:
        title = COLLECTION_TITLES.get(artifact_type)
        if title is None:
            raise CollectionError(
                f"no collection configured for artifact type {artifact_type!r}"
            )
        if "/" not in repo_id:
            raise CollectionError(f"cannot derive collection owner from {repo_id!r}")
        owner = repo_id.split("/", maxsplit=1)[0]
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
