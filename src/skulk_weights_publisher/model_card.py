"""Self-describing model cards for published Foxlight artifacts.

Every artifact SWP publishes to the Hub (MTP sidecar, LARQL vindex, vision
sidecar) should carry a model card so anyone browsing — or future us — can tell
what it is, what it's for, and where it came from. Today a publish uploads only
the weight file, which is fully decontextualized.

This module is the **pure** card renderer: given the facts SWP already has at
publish time (the artifact kind, the repo, the source it was derived from at a
pinned revision, the target it serves, quant, inherited license, …), it returns
the full ``README.md`` content (YAML frontmatter + body). It performs no network
or filesystem access and is deterministic — the caller passes the timestamp and
tool version — so it is trivially testable and the publishers (PR B) just upload
its output alongside the weights.

Design notes:

- **Provenance is structured, not just prose.** ``base_model`` in the frontmatter
  is the source repo, which the Hub renders as a derivation relationship; the
  exact source commit (``source_revision``) is recorded so the artifact is
  reproducible. A ``provenance`` frontmatter block + a body table carry the rest.
- **Licensing is inherited unchanged.** Derived artifacts (MTP heads, vindexes)
  are published under the *source model's* original license — never re-licensed.
  The caller supplies the detected license; we propagate it verbatim.
- **An MTP sidecar is not a standalone model**, and the Gemma 4 assistant case is
  a pointer to a separate model, so none of HF's ``base_model_relation`` values
  (finetune/quantized/adapter/merge) fit cleanly — we use ``base_model`` for the
  link plus an honest ``artifact_type`` tag rather than forcing a relation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import yaml

ArtifactType = Literal["mtp-sidecar", "vindex", "vision-sidecar"]

# Human-facing labels and one-line summaries per artifact kind.
_ARTIFACT_LABELS: dict[ArtifactType, str] = {
    "mtp-sidecar": "MTP speculative-decoding sidecar",
    "vindex": "LARQL vindex",
    "vision-sidecar": "vision encoder sidecar",
}


@dataclass(frozen=True)
class CardInfo:
    """Everything needed to render a self-describing model card.

    All fields are plain data the publisher already holds at publish time. The
    renderer is pure, so ``generated_at`` and ``tool_version`` are passed in
    rather than read from the clock/environment.
    """

    artifact_type: ArtifactType
    repo_id: str
    """The repo being published, e.g. ``FoxlightAI/qwen3-6-35b-a3b-mtp-q4k``."""
    source_repo: str
    """Upstream repo the artifact was derived from (the provenance link)."""
    source_revision: str | None = None
    """Exact source commit SHA at extraction time, for reproducibility."""
    target_model: str | None = None
    """The model this artifact serves/accelerates (may equal ``source_repo``)."""
    quant: str | None = None
    """Quantization scheme, e.g. ``q4k`` (None for un-quantized artifacts)."""
    license: str | None = None
    """SPDX-ish license id inherited from the source model (e.g. ``apache-2.0``)."""
    license_name: str | None = None
    """For custom licenses (e.g. Gemma/Llama): the ``other`` license display name."""
    license_link: str | None = None
    """For custom licenses: a link to the license text."""
    catalog_key: str | None = None
    """The SWP catalog key this artifact corresponds to."""
    tool_version: str | None = None
    """Version of skulk-weights-publisher that produced the artifact."""
    generated_at: str | None = None
    """ISO-8601 timestamp the card was generated (caller-supplied, for purity)."""
    extra_tags: tuple[str, ...] = ()
    """Additional Hub tags to merge in beyond the defaults."""
    human_blurb: str | None = None
    """Optional curated paragraph; appended verbatim when provided."""
    skulk_usage: str | None = None
    """Optional override for the usage section; a sensible default is generated."""

    weight_filename: str | None = None
    """The primary weight file in the repo, surfaced in usage (e.g. mtp.safetensors)."""

    extras: dict[str, str] = field(default_factory=dict)
    """Any extra provenance key/values to record (kept simple: string→string)."""


def render_model_card(info: CardInfo) -> str:
    """Render the full ``README.md`` (frontmatter + body) for an artifact.

    Pure and deterministic: identical ``CardInfo`` always yields identical output.
    """

    frontmatter = _render_frontmatter(info)
    body = _render_body(info)
    return f"---\n{frontmatter}---\n\n{body}"


def _render_frontmatter(info: CardInfo) -> str:
    # Insertion order is preserved by PyYAML with sort_keys=False, so build the
    # mapping in the order we want it to appear on the Hub.
    data: dict[str, object] = {}
    if info.source_repo:
        data["base_model"] = info.source_repo
    data["tags"] = _tags(info)
    if info.license:
        data["license"] = info.license
    if info.license_name:
        data["license_name"] = info.license_name
    if info.license_link:
        data["license_link"] = info.license_link

    provenance: dict[str, object] = {
        "artifact_type": info.artifact_type,
        "source_repo": info.source_repo,
    }
    if info.source_revision:
        provenance["source_revision"] = info.source_revision
    if info.target_model:
        provenance["target_model"] = info.target_model
    if info.quant:
        provenance["quant"] = info.quant
    if info.catalog_key:
        provenance["catalog_key"] = info.catalog_key
    if info.tool_version:
        provenance["extracted_with"] = f"skulk-weights-publisher {info.tool_version}"
    if info.generated_at:
        provenance["generated_at"] = info.generated_at
    for key, value in info.extras.items():
        provenance.setdefault(key, value)
    # Namespaced under `foxlight` so it never collides with Hub-reserved keys.
    data["foxlight"] = provenance

    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def _tags(info: CardInfo) -> list[str]:
    tags = [info.artifact_type, "skulk", "foxlight"]
    if info.quant:
        tags.append(info.quant)
    for tag in info.extra_tags:
        if tag not in tags:
            tags.append(tag)
    return tags


def _render_body(info: CardInfo) -> str:
    label = _ARTIFACT_LABELS[info.artifact_type]
    sections: list[str] = [f"# {info.repo_id}", "", _summary(info, label), ""]

    sections += ["## Provenance", "", _provenance_table(info), ""]
    sections += ["## Usage", "", info.skulk_usage or _default_usage(info), ""]
    sections += ["## License", "", _license_note(info), ""]

    if info.human_blurb:
        sections += ["## Notes", "", info.human_blurb.strip(), ""]

    return "\n".join(sections).rstrip() + "\n"


def _summary(info: CardInfo, label: str) -> str:
    target = info.target_model or info.source_repo
    quant = f", quantized to `{info.quant}`" if info.quant else ""
    if info.artifact_type == "mtp-sidecar":
        return (
            f"This repository contains an **{label}** derived from "
            f"[`{info.source_repo}`](https://huggingface.co/{info.source_repo}). "
            "It is **not a standalone model** — it provides the multi-token-"
            "prediction heads used by "
            "[Skulk](https://github.com/Foxlight-Foundation/Skulk) "
            f"to speculatively decode for the target model **`{target}`**{quant}."
        )
    if info.artifact_type == "vindex":
        return (
            f"This repository contains a **{label}** for "
            f"[`{info.source_repo}`](https://huggingface.co/{info.source_repo})"
            f"{quant}. Vindexes are LARQL retrieval indexes consumed by "
            "[Skulk](https://github.com/Foxlight-Foundation/Skulk)."
        )
    return (
        f"This repository contains a **{label}** for "
        f"[`{info.source_repo}`](https://huggingface.co/{info.source_repo}) — the "
        "vision-tower weights needed to run the VLM when the main checkpoint omits "
        "them, published Foxlight-owned so the artifact chain does not depend on a "
        "third party."
    )


def _provenance_table(info: CardInfo) -> str:
    rows: list[tuple[str, str]] = [("Artifact type", info.artifact_type)]
    rows.append(("Source model", f"`{info.source_repo}`"))
    if info.source_revision:
        rows.append(("Source revision", f"`{info.source_revision}`"))
    if info.target_model:
        rows.append(("Target model", f"`{info.target_model}`"))
    if info.quant:
        rows.append(("Quantization", f"`{info.quant}`"))
    if info.catalog_key:
        rows.append(("Catalog key", f"`{info.catalog_key}`"))
    if info.tool_version:
        rows.append(
            ("Extracted with", f"skulk-weights-publisher `{info.tool_version}`")
        )
    if info.generated_at:
        rows.append(("Generated", info.generated_at))
    lines = ["| Field | Value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)


def _default_usage(info: CardInfo) -> str:
    fname = info.weight_filename
    if info.artifact_type == "mtp-sidecar":
        loc = f" (`{fname}`)" if fname else ""
        return (
            f"Skulk loads this sidecar{loc} alongside the target model to enable "
            "MTP speculative decoding. It is referenced from the Skulk Weights "
            "Publisher catalog and fetched automatically by the Skulk shard "
            "downloader; it is not intended to be loaded standalone."
        )
    if info.artifact_type == "vindex":
        return (
            "Skulk loads this vindex for LARQL retrieval. It is referenced from the "
            "Skulk Weights Publisher catalog and fetched automatically; it is not a "
            "standalone model."
        )
    return (
        "Skulk fetches these vision-encoder weights alongside the language model "
        "when running the VLM. Referenced from the Skulk Weights Publisher catalog."
    )


def _license_note(info: CardInfo) -> str:
    if info.license or info.license_name:
        named = info.license_name or info.license
        return (
            f"This artifact is derived from [`{info.source_repo}`]"
            f"(https://huggingface.co/{info.source_repo}) and is published under "
            f"that model's original license (**{named}**), preserved unchanged. "
            "Refer to the source model's card for the full terms."
        )
    return (
        f"This artifact is derived from [`{info.source_repo}`]"
        f"(https://huggingface.co/{info.source_repo}); its license follows that of "
        "the source model. Refer to the source model's card for the full terms."
    )
