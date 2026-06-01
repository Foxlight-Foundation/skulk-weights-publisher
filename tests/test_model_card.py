from __future__ import annotations

import yaml

from skulk_weights_publisher.model_card import CardInfo, render_model_card


def _split_card(card: str) -> tuple[dict[str, object], str]:
    """Split a rendered card into (parsed frontmatter, body)."""
    assert card.startswith("---\n")
    _, fm, body = card.split("---\n", 2)
    return yaml.safe_load(fm), body


def _mtp_info(**overrides: object) -> CardInfo:
    base: dict[str, object] = dict(
        artifact_type="mtp-sidecar",
        repo_id="FoxlightAI/qwen3-6-35b-a3b-mtp-q4k",
        source_repo="Qwen/Qwen3.6-35B-A3B",
        source_revision="abc123def456",
        target_model="Qwen/Qwen3.6-35B-A3B",
        quant="q4k",
        license="apache-2.0",
        catalog_key="foxlight/qwen3-6-35b-a3b-full-q4-k",
        tool_version="0.1.0",
        generated_at="2026-06-01T00:00:00Z",
        weight_filename="mtp.safetensors",
    )
    base.update(overrides)
    return CardInfo(**base)  # type: ignore[arg-type]


def test_card_has_frontmatter_and_body() -> None:
    card = render_model_card(_mtp_info())
    fm, body = _split_card(card)

    assert isinstance(fm, dict)
    assert body.strip().startswith("# FoxlightAI/qwen3-6-35b-a3b-mtp-q4k")


def test_frontmatter_carries_structured_provenance() -> None:
    fm, _ = _split_card(render_model_card(_mtp_info()))

    assert fm["base_model"] == "Qwen/Qwen3.6-35B-A3B"
    prov = fm["foxlight"]
    assert prov["artifact_type"] == "mtp-sidecar"
    assert prov["source_repo"] == "Qwen/Qwen3.6-35B-A3B"
    assert prov["source_revision"] == "abc123def456"
    assert prov["target_model"] == "Qwen/Qwen3.6-35B-A3B"
    assert prov["quant"] == "q4k"
    assert prov["catalog_key"] == "foxlight/qwen3-6-35b-a3b-full-q4-k"
    assert "skulk-weights-publisher 0.1.0" in str(prov["extracted_with"])


def test_tags_include_artifact_type_quant_and_defaults() -> None:
    fm, _ = _split_card(render_model_card(_mtp_info(extra_tags=("qwen",))))
    tags = fm["tags"]
    for expected in ("mtp-sidecar", "skulk", "foxlight", "q4k", "qwen"):
        assert expected in tags


def test_license_inherited_and_noted_not_relicensed() -> None:
    card = render_model_card(_mtp_info(license="apache-2.0"))
    fm, body = _split_card(card)

    assert fm["license"] == "apache-2.0"
    assert "preserved unchanged" in body
    assert "original license" in body


def test_custom_license_uses_name_and_link() -> None:
    fm, body = _split_card(
        render_model_card(
            _mtp_info(
                license="other",
                license_name="gemma",
                license_link="https://ai.google.dev/gemma/terms",
            )
        )
    )
    assert fm["license"] == "other"
    assert fm["license_name"] == "gemma"
    assert fm["license_link"] == "https://ai.google.dev/gemma/terms"
    assert "gemma" in body


def test_mtp_summary_states_not_standalone() -> None:
    _, body = _split_card(render_model_card(_mtp_info()))
    assert "not a standalone model" in body
    assert "speculatively decode" in body
    assert "mtp.safetensors" in body  # usage references the weight file


def test_provenance_table_present_in_body() -> None:
    _, body = _split_card(render_model_card(_mtp_info()))
    assert "## Provenance" in body
    assert "Source revision" in body
    assert "`abc123def456`" in body


def test_vindex_card_renders() -> None:
    info = CardInfo(
        artifact_type="vindex",
        repo_id="FoxlightAI/gemma-3-4b-it-full-q4-k-vindex",
        source_repo="google/gemma-3-4b-it",
        quant="q4k",
        license="gemma",
    )
    fm, body = _split_card(render_model_card(info))
    assert fm["foxlight"]["artifact_type"] == "vindex"
    assert "LARQL" in body


def test_vision_card_renders() -> None:
    info = CardInfo(
        artifact_type="vision-sidecar",
        repo_id="FoxlightAI/kimi-k2-5-vision",
        source_repo="thirdparty/Kimi-K2.5-vision",
        target_model="moonshotai/Kimi-K2.5",
    )
    fm, body = _split_card(render_model_card(info))
    assert fm["foxlight"]["artifact_type"] == "vision-sidecar"
    assert "vision" in body.lower()
    # No quant → no quant tag.
    assert "q4k" not in fm["tags"]


def test_human_blurb_appended_when_present() -> None:
    card = render_model_card(_mtp_info(human_blurb="Hand-written note."))
    _, body = _split_card(card)
    assert "## Notes" in body
    assert "Hand-written note." in body


def test_human_blurb_absent_by_default() -> None:
    _, body = _split_card(render_model_card(_mtp_info()))
    assert "## Notes" not in body


def test_render_is_deterministic() -> None:
    assert render_model_card(_mtp_info()) == render_model_card(_mtp_info())


def test_optional_fields_omitted_cleanly() -> None:
    # Minimal info: no revision/target/quant/license/etc.
    info = CardInfo(
        artifact_type="mtp-sidecar",
        repo_id="FoxlightAI/x-mtp",
        source_repo="org/x",
    )
    fm, body = _split_card(render_model_card(info))
    assert "license" not in fm
    prov = fm["foxlight"]
    assert "source_revision" not in prov
    assert "quant" not in prov
    # Still valid, still self-describing.
    assert "## Provenance" in body
