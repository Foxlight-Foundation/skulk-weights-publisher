from __future__ import annotations

from pathlib import Path

import pytest

from skulk_vindex_publisher.catalogue import (
    filter_catalogue_entries,
    find_catalogue_entry,
    load_catalogue_view,
    write_default_config,
)
from skulk_vindex_publisher.manifest import ManifestError


def _write_operator_manifest(
    path: Path,
    *,
    key: str = "local-7b-full-q4-k",
    output_name: str = "local-7b-full-q4-k.vindex",
    hf_repo: str = "acme/local-7b-full-q4-k-vindex",
) -> None:
    path.write_text(
        f"""
models:
  - key: {key}
    source_model: acme/Local-7B-Instruct
    quant: q4k
    tier: smoke
    slices:
      - full
    output_name: {output_name}
    hf_repo: {hf_repo}
""",
        encoding="utf-8",
    )


def test_default_catalogue_loads_packaged_foxlight_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    view = load_catalogue_view()

    assert len(view.sources) == 1
    assert view.sources[0].kind == "builtin"
    assert view.sources[0].namespace == "foxlight"
    assert len(view.entries) == 9
    assert view.entries[0].key == "foxlight/gemma-3-4b-full-q4-k"
    assert [entry.key for entry in filter_catalogue_entries(view, "smoke")] == [
        "foxlight/gemma-3-4b-full-q4-k",
        "foxlight/llama-3-2-3b-full-q4-k",
        "foxlight/qwen-2-5-7b-full-q4-k",
    ]


def test_config_adds_operator_catalogue_without_listing_builtin(
    tmp_path: Path,
) -> None:
    operator_manifest = tmp_path / "operator-vindexes.yaml"
    _write_operator_manifest(operator_manifest)
    config = tmp_path / "skulk-vindex.yaml"
    config.write_text(
        """
catalogues:
  - path: ./operator-vindexes.yaml
    namespace: acme
    hf_owner: acme
""",
        encoding="utf-8",
    )

    view = load_catalogue_view(config_path=config)
    entry = find_catalogue_entry("acme/local-7b-full-q4-k", view)

    assert [source.name for source in view.sources] == ["foxlight", "acme"]
    assert len(view.entries) == 10
    assert entry.hf_repo == "acme/local-7b-full-q4-k-vindex"


def test_empty_config_is_valid_because_foxlight_is_automatic(
    tmp_path: Path,
) -> None:
    config = tmp_path / "skulk-vindex.yaml"
    write_default_config(config)

    view = load_catalogue_view(config_path=config)

    assert len(view.entries) == 9
    assert view.sources[0].name == "foxlight"
    with pytest.raises(ManifestError, match="already exists"):
        write_default_config(config)


def test_duplicate_effective_key_across_sources_is_rejected(tmp_path: Path) -> None:
    operator_manifest = tmp_path / "operator-vindexes.yaml"
    _write_operator_manifest(
        operator_manifest,
        key="gemma-3-4b-full-q4-k",
        output_name="operator-gemma-3-4b-full-q4-k.vindex",
        hf_repo="acme/operator-gemma-3-4b-full-q4-k-vindex",
    )
    config = tmp_path / "skulk-vindex.yaml"
    config.write_text(
        """
catalogues:
  - path: ./operator-vindexes.yaml
    namespace: foxlight
    hf_owner: acme
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="duplicate catalogue key"):
        load_catalogue_view(config_path=config)


def test_duplicate_output_across_sources_is_rejected(tmp_path: Path) -> None:
    operator_manifest = tmp_path / "operator-vindexes.yaml"
    _write_operator_manifest(
        operator_manifest,
        output_name="gemma-3-4b-it-full-q4-k.vindex",
    )
    config = tmp_path / "skulk-vindex.yaml"
    config.write_text(
        """
catalogues:
  - path: ./operator-vindexes.yaml
    namespace: acme
    hf_owner: acme
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="duplicate output_name"):
        load_catalogue_view(config_path=config)


def test_duplicate_hugging_face_repo_across_sources_is_rejected(
    tmp_path: Path,
) -> None:
    operator_manifest = tmp_path / "operator-vindexes.yaml"
    _write_operator_manifest(
        operator_manifest,
        output_name="operator-gemma-3-4b-full-q4-k.vindex",
        hf_repo="skulk/gemma-3-4b-it-full-q4-k-vindex",
    )
    config = tmp_path / "skulk-vindex.yaml"
    config.write_text(
        """
catalogues:
  - path: ./operator-vindexes.yaml
    namespace: acme
    hf_owner: skulk
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="duplicate hf_repo"):
        load_catalogue_view(config_path=config)


def test_operator_hugging_face_owner_must_match_config(tmp_path: Path) -> None:
    operator_manifest = tmp_path / "operator-vindexes.yaml"
    _write_operator_manifest(
        operator_manifest,
        hf_repo="other/local-7b-full-q4-k-vindex",
    )
    config = tmp_path / "skulk-vindex.yaml"
    config.write_text(
        """
catalogues:
  - path: ./operator-vindexes.yaml
    namespace: acme
    hf_owner: acme
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="hf_repo owner must be 'acme'"):
        load_catalogue_view(config_path=config)
