from __future__ import annotations

import pytest
from pytest import CaptureFixture

from skulk_weights_publisher.cli import run
from skulk_weights_publisher.defaults import DEFAULT_FOXLIGHT_VINDEX_COLLECTION
from skulk_weights_publisher.mtp_extractor import MtpExtractionError


def test_cli_catalog_validate(capsys: CaptureFixture[str]) -> None:
    exit_code = run(["catalog", "validate"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "catalog valid: 11 entries from 1 sources" in captured.out


def test_cli_catalog_sources(capsys: CaptureFixture[str]) -> None:
    exit_code = run(["catalog", "sources"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "builtin foxlight namespace=foxlight hf_owner=FoxlightAI" in captured.out
    assert f"hf_collection={DEFAULT_FOXLIGHT_VINDEX_COLLECTION}" in captured.out


def test_cli_legacy_catalogue_alias_still_works(
    capsys: CaptureFixture[str],
) -> None:
    exit_code = run(["catalogue", "validate"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "catalog valid: 11 entries from 1 sources" in captured.out


def test_cli_publish_dry_run(capsys: CaptureFixture[str]) -> None:
    exit_code = run(
        [
            "publish",
            "--model",
            "foxlight/gemma-3-4b-full-q4-k",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "model key: foxlight/gemma-3-4b-full-q4-k" in captured.out
    assert (
        f"collection: https://huggingface.co/collections/{DEFAULT_FOXLIGHT_VINDEX_COLLECTION}"
        in captured.out
    )
    assert "extract command:" in captured.out
    assert "dry run complete" in captured.out


def test_cli_legacy_manifest_publish_dry_run(capsys: CaptureFixture[str]) -> None:
    exit_code = run(
        [
            "--manifest",
            "models.yaml",
            "publish",
            "--model",
            "gemma-3-4b-full-q4-k",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "model key: gemma-3-4b-full-q4-k" in captured.out


def test_cli_catalog_add_dry_run(
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.catalog_adder as adder_mod

    fake_info = {
        "id": "mlx-community/TestModel-4bit",
        "tags": ["base_model:quantized:Acme/TestModel", "text-generation"],
    }
    monkeypatch.setattr(adder_mod, "fetch_hf_model_info", lambda *a, **kw: fake_info)
    monkeypatch.setattr(adder_mod, "detect_mtp_keys", lambda *a, **kw: [])

    exit_code = run(
        ["catalog", "add", "mlx-community/TestModel-4bit", "--dry-run"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "testmodel-full-q4-k" in captured.out
    assert "dry run" in captured.out


def test_cli_catalog_add_error_propagates(
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.catalog_adder as adder_mod
    from skulk_weights_publisher.catalog_adder import CatalogAddError

    monkeypatch.setattr(
        adder_mod,
        "fetch_hf_model_info",
        lambda *a, **kw: (_ for _ in ()).throw(CatalogAddError("HF API returned 404")),
    )

    exit_code = run(["catalog", "add", "bad/model", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "HF API returned 404" in captured.err


def test_cli_mtp_extraction_error_caught_by_run(
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.cli as cli_mod

    def _raise(*a: object, **kw: object) -> None:
        raise MtpExtractionError("no mtp.* keys found in test/repo")

    monkeypatch.setattr(cli_mod, "execute_publish_plan", _raise)

    exit_code = run(
        ["--manifest", "models.yaml", "publish", "--model", "gemma-3-4b-full-q4-k"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "no mtp.* keys found" in captured.err
