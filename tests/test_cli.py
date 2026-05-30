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
    assert "catalog valid: 10 entries from 1 sources" in captured.out


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
    assert "catalog valid: 10 entries from 1 sources" in captured.out


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
