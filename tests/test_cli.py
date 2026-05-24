from __future__ import annotations

from pytest import CaptureFixture

from skulk_vindex_publisher.cli import run
from skulk_vindex_publisher.defaults import DEFAULT_FOXLIGHT_VINDEX_COLLECTION


def test_cli_catalog_validate(capsys: CaptureFixture[str]) -> None:
    exit_code = run(["catalog", "validate"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "catalog valid: 9 entries from 1 sources" in captured.out


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
    assert "catalog valid: 9 entries from 1 sources" in captured.out


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
