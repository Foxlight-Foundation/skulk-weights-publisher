from __future__ import annotations

from pytest import CaptureFixture

from skulk_vindex_publisher.cli import run


def test_cli_manifest_validate(capsys: CaptureFixture[str]) -> None:
    exit_code = run(["manifest", "validate"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "models.yaml valid: 9 entries" in captured.out


def test_cli_publish_dry_run(capsys: CaptureFixture[str]) -> None:
    exit_code = run(["publish", "--model", "gemma-3-4b-full-q4-k", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "extract command:" in captured.out
    assert "dry run complete" in captured.out
