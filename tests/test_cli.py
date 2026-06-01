from __future__ import annotations

from pathlib import Path

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


def _patch_catalog_add_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    id: str = "mlx-community/TestModel-4bit",
    tags: list[str] | None = None,
    empty_catalog: bool = True,
) -> None:
    """Patch the three external calls made by _cmd_catalog_add."""
    import skulk_weights_publisher.catalog_adder as adder_mod
    import skulk_weights_publisher.cli as cli_mod

    fake_info = {"id": id, "tags": tags or ["text-generation"]}
    monkeypatch.setattr(adder_mod, "fetch_hf_model_info", lambda *a, **kw: fake_info)
    monkeypatch.setattr(adder_mod, "detect_mtp_keys", lambda *a, **kw: [])
    if empty_catalog:
        from unittest.mock import MagicMock

        view = MagicMock()
        view.entries = []
        monkeypatch.setattr(cli_mod, "load_catalogue_view", lambda **kw: view)


def test_cli_catalog_add_dry_run(
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_catalog_add_deps(monkeypatch)

    exit_code = run(
        ["catalog", "add", "mlx-community/TestModel-4bit", "--dry-run"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "testmodel-full-q4-k" in captured.out
    assert "dry run" in captured.out


def test_cli_catalog_add_rejects_unsupported_quant(
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_catalog_add_deps(
        monkeypatch,
        id="mlx-community/BigModel-8bit",
        tags=["8-bit", "text-generation"],
    )

    exit_code = run(
        ["catalog", "add", "mlx-community/BigModel-8bit", "--dry-run"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "q8k" in captured.err
    assert "not supported" in captured.err


def test_cli_catalog_add_rejects_duplicate_key(
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    import skulk_weights_publisher.catalog_adder as adder_mod
    import skulk_weights_publisher.cli as cli_mod

    fake_info = {"id": "mlx-community/TestModel-4bit", "tags": ["text-generation"]}
    monkeypatch.setattr(adder_mod, "fetch_hf_model_info", lambda *a, **kw: fake_info)
    monkeypatch.setattr(adder_mod, "detect_mtp_keys", lambda *a, **kw: [])

    existing = MagicMock()
    existing.key = "foxlight/testmodel-full-q4-k"
    existing.hf_repo = "FoxlightAI/testmodel-full-q4-k-vindex"
    existing.output_name = "testmodel-full-q4-k.vindex"
    view = MagicMock()
    view.entries = [existing]
    monkeypatch.setattr(cli_mod, "load_catalogue_view", lambda **kw: view)

    exit_code = run(
        ["catalog", "add", "mlx-community/TestModel-4bit", "--dry-run"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "already exists" in captured.err


def test_cli_catalog_add_rejects_duplicate_output_name(
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    import skulk_weights_publisher.catalog_adder as adder_mod
    import skulk_weights_publisher.cli as cli_mod

    fake_info = {"id": "mlx-community/TestModel-4bit", "tags": ["text-generation"]}
    monkeypatch.setattr(adder_mod, "fetch_hf_model_info", lambda *a, **kw: fake_info)
    monkeypatch.setattr(adder_mod, "detect_mtp_keys", lambda *a, **kw: [])

    existing = MagicMock()
    existing.key = "foxlight/other-key-full-q4-k"
    existing.hf_repo = "FoxlightAI/other-hf-repo-vindex"
    existing.output_name = "testmodel-full-q4-k.vindex"
    view = MagicMock()
    view.entries = [existing]
    monkeypatch.setattr(cli_mod, "load_catalogue_view", lambda **kw: view)

    exit_code = run(
        ["catalog", "add", "mlx-community/TestModel-4bit", "--dry-run"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "output_name" in captured.err
    assert "already exists" in captured.err


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


def test_scratch_clean_deletes_directory(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.cli as cli_mod

    scratch = tmp_path / "scratch" / "deep" / "dir"
    scratch.mkdir(parents=True)
    monkeypatch.setattr(cli_mod, "default_scratch_root", lambda: scratch)

    exit_code = run(["scratch", "clean", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "deleted" in captured.out
    assert not scratch.exists()


def test_scratch_clean_missing_dir(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.cli as cli_mod

    scratch = tmp_path / "scratch" / "deep" / "dir"
    monkeypatch.setattr(cli_mod, "default_scratch_root", lambda: scratch)

    exit_code = run(["scratch", "clean", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "does not exist" in captured.out


def test_scratch_clean_aborts_on_no(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.cli as cli_mod

    scratch = tmp_path / "scratch" / "deep" / "dir"
    scratch.mkdir(parents=True)
    monkeypatch.setattr(cli_mod, "default_scratch_root", lambda: scratch)
    monkeypatch.setattr("builtins.input", lambda _: "n")

    exit_code = run(["scratch", "clean"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "aborted" in captured.out
    assert scratch.exists()


def test_scratch_clean_aborts_on_eof(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.cli as cli_mod

    scratch = tmp_path / "scratch" / "deep" / "dir"
    scratch.mkdir(parents=True)
    monkeypatch.setattr(cli_mod, "default_scratch_root", lambda: scratch)

    def _raise(_: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)

    exit_code = run(["scratch", "clean"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "aborted" in captured.out
    assert scratch.exists()


def test_scratch_clean_rejects_dangerous_path(
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.cli as cli_mod

    monkeypatch.setattr(cli_mod, "default_scratch_root", lambda: Path.home())

    exit_code = run(["scratch", "clean", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "refusing to delete" in captured.err


def test_scratch_clean_rejects_cwd(
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.cli as cli_mod

    monkeypatch.setattr(cli_mod, "default_scratch_root", lambda: Path.cwd())

    exit_code = run(["scratch", "clean", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "refusing to delete" in captured.err


def test_scratch_clean_rejects_ancestor_of_cwd(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.cli as cli_mod

    # Place cwd inside the scratch root so deleting scratch would delete cwd.
    nested = tmp_path / "project" / "subdir"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.setattr(cli_mod, "default_scratch_root", lambda: tmp_path)

    exit_code = run(["scratch", "clean", "--yes"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "refusing to delete" in captured.err


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


def test_cli_vision_extraction_error_caught_by_run(
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.cli as cli_mod
    from skulk_weights_publisher.vision_extractor import VisionExtractionError

    def _raise(*a: object, **kw: object) -> None:
        raise VisionExtractionError("no .safetensors weights found in test/repo")

    monkeypatch.setattr(cli_mod, "execute_publish_plan", _raise)

    exit_code = run(
        ["--manifest", "models.yaml", "publish", "--model", "gemma-3-4b-full-q4-k"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "no .safetensors weights found" in captured.err
