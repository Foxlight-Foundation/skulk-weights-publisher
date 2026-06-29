"""Unit tests for the GGUF converter (own-the-artifact for the served path)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import skulk_weights_publisher.gguf_converter as gc
from skulk_weights_publisher.gguf_converter import (
    GgufConversionError,
    convert_and_publish_gguf,
    output_filename,
)


def test_output_filename_is_deterministic() -> None:
    assert (
        output_filename("google/gemma-4-31B-it-assistant", "Q8_0")
        == "gemma-4-31B-it-assistant-Q8_0.gguf"
    )


def test_dry_run_returns_planned_filename_without_work() -> None:
    # No tooling, no HF, no token needed: dry-run is pure planning.
    name = convert_and_publish_gguf(
        "google/gemma-4-31B-it-assistant",
        "foxlight/gemma-4-31B-mtp-draft-GGUF",
        Path("/tmp/scratch"),
        token=None,
        precision="Q8_0",
        dry_run=True,
    )
    assert name == "gemma-4-31B-it-assistant-Q8_0.gguf"


def test_unknown_precision_rejected() -> None:
    with pytest.raises(GgufConversionError, match="unknown precision"):
        convert_and_publish_gguf(
            "org/src", "org/dst", Path("/tmp"), token=None, precision="Q3_K_XL"
        )


def test_missing_llama_cpp_dir_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gc.LLAMA_CPP_DIR_ENV, raising=False)
    # Not already published -> proceeds to tool resolution, which must raise.
    with (
        patch.object(gc, "gguf_already_published", return_value=False),
        pytest.raises(GgufConversionError, match="LLAMA_CPP_DIR is not set"),
    ):
        convert_and_publish_gguf(
            "org/src", "org/dst", Path("/tmp"), token="t", precision="Q8_0"
        )


def test_already_published_skips_without_force(monkeypatch: pytest.MonkeyPatch) -> None:
    with patch.object(gc, "gguf_already_published", return_value=True) as pub:
        name = convert_and_publish_gguf(
            "org/src", "org/dst", Path("/tmp"), token="t", precision="Q8_0"
        )
    assert name == "src-Q8_0.gguf"
    pub.assert_called_once()


def test_convert_quantize_publish_flow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Wire up a fake llama.cpp checkout so tool resolution succeeds.
    convert_script = tmp_path / "convert_hf_to_gguf.py"
    convert_script.write_text("# stub")
    qbin = tmp_path / "build" / "bin" / "llama-quantize"
    qbin.parent.mkdir(parents=True)
    qbin.write_text("#!/bin/sh\n")
    qbin.chmod(0o755)
    monkeypatch.setenv(gc.LLAMA_CPP_DIR_ENV, str(tmp_path))

    calls: list[list[str]] = []

    def fake_run(cmd, check):  # noqa: ANN001
        calls.append([str(c) for c in cmd])
        return MagicMock(returncode=0)

    hf = MagicMock()
    hf["snapshot_download"].return_value = str(tmp_path / "source")
    with (
        patch.object(gc, "gguf_already_published", return_value=False),
        patch.object(gc.subprocess, "run", side_effect=fake_run),
        patch("huggingface_hub.create_repo") as create_repo,
        patch("huggingface_hub.snapshot_download", return_value=str(tmp_path / "src")),
        patch("huggingface_hub.upload_file") as upload_file,
        patch("huggingface_hub.utils.tqdm.disable_progress_bars"),
    ):
        name = convert_and_publish_gguf(
            "google/gemma-4-31B-it-assistant",
            "foxlight/gemma-4-31B-mtp-draft-GGUF",
            tmp_path / "scratch",
            token="hf_xxx",
            precision="Q8_0",
        )

    assert name == "gemma-4-31B-it-assistant-Q8_0.gguf"
    create_repo.assert_called_once()
    # Convert (f16) then quantize (Q8_0) were both invoked.
    assert any("convert_hf_to_gguf.py" in " ".join(c) for c in calls)
    assert any(c[-1] == "Q8_0" for c in calls)
    upload_file.assert_called_once()
    kwargs = upload_file.call_args.kwargs
    assert kwargs["path_in_repo"] == "gemma-4-31B-it-assistant-Q8_0.gguf"
    assert kwargs["repo_id"] == "foxlight/gemma-4-31B-mtp-draft-GGUF"


def test_f16_precision_skips_quantize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    convert_script = tmp_path / "convert_hf_to_gguf.py"
    convert_script.write_text("# stub")
    monkeypatch.setenv(gc.LLAMA_CPP_DIR_ENV, str(tmp_path))

    calls: list[list[str]] = []

    def fake_run(cmd, check):  # noqa: ANN001
        calls.append([str(c) for c in cmd])
        return MagicMock(returncode=0)

    with (
        patch.object(gc, "gguf_already_published", return_value=False),
        patch.object(gc.subprocess, "run", side_effect=fake_run),
        patch("huggingface_hub.create_repo"),
        patch("huggingface_hub.snapshot_download", return_value=str(tmp_path / "src")),
        patch("huggingface_hub.upload_file"),
        patch("huggingface_hub.utils.tqdm.disable_progress_bars"),
    ):
        convert_and_publish_gguf(
            "org/src", "org/dst", tmp_path / "scratch", token="t", precision="F16"
        )

    # Only the convert step runs; no quantize invocation (no llama-quantize needed).
    assert len(calls) == 1
    assert "convert_hf_to_gguf.py" in " ".join(calls[0])
