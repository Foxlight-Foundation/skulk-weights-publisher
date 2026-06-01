from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from skulk_weights_publisher.vision_extractor import (
    VisionExtractionError,
    _print_dry_run_plan,
    extract_and_publish_vision,
)


def test_print_dry_run_plan_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _print_dry_run_plan("thirdparty/model-vision", "acme/model-vision", tmp_path)

    out = capsys.readouterr().out
    assert "hf://thirdparty/model-vision" in out
    assert "hf://acme/model-vision" in out
    assert "no quantization" in out


def test_dry_run_does_not_touch_the_hub(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # No huggingface_hub functions are patched: if dry-run tried to reach the
    # hub, it would fail. It must return after only printing the plan.
    extract_and_publish_vision(
        "thirdparty/model-vision",
        "acme/model-vision",
        tmp_path,
        token=None,
        dry_run=True,
    )

    assert "vision source repo" in capsys.readouterr().out


def test_extract_and_publish_vision_mirrors_weights(tmp_path: Path) -> None:
    create_calls: list[str] = []
    upload_calls: list[dict[str, Any]] = []

    def fake_create_repo(repo_id: str, **_kw: Any) -> None:
        create_calls.append(repo_id)

    def fake_snapshot_download(
        repo_id: str,
        *,
        local_dir: str,
        **_kw: Any,
    ) -> str:
        # Simulate the hub writing a weight file into the local dir.
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "model.safetensors").write_bytes(b"\x00\x01")
        (Path(local_dir) / "config.json").write_text("{}", encoding="utf-8")
        return local_dir

    def fake_upload_folder(**kw: Any) -> None:
        upload_calls.append(kw)

    with (
        patch("huggingface_hub.create_repo", side_effect=fake_create_repo),
        patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download),
        patch("huggingface_hub.upload_folder", side_effect=fake_upload_folder),
        patch("huggingface_hub.utils.tqdm.disable_progress_bars", lambda: None),
    ):
        extract_and_publish_vision(
            "thirdparty/model-vision",
            "acme/model-vision",
            tmp_path,
            token="hf_tok",
        )

    assert create_calls == ["acme/model-vision"]
    assert len(upload_calls) == 1
    assert upload_calls[0]["repo_id"] == "acme/model-vision"
    assert upload_calls[0]["repo_type"] == "model"
    # delete_patterns must be passed so stale remote files are pruned (true mirror).
    assert upload_calls[0]["delete_patterns"]


def test_extract_and_publish_vision_clears_stale_scratch(tmp_path: Path) -> None:
    # A prior run left a stale shard in the deterministic scratch dir; it must be
    # gone after the next run so it can't be uploaded alongside the fresh snapshot.
    local_dir = tmp_path / "vision" / "acme--model-vision"
    local_dir.mkdir(parents=True)
    (local_dir / "stale-old-shard.safetensors").write_bytes(b"\xff")

    uploaded_files: list[list[str]] = []

    def fake_snapshot_download(
        repo_id: str,
        *,
        local_dir: str,
        **_kw: Any,
    ) -> str:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "model.safetensors").write_bytes(b"\x00")
        return local_dir

    def fake_upload_folder(*, folder_path: str, **_kw: Any) -> None:
        uploaded_files.append(
            sorted(p.name for p in Path(folder_path).rglob("*.safetensors"))
        )

    with (
        patch("huggingface_hub.create_repo", lambda *a, **k: None),
        patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download),
        patch("huggingface_hub.upload_folder", side_effect=fake_upload_folder),
        patch("huggingface_hub.utils.tqdm.disable_progress_bars", lambda: None),
    ):
        extract_and_publish_vision(
            "thirdparty/model-vision",
            "acme/model-vision",
            tmp_path,
            token="hf_tok",
        )

    # Only the fresh file is present — the stale shard was removed before download.
    assert uploaded_files == [["model.safetensors"]]
    assert not (local_dir / "stale-old-shard.safetensors").exists()


def test_extract_and_publish_vision_errors_when_no_weights(tmp_path: Path) -> None:
    def fake_snapshot_download(
        repo_id: str,
        *,
        local_dir: str,
        **_kw: Any,
    ) -> str:
        # Hub returns nothing matching the allow-patterns: no .safetensors lands.
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return local_dir

    with (
        patch("huggingface_hub.create_repo", lambda *a, **k: None),
        patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download),
        patch("huggingface_hub.upload_folder", lambda **k: None),
        patch("huggingface_hub.utils.tqdm.disable_progress_bars", lambda: None),
        pytest.raises(VisionExtractionError, match="no .safetensors weights found"),
    ):
        extract_and_publish_vision(
            "thirdparty/model-vision",
            "acme/model-vision",
            tmp_path,
            token="hf_tok",
        )
