from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from skulk_weights_publisher import publisher
from skulk_weights_publisher.defaults import (
    COLLECTION_ENV_VAR,
    DEFAULT_FOXLIGHT_VINDEX_COLLECTION,
)
from skulk_weights_publisher.manifest import find_entry
from skulk_weights_publisher.publisher import (
    PublishError,
    build_publish_plan,
    default_scratch_root,
    execute_publish_plan,
    resolve_publish_collection,
)


def test_build_publish_plan_for_full_slice(tmp_path: Path) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)

    assert plan.output_path == tmp_path / "gemma-3-4b-it-full-q4-k.vindex"
    assert plan.collection_slug == DEFAULT_FOXLIGHT_VINDEX_COLLECTION
    assert plan.extract_command == (
        "larql",
        "extract",
        "google/gemma-3-4b-it",
        "-o",
        str(plan.output_path),
        "--quant",
        "q4k",
    )
    assert plan.publish_command[-1] == "none"
    assert (
        f"https://huggingface.co/collections/{DEFAULT_FOXLIGHT_VINDEX_COLLECTION}"
        in "\n".join(plan.summary_lines(force=False))
    )


def test_build_publish_plan_for_expert_server_slice(tmp_path: Path) -> None:
    entry = find_entry("gemma-4-26b-a4b-expert-server-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)

    assert plan.publish_command[-1] == "expert-server"
    assert "expert-server" in "\n".join(plan.summary_lines(force=False))


def test_execute_publish_plan_dry_run_skips_preflight(tmp_path: Path) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)

    execute_publish_plan(plan, dry_run=True, force=False, environ={})


def test_execute_publish_plan_requires_hf_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)
    monkeypatch.setattr(publisher.shutil, "which", lambda _name: "/usr/bin/larql")
    monkeypatch.setenv("HF_TOKEN", "host-token")

    with pytest.raises(PublishError, match="HF_TOKEN"):
        execute_publish_plan(plan, dry_run=False, force=False, environ={})


def test_resolve_publish_collection_honors_environment_override() -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))

    assert resolve_publish_collection(entry, environ={}) == entry.hf_collection
    assert (
        resolve_publish_collection(
            entry,
            environ={COLLECTION_ENV_VAR: "acme/vindexes-0123456789abcdef01234567"},
        )
        == "acme/vindexes-0123456789abcdef01234567"
    )
    assert (
        resolve_publish_collection(entry, environ={COLLECTION_ENV_VAR: "none"}) is None
    )


def test_resolve_publish_collection_rejects_bad_override() -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))

    with pytest.raises(PublishError, match=COLLECTION_ENV_VAR):
        resolve_publish_collection(entry, environ={COLLECTION_ENV_VAR: "not-a-slug"})


def test_execute_publish_plan_adds_repo_to_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)
    commands: list[tuple[str, ...]] = []
    collection_calls: list[tuple[str, str, str | None]] = []

    def fake_run(
        command: tuple[str, ...], *, check: bool
    ) -> subprocess.CompletedProcess[tuple[str, ...]]:
        assert check
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    def fake_file_in_collection(
        repo_id: str,
        artifact_type: str,
        *,
        token: str | None,
        collection_slug: str | None = None,
        note: str | None = None,
    ) -> None:
        collection_calls.append((artifact_type, repo_id, collection_slug))

    card_calls: list[dict[str, object]] = []
    monkeypatch.setattr(publisher.shutil, "which", lambda _name: "/usr/bin/larql")
    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    monkeypatch.setattr(
        publisher, "file_artifact_in_collection", fake_file_in_collection
    )
    monkeypatch.setattr(
        publisher, "publish_model_card", lambda **kw: card_calls.append(kw)
    )

    execute_publish_plan(
        plan,
        dry_run=False,
        force=False,
        environ={"HF_TOKEN": "hf_write_token"},
    )

    assert commands == [plan.extract_command, plan.publish_command]
    # The vindex is filed into the configured collection slug (honored exactly).
    assert collection_calls == [
        ("vindex", entry.hf_repo, DEFAULT_FOXLIGHT_VINDEX_COLLECTION)
    ]
    # A self-describing card is published to the vindex repo.
    assert len(card_calls) == 1
    assert card_calls[0]["repo_id"] == entry.hf_repo
    assert card_calls[0]["artifact_type"] == "vindex"
    assert card_calls[0]["catalog_key"] == entry.key


def test_execute_publish_plan_does_not_publish_after_extract_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)
    commands: list[tuple[str, ...]] = []
    collection_calls: list[tuple[str, str, str | None]] = []

    def fake_run(
        command: tuple[str, ...], *, check: bool
    ) -> subprocess.CompletedProcess[tuple[str, ...]]:
        assert check
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    def fake_file_in_collection(
        repo_id: str,
        artifact_type: str,
        *,
        token: str | None,
        note: str | None = None,
    ) -> None:
        collection_calls.append((artifact_type, repo_id, token))

    monkeypatch.setattr(publisher.shutil, "which", lambda _name: "/usr/bin/larql")
    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    monkeypatch.setattr(
        publisher, "file_artifact_in_collection", fake_file_in_collection
    )
    monkeypatch.setattr(publisher, "publish_model_card", lambda **kw: None)

    with pytest.raises(subprocess.CalledProcessError):
        execute_publish_plan(
            plan,
            dry_run=False,
            force=False,
            environ={"HF_TOKEN": "hf_write_token"},
        )

    assert commands == [plan.extract_command]
    assert collection_calls == []


def _make_mtp_entry() -> object:
    from skulk_weights_publisher.manifest import ManifestEntry

    return ManifestEntry(
        key="acme/qwen3-6-7b-full-q4-k",
        source_model="Qwen/Qwen3-6-7B",
        quant="q4k",
        tier="smoke",
        slices=("full",),
        output_name="qwen3-6-7b-full-q4-k.vindex",
        hf_repo="acme/qwen3-6-7b-full-q4-k-vindex",
        mtp_source_repo="Qwen/Qwen3-6-7B",
        mtp_sidecar_repo="acme/qwen3-6-7b-mtp-int4",
        mtp_quant="q4k",
    )


def test_build_publish_plan_populates_mtp_step(tmp_path: Path) -> None:
    from skulk_weights_publisher.publisher import MtpSidecarStep

    entry = _make_mtp_entry()
    plan = build_publish_plan(entry, scratch_root=tmp_path)  # type: ignore[arg-type]

    assert plan.mtp_step == MtpSidecarStep(
        source_repo="Qwen/Qwen3-6-7B",
        sidecar_repo="acme/qwen3-6-7b-mtp-int4",
        mtp_quant="q4k",
    )


def test_build_publish_plan_no_mtp_step_when_not_configured(tmp_path: Path) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)

    assert plan.mtp_step is None


def test_summary_lines_includes_mtp_details(tmp_path: Path) -> None:
    entry = _make_mtp_entry()
    plan = build_publish_plan(entry, scratch_root=tmp_path)  # type: ignore[arg-type]
    summary = "\n".join(plan.summary_lines(force=False, artifact="mtp"))

    assert "Qwen/Qwen3-6-7B" in summary
    assert "acme/qwen3-6-7b-mtp-int4" in summary
    assert "q4k" in summary


def test_summary_lines_mtp_not_configured_note(tmp_path: Path) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)
    summary = "\n".join(plan.summary_lines(force=False, artifact="mtp"))

    assert "not configured" in summary


def test_summary_lines_reports_per_type_sidecar_collections(tmp_path: Path) -> None:
    # Sidecars go into their own per-type collections; the dry-run must say so
    # (the configured slug is the vindex collection only).
    entry = _make_mtp_entry()
    plan = build_publish_plan(
        entry,  # type: ignore[arg-type]
        scratch_root=tmp_path,
        collection_slug="FoxlightAI/vindexes-0123456789abcdef01234567",
    )
    summary = "\n".join(plan.summary_lines(force=False, artifact="all"))

    assert "vindex collection:" in summary
    assert "mtp collection:   MTP Sidecars" in summary


def test_execute_publish_plan_mtp_calls_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk_weights_publisher.publisher as pub_mod

    entry = _make_mtp_entry()
    plan = build_publish_plan(entry, scratch_root=tmp_path)  # type: ignore[arg-type]
    extract_calls: list[tuple[str, str, str, Path]] = []

    def fake_extract_mtp(
        source_repo: str,
        sidecar_repo: str,
        mtp_quant: str,
        scratch_root: Path,
        *,
        token: str | None,
        dry_run: bool = False,
        force: bool = False,
        catalog_key: str | None = None,
    ) -> None:
        extract_calls.append((source_repo, sidecar_repo, mtp_quant, scratch_root))

    monkeypatch.setattr(pub_mod.shutil, "which", lambda _name: "/usr/bin/larql")
    monkeypatch.setattr(pub_mod.subprocess, "run", lambda *a, **kw: None)

    import skulk_weights_publisher.mtp_extractor as mtp_mod
    monkeypatch.setattr(mtp_mod, "extract_mtp", fake_extract_mtp)

    execute_publish_plan(
        plan,
        dry_run=False,
        force=False,
        artifact="mtp",
        environ={"HF_TOKEN": "hf_tok"},
    )

    assert len(extract_calls) == 1
    assert extract_calls[0][:3] == (
        "Qwen/Qwen3-6-7B",
        "acme/qwen3-6-7b-mtp-int4",
        "q4k",
    )


def test_execute_publish_plan_mtp_artifact_without_step_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)
    monkeypatch.setattr(publisher.shutil, "which", lambda _name: "/usr/bin/larql")

    with pytest.raises(PublishError, match="no MTP sidecar configured"):
        execute_publish_plan(
            plan,
            dry_run=False,
            force=False,
            artifact="mtp",
            environ={"HF_TOKEN": "hf_tok"},
        )


def _make_vision_entry() -> object:
    from skulk_weights_publisher.manifest import ManifestEntry

    return ManifestEntry(
        key="acme/kimi-k2-5-full-q4-k",
        source_model="acme/Kimi-K2.5",
        quant="q4k",
        tier="moe",
        slices=("full",),
        output_name="kimi-k2-5-full-q4-k.vindex",
        hf_repo="acme/kimi-k2-5-full-q4-k-vindex",
        vision_source_repo="thirdparty/Kimi-K2.5-vision",
        vision_sidecar_repo="acme/kimi-k2-5-vision",
    )


def test_build_publish_plan_populates_vision_step(tmp_path: Path) -> None:
    from skulk_weights_publisher.publisher import VisionSidecarStep

    entry = _make_vision_entry()
    plan = build_publish_plan(entry, scratch_root=tmp_path)  # type: ignore[arg-type]

    assert plan.vision_step == VisionSidecarStep(
        source_repo="thirdparty/Kimi-K2.5-vision",
        sidecar_repo="acme/kimi-k2-5-vision",
    )


def test_build_publish_plan_no_vision_step_when_not_configured(tmp_path: Path) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)

    assert plan.vision_step is None


def test_summary_lines_includes_vision_details(tmp_path: Path) -> None:
    entry = _make_vision_entry()
    plan = build_publish_plan(entry, scratch_root=tmp_path)  # type: ignore[arg-type]
    summary = "\n".join(plan.summary_lines(force=False, artifact="vision"))

    assert "thirdparty/Kimi-K2.5-vision" in summary
    assert "acme/kimi-k2-5-vision" in summary
    assert "no quantization" in summary


def test_summary_lines_vision_not_configured_note(tmp_path: Path) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)
    summary = "\n".join(plan.summary_lines(force=False, artifact="vision"))

    assert "not configured" in summary


def test_execute_publish_plan_vision_calls_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _make_vision_entry()
    plan = build_publish_plan(entry, scratch_root=tmp_path)  # type: ignore[arg-type]
    calls: list[tuple[str, str, Path]] = []

    def fake_extract_vision(
        source_repo: str,
        sidecar_repo: str,
        scratch_root: Path,
        *,
        token: str | None,
        dry_run: bool = False,
        target_model: str | None = None,
        catalog_key: str | None = None,
    ) -> None:
        calls.append((source_repo, sidecar_repo, scratch_root))

    import skulk_weights_publisher.vision_extractor as vision_mod

    monkeypatch.setattr(vision_mod, "extract_and_publish_vision", fake_extract_vision)

    execute_publish_plan(
        plan,
        dry_run=False,
        force=False,
        artifact="vision",
        environ={"HF_TOKEN": "hf_tok"},
    )

    assert len(calls) == 1
    assert calls[0][:2] == ("thirdparty/Kimi-K2.5-vision", "acme/kimi-k2-5-vision")


def test_execute_publish_plan_vision_artifact_without_step_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)
    monkeypatch.setattr(publisher.shutil, "which", lambda _name: "/usr/bin/larql")

    with pytest.raises(PublishError, match="no vision sidecar configured"):
        execute_publish_plan(
            plan,
            dry_run=False,
            force=False,
            artifact="vision",
            environ={"HF_TOKEN": "hf_tok"},
        )


def test_default_scratch_root_honors_injected_empty_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_WEIGHTS_SCRATCH", "/tmp/host-scratch")

    assert default_scratch_root(environ={}) == Path.cwd() / ".scratch"


def test_summary_lines_includes_mtp_output_path(tmp_path: Path) -> None:
    entry = _make_mtp_entry()
    plan = build_publish_plan(entry, scratch_root=tmp_path)  # type: ignore[arg-type]
    summary = "\n".join(plan.summary_lines(force=False, artifact="mtp"))

    expected_path = tmp_path / "acme--qwen3-6-7b-mtp-int4-mtp.safetensors"
    assert str(expected_path) in summary
