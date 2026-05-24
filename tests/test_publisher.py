from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from skulk_vindex_publisher import publisher
from skulk_vindex_publisher.defaults import (
    COLLECTION_ENV_VAR,
    DEFAULT_FOXLIGHT_VINDEX_COLLECTION,
)
from skulk_vindex_publisher.manifest import find_entry
from skulk_vindex_publisher.publisher import (
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

    def fake_add_vindex_to_collection(
        collection_slug: str,
        repo_id: str,
        *,
        token: str | None,
    ) -> None:
        collection_calls.append((collection_slug, repo_id, token))

    monkeypatch.setattr(publisher.shutil, "which", lambda _name: "/usr/bin/larql")
    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    monkeypatch.setattr(
        publisher, "add_vindex_to_collection", fake_add_vindex_to_collection
    )

    execute_publish_plan(
        plan,
        dry_run=False,
        force=False,
        environ={"HF_TOKEN": "hf_write_token"},
    )

    assert commands == [plan.extract_command, plan.publish_command]
    assert collection_calls == [
        (DEFAULT_FOXLIGHT_VINDEX_COLLECTION, entry.hf_repo, "hf_write_token")
    ]


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

    def fake_add_vindex_to_collection(
        collection_slug: str,
        repo_id: str,
        *,
        token: str | None,
    ) -> None:
        collection_calls.append((collection_slug, repo_id, token))

    monkeypatch.setattr(publisher.shutil, "which", lambda _name: "/usr/bin/larql")
    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    monkeypatch.setattr(
        publisher, "add_vindex_to_collection", fake_add_vindex_to_collection
    )

    with pytest.raises(subprocess.CalledProcessError):
        execute_publish_plan(
            plan,
            dry_run=False,
            force=False,
            environ={"HF_TOKEN": "hf_write_token"},
        )

    assert commands == [plan.extract_command]
    assert collection_calls == []


def test_default_scratch_root_honors_injected_empty_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_VINDEX_SCRATCH", "/tmp/host-scratch")

    assert default_scratch_root(environ={}) == Path.cwd() / ".scratch"
