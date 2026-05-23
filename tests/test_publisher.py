from __future__ import annotations

from pathlib import Path

import pytest

from skulk_vindex_publisher import publisher
from skulk_vindex_publisher.manifest import find_entry
from skulk_vindex_publisher.publisher import (
    PublishError,
    build_publish_plan,
    default_scratch_root,
    execute_publish_plan,
)


def test_build_publish_plan_for_full_slice(tmp_path: Path) -> None:
    entry = find_entry("gemma-3-4b-full-q4-k", Path("models.yaml"))
    plan = build_publish_plan(entry, scratch_root=tmp_path)

    assert plan.output_path == tmp_path / "gemma-3-4b-it-full-q4-k.vindex"
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


def test_default_scratch_root_honors_injected_empty_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_VINDEX_SCRATCH", "/tmp/host-scratch")

    assert default_scratch_root(environ={}) == Path.cwd() / ".scratch"
