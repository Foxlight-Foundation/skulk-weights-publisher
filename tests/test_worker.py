"""Tests for the headless registry-leased SWP worker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from skulk_weights_publisher.worker import (
    RegistryWorkerClient,
    SidecarJob,
    _cleanup_failed_job,
    _execute_leased_publication,
    _file_mtp_collection_best_effort,
    _lease_best_effort,
    _preflight_scratch,
    _progress_reporter,
    _report_failure_best_effort,
)


def test_sidecar_job_accepts_exact_revision_and_deterministic_destination() -> None:
    job = SidecarJob.from_json(
        {
            "kind": "mtp_sidecar",
            "job_id": "job-1",
            "source_repository": "Qwen/Qwen3.8-Base",
            "source_revision": "a" * 40,
            "destination_repository": "FoxlightAI/qwen3-8-base-mtp",
        }
    )

    assert job.source_revision == "a" * 40


@pytest.mark.parametrize("revision", ["main", "A" * 40, "a" * 39])
def test_sidecar_job_rejects_mutable_or_noncanonical_revision(revision: str) -> None:
    with pytest.raises(ValueError, match="mutable source revision"):
        SidecarJob.from_json(
            {
                "kind": "mtp_sidecar",
                "job_id": "job-1",
                "source_repository": "Qwen/Qwen3.8-Base",
                "source_revision": revision,
                "destination_repository": "FoxlightAI/qwen3-8-base-mtp",
            }
        )


def test_sidecar_job_rejects_destination_outside_allowlist() -> None:
    with pytest.raises(ValueError, match="does not match allowed"):
        SidecarJob.from_json(
            {
                "kind": "mtp_sidecar",
                "job_id": "job-1",
                "source_repository": "Qwen/Qwen3.8-Base",
                "source_revision": "a" * 40,
                "destination_repository": "attacker/redirected",
            }
        )


@pytest.mark.parametrize("job_id", ["../outside", "/var/lib/swp", "..", "job/child"])
def test_sidecar_job_rejects_unsafe_path_components(job_id: str) -> None:
    """Registry queue data can never escape the bounded scratch root."""
    with pytest.raises(ValueError, match="unsafe sidecar job ID"):
        SidecarJob.from_json(
            {
                "kind": "mtp_sidecar",
                "job_id": job_id,
                "source_repository": "Qwen/Qwen3.8-Base",
                "source_revision": "a" * 40,
                "destination_repository": "FoxlightAI/qwen3-8-base-mtp",
            }
        )


def test_sidecar_job_rejects_other_job_kinds() -> None:
    """The dedicated worker cannot consume a future or misrouted job type."""
    with pytest.raises(ValueError, match="non-MTP sidecar job"):
        SidecarJob.from_json(
            {
                "kind": "full_model_conversion",
                "job_id": "job-1",
                "source_repository": "Qwen/Qwen3.8-Base",
                "source_revision": "a" * 40,
                "destination_repository": "FoxlightAI/qwen3-8-base-mtp",
            }
        )


def test_preflight_scratch_rejects_capacity_below_floor(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="scratch volume"):
        _preflight_scratch(tmp_path / "scratch", 2**63)


def test_preflight_credits_retained_cache_for_exact_job_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumable job is judged against its pre-download effective capacity."""
    job_root = tmp_path / "job"
    cache = job_root / "_hf_cache"
    cache.mkdir(parents=True)
    (cache / "shard.partial").write_bytes(b"x" * 60)
    usage = MagicMock(free=50)
    monkeypatch.setattr("shutil.disk_usage", lambda path: usage)

    _preflight_scratch(job_root, 100)


def test_failed_job_cleanup_retains_retry_cache_and_removes_terminal_scratch(
    tmp_path: Path,
) -> None:
    """Retriable downloads remain resumable while terminal failures are removed."""
    job_root = tmp_path / "job"
    job_root.mkdir()
    (job_root / "partial").write_bytes(b"cached")

    _cleanup_failed_job(job_root, terminal=False)
    assert job_root.exists()

    _cleanup_failed_job(job_root, terminal=True)
    assert not job_root.exists()


def test_progress_is_best_effort_during_registry_interruption() -> None:
    """A telemetry timeout cannot fail an otherwise healthy extraction."""
    client = RegistryWorkerClient("https://registry.example", "token", "worker")
    client._client = MagicMock()  # pyright: ignore[reportPrivateUsage]
    client._client.post.side_effect = httpx.ReadTimeout("temporary")

    assert client.progress("job", "downloading") is False
    assert client._client.post.call_args.kwargs["timeout"] == 2  # pyright: ignore[reportPrivateUsage]


def test_progress_reporter_opens_circuit_after_first_failure() -> None:
    """Repeated upload increments never accumulate registry timeout stalls."""
    client = MagicMock()
    client.progress.return_value = False
    report = _progress_reporter(client, "job")

    report("uploading 2%")
    report("uploading 4%")

    client.progress.assert_called_once_with("job", "uploading 2%")


def test_lease_is_retried_after_registry_interruption() -> None:
    """A transient lease request failure keeps the serial worker alive."""
    client = MagicMock()
    client.lease.side_effect = httpx.ReadTimeout("temporary")

    assert _lease_best_effort(client) is None


def test_malformed_lease_response_is_retried_without_worker_exit() -> None:
    """Invalid queue data behaves as a failed poll instead of crashing startup."""
    client = MagicMock()
    client.lease.side_effect = ValueError("malformed sidecar job")

    assert _lease_best_effort(client) is None


def test_failure_reporting_preserves_job_during_registry_interruption() -> None:
    """An unavailable failure endpoint cannot terminate or clean retry state."""
    client = MagicMock()
    client.fail.side_effect = httpx.ReadTimeout("temporary")

    assert _report_failure_best_effort(client, "job", "upload failed") is None


def test_malformed_failure_response_preserves_worker_retry() -> None:
    """Invalid failure JSON behaves like an unavailable reporting endpoint."""
    client = MagicMock()
    client.fail.side_effect = ValueError("invalid JSON")

    assert _report_failure_best_effort(client, "job", "upload failed") is None


def test_worker_files_published_sidecar_in_mtp_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry-driven publications remain visible in the standard collection."""
    filed: list[tuple[str, str, str | None]] = []
    messages: list[str] = []
    monkeypatch.setattr(
        "skulk_weights_publisher.worker.collections_disabled", lambda: False
    )
    monkeypatch.setattr(
        "skulk_weights_publisher.worker.file_artifact_in_collection",
        lambda repository, artifact_type, token: filed.append(
            (repository, artifact_type, token)
        ),
    )

    _file_mtp_collection_best_effort(
        "FoxlightAI/qwen3-8-mtp", "hf_token", messages.append
    )

    assert filed == [("FoxlightAI/qwen3-8-mtp", "mtp-sidecar", "hf_token")]
    assert messages == ["mtp: filed in the MTP Sidecars collection"]


def test_heartbeat_covers_collection_filing_and_registry_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lease renewal stops only after every publication effect is acknowledged."""
    active = False
    observed: list[str] = []

    class FakeThread:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            nonlocal active
            active = True

        def join(self, timeout: int) -> None:
            nonlocal active
            assert timeout == 5
            active = False

    publication = MagicMock(repository="FoxlightAI/qwen3-8-base-mtp")
    monkeypatch.setattr("skulk_weights_publisher.worker.threading.Thread", FakeThread)
    monkeypatch.setattr(
        "skulk_weights_publisher.worker.extract_mtp",
        lambda **_kwargs: publication if active else None,
    )
    monkeypatch.setattr(
        "skulk_weights_publisher.worker._file_mtp_collection_best_effort",
        lambda *_args: observed.append("collection-active" if active else "stopped"),
    )
    client = MagicMock()
    client.progress.return_value = True
    client.complete.side_effect = lambda *_args: observed.append(
        "completion-active" if active else "stopped"
    )
    job = SidecarJob(
        job_id="job-1",
        source_repository="Qwen/Qwen3.8-Base",
        source_revision="a" * 40,
        destination_repository="FoxlightAI/qwen3-8-base-mtp",
    )

    _execute_leased_publication(client, job, tmp_path, "hf_token")

    assert observed == ["collection-active", "completion-active"]
    assert active is False
