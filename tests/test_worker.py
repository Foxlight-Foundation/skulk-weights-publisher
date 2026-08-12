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
    _lease_best_effort,
    _preflight_scratch,
    _report_failure_best_effort,
)


def test_sidecar_job_accepts_exact_revision_and_deterministic_destination() -> None:
    job = SidecarJob.from_json(
        {
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
                "job_id": "job-1",
                "source_repository": "Qwen/Qwen3.8-Base",
                "source_revision": "a" * 40,
                "destination_repository": "attacker/redirected",
            }
        )


def test_preflight_scratch_rejects_capacity_below_floor(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="scratch volume"):
        _preflight_scratch(tmp_path / "scratch", 2**63)


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

    client.progress("job", "downloading")


def test_lease_is_retried_after_registry_interruption() -> None:
    """A transient lease request failure keeps the serial worker alive."""
    client = MagicMock()
    client.lease.side_effect = httpx.ReadTimeout("temporary")

    assert _lease_best_effort(client) is None


def test_failure_reporting_preserves_job_during_registry_interruption() -> None:
    """An unavailable failure endpoint cannot terminate or clean retry state."""
    client = MagicMock()
    client.fail.side_effect = httpx.ReadTimeout("temporary")

    assert _report_failure_best_effort(client, "job", "upload failed") is None
