"""Headless, lease-driven MTP sidecar publication worker."""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from skulk_weights_publisher.catalog_adder import base_model_slug
from skulk_weights_publisher.collection_publish import (
    CollectionError,
    collections_disabled,
    file_artifact_in_collection,
)
from skulk_weights_publisher.defaults import DEFAULT_FOXLIGHT_HF_OWNER
from skulk_weights_publisher.mtp_extractor import MtpPublication, extract_mtp

_REVISION_LENGTH = 40
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,159}$")


@dataclass(frozen=True)
class SidecarJob:
    """One immutable-revision extraction job leased from the registry."""

    job_id: str
    source_repository: str
    source_revision: str
    destination_repository: str

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> SidecarJob:
        """Validate a registry job without accepting arbitrary destinations."""

        if raw.get("kind") != "mtp_sidecar":
            raise ValueError("registry returned a non-MTP sidecar job")
        values = {
            key: raw.get(key)
            for key in (
                "job_id",
                "source_repository",
                "source_revision",
                "destination_repository",
            )
        }
        if not all(isinstance(value, str) and value for value in values.values()):
            raise ValueError("registry returned an incomplete sidecar job")
        job_id = str(values["job_id"])
        if _SAFE_JOB_ID.fullmatch(job_id) is None:
            raise ValueError("registry returned an unsafe sidecar job ID")
        revision = str(values["source_revision"])
        if len(revision) != _REVISION_LENGTH or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError("registry returned a mutable source revision")
        source = str(values["source_repository"])
        destination = str(values["destination_repository"])
        expected = f"{DEFAULT_FOXLIGHT_HF_OWNER}/{base_model_slug(source)}-mtp"
        if destination != expected:
            raise ValueError(
                f"destination {destination!r} does not match allowed {expected!r}"
            )
        return cls(
            job_id=job_id,
            source_repository=source,
            source_revision=revision,
            destination_repository=destination,
        )


class RegistryWorkerClient:
    """HTTP client for the registry's backend-only SWP work queue."""

    def __init__(self, base_url: str, token: str, owner: str) -> None:
        """Create a scoped client that cannot access registry administration."""

        self._owner = owner
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )

    def close(self) -> None:
        """Close network resources."""

        self._client.close()

    def lease(self) -> SidecarJob | None:
        """Lease the oldest eligible MTP job, if one exists."""

        response = self._client.post(
            "/api/v1/swp/jobs/lease", params={"owner": self._owner}
        )
        response.raise_for_status()
        body = response.json()
        return SidecarJob.from_json(body) if isinstance(body, dict) else None

    def heartbeat(self, job_id: str) -> None:
        """Renew a lease while extraction or Hub upload remains active."""

        response = self._client.post(
            f"/api/v1/swp/jobs/{job_id}/heartbeat", json={"owner": self._owner}
        )
        response.raise_for_status()

    def progress(self, job_id: str, message: str) -> bool:
        """Persist progress with a short timeout and report endpoint health."""

        try:
            response = self._client.post(
                f"/api/v1/swp/jobs/{job_id}/progress",
                json={"owner": self._owner, "message": message[:1000]},
                timeout=2,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            # Progress is observability, not a publication precondition. The
            # independent heartbeat and lease-bound completion retain safety.
            return False
        return True

    def complete(self, job: SidecarJob, result: MtpPublication) -> None:
        """Commit an immutable publication result to the owning campaign."""

        if result.source_revision != job.source_revision:
            raise ValueError("publication source revision does not match leased job")
        response = self._client.post(
            f"/api/v1/swp/jobs/{job.job_id}/complete",
            json={
                "owner": self._owner,
                "sidecar_repository": result.repository,
                "sidecar_filename": result.filename,
                "source_revision": result.source_revision,
                "sidecar_revision": result.sidecar_revision,
            },
        )
        response.raise_for_status()

    def fail(self, job_id: str, error: str) -> bool:
        """Release a failed job and report whether retry is terminal."""

        response = self._client.post(
            f"/api/v1/swp/jobs/{job_id}/fail",
            json={"owner": self._owner, "error": error[:4000]},
        )
        response.raise_for_status()
        body = response.json()
        return isinstance(body, dict) and body.get("state") == "terminal"


def _heartbeat_loop(
    client: RegistryWorkerClient, job_id: str, stop: threading.Event
) -> None:
    """Renew the active lease until the extraction thread finishes."""

    while not stop.wait(60):
        try:
            client.heartbeat(job_id)
        except httpx.HTTPError:
            # A transient registry interruption must not silently end lease
            # renewal during a multi-gigabyte shard download. Progress and
            # completion still enforce ownership if the lease is truly lost.
            continue


def _preflight_scratch(path: Path, minimum_free_bytes: int) -> None:
    """Refuse extraction before the configured scratch safety floor is crossed."""

    path.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path).free < minimum_free_bytes:
        raise RuntimeError(
            f"scratch volume has less than {minimum_free_bytes} free bytes"
        )


def _cleanup_failed_job(path: Path, *, terminal: bool) -> None:
    """Retain resumable cache for retries and remove terminal job scratch."""

    if terminal:
        shutil.rmtree(path, ignore_errors=True)


def _lease_best_effort(client: RegistryWorkerClient) -> SidecarJob | None:
    """Treat a transient registry outage as an empty poll, not worker exit."""

    try:
        return client.lease()
    except (httpx.HTTPError, ValueError):
        return None


def _report_failure_best_effort(
    client: RegistryWorkerClient, job_id: str, error: str
) -> bool | None:
    """Report failure when reachable while preserving retry state on outage."""

    try:
        return client.fail(job_id, error)
    except (httpx.HTTPError, ValueError):
        return None


def _progress_reporter(
    client: RegistryWorkerClient, job_id: str
) -> Callable[[str], None]:
    """Disable job telemetry after its first failed short-timeout request."""
    enabled = True

    def emit(message: str) -> None:
        nonlocal enabled
        if enabled:
            enabled = client.progress(job_id, message)

    return emit


def _file_mtp_collection_best_effort(
    repository: str, token: str, emit: Callable[[str], None]
) -> None:
    """Index a published sidecar without making collection health authoritative."""

    if collections_disabled():
        emit("mtp: collection filing disabled")
        return
    try:
        file_artifact_in_collection(repository, "mtp-sidecar", token=token)
        emit("mtp: filed in the MTP Sidecars collection")
    except CollectionError as error:
        emit(f"mtp: warning: could not file into collection: {error}")


def _execute_leased_publication(
    client: RegistryWorkerClient,
    job: SidecarJob,
    job_root: Path,
    hub_token: str,
) -> None:
    """Keep one lease alive through extraction, filing, and completion."""
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(client, job.job_id, stop),
        daemon=True,
    )
    heartbeat.start()
    report_progress = _progress_reporter(client, job.job_id)
    try:
        result = extract_mtp(
            source_repo=job.source_repository,
            source_revision=job.source_revision,
            sidecar_repo=job.destination_repository,
            scratch_root=job_root,
            token=hub_token,
            log=report_progress,
        )
        if result is None:
            raise RuntimeError("pinned SWP publication returned no result")
        _file_mtp_collection_best_effort(
            result.repository,
            hub_token,
            report_progress,
        )
        client.complete(job, result)
    finally:
        stop.set()
        heartbeat.join(timeout=5)


def run_forever() -> None:
    """Lease and publish sidecars serially until the process is stopped."""

    base_url = os.environ["SWP_REGISTRY_INGESTION_URL"]
    ingestion_token = os.environ["SWP_REGISTRY_INGESTION_TOKEN"]
    hub_token = os.environ["HF_TOKEN"]
    scratch_root = Path(os.environ.get("SWP_SCRATCH", "/var/lib/swp"))
    minimum_free_bytes = int(
        os.environ.get("SWP_MINIMUM_FREE_BYTES", str(20 * 1024**3))
    )
    owner = f"swp-{uuid4().hex}"
    client = RegistryWorkerClient(base_url, ingestion_token, owner)
    try:
        while True:
            job = _lease_best_effort(client)
            if job is None:
                time.sleep(30)
                continue
            job_id = job.job_id
            job_root = scratch_root / "jobs" / job_id
            try:
                _preflight_scratch(job_root, minimum_free_bytes)
                _execute_leased_publication(client, job, job_root, hub_token)
                shutil.rmtree(job_root, ignore_errors=True)
            except Exception as error:  # noqa: BLE001 - report and retry remotely
                terminal = _report_failure_best_effort(client, job_id, str(error))
                if terminal is not None:
                    _cleanup_failed_job(job_root, terminal=terminal)
                time.sleep(30)
    finally:
        client.close()


def main() -> None:
    """Run the headless SWP worker console entry point."""

    run_forever()
