"""Local environment checks for vindex publication."""

from __future__ import annotations

import importlib.util
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from skulk_vindex_publisher.catalogue import load_catalogue_view
from skulk_vindex_publisher.manifest import ManifestError, validate_manifest
from skulk_vindex_publisher.publisher import default_scratch_root


@dataclass(frozen=True)
class DoctorCheck:
    """One preflight check result."""

    name: str
    ok: bool
    message: str


@dataclass(frozen=True)
class DoctorReport:
    """Collection of doctor check results."""

    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        """Return whether every check passed."""

        return all(check.ok for check in self.checks)


def run_doctor(
    *,
    publish: bool = False,
    config_path: Path | None = None,
    manifest_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> DoctorReport:
    """Run local preflight checks and return structured results."""

    env = os.environ if environ is None else environ
    checks: list[DoctorCheck] = []

    if publish:
        larql_path = shutil.which("larql")
        checks.append(
            DoctorCheck(
                name="larql",
                ok=larql_path is not None,
                message=f"found command: larql ({larql_path})"
                if larql_path
                else "missing command: larql",
            )
        )

    has_yaml = importlib.util.find_spec("yaml") is not None
    checks.append(
        DoctorCheck(
            name="pyyaml",
            ok=has_yaml,
            message="found Python package: PyYAML"
            if has_yaml
            else "missing Python package: PyYAML",
        )
    )

    if publish:
        checks.append(
            DoctorCheck(
                name="hf-token",
                ok=bool(env.get("HF_TOKEN")),
                message="HF_TOKEN is set"
                if env.get("HF_TOKEN")
                else "HF_TOKEN is not set; publication will fail",
            )
        )
        has_huggingface_hub = importlib.util.find_spec("huggingface_hub") is not None
        checks.append(
            DoctorCheck(
                name="huggingface-hub",
                ok=has_huggingface_hub,
                message="found Python package: huggingface_hub"
                if has_huggingface_hub
                else "missing Python package: huggingface_hub",
            )
        )

    scratch_root = default_scratch_root(env)
    try:
        scratch_root.mkdir(parents=True, exist_ok=True)
        writable = os.access(scratch_root, os.W_OK)
    except OSError:
        writable = False
    checks.append(
        DoctorCheck(
            name="scratch",
            ok=writable,
            message=f"scratch root writable: {scratch_root}"
            if writable
            else f"scratch root is not writable: {scratch_root}",
        )
    )

    try:
        if manifest_path is not None:
            entry_count = len(validate_manifest(manifest_path))
            message = f"{manifest_path} valid: {entry_count} entries"
        else:
            view = load_catalogue_view(config_path=config_path)
            entry_count = len(view.entries)
            message = (
                f"catalogue valid: {entry_count} entries from "
                f"{len(view.sources)} sources"
            )
        checks.append(DoctorCheck(name="catalogue", ok=True, message=message))
    except ManifestError as exc:
        checks.append(DoctorCheck(name="catalogue", ok=False, message=str(exc)))

    return DoctorReport(checks=tuple(checks))
