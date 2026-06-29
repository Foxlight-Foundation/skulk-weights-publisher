"""Convert an upstream HF model to a GGUF and publish it (own-the-artifact).

The served (llama.cpp / ``llama_server``) engine consumes GGUF only (it cannot
load safetensors at runtime the way mlx-lm can). So for the AMD/served MTP path,
the draft artifact a card points at must be a GGUF. Rather than depend on a
third-party community conversion (whose template/heads quality we don't control
-- the source of the pearsonkyle channel-marker and GLM dropped-heads failures),
SWP converts the official upstream itself, exactly as it re-extracts MTP heads
for the MLX sidecars.

Two kinds of GGUF artifact share this converter:

- **Draft model** (Gemma 4): convert the official standalone assistant/drafter
  (e.g. ``google/gemma-4-31B-it-assistant``) to a small GGUF that a served card
  passes as ``llama-server --model-draft``. The draft's template is irrelevant
  (the base templates), so a clean, controlled draft is all we need.
- **Heads-preserving base** (DeepSeek/GLM/Qwen nextn families): convert the
  ORIGINAL HF base (which still carries the ``nextn`` tensors) to GGUF. Because
  the conversion reads the source checkpoint directly, the MTP heads come along
  -- this is the fix for community quants that silently drop them (GLM).

The conversion itself is a deterministic format transcription via llama.cpp's
``convert_hf_to_gguf.py`` (+ optional ``llama-quantize``); both come from a
llama.cpp checkout, resolved from ``LLAMA_CPP_DIR`` (or on ``PATH``). This module
fails loudly when the tooling is absent, mirroring how the vindex path requires
``larql``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

#: Quantizations llama-quantize accepts that we expose for drafts/bases. A draft
#: is tiny, so Q8_0 (near-lossless) is the sensible default; F16 skips the
#: quantize step entirely.
_KNOWN_PRECISIONS: Final[frozenset[str]] = frozenset(
    {"F16", "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"}
)

#: Env var pointing at a llama.cpp checkout that contains ``convert_hf_to_gguf.py``
#: and a built ``llama-quantize`` (under ``build/bin`` or on ``PATH``).
LLAMA_CPP_DIR_ENV: Final = "LLAMA_CPP_DIR"


class GgufConversionError(RuntimeError):
    """Raised when GGUF conversion or its tooling is unavailable or fails."""


def _resolve_convert_script(environ: os._Environ[str] | None = None) -> Path:
    """Locate llama.cpp's ``convert_hf_to_gguf.py``.

    Resolution order: ``$LLAMA_CPP_DIR/convert_hf_to_gguf.py``. Raises a clear
    error (not a bare FileNotFoundError) when the checkout isn't configured, so a
    misconfigured runner fails actionably before the expensive model download.
    """
    env = os.environ if environ is None else environ
    root = env.get(LLAMA_CPP_DIR_ENV, "").strip()
    if not root:
        raise GgufConversionError(
            f"{LLAMA_CPP_DIR_ENV} is not set; point it at a llama.cpp checkout "
            "containing convert_hf_to_gguf.py (and a built llama-quantize)"
        )
    script = Path(root) / "convert_hf_to_gguf.py"
    if not script.is_file():
        raise GgufConversionError(
            f"convert_hf_to_gguf.py not found under {LLAMA_CPP_DIR_ENV}={root!r}"
        )
    return script


def _resolve_quantize_bin(environ: os._Environ[str] | None = None) -> str:
    """Locate the ``llama-quantize`` binary (under the checkout or on PATH)."""
    import shutil

    env = os.environ if environ is None else environ
    root = env.get(LLAMA_CPP_DIR_ENV, "").strip()
    if root:
        for candidate in (
            Path(root) / "build" / "bin" / "llama-quantize",
            Path(root) / "llama-quantize",
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    found = shutil.which("llama-quantize")
    if found is None:
        raise GgufConversionError(
            "llama-quantize not found (looked under "
            f"{LLAMA_CPP_DIR_ENV}/build/bin and PATH); build llama.cpp or set "
            f"{LLAMA_CPP_DIR_ENV}"
        )
    return found


def gguf_already_published(
    draft_repo: str, filename: str, *, token: str | None
) -> bool:
    """True when ``filename`` already exists in ``draft_repo`` on the Hub."""
    from huggingface_hub import HfApi

    return bool(HfApi().file_exists(draft_repo, filename, token=token))


def output_filename(source_repo: str, precision: str) -> str:
    """Deterministic GGUF filename for a source repo + precision.

    ``org/Foo-assistant`` + ``Q8_0`` -> ``Foo-assistant-Q8_0.gguf``.
    """
    base = source_repo.split("/")[-1]
    return f"{base}-{precision}.gguf"


def convert_and_publish_gguf(
    source_repo: str,
    draft_repo: str,
    scratch_root: Path,
    *,
    token: str | None,
    precision: str = "Q8_0",
    dry_run: bool = False,
    force: bool = False,
    emit: object = None,
) -> str:
    """Convert ``source_repo`` to a GGUF and publish it to ``draft_repo``.

    Downloads the source HF model, runs ``convert_hf_to_gguf.py`` to an F16 GGUF,
    optionally quantizes to ``precision``, and uploads the result. Returns the
    published filename. A re-run is a no-op unless ``force`` (the file is checked
    on the Hub first, like the MTP sidecar path). ``dry_run`` returns the planned
    filename without doing any work.

    The same call serves both artifact shapes (a Gemma draft model and a
    heads-preserving base) because the difference is only which ``source_repo``
    is converted.
    """
    log = emit if callable(emit) else (lambda message: print(message, file=sys.stderr))

    if precision not in _KNOWN_PRECISIONS:
        raise GgufConversionError(
            f"unknown precision {precision!r}; one of {sorted(_KNOWN_PRECISIONS)}"
        )

    filename = output_filename(source_repo, precision)
    if dry_run:
        return filename

    if not force and gguf_already_published(draft_repo, filename, token=token):
        log(f"gguf: {draft_repo}/{filename} already published; skipping (use --force)")
        return filename

    # Resolve tooling up front so a misconfigured runner fails before the
    # multi-GB source download rather than after it.
    convert_script = _resolve_convert_script()
    quantize_bin = None if precision in ("F16", "BF16") else _resolve_quantize_bin()

    from huggingface_hub import create_repo, snapshot_download
    from huggingface_hub.utils.tqdm import (
        disable_progress_bars,  # type: ignore[import-untyped]
    )

    disable_progress_bars()
    create_repo(
        draft_repo, repo_type="model", private=False, exist_ok=True, token=token
    )

    scratch_root.mkdir(parents=True, exist_ok=True)
    cache_dir = str(scratch_root / "_hf_cache")
    log(f"gguf: downloading source model {source_repo}")
    source_dir = snapshot_download(
        repo_id=source_repo,
        token=token,
        cache_dir=cache_dir,
        # The LM weights + config + tokenizer; skip GGUFs already in the source
        # repo and any vision projector (a draft is text-only).
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
    )

    f16_path = scratch_root / output_filename(source_repo, "F16")
    log(f"gguf: converting to F16 -> {f16_path.name}")
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            source_dir,
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ],
        check=True,
    )

    if quantize_bin is None:
        out_path = f16_path
    else:
        out_path = scratch_root / filename
        log(f"gguf: quantizing F16 -> {precision} ({out_path.name})")
        subprocess.run(
            [quantize_bin, str(f16_path), str(out_path), precision], check=True
        )

    from huggingface_hub import upload_file

    log(f"gguf: uploading to hf://{draft_repo}/{filename}")
    upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=filename,
        repo_id=draft_repo,
        repo_type="model",
        token=token,
        commit_message=f"Add GGUF ({precision}) converted from {source_repo}",
    )
    log(f"gguf: published hf://{draft_repo}/{filename}")
    return filename
