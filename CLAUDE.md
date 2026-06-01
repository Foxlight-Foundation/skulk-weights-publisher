# CLAUDE.md — Skulk Weights Publisher (SWP)

Guidance for AI agents working in this repo.

## What this is

**SWP** publishes Skulk model weights — LARQL **vindexes** and **MTP sidecars** —
to HuggingFace. The repo name is `skulk-weights-publisher`; note the local clone
directory is often `skulk-vindex-publisher` (dir name ≠ repo name).

## Toolchain (important)

- **`uv`, not pip.** Common commands:
  - `uv sync --extra dev` — install dev deps
  - `uv run pytest tests/ -q` — tests
  - `uv run --extra dev basedpyright` — type check
  - `uv run ruff check src/ tests/` — lint (line-length 88; select E,F,I,UP,B,SIM)
  - `uv run --extra ui --extra mtp --extra dev pytest tests/` — full suite incl. server + mtp
- **CLI entry:** `skulk-weights`. **GUI entry:** `skulk-ui` (local React/Vite app
  under `ui/`, FastAPI server under `src/skulk_weights_publisher/server/`).
- The `ui/` app uses **yarn** (Yarn 1 classic; `packageManager` pinned). `skulk-ui`
  serves `ui/dist/` at runtime and builds it on first run — it needs a source
  checkout, not a bare wheel install (`SKULK_UI_DIST` overrides the dist path).

## Conventions

- Always branch + PR; never push to `main` directly.
- Gemma 4 differs from Qwen3/DeepSeek: it ships a separate `{model}-assistant`
  drafter rather than embedded `mtp.*` heads. SWP records `assistant_model_repo`
  in the catalog instead of extracting tensors.

## Foxlight docs hub

Cross-project planning, roadmaps, and decision logs (spanning Skulk, SWP,
FoxlightWeb) live in the **private** repo `Foxlight-Foundation/foxlight-docs`.
Read its `INDEX.md` first for anything cross-cutting (e.g. the Gemma 4 MTP
initiative at `initiatives/gemma4-mtp/`). Clone:
`gh repo clone Foxlight-Foundation/foxlight-docs`.
