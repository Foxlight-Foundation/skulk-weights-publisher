# Operator Setup

Real weight publication is designed for a self-hosted runner because extraction
writes large local files before upload. SWP publishes three artifact types:

- **LARQL vindexes** (`--artifact vindex`) via `larql extract` + `larql publish`
  — lets Skulk keep expensive GPU memory focused on the attention path while
  CPU/high-memory LARQL servers host the weight-heavy FFN and expert pieces.
- **MTP sidecars** (`--artifact mtp`) — extracts and quantizes native
  multi-token prediction heads (`mtp.*` tensor keys) from the BF16 checkpoint
  and publishes them as `mtp.safetensors` to a dedicated Hugging Face repo.
- **Vision sidecars** (`--artifact vision`) — mirrors a third-party vision
  encoder into a Foxlight-owned repo **byte-for-byte** (no quantization, no
  dtype conversion) so the published encoder is numerically identical to
  upstream. Configured with the `vision_source_repo` (upstream weights) and
  `vision_sidecar_repo` (Foxlight-owned mirror) catalog fields, which are set
  together or not at all.

Use hosted GitHub runners for safe validation: install the package, validate the
catalog, run tests, and dry-run every entry. Use the self-hosted runner when
you are ready to perform real extraction and publication.

The Foxlight catalog is included by default and publishes to the `FoxlightAI`
Hugging Face organization. Each publish is filed into a per-artifact-type
Hugging Face collection — `Vindexes`, `MTP Sidecars`, and `Vision Sidecars`.
The sidecar collections are resolved by title and created if missing; the
vindex collection is the configured `Vindexes` slug:

```text
https://huggingface.co/collections/FoxlightAI/vindexes-6a124406dd5fb439c431b051
```

Set `SKULK_WEIGHTS_COLLECTION` to one of `none`, `0`, `false`, `no`, `off`, or
`disabled` to skip collection filing for a run.

## Self-describing model cards

Every real publish — vindex, MTP, or vision — also uploads a self-describing
`README.md` model card to the published repo. The card records provenance
(`base_model` set to the source repo, the pinned source commit SHA, quant,
catalog key, and `extracted_with`) and the source model's license. The source
license and commit SHA are resolved best-effort from the Hub using `HF_TOKEN`,
and published artifacts inherit the source model's license **unchanged** — SWP
never re-licenses.

If you maintain your own weight library, add a checked-in `skulk-weights.yaml`
that points at your operator manifest and pass that path through `catalog_config`
when dispatching the workflow.

## GitHub Actions Runner

Register a Linux self-hosted runner with these labels:

```text
self-hosted
linux
larql
vindex
```

First-publish capacity targets:

**Vindex publication:**
- at least 200 GB scratch space for smoke-tier entries
- stable network path to Hugging Face
- `larql` available on `PATH`

**MTP sidecar publication (additional):**
- scratch space for the BF16 checkpoint download before quantization
  (typically 15–30 GB per model, on top of vindex scratch)
- `HF_TOKEN` must have read access to the MTP source repo (usually the same
  upstream model as the vindex entry)
- install the `mtp` extras on the extraction runner: `uv sync --extra mtp`;
  this installs `safetensors` on all platforms and `mlx` on macOS Apple Silicon
  only (platform-gated in pyproject.toml — Linux runners will not get mlx)
- real MTP extraction requires macOS (Apple Silicon); the standard Linux runner
  (`self-hosted, linux, larql, vindex`) cannot run mlx — a separate macOS
  Apple Silicon runner or step is needed for MTP publication

The runner does not need to be the eventual runtime server. It needs enough disk
and network to extract and upload safely. MoE-tier entries are manual by default
and require substantially more scratch space than the smoke tier. Do not enable
broad MoE publication until the runner has enough disk for the selected model
family.

Set `SKULK_WEIGHTS_SCRATCH` on the runner if the scratch directory should live
outside the checkout.

## Secrets

Configure this GitHub Actions secret:

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token with write access to target repos and collections; also needs read access to MTP source repos |

Do not commit tokens to this repository.

## Validation

Before enabling the scheduled workflow, install the package and run the local
validation path:

```bash
uv sync --extra dev
skulk-weights doctor
skulk-weights catalog validate
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

Before real publication, run the stricter preflight:

```bash
skulk-weights doctor --publish
skulk-weights publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

For entries with MTP configured, dry-run the MTP step too (no extra packages
needed for dry-run — the download and quantization are skipped):

```bash
skulk-weights publish --model my-org/my-model --artifact mtp --dry-run
```

For real MTP extraction, run the publish command directly on a macOS (Apple
Silicon) machine — the GitHub Actions workflow is pinned to the Linux vindex
runner and does not install `.[mtp]` or support MTP extraction. MTP publication
is a CLI-only operation until a separate macOS workflow path is added:

```bash
uv sync --extra mtp
skulk-weights publish --model my-org/my-model --artifact mtp
```

Then use manual workflow dispatch for a single smoke-tier **vindex** entry
before expanding vindex publication to more keys.

## Workflow Dispatch

- `model=all`, `tier=smoke`, `dry_run=false` publishes the scheduled smoke set.
- `model=<catalog-key>` publishes one entry and ignores `tier`.
- `catalog_config=skulk-weights.yaml` includes operator catalog sources.
- `SKULK_WEIGHTS_COLLECTION` can be set as a repository variable to override
  collection updates for a run; set it to `none` to skip collection updates.
- `dry_run=true` runs the same catalog resolution path but only prints commands.
- `tier=moe` is for explicit large-model and expert-server publication after
  capacity has been verified.
