# Operator Setup

Real vindex publication is designed for a self-hosted runner because LARQL
extraction writes large local vindex directories before upload. The published
output is what later lets Skulk keep expensive GPU memory focused on the
latency-sensitive inference path while CPU/high-memory LARQL servers host
weight-heavy FFN and expert pieces.

Use hosted GitHub runners for safe validation: install the package, validate the
catalogue, run tests, and dry-run every entry. Use the self-hosted runner when
you are ready to run `larql extract` and `larql publish`.

The Foxlight catalogue is included by default and publishes to the `FoxlightAI`
Hugging Face organization. Successful Foxlight publishes are also added to the
public `Vindexes` collection:

```text
https://huggingface.co/collections/FoxlightAI/vindexes-6a124406dd5fb439c431b051
```

If you maintain your own vindex library, add a checked-in
`skulk-vindex.yaml` that points at your operator manifest and pass that path
through `catalogue_config` when dispatching the workflow.

## GitHub Actions Runner

Register a Linux self-hosted runner with these labels:

```text
self-hosted
linux
larql
vindex
```

First-publish capacity target:

- at least 200 GB scratch space
- stable network path to Hugging Face
- `larql` available on `PATH`

The runner does not need to be the eventual runtime server. It needs enough disk
and network to extract and upload the vindex safely. MoE-tier entries are manual
by default and require substantially more scratch space than the smoke tier. Do
not enable broad MoE publication until the runner has enough disk for the
selected model family.

Set `SKULK_VINDEX_SCRATCH` on the runner if the scratch directory should live
outside the checkout.

## Secrets

Configure this GitHub Actions secret:

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token with write access to the target repos and collections |

Do not commit tokens to this repository.

## Validation

Before enabling the scheduled workflow, install the package and run the local
validation path:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
skulk-vindex doctor
skulk-vindex catalogue validate
skulk-vindex publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

Before real publication, run the stricter preflight:

```bash
skulk-vindex doctor --publish
skulk-vindex publish --model foxlight/gemma-3-4b-full-q4-k --dry-run
```

Then use manual workflow dispatch for a single smoke-tier entry before expanding
publication to more keys.

## Workflow Dispatch

- `model=all`, `tier=smoke`, `dry_run=false` publishes the scheduled smoke set.
- `model=<catalogue-key>` publishes one entry and ignores `tier`.
- `catalogue_config=skulk-vindex.yaml` includes operator catalogue sources.
- `SKULK_VINDEX_COLLECTION` can be set as a repository variable to override
  collection updates for a run; set it to `none` to skip collection updates.
- `dry_run=true` runs the same catalogue resolution path but only prints LARQL
  commands.
- `tier=moe` is for explicit large-model and expert-server publication after
  capacity has been verified.
