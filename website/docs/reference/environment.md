---
title: Environment Reference
---

The publisher reads a small set of environment values. These values are about
publication, not ordinary catalogue validation.

## `HF_TOKEN`

Required for real publication.

The token must have write access to every target Hugging Face repository used by
the selected catalogue entries and any configured collection. In GitHub
Actions, configure it as a repository secret named `HF_TOKEN`.

## `SKULK_VINDEX_SCRATCH`

Optional.

Sets the scratch root used for local extraction output. LARQL writes vindex
directories here before publication.

If unset, the publisher uses `.scratch` inside the current checkout.

Use fast storage with enough capacity for the selected vindex shape. Full and
expert-server outputs exist to support Skulk's runtime placement split, but the
publisher still needs local scratch space before anything reaches Hugging Face.

Example:

```bash
export SKULK_VINDEX_SCRATCH=/fast/scratch/skulk-vindexes
```

You can override this for one command with `--scratch`:

```bash
skulk-vindex publish \
  --model foxlight/gemma-3-4b-full-q4-k \
  --scratch /fast/scratch/skulk-vindexes
```

## `PATH`

Real publication requires `larql` to be discoverable on `PATH`.

Dry-runs do not require `larql`, because they only print the command plan.

Check the publishing runner with:

```bash
skulk-vindex doctor --publish
```

## `SKULK_VINDEX_COLLECTION`

Optional.

Overrides the collection target for a publish command. If unset, each catalogue
entry uses its own `hf_collection` value. The built-in Foxlight catalogue uses:

```text
FoxlightAI/vindexes-6a124406dd5fb439c431b051
```

Set this only when you intentionally want all selected entries in that publish
run to land in a different collection. Set it to `none` to skip collection
updates for a one-off run.
