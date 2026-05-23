---
title: Environment Reference
---

The publisher reads a small set of environment values. These values are about
publication, not ordinary catalogue validation.

## `HF_TOKEN`

Required for real publication.

The token must have write access to every target Hugging Face repository used by
the selected manifest entries. In GitHub Actions, configure it as a repository
secret named `HF_TOKEN`.

## `SKULK_VINDEX_SCRATCH`

Optional.

Sets the scratch root used for local extraction output. LARQL writes vindex
directories here before publication.

If unset, the publisher uses `.scratch` inside the current checkout.

Example:

```bash
export SKULK_VINDEX_SCRATCH=/fast/scratch/skulk-vindexes
```

You can override this for one command with `--scratch`:

```bash
skulk-vindex publish \
  --model gemma-3-4b-full-q4-k \
  --scratch /fast/scratch/skulk-vindexes
```

## `PATH`

Real publication requires `larql` to be discoverable on `PATH`.

Dry-runs do not require `larql`, because they only print the command plan.

Check the publishing runner with:

```bash
skulk-vindex doctor --publish
```
