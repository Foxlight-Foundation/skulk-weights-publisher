---
title: Troubleshooting
---

## `models.yaml not found`

Run commands from the repository root, or pass `--manifest PATH`.

## `missing command: larql`

Install LARQL and make sure it is on `PATH`. This is required only for real
publication, not for normal validation or dry-runs.

## `HF_TOKEN is not set`

Real publication needs a Hugging Face token with write access to the target
repository.

## `output path already exists`

The publisher refuses to overwrite local extraction output by default. Remove
the directory manually, choose another scratch root, or rerun with `--force`
when replacement is intentional.
