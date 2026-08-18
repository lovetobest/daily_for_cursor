---
name: add-sample
description: Add a new self-contained coding sample to this practice repo. Use when the user wants a new demo, algorithm, or Cursor exercise alongside beamsearch.
---

# Add a sample

Create a small, dependency-free demo that matches the existing `beamsearch/` layout.

## Layout

| Path | Role |
|------|------|
| `<name>/__init__.py` | Public exports only |
| `<name>/<module>.py` | Library code |
| `<name>/__main__.py` | CLI: `python3 -m <name>` |
| `tests/test_<name>.py` | `unittest` cases |

## Steps

1. Pick a short package name (`snake_case`). Do not nest it under `beamsearch/`.
2. Implement the smallest version that demonstrates the idea. Prefer a toy example with a surprising greedy vs better result when it fits.
3. Add tests that lock the interesting behavior, not the implementation details.
4. Run `python3 -m unittest discover -s tests -v` and keep it green.
5. Add a README section: what it is, how to run it, how it differs from a naive approach.
6. Do not add requirements.txt, Docker, or CI unless the user asks.

## Canonical example

Follow `beamsearch/` and `tests/test_search.py`. Read those files before writing new ones.
