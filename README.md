# daily_for_cursor

Practice repo for Cursor: small coding samples plus a checked-in Agent setup (`AGENTS.md`, `.cursor/rules/`, `.cursor/skills/`).

Read **[Cursor 最佳实践](docs/cursor-best-practices.md)** for the workflow (Plan Mode, context, Rules vs Skills, Cloud Agent). In Agent chat, `/run-tests` runs the suite and `/add-sample` follows the layout below.

## Beam search sample

A small, dependency-free Python demo of **beam search**: keep the top *k* partial sequences while decoding, instead of always taking the single best next token (greedy search).

## Run the demo

```bash
python3 -m beamsearch
```

Greedy decoding locks onto `the cat sat on the mat` because `cat` is the locally better choice after `the`. A wider beam also keeps `dog`, and finishes with the globally better sequence `the dog ran away`.

```bash
python3 -m beamsearch --beam-width 3
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Layout

| Path | Role |
|------|------|
| `docs/cursor-best-practices.md` | Cursor 最佳实践 |
| `AGENTS.md` | Project facts for Agent |
| `.cursor/rules/` | Scoped coding conventions |
| `.cursor/skills/` | Repeatable Agent workflows |
| `beamsearch/search.py` | `beam_search` and `greedy_search` |
| `beamsearch/toy_lm.py` | Tiny hand-written next-token model |
| `beamsearch/__main__.py` | CLI demo |
| `tests/test_search.py` | Unit tests |

No third-party packages are required (Python 3.9+).
