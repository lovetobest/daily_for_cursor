# Beam search sample

A small, dependency-free Python demo of **beam search**: keep the top *k* partial sequences while decoding, instead of always taking the single best next token (greedy search).

## Run the demo

```bash
python3 -m beamsearch
```

Greedy decoding locks onto `the cat sat on the mat` because `cat` is the locally better choice after `the`. A wider beam also keeps `dog`, and finishes with the globally better sequence `the dog ran away`.

```bash
python3 -m beamsearch --beam-width 3
```

## DAG scheduler

A second dependency-free demo schedules a directed acyclic graph of tasks. A task starts when its own parents have succeeded, so independent branches overlap. A failed task skips its descendants; other branches keep running.

```bash
python3 -m dag
python3 -m dag --fail-check
```

The sample sales pipeline totals 65. `--fail-check` fails `quality_check`, skips `publish`, and still produces the report. Design notes, in Chinese: [docs/dag.md](docs/dag.md).

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Layout

| Path | Role |
|------|------|
| `beamsearch/search.py` | `beam_search` and `greedy_search` |
| `beamsearch/toy_lm.py` | Tiny hand-written next-token model |
| `beamsearch/__main__.py` | CLI demo |
| `dag/graph.py` | Tasks, edges, cycle check, topological levels |
| `dag/scheduler.py` | Dynamic ready-set scheduler |
| `dag/demo.py` | Sales-pipeline DAG |
| `dag/__main__.py` | CLI demo |
| `docs/dag.md` | What a DAG is, and how this scheduler is put together |
| `tests/` | Unit tests |

No third-party packages are required (Python 3.9+).
