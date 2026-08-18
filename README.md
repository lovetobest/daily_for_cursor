# Beam search sample

A small, dependency-free Python demo of **beam search**: keep the top *k* partial sequences while decoding, instead of always taking the single best next token (greedy search).

## Run the demo

```bash
python -m beamsearch
```

Greedy decoding locks onto `the cat sat on the mat` because `cat` is the locally better choice after `the`. A wider beam also keeps `dog`, and finishes with the globally better sequence `the dog ran away`.

```bash
python -m beamsearch --beam-width 3
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Layout

| Path | Role |
|------|------|
| `beamsearch/search.py` | `beam_search` and `greedy_search` |
| `beamsearch/toy_lm.py` | Tiny hand-written next-token model |
| `beamsearch/__main__.py` | CLI demo |
| `tests/test_search.py` | Unit tests |

No third-party packages are required (Python 3.9+).
