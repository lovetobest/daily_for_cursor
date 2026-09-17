# daily_for_cursor

Small, dependency-free Python samples (Python 3.9+).

```bash
python3 -m unittest discover -s tests -v
```

## Beam search

Keep the top *k* partial sequences while decoding, instead of always taking the single best next token (greedy search).

```bash
python3 -m beamsearch
python3 -m beamsearch --beam-width 3
```

Greedy decoding locks onto `the cat sat on the mat` because `cat` is the locally better choice after `the`. A wider beam also keeps `dog`, and finishes with the globally better sequence `the dog ran away`.

| Path | Role |
|------|------|
| `beamsearch/search.py` | `beam_search` and `greedy_search` |
| `beamsearch/toy_lm.py` | Tiny hand-written next-token model |
| `beamsearch/__main__.py` | CLI demo |
| `tests/test_search.py` | Unit tests |

## Recommendation models (two-tower / Transformer / BERT)

These are not three drop-in SOTA replacements. They sit on **different layers of the funnel**: two-tower (and SASRec-as-user-tower, semantic BERT) for **recall**; DIN/BST for **candidate-aware ranking**; BERT4Rec for sequential MLM.

Same session `coffee → cake → ramen` (cafe, cafe, then ramen):

- **Two-tower (双塔)** — one user vector, cosine ANN. Majority cafe → `latte`.
- **SASRec user tower** — still one user vector (last hidden), so still ANN-able, but last-click wins → `sushi`.
- **DIN** — a *different* user vector per candidate (target attention). Ranking only.
- **BERT4Rec** — end `[MASK]` looks like a session bag; cloze `coffee → [MASK] → ramen` is where bidirectionality matters.
- **Semantic BERT dual-encoder** — encode title tokens; cold-start `udon` retrieves without clicks.

```bash
python3 -m recsys_models              # full walkthrough (漏斗 / 损失 / attention)
python3 -m recsys_models --short      # three next-item lists only
python3 -m recsys_models --history coffee,cake,ramen
```

Longer notes: [`docs/recsys-models.md`](docs/recsys-models.md).

| Path | Role |
|------|------|
| `recsys_models/catalog.py` | id embeddings + title tokens, including cold-start `udon` |
| `recsys_models/models.py` | `TwoTower`, `CausalTransformer`, `Bert4Rec` |
| `recsys_models/din.py` | candidate-aware ranker |
| `recsys_models/semantic.py` | BERT dual-encoder over tokens |
| `recsys_models/losses.py` | InfoNCE |
| `recsys_models/pipeline.py` | recall then rank |
| `recsys_models/lecture.py` | CLI walkthrough |
| `tests/test_recsys_models.py` | Unit tests |

No third-party packages are required.
