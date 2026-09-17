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

The three families that still dominate industrial recsys, as tiny pooling and attention models over a hand-written food-delivery catalog. No PyTorch.

The demo user clicks `coffee → cake → ramen` (two cafe, then Japanese):

- **Two-tower (双塔 / DSSM)** — user = mean of history, cosine vs item vectors. Order is ignored, so cafe wins (`latte`). Towers are independent, which is why this is still the default *recall* / ANN architecture.
- **Transformer (SASRec)** — causal self-attention + recency. Last-click ramen dominates ranking (`sushi`).
- **BERT (BERT4Rec)** — bidirectional `[MASK]`. End-of-sequence mask behaves like a session bag (cafe-like). The distinctive trick is **cloze**: `coffee → [MASK] → ramen` uses the future click, so BERT fills `sushi` while a causal mask (left only) still fills `latte`.

```bash
python3 -m recsys_models
python3 -m recsys_models --history coffee,cake,ramen
```

| Path | Role |
|------|------|
| `recsys_models/catalog.py` | 3-d item embeddings and default history |
| `recsys_models/vectors.py` | dot, mean-pool, softmax |
| `recsys_models/models.py` | `TwoTower`, `CausalTransformer`, `Bert4Rec` |
| `recsys_models/__main__.py` | CLI comparison |
| `tests/test_recsys_models.py` | Unit tests |

No third-party packages are required.
