# Golden-set retrieval evaluation & ablations

**Added:** 2026-10-22 · **Files:** `eval_harness.py`, `scripts/golden_set.py`, `scripts/run_ablation.py`, `tests/test_eval_harness.py`

The RAGAS harness (`docs/EVALUATION.md`) scores *answers* and needs an LLM judge per case.
This harness scores *retrieval* against a human-confirmed, **document-level** golden set, with
no LLM in the loop at scoring time. It is the tool for deciding which of the ~40 retrieval
flags actually help on this vault.

Why document level: chunk ids change on every re-index; `file_path` does not. A golden set
keyed on documents survives chunking, embedding and index-time changes — the things you most
want to ablate.

## 1. Build the golden set (once, then grow it)

```bash
# 1. mine real queries from data/users/*/query_history.json (no API calls)
uv run python -m scripts.golden_set mine --n 100 --min-greek 20 \
    --add "Τι έχω σημειώσει για τον ύπνο;" --add "my notes on RAG evaluation"

# 2. retrieve + LLM-grade candidate documents (uses the configured LLM; ~1 call per 15 docs)
uv run python -m scripts.golden_set propose            # per-query scope from the @prefix used
uv run python -m scripts.golden_set propose --scope vault   # or force vault-only

# 3. review: open tests/golden/review.md to read, edit tests/golden/proposed.json:
#    - fix `relevant` (doc_id -> 1 | 2), remove wrong docs, add docs you know should be there
#    - set "status": "confirmed"
# 4. finalize
uv run python -m scripts.golden_set finalize
uv run python -m scripts.golden_set stats
```

Notes
- History yields ~44 unique queries today (lots of re-runs). Add 30–40 by hand with `--add`
  to reach ~80, and make sure ~20 are Greek. Prune generation-style queries ("create a morning
  routine…") — they don't test retrieval.
- Candidates are pooled from the full pipeline **plus** vector-only **plus** BM25-only so the
  labels aren't biased toward one configuration. Documents none of them surface are never
  seen by the grader; add those by hand when you know them.
- `finalize --accept-proposed` skips human review. Fine for a first provisional baseline;
  don't tune against it for long.
- `golden_set.json` is merged, not replaced, on `finalize` — hand-written entries survive.

Format (`tests/golden/golden_set.json`):
```json
{"version": 1, "queries": [
  {"id": "g0003", "query": "…", "relevant": {"Notes/Habits.md": 2, "Notes/Sleep.md": 1},
   "language": "el", "category": "exhaustive", "scope": "vault", "status": "confirmed"}
]}
```
`relevant` keys are the `file_path` metadata of vault/book/conversation nodes, or `url` for
TheSource items (see `eval_harness.doc_id`).

## 2. Run ablations

```bash
uv run python -m scripts.run_ablation --save-baseline                      # current .env
uv run python -m scripts.run_ablation --grid QUERY_TRANSFORM_METHOD=none,hyde,multi_query
uv run python -m scripts.run_ablation --grid RRF_K=20,60,100 --grid FUSION_BM25_WEIGHT=0.5,1.0,1.5
uv run python -m scripts.run_ablation --grid USE_SELF_CORRECTION=true,false --grid ENABLE_MMR=true,false
uv run python -m scripts.run_ablation --grid RERANK_SCORE_THRESHOLD=0,0.1,0.2
uv run python -m scripts.run_ablation --grid RETRIEVAL_CANDIDATES=100,150,250
```

Output: `data/evaluation/ablations/<timestamp>/results.md` (one row per config) and
`results.json`, plus one JSON per config with per-query rows, the ranked doc ids, and the
relevant docs that were **missed** — the fastest way to see *why* a config loses.

Metrics (all document level):
| metric | stage | tells you |
|---|---|---|
| `recall@20 (pre)`, `recall@50 (pre)` | candidate set, before reranker | is fusion/transform finding the docs at all? If low, no reranker can fix it |
| `ndcg@10`, `recall@10`, `mrr`, `hit@1`, `p@10` | after reranker + postprocessors | what the synthesizer actually sees |
| `t_total_ms` | — | retrieval latency incl. any LLM calls (HyDE, self-correction) |

Breakdowns by `category` (factual / conceptual / exhaustive / comparison) and `language`
(en / el / mixed) are in each config's section — Greek and exhaustive slices are where
configs usually diverge.

### Index-level flags
`EMBEDDING_MODEL`, `CHUNKING_STRATEGY`, `USE_CONTEXTUAL_RETRIEVAL`, `VAULT_TABLE`, … change
the index, so the runner reloads the system for them. To ablate contextual retrieval:

```bash
# build the alternate table once
USE_CONTEXTUAL_RETRIEVAL=true VAULT_TABLE=obsidian_ctx uv run python scripts/reindex_vault.py
# then compare
uv run python -m scripts.run_ablation --grid VAULT_TABLE=obsidian_embeddings,obsidian_ctx
```

### Regression gate
```bash
uv run python -m scripts.run_ablation --baseline data/evaluation/baseline.json --gate 3
```
Exit code 1 if `ndcg@10` or `recall@20 (pre)` drops more than 3 points vs the saved baseline.
Run it before merging any retrieval change; re-save the baseline when a change is accepted.

## 3. How the probe works
`eval_harness.RetrievalProbe` builds the same `HybridQueryEngine` / `FederatedQueryEngine`
as `main.UltraRAG`, then calls the engine's **retriever** (→ pre-rerank candidates) and its
**node postprocessors** (→ post-rerank) directly. One retrieval pass, no synthesis, no result
cache. Self-correction and HyDE still run when enabled because they are part of retrieval.
The federated scope calls `retrieve_candidates` and `rerank_candidates`, the same steps as
`FederatedQueryEngine.retrieve`, which `UltraRAG.query_federated` uses.

Known limits
- Research mode (iterative) is not probed; it is an answer-level feature — use RAGAS for it.
- Pooling bias (above). Grow `relevant` by hand over time.
- The LLM grader uses the configured `LLM_MODEL`. Use a different family from your
  synthesizer when you later judge answers with RAGAS, to limit self-preference.
