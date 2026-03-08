# Session Report: Federated Retrieval Optimizations (2026-02-20)

## Why This Was Done
Your federated query logs showed the system was working, but it was doing too much work per query:

- HyDE query transformation was triggered in parallel per source (same query transformed multiple times).
- Retrieval volume was very high (`600 -> 200` repeatedly).
- Source list often contained duplicates/near-duplicates.
- Long latency for broad questions due to repeated retries and large candidate sets.

This report explains what was changed, why it helps, and what trade-offs to expect.

---

## What Changed (High Level)

We optimized 4 areas:

1. **HyDE deduped across parallel sources**
2. **Federated retrieval limits made explicit/configurable**
3. **Stronger chunk deduplication (content-level)**
4. **Per-document diversity cap before final ranking**

Files changed:

- `/Users/silver/Projects/UltraRAG/query_transform.py`
- `/Users/silver/Projects/UltraRAG/federated_query.py`
- `/Users/silver/Projects/UltraRAG/config.py`

---

## 1) HyDE Deduped Across Parallel Sources

### Problem (Before)
In federated mode, multiple sources (vault/conversations/books) can be queried in parallel.
Each source independently asked the LLM to generate a HyDE transformation for the same query.

Result:
- multiple identical LLM calls,
- extra latency,
- extra token usage.

### Fix (After)
Added an **in-flight lock + event map** in `query_transform.py`.

Behavior now:
- First thread starts HyDE generation.
- Other threads with same normalized query wait briefly.
- When first thread finishes, result is cached and waiting threads reuse it.

### Why It Helps
- Fewer duplicate LLM calls.
- Lower latency under federated parallel retrieval.
- Lower model usage/cost for the same query.

### Trade-off
- Very small synchronization overhead.
- If first HyDE call fails, waiting threads may fall back to normal behavior after timeout.

---

## 2) Federated Retrieval Limits (New Config Knobs)

### Problem (Before)
Federated retrieval used broad limits by default, often returning large candidate pools.
That increases retrieval/reranking load and can slow response time.

### Fix (After)
Added explicit retrieval controls in `config.py`:

- `federated_top_k_per_source` (default: `100`)
- `federated_final_top_k` (default: `120`)
- `federated_max_chunks_per_document` (default: `3`)

Environment variable support:

- `FEDERATED_TOP_K_PER_SOURCE`
- `FEDERATED_FINAL_TOP_K`
- `FEDERATED_MAX_CHUNKS_PER_DOCUMENT`

### Why It Helps
- Predictable performance envelope.
- Easier tuning without code changes.
- Reduces unnecessary context volume for synthesis.

### Trade-off
- Lower limits can miss edge-case relevant chunks (recall loss) if set too low.
- Recommended to tune gradually and validate on real queries.

---

## 3) Stronger Deduplication (Content-Aware)

### Problem (Before)
Dedup relied mainly on node IDs. Similar or repeated content could still pass through if node IDs differed.

### Fix (After)
In `federated_query.py`, dedup now uses:

- `node_id` (existing), and
- **content signature**: `file identity + normalized text hash`.

This catches practical duplicates across repeated chunks/doc variants.

### Why It Helps
- Cleaner source lists.
- Less repeated context sent into synthesis.
- Better signal-to-noise.

### Trade-off
- Tiny CPU overhead for normalization + hashing.
- Very similar but not identical text may still appear (by design; avoids over-aggressive removal).

---

## 4) Per-Document Diversity Cap

### Problem (Before)
A single long document could dominate top results with many chunks, reducing source diversity.

### Fix (After)
Added per-document cap in `federated_query.py`:
- Keep at most `federated_max_chunks_per_document` chunks from the same document before final top-k cutoff.

Default is `3`.

### Why It Helps
- More diverse evidence in final retrieval set.
- Better downstream summaries and citations.
- Reduces “same doc repeated 10 times” effect.

### Trade-off
- If one document truly contains most relevant material, cap can suppress some useful detail.
- Increase cap for deep single-document analysis tasks.

---

## “Explain It Like I’m New”: End-to-End Flow (Now)

When you ask a federated question:

1. Query sent to federated retriever.
2. Sources retrieve in parallel.
3. HyDE transformation is generated once and reused if concurrent.
4. Each source returns candidates.
5. Candidates are merged.
6. Duplicates removed (ID + content hash).
7. Per-document cap applied for diversity.
8. Final top-k selected.
9. Self-correction decides if retrieval quality is enough; may retry with refined query.
10. Final synthesis runs on curated set.

---

## What You Should Expect in Logs

Positive indicators:

- Lower repeated HyDE LLM calls under parallel source retrieval.
- Federated retrieval log now includes a diversification stage:
  - `total -> unique -> diversified -> final`
- Fewer duplicate source snippets in displayed results.

---

## Suggested Tuning Defaults (Starting Point)

If you want a balanced setup:

- `FEDERATED_TOP_K_PER_SOURCE=80`
- `FEDERATED_FINAL_TOP_K=100`
- `FEDERATED_MAX_CHUNKS_PER_DOCUMENT=3`
- keep `SELF_CORRECTION_MAX_RETRIES=2` (or `1` for lower latency)

If quality appears too narrow, increase in this order:

1. `FEDERATED_FINAL_TOP_K`
2. `FEDERATED_TOP_K_PER_SOURCE`
3. `FEDERATED_MAX_CHUNKS_PER_DOCUMENT`

---

## Known Limitation

This optimization reduces duplicate and redundant retrieval work, but does not by itself guarantee perfect relevance grading by the self-correction module. Broad advisory queries may still trigger retries.

---

## Summary

These changes optimize **efficiency and result quality structure** without changing your core architecture:

- fewer redundant transformation calls,
- better dedupe,
- more source diversity,
- tunable performance via config/env.

This should reduce latency/cost noise while preserving strong retrieval quality for federated queries.

