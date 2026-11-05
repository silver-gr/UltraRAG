# Federated Scoring

How `FederatedRetriever` merges sources before the reranker, and how saved items are filtered.

## Rank-normalised merging

Each source scores on its own scale: hybrid sources return RRF scores (about 0.02), saved items
return cosine similarity (0.4-0.9). Those values can't be compared, so the merge ignores them and
uses each node's rank within its source:

```
score = 1 / (k + rank / weight)      # rank starts at 1, k = RRF_K (60)
```

- A rank-r node of a weight-0.5 source ties with a rank-2r node of a weight-1.0 source.
- Each source's share of the `FEDERATED_FINAL_TOP_K` cut follows its weight. With the default
  weights and all four sources active, the 120 slots split roughly into vault 38, books 34,
  conversations 30 and saved items 18, before deduplication and the per-document cap.
- A weight of 0 scores every node 0, so the source only fills slots nothing else wants.

Merged scores say nothing about absolute relevance, so don't compare them to a threshold. Only a
reranker adds that signal, and not every path runs one (see below).

## Reading scores downstream

| Where | Reranker runs? | `node.score` is |
|---|---|---|
| `FederatedQueryEngine.query` (`retrieval.federated_query`) | Yes | Reranker score |
| `FederatedQueryEngine.retrieve` (`UltraRAG.query_federated`: `@all`, the web app, x3lixi) | No | Merge score |
| Research mode (`FederatedRetriever` directly) | No | Merge score |

On the two paths without a reranker, the merge order alone decides which nodes get cut. For example,
`query_federated` keeps the first `RERANK_TOP_N` nodes unless the caller passes `max_sources`.

The source's own score stays in `node.metadata["native_score"]`, for debugging. It is excluded from
the text sent to the reranker and the LLM. Native scores still mix scales, so research mode skips its
score floor for federated results. Information gain and query similarity still stop research.

Without a real reranker, the federated engine applies no `SIMILARITY_THRESHOLD`, and it ignores
`get_reranker`'s cosine-cutoff fallback: on merge scores either would remove every result.

To see the per-source mix of the final list, enable debug logging for `federated_query`:

```
Federated final per-source counts: {'vault': 38, 'books': 34, 'conversations': 30, 'saved_items': 18}
```

## Saved-items filter

`SavedItemsRetriever` pre-filters its kNN search, so filtered rows don't use up `top_k`.

| Variable | Default | Effect |
|---|---|---|
| `SAVED_ITEMS_MIN_QUALITY_FILTER` | `true` | Keep rows where `usefulness_score > 0 OR relevance_score > 0`, the rule x3lixi's unscoped search uses |
| `SAVED_ITEMS_WHERE` | unset | Extra LanceDB SQL predicate, AND-ed with the filter above |

The active predicate is logged once when the table is opened. Items that x3lixi no longer considers
eligible (dead links, rejected, archived) can't be detected from these columns. The x3lixi side has
to remove them from the table.

**Scoped queries:** x3lixi's own search skips this rule when scoped to a viewer's library, because
user items may have no enrichment scores. `ultrarag_service.py` scopes UltraRAG by wrapping
`_retrieve`, and the filter runs inside the wrapped call. A viewer's unscored items are therefore
removed before the scope check, unless x3lixi sets `SAVED_ITEMS_MIN_QUALITY_FILTER=false` for its
scoped calls. It starts a new process per query, so it can set this per call.
