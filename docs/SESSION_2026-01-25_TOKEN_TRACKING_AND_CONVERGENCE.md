# Session Report: Token Tracking & Convergence Detection Enhancements

**Date:** 2026-01-25
**Session Focus:** Voyage AI token tracking improvements, convergence detection for research mode

---

## Summary

This session focused on two main areas:
1. **Token tracking improvements** — Separate tracking for embedding models and index/query usage
2. **Research mode convergence detection** — Enhanced stopping criteria with multiple signals

---

## 1. Token Tracking Enhancements

### 1.1 Per-Model History Tracking

**Problem:** When switching embedding models (e.g., voyage-3.5-lite → voyage-4-lite), the previous model's usage was lost.

**Solution:** Added `embedding_history` to preserve historical usage when models change.

**Files Modified:**
- `token_tracker.py` — Added `embedding_history` list, `_archive_embedding_model()` method

**New JSON Structure:**
```json
{
  "embedding": { "model": "voyage-4-lite", "tokens_used": 38844416, ... },
  "embedding_history": [
    { "model": "voyage-3.5-lite", "tokens_used": 63069661, "archived_at": "..." }
  ]
}
```

### 1.2 Index vs Query Token Tracking

**Problem:** Could not distinguish between tokens used for indexing documents vs querying.

**Solution:** Added separate counters for index and query operations.

**Files Modified:**
- `token_tracker.py` — Added `index_tokens`, `query_tokens`, `index_requests`, `query_requests` fields
- `embeddings.py` — Added `usage_type` parameter to tracking calls

**New TokenUsage Fields:**
```python
@dataclass
class TokenUsage:
    model: str
    tokens_used: int = 0
    # NEW: Separate tracking
    index_tokens: int = 0
    query_tokens: int = 0
    index_requests: int = 0
    query_requests: int = 0
```

**Usage Pattern:**
```python
# In embeddings.py
# Document embedding → usage_type="index"
self._tracker.record_embedding_usage(tokens, model, usage_type="index")

# Query embedding → usage_type="query"
self._tracker.record_embedding_usage(tokens, model, usage_type="query")
```

---

## 2. Research Mode Convergence Detection

### 2.1 Previous Implementation

Single criterion: Information gain < 5%

```python
information_gain = new_unique_nodes / total_nodes
if information_gain < 0.05:
    stop()
```

### 2.2 Enhanced Implementation

Multiple stopping criteria with configurable thresholds:

| Criterion | Threshold | Purpose |
|-----------|-----------|---------|
| **Information Gain** | <8% | Primary: few new unique nodes |
| **Score Floor** | avg <0.25 | New nodes have low relevance |
| **Query Similarity** | ≥85% | Reformulated query too similar |
| **Redundancy** | ≥60% | Most retrieved nodes already seen |

**Files Modified:**
- `research_mode.py` — Added `_check_convergence()`, `_check_score_floor()`, `_check_query_similarity()`, `_check_redundancy()` methods

**New Constants:**
```python
CONVERGENCE_THRESHOLD = 0.08      # Was 0.05
SCORE_FLOOR_THRESHOLD = 0.25      # New
QUERY_SIMILARITY_THRESHOLD = 0.85  # New
REDUNDANCY_THRESHOLD = 0.60        # New
```

### 2.3 How It Works

```
Iteration 2:
├── Retrieved 50 nodes
├── Info gain: 10.9% ✓ (>8%)
├── Avg score: 0.42 ✓ (>0.25)
├── Query similarity: 45% ✓ (<85%)
├── Redundancy: 24% ✓ (<60%)
└── Continue to iteration 3...

Iteration 3:
├── Retrieved 45 nodes
├── Info gain: 2.5% ✗ (<8%)
└── STOP: "info_gain=2.5% < 8%"
```

### 2.4 Query Similarity Detection

Uses Jaccard word overlap to detect when query reformulation isn't finding new angles:

```python
def _check_query_similarity(self, new_query: str, previous_queries: List[str]):
    # Tokenize queries
    new_words = set(re.findall(r'\w+', new_query.lower()))

    for prev_query in previous_queries:
        prev_words = set(re.findall(r'\w+', prev_query.lower()))
        # Jaccard similarity
        similarity = len(new_words & prev_words) / len(new_words | prev_words)
        if similarity >= 0.85:
            return True, similarity

    return False, 0.0
```

---

## 3. Other Fixes

### 3.1 Temporal Filter Timezone Fix

**Problem:** `TypeError: can't compare offset-naive and offset-aware datetimes`

**Solution:** Strip timezone info before comparison.

**File:** `temporal_filter.py`
```python
cmp_date = date.replace(tzinfo=None) if hasattr(date, 'tzinfo') and date.tzinfo else date
```

### 3.2 Content Research Page Type Checking

**Problem:** `isinstance(r, ResearchResult)` failing for dict results.

**Solution:** Use `hasattr(r, 'source')` instead.

**File:** `pages/content_research.py`

---

## 4. PWA Icon Update

Added custom icons for PWA installation:

| File | Size | Purpose |
|------|------|---------|
| `static/icon-512.png` | 512x512 | PWA large icon |
| `static/icon-192.png` | 125x125 | PWA small icon |
| `static/favicon-32.png` | 35x35 | Browser favicon |
| `static/apple-touch-icon.png` | 125x125 | iOS home screen |

**Files Modified:**
- `static/manifest.json` — Updated icon sizes
- `app.py` — Changed `page_icon` to use custom icon file

---

## 5. Commits

### Commit 1: `73b4e52`
```
feat: add per-model token history and index/query usage tracking

- Track embedding tokens separately for indexing vs querying operations
- Add embedding_history to preserve usage when switching models
- Auto-archive model stats when model changes
- Fix temporal filter timezone comparison (naive vs aware datetime)
- Fix content research page type checking (hasattr vs isinstance)
- Remove chunk preview truncation in markdown export
```

### Commit 2: (pending)
```
feat: enhance research mode with multi-criteria convergence detection

- Increase info gain threshold from 5% to 8%
- Add score floor criterion (avg score < 0.25)
- Add query similarity detection (Jaccard >= 85%)
- Add redundancy detection (>= 60% duplicates)
- Track previous queries for similarity comparison
```

---

## 6. Current Token Usage (Voyage AI)

| Model | Used | Limit | Remaining |
|-------|------|-------|-----------|
| voyage-4-lite | 38,844,416 | 200,000,000 | 161,155,584 |
| voyage-3.5-lite (archived) | 63,069,661 | — | — |
| rerank-2.5 | 2,385,650 | 200,000,000 | 197,614,350 |

---

## 7. Remaining Optimization Opportunities

### 7.1 Shared Embedding Space (voyage-4)

**Opportunity:** voyage-4-lite, voyage-4, voyage-4-large share embedding space and have **separate 200M quotas**.

**Potential Optimization:**
1. Index documents with `voyage-4-large` (best quality)
2. Query with `voyage-4-lite` (cheapest)
3. No re-indexing needed when switching query models

**Status:** Not yet implemented. Would require config separation for index vs query models.

### 7.2 Contextual Retrieval

**Opportunity:** Anthropic's technique prepends LLM-generated context to chunks before embedding (35-67% fewer retrieval failures).

**Status:** Exists as opt-in feature but not enabled by default due to LLM cost at indexing time.

### 7.3 Research Content Index

**Consideration:** User plans to separate personal notes from research content. Research content may benefit from:
- Lower convergence threshold (5%)
- Higher redundancy tolerance (80%)
- More iterations by default

**Status:** Not yet implemented. Would require per-index configuration.

---

## 8. Files Changed This Session

| File | Changes |
|------|---------|
| `token_tracker.py` | embedding_history, index/query tracking |
| `embeddings.py` | usage_type parameter |
| `research_mode.py` | Multi-criteria convergence |
| `temporal_filter.py` | Timezone fix |
| `pages/content_research.py` | Type checking fix |
| `research_storage.py` | Chunk preview fix |
| `static/manifest.json` | PWA icons |
| `static/*.png` | New icon files |
| `app.py` | Custom favicon |

---

*Report generated: 2026-01-25*
