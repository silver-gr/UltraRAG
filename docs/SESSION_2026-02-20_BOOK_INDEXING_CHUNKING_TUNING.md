# Session Report: Book Indexing & Chunking Tuning (2026-02-20)

## Context
- Dataset was reduced to **100 books** (all EPUB, no PDF) for an initial indexing phase.
- Indexing run showed healthy batch behavior:
  - Batch 1: create `books` table
  - Batch 2+ : append (`Adding to books index...`)
- No hard failures were observed in the provided logs.

## Run Snapshot
- Books discovered: **100**
- Documents loaded: **100** (1 per EPUB)
- Chunking mode: batched (10 docs per batch)

## Measured Embedding Usage (Current Settings)
- Current chunk settings: `chunk_size=1024`, `chunk_overlap=128`
- Total nodes: **15,195**
- Total embedding tokens: **10,614,394**
- Avg tokens/node: **698.55**
- Max tokens/node: **1274**

## Chunking Sweep (Same 100-Book Dataset)
Measured by re-chunking the same set with alternative settings.

| Label | chunk_size | overlap | nodes | embedding tokens | node delta vs current | token delta vs current |
|---|---:|---:|---:|---:|---:|---:|
| current | 1024 | 128 | 15,195 | 10,614,394 | baseline | baseline |
| balanced+ | 1200 | 96 | 13,473 | 10,337,906 | -11.3% | -2.6% |
| cost-lean | 1400 | 96 | 12,279 | 10,290,817 | -19.2% | -3.0% |
| leaner | 1600 | 80 | 11,409 | 10,211,849 | -24.9% | -3.8% |
| aggressive | 1800 | 80 | 10,753 | 10,198,569 | -29.2% | -3.9% |
| same-size lower-overlap | 1024 | 64 | 14,839 | 10,226,087 | -2.3% | -3.7% |

## Interpretation
- Increasing chunk size reduces node count significantly.
- Total embedding tokens drop only modestly (~3–4% max in this sweep).
- Main trade-off is **retrieval behavior**, not cost:
  - Smaller chunks / higher overlap: better pinpoint recall, more redundancy/noisy top-k.
  - Larger chunks / lower overlap: better coherence, fewer nodes, lower granularity for narrow facts.

## Quality-First Recommendation
If quality is the priority, tune conservatively:
1. Keep chunk size near current (`1024` to `1200`).
2. Reduce overlap moderately (`128 -> 96` or `64`) to cut redundancy.
3. Evaluate on real queries before moving to large chunks (`1600+`).

Suggested first A/B candidate:
- `chunk_size=1200`, `chunk_overlap=96`

## Minor Data Quality Issue Noted
- Some loaded/chunked titles include a leading separator (example: `" - Mindfulness in Plain English"`).
- This is not a blocker, but title normalization cleanup is recommended for metadata quality.

## Next Test Plan
When retuning later:
1. Keep a fixed query set (representative user questions).
2. Compare current (`1024/128`) vs candidate (`1200/96`) on:
   - answer relevance
   - citation quality
   - retrieval stability for narrow factual queries
3. Promote new settings only if quality is equal or better.

