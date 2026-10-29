# UltraRAG — State-of-the-Art Review, October 2026

**Date:** 2026-10-22
**Scope:** What changed in embeddings, rerankers, retrieval techniques and evaluation since the January 2026 review; what UltraRAG (v1.6.x, commit 7b54a59) should actually change.
**Reviewer note:** My training data ends June 2026. Everything dated after that below comes from web sources fetched today and is cited. Confidence is marked per recommendation.

---

## 0. Bottom line

**UltraRAG is not archaic.** Checked against today's landscape, the component choices are current:

| Component | UltraRAG today | Landscape, Oct 2026 | Verdict |
|---|---|---|---|
| Embeddings | `voyage-4-large` (Jan 2026) | Still Voyage's top general-text model; nothing newer from Voyage for general text except `voyage-context-4` | Keep; spike `voyage-context-4` for long sources |
| Reranker | `rerank-3` (Sept 30, 2026) | Three weeks old; +0.96% over rerank-2.5 | Done, nothing to do |
| Fusion | Vector + BM25, RRF, tunable | Still the production default everywhere | Keep; tune with data |
| Contextual retrieval | Implemented, **off** | Best-evidenced single technique | Turn on, measure |
| Late chunking | Implemented, **off** | Superseded by contextualized-embedding models | Replace with `voyage-context-4` spike |
| Self-RAG/CRAG, research mode, RAPTOR, graph expansion | All present | Still current patterns | Keep |
| Evaluation | RAGAS harness + IR metrics exist | Harness exists, **no golden set in use** | **This is the real gap** |

The honest diagnosis: you have ~40 retrieval flags and no measurement telling you which ones help on *your* corpus. The 2025 chunking paper below found technique gains are **model- and corpus-dependent and often small**; that means the next 15% is not in a new model, it is in an eval loop that lets you tune what you already built. Everything in this document is ordered on that premise.

Expected gain from the P1 items combined: **+10–20% on retrieval nDCG@10 / recall@20**, confidence medium. I can't be more precise without the baseline — which is the point.

---

## 1. Priority 1 — Build the measurement loop (week 1)

### 1.1 Golden set from your own query history
**Current state:** `tests/evaluation_dataset.json` exists as a format; `evaluation.py` computes precision/recall/MRR/nDCG when `relevant_docs` is supplied, but nothing supplies it. `ENABLE_EVALUATION_LOGGING` logs score stats only.
**Action:**
1. Mine `data/users/*/query_history.json` for ~150 real queries. Dedupe, stratify: factual / conceptual / exhaustive ("all my notes on…") / cross-lingual (EN→EL, EL→EN) / source-specific (@books, @conv).
2. For each, label **document-level** relevance (which notes/books/conversations should appear), not chunk-level. Cheapest path: run current retrieval at top-40, have an LLM propose relevant/not per doc, **you confirm** in a spreadsheet. Budget 2–3 hours of your time for ~80 queries. Doc-level labels are what `RetrievalEvaluator` already consumes (`file_path`).
3. Keep 20 of those as Greek-query cases. Your Jan 2026 `BILINGUAL_RETRIEVAL_REPORT` showed 50–65% cross-lingual hit rates — that number should be tracked, not assumed.
4. Separately, use RAGAS `TestsetGenerator` to synthesize ~40 Q/A pairs for the end-to-end metrics (faithfulness, answer relevancy). Synthetic sets are fine for generation metrics; they are a poor proxy for retrieval quality on personal notes, so keep the two sets distinct.

**Metrics to report per run:** recall@20 on the *pre-rerank* candidate set (tells you whether the fusion stage is the bottleneck), nDCG@10 and MRR *post-rerank*, faithfulness, p50/p95 latency, cost per query. One table, one row per config.

### 1.2 Ablation runner
**Action:** a script that takes a list of env overrides and runs the retrieval-only metrics (no LLM judge needed, so it's fast and ~free) across the golden set. First matrix:

| Flag | Values | Why |
|---|---|---|
| `USE_CONTEXTUAL_RETRIEVAL` | off / on | Best-evidenced technique, currently off |
| `QUERY_TRANSFORM_METHOD` | none / hyde / multi_query | HyDE is the default; see §3.2 |
| `RRF_K`, `FUSION_BM25_WEIGHT` | 20/60/100 × 0.5/1.0/1.5 | Greek morphology may want more or less BM25 |
| `RERANK_SCORE_THRESHOLD` | 0 / 0.1 / 0.2 | Currently disabled |
| `ENABLE_MMR` | off / on | Exhaustive queries |
| `RETRIEVAL_CANDIDATES` | 100 / 150 / 250 | Over-fetch ratio |
| `ENABLE_QUERY_DECOMPOSITION` | off / on | Multi-part queries only |

Run it once, keep the table in `docs/`. From then on, every config change gets a row.

### 1.3 LLM-as-judge hygiene
Use a judge model from a different family than the generator (your config generates with `gpt-6.1-sol`; judge with Gemini 3.8 Flash, or vice-versa) to blunt self-preference bias. RAGAS `answer_relevancy` embeds with an embedding model — set it explicitly to your Voyage model so scores are comparable across runs.

### 1.4 Regression gate
Single-user system: a nightly CI job is overkill. A `make eval` you run before merging config changes, that exits non-zero if nDCG@10 or recall@20 drops >3 points from the stored baseline, is enough. DeepEval's `assert_test` under pytest is the easiest way to express that if you want it in `run_tests.sh`; RAGAS remains better for the dataset-level comparisons ([comparison](https://qaskills.sh/blog/deepeval-vs-ragas-llm-evaluation-2026)).

**Impact:** High — it converts every other item here from a guess into a measurement. **Effort:** Medium (3–4 days). **Confidence that this is the right first move:** high.

---

## 2. Embeddings — what's new and what to do

### 2.1 Landscape (verified today)

| Model | Date | Key facts | Relevance to you |
|---|---|---|---|
| **voyage-4-large** (current) | Jan 2026 | MoE, shared space with voyage-4/-lite/-nano, 32K ctx, 1024-d default | Still Voyage's best general model. No successor in 2026 ([Voyage blog index](https://blog.voyageai.com/)) |
| **voyage-context-4** | Jun 29, 2026 | Contextualized chunk embeddings; chunk vector reflects whole doc. +0.45% avg vs voyage-4-large, **+7.1% on long documents (LongEmbed)**, +2.1% chunk-level vs context-3; beat Anthropic-style contextual retrieval by 6.8% and Jina late chunking by 23.7% in v3 evals. $0.12/1M, 200M free. Auto-chunking optional. **Shared-space compatibility with voyage-4 not stated** ([Voyage](https://blog.voyageai.com/2026/06/29/voyage-context-4/)) | The native version of your `late_chunking` strategy. Biggest expected win on books and long conversations, small on the vault |
| **voyage-code-4** | Aug 13, 2026 | +14–28% over code-3 on code retrieval; $0.12/1M; no prose benchmarks ([Voyage](https://blog.voyageai.com/2026/08/13/voyage-code-4/)) | Not for a mixed vault. Skip |
| **Microsoft Harrier-OSS-v1** (27B / 0.6B / 270M) | Apr 7, 2026 | Open weights (MIT per [supermemory ranking](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)); MTEB-multilingual-v2 **74.3** (27B, rank 1), 69.0 (0.6B), 100+ languages, 32K ctx ([Bing blog](https://blogs.bing.com/search/April-2026/Microsoft-Open-Sources-Industry-Leading-Embedding-Model)) | The new open-weight leader, ahead of Qwen3-8B (70.9). 27B won't fit your RTX 3060; **0.6B is a credible local/offline model** |
| Qwen3-Embedding-8B / 4B | Jun 2025 | 70.9 / 69.6 multilingual, Apache 2.0 | Already in your config as `qwen3-8b`; now second-tier open |
| **Gemini Embedding 2** | May 20, 2026 | Multimodal, **8K context only**, $0.20/1M ([OpenRouter](https://openrouter.ai/google/gemini-embedding-2)) | Short context and higher price; voyage-code-4 beat it by 31% on code. Skip |
| jina-embeddings-v5 | 2026 | CC BY-NC 4.0 | Non-commercial licence is fine for a personal system, but I found no benchmark numbers today — unverified |
| OpenAI text-embedding-3-large | 2024 | Trails Voyage by 14–29% on Voyage's evals | Skip |

Caveat on comparability: Voyage reports on its own 100-dataset suite and RTEB; open models report MTEB. **Cross-vendor numbers are not comparable**; only your golden set is.

### 2.2 Recommendation
- **Vault:** stay on `voyage-4-large`. Confidence high. Nothing released in 2026 beats it for API-served general text on evidence I could verify, and your Greek cross-lingual retrieval is empirically working.
- **Books and conversations tables:** run a **`voyage-context-4` spike**. These are exactly the long-document case where it reports +7%. Your `BookChunker` already produces chunk lists per document, so use `contextualized_embed(..., enable_auto_chunking=False)` with your own chunks — one context-aware vector per chunk, chunk boundaries unchanged. Because shared-space compatibility is undocumented, keep it in **separate LanceDB tables** (you already have `books` and `conversations` tables) and query each table with the model that built it. Cost to embed the book library once at $0.12/1M is trivial (likely <$10; free tier covers it). Confidence medium: the +7% is Voyage's own number on LongEmbed, not your corpus.
- **Retire the DIY `late_chunking` strategy** once context-4 is measured. The 2025 study below found late chunking's effect flips sign depending on the embedding model; a model trained for contextual chunk embeddings is the principled replacement.
- **Local fallback:** add `harrier-oss-v1-0.6b` as a `sentence-transformers` option for offline/degraded mode or for the TheSource ingestion box if you want to stop paying per-token there. Don't mix its vectors with Voyage tables.

### 2.3 Dimensions / quantization
18–20K vault chunks + books is small. Leave 1024-d float. `ENABLE_QUANTIZATION` stays off; it only matters at ~1M+ vectors.

---

## 3. Retrieval techniques — what the 2025–26 evidence actually says

### 3.1 Contextual retrieval — turn it on (P1)
**Current state:** `USE_CONTEXTUAL_RETRIEVAL=false` with the comment "costs tokens".
**Evidence:** Anthropic's original numbers (35% → 49% → 67% fewer retrieval failures as you add contextual BM25 and reranking) remain the best-documented technique gains in the field; the academic replication ([Reconstructing Context, arXiv 2504.19754](https://arxiv.org/html/2504.19754v1)) found contextual retrieval slightly ahead of late chunking on the same setup (nDCG@10 0.308 vs 0.294) and that **fusion + reranking was the consistent winner** — which you already have. Unlike late chunking, the context blurb also improves BM25 and the reranker's input ([comparison](https://dreaming.press/posts/2026-06-23-late-chunking-vs-contextual-retrieval.html)).
**Cost:** With Gemini 3.8 Flash at $0.75/M input and context caching of the parent document, ~20K vault chunks × ~100 output tokens is on the order of **$5–15 one-time**, then incremental for changed notes. Your `ContextualRetrieval` doc already describes the per-chunk prompt; just route it through a cached-document call.
**Note:** heading ancestry (added in 1.6.1) is a cheap, deterministic subset of this. Keep both; measure whether the LLM blurb adds on top of ancestry for the vault — it may not for short, well-titled notes, and will for books and conversations.
**Impact:** High. **Effort:** Low (flag + one re-index). **Confidence:** medium-high.

### 3.2 HyDE as default — probably wrong in 2026 (P1 to measure)
HyDE's gains were demonstrated on 2023-era embedders without rerankers. With voyage-4-large + rerank-3 the upside is unproven, and the downsides are real: an extra LLM round-trip on every query, and hypothetical documents that anchor retrieval on hallucinated specifics for personal-vault queries ("my notes on X" has no public prior). The 2026 synthesis documents I checked don't list it at all ([SOTA 2026 synthesis](https://github.com/dpsdagain/RAG/blob/main/rag_state_of_the_art_2026.md)). Run the ablation; my expectation (confidence low-medium) is `none` or `multi_query` wins for interactive queries and HyDE at most helps research mode.

### 3.3 Late interaction (ColBERT) — available, not a priority (P3)
LanceDB now supports **multivector columns with MaxSim scoring natively** (list-of-vectors schema, IVF_PQ index, cosine only; PyLate + `GTE-ModernColBERT-v1` example) ([LanceDB docs](https://docs.lancedb.com/search/multivector-search)). So the plumbing cost is low. But: the strong open ColBERT models are English-first; multilingual options (jina-colbert-v2) are non-commercial-licensed and unbenchmarked on Greek; index size is 30–100× single-vector; and `rerank-3` already does token-level interaction at stage 2 over your 150 candidates. For a 20K-chunk corpus a ColBERT first stage buys recall at the candidate stage only if recall@150 is actually your bottleneck — which §1 will tell you. Experiment only if recall@20 pre-rerank comes back below ~0.8.

### 3.4 Fusion tuning (P2, cheap)
`RRF_K=60` and 1.0/1.0 weights are defaults, not tuned values. Greek inflection means BM25 (with your Greek stemmer) and dense search disagree more than in English; the right weight is an empirical question. Grid in §1.2.

### 3.5 Research mode — move from fixed iterations to tool-loop (P2)
Your research mode (gap analysis → sub-queries → convergence at 8% gain) is a 2024–25 design. The pattern that has displaced it in 2026 is simpler: expose `search(query, source, filters)` as a **tool** and let a capable model (GPT-6.1 Sol, Gemini 3.8 Flash/4 Argon) decide when to search, what to search, and when to stop, with a hard cap on calls. It typically needs fewer iterations because the model sees the raw results and reformulates precisely instead of your prompt guessing gaps. Keep your convergence guards as the hard cap. Measure on the "exhaustive" slice of the golden set — that's where it should show. Confidence medium.

### 3.6 Graph — keep wikilink expansion, skip entity-graph RAG
1-hop wikilink expansion is the right amount of graph for a personal vault: deterministic, zero cost, matches how you authored links. Full GraphRAG entity extraction (even LazyGraphRAG) adds an LLM indexing pass and a maintenance burden with gains demonstrated on multi-hop QA over corpora you didn't write. You already have Neo4j in x3lixi-OS for other reasons; don't route retrieval through it unless an eval slice shows multi-hop failures.

### 3.7 RAPTOR — keep for books only
Collapsed-tree RAPTOR over books gives chapter/book-level summaries that chunk retrieval can't; for the vault, heading ancestry + contextual blurbs cover the same need more cheaply.

### 3.8 Self-correction — keep, but gate it
2 retries with LLM relevance grading is fine. Add a skip rule: if post-rerank top-1 score is above your (to-be-set) `RERANK_SCORE_THRESHOLD` high-water mark, skip grading. Saves an LLM call on the easy 70% of queries; §1 tells you where that threshold sits because rerank-3 scores are calibrated to rerank-2.5's distribution ([Voyage](https://blog.voyageai.com/2026/09/30/rerank-3/)).

### 3.9 Caching
Exact-match LRU exists. A semantic cache (embed query, hit if cosine > 0.95 within TTL) is cheap to add but single-user value is low; P3.

---

## 4. Reranker — nothing to do

`rerank-3`: +0.96% over 2.5 overall, +3.35% on long docs, +2.7% over Cohere Rerank 4 Pro, +3.0% over Qwen3-Reranker-8B; same price ($0.05/M, 200M free), same 32K context, scores calibrated to 2.5 ([Voyage](https://blog.voyageai.com/2026/09/30/rerank-3/)). You're on it. `rerank-3-lite` matches 2.5 quality at 40% of the price if the free tier ever runs out — a latency option, not a quality one. Greek isn't named in the 31-language eval set; your cross-lingual test covers that.

---

## 5. LLM layer (brief; this is your Council's domain, not a RAG question)

- Your config now defaults synthesis to `gpt-6.1-sol` (released Sept 29, 2026; the Astra variant was cancelled on safety grounds) ([Wikipedia](https://en.wikipedia.org/wiki/GPT-6.1)). Fine for synthesis.
- For the **cheap, high-volume** calls — contextual blurbs, relevance grading, gap analysis, RAGAS judging — **Gemini 3.8 Flash** (Sept 2, 2026; $0.75/$3.75 per M, 1M context, cache reads $0.075/M, `thinking_level` minimal→high) is the current price/performance point ([listing](https://llmgateway.io/models/gemini-3.8-flash)). `gemini-3.5-flash` is deprecated and auto-routes to 3.6; update `LLM_FALLBACK_MODELS` and the gap-analysis model string accordingly ([Google docs](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5)). Gemini 4 Argon exists; I did not verify its pricing today.
- `llm_factory.py` maps your `thinking_effort` to `thinking_level`; note Google changed the **default from high to medium** — set it explicitly for grading (`minimal`) vs synthesis (`high`).

---

## 6. Per-source notes (TheSource / conversations / books)

The January per-source strategy docs still hold. Two updates:
- **Books**: this is where `voyage-context-4` should pay off most (long, structurally coherent documents). Spike here first.
- **Conversations**: your turn-aware chunker + heading ancestry is good; the conversation-level summary node pattern remains the right answer for "find the conversation where…" queries. Context-4 second.
- **TheSource**: authority/freshness weighting is a scoring problem, not an embedding problem; leave the model alone, tune weights with a TheSource slice in the golden set.

---

## 7. Execution plan

| When | Item | Effort | Expected effect |
|---|---|---|---|
| Week 1 | §1 golden set + ablation runner + baseline table | 3–4 days | Everything else becomes measurable |
| Week 1 | §3.1 contextual retrieval on; re-index; measure | 1 day + ~$10 | Likely the single largest retrieval gain |
| Week 2 | §3.2 HyDE/multi-query/none ablation; set default from data | ½ day | Latency −1 LLM call/query if `none` wins |
| Week 2 | §3.4 RRF grid; §3.8 rerank threshold from score distribution | ½ day | Small, free |
| Week 2 | §2.2 `voyage-context-4` spike on `books` table (separate table) | 1 day | +5–7% on book retrieval if Voyage's LongEmbed number transfers |
| Week 3 | §3.5 tool-loop research mode behind a flag; measure on exhaustive slice | 2–3 days | Fewer iterations, better coverage on "all my notes on X" |
| Week 3 | §5 model-string hygiene (3.5 → 3.8 Flash), thinking levels per call type | 1 hour | Cost/latency |
| Backlog | §3.3 ColBERT first stage, §3.9 semantic cache, harrier-0.6b local | — | Only if evals point there |

Total new spend: well under $50 one-time; monthly unchanged.

---

## 8. What I'm not confident about

- Whether `voyage-context-4` vectors are compatible with `voyage-4-large` vectors. Voyage doesn't say. Separate tables sidestep it; if you want one table, test cosine agreement on 100 identical chunks before trusting it.
- Whether contextual blurbs add anything on top of heading ancestry for *short vault notes*. They clearly should for books/conversations.
- HyDE's sign on your corpus. I expect it to be neutral-to-negative interactively; that's a prior, not a result.
- Harrier's Greek quality specifically; "100+ languages" is all the source gives.
- Gemini 4 Argon pricing and whether it changes the cheap-call recommendation; unverified today.

---

## Sources

- [Voyage: rerank-3 and rerank-3-lite (2026-09-30)](https://blog.voyageai.com/2026/09/30/rerank-3/)
- [Voyage: voyage-context-4 (2026-06-29)](https://blog.voyageai.com/2026/06/29/voyage-context-4/)
- [Voyage: voyage-code-4 (2026-08-13)](https://blog.voyageai.com/2026/08/13/voyage-code-4/)
- [Voyage blog index (2026 releases)](https://blog.voyageai.com/)
- [Microsoft Bing: Harrier open-source embedding models (2026-04-07)](https://blogs.bing.com/search/April-2026/Microsoft-Open-Sources-Industry-Leading-Embedding-Model)
- [supermemory: open-source embedding ranking, rechecked 2026-10-16](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)
- [OpenRouter: Gemini Embedding 2](https://openrouter.ai/google/gemini-embedding-2)
- [arXiv 2504.19754: Reconstructing Context — late chunking vs contextual retrieval](https://arxiv.org/html/2504.19754v1)
- [Late chunking vs contextual retrieval (2026-06-23)](https://dreaming.press/posts/2026-06-23-late-chunking-vs-contextual-retrieval.html)
- [LanceDB docs: multivector search](https://docs.lancedb.com/search/multivector-search)
- [RAG state of the art 2026 synthesis (May 2026)](https://github.com/dpsdagain/RAG/blob/main/rag_state_of_the_art_2026.md)
- [DeepEval vs Ragas (2026)](https://qaskills.sh/blog/deepeval-vs-ragas-llm-evaluation-2026)
- [Google: What's new in Gemini 3.5 Flash (updated 2026-10-12)](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5)
- [Gemini 3.8 Flash listing](https://llmgateway.io/models/gemini-3.8-flash)
- [Wikipedia: GPT-6.1](https://en.wikipedia.org/wiki/GPT-6.1)
- [FutureAGI: rerankers compared (Jul 2025)](https://futureagi.com/blog/best-rerankers-for-rag-2026/)
