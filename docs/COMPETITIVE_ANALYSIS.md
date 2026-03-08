# UltraRAG Competitive Analysis

*Generated 2026-03-08. Based on README analysis of 12 open-source RAG systems.*

## Methodology

Each competitor's GitHub README was fetched and analyzed for documented features only. No capabilities were inferred from source code inspection. UltraRAG's feature set was extracted from CLAUDE.md and codebase documentation.

---

## 1. Feature Matrix

### 1.1 Retrieval Quality

| Feature | UltraRAG | RAGFlow | STORM | DeepTutor | DeepSearcher | Kotaemon | Quivr | Cognita | MemoRAG | fast-graphrag | AutoFlow | Morphik | PageIndex |
|---------|----------|---------|-------|-----------|--------------|----------|-------|---------|---------|---------------|----------|---------|-----------|
| Vector search | Y | Y | Via VectorRM | Y | Y | Y | Y | Y | FAISS | Via graph | Y (TiDB) | Y (pgvector) | X (vectorless) |
| BM25 / full-text | Y (bilingual stemmer) | Y (Elasticsearch) | X | X | X | Y | X | X | X | X | X | X | X |
| Hybrid (vector+BM25) | Y | Y (fused re-ranking) | X | X | X | Y | X | X | X | X | Partial | X | X |
| Reranking | Y (Voyage rerank-2.5) | Y (fused) | X | X | X | Y | Y (Cohere multilingual) | Y (Infinity/mxbai) | X | X | Y (Jina/Cohere/vLLM) | Y (FlagReranker) | X |
| Graph retrieval | Y (wikilink expansion) | X | X | Y (knowledge graph) | X | Y (3 GraphRAG backends) | X | X | X | Y (PageRank) | Y (TiDB KG) | X | X |
| RAPTOR hierarchical | Y | X | X | X | X | X | X | X | X | X | X | X | X |
| Federated multi-index | Y (vault+conversations) | X | X | X | X | X | X | X | X | X | X | X | X |

**Takeaway:** UltraRAG has the broadest retrieval stack. Only RAGFlow and Kotaemon match on hybrid search. RAPTOR and federated multi-index are unique to UltraRAG. Graph retrieval is available in 5 systems but via different approaches (wikilinks vs knowledge graphs vs PageRank).

### 1.2 Query Intelligence

| Feature | UltraRAG | RAGFlow | STORM | DeepTutor | DeepSearcher | Kotaemon | Quivr | Cognita | MemoRAG | fast-graphrag | AutoFlow | Morphik | PageIndex |
|---------|----------|---------|-------|-----------|--------------|----------|-------|---------|---------|---------------|----------|---------|-----------|
| HyDE | Y | X | X | X | X | X | X | X | X | X | X | X | X |
| Multi-query expansion | Y | X | X | X | X | X | X | X | X | X | X | X | X |
| Query decomposition | X | X | X | Y (DecomposeAgent) | X | Y (FullDecompose) | X | Y | X | X | X | X | X |
| Query rewriting | X | X | X | Y (RephraseAgent) | X | X | Y | X | Y (memory rewrite) | X | X | X | X |
| Bilingual expansion | Y (12 languages) | Cross-language | X | X | X | X | X | X | X | X | X | X | X |
| Multi-perspective | X | X | Y (core innovation) | X | X | X | X | X | X | X | X | X | X |
| Memory-based clues | X | X | X | X | X | X | X | X | Y (core innovation) | X | X | X | X |

**Takeaway:** UltraRAG is the only system with HyDE + multi-query + bilingual expansion. However, it lacks query decomposition (splitting complex queries into sub-queries), which DeepTutor, Kotaemon, and Cognita offer. MemoRAG's memory-based clue generation is a genuinely novel approach worth studying.

### 1.3 Answer Quality

| Feature | UltraRAG | RAGFlow | STORM | DeepTutor | DeepSearcher | Kotaemon | Quivr | Cognita | MemoRAG | fast-graphrag | AutoFlow | Morphik | PageIndex |
|---------|----------|---------|-------|-----------|--------------|----------|-------|---------|---------|---------------|----------|---------|-----------|
| Self-correction (CRAG) | Y | X | X | X | X | X | X | X | X | X | X | X | X |
| Relevance grading | Y | X | X | X | X | Low-relevance warning | X | X | X | X | X | X | X |
| Inline citations | Y ([1],[2],[3]) | Y (traceable) | Y (Wikipedia-style) | Y | X | Y (highlighted in PDF) | X | X | X | Y (configurable) | X | X | X |
| Answer validation loop | Y (CheckAgent equivalent) | X | X | Y (CheckAgent) | X | X | X | X | X | X | X | X | X |

**Takeaway:** UltraRAG's self-correction pipeline (Self-RAG/CRAG with relevance grading) is unique among all 12 competitors. Only DeepTutor has a comparable validation loop (CheckAgent). Citations are common but UltraRAG and Kotaemon offer the most integrated experience.

### 1.4 Research Capability

| Feature | UltraRAG | RAGFlow | STORM | DeepTutor | DeepSearcher | Kotaemon | Quivr | Cognita | MemoRAG | fast-graphrag | AutoFlow | Morphik | PageIndex |
|---------|----------|---------|-------|-----------|--------------|----------|-------|---------|---------|---------------|----------|---------|-----------|
| Iterative retrieval | Y | X | Y (core) | Y (DR-in-KG) | X | X | X | X | X | X | X | X | X |
| Gap analysis | Y | X | X | X | X | X | X | X | X | X | X | X | X |
| Convergence detection | Y (5% threshold) | X | X | X | X | X | X | X | X | X | X | X | X |
| Exhaustive mode | Y (@all) | X | X | X | X | X | X | X | X | X | X | X | X |
| Dual-model architecture | Y (flash+pro) | X | Y (cheap+powerful) | X | X | X | X | X | Y (memory+generator) | X | X | X | X |
| Dynamic topic discovery | X | X | Y (perspective-driven) | Y (topic queue) | X | X | X | X | X | X | X | X | X |
| Report generation | X | X | Y (full articles) | Y (reports with citations) | Y (reports) | X | X | X | X | X | X | X | X |

**Takeaway:** UltraRAG has the most sophisticated iterative retrieval with gap analysis, convergence detection, and exhaustive mode -- no competitor matches this combination. STORM and DeepTutor offer research-style workflows but without convergence detection. UltraRAG lacks report generation (long-form output) and dynamic topic discovery.

### 1.5 Document Handling

| Feature | UltraRAG | RAGFlow | STORM | DeepTutor | DeepSearcher | Kotaemon | Quivr | Cognita | MemoRAG | fast-graphrag | AutoFlow | Morphik | PageIndex |
|---------|----------|---------|-------|-----------|--------------|----------|-------|---------|---------|---------------|----------|---------|-----------|
| Markdown/Obsidian | Y (obsidian-aware) | X | X | Y | X | X | Y | Y | X | X | Y | X | Y |
| PDF | Y (via Calibre) | Y (deepdoc) | X | Y (MinerU) | Y | Y (multimodal) | Y | X | X | X | Y | Y (ColPali) | Y (vision) |
| Office (DOCX/XLSX/PPTX) | X | Y | X | X | X | Y (full image) | X | X | X | X | Y | X | X |
| Audio/Video | X | X | X | Y | X | X | X | Y | X | X | X | Y | X |
| Web crawling | X | X | Y (search APIs) | X | Y (4 crawlers) | X | X | X | X | X | Y (sitemap) | X | X |
| AI conversation exports | Y (ChatGPT/Claude/Gemini) | X | X | X | X | X | X | X | X | X | X | X | X |
| Multiple chunking strategies | Y (5 strategies) | Y (template-based) | X | X | X | X | X | X | X | X | X | Y (contextual) | X (no chunking) |
| Visual/multimodal parsing | X | Y (deepdoc) | X | X | X | Y (Azure/Adobe/Docling) | X | Y (GPT-4 vision) | X | X | X | Y (ColPali) | Y (vision LLM) |

**Takeaway:** UltraRAG excels at Obsidian-specific handling and AI conversation indexing (unique). Major gap: no Office format support and no multimodal/vision document parsing. RAGFlow, Kotaemon, and Morphik lead in document format breadth.

### 1.6 Observability

| Feature | UltraRAG | RAGFlow | STORM | DeepTutor | DeepSearcher | Kotaemon | Quivr | Cognita | MemoRAG | fast-graphrag | AutoFlow | Morphik | PageIndex |
|---------|----------|---------|-------|-----------|--------------|----------|-------|---------|---------|---------------|----------|---------|-----------|
| Embedding token tracking | Y (Voyage) | X | X | X | X | X | X | X | X | X | X | X | X |
| LLM cost tracking | Y (per-day, EUR+VAT) | X | X | X | X | X | X | X | X | X | X | X | X |
| Query history | Y (persistent) | X | X | X | X | X | X | X | X | X | X | X | X |
| Chunk visualization | X | Y (human-in-the-loop) | X | X | X | X | X | X | X | X | X | X | X |
| Evaluation framework | Y (RAGAS) | Y (planned) | X | X | X | X | X | X | X | X | Y (built-in) | X | X |

**Takeaway:** UltraRAG is the only system with comprehensive cost observability (embedding + LLM tracking with currency conversion). No competitor offers this. RAGFlow's chunk visualization is a notable UI feature UltraRAG lacks.

### 1.7 Developer Experience

| Feature | UltraRAG | RAGFlow | STORM | DeepTutor | DeepSearcher | Kotaemon | Quivr | Cognita | MemoRAG | fast-graphrag | AutoFlow | Morphik | PageIndex |
|---------|----------|---------|-------|-----------|--------------|----------|-------|---------|---------|---------------|----------|---------|-----------|
| Web UI | Y (Streamlit/PWA) | Y (React) | Y (Streamlit) | Y (Next.js) | X (Swagger) | Y (Gradio) | Y (hosted) | Y (React) | X | X | Y (Next.js) | Y (Console) | Y (chat platform) |
| Interactive CLI | Y | X | X | Y | Y | X | Y | X | X | X | X | X | Y |
| Non-interactive CLI | Y (agent-friendly) | X | X | X | Y | X | X | X | X | X | X | X | X |
| REST API | X | Y | X | Y (FastAPI) | Y (FastAPI) | X | Y | Y (FastAPI) | X | X | Y | Y (FastAPI) | Y (beta) |
| Python library | X | X | Y | X | Y | X | Y (core) | X | Y | Y | X | Y (SDK) | X |
| MCP integration | X | Y (agent) | X | X | X | X | X | X | X | X | X | Y | Y |
| Docker deployment | X | Y | X | Y | X | Y | X | Y | X | X | Y | Y | X |
| Multi-user/multi-tenant | X | X | X | X | X | Y | X | Y | X | X | X | Y (folders) | X |

**Takeaway:** UltraRAG has strong CLI coverage (interactive + non-interactive for agents) but lacks REST API and Docker deployment. Several competitors offer both. The non-interactive CLI for agent automation is a differentiator. MCP integration is an emerging gap.

### 1.8 Production Readiness

| Feature | UltraRAG | RAGFlow | STORM | DeepTutor | DeepSearcher | Kotaemon | Quivr | Cognita | MemoRAG | fast-graphrag | AutoFlow | Morphik | PageIndex |
|---------|----------|---------|-------|-----------|--------------|----------|-------|---------|---------|---------------|----------|---------|-----------|
| Disk cache | Y (docstore + embeddings) | X | X | X | X | X | X | X | Y (KV cache) | X | X | X | X |
| Checkpointing | Y (index recovery) | X | X | X | X | X | X | X | X | Y | X | X | X |
| Incremental indexing | X | X | X | X | X | X | X | Y | X | Y (real-time) | X | X | X |
| Context caching (LLM) | Y (Gemini) | X | X | X | X | X | X | X | X | X | X | X | X |
| File exclusions | Y (glob/exact/regex) | X | X | X | X | X | X | X | X | X | X | X | X |
| Error recovery | Y (checkpoint resume) | X | X | X | X | X | X | X | X | Y (corruption prevention) | X | X | X |
| Progressive retry | Y (MAX_TOKENS) | X | X | X | X | X | X | X | X | X | X | X | X |

**Takeaway:** UltraRAG leads in production hardening features. Gemini context caching, progressive retry, and file exclusions are unique. The main gap is incremental indexing (Cognita and fast-graphrag have it), which would avoid full re-indexing when adding new documents.

---

## 2. UltraRAG Strengths (Where It Leads)

### Unmatched by any competitor:
1. **Self-correction pipeline (Self-RAG/CRAG)** -- No other system has automated relevance grading with answer correction. DeepTutor has CheckAgent but it's validation, not correction.
2. **Research mode with convergence detection** -- Gap analysis + 5% convergence threshold + exhaustive mode is unique across all 12 systems.
3. **Cost observability** -- Per-day LLM cost tracking with currency conversion (EUR+VAT), embedding token tracking. Zero competitors offer this.
4. **Bilingual query expansion** -- 12-language expansion augmenting HyDE/multi-query. RAGFlow has "cross-language query" but without the expansion architecture.
5. **AI conversation indexing** -- Federated search across Obsidian vault + ChatGPT/Claude/Gemini exports is completely unique.
6. **Non-interactive CLI for agents** -- Structured YAML/JSON output designed for agent consumption (`python -m cli`). Only DeepSearcher has comparable CLI but without structured output.
7. **Production hardening combination** -- Disk cache + embedding cache + checkpointing + context caching + progressive retry + file exclusions. No competitor matches this breadth.

### Top 5 features where UltraRAG exceeds the field:
1. **Iterative research with gap analysis + convergence** -- STORM/DeepTutor have iterative modes but no convergence detection or confidence thresholds
2. **Self-RAG/CRAG self-correction** -- Completely absent in all 12 competitors
3. **Hybrid search with bilingual BM25 stemmer** -- Greek/English BilingualStemmer is unique engineering
4. **Dual-model research architecture** -- Separating gap analysis model from synthesis model for rate limit management
5. **RAPTOR hierarchical summaries** -- No competitor offers recursive clustering with LLM summarization for multi-document reasoning

---

## 3. UltraRAG Gaps (Where It Lags)

### Critical gaps (high impact, multiple competitors have these):
1. **No query decomposition** -- DeepTutor, Kotaemon, Cognita all break complex queries into sub-queries. UltraRAG's multi-query expansion is different (generates similar queries, not sub-questions).
2. **No multimodal/vision document parsing** -- RAGFlow (deepdoc), Kotaemon (Azure/Docling), Morphik (ColPali), PageIndex (vision LLM) all handle visual elements in PDFs (charts, tables, figures). UltraRAG treats PDFs as text only.
3. **No Docker deployment** -- RAGFlow, DeepTutor, Kotaemon, Cognita, AutoFlow, Morphik all offer Docker Compose. UltraRAG requires manual Python/venv setup.
4. **No REST API** -- 7 of 12 competitors offer REST APIs. UltraRAG has CLI only (interactive + non-interactive).
5. **No Office format support** -- RAGFlow, Kotaemon, AutoFlow handle DOCX/XLSX/PPTX. UltraRAG handles only Markdown and PDF.

### Moderate gaps (nice-to-have, some competitors lead):
6. **No knowledge graph construction** -- DeepTutor, Kotaemon (3 backends!), fast-graphrag, AutoFlow all build knowledge graphs from documents. UltraRAG's wikilink graph is manual (existing Obsidian links), not auto-constructed.
7. **No incremental indexing** -- Cognita and fast-graphrag support adding documents without full re-index. UltraRAG requires complete re-indexing.
8. **No chunk visualization** -- RAGFlow's human-in-the-loop chunk inspection is a quality differentiator UltraRAG lacks.
9. **No web crawling / URL ingestion** -- DeepSearcher, AutoFlow, STORM all ingest web content. UltraRAG is file-based only.
10. **No MCP integration** -- Morphik, PageIndex, RAGFlow offer MCP. UltraRAG's non-interactive CLI could be wrapped as an MCP server easily.

---

## 4. Recommendations (Prioritized)

### Tier 1: High-impact, proven patterns to adopt

| Priority | Feature | Reference System | Why |
|----------|---------|-----------------|-----|
| 1 | **Query decomposition** | DeepTutor (DecomposeAgent), Kotaemon (FullDecomposeQAPipeline) | Handles multi-hop questions that HyDE/multi-query can't. Complements existing pipeline. |
| 2 | **Multimodal PDF parsing** | Morphik (ColPali), Kotaemon (Docling) | Charts, tables, and figures in PDFs are currently invisible to UltraRAG. Docling is open-source and lightweight. |
| 3 | **Incremental indexing** | fast-graphrag, Cognita | Avoids full re-index when adding notes. Track which documents are already indexed, process only new/modified ones. |
| 4 | **REST API** | Cognita (FastAPI) | Enables browser extensions, mobile apps, and third-party integrations. FastAPI is trivial to add atop existing query functions. |
| 5 | **MCP server wrapper** | PageIndex, Morphik | Expose UltraRAG as an MCP tool for Claude Code, Cursor, etc. The non-interactive CLI already has structured output -- just needs an MCP adapter. |

### Tier 2: Differentiation opportunities

| Priority | Feature | Reference System | Why |
|----------|---------|-----------------|-----|
| 6 | **Knowledge graph auto-construction** | fast-graphrag (cheapest: 6x less than MS GraphRAG) | Would upgrade wikilink graph from manual links to auto-extracted entity relationships. fast-graphrag's PageRank approach is elegant. |
| 7 | **Dynamic topic discovery** | DeepTutor (Dynamic Topic Queue), STORM (multi-perspective) | Research mode could discover related topics the user didn't ask about, broadening coverage. |
| 8 | **Report generation** | STORM (outline-first), DeepTutor (3-level outline + report) | Transform research mode output from answer synthesis into structured long-form reports with outline generation. |
| 9 | **Chunk visualization** | RAGFlow (visual chunking) | Show users how documents were chunked with option to correct. Valuable for Obsidian where note structure varies. |
| 10 | **Docker Compose deployment** | Cognita (single-command) | Lower barrier to entry. Bundle Streamlit + LanceDB + venv in a Docker image. |

### Tier 3: Future exploration

| Priority | Feature | Reference System | Why |
|----------|---------|-----------------|-----|
| 11 | **Memory-based retrieval** | MemoRAG | Novel approach for "unsearchable" queries. Could help with thematic/synthesis questions across the vault. Research-stage only. |
| 12 | **Vectorless tree-search** | PageIndex | Interesting alternative for long structured documents. Not a replacement but could complement vector search for books/manuals. |
| 13 | **Agent framework** | RAGFlow, DeepTutor | Multi-agent research with tool use. UltraRAG's research mode is single-agent; multi-agent could enable web search + code execution alongside vault retrieval. |

---

## 5. Architectural Insights

### Patterns worth studying:

**1. fast-graphrag's Personalized PageRank (from HippoRAG)**
Instead of vector similarity, traverse a knowledge graph using PageRank seeded from query entities. This finds contextually relevant information that's topologically close in the knowledge space, not just semantically similar. Key insight: graph structure captures relationships that embedding spaces miss.

**2. STORM's multi-model cost optimization**
Explicitly assigns cheap models (GPT-3.5 tier) to high-volume tasks (conversation simulation, question generation) and expensive models (GPT-4 tier) to quality-critical tasks (article writing, polishing). UltraRAG's dual-model research architecture already does this for gap analysis vs synthesis -- the pattern could extend to other pipeline stages.

**3. MemoRAG's global memory model**
A fine-tuned LLM compresses an entire corpus (up to 1M tokens) into latent memory, then generates "clues" that bridge the gap between a query and relevant passages. This addresses the fundamental limitation of chunk-level retrieval: when the answer requires synthesizing information spread across the entire corpus. UltraRAG's RAPTOR hierarchical summaries approach the same problem differently (bottom-up clustering vs top-down memory).

**4. Kotaemon's multi-backend GraphRAG**
Supports MS GraphRAG, NanoGraphRAG, and LightRAG as interchangeable graph backends. This modular approach means users can pick the best graph implementation for their use case without changing application code. UltraRAG could adopt this pattern if adding knowledge graph construction.

**5. Morphik's ColPali multimodal search**
A vision-language model that "sees" document layouts, enabling search over charts, tables, and diagrams without OCR or text extraction. This is the future of PDF-heavy RAG systems. The key architectural insight: treat document pages as images, not text, and use vision models for both indexing and retrieval.

**6. PageIndex's vectorless tree search**
Eliminates the entire embedding/chunking/vector pipeline. Instead, builds a hierarchical table-of-contents tree and uses LLM reasoning to navigate it -- like a human expert scanning a book's index. Achieves 98.7% accuracy on financial QA. The insight: for structured professional documents, LLM reasoning about document structure beats vector similarity.

**7. DeepTutor's dual-loop solver**
Analysis Loop (investigate -> note) feeds into Solve Loop (plan -> manage -> solve -> check -> format). The check step validates solutions before presenting them. This is architecturally similar to UltraRAG's self-correction but more granular -- separating investigation from solution generation allows each to iterate independently.

### Anti-patterns observed:

1. **Provider breadth over depth** -- DeepSearcher supports 15+ LLM providers but has no query transformation, no self-correction, no research mode. Width without depth is a weak competitive position.
2. **Graph RAG without hybrid fallback** -- fast-graphrag and AutoFlow rely entirely on graph retrieval. When the knowledge graph misses an entity, there's no BM25/vector fallback. UltraRAG's layered approach (hybrid + graph expansion) is more robust.
3. **No observability** -- 11 of 12 competitors have zero cost tracking. This is technically debt that becomes a user complaint at scale.

---

## Appendix: Competitor Summary Table

| System | Stars | Category | Best For | Architecture |
|--------|-------|----------|----------|-------------|
| RAGFlow | 73.2k | Enterprise RAG | Complex document parsing, visual QA | Elasticsearch + deepdoc + agents |
| STORM | 27.9k | Research generator | Wikipedia-style article creation | Search APIs + multi-model + litellm |
| DeepTutor | 10.5k | Education RAG | Learning assistance, problem solving | Knowledge graph + multi-agent |
| DeepSearcher | 7.6k | Milvus-native RAG | Chinese AI ecosystem, web crawling | Milvus + many providers |
| Kotaemon | - | Document chat | PDF-heavy workflows, multi-user | Gradio + 3 GraphRAG backends |
| Quivr | - | RAG library | Quick integration, 5-line setup | YAML DAG + Megaparse |
| Cognita | - | Production RAG | Enterprise multi-tenant deployment | FastAPI + plugin system |
| MemoRAG | - | Research framework | Unsearchable queries, corpus-level QA | Memory model + FAISS |
| fast-graphrag | - | Graph RAG | Knowledge graph construction | PageRank + dynamic ontology |
| AutoFlow | - | Graph RAG + chat | Customer support, docs-as-chatbot | TiDB + DSPy + embeddable widget |
| Morphik | - | Multimodal RAG | Visual documents, charts, diagrams | ColPali + pgvector + LiteLLM |
| PageIndex | - | Vectorless RAG | Long structured documents (finance, legal) | LLM tree search, no vectors |

---

*UltraRAG's competitive position: strongest in iterative research, self-correction, cost observability, and Obsidian-specific features. Primary gaps in multimodal parsing, query decomposition, and deployment packaging.*

---

## 6. Implementation-Level Optimizations

*Derived from cross-referencing RAG best-practice patterns against UltraRAG's current architecture. These are improvements within the existing system, not new features.*

### 6.1 Retrieval Pipeline

| # | Optimization | What to Change | Impact | Effort |
|---|-------------|---------------|--------|--------|
| 1 | **Verify RRF fusion** | Check if `query_engine.py` hybrid search uses proper Reciprocal Rank Fusion (`1/(k+rank+1)`, k=60) vs naive linear combination. RRF doesn't require score normalization across backends. | High | Low |
| 2 | **MMR diversity** | Add Maximal Marginal Relevance to prevent returning 5 chunks from the same note. LanceDB supports MMR natively. Balance `lambda_mult` between relevance (1.0) and diversity (0.0). | High | Low |
| 3 | **Parent Document Retriever** | Embed small chunks (400 tokens) for precise retrieval but return the parent section (2000 tokens) for context. The `obsidian_aware` chunker already has note/section boundaries to exploit. | High | Medium |
| 4 | **Contextual Compression** | Before synthesis, filter retrieved chunks to extract only query-relevant portions. Especially impactful in research mode where 100+ chunks accumulate. Reduces LLM input noise. | Medium | Medium |
| 5 | **Post-rerank score threshold** | `similarity_threshold: 0.3` is only applied when no reranker is configured. Apply a threshold *after* reranking too, to eliminate low-relevance results that survived reranking. | Medium | Low |
| 6 | **Query-time metadata filtering** | Allow users to filter by `#tag`, date range, or note folder at query time. Pre-filtering reduces search space before vector comparison. | Medium | Medium |

### 6.2 Index & Embeddings

| # | Optimization | What to Change | Impact | Effort |
|---|-------------|---------------|--------|--------|
| 7 | **HNSW parameter tuning** | LanceDB uses HNSW internally. For <1M vectors: M=16, efConstruction=128, efSearch=128 targets 95%+ recall. Check if LanceDB exposes these parameters. | Medium | Low |
| 8 | **Retrieval quality metrics** | Add precision@k, recall@k, MRR, NDCG@k evaluation alongside existing RAGAS metrics. These measure *retrieval* quality specifically (RAGAS measures end-to-end). | High | Medium |
| 9 | **Tunable fusion weights by query type** | Route factual queries (names, dates, specific terms) to higher BM25 weight; conceptual queries to higher vector weight. Connects to planned query intent classification. | Medium | Medium |

### 6.3 Output Quality

| # | Optimization | What to Change | Impact | Effort |
|---|-------------|---------------|--------|--------|
| 10 | **Structured confidence output** | Return Pydantic model with `answer`, `confidence: float`, `sources: list`, `reasoning: str`. Self-correction already grades relevance -- expose this as a user-facing confidence score. | Low | Low |
| 11 | **Search latency monitoring** | Track p50/p95/p99 search latency alongside existing token/cost tracking. Reveals slow queries and regression. | Low | Low |

### 6.4 Priority Implementation Order

```
Phase 1 (quick wins, high impact):
  #1 Verify RRF fusion → #2 Add MMR diversity → #5 Post-rerank threshold → #7 HNSW tuning

Phase 2 (medium effort, high value):
  #3 Parent Document Retriever → #8 Retrieval quality metrics → #4 Contextual compression

Phase 3 (feature work):
  #6 Query-time metadata filtering → #9 Tunable fusion weights → #10 Structured output → #11 Latency monitoring
```
