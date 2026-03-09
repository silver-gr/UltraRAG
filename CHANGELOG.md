# Changelog

All notable changes to UltraRAG are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning Scheme

- **MAJOR** (X.0.0): Breaking changes to configuration or API
- **MINOR** (1.X.0): New features, backward compatible
- **PATCH** (1.0.X): Bug fixes, documentation updates

---

## [1.6.0] - 2026-03-08

### Added
- **Post-rerank score threshold** (`RERANK_SCORE_THRESHOLD`)
  - Separate threshold for reranked scores (different scale from cosine similarity)
  - Applied in `RAGQueryEngine`, `HybridQueryEngine`, and `FederatedQueryEngine`
- **Query decomposition** (`ENABLE_QUERY_DECOMPOSITION`)
  - LLM-powered pre-retrieval splitting of complex multi-part queries
  - Sub-queries retrieved independently and merged with RRF
  - Example: "Compare HyDE and multi-query" → retrieves each independently
- **Contextual compression** (`ENABLE_CONTEXTUAL_COMPRESSION`)
  - Post-rerank extraction of query-relevant sentences from chunks
  - Reduces synthesis token usage while preserving answer quality
- **Query-time metadata filtering**
  - `MetadataFilterPostprocessor` filters by tags, dates, path prefix before reranking
  - CLI flags: `--tag`, `--after`, `--before`, `--path-prefix`
- **Tunable RRF fusion** (`RRF_K`, `FUSION_VECTOR_WEIGHT`, `FUSION_BM25_WEIGHT`)
  - All 5 hardcoded `RRF_K=60` sites now configurable
- **Structured confidence output on `QueryResult`**
  - New fields: `confidence`, `relevance_grade`, `validation_result`, `timings`
  - CLI outputs these in structured YAML/JSON when present
- **Search latency monitoring**
  - Per-stage timing breakdown: `engine_build_ms`, `query_ms`, `total_ms`
  - `RetrievalMetrics` extended with latency fields

### Fixed
- **CLI stub engine** (HIGH): `_build_query_engine()` was a stub — non-interactive CLI got no reranking, no BM25, no postprocessors. Now uses real `RAGQueryEngine`/`HybridQueryEngine`.
- **Similarity threshold after reranker**: `RAGQueryEngine` unconditionally applied cosine similarity cutoff (0.3) which dropped valid reranked results. Now guarded with `not self.reranker`.
- **No cross-source reranking in federated path**: `federated_query()` used ad-hoc merging with no reranking. Now delegates to `FederatedQueryEngine` with full pipeline.
- **Dead code cleanup**: Removed `_federated_retrieve()` and `synthesize_answer()` (130 lines)

### Changed
- Default config version bumped to v1.6.0
- `book_raptor.py` uses `get_response_synthesizer` instead of removed `synthesize_answer`

---

## [1.5.0] - 2026-03-08

### Added
- **Authentication & Multi-User**
  - In-app authentication with admin/user roles (`ULTRARAG_AUTH_ENABLED`)
  - Per-user storage isolation (query history, research exports)
  - Rate limiting: global concurrency control + per-user cooldown
  - User management CLI (`python -m scripts.manage_users`)
  - Session timeout configuration
- **Retrieval Stack Audit** (5 bug fixes + 6 features)
  - Over-fetch ratio: `retrieval_candidates=150` feeds reranker, `rerank_top_n=20` keeps best (was 75 in, 100 out = no filtering)
  - Multi-query RRF: proper Reciprocal Rank Fusion (k=60) replaces incorrect `max(score_a, score_b)` across 4 fusion sites
  - SelfRAGValidator wired: post-generation hallucination check was dead code, now active
  - All sub-queries used in research mode (was discarding `subqueries[1:]`)
  - QueryFusionRetriever `num_queries=1` eliminates redundant LLM query expansion
  - Document diversity postprocessor (`ENABLE_MMR`, `MAX_CHUNKS_PER_DOCUMENT=5`)
  - HyDE temperature separation: dedicated LLM at temp=0.7 for hypothetical doc generation
  - Retrieval evaluation framework: precision@k, recall@k, MRR, NDCG per query
- **LanceDB Index Optimization**
  - Auto-creates IVF_HNSW_SQ or IVF_PQ index on tables with 1000+ rows
  - HNSW parameter exposure: `HNSW_M`, `HNSW_EF_CONSTRUCTION`, `HNSW_EF_SEARCH`
  - Scalar/product quantization support (`ENABLE_QUANTIZATION`, `QUANTIZATION_TYPE`)
- **Saved Items Integration (TheSource)**
  - Federated search across vault + TheSource saved items
  - Configurable weight (`SAVED_ITEMS_WEIGHT`) and LanceDB path
- **Infrastructure**
  - HTTPS via auto-generated dev certificates (`scripts/generate_dev_certs.sh`)
  - Startup script with TLS automation (`start.sh`)

### Changed
- Default `rerank_top_n` from 100 to 20 (reranker now actually filters)
- Default `retrieval_candidates` introduced at 150 (over-fetch for reranker)
- Embedding model default: `voyage-4-lite` (was `voyage-3.5-lite`)
- Verified Voyage `input_type` dispatched correctly by LlamaIndex (no change needed)
- Per-user query history replaces global `data/query_history.json`

### Fixed
- Reranker received fewer candidates than it returned (no-op reranking)
- Multi-query fusion used max-score instead of RRF (documents in multiple variations not boosted)
- SelfRAGValidator was dead code (never called post-generation)
- Research mode discarded all sub-queries except first (~3x coverage lost)
- QueryFusionRetriever generated 3 redundant LLM query variations
- HyDE duplicate generation race condition
- LanceDB `list_tables` API compatibility across versions
- Conversation loader excluded `_excluded` directories

---

## [1.4.0] - 2026-02-19

### Added
- **Book Library** (EPUB/PDF federated search)
  - Book indexing with PDF page merging and configurable chunking
  - Calibre metadata enrichment (fuzzy matching, stale cache handling)
  - Optional web metadata enrichment utilities
  - Book category/author filter UI in Streamlit
  - Book filtering via native LanceDB WHERE clauses
  - Book commands in interactive CLI (`@books`) and non-interactive CLI
  - 2-stage Book-Summary RAPTOR retrieval
  - Configurable: `BOOKS_ENABLED`, `BOOKS_PATH`, `BOOKS_WEIGHT`, `BOOKS_TABLE_NAME`
- **Obsession Radar**
  - Recurring theme detection across queries and research sessions
- **Non-Interactive CLI** (`cli.py`)
  - Agent/automation-friendly interface: `python -m cli research|query|status`
  - Structured YAML/JSON output on stdout, logs on stderr
  - Depth modes: `quick` (vault), `standard` (vault+conversations), `deep` (iterative)
- **Glass Morphism UI Redesign**
  - Modern visual design with glass morphism effects
  - Citation renumbering fix

### Changed
- Source numbering added to LLM context for accurate inline citations
- Federated retrieval uses reserved-slot source diversity (prevents single-source dominance)
- Book chunking parameters made configurable via env vars
- Streamlit port explicitly set to 9001

### Fixed
- Citation renumbering in UI after source filtering
- Source numbering mismatch between LLM context and displayed sources
- Default table name restored to `obsidian_embeddings`

---

## [1.3.0] - 2026-01-21

### Added
- **Research Mode Enhancements**
  - Exhaustive query mode with `@all` prefix for comprehensive retrieval
  - Auto-detection of exhaustive patterns ("all", "every", "complete list")
  - Dual-model architecture: `gemini-flash-latest` for gap analysis, main LLM for synthesis
  - Progressive retry for MAX_TOKENS errors (100% → 80% → 66% → 300 nodes)
  - Inter-iteration delay (5s) to prevent rate limiting
- **Citation Filtering**: Show only cited sources in results
- **Gap Analysis Insights**: Display coverage metrics in research mode
- **Euro Currency Display**: Costs shown in EUR with VAT for Google Cloud billing alignment
- **LLM Token Tracking**: Per-day cost aggregation with detailed breakdown dialog

### Changed
- Source excerpt display limit increased from 500 to 1500 characters
- Research mode uses `RESEARCH_MAX_SYNTHESIS_SOURCES` (not UI dropdown)
- Gap analysis model has AFC (Automatic Function Calling) disabled

### Fixed
- Rate limiting issues on research iteration 3-4
- MAX_TOKENS errors during large research synthesis

---

## [1.2.0] - 2026-01-16

### Added
- **Research Mode** (Khoj-inspired)
  - Iterative multi-step retrieval with LLM-powered gap analysis
  - 141% accuracy improvement on complex queries (benchmark)
  - CLI: `@research <query>` prefix
  - Web UI: 🔬 checkbox toggle
- **RAPTOR Hierarchical Summaries**
  - Recursive clustering with LLM summarization
  - Two modes: `collapsed` (flat) and `tree_traversal` (hierarchical)
  - CLI: `raptor` command to build, `@raptor <query>` to search
- **RAGAS Evaluation Framework**
  - Automated metrics: faithfulness, answer_relevancy, context_precision, context_recall
  - Test dataset support in `tests/evaluation_dataset.json`
- **Bilingual Query Expansion**
  - Translate key terms to additional languages (Greek, Spanish, etc.)
  - `ENABLE_BILINGUAL_EXPANSION=true` + `EXPANSION_LANGUAGES=el`
- **PWA Support**
  - Install as standalone app on macOS/iOS/Android
  - Service worker for offline capability
- **Persistent Query History**
  - History saved to `data/query_history.json`
  - Browse and re-run past queries
- **Disk Cache for Fast Restarts**
  - Docstore nodes cached to `data/cache/docstore_nodes.pkl`
  - ~2s restart vs ~10s without cache

### Changed
- Default embedding model: `voyage-3.5-lite` (was `voyage-3-large`)
- Default reranker: `rerank-2.5` (was `voyage-rerank-2`)
- Rerank top_n increased to 100 (UI controls final display count)
- Similarity threshold lowered to 0.3 (was 0.7)

### Fixed
- Schema stability issues across document batches
- Reranker detection for hybrid search

---

## [1.1.0] - 2026-01-10

### Added
- **Federated Retrieval**
  - Search across vault AND AI conversation exports
  - Support for ChatGPT, Claude, Gemini exports
  - Configurable weight balancing (`CONVERSATIONS_WEIGHT`)
  - Search scope prefixes: `@vault`, `@conv`, `@all`
- **AI Conversations Index**
  - Turn-aware chunking for conversation context
  - Separate LanceDB table (`conversations`)
  - Compatible with [AI Conversation Toolkit](https://github.com/silver-gr/ai-conversation-toolkit)
- **Gemini CLI Backend**
  - Alternative to API with separate quota (1000 req/day)
  - `LLM_BACKEND=cli` option
- **Inline Citations**
  - Clickable `[1]`, `[2]`, `[3]` citations in responses
  - Auto-scroll to source in web UI

### Changed
- Docstore reconstruction from LanceDB metadata
- Graph retrieval uses in-memory wikilink graph (Neo4j deferred)

### Fixed
- Docstore reconstruction for hybrid search
- Schema mismatch errors with varying frontmatter

---

## [1.0.0] - 2026-01-05

### Added
- **Core RAG Pipeline**
  - Obsidian markdown parsing with frontmatter, wikilinks, tags
  - Multiple chunking strategies: `obsidian_aware`, `markdown_semantic`, `late_chunking`, `semantic`, `simple`
  - Late chunking for +10-12% retrieval accuracy
- **Embedding Models**
  - Voyage AI: `voyage-3-large`, `voyage-3.5-lite`
  - Self-hosted: `qwen3-8b`
  - Token tracking with usage quotas
- **Vector Databases**
  - LanceDB (embedded, zero-config)
  - Qdrant (production, scalable)
- **Retrieval Features**
  - Hybrid search (vector + BM25 fusion)
  - Wikilink graph retrieval
  - Query transformation: HyDE, multi-query, both
  - Self-correction: Self-RAG/CRAG patterns
- **Reranking**
  - Voyage `rerank-2.5` (default)
  - Token tracking with usage quotas
- **LLM Integration**
  - Gemini 3 Flash (`gemini-3-flash-preview`)
  - PTCF prompting framework
  - 65,536 max output tokens
- **Web Interface**
  - Streamlit-based UI
  - Auto-load index on startup
  - Obsidian URI links (clickable sources)
- **Checkpointing**
  - Recovery for interrupted indexing
  - `data/index_checkpoint.json`

### Infrastructure
- Pydantic configuration with validation
- Comprehensive test suite with pytest
- Documentation in `docs/`

---

## Pre-release

### [0.1.0] - 2024-11-XX (Initial Development)

- Research and planning phase
- Strategy document: `docs/reference/RAG_STRATEGY.md`
- Core architecture design
