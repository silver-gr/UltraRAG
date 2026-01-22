# Changelog

All notable changes to UltraRAG are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning Scheme

- **MAJOR** (X.0.0): Breaking changes to configuration or API
- **MINOR** (1.X.0): New features, backward compatible
- **PATCH** (1.0.X): Bug fixes, documentation updates

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
