# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**UltraRAG v1.3.0** - A production-grade RAG system for Obsidian vaults implementing late chunking, hybrid retrieval, query transformation (HyDE/Multi-Query), self-correction patterns, and iterative research mode.

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Development Commands

```bash
# Activate environment
source venv/bin/activate

# Run CLI interface
python main.py

# Run web interface
streamlit run app.py

# Run all tests
pytest

# Run specific test file/class/method
pytest tests/test_loader.py
pytest tests/test_loader.py::TestWikilinkExtraction
pytest tests/test_loader.py::TestWikilinkExtraction::test_extract_simple_wikilinks

# Coverage report
pytest --cov=. --cov-report=html && open htmlcov/index.html

# Check setup
python check_setup.py
```

## Web Interface Auto-Load
The Streamlit web interface (`app.py`) auto-loads the existing index on startup:
```bash
# Add to .env (enabled by default)
AUTOLOAD_INDEX=true
```
- When `true`: Index loads automatically when Streamlit starts (no button click needed)
- When `false`: Manual "Load Existing Index" button required (useful for debugging)
- Status shows "(auto-loaded)" to indicate automatic loading

## Disk Cache (Fast Restarts)
Docstore nodes are cached to disk for fast app restarts (~2s vs ~10s):
```bash
# Cache location
data/cache/docstore_nodes.pkl

# Invalidate cache (terminal)
touch data/.cache_invalid

# Invalidate cache (CLI)
python main.py  # then type: cache

# Invalidate cache (Python)
from vector_store import invalidate_cache
invalidate_cache()
```
Cache auto-invalidates when index row count changes. Manual invalidation needed after re-indexing with same document count.

## Install as App (PWA)
UltraRAG can be installed as a standalone app on your dock/home screen:

**macOS (Chrome/Edge):**
1. Open `http://localhost:8501` in Chrome/Edge
2. Click the install icon in the URL bar (or Menu → "Install UltraRAG...")
3. App appears in your Applications folder and Dock

**macOS (Safari):**
1. Open `http://localhost:8501` in Safari
2. File → "Add to Dock"

**iOS (iPhone/iPad):**
1. Open `http://your-mac-ip:8501` in Safari
2. Tap Share → "Add to Home Screen"
3. App icon appears on home screen

**Android:**
1. Open `http://your-mac-ip:8501` in Chrome
2. Tap Menu → "Add to Home Screen" or "Install App"

Note: For mobile access, run Streamlit with `--server.address 0.0.0.0` to allow network access.

## CLI Commands

When running `python main.py`, the following commands are available:

| Command | Description |
|---------|-------------|
| `<query>` | Standard RAG query |
| `@research <query>` | Research mode with gap analysis |
| `@all <query>` | Exhaustive research (all iterations) |
| `@raptor <query>` | Query RAPTOR hierarchical index |
| `@vault <query>` | Search vault only (no conversations) |
| `@conv <query>` | Search conversations only |
| `index` | Re-index the vault |
| `conv` | Index AI conversations |
| `raptor` | Build RAPTOR index |
| `cache` | Invalidate docstore cache |
| `stats` | Show index statistics |
| `help` | Show help |
| `quit` / `exit` | Exit CLI |

## Architecture

### Entry Points
- `main.py` - CLI orchestrator with `UltraRAG` class
- `app.py` - Streamlit web interface

### Ingestion Pipeline
```
loader.py (ObsidianLoader) → chunking.py (ObsidianChunker) → embeddings.py → vector_store.py
```
- **loader.py**: Parses .md files, extracts frontmatter, wikilinks `[[note]]`, tags `#tag`
- **chunking.py**: 5 strategies: `obsidian_aware` (default), `markdown_semantic`, `late_chunking`, `semantic`, `simple`
- **embeddings.py**: Voyage, Qwen, OpenAI models with token tracking
- **vector_store.py**: LanceDB (embedded) or Qdrant

### Query Pipeline
```
query_transform.py → query_engine.py → self_correction.py → LLM response
```
- **query_transform.py**: HyDE and multi-query expansion
- **query_engine.py**: `RAGQueryEngine`, `HybridQueryEngine`, reranking, caching
- **self_correction.py**: Self-RAG/CRAG patterns with relevance grading

### Research Mode (Iterative Retrieval)
```
research_mode.py → multi-step retrieval → gap analysis → refined queries
```
- **research_mode.py**: `ResearchRetriever` performs iterative retrieval with LLM-powered gap analysis
- Activates via `@research <query>` in CLI or 🔬 checkbox in web UI
- **Exhaustive mode**: Use `@all <query>` prefix to force all iterations regardless of confidence (for "all/every" queries)
- **Important:** Research mode uses its own synthesis limit (`RESEARCH_MAX_SYNTHESIS_SOURCES=0`), NOT the UI "Max sources" dropdown. This allows deep analysis while the dropdown only controls display count.
- Config: `research_max_iterations=3`, `research_confidence_threshold=0.8`, `research_max_synthesis_sources=0` (0 = unlimited)

**Dual-Model Architecture:**
Research mode uses two different models to optimize for speed and reduce rate limiting:

| Function | Model | Purpose |
|----------|-------|---------|
| Gap Analysis | `gemini-flash-latest` (Gemini 2.5 Flash) | Fast, lightweight analysis with AFC disabled |
| Sub-query Generation | `gemini-3-flash-preview` | Main LLM for quality query generation |
| Final Synthesis | `gemini-3-flash-preview` | Main LLM for comprehensive output |

- Gap analysis LLM has AFC (Automatic Function Calling) disabled to prevent rate limiting
- 5-second delay between iterations to spread API calls
- Progressive retry on MAX_TOKENS: 100% → 80% → 66% → 300 nodes
- See `docs/features/RESEARCH_MODE_ENHANCEMENTS.md` for full documentation

### Federated Retrieval (AI Conversations)
```
conversation_loader.py → federated_query.py → merged results
```
- **conversation_loader.py**: Parses ChatGPT/Claude/Gemini exports, turn-aware chunking
- **federated_query.py**: `FederatedQueryEngine` queries vault + conversations indexes in parallel, merges with configurable weights

### RAPTOR Hierarchical Summaries
```
raptor_index.py → recursive clustering → LLM summarization → tree traversal
```
- **raptor_index.py**: `RaptorIndexManager` builds hierarchical summary trees
- Two retrieval modes: `collapsed` (flat search) and `tree_traversal` (hierarchical)
- Activates via `@raptor <query>` in CLI or RAPTOR checkbox in web UI
- Config: `ENABLE_RAPTOR=true`, `RAPTOR_MODE=collapsed`

### Configuration
- **config.py**: Pydantic models (`RAGConfig`, `EmbeddingConfig`, `LLMConfig`, etc.)
- **.env**: Runtime configuration (copy from `.env.example`)

### Web UI Features
The Streamlit web interface (`app.py`) provides:
- **Search**: Query input with Research Mode toggle
- **Results**: Answer with inline citations `[1]`, `[2]`, `[3]`
- **Sources**: Expandable source list with clickable Obsidian links
- **Settings**: File exclusions, pattern matching preview
- **History**: Past queries with one-click re-run
- **Stats**: Index info, LLM costs dialog, Voyage token usage
- **Scope**: Toggle between Vault, Conversations, or Both

## Key Patterns

### Token Tracking (Voyage AI)
`token_tracker.py` wraps Voyage API calls to track embedding/reranking token usage against quotas. Check `data/voyage_usage.json` for current usage.

### LLM Token & Cost Tracking (Gemini)
`llm_token_tracker.py` tracks all Gemini LLM calls with per-day cost aggregation.
- **Storage**: `data/llm_usage.json`
- **Pricing**: gemini-3-flash-preview ($0.50/$3.00 per 1M tokens), gemini-3-pro-preview ($2.00/$12.00)
- **UI**: Click "LLM Costs" button in sidebar to view daily breakdown table
- **Currency**: Costs displayed in EUR with VAT (configurable via `.env`)
  - `CURRENCY_EXCHANGE_RATE=0.8633` - USD to EUR conversion rate
  - `VAT_RATE=0.24` - VAT percentage (24% for Greece)
- **API**: `from llm_token_tracker import get_llm_tracker; tracker.get_total_stats()`

The `TrackedLLM` wrapper (`tracked_llm.py`) automatically intercepts all LLM calls and records token usage.

### LanceDB Schema Stability
`loader.py` uses a fixed schema with `extra_metadata` JSON field to prevent schema mismatch errors across document batches with varying frontmatter.

### Embedding Cache
`cache.py` caches computed embeddings to `data/embedding_cache/` to reduce API costs on re-indexing.

### Checkpointing
Indexing uses checkpoints (`data/index_checkpoint.json`) for recovery if interrupted.

### Docstore Reconstruction
When loading a persisted LanceDB index, `vector_store.py:_reconstruct_nodes_from_lancedb()` rebuilds the docstore from `_node_content` metadata stored in vectors. This enables hybrid search and graph retrieval which require full node access beyond just embeddings.

## Test Markers
```bash
pytest -m "not slow"      # Skip slow tests
pytest -m integration     # Integration tests only
pytest -m unit            # Unit tests only
```

## Data Directories
- `data/lancedb/` - Vector index (tables: `vectors` for vault, `conversations` for AI chats, `settings` for exclusions)
- `data/raptor/` - RAPTOR hierarchical index (LanceDB table: `raptor_embeddings`)
- `data/cache/docstore_nodes.pkl` - Cached docstore for fast restarts
- `data/embedding_cache/` - Cached embeddings
- `data/voyage_usage.json` - Voyage AI (embeddings/rerank) token tracking
- `data/llm_usage.json` - Gemini LLM token tracking with daily costs
- `data/index_checkpoint.json` - Indexing checkpoint
- `data/query_history.json` - Persistent query history

## AI Conversations Integration
Enable federated search across vault + AI conversation exports:
```bash
CONVERSATIONS_ENABLED=true
CONVERSATIONS_PATH=/path/to/ai-conversation-toolkit/output
```
CLI query prefixes: `@vault`, `@conv`, `@all` to filter search scope.
Compatible with exports from [AI Conversation Toolkit](https://github.com/silver-gr/ai-conversation-toolkit).

## RAPTOR Hierarchical Summaries
Enable RAPTOR for better multi-document reasoning through hierarchical clustering:
```bash
ENABLE_RAPTOR=true
RAPTOR_MODE=collapsed  # or "tree_traversal"
RAPTOR_CHUNK_SIZE=1024  # Must be > metadata size
RAPTOR_TOP_K=10
```
CLI: `raptor` to build index, `@raptor <query>` to search.
Note: Building RAPTOR index uses LLM calls for cluster summarization (slower initial indexing).

## Bilingual Query Expansion
Enable bilingual expansion to search English queries in other languages (e.g., Greek notes):
```bash
ENABLE_BILINGUAL_EXPANSION=true
EXPANSION_LANGUAGES=el  # Comma-separated: el,es,de for Greek, Spanish, German
```
- Translates key nouns/concepts (not full query) to target languages
- Augments existing query transformation (HyDE, multi-query) rather than replacing
- Supported languages: el (Greek), es (Spanish), de (German), fr (French), it (Italian), pt (Portuguese), nl (Dutch), ru (Russian), zh (Chinese), ja (Japanese), ko (Korean), ar (Arabic)
- Example: "habits for productivity" also searches for "συνήθειες για παραγωγικότητα"

## LLM Backend Options

Two LLM backends are available (set `LLM_BACKEND` in .env):

| Backend | Quota | Requires | Use Case |
|---------|-------|----------|----------|
| `api` | API-based | `GOOGLE_API_KEY` | Default, direct API access |
| `cli` | 1000 req/day, 60/min | Gemini CLI installed | Cost savings, separate quota |

**CLI Setup:**
```bash
npm install -g @google/gemini-cli
gemini  # authenticate once
```

Then set `LLM_BACKEND=cli` in your .env file.

## Default Configuration (v1.3.0)
- LLM: `gemini-3-flash-preview` (backend: `api`)
- Embeddings: `voyage-3.5-lite` (200M free tokens/month)
- Reranker: `rerank-2.5` (200M free tokens/month)
- Chunk size: 512 tokens, overlap: 75
- Retrieval: top_k=75 → rerank to top_n=100 (UI controls display count)
- Similarity threshold: 0.3 (only applied when no reranker is configured)

## Obsidian URI Links
Enable clickable source links that open notes directly in Obsidian:
```bash
# Add to .env - must match your vault name exactly (case-sensitive)
OBSIDIAN_VAULT_NAME=Silver Personal
```
- When configured, source file paths in the web UI become clickable
- Clicking opens the note directly in Obsidian via `obsidian://` URI scheme
- Works for vault sources only (not AI conversation sources)

## File Exclusions
Exclude files or folders from indexing via Settings UI or programmatically:
```bash
# Web UI: Click "Settings" button in sidebar after loading index

# Programmatic usage:
from settings_store import add_exclusion, get_exclusions
add_exclusion("data/lancedb", "Archive/**", "glob")  # Exclude Archive folder
add_exclusion("data/lancedb", "*.excalidraw.md", "glob")  # Exclude Excalidraw files
```
- Pattern types: `glob` (wildcards), `exact` (path match), `regex`
- Live removal: Files are removed from index immediately when pattern is added
- Persistence: Patterns saved in LanceDB `settings` table
- See `docs/features/FILE_EXCLUSIONS.md` for full documentation

## RAGAS Evaluation
Run automated evaluation with RAGAS metrics:
```bash
# Run evaluation against test dataset
python -m evaluation --dataset tests/evaluation_dataset.json

# Metrics: faithfulness, answer_relevancy, context_precision, context_recall
```

## Web Search Integration
Enable real-time web search to augment vault knowledge:
```bash
WEB_SEARCH_ENABLED=true
WEB_SEARCH_WEIGHT=0.7  # Score multiplier vs vault (vault=1.0)
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_IN_RESEARCH=true  # Include in research mode
```
Requires Tavily API key (set in environment).

## Query History
Query history is automatically saved to `data/query_history.json`:
- Browse past queries in web UI sidebar
- Re-run previous queries with one click
- History persists across sessions

## Key Source Files
| File | Purpose |
|------|---------|
| `main.py` | CLI orchestrator, `UltraRAG` class |
| `app.py` | Streamlit web interface |
| `config.py` | Pydantic configuration models |
| `loader.py` | Obsidian vault parsing |
| `chunking.py` | Document chunking strategies |
| `embeddings.py` | Embedding model wrappers |
| `vector_store.py` | LanceDB/Qdrant integration |
| `query_engine.py` | RAG/Hybrid query engines |
| `query_transform.py` | HyDE/multi-query expansion |
| `self_correction.py` | Self-RAG/CRAG patterns |
| `research_mode.py` | Iterative research retrieval |
| `federated_query.py` | Multi-index federated search |
| `raptor_index.py` | RAPTOR hierarchical summaries |
| `tracked_llm.py` | LLM wrapper with token tracking |
| `llm_token_tracker.py` | Cost tracking and reporting |
| `token_tracker.py` | Voyage API token tracking |
| `settings_store.py` | Persistent settings (exclusions) |
| `exclusion_matcher.py` | File exclusion pattern matching |

## Documentation
Extended documentation is in `docs/`:
- `docs/QUICKSTART.md` - Getting started guide
- `docs/ARCHITECTURE.md` - System diagrams and data flow
- `docs/EVALUATION.md` - RAGAS evaluation setup and usage
- `docs/TESTING.md` - Test suite guide
- `docs/features/` - Feature guides:
  - `LATE_CHUNKING.md` - Document-aware embeddings
  - `QUERY_TRANSFORMATION.md` - HyDE and multi-query
  - `SELF_CORRECTION.md` - Self-RAG/CRAG patterns
  - `GRAPH_RETRIEVAL.md` - Wikilink graph expansion
  - `CONTEXTUAL_RETRIEVAL.md` - LLM-enhanced context (opt-in)
  - `FILE_EXCLUSIONS.md` - File/folder exclusion patterns
  - `RESEARCH_MODE_ENHANCEMENTS.md` - Exhaustive queries, dual-model architecture
  - `QUERY_INTENT_CLASSIFICATION.md` - Future: LLM-based query classification
- `docs/reference/RAG_STRATEGY.md` - Original planning document (historical)
