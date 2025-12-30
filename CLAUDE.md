# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UltraRAG is a production-grade RAG system for Obsidian vaults implementing late chunking, hybrid retrieval, query transformation (HyDE/Multi-Query), and self-correction patterns.

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

### Federated Retrieval (AI Conversations)
```
conversation_loader.py → federated_query.py → merged results
```
- **conversation_loader.py**: Parses ChatGPT/Claude/Gemini exports, turn-aware chunking
- **federated_query.py**: `FederatedQueryEngine` queries vault + conversations indexes in parallel, merges with configurable weights

### Configuration
- **config.py**: Pydantic models (`RAGConfig`, `EmbeddingConfig`, `LLMConfig`, etc.)
- **.env**: Runtime configuration (copy from `.env.example`)

## Key Patterns

### Token Tracking
`token_tracker.py` wraps Voyage API calls to track embedding/reranking token usage against quotas. Check `data/voyage_usage.json` for current usage.

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
- `data/lancedb/` - Vector index (tables: `vectors` for vault, `conversations` for AI chats)
- `data/embedding_cache/` - Cached embeddings
- `data/voyage_usage.json` - API token tracking
- `data/index_checkpoint.json` - Indexing checkpoint

## AI Conversations Integration
Enable federated search across vault + AI conversation exports:
```bash
CONVERSATIONS_ENABLED=true
CONVERSATIONS_PATH=/path/to/ai-conversation-toolkit/output
```
CLI query prefixes: `@vault`, `@conv`, `@all` to filter search scope.
Compatible with exports from [AI Conversation Toolkit](https://github.com/silver-gr/ai-conversation-toolkit).

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

## Default Configuration
- LLM: `gemini-3-flash-preview` (backend: `api`)
- Embeddings: `voyage-3.5-lite` (200M free tokens/month)
- Reranker: `rerank-2.5`
- Chunk size: 512 tokens, overlap: 75
- Retrieval: top_k=75 → rerank to top_n=10
- Similarity threshold: 0.3 (only applied when no reranker is configured)

## Documentation
Extended documentation is in `docs/`:
- `docs/ARCHITECTURE.md` - System diagrams and data flow
- `docs/features/` - Feature guides (late chunking, query transformation, self-correction, graph retrieval)
