# AGENTS.md

Instructions for coding agents (Claude Code, Codex, and others) working in UltraRAG.

This is the only agent-instructions file. Claude Code (v2.1.277+) reads `AGENTS.md`
natively. Do not add a `CLAUDE.md`: when both exist, Claude Code reads only `CLAUDE.md`
and ignores this file.

Treat drift between this file and the code as a bug. The facts below were verified
against the code on 2026-10-20. If they disagree with the code, the code wins; fix this
file in the same change.

## 1. Project at a glance

UltraRAG is a RAG system for an Obsidian vault. It can also federate over AI-conversation
exports, an EPUB/PDF book library, "saved items" in an external LanceDB, and Tavily web
search. Retrieval is hybrid (vector + BM25), reranked, optionally query-transformed and
self-correcting, with an iterative research mode and RAPTOR summary trees.

- Stack: Python >= 3.13 managed with uv; llama_index; LanceDB (Qdrant optional); Voyage AI
  embeddings and reranker; LLM via the Gemini API or an OpenAI-compatible proxy.
- Interfaces: Streamlit web app (multi-user auth, Greek UI), interactive CLI, and a
  non-interactive YAML/JSON CLI for agents.
- Version: CHANGELOG.md's latest entry is 1.6.1; `pyproject.toml` still says 1.6.0.

## 2. Commands

```bash
uv sync                       # runtime deps + dev group (pytest, pytest-cov)
uv sync --extra eval          # adds RAGAS evaluation deps (ragas, datasets, langsmith)

uv run streamlit run app.py   # web app at https://localhost:9001
uv run python main.py         # interactive CLI (exits if .env is missing)
uv run python -m cli status   # non-interactive CLI, see section 5

uv run pytest -m unit         # fast, no API calls - run first
uv run pytest                 # full suite
uv run pytest tests/test_retrieval.py::TestResearch -v
```

First-time setup:

```bash
uv sync
cp -n .env.example .env               # fill in API keys and OBSIDIAN_VAULT_PATH
./scripts/generate_dev_certs.sh       # writes certs/frontend.pem + certs/frontend-key.pem
uv run python -m scripts.manage_users init --admin <username>
```

User management (`uv run python -m scripts.manage_users <cmd>`): `init --admin <name>`,
`add-user --username <name> --role user|admin`, `set-password --username <name>`,
`set-role --username <name> --role user|admin`, `list-users`.

Wrapper scripts:
- `./setup.sh`: `uv sync`, then creates `.env` from `.env.example` if missing.
- `./start.sh`: HTTPS launch via `uv run`, generating dev certs if missing.
  `ULTRARAG_SERVER_ADDRESS`, `ULTRARAG_SERVER_PORT` and `ULTRARAG_BROWSER_HOST` are shell
  overrides for `start.sh` only.
- `run_tests.sh`: `uv run pytest`.

Dependencies live in `pyproject.toml` + `uv.lock` only; there is no `requirements.txt`.

## 3. Architecture

### Two orchestration stacks - read before changing behavior

| Stack | Code | Used by |
|---|---|---|
| Stateful | `main.py` class `UltraRAG`: `index_vault`, `index_conversations`, `index_books`, `index_raptor`, `query`, `query_federated`, `query_research`, `query_raptor`, ... | `app.py`, `pages/`, interactive CLI, `scripts/reindex_*.py` |
| Stateless | `indexing.py`: `index_vault`, `index_conversations`, `index_books`, `enrich_books`, `load_index`, `index_exists`. `retrieval.py`: `query`, `federated_query`, `research` | `cli.py`, tests |

Both stacks share the layers below. `app.py` does not call `retrieval.py`; it drives
`main.UltraRAG`. A retrieval or indexing behavior change usually has to land in both
stacks - check and test both.

### Pipelines

- **Ingestion:** `loader.py` (`ObsidianLoader`: frontmatter, wikilinks, tags) ->
  `chunking.py` (`ObsidianChunker`) -> `embeddings.py` -> `vector_store.py`.
  - Chunking strategies (`CHUNKING_STRATEGY`): `obsidian_aware` (default),
    `markdown_semantic`, `late_chunking`, `semantic`, `markdown`, `simple`.
  - The embedding provider is chosen by substring of the model name: Voyage (default
    `voyage-4-lite`), local HuggingFace `gte-Qwen2`, or OpenAI.
- **Query:** `query_engine.py` provides `RAGQueryEngine` (vector) and `HybridQueryEngine`
  (vector + BM25 with `BilingualStemmer` for Greek/English, fused with RRF), Voyage
  reranking, and query caching.
  - Engines use `query_transform.py` (`QueryTransformer`: HyDE, multi-query, bilingual
    expansion, decomposition).
  - Retrieval is wrapped by `self_correction.py` (`SelfCorrectingRetriever` relevance
    grading; `SelfRAGValidator` post-generation hallucination check).
- **Federated:** `federated_query.py` `FederatedQueryEngine` queries sources in parallel:
  `vault`, `conversations`, `books`, `saved_items`, `web`.
  - Each node's score is multiplied by its source weight, then results are deduplicated
    by node id and content and sorted.
  - RRF is used within a source, for hybrid and multi-query fusion. It is never applied
    across sources.
- **Research:** `research_mode.py` `ResearchRetriever` loops retrieval -> LLM gap analysis
  -> refined sub-queries.
  - It stops at `RESEARCH_CONFIDENCE_THRESHOLD`, or when information gain (new unique
    nodes / total nodes) falls below the profile threshold: balanced 8%, personal 10%,
    research 5%.
  - Exhaustive mode raises the iteration count. It is triggered by an `@all ` or `!all `
    prefix on a research query, or by "all/every"-style queries detected by regex.
  - Final synthesis lives in `main.UltraRAG.query_research`. It retries on a token budget
    (66% -> 50% -> 33% of nodes, floor 150) and switches to chunked synthesis above
    ~0.9M estimated tokens.
  - Synthesis uses `RESEARCH_MAX_SYNTHESIS_SOURCES` (0 = unlimited), not the UI
    "Max sources" selector, which only controls display.
- **RAPTOR:** `raptor_index.py` `RaptorIndexManager` builds a clustered summary tree in
  LanceDB table `raptor_embeddings` under `RAPTOR_PATH`.
  - Retrieval modes: `collapsed` and `tree_traversal`.
  - `book_raptor.py` is a two-stage book-summary variant.
  - Building it makes LLM calls for cluster summaries.

### LLM construction

- Every LLM is built through `llm_factory.py` (`create_llm`,
  `create_llm_from_config(config)`, `create_llm_from_env(model)`). That is why
  `LLM_BACKEND`, proxy settings and thinking effort apply everywhere.
- `tracked_llm.wrap_llm_with_tracking` adds token/cost tracking. It wraps only:
  - the main LLM, in `main.py` and in `retrieval.get_llm`; these two sites also add the
    `LLM_FALLBACK_MODELS` chain
  - the research gap-analysis LLM (built with function calling disabled)
- HyDE's LLM is not tracked.

## 4. Module map

**Entry points**
- `app.py`: Streamlit app (auth gate, search, admin controls, PWA wiring); uses `main.UltraRAG`.
- `pages/content_research.py`: Content Research page (weighted multi-source search + export).
- `main.py`: `UltraRAG` orchestrator and the interactive CLI (`main()`).
- `cli.py`: non-interactive CLI for agents/automation.

**Core**
- `config.py`: Pydantic config models. `load_config()` reads env; `.env` is loaded by
  `load_dotenv()` at import.
- `models.py`: shared dataclasses, exceptions, protocols, `BookFilter`.
- `indexing.py`, `retrieval.py`: stateless public API (section 3).
- Ingestion: `loader.py`, `chunking.py`, `embeddings.py`, `vector_store.py`.
- Query: `query_engine.py`, `query_transform.py`, `self_correction.py`.
- Orchestration: `federated_query.py`, `research_mode.py`, `context_cache.py`, `raptor_index.py`.
- LLMs: `llm_factory.py`, `tracked_llm.py`.

**Sources and features**
- `conversation_loader.py`: ChatGPT/Claude/Gemini exports, turn-aware chunking.
- Books: `book_loader.py`, `book_chunker.py`, `book_raptor.py`, `books_retriever.py`,
  `calibre_metadata.py`, `web_metadata_enricher.py`.
- `saved_items_retriever.py`: retriever over the external saved-items LanceDB (flat schema
  to `NodeWithScore`).
- `web_retriever.py`: Tavily web search, optional Crawl4AI.
- `content_research_engine.py`, `research_storage.py`: backend of the Content Research page.
- `citations.py`: citation offsets/validation for chunked synthesis.
- `temporal_filter.py`: date-range filtering.
- `obsession_radar.py`: cross-source topic convergence (`uv run python -m obsession_radar`).
- `evaluation.py`: RAGAS evaluation.
- `observability.py`: optional LangSmith tracing.

**Platform**
- `auth.py`: login, sessions, roles. `user_storage.py`: per-user paths.
- `rate_limit.py`: `run_with_limits` concurrency and cooldown. Streamlit-only.
- `settings_store.py`, `exclusion_matcher.py`: file exclusions, stored in the LanceDB
  `settings` table.
- Usage tracking: `token_tracker.py` (Voyage quota), `llm_token_tracker.py` (LLM cost),
  `usage.py` (Voyage usage CLI).
- `ui_theme.py`: shared Streamlit CSS.
- `cache.py`: `EmbeddingCache`. It is instantiated in `main.py` but never read or written.

**`scripts/`** (run from the repo root):
- `manage_users.py`, `generate_dev_certs.sh`
- `reindex_vault.py`, `reindex_conversations.py`
- `update_paths.py` (fixes stored paths after a vault reorganization)
- `score_notes.py`, `build_raptor_filtered.py`, `build_raptor_simple.py`
- book-metadata enrichment scripts, `test_crosslingual.py`

**Manual scripts:** root-level `quick_test.py`, `test_late_chunking.py` and
`test_self_correction.py` are not part of the test suite; `pytest.ini` collects only `tests/`.

## 5. Interfaces

### Non-interactive CLI (`cli.py`) - preferred for agents

Output is YAML (default) or JSON on stdout; logs go to stderr. `--format yaml|json` and
`--quiet` work before or after the subcommand.

```bash
uv run python -m cli research --topic "sleep optimization" --depth quick|standard|deep
uv run python -m cli query --query "meditation" --source vault|conversations|books|all --mode simple|hybrid
uv run python -m cli query --query "..." --tag a,b --after 2025-01-01 --before 2025-06-30 --path-prefix Projects/
uv run python -m cli query --query "..." --source books --category <category> --author <name>
uv run python -m cli status                  # index availability: vault, conversations, books
uv run python -m cli enrich [--clear-cache]  # book metadata enrichment
```

| `--depth` | Sources | Function |
|---|---|---|
| `quick` | vault | `retrieval.query()` |
| `standard` (default) | vault + conversations | `retrieval.federated_query()` |
| `deep` | vault + conversations, iterative | `retrieval.research()` |

- `query` defaults: `--source vault`, `--mode hybrid`.
- With one source and no book filter, `query` calls `retrieval.query()`; otherwise it
  calls `retrieval.federated_query()`.

### Interactive CLI (`main.py`)

| Input | Action |
|---|---|
| `<query>` | Federated query when extra sources are loaded, otherwise vault only |
| `@vault <q>` / `@conv <q>` | Vault only / conversations only |
| `@all <q>` | Federated query across all loaded sources |
| `@research <q>` | Research mode; `@research @all <q>` (or `!all`) forces exhaustive mode |
| `@books <q>`, `@books:<category> <q>` | Books, optionally filtered by category |
| `@raptor <q>`, `@raptor-books <q>` | RAPTOR index / book RAPTOR |
| `conv`, `books`, `raptor` | Index conversations / index books / build RAPTOR |
| `enrich` | Enrich book metadata |
| `cache` | Invalidate the docstore cache and the result cache |
| `usage` | Print token usage |
| `quit`, `exit`, `q` | Exit |

Indexing happens at startup, not through a command:
- With an existing index, a menu offers 1 = load, 2 = rebuild, 3 = exit.
- Without one, a y/n prompt offers to create it.

There is no `index`, `stats` or `help` command.

### Web app (`app.py`)

- HTTPS on port 9001, bound to 0.0.0.0. `.streamlit/config.toml` sets
  `sslCertFile = "certs/frontend.pem"`, `sslKeyFile = "certs/frontend-key.pem"` and
  `enableStaticServing = true`.
  - If you change these, update the cert script and the docs in the same change.
- `AUTOLOAD_INDEX=true` (default) loads the index at startup. `false` requires the manual
  load button.
- UI text and synthesis prompts are Greek.
- Search controls: source checkboxes (vault, conversations, books, saved items),
  research-mode and RAPTOR toggles, a search-type radio, and a "Max sources" display
  selector.
- Results: answers with inline citations `[1]`, `[2]`. Sources become `obsidian://` links
  when `OBSIDIAN_VAULT_NAME` matches the vault name exactly (case-sensitive).
- Admin dialogs:
  - Settings: file exclusions. Adding a pattern also deletes matching files from the
    index.
  - LLM costs: shown in EUR including VAT via `CURRENCY_EXCHANGE_RATE` (default 0.8633)
    and `VAT_RATE` (default 0.24).
- Installable as a PWA (Chrome/Edge install, Safari "Add to Dock", mobile "Add to Home
  Screen"). `static/manifest.json` and `static/sw.js` are injected by `app.py`.

## 6. Multi-user, auth and rate limiting

- **Auth:** on by default (`ULTRARAG_AUTH_ENABLED=true`).
  - Users live in `ULTRARAG_USERS_PATH` (default `data/auth/users.json`).
  - Sessions expire after `ULTRARAG_SESSION_TIMEOUT_MINUTES` (720).
  - Signup is off (`ULTRARAG_ALLOW_SIGNUP=false`).
  - With auth disabled the app runs as a local admin.
- **Roles:** exactly `admin` and `user`.
  - Admin-only: the management panel (load/init/index/RAPTOR), the settings and LLM-cost
    dialogs (`require_admin`), index deletion and cost resets.
  - Users can only query. `pages/content_research.py` is open to every signed-in user.
  - Never introduce silent role escalation.
- **Per-user data:** only under `data/users/<username>/` (`user_storage.py`):
  `query_history.json`, `research-exports/`.
  - Never write user data to shared global files.
  - A legacy global `data/query_history.json` is migrated once into the admin's history.
- **Rate limiting:** wrap every expensive operation in `with run_with_limits(ctx, job_name):`
  (a context manager in `rate_limit.py`). That covers queries, research, indexing, exports
  and index/settings mutations.
  - Limits: `ULTRARAG_MAX_CONCURRENT_JOBS=1`, `ULTRARAG_MIN_SECONDS_BETWEEN_QUERIES=2`,
    `ULTRARAG_JOB_ACQUIRE_TIMEOUT_SECONDS=120`.
  - It is Streamlit-bound; the CLIs do not use it.

## 7. Configuration

- `config.py` holds Pydantic models, and `load_config()` builds `RAGConfig` from env.
- When changing a default, keep the Pydantic `Field` default equal to its `load_config()`
  fallback.
- Update `.env.example` and this file in the same change.

Code defaults (your `.env` may override them):

| Area | Env var(s) | Default |
|---|---|---|
| LLM | `LLM_MODEL` | `gpt-6.1-sol` |
| | `LLM_BACKEND` | `api` (`api`, `openai`) |
| | `LLM_TEMPERATURE`, `LLM_MAX_TOKENS` | 0.1, 65536 |
| | `LLM_THINKING_EFFORT` | empty = backend default (`none`, `minimal`, `low`, `medium`, `high`) |
| | `LLM_FALLBACK_MODELS` | empty (comma-separated) |
| Embeddings | `EMBEDDING_MODEL` | `voyage-4-lite` (200M free tokens) |
| Chunking | `CHUNKING_STRATEGY`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | `obsidian_aware`, 512, 75 |
| Retrieval | `RETRIEVAL_CANDIDATES` -> `RERANK_TOP_N` | 150 -> 20 |
| | `TOP_K` (UI display) | 75 |
| | `RERANKER_MODEL` | `rerank-2.5` |
| | `SIMILARITY_THRESHOLD` | 0.3, applied only when no reranker is configured |
| | `RERANK_SCORE_THRESHOLD` | 0.0 = off (0.1 filters weak reranked hits) |
| | `ENABLE_HYBRID_SEARCH`, `RRF_K` | true, 60 |
| | `ENABLE_GRAPH_RETRIEVAL` (wikilink expansion) | true (depth 1, max 3 links) |
| | `ENABLE_MMR`, `MAX_CHUNKS_PER_DOCUMENT` | false, 5 |
| Query transform | `HYDE_TEMPERATURE` | 0.7 (synthesis uses `LLM_TEMPERATURE`) |
| | `ENABLE_QUERY_DECOMPOSITION`, `ENABLE_CONTEXTUAL_COMPRESSION` | false, false |
| | `ENABLE_BILINGUAL_EXPANSION`, `EXPANSION_LANGUAGES` | false, `el` |
| Self-correction | `USE_SELF_CORRECTION`, `VALIDATE_RESPONSES` | true, true |
| Research | `RESEARCH_MAX_ITERATIONS`, `RESEARCH_CONFIDENCE_THRESHOLD` | 3, 0.8 |
| | `RESEARCH_MAX_SUBQUERIES`, `RESEARCH_GAP_ANALYSIS_TOP_N`, `RESEARCH_MAX_SYNTHESIS_SOURCES` | 3, 40, 0 |
| Conversations | `CONVERSATIONS_ENABLED`, `CONVERSATIONS_PATH`, `CONVERSATIONS_WEIGHT` | false, unset, 0.8 |
| Books | `BOOKS_ENABLED`, `BOOKS_PATH`, `CALIBRE_DB_PATH`, `BOOKS_WEIGHT` | false, unset, unset, 0.9 |
| Saved items | `SAVED_ITEMS_ENABLED`, `SAVED_ITEMS_LANCEDB_PATH`, `SAVED_ITEMS_WEIGHT` | false, `~/Projects/shared-data/lancedb`, 0.5 |
| Web search | `WEB_SEARCH_ENABLED`, `WEB_SEARCH_WEIGHT`, `WEB_SEARCH_MAX_RESULTS`, `WEB_SEARCH_IN_RESEARCH` | false, 0.7, 5, true |
| RAPTOR | `ENABLE_RAPTOR`, `RAPTOR_MODE`, `RAPTOR_CHUNK_SIZE`, `RAPTOR_TOP_K` | false, `collapsed`, 1024 (must exceed metadata size), 10 |
| Federated | `FEDERATED_TOP_K_PER_SOURCE`, `FEDERATED_FINAL_TOP_K` | 100, 120 |
| Result cache | `RESULT_CACHE_SIZE` | 64 |

- Bilingual expansion translates key concepts rather than the full query, and adds to
  HyDE/multi-query rather than replacing them. Supported languages: el, es, de, fr, it,
  pt, nl, ru, zh, ja, ko, ar.
- `CONVERSATIONS_PATH` points at the output of
  [ai-conversation-toolkit](https://github.com/silver-gr/ai-conversation-toolkit).

**Keys and paths:**
- `OBSIDIAN_VAULT_PATH` (required).
- `VOYAGE_API_KEY`: embeddings and rerank. Every query embeds through Voyage, so an
  invalid key breaks all search.
- `GOOGLE_API_KEY` (api backend), `LLM_BASE_URL` + `LLM_API_KEY` (openai backend).
- `TAVILY_API_KEY` (web search), `OBSIDIAN_VAULT_NAME`, `LANGSMITH_*` (optional tracing).

**LLM backends (`LLM_BACKEND`):**
- `api`: Google GenAI with `GOOGLE_API_KEY`. `LLM_THINKING_EFFORT` maps to the Gemini
  thinking config: `none`/`minimal` -> budget 0, `low` -> level low, `medium`/`high` ->
  level high.
- `openai`: an OpenAI-compatible proxy (e.g. LiteLLM). Configure it with `LLM_BASE_URL`
  (the `/v1` base) and `LLM_API_KEY`, which falls back to `GOOGLE_API_KEY`.
  - `LLM_THINKING_EFFORT` is forwarded as `reasoning_effort`.
  - Requests time out after 600 s, so long research responses are not cut off.

**Cost tracking:**
- `llm_token_tracker.py` aggregates LLM calls per day in `data/llm_usage.json`; the API is
  `get_llm_tracker().get_total_stats()`.
- `token_tracker.py` tracks Voyage usage against the free-tier limits in
  `data/voyage_usage.json`.
- Only Gemini models have prices. Any other model, including the default `gpt-6.1-sol`, is
  costed at the `default` entry ($0.50 in / $3.00 out per 1M tokens), so UI costs are
  estimates.

## 8. Data layout (`data/`, gitignored)

| Path | Contents |
|---|---|
| `lancedb/` | Tables `obsidian_embeddings` (vault, `VAULT_TABLE`), `conversations`, `books`, `settings` (exclusions) |
| `raptor/` | RAPTOR table `raptor_embeddings` |
| `cache/docstore_nodes.pkl` | docstore cache for fast restarts |
| `.cache_invalid` | touch to force a docstore reload |
| `index_checkpoint.json` | indexing checkpoint (`main.UltraRAG` only) |
| `auth/users.json` | user accounts |
| `users/<username>/` | per-user query history and research exports |
| `llm_usage.json`, `voyage_usage.json` | usage and cost tracking |
| `book_metadata_cache.json` | Calibre/web book metadata |
| `embedding_cache/` | unused (see `cache.py`) |

Saved items live outside the repo, in `SAVED_ITEMS_LANCEDB_PATH` (table `saved_items`).

**Docstore cache:**
- Invalidate it with `touch data/.cache_invalid`, the interactive `cache` command, or
  `vector_store.invalidate_cache()`.
- The cache key is a hash of the table name and row count, so re-indexing to the same
  row count needs a manual invalidation.
- When a persisted index is loaded, `vector_store.reconstruct_nodes_from_lancedb()`
  rebuilds the docstore from each row's `_node_content`. Hybrid (BM25) and graph retrieval
  depend on it.

**LanceDB rules:**
- `loader.py` uses a fixed schema. Variable frontmatter goes into the `extra_metadata` JSON
  field, so batches with different frontmatter never cause schema mismatches. Keep it that
  way.
- List tables with `list_tables()`; `vector_store.py` and `settings_store.py` each have a
  `_list_tables()` helper.
  - `table_names()` is deprecated in lancedb 0.40.
  - It is still called in `content_research_engine.py`, `raptor_index.py` and
    `saved_items_retriever.py`; migrate those calls when you touch the files.
- No schema changes without a migration note and tests.
- Indexing and delete-from-index destroy data. Never run them against the real `data/`
  from tests or experiments.

## 9. Testing

- Markers: `unit`, `integration`, `slow`. `--strict-markers` is on, so mark new tests.
- `pytest.ini` always adds coverage (`--cov=.`, writing `htmlcov/` and `coverage.xml`).
  Use `--no-cov` for quick runs.
- Tests must not touch real data, real `.env` values or paid APIs:
  - Pass the `test_config` fixture (temp LanceDB plus `tests/fixtures/sample_vault`).
    Calling functions with `config=None` falls back to `load_config()` and the real `.env`.
  - Mock embeddings with `patch('embeddings.get_embedding_model',
    return_value=mock_embedding_model)` (a `MockEmbedding`).
  - Test `retrieval.query()` by patching `retrieval._build_query_engine`. Otherwise the
    engine reads llama_index's global `Settings.llm`, which resolves to OpenAI.
  - The autouse `isolate_docstore_cache` fixture already redirects the docstore cache.
- `retrieval.py` keeps a module-level result cache keyed by query text + mode + filter.
  The same query string in two tests can return a cached result; use
  `retrieval.invalidate_result_cache()` to clear it.
- Which tests to run: `tests/test_indexing.py` after changing `indexing.py`,
  `tests/test_retrieval.py` after `retrieval.py`, and `tests/test_retrieval.py::TestResearch`
  for research mode.
- Evaluation needs `uv sync --extra eval`. Run `uv run python -m evaluation --dataset
  tests/evaluation_dataset.json [--output data/evaluation] [--threshold 0.5] [--no-save]`.
  Metrics: faithfulness, answer_relevancy, context_precision, context_recall.

## 10. Errors

All are defined in `models.py` and subclass `UltraRAGError`.

| Error | Likely cause | Fix |
|---|---|---|
| `IndexNotFoundError` | index not built | Build it (web admin panel, `main.py` startup prompt, or `indexing.index_vault(config)`); check `OBSIDIAN_VAULT_PATH` |
| `LLMRateLimitError` | 429 from the LLM | Wait, or set `LLM_FALLBACK_MODELS` |
| `EmbeddingQuotaError` | Voyage quota | Check `data/voyage_usage.json`. Defined, but not currently raised |
| `ConfigurationError` | missing or invalid path/key | The message names the env var |
| `voyageai.error.AuthenticationError` | invalid `VOYAGE_API_KEY` | Rotate the key |

## 11. Common tasks

- **New index source:**
  - `index_<source>()` in `indexing.py` (follow `index_books`)
  - the matching method in `main.UltraRAG` if the web app needs it
  - a `source_type` in `federated_query.IndexSource`
  - config in `config.py` and `.env.example`
  - a `--source` choice in `cli.py`
  - tests in `tests/test_indexing.py`
- **New query mode:** `retrieval._build_query_engine()` (`simple` -> `RAGQueryEngine`,
  anything else -> `HybridQueryEngine`) plus the equivalent path in `main.UltraRAG`.
  Tests go in `tests/test_retrieval.py`.
- **Research-mode bug:** `research_mode.py`, plus `main.UltraRAG.query_research` for
  synthesis or `retrieval.research()` for the CLI path.
- **File exclusions:** `settings_store.add_exclusion(db_path, pattern, pattern_type="glob")`
  with `glob`, `exact` or `regex`. See `docs/features/FILE_EXCLUSIONS.md`.

## 12. Change discipline

When you change runtime behavior, update in the same change wherever applicable:
- code and tests
- `.env.example`, if the config surface changed
- `docs/QUICKSTART.md` (startup flow), `docs/DEPLOYMENT.md` (ops), `docs/TESTING.md` (tests)
- README.md, for user-visible behavior
- `CHANGELOG.md`
- this file

High-risk areas - keep edits small and re-verify:
- `app.py` (~2.4k lines, stateful Streamlit flow) and `main.py` (~2.9k lines)
- auth and session logic in `auth.py`
- per-user path handling and history migration
- indexing and delete-from-index (both destroy data)
- defaults that change model choice or cost

Guardrails:
- Keep changes ASCII unless the file already requires Unicode. UI strings and synthesis
  prompts are Greek by design.
- No machine-specific absolute paths in runtime code or docs.
- No dead commands or doc references.
- Preserve backward compatibility where feasible; document breaking changes and migrations.

## 13. Known inconsistencies (verified 2026-10-20)

Fix these when you touch the area, and do not copy them elsewhere:
- **LLM defaults:** the code defaults pair `LLM_MODEL=gpt-6.1-sol` with `LLM_BACKEND=api`.
  That model is only reachable through the `openai` proxy backend. A config that sets
  neither needs `LLM_BACKEND=openai` + `LLM_BASE_URL`, or a Gemini `LLM_MODEL`.
- **Stale model names:**
  - `tracked_llm.py` and `llm_token_tracker.record_llm_usage` fall back to the model
    name `gemini-3-flash-preview`.
  - `research_mode.GAP_ANALYSIS_MODEL` is hard-coded to `gpt-6.1-sol`.
  - HyDE in `query_transform.py` follows the main LLM's model, falling back to the
    literal `gpt-6.1-sol` only when that LLM has no `model` attribute.
- **`context_cache.py`:** Gemini-only (needs `GOOGLE_API_KEY`). Errors are swallowed at
  debug level, and it is not wired to the `openai` backend.
- **`IndexNotFoundError` hint:** it says `VAULT_PATH`; the variable is `OBSIDIAN_VAULT_PATH`.
- **Duplicate types:** `models.py` duplicates `IndexProfile`, `ConvergenceConfig` and
  `IndexSource`. The live definitions are in `research_mode.py` and `federated_query.py`.
- **lancedb 0.40 warning:** opening a table logs `Index vector_idx has unrecognized type
  /lance.table.VectorIndexDetails, ignoring it`. It is harmless; vector search still uses
  the ANN index.
- **Version mismatch:** 1.6.1 in CHANGELOG.md, 1.6.0 in `pyproject.toml`.

## 14. Documentation

- Current:
  - `docs/ARCHITECTURE.md`, `docs/QUICKSTART.md`, `docs/DEPLOYMENT.md`,
    `docs/TESTING.md`, `docs/EVALUATION.md`, `docs/LANGSMITH_INTEGRATION.md`
  - `docs/features/`: BOOK_LIBRARY, CONTEXTUAL_RETRIEVAL, FILE_EXCLUSIONS, GRAPH_RETRIEVAL,
    LATE_CHUNKING, OBSESSION_RADAR, QUERY_INTENT_CLASSIFICATION (future work),
    QUERY_TRANSFORMATION, RESEARCH_MODE_ENHANCEMENTS, SELF_CORRECTION, V1_4_UPGRADE
  - `CHANGELOG.md`: version history.
- Historical, not authoritative: `docs/plans/`, `docs/prompts/`,
  `docs/reference/RAG_STRATEGY.md`, `docs/SESSION_*`, the `*_REPORT.md` / `*_EVALUATION.md`
  files and the competitive analyses.
