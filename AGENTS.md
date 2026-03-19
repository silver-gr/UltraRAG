# AGENTS.md

This file is the canonical maintenance guide for coding agents working in UltraRAG.

## 1) Project Purpose

UltraRAG is a production-oriented RAG system for Obsidian knowledge bases with:
- vault indexing and retrieval
- optional federated sources (conversations, books, saved items)
- research mode / self-correction / RAPTOR
- Streamlit web UI + CLI entry points

Primary goals when changing code:
- correctness and reproducibility
- privacy-safe multi-user behavior
- documentation/config/runtime consistency

## 2) Canonical Runtime Defaults (Do Not Drift)

These defaults are contract-level and should remain aligned across code + docs + scripts:

- Streamlit HTTPS endpoint: `https://localhost:9001`
- `.streamlit/config.toml` uses:
  - `sslCertFile = "certs/frontend.pem"`
  - `sslKeyFile = "certs/frontend-key.pem"`
  - `port = 9001`
- In-app auth/roles are supported for multi-user usage.

If you change any of the above, update all relevant docs and bootstrap scripts in the same change.

## 3) Core Entry Points

- `app.py`: main Streamlit UI (auth gate, query flow, admin controls)
- `pages/content_research.py`: content research UI page
- `main.py`: interactive CLI
- `cli.py`: non-interactive CLI for automation/agents

## 4) Module Ownership Boundaries

- `indexing.py`: ingestion/index creation/load/exists (vault, conversations, books)
- `retrieval.py`, `query_engine.py`, `federated_query.py`, `research_mode.py`: query/retrieval behavior
- `config.py`: Pydantic config models + env loading
- `vector_store.py`, `settings_store.py`: LanceDB/Qdrant integration
- `auth.py`: auth context/session/password helpers
- `rate_limit.py`: global concurrency + per-user cooldown controls
- `user_storage.py`: per-user path resolution helpers
- `research_storage.py`: per-user content research persistence/export
- `ui_theme.py`: shared Streamlit theme injection

Prefer focused changes per module; avoid cross-module coupling unless necessary.

## 5) Multi-User Privacy and Security Rules

Never introduce shared user data in global files.

Per-user storage contract:
- user root: `data/users/<username>/`
- query history: `data/users/<username>/query_history.json`
- research exports: `data/users/<username>/research-exports/`

Auth/role model:
- roles: `admin`, `user` only
- admin-only actions: indexing/reindexing/settings/cost resets/other mutating controls
- user role is query-capable and must not see mutation controls

If auth is disabled (`ULTRARAG_AUTH_ENABLED=false`), app falls back to local admin context.

## 6) Concurrency / Stampede Protection Rules

Use `run_with_limits(ctx, job_name)` for expensive operations:
- queries/search/research
- index build/rebuild
- export-heavy operations
- index mutations/settings actions

Config keys:
- `ULTRARAG_MAX_CONCURRENT_JOBS`
- `ULTRARAG_MIN_SECONDS_BETWEEN_QUERIES`
- `ULTRARAG_JOB_ACQUIRE_TIMEOUT_SECONDS`

Do not bypass limiter wrappers for new heavy operations.

## 7) Config Discipline

When changing defaults in `config.py`:
- keep Pydantic defaults and `load_config()` fallbacks aligned
- update `.env.example`
- update docs mentioning those defaults

Treat config/documentation drift as a bug.

## 8) Storage / LanceDB Rules

- Prefer `db.list_tables()` compatibility path, not `table_names()`.
- Keep table-name handling consistent with existing patterns in `main.py`.
- Avoid schema changes without migration notes/tests.

## 9) Setup / Deployment Workflow

Recommended local bootstrap:

```bash
cd /Users/silver/Projects/UltraRAG
./setup.sh
./scripts/generate_dev_certs.sh
python -m scripts.manage_users init --admin <username>
source venv/bin/activate
streamlit run app.py
```

User management CLI (`scripts/manage_users.py`):
- `init --admin <name>`
- `add-user --username <name> --role user|admin`
- `set-password --username <name>`
- `set-role --username <name> --role user|admin`
- `list-users`

## 10) Testing Expectations

Minimum pre-merge check for most changes:

```bash
python -m pytest -m unit
```

Run broader tests when touching retrieval/indexing/integration behavior:

```bash
python -m pytest
python -m pytest -m integration
```

When adding core behavior, add/update tests under `tests/` with proper markers.

## 11) Required Update Checklist for Behavior Changes

If you change runtime behavior, verify all applicable items are updated in the same PR:

- Code implementation
- Tests
- `.env.example` (if config surface changed)
- `docs/QUICKSTART.md` (user startup flow)
- `docs/DEPLOYMENT.md` (ops/deploy assumptions)
- `docs/TESTING.md` (test layout/commands/markers)
- `CLAUDE.md` (agent-focused command and architecture references)
- `README.md` if user-visible setup/feature semantics changed

## 12) High-Risk Areas (Extra Care)

- `app.py` (large UI/stateful flow): prefer small, explicit edits and re-check syntax
- auth/session logic in `auth.py`: avoid silent role escalation
- file path handling and migrations in user storage/history
- indexing and delete-from-index operations (can destroy data)
- config defaults that influence model/cost behavior

## 13) Guardrails

- Keep changes ASCII unless file already requires Unicode.
- Avoid hardcoding machine-specific absolute paths in runtime logic.
- Do not introduce dead commands/docs references.
- Preserve backward compatibility where feasible; if breaking, document migration clearly.

<!-- gitnexus:start -->
# GitNexus MCP

This project is indexed by GitNexus as **UltraRAG** (3621 symbols, 6705 relationships, 133 execution flows).

## Always Start Here

1. **Read `gitnexus://repo/{name}/context`** — codebase overview + check index freshness
2. **Match your task to a skill below** and **read that skill file**
3. **Follow the skill's workflow and checklist**

> If step 1 warns the index is stale, run `npx gitnexus analyze` in the terminal first.

## Skills

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
