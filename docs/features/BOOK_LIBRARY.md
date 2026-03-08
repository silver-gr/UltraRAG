# Book Library Integration

Index, search, and filter your personal book collection (EPUB/PDF) alongside your Obsidian vault. Books are enriched with metadata from Calibre and optional web sources, enabling category/author filtering and intelligent 2-stage RAPTOR retrieval.

## Quick Start

```bash
# 1. Enable books in .env
BOOKS_ENABLED=true
BOOKS_PATH=/path/to/your/books

# 2. (Optional) Connect Calibre for rich metadata
CALIBRE_DB_PATH=/path/to/calibre/metadata.db

# 3. Enrich metadata
python -m cli enrich

# 4. Index books
python main.py   # then type: books

# 5. Search
python main.py   # then type: @books what is deep work
```

## Architecture Overview

```
Book Files (EPUB/PDF)
        │
        ▼
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Calibre Metadata  │────▶│  BookMetadata     │────▶│  Metadata Cache  │
│   Extraction        │     │  Cache (.json)    │     │  (persistent)    │
│   (calibre_metadata │     │                   │     │                  │
│    .py)             │     └──────────────────┘     └──────────────────┘
└─────────────────────┘              │
        │                            ▼
        │              ┌──────────────────────┐
        │              │  Web Enrichment      │
        │              │  (optional fallback)  │
        │              └──────────────────────┘
        │                            │
        ▼                            ▼
┌─────────────────────┐     ┌──────────────────┐
│  BookLoader         │◀────│  Enriched        │
│  (book_loader.py)   │     │  Metadata        │
└─────────────────────┘     └──────────────────┘
        │
        ▼
┌─────────────────────┐     ┌──────────────────┐
│  BookChunker        │────▶│  LanceDB         │
│  (1024-token chunks │     │  "books" table   │
│   chapter-aware)    │     │                  │
└─────────────────────┘     └──────────────────┘
```

## Configuration

Add these to your `.env` file:

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `BOOKS_ENABLED` | `false` | Enable the books integration |
| `BOOKS_PATH` | _(required)_ | Path to your books directory (EPUB/PDF files, searched recursively) |
| `BOOKS_WEIGHT` | `0.9` | Score weight in federated retrieval (vault = 1.0) |
| `BOOKS_IN_DEFAULT_SEARCH` | `true` | Include books in default federated ("Both") search |
| `BOOKS_TABLE_NAME` | `books` | LanceDB table name for the books index |

### Calibre Integration

| Variable | Default | Description |
|----------|---------|-------------|
| `CALIBRE_DB_PATH` | _(optional)_ | Path to Calibre's `metadata.db` SQLite file |
| `CALIBRE_MATCH_THRESHOLD` | `0.75` | Fuzzy match threshold (0.0-1.0). Higher = stricter matching |
| `BOOKS_EXCLUDE_TAGS` | `audio,notbooks,...` | Calibre tags to exclude from category metadata |

### Web Enrichment (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `BOOKS_WEB_ENRICH` | `false` | Enable web-based metadata enrichment fallback |
| `BOOKS_WEB_ENRICH_MAX` | `50` | Maximum books to web-enrich per run |

### Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `BOOKS_METADATA_CACHE_PATH` | `data/book_metadata_cache.json` | Path to persistent metadata cache |

## Metadata Enrichment

The enrichment pipeline populates a JSON cache with rich metadata for each book. This runs as a separate step **before indexing**.

### How Calibre Matching Works

1. Parses the filename using the `"Author - Title.ext"` convention
2. Runs dual-gate fuzzy matching against Calibre's database:
   - Both author AND title must individually exceed the match threshold
   - The best combined `(title_similarity + author_similarity) / 2` score wins
3. If only a title is available (no `" - "` separator), title-only matching is used with a penalty
4. Tags listed in `BOOKS_EXCLUDE_TAGS` are stripped from categories

### Enrichment Commands

```bash
# Run enrichment (Calibre + optional web)
python -m cli enrich

# Clear cache and re-enrich from scratch
python -m cli enrich --clear-cache

# Interactive CLI
python main.py   # then type: enrich
```

### Enrichment Stats

After enrichment, you'll see stats like:

```yaml
total: 834
calibre_matched: 712
avg_confidence: 0.89
web_enriched: 45
no_match: 77
already_cached: 0
```

### Metadata Fields

Each book can have:

| Field | Source | Description |
|-------|--------|-------------|
| `title` | Calibre/filename | Book title |
| `author` | Calibre/filename | Author name |
| `categories` | Calibre tags | Genre/topic categories (list) |
| `description` | Calibre/web | Book synopsis |
| `language` | Calibre | Language code (e.g., `en`) |
| `publisher` | Calibre | Publisher name |
| `isbn` | Calibre | ISBN identifier |
| `calibre_id` | Calibre | Calibre database ID |
| `match_confidence` | Computed | Fuzzy match score (0.0-1.0) |
| `metadata_source` | System | One of: `calibre`, `web`, `calibre+web`, `filename` |

## Indexing

After enrichment, index your books:

```bash
# Interactive CLI
python main.py   # then type: books

# If index already exists, you'll be prompted:
#   [L]oad existing / [R]ecreate / [S]kip
```

### Chunking Strategy

Books use a specialized chunker (`BookChunker`):
- **Chunk size**: 1024 tokens (vs 512 for vault notes)
- **Overlap**: 128 tokens
- **Chapter-aware**: Recognizes `Chapter N`, `Part N`, `Step N`, `Lesson N`, `Week N`, `Day N` headers
- **Paragraph-aware**: Merges short paragraphs to avoid fragment chunks

## Searching Books

### Interactive CLI

```bash
python main.py

# Search all books
> @books what is deep work

# Search books filtered by category
> @books:productivity what is deep work

# Search books filtered by category (multi-word)
> @books:rest-recovery how to improve sleep

# 2-stage RAPTOR search (see RAPTOR section below)
> @raptor-books what are the best productivity techniques
```

### Non-Interactive CLI

```bash
# Search books only
python -m cli query --query "habits for productivity" --source books

# Search everything including books
python -m cli query --query "meditation techniques" --source all

# Filter by category
python -m cli query --query "deep work" --source books --category "productivity"

# Filter by author
python -m cli query --query "atomic habits" --source books --author "James Clear"

# Multiple filters (categories OR within field, AND across fields)
python -m cli query --query "focus" --source books \
  --category "productivity,habits" \
  --author "Newport,Clear"
```

### Web UI (Streamlit)

1. Run `streamlit run app.py`
2. Set search scope to **Both** (federated)
3. Expand **Book Filters** in the sidebar:
   - **Category multiselect**: Shows all categories with document counts
   - **Author text input**: Free-text, comma-separated
4. The active WHERE clause is displayed as an info message
5. Click **Clear filters** to reset

> Book filters only apply in "Both" (federated) search scope.

## Book Filtering (Technical Details)

Filters use native LanceDB DataFusion SQL via `BookFilter.to_lance_where()`:

```python
BookFilter(
    categories=["productivity", "habits"],
    authors=["Cal Newport"]
).to_lance_where()

# Generates:
# array_has_any(metadata.book_categories, make_array('productivity', 'habits'))
# AND metadata.book_author IN ('Cal Newport')
```

- **Categories**: OR semantics within selection (any match)
- **Authors**: OR semantics within selection
- **Cross-field**: AND semantics (must match both category AND author)
- **SQL-safe**: Single quotes are escaped (`O'Reilly` -> `O''Reilly`)
- **Normalized**: All categories are lowercased and whitespace-collapsed before comparison

## 2-Stage Book-Summary RAPTOR Retrieval

Standard vector search across 800+ books can produce topic collision (chunks from unrelated books surfacing due to shared vocabulary). The 2-stage RAPTOR approach solves this.

### How It Works

```
Query: "best productivity techniques"
              │
              ▼
┌─────────────────────────────┐
│  Stage 1: RAPTOR Summary    │
│  Which books are relevant?  │
│  (searches book summaries)  │
│  → Returns top 5 book UIDs  │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 2: Filtered Search   │
│  Search ONLY within those   │
│  5 books (WHERE clause)     │
│  → Returns actual chunks    │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Synthesis                  │
│  LLM generates answer from  │
│  focused, relevant chunks   │
└─────────────────────────────┘
```

**Stage 1** uses a separate RAPTOR index built from book summaries (title + author + categories + description). Each book is one document. The `book_uid` is encoded into the `file_path` metadata field to survive RAPTOR's metadata stripping.

**Stage 2** uses `BookFilter(book_uids=[...])` to create a native LanceDB WHERE clause that limits the search to only the relevant books.

### Commands

```bash
# Interactive CLI
python main.py
> @raptor-books what are the best productivity techniques

# The RAPTOR summary index is built automatically on first use
# (requires metadata cache from 'enrich' step)
```

### Index Location

The books summary RAPTOR index is stored at `data/raptor/books_summary` (separate from the vault RAPTOR at `data/raptor`).

> After re-running `enrich`, the RAPTOR summary index may be stale. It will be rebuilt automatically on next `@raptor-books` query.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Books not appearing in search | Check `BOOKS_ENABLED=true` and that you've run both `enrich` and `books` index |
| Poor Calibre matching | Lower `CALIBRE_MATCH_THRESHOLD` (e.g., 0.6) or check filename format (`Author - Title.ext`) |
| Category filter returns nothing | Use `@books:category` with the normalized category name (lowercase, no extra spaces) |
| `@raptor-books` is slow first time | The RAPTOR summary index is built on first use; subsequent queries are fast |
| Stale metadata after adding books | Re-run `python -m cli enrich` then re-index with `books` command |

## Key Files

| File | Purpose |
|------|---------|
| `book_loader.py` | Book discovery and loading (EPUB/PDF) |
| `book_chunker.py` | Book-specific chunking (1024 tokens, chapter-aware) |
| `calibre_metadata.py` | Calibre DB extraction, fuzzy matching, metadata cache |
| `web_metadata_enricher.py` | Optional web-based metadata fallback |
| `books_retriever.py` | Shared retriever with WHERE clause support |
| `book_raptor.py` | 2-stage RAPTOR summary retrieval |
| `models.py` | `BookMetadata`, `BookFilter`, `_normalize_category()` |
| `config.py` | `BooksConfig` configuration class |
