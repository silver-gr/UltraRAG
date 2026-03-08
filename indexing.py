"""
UltraRAG Indexing Module

Handles all document indexing: vault, conversations, books.
Agent note: This is the ONLY file you need to read for indexing tasks.

PUBLIC API:
- index_vault(config) -> VectorStoreIndex
- index_conversations(config) -> VectorStoreIndex
- index_books(config) -> VectorStoreIndex
- enrich_books(config) -> dict
- load_index(config, source) -> VectorStoreIndex | None
- index_exists(config, source) -> bool
"""

import logging
from pathlib import Path
from typing import Literal

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, Document

from config import RAGConfig
from models import IndexNotFoundError, ConfigurationError

logger = logging.getLogger("ultrarag.indexing")


# === PUBLIC API ===

def index_vault(config: RAGConfig) -> VectorStoreIndex | None:
    """
    Index Obsidian vault and return ready-to-query index.

    Args:
        config: RAG configuration with vault_path set

    Returns:
        VectorStoreIndex ready for queries, or None if vault is empty

    Raises:
        ConfigurationError: If vault_path is invalid
        EmbeddingQuotaError: If Voyage quota exceeded

    Example:
        >>> config = load_config()
        >>> index = index_vault(config)
        >>> # Now ready to query
    """
    logger.info("Starting vault indexing | path=%s", config.vault_path)

    # Validate config
    if not config.vault_path or not Path(config.vault_path).exists():
        raise ConfigurationError("OBSIDIAN_VAULT_PATH", "/path/to/obsidian/vault")

    # Load documents
    from loader import ObsidianLoader
    loader = ObsidianLoader(Path(config.vault_path))

    exclusion_patterns = _get_exclusion_patterns(config)
    notes = loader.load_vault(exclusion_patterns=exclusion_patterns)

    if not notes:
        logger.warning("No notes found in vault | path=%s", config.vault_path)
        return None

    documents = loader.notes_to_documents(notes)
    wikilink_graph = loader.build_wikilink_graph(notes)

    logger.info("Loaded documents | count=%d", len(documents))

    # Chunk documents
    from chunking import ObsidianChunker
    from embeddings import get_embedding_model

    embed_model = get_embedding_model(config.embedding)
    chunker = ObsidianChunker(config.embedding, embed_model)
    nodes = chunker.chunk_documents(documents)

    logger.info("Chunked into nodes | count=%d", len(nodes))

    # Create index
    from vector_store import get_vector_store, create_vector_index

    vector_store = get_vector_store(
        config.vector_db,
        mode="overwrite",
        table_name=config.vector_db.vault_table
    )
    index = create_vector_index(nodes, vector_store, embed_model)

    # Store wikilink graph in index metadata for graph retrieval
    index.wikilink_graph = wikilink_graph

    logger.info("Index created | nodes=%d", len(nodes))

    return index


def index_conversations(config: RAGConfig) -> VectorStoreIndex | None:
    """
    Index AI conversation exports.

    Args:
        config: RAG configuration with conversations_path set

    Returns:
        VectorStoreIndex of conversations, or None if empty

    Raises:
        ConfigurationError: If conversations_path is invalid
    """
    conversations_path = config.conversations.path
    logger.info("Starting conversations indexing | path=%s", conversations_path)

    if not conversations_path or not Path(conversations_path).exists():
        raise ConfigurationError("CONVERSATIONS_PATH", "/path/to/exports")

    from conversation_loader import ConversationLoader, ConversationChunker
    from embeddings import get_embedding_model
    from vector_store import get_vector_store, create_vector_index

    # Load conversations
    loader = ConversationLoader(Path(conversations_path))
    documents = loader.load_all()

    if not documents:
        logger.warning("No conversations found")
        return None

    # Chunk with turn-aware strategy
    embed_model = get_embedding_model(config.embedding)
    chunker = ConversationChunker(config.embedding)
    nodes = chunker.chunk_conversations(documents)

    # Create index in separate table
    vector_store = get_vector_store(
        config.vector_db,
        mode="overwrite",
        table_name=config.vector_db.conversations_table
    )
    index = create_vector_index(nodes, vector_store, embed_model)

    logger.info("Conversations indexed | count=%d", len(nodes))

    return index


def index_books(config: RAGConfig) -> VectorStoreIndex | None:
    """
    Index EPUB/PDF books.

    Args:
        config: RAG configuration with books_path set

    Returns:
        VectorStoreIndex of books, or None if empty

    Raises:
        ConfigurationError: If books_path is invalid
    """
    books_path = config.books.path
    logger.info("Starting books indexing | path=%s", books_path)

    if not books_path or not Path(books_path).exists():
        raise ConfigurationError("BOOKS_PATH", "/path/to/books")

    from book_loader import BookLoader
    from book_chunker import BookChunker, BookChunkConfig
    from embeddings import get_embedding_model
    from vector_store import get_vector_store, create_vector_index

    loader = BookLoader(Path(books_path))

    # Load enrichment cache if it exists
    cache_path = config.books.metadata_cache_path
    if cache_path and Path(cache_path).exists():
        from calibre_metadata import BookMetadataCache
        cache = BookMetadataCache(Path(cache_path))
        loader.set_enrichment_cache(cache)
        logger.info("Loaded book metadata cache | entries=%d", len(cache))

    documents = loader.load_all_books(show_progress=True)

    if not documents:
        logger.warning("No books found")
        return None

    logger.info("Loaded documents | count=%d", len(documents))

    embed_model = get_embedding_model(config.embedding)
    chunk_config = BookChunkConfig(
        chunk_size=config.books.book_chunk_size,
        chunk_overlap=config.books.book_chunk_overlap,
        min_chunk_size=config.books.book_min_chunk_size,
        respect_chapters=config.books.book_respect_chapters,
        respect_paragraphs=config.books.book_respect_paragraphs,
    )
    chunker = BookChunker(config=chunk_config)
    nodes = chunker.chunk_documents(documents)

    logger.info("Chunked into nodes | count=%d", len(nodes))

    vector_store = get_vector_store(
        config.vector_db,
        mode="overwrite",
        table_name=config.books.table_name,
        flat_metadata=False,
    )
    index = create_vector_index(nodes, vector_store, embed_model)

    logger.info("Books indexed | count=%d", len(nodes))

    return index


def enrich_books(config: RAGConfig) -> dict:
    """
    Run metadata enrichment pipeline: Calibre extract -> web enrich -> save cache.

    This is an explicit pre-step before indexing. Run 'enrich' first, then 'index'.

    Args:
        config: RAG configuration with books settings

    Returns:
        Dict with enrichment statistics

    Example:
        >>> stats = enrich_books(config)
        >>> print(f"Matched: {stats['calibre_matched']}, Web: {stats['web_enriched']}")
    """
    books_path = config.books.path
    if not books_path or not Path(books_path).exists():
        raise ConfigurationError("BOOKS_PATH", "/path/to/books")

    from book_loader import BookLoader
    from calibre_metadata import CalibreMetadataExtractor, BookMetadataCache

    loader = BookLoader(Path(books_path))
    book_files = loader.discover_books()

    if not book_files:
        logger.warning("No books found for enrichment")
        return {"total": 0}

    cache = BookMetadataCache(Path(config.books.metadata_cache_path))

    stats = {
        "total": len(book_files),
        "calibre_matched": 0,
        "web_enriched": 0,
        "no_match": 0,
        "already_cached": 0,
        "excluded_tags": 0,
        "avg_confidence": 0.0,
    }

    # Phase 1: Calibre extraction
    calibre_extractor = None
    if config.books.calibre_db_path and Path(config.books.calibre_db_path).exists():
        calibre_extractor = CalibreMetadataExtractor(
            db_path=Path(config.books.calibre_db_path),
            match_threshold=config.books.calibre_match_threshold,
            exclude_tags=frozenset(config.books.exclude_tags),
        )
        logger.info("Calibre DB loaded: %s", config.books.calibre_db_path)
    else:
        logger.warning("No Calibre DB configured — using filename metadata only")

    needs_web = []
    confidences = []

    for book_path in book_files:
        # Reuse cache when possible, but allow incremental upgrades:
        # - retry Calibre matching for cached filename/web-only entries
        # - allow web enrichment for cached entries missing categories/description
        cached_meta = cache.lookup(book_path)
        if cached_meta is not None:
            # Try upgrading cached non-Calibre entries if Calibre is now available
            if calibre_extractor and cached_meta.calibre_id is None:
                upgraded = calibre_extractor.match_book(book_path)
                if upgraded:
                    # Preserve cached fields if Calibre doesn't provide them
                    if not upgraded.categories and cached_meta.categories:
                        upgraded.categories = cached_meta.categories
                    if not upgraded.description and cached_meta.description:
                        upgraded.description = cached_meta.description
                    if cached_meta.metadata_source in {"web", "calibre+web"}:
                        upgraded.metadata_source = "calibre+web"
                    cache.put(upgraded)
                    cached_meta = upgraded
                    stats["calibre_matched"] += 1
                    confidences.append(upgraded.match_confidence)

            # Cached but incomplete entries can still go through web enrichment
            if config.books.web_enrich_enabled and (
                not cached_meta.categories or not cached_meta.description
            ):
                needs_web.append(cached_meta)
                continue

            stats["already_cached"] += 1
            continue

        meta = None
        if calibre_extractor:
            meta = calibre_extractor.match_book(book_path)

        if meta:
            stats["calibre_matched"] += 1
            confidences.append(meta.match_confidence)
            cache.put(meta)
            if not meta.categories or not meta.description:
                needs_web.append(meta)
        else:
            # Filename-only metadata
            from models import BookMetadata
            file_type = book_path.suffix.lower().lstrip(".")
            file_size = book_path.stat().st_size if book_path.exists() else 0
            meta = BookMetadata(
                title=book_path.stem,
                file_path=str(book_path),
                file_type=file_type,
                file_size=file_size,
                metadata_source="filename",
            )
            cache.put(meta)
            needs_web.append(meta)
            stats["no_match"] += 1

    # Phase 2: Web enrichment (optional)
    if config.books.web_enrich_enabled and needs_web:
        try:
            from web_metadata_enricher import WebMetadataEnricher
            enricher = WebMetadataEnricher(max_per_run=config.books.web_enrich_max_per_run)
            if enricher.available:
                _, web_count = enricher.enrich_batch(needs_web)
                stats["web_enriched"] = web_count
                # Re-cache the web-enriched entries
                for meta in needs_web:
                    cache.put(meta)
        except Exception as e:
            logger.warning("Web enrichment failed: %s", e)

    # Save cache
    cache.save()

    if confidences:
        stats["avg_confidence"] = round(sum(confidences) / len(confidences), 4)

    # Check if RAPTOR index might be stale
    raptor_path = Path("data/raptor/books_summary")
    if raptor_path.exists():
        logger.warning(
            "RAPTOR books_summary index exists — may be stale after enrichment. "
            "Rebuild with 'raptor-books' command."
        )

    logger.info(
        "Enrichment complete: %d matched (avg confidence %.2f), "
        "%d web-enriched, %d no match, %d already cached",
        stats["calibre_matched"], stats["avg_confidence"],
        stats["web_enriched"], stats["no_match"], stats["already_cached"],
    )

    return stats


def load_index(
    config: RAGConfig,
    source: Literal["vault", "conversations", "books"] = "vault"
) -> VectorStoreIndex | None:
    """
    Load existing index from disk.

    Args:
        config: RAG configuration
        source: Which index to load

    Returns:
        VectorStoreIndex if exists, None otherwise

    Example:
        >>> index = load_index(config, source="vault")
        >>> if index is None:
        ...     index = index_vault(config)
    """
    table_names = {
        "vault": config.vector_db.vault_table,
        "conversations": config.vector_db.conversations_table,
        "books": config.books.table_name,
    }

    table_name = table_names.get(source, config.vector_db.vault_table)

    if not index_exists(config, source):
        logger.debug("Index not found | source=%s", source)
        return None

    from embeddings import get_embedding_model
    from vector_store import load_vector_index, get_vector_store

    embed_model = get_embedding_model(config.embedding)
    vector_store = get_vector_store(
        config.vector_db,
        mode="append",
        table_name=table_name,
        flat_metadata=(source != "books"),
    )

    index = load_vector_index(vector_store, embed_model, config.vector_db, table_name=table_name)

    logger.info("Index loaded | source=%s", source)

    return index


def index_exists(
    config: RAGConfig,
    source: Literal["vault", "conversations", "books"] = "vault"
) -> bool:
    """
    Check if index exists without loading it.

    Args:
        config: RAG configuration
        source: Which index to check

    Returns:
        True if index exists on disk
    """
    from vector_store import index_exists as vs_index_exists

    table_names = {
        "vault": config.vector_db.vault_table,
        "conversations": config.vector_db.conversations_table,
        "books": config.books.table_name,
    }

    table_name = table_names.get(source, config.vector_db.vault_table)

    return vs_index_exists(config.vector_db, table_name=table_name)


# === INTERNAL HELPERS ===

def _get_exclusion_patterns(config: RAGConfig) -> list[dict] | None:
    """Get exclusion patterns from settings store."""
    try:
        from settings_store import get_exclusions
        return get_exclusions(str(config.vector_db.lancedb_path))
    except Exception:
        return None
