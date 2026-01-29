"""
UltraRAG Indexing Module

Handles all document indexing: vault, conversations, books.
Agent note: This is the ONLY file you need to read for indexing tasks.

PUBLIC API:
- index_vault(config) -> VectorStoreIndex
- index_conversations(config) -> VectorStoreIndex
- index_books(config) -> VectorStoreIndex
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
        raise ConfigurationError("VAULT_PATH", "/path/to/obsidian/vault")

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

    vector_store = get_vector_store(config.vector_db, mode="overwrite", table_name="vectors")
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
    logger.info("Starting conversations indexing | path=%s", config.conversations_path)

    if not config.conversations_path or not Path(config.conversations_path).exists():
        raise ConfigurationError("CONVERSATIONS_PATH", "/path/to/exports")

    from conversation_loader import ConversationLoader, ConversationChunker
    from embeddings import get_embedding_model
    from vector_store import get_vector_store, create_vector_index

    # Load conversations
    loader = ConversationLoader(Path(config.conversations_path))
    documents = loader.load_all()

    if not documents:
        logger.warning("No conversations found")
        return None

    # Chunk with turn-aware strategy
    embed_model = get_embedding_model(config.embedding)
    chunker = ConversationChunker(config.embedding)
    nodes = chunker.chunk_conversations(documents)

    # Create index in separate table
    vector_store = get_vector_store(config.vector_db, mode="overwrite", table_name="conversations")
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
    """
    logger.info("Starting books indexing | path=%s", config.books_path)

    if not config.books_path or not Path(config.books_path).exists():
        raise ConfigurationError("BOOKS_PATH", "/path/to/books")

    from book_loader import BookLoader
    from book_chunker import BookChunker
    from embeddings import get_embedding_model
    from vector_store import get_vector_store, create_vector_index

    loader = BookLoader(Path(config.books_path))
    documents = loader.load_all()

    if not documents:
        logger.warning("No books found")
        return None

    embed_model = get_embedding_model(config.embedding)
    chunker = BookChunker(config.embedding)
    nodes = chunker.chunk_books(documents)

    vector_store = get_vector_store(config.vector_db, mode="overwrite", table_name="books")
    index = create_vector_index(nodes, vector_store, embed_model)

    logger.info("Books indexed | count=%d", len(nodes))

    return index


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
        "vault": "vectors",
        "conversations": "conversations",
        "books": "books",
    }

    table_name = table_names.get(source, "vectors")

    if not index_exists(config, source):
        logger.debug("Index not found | source=%s", source)
        return None

    from embeddings import get_embedding_model
    from vector_store import load_vector_index, get_vector_store

    embed_model = get_embedding_model(config.embedding)
    vector_store = get_vector_store(config.vector_db, mode="append", table_name=table_name)

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
        "vault": "vectors",
        "conversations": "conversations",
        "books": "books",
    }

    table_name = table_names.get(source, "vectors")

    return vs_index_exists(config.vector_db, table_name=table_name)


# === INTERNAL HELPERS ===

def _get_exclusion_patterns(config: RAGConfig) -> list[dict] | None:
    """Get exclusion patterns from settings store."""
    try:
        from settings_store import get_exclusions
        return get_exclusions(str(config.vector_db.lancedb_path))
    except Exception:
        return None
