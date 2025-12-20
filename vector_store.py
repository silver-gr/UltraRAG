"""Vector database initialization and management."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from config import VectorDBConfig

logger = logging.getLogger(__name__)


def index_exists(config: VectorDBConfig) -> bool:
    """Check if an index already exists in the vector store.

    Args:
        config: Vector database configuration

    Returns:
        True if index exists, False otherwise
    """
    db_type = config.db_type.lower()

    if db_type == "lancedb":
        import lancedb

        # Check if LanceDB directory and table exist
        if not config.lancedb_path.exists():
            return False

        try:
            db = lancedb.connect(str(config.lancedb_path))
            table_names = db.table_names()
            return "obsidian_embeddings" in table_names
        except Exception:
            return False

    elif db_type == "qdrant":
        from qdrant_client import QdrantClient

        try:
            client = QdrantClient(
                host=config.qdrant_host,
                port=config.qdrant_port
            )
            collections = client.get_collections()
            return config.qdrant_collection in [c.name for c in collections.collections]
        except Exception:
            return False

    return False


def get_vector_store(config: VectorDBConfig, mode: str = "append") -> Any:
    """Initialize vector store based on configuration.

    Args:
        config: Vector database configuration
        mode: Mode for LanceDB - "append" (default) or "overwrite"

    Returns:
        Initialized vector store
    """
    db_type = config.db_type.lower()
    logger.info(f"Initializing vector store: {db_type}")

    try:
        if db_type == "lancedb":
            from llama_index.vector_stores.lancedb import LanceDBVectorStore
            import lancedb

            # Create directory if it doesn't exist
            try:
                config.lancedb_path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Vector store directory created/verified: {config.lancedb_path.parent}")
            except PermissionError as e:
                logger.error(f"Permission denied creating vector store directory: {config.lancedb_path.parent}")
                raise PermissionError(
                    f"Cannot create vector store directory at {config.lancedb_path.parent}. "
                    "Please check file permissions."
                ) from e

            # Initialize LanceDB
            try:
                db = lancedb.connect(str(config.lancedb_path))
                logger.debug(f"Connected to LanceDB at: {config.lancedb_path}")
            except Exception as e:
                logger.error(f"Failed to connect to LanceDB: {e}", exc_info=True)
                raise RuntimeError(f"Failed to connect to LanceDB: {e}") from e

            return LanceDBVectorStore(
                uri=str(config.lancedb_path),
                table_name="obsidian_embeddings",
                mode=mode
            )

        elif db_type == "qdrant":
            from llama_index.vector_stores.qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient

            try:
                client = QdrantClient(
                    host=config.qdrant_host,
                    port=config.qdrant_port
                )
                logger.debug(f"Connected to Qdrant at {config.qdrant_host}:{config.qdrant_port}")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant at {config.qdrant_host}:{config.qdrant_port}: {e}")
                raise ConnectionError(
                    f"Cannot connect to Qdrant at {config.qdrant_host}:{config.qdrant_port}. "
                    "Make sure Qdrant server is running."
                ) from e

            return QdrantVectorStore(
                client=client,
                collection_name=config.qdrant_collection
            )

        else:
            logger.error(f"Unsupported vector database type: {db_type}")
            raise ValueError(
                f"Unsupported vector database: {db_type}. "
                "Supported databases: lancedb, qdrant"
            )

    except ImportError as e:
        logger.error(f"Failed to import vector store library: {e}")
        raise ImportError(
            f"Missing required library for {db_type}. "
            f"Install it with: pip install llama-index-vector-stores-{db_type}"
        ) from e


def create_vector_index(
    nodes: list[TextNode],
    vector_store: Any,
    embed_model: BaseEmbedding,
    show_progress: bool = True
) -> VectorStoreIndex:
    """Create vector index from nodes."""
    logger.info(f"Creating vector index with {len(nodes)} nodes")

    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=show_progress
        )

        logger.info("Vector index created successfully")
        return index

    except Exception as e:
        logger.error(f"Failed to create vector index: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create vector index: {e}") from e


def load_vector_index(
    vector_store: Any,
    embed_model: BaseEmbedding
) -> VectorStoreIndex:
    """Load existing vector index."""
    logger.info("Loading existing vector index")

    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            storage_context=storage_context
        )

        logger.info("Vector index loaded successfully")
        return index

    except Exception as e:
        logger.error(f"Failed to load vector index: {e}", exc_info=True)
        raise RuntimeError(
            f"Failed to load vector index: {e}. "
            "The index may not exist yet - try running indexing first."
        ) from e
