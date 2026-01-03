"""RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) implementation.

RAPTOR works by recursively clustering documents and generating summaries at multiple
abstraction levels. This enables both detailed and high-level retrieval.

Two retrieval modes:
- collapsed: Treats entire tree as flat list, simple top-k (faster)
- tree_traversal: Traverses hierarchy, top-k at each level (more comprehensive)

Reference: https://arxiv.org/abs/2401.18059
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Literal

from llama_index.core import Document
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# Type alias for RAPTOR modes
RaptorMode = Literal["collapsed", "tree_traversal"]


class RaptorIndexManager:
    """Manages RAPTOR index creation, persistence, and retrieval.

    RAPTOR creates hierarchical summaries through recursive clustering:
    1. Original documents are chunked and embedded
    2. Similar chunks are clustered (using k-means or UMAP + GMM)
    3. Each cluster is summarized by the LLM
    4. Summaries become new nodes for the next level
    5. Process repeats until convergence

    This creates a tree where:
    - Leaf nodes = original document chunks
    - Internal nodes = summaries of clusters
    - Root = highest-level summary
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
        llm: LLM,
        raptor_path: Path,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        similarity_top_k: int = 10,
        default_mode: RaptorMode = "collapsed"
    ):
        """Initialize RAPTOR index manager.

        Args:
            embed_model: Embedding model for clustering
            llm: LLM for generating cluster summaries
            raptor_path: Path to store RAPTOR index (LanceDB)
            chunk_size: Size of initial document chunks
            chunk_overlap: Overlap between chunks
            similarity_top_k: Number of results to retrieve
            default_mode: Default retrieval mode ("collapsed" or "tree_traversal")
        """
        self.embed_model = embed_model
        self.llm = llm
        self.raptor_path = Path(raptor_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.default_mode = default_mode

        # Initialize components
        self._pack = None
        self._retriever = None
        self._vector_store = None

        logger.info(
            f"RaptorIndexManager initialized: path={raptor_path}, "
            f"chunk_size={chunk_size}, mode={default_mode}"
        )

    def _get_vector_store(self, mode: str = "append"):
        """Get or create LanceDB vector store for RAPTOR.

        Args:
            mode: "append" to add to existing, "overwrite" to recreate

        Returns:
            LanceDBVectorStore instance
        """
        from llama_index.vector_stores.lancedb import LanceDBVectorStore
        import lancedb

        # Ensure directory exists
        self.raptor_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to LanceDB
        db = lancedb.connect(str(self.raptor_path))
        table_name = "raptor_embeddings"

        # Check if table exists
        existing_tables = db.table_names()
        table_exists = table_name in existing_tables

        if mode == "overwrite" and table_exists:
            logger.info(f"Dropping existing RAPTOR table: {table_name}")
            db.drop_table(table_name)
            table_exists = False

        self._vector_store = LanceDBVectorStore(
            uri=str(self.raptor_path),
            table_name=table_name,
            mode="overwrite" if not table_exists else "append"
        )

        return self._vector_store

    def index_exists(self) -> bool:
        """Check if RAPTOR index already exists.

        RAPTOR persistence is handled by the LanceDB vector store, not file-based.
        We check if the LanceDB table exists and has data.

        Returns:
            True if LanceDB table exists with data
        """
        try:
            import lancedb

            if not self.raptor_path.exists():
                return False

            db = lancedb.connect(str(self.raptor_path))
            table_name = "raptor_embeddings"

            # Check if table exists and has data
            if table_name not in db.table_names():
                return False

            table = db.open_table(table_name)
            return len(table) > 0

        except Exception as e:
            logger.warning(f"Error checking RAPTOR index: {e}")
            return False

    def build_index(
        self,
        documents: List[Document],
        force_rebuild: bool = False,
        show_progress: bool = True
    ) -> bool:
        """Build RAPTOR hierarchical index from documents.

        This creates the recursive tree structure with clustered summaries.

        Args:
            documents: List of LlamaIndex Document objects
            force_rebuild: If True, rebuild even if index exists
            show_progress: Show progress bar during indexing

        Returns:
            True if index was built successfully
        """
        try:
            from llama_index.packs.raptor import RaptorPack
        except ImportError:
            logger.error(
                "llama-index-packs-raptor not installed. "
                "Install with: pip install llama-index-packs-raptor"
            )
            raise ImportError(
                "RAPTOR pack not available. Install with: "
                "pip install llama-index-packs-raptor"
            )

        if not documents:
            logger.warning("No documents provided for RAPTOR indexing")
            return False

        # Check existing index
        if not force_rebuild and self.index_exists():
            logger.info("RAPTOR index already exists. Use force_rebuild=True to recreate.")
            return self._load_retriever()

        logger.info(f"Building RAPTOR index from {len(documents)} documents...")

        # Strip large metadata and add required RAPTOR fields for schema stability
        # LanceDB requires consistent schema - RAPTOR adds 'level' and 'parent_id' during
        # hierarchical clustering, so we must include them from the start
        stripped_docs = []
        for doc in documents:
            essential_metadata = {
                'title': doc.metadata.get('title', 'Unknown'),
                'file_path': doc.metadata.get('file_path', ''),
                # Required for LanceDB schema stability - RAPTOR adds these during clustering
                'level': 0,
                'parent_id': '',
            }
            # Create new document with stripped metadata
            from llama_index.core.schema import Document
            stripped_doc = Document(
                text=doc.text,
                metadata=essential_metadata
            )
            stripped_docs.append(stripped_doc)

        logger.info(f"Stripped metadata from {len(stripped_docs)} documents for RAPTOR")

        # Get vector store (overwrite mode for fresh build)
        vector_store = self._get_vector_store(mode="overwrite")

        # Configure transformations for initial chunking
        transformations = [
            SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        ]

        try:
            # Create RAPTOR pack - this does the hierarchical clustering
            # Persistence is automatic via LanceDB vector store
            self._pack = RaptorPack(
                stripped_docs,
                embed_model=self.embed_model,
                llm=self.llm,
                vector_store=vector_store,
                similarity_top_k=self.similarity_top_k,
                mode=self.default_mode,
                transformations=transformations
            )

            # Verify index was created in LanceDB
            import lancedb
            db = lancedb.connect(str(self.raptor_path))
            if "raptor_embeddings" in db.table_names():
                table = db.open_table("raptor_embeddings")
                logger.info(f"RAPTOR index built successfully! {len(table)} nodes in LanceDB")
            else:
                logger.warning("RAPTOR pack created but LanceDB table not found")

            return True

        except Exception as e:
            logger.error(f"Failed to build RAPTOR index: {e}", exc_info=True)
            raise RuntimeError(f"RAPTOR indexing failed: {e}") from e

    def _load_retriever(self) -> bool:
        """Load existing RAPTOR retriever by reconnecting to vector store.

        RAPTOR persistence is handled by the LanceDB vector store.
        To reload, we create a new RaptorRetriever with empty docs
        and the existing vector store.

        Returns:
            True if retriever was loaded successfully
        """
        try:
            from llama_index.packs.raptor import RaptorRetriever
        except ImportError:
            logger.error("llama-index-packs-raptor not installed")
            return False

        if not self.index_exists():
            logger.warning("No RAPTOR index found to load")
            return False

        try:
            # Get vector store in append mode (connects to existing table)
            vector_store = self._get_vector_store(mode="append")

            # Reconnect to existing index by passing empty docs + existing vector store
            # This is the documented way to reload a RAPTOR index with LanceDB
            self._retriever = RaptorRetriever(
                [],  # Empty docs - we're loading from existing vector store
                embed_model=self.embed_model,
                llm=self.llm,
                vector_store=vector_store,
                similarity_top_k=self.similarity_top_k,
                mode=self.default_mode
            )

            logger.info(f"RAPTOR retriever loaded from {self.raptor_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load RAPTOR retriever: {e}", exc_info=True)
            return False

    def load_index(self) -> bool:
        """Load existing RAPTOR index.

        Returns:
            True if index was loaded successfully
        """
        return self._load_retriever()

    def retrieve(
        self,
        query: str,
        mode: Optional[RaptorMode] = None,
        top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """Retrieve relevant nodes using RAPTOR.

        Args:
            query: Search query
            mode: Retrieval mode (defaults to self.default_mode)
                - "collapsed": Flat search across all levels
                - "tree_traversal": Hierarchical search through tree
            top_k: Number of results (defaults to self.similarity_top_k)

        Returns:
            List of NodeWithScore objects with relevance scores
        """
        # Ensure retriever is loaded
        if self._pack is None and self._retriever is None:
            if not self._load_retriever():
                raise RuntimeError(
                    "RAPTOR index not available. "
                    "Run build_index() first or ensure index exists."
                )

        mode = mode or self.default_mode

        try:
            if self._pack is not None:
                # Use pack's run method if we built the index this session
                nodes = self._pack.run(query, mode=mode)
            else:
                # Use retriever for loaded index
                # RaptorRetriever.retrieve() returns List[NodeWithScore]
                nodes = self._retriever.retrieve(query)

            # Apply top_k limit if specified
            if top_k and len(nodes) > top_k:
                nodes = nodes[:top_k]

            logger.info(
                f"RAPTOR retrieved {len(nodes)} nodes "
                f"(mode={mode}, query='{query[:50]}...')"
            )

            return nodes

        except Exception as e:
            logger.error(f"RAPTOR retrieval failed: {e}", exc_info=True)
            raise RuntimeError(f"RAPTOR retrieval failed: {e}") from e

    def get_summary_stats(self) -> dict:
        """Get statistics about the RAPTOR index.

        Returns:
            Dictionary with index statistics
        """
        import lancedb

        stats = {
            "exists": False,
            "node_count": 0,
            "path": str(self.raptor_path),
            "default_mode": self.default_mode,
            "similarity_top_k": self.similarity_top_k
        }

        if not self.index_exists():
            return stats

        try:
            db = lancedb.connect(str(self.raptor_path))
            table = db.open_table("raptor_embeddings")
            stats["exists"] = True
            stats["node_count"] = len(table)
        except Exception as e:
            logger.warning(f"Error getting RAPTOR stats: {e}")

        return stats


def create_raptor_index(
    documents: List[Document],
    embed_model: BaseEmbedding,
    llm: LLM,
    raptor_path: Path,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    similarity_top_k: int = 10,
    mode: RaptorMode = "collapsed",
    force_rebuild: bool = False
) -> RaptorIndexManager:
    """Convenience function to create and build a RAPTOR index.

    Args:
        documents: Documents to index
        embed_model: Embedding model
        llm: LLM for summarization
        raptor_path: Path for index storage
        chunk_size: Initial chunk size
        chunk_overlap: Chunk overlap
        similarity_top_k: Number of results to retrieve
        mode: Default retrieval mode
        force_rebuild: Force rebuild if index exists

    Returns:
        Initialized RaptorIndexManager with built index
    """
    manager = RaptorIndexManager(
        embed_model=embed_model,
        llm=llm,
        raptor_path=raptor_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_top_k=similarity_top_k,
        default_mode=mode
    )

    manager.build_index(documents, force_rebuild=force_rebuild)

    return manager


def load_raptor_index(
    embed_model: BaseEmbedding,
    llm: LLM,
    raptor_path: Path,
    similarity_top_k: int = 10,
    mode: RaptorMode = "collapsed"
) -> Optional[RaptorIndexManager]:
    """Load an existing RAPTOR index.

    Args:
        embed_model: Embedding model
        llm: LLM for retrieval
        raptor_path: Path to existing index
        similarity_top_k: Number of results to retrieve
        mode: Default retrieval mode

    Returns:
        RaptorIndexManager if index exists, None otherwise
    """
    manager = RaptorIndexManager(
        embed_model=embed_model,
        llm=llm,
        raptor_path=raptor_path,
        similarity_top_k=similarity_top_k,
        default_mode=mode
    )

    if manager.load_index():
        return manager

    return None
