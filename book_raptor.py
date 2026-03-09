"""Book-Summary RAPTOR — 2-stage hierarchical retrieval for books.

Stage 1: Query RAPTOR index built from book summaries/descriptions
         to identify the top N relevant books (by book_uid).
Stage 2: Use BookFilter(book_uids=[...]) with get_books_retriever()
         to retrieve detailed chunks from only those books.

This avoids topic collision across ~800 books by first narrowing to
relevant books, then doing detailed retrieval within them.

RAPTOR metadata stripping: The standard RAPTOR build_index() strips metadata
to {title, file_path, level, parent_id}. We encode book_uid into the
file_path field so it survives the stripping process.
"""

import logging
from pathlib import Path
from typing import Optional

from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM

from raptor_index import RaptorIndexManager, RaptorMode
from models import BookFilter, QueryResult
from calibre_metadata import BookMetadataCache

logger = logging.getLogger(__name__)

BOOKS_SUMMARY_RAPTOR_PATH = Path("data/raptor/books_summary")


class BookRaptorManager:
    """2-stage RAPTOR retrieval for books.

    Stage 1: RAPTOR query on book summaries -> list of book_uids
    Stage 2: BookFilter(book_uids=[...]) -> detailed chunk retrieval
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
        llm: LLM,
        raptor_path: Path | None = None,
        similarity_top_k: int = 10,
        mode: RaptorMode = "collapsed",
    ):
        self.raptor = RaptorIndexManager(
            embed_model=embed_model,
            llm=llm,
            raptor_path=raptor_path or BOOKS_SUMMARY_RAPTOR_PATH,
            chunk_size=1024,
            chunk_overlap=100,
            similarity_top_k=similarity_top_k,
            default_mode=mode,
        )

    def build_summary_index(
        self,
        metadata_cache: BookMetadataCache,
        force_rebuild: bool = False,
    ) -> bool:
        """Build RAPTOR from book summaries.

        For each cached book, creates a Document with:
        - text = Calibre description (or title + author as fallback)
        - metadata.title = book_title
        - metadata.file_path = book_uid (encoded for RAPTOR stripping survival)

        Args:
            metadata_cache: Populated BookMetadataCache
            force_rebuild: Force rebuild even if index exists

        Returns:
            True if index was built successfully
        """
        if not force_rebuild and self.raptor.index_exists():
            logger.info("Book-summary RAPTOR index already exists")
            return self.raptor.load_index()

        documents = []
        for key, entry in metadata_cache.items():
            title = entry.get("title", "Unknown")
            author = entry.get("author", "")
            description = entry.get("description", "")
            categories = entry.get("categories", [])

            # Build summary text
            parts = [f"Title: {title}"]
            if author:
                parts.append(f"Author: {author}")
            if categories:
                parts.append(f"Categories: {', '.join(categories)}")
            if description:
                parts.append(f"Description: {description}")

            text = "\n".join(parts)

            # Minimal text fallback
            if len(text) < 20:
                text = f"{title} by {author}" if author else title

            # Encode book_uid into file_path — RAPTOR strips to
            # {title, file_path, level, parent_id} so this survives
            book_uid = key  # Cache key IS the book_uid (calibre:N or hash:xxx)

            documents.append(Document(
                text=text,
                metadata={
                    "title": title,
                    "file_path": book_uid,  # book_uid encoded here
                },
            ))

        if not documents:
            logger.warning("No books in metadata cache for RAPTOR summary index")
            return False

        logger.info("Building book-summary RAPTOR from %d book summaries", len(documents))
        return self.raptor.build_index(documents, force_rebuild=True)

    def index_exists(self) -> bool:
        return self.raptor.index_exists()

    def load_index(self) -> bool:
        return self.raptor.load_index()

    def retrieve_book_uids(self, query: str, top_n_books: int = 5) -> list[str]:
        """Stage 1: RAPTOR query -> list of book_uids.

        Args:
            query: Search query
            top_n_books: Maximum number of books to return

        Returns:
            List of book_uid strings (e.g. ["calibre:42", "hash:abc123"])
        """
        nodes = self.raptor.retrieve(query, top_k=top_n_books)

        book_uids = []
        seen = set()
        for node in nodes:
            inner = node.node if hasattr(node, "node") else node
            uid = inner.metadata.get("file_path", "")
            if uid and uid not in seen:
                seen.add(uid)
                book_uids.append(uid)

        logger.info(
            "Stage 1 RAPTOR: query='%s' -> %d books: %s",
            query[:50], len(book_uids), book_uids,
        )
        return book_uids

    def two_stage_query(
        self,
        query: str,
        books_index,
        config,
        top_n_books: int = 5,
    ) -> QueryResult:
        """Full 2-stage query: RAPTOR -> filtered chunk retrieval.

        Stage 1: RAPTOR identifies top N books by uid.
        Stage 2: BookFilter(book_uids=[...]) on the books vector index.

        Args:
            query: Search query
            books_index: VectorStoreIndex for books chunks
            config: RAGConfig
            top_n_books: Number of books from Stage 1

        Returns:
            QueryResult from filtered retrieval
        """
        import time
        from books_retriever import get_books_retriever

        start_time = time.time()

        # Stage 1: Get relevant book uids
        book_uids = self.retrieve_book_uids(query, top_n_books=top_n_books)

        if not book_uids:
            return QueryResult(
                answer="No relevant books found in RAPTOR summary index.",
                sources=[],
                tokens_used=0,
                exec_time=time.time() - start_time,
                query=query,
                mode="raptor-books",
            )

        # Stage 2: Filter retrieval by book_uids
        book_filter = BookFilter(book_uids=book_uids)
        retriever = get_books_retriever(
            books_index, config.retrieval.top_k, book_filter,
        )

        nodes = retriever.retrieve(query)

        # Synthesize answer
        from retrieval import get_llm, extract_sources
        from llama_index.core.response_synthesizers import get_response_synthesizer

        llm = get_llm(config)
        synth = get_response_synthesizer(response_mode="compact")
        response = synth.synthesize(query, nodes)
        answer = str(response) if response else "No relevant information found."
        sources = extract_sources(nodes)

        return QueryResult(
            answer=answer,
            sources=sources,
            tokens_used=0,
            exec_time=time.time() - start_time,
            query=query,
            mode="raptor-books",
        )

    def get_stats(self) -> dict:
        return self.raptor.get_summary_stats()
