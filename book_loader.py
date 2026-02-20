"""Book document loader for EPUB and PDF files."""
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.schema import TextNode
from tqdm import tqdm

from models import BookMetadata

logger = logging.getLogger(__name__)


class BookLoader:
    """Load and parse book documents (EPUB and PDF)."""

    SUPPORTED_EXTENSIONS = [".epub", ".pdf"]

    def __init__(self, books_path: Path):
        """Initialize book loader.

        Args:
            books_path: Path to books directory
        """
        self.books_path = books_path
        self._cache = None

    @staticmethod
    def _normalize_line(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _looks_like_page_number(line: str) -> bool:
        s = line.strip().lower()
        if not s:
            return False
        if re.fullmatch(r"(?:page\s*)?\d{1,5}", s):
            return True
        if re.fullmatch(r"\d{1,5}\s*/\s*\d{1,5}", s):
            return True
        return False

    def _merge_pdf_documents(self, documents: List[Document]) -> List[Document]:
        """Merge page-level PDF Documents into a single Document.

        SimpleDirectoryReader often returns one Document per page for PDFs.
        Chunking works better (chapter detection, paragraph recovery, overlap)
        when we feed the chunker a full-book stream instead of isolated pages.
        """
        if not documents:
            return []
        if len(documents) == 1:
            return documents

        page_texts: list[list[str]] = []
        # Track likely headers/footers by looking at top/bottom lines across pages.
        header_counts: dict[str, int] = {}
        footer_counts: dict[str, int] = {}

        for doc in documents:
            text = (doc.text or "").replace("\r\n", "\n").replace("\r", "\n")
            lines = [self._normalize_line(ln) for ln in text.split("\n")]
            lines = [ln for ln in lines if ln]
            page_texts.append(lines)

            top = [ln for ln in lines[:4] if not self._looks_like_page_number(ln)]
            bottom = [ln for ln in lines[-4:] if not self._looks_like_page_number(ln)]

            for ln in top:
                if 4 <= len(ln) <= 120:
                    header_counts[ln] = header_counts.get(ln, 0) + 1
            for ln in bottom:
                if 4 <= len(ln) <= 120:
                    footer_counts[ln] = footer_counts.get(ln, 0) + 1

        page_count = len(page_texts)
        # Remove repeated headers/footers that show up across many pages.
        threshold = max(3, int(page_count * 0.20))
        repeated_headers = {k for k, v in header_counts.items() if v >= threshold}
        repeated_footers = {k for k, v in footer_counts.items() if v >= threshold}

        cleaned_pages: list[str] = []
        for lines in page_texts:
            cleaned: list[str] = []
            for ln in lines:
                if self._looks_like_page_number(ln):
                    continue
                if ln in repeated_headers or ln in repeated_footers:
                    continue
                cleaned.append(ln)
            cleaned_pages.append("\n".join(cleaned).strip())

        merged_text = "\n\n".join([p for p in cleaned_pages if p])

        # Prefer metadata from the first page; drop page-specific keys if present.
        merged_meta = dict(documents[0].metadata or {})
        for k in ("page_label", "page_number", "page", "page_id", "page_index"):
            merged_meta.pop(k, None)
        merged_meta["pdf_page_count"] = page_count
        merged_doc = Document(text=merged_text, metadata=merged_meta)
        return [merged_doc]

    def set_enrichment_cache(self, cache):
        """Set optional enrichment cache for metadata lookup during loading."""
        self._cache = cache

    def discover_books(self) -> List[Path]:
        """Discover all supported book files in the directory.

        Returns:
            List of paths to book files
        """
        if not self.books_path.exists():
            logger.warning(f"Books path does not exist: {self.books_path}")
            return []

        books = []
        for ext in self.SUPPORTED_EXTENSIONS:
            books.extend(self.books_path.rglob(f"*{ext}"))

        logger.info(f"Discovered {len(books)} books in {self.books_path}")
        return sorted(books)

    def load_book(self, book_path: Path) -> List[Document]:
        """Load a single book file.

        Args:
            book_path: Path to book file

        Returns:
            List of Document objects
        """
        try:
            # Use SimpleDirectoryReader which has built-in EPUB and PDF support
            reader = SimpleDirectoryReader(
                input_files=[str(book_path)],
            )
            documents = reader.load_data()

            # Enhance metadata for each document
            file_type = book_path.suffix.lower().lstrip(".")
            title = self._extract_title(book_path, documents)

            # For PDFs, SimpleDirectoryReader commonly returns one doc per page.
            # Merge pages into a single book-level doc to improve downstream chunking.
            if file_type == "pdf" and documents:
                documents = self._merge_pdf_documents(documents)

            # Lookup enriched metadata if cache available
            enriched = self._cache.lookup(book_path) if self._cache else None

            # Compute hash-based uid fallback
            hash_uid = f"hash:{hashlib.sha256(f'{str(book_path)}:{book_path.stat().st_size}'.encode()).hexdigest()[:16]}"

            for doc in documents:
                doc.metadata.update({
                    "source_type": "books",
                    "book_title": enriched.title if enriched else title,
                    "book_author": enriched.author if enriched else "",
                    "book_uid": enriched.book_uid if enriched else hash_uid,
                    "book_categories": enriched.categories if enriched else [],
                    "book_language": enriched.language if enriched else "",
                    "book_description": (enriched.description[:500]) if enriched and enriched.description else "",
                    "metadata_source": enriched.metadata_source if enriched else "filename",
                    "file_path": str(book_path),
                    "file_type": file_type,
                    "file_name": book_path.name,
                })

            logger.info(f"Loaded {len(documents)} documents from: {book_path.name}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load book {book_path}: {e}")
            return []

    def _extract_title(self, book_path: Path, documents: List[Document]) -> str:
        """Extract book title from metadata or filename.

        Args:
            book_path: Path to book file
            documents: Loaded documents (may contain title in metadata)

        Returns:
            Book title
        """
        # Try to get title from document metadata
        if documents and documents[0].metadata:
            meta = documents[0].metadata
            if "title" in meta and meta["title"]:
                return meta["title"]

        # Fallback: clean up filename
        # "ADHD Explained - Edward Hallowell.epub" -> "ADHD Explained - Edward Hallowell"
        return book_path.stem

    def load_all_books(
        self,
        show_progress: bool = True
    ) -> List[Document]:
        """Load all books from the directory.

        Args:
            show_progress: Show progress bar

        Returns:
            List of all Document objects
        """
        books = self.discover_books()

        if not books:
            logger.warning(f"No books found in {self.books_path}")
            return []

        all_documents = []
        iterator = tqdm(books, desc="Loading books") if show_progress else books

        for book_path in iterator:
            docs = self.load_book(book_path)
            all_documents.extend(docs)

        logger.info(f"Loaded {len(all_documents)} total documents from {len(books)} books")
        return all_documents

    def get_book_stats(self) -> Dict[str, Any]:
        """Get statistics about books in the directory.

        Returns:
            Dictionary with book statistics
        """
        books = self.discover_books()

        stats = {
            "total_books": len(books),
            "by_type": {"epub": 0, "pdf": 0},
            "total_size_mb": 0,
            "books": []
        }

        for book_path in books:
            file_type = book_path.suffix.lower().lstrip(".")
            file_size = book_path.stat().st_size

            stats["by_type"][file_type] = stats["by_type"].get(file_type, 0) + 1
            stats["total_size_mb"] += file_size / (1024 * 1024)
            stats["books"].append({
                "title": book_path.stem,
                "type": file_type,
                "size_mb": round(file_size / (1024 * 1024), 2)
            })

        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats
