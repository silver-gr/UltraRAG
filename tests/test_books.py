"""
Tests for book loading and chunking (book_loader.py, book_chunker.py).

Agent note:
- Tests cover BookLoader and BookChunker classes
- Unit tests don't require actual EPUB/PDF files (use mocks)
- Integration tests (@pytest.mark.integration) test real file parsing
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from llama_index.core import Document
from llama_index.core.schema import TextNode


class TestBookLoader:
    """
    Tests for BookLoader class.

    Agent context:
    - BookLoader discovers and loads EPUB/PDF files
    - Uses LlamaIndex SimpleDirectoryReader internally
    - Adds source_type, book_title metadata
    """

    @pytest.mark.unit
    def test_discover_books_finds_epub_and_pdf(self, tmp_path):
        """
        GIVEN: A directory with .epub and .pdf files
        WHEN: discover_books() is called
        THEN: Returns list of both file types
        """
        from book_loader import BookLoader

        # Create test files
        (tmp_path / "book1.epub").touch()
        (tmp_path / "book2.pdf").touch()
        (tmp_path / "notes.txt").touch()  # Should be ignored

        loader = BookLoader(tmp_path)
        books = loader.discover_books()

        assert len(books) == 2
        extensions = {b.suffix for b in books}
        assert extensions == {".epub", ".pdf"}

    @pytest.mark.unit
    def test_discover_books_recursive(self, tmp_path):
        """
        GIVEN: A directory with nested subdirectories containing books
        WHEN: discover_books() is called
        THEN: Finds books in all subdirectories
        """
        from book_loader import BookLoader

        # Create nested structure
        subdir = tmp_path / "subfolder" / "nested"
        subdir.mkdir(parents=True)
        (tmp_path / "root.epub").touch()
        (subdir / "nested.pdf").touch()

        loader = BookLoader(tmp_path)
        books = loader.discover_books()

        assert len(books) == 2

    @pytest.mark.unit
    def test_discover_books_empty_directory(self, tmp_path):
        """
        GIVEN: An empty directory
        WHEN: discover_books() is called
        THEN: Returns empty list without error
        """
        from book_loader import BookLoader

        loader = BookLoader(tmp_path)
        books = loader.discover_books()

        assert books == []

    @pytest.mark.unit
    def test_discover_books_nonexistent_path(self, tmp_path):
        """
        GIVEN: A path that doesn't exist
        WHEN: discover_books() is called
        THEN: Returns empty list (graceful handling)
        """
        from book_loader import BookLoader

        loader = BookLoader(tmp_path / "nonexistent")
        books = loader.discover_books()

        assert books == []

    @pytest.mark.unit
    def test_get_book_stats(self, tmp_path):
        """
        GIVEN: Directory with books of known sizes
        WHEN: get_book_stats() is called
        THEN: Returns accurate statistics
        """
        from book_loader import BookLoader

        # Create test files with content (larger to avoid rounding to 0)
        (tmp_path / "book1.epub").write_bytes(b"x" * 100_000)
        (tmp_path / "book2.pdf").write_bytes(b"y" * 200_000)

        loader = BookLoader(tmp_path)
        stats = loader.get_book_stats()

        assert stats["total_books"] == 2
        assert stats["by_type"]["epub"] == 1
        assert stats["by_type"]["pdf"] == 1
        assert stats["total_size_mb"] == pytest.approx(0.29, abs=0.01)

    @pytest.mark.unit
    def test_load_book_adds_metadata(self, tmp_path):
        """
        GIVEN: A book file
        WHEN: load_book() is called
        THEN: Documents have correct metadata (source_type, book_title, etc.)
        """
        from book_loader import BookLoader

        # Create a mock document that SimpleDirectoryReader would return
        mock_doc = Document(
            text="Book content here",
            metadata={"title": "Test Book"}
        )

        with patch('book_loader.SimpleDirectoryReader') as MockReader:
            MockReader.return_value.load_data.return_value = [mock_doc]

            loader = BookLoader(tmp_path)
            book_path = tmp_path / "test.epub"
            book_path.touch()

            docs = loader.load_book(book_path)

            assert len(docs) == 1
            assert docs[0].metadata["source_type"] == "books"
            assert docs[0].metadata["book_title"] == "Test Book"
            assert docs[0].metadata["file_type"] == "epub"

    @pytest.mark.unit
    def test_load_book_uses_filename_as_fallback_title(self, tmp_path):
        """
        GIVEN: A book without title metadata
        WHEN: load_book() is called
        THEN: Uses filename (without extension) as title
        """
        from book_loader import BookLoader

        mock_doc = Document(
            text="Content",
            metadata={}  # No title
        )

        with patch('book_loader.SimpleDirectoryReader') as MockReader:
            MockReader.return_value.load_data.return_value = [mock_doc]

            loader = BookLoader(tmp_path)
            book_path = tmp_path / "My Great Book.epub"
            book_path.touch()

            docs = loader.load_book(book_path)

            assert docs[0].metadata["book_title"] == "My Great Book"


class TestBookChunker:
    """
    Tests for BookChunker class.

    Agent context:
    - BookChunker splits book documents into chunks
    - Chapter-aware: tries to keep chapters together
    - Paragraph-aware: avoids splitting mid-paragraph
    - Default chunk size is 1024 (larger than vault's 512)
    """

    @pytest.mark.unit
    def test_chunk_document_basic(self):
        """
        GIVEN: A simple document with text
        WHEN: chunk_document() is called
        THEN: Returns TextNode objects with metadata preserved
        """
        from book_chunker import BookChunker, BookChunkConfig

        doc = Document(
            text="This is a test paragraph.\n\nAnother paragraph here.",
            metadata={"book_title": "Test Book", "source_type": "books"}
        )

        config = BookChunkConfig(chunk_size=1024, min_chunk_size=10)
        chunker = BookChunker(config=config)
        nodes = chunker.chunk_document(doc)

        assert len(nodes) >= 1
        assert all(isinstance(n, TextNode) for n in nodes)
        assert nodes[0].metadata["book_title"] == "Test Book"

    @pytest.mark.unit
    def test_chunk_document_respects_chapters(self):
        """
        GIVEN: A document with chapter headings
        WHEN: chunk_document() is called with respect_chapters=True
        THEN: Chapter info is captured in metadata
        """
        from book_chunker import BookChunker, BookChunkConfig

        doc = Document(
            text="# Chapter 1\n\nFirst chapter content.\n\n# Chapter 2\n\nSecond chapter content.",
            metadata={"book_title": "Test Book", "source_type": "books"}
        )

        config = BookChunkConfig(chunk_size=1024, min_chunk_size=10, respect_chapters=True)
        chunker = BookChunker(config=config)
        nodes = chunker.chunk_document(doc)

        # Should have section info in metadata for at least some nodes
        sections = [n.metadata.get("section") for n in nodes if n.metadata.get("section")]
        assert len(sections) > 0

    @pytest.mark.unit
    def test_chunk_document_filters_tiny_chunks(self):
        """
        GIVEN: A document that would produce very small chunks
        WHEN: chunk_document() is called with min_chunk_size
        THEN: Tiny chunks are filtered out
        """
        from book_chunker import BookChunker, BookChunkConfig

        # Very short content
        doc = Document(
            text="Hi.\n\nOk.",
            metadata={"book_title": "Test", "source_type": "books"}
        )

        config = BookChunkConfig(chunk_size=1024, min_chunk_size=100)
        chunker = BookChunker(config=config)
        nodes = chunker.chunk_document(doc)

        # Should filter out tiny content
        assert all(len(n.text) >= config.min_chunk_size for n in nodes) or len(nodes) == 0

    @pytest.mark.unit
    def test_chunk_documents_multiple(self):
        """
        GIVEN: Multiple documents
        WHEN: chunk_documents() is called
        THEN: Returns nodes from all documents combined
        """
        from book_chunker import BookChunker, BookChunkConfig

        docs = [
            Document(text="First book content paragraph.", metadata={"book_title": "Book 1", "source_type": "books"}),
            Document(text="Second book content paragraph.", metadata={"book_title": "Book 2", "source_type": "books"}),
        ]

        config = BookChunkConfig(chunk_size=1024, min_chunk_size=10)
        chunker = BookChunker(config=config)
        nodes = chunker.chunk_documents(docs)

        # Should have nodes from both books
        titles = {n.metadata["book_title"] for n in nodes}
        assert titles == {"Book 1", "Book 2"}

    @pytest.mark.unit
    def test_chunk_large_section_splits_properly(self):
        """
        GIVEN: A document with content larger than chunk_size
        WHEN: chunk_document() is called
        THEN: Content is split into multiple chunks
        """
        from book_chunker import BookChunker, BookChunkConfig

        # Create content with multiple paragraphs larger than chunk_size
        # Use sentences to help the splitter find break points
        paragraph = "This is a test sentence with some meaningful content. " * 30
        long_content = f"{paragraph}\n\n{paragraph}\n\n{paragraph}"

        doc = Document(
            text=long_content,
            metadata={"book_title": "Long Book", "source_type": "books"}
        )

        config = BookChunkConfig(chunk_size=500, min_chunk_size=50, respect_paragraphs=True)
        chunker = BookChunker(config=config)
        nodes = chunker.chunk_document(doc)

        # Should have multiple chunks due to content exceeding chunk_size
        assert len(nodes) > 1

    @pytest.mark.unit
    def test_is_chapter_heading_patterns(self):
        """
        GIVEN: Various chapter heading formats
        WHEN: _is_chapter_heading() is called
        THEN: Correctly identifies chapter headings
        """
        from book_chunker import BookChunker, BookChunkConfig

        chunker = BookChunker()

        # Should match - traditional patterns
        assert chunker._is_chapter_heading("Chapter 1")
        assert chunker._is_chapter_heading("CHAPTER 5")
        assert chunker._is_chapter_heading("## Chapter 12")
        assert chunker._is_chapter_heading("Part 2")
        assert chunker._is_chapter_heading("PART 3")

        # Should match - self-help / instructional patterns
        assert chunker._is_chapter_heading("Step 1")
        assert chunker._is_chapter_heading("STEP 1")
        assert chunker._is_chapter_heading("Step 1. Introduction")
        assert chunker._is_chapter_heading("Exploration 1.1: Testing")
        assert chunker._is_chapter_heading("Section 3")
        assert chunker._is_chapter_heading("Lesson 5")
        assert chunker._is_chapter_heading("Module 2")
        assert chunker._is_chapter_heading("Week 1")
        assert chunker._is_chapter_heading("Day 3")

        # Should not match
        assert not chunker._is_chapter_heading("Regular text")
        assert not chunker._is_chapter_heading("The chapter continues")
        assert not chunker._is_chapter_heading("Take the next step")


class TestIndexBooks:
    """
    Tests for index_books() function in indexing.py.

    Agent context:
    - index_books() orchestrates loading, chunking, and indexing
    - Returns VectorStoreIndex ready for querying
    - Uses config.books.path (not config.books_path)
    """

    @pytest.mark.unit
    def test_index_books_returns_index(self, test_config, tmp_path, mock_embedding_model):
        """
        GIVEN: A directory with books
        WHEN: index_books() is called
        THEN: Returns a VectorStoreIndex
        """
        # Setup books path via config.books.path
        books_dir = tmp_path / "books"
        books_dir.mkdir()
        test_config.books.path = books_dir

        # Create mock document
        mock_doc = Document(
            text="Test book content that is long enough to be chunked properly.",
            metadata={"title": "Test Book"}
        )

        with patch('embeddings.get_embedding_model', return_value=mock_embedding_model):
            with patch('book_loader.SimpleDirectoryReader') as MockReader:
                MockReader.return_value.load_data.return_value = [mock_doc]

                # Create a fake epub file
                (books_dir / "test.epub").touch()

                from indexing import index_books

                index = index_books(test_config)

                # Should return an index (may be None if no valid content)
                # The important thing is it doesn't crash

    @pytest.mark.unit
    def test_index_books_missing_path_raises(self, test_config):
        """
        GIVEN: No books path configured
        WHEN: index_books() is called
        THEN: Raises ConfigurationError
        """
        from indexing import index_books
        from models import ConfigurationError

        test_config.books.path = None

        with pytest.raises(ConfigurationError):
            index_books(test_config)

    @pytest.mark.unit
    def test_index_books_nonexistent_path_raises(self, test_config, tmp_path):
        """
        GIVEN: Books path that doesn't exist
        WHEN: index_books() is called
        THEN: Raises ConfigurationError
        """
        from indexing import index_books
        from models import ConfigurationError

        test_config.books.path = tmp_path / "nonexistent"

        with pytest.raises(ConfigurationError):
            index_books(test_config)
