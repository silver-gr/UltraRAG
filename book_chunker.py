"""Book-specific chunking strategies for EPUB and PDF documents."""
import logging
import re
from typing import List, Optional
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)


@dataclass
class BookChunkConfig:
    """Configuration for book chunking."""
    chunk_size: int = 1024  # Larger chunks for books (more context)
    chunk_overlap: int = 128
    min_chunk_size: int = 100  # Skip tiny chunks
    respect_chapters: bool = True
    respect_paragraphs: bool = True


class BookChunker:
    """Smart chunking for book documents (EPUB/PDF).

    Key differences from ObsidianChunker:
    - Larger default chunk size (1024 vs 512) - books benefit from more context
    - Chapter-aware splitting - tries to keep chapter content together
    - Paragraph-aware - avoids splitting mid-paragraph
    - No wikilink handling (books don't have them)
    - Simpler metadata (no tags, frontmatter)
    """

    # Chapter heading patterns
    CHAPTER_PATTERNS = [
        r'^#+\s+Chapter\s+\d+',  # Markdown: ## Chapter 1
        r'^Chapter\s+\d+',  # Plain: Chapter 1
        r'^CHAPTER\s+\d+',  # Uppercase: CHAPTER 1
        r'^Part\s+\d+',  # Part 1
        r'^PART\s+\d+',  # PART 1
        r'^#+\s+\d+\.',  # Numbered: ## 1. Introduction
        # Self-help / instructional book patterns
        r'^Step\s+\d+',  # Step 1
        r'^STEP\s+\d+',  # STEP 1
        r'^Exploration\s+\d+',  # Exploration 1.1
        r'^Section\s+\d+',  # Section 1
        r'^Lesson\s+\d+',  # Lesson 1
        r'^Module\s+\d+',  # Module 1
        r'^Unit\s+\d+',  # Unit 1
        r'^Week\s+\d+',  # Week 1
        r'^Day\s+\d+',  # Day 1
    ]

    def __init__(self, config: Optional[BookChunkConfig] = None):
        """Initialize book chunker.

        Args:
            config: Chunking configuration (uses defaults if not provided)
        """
        self.config = config or BookChunkConfig()
        self._chapter_regex = re.compile(
            '|'.join(self.CHAPTER_PATTERNS),
            re.MULTILINE | re.IGNORECASE
        )

        # Fallback sentence splitter for large sections
        self._sentence_splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def _is_chapter_heading(self, line: str) -> bool:
        """Check if a line is a chapter heading."""
        return bool(self._chapter_regex.match(line.strip()))

    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into sections based on chapter/section headings.

        Returns list of sections, each starting with a heading (if present).
        """
        lines = text.split('\n')
        sections = []
        current_section = []

        for line in lines:
            # Check for chapter/section heading
            if self._is_chapter_heading(line) or (line.startswith('#') and len(line) < 100):
                # Save current section if not empty
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        # Don't forget the last section
        if current_section:
            sections.append('\n'.join(current_section))

        return sections

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs (double newline separated)."""
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)
        # Clean up and filter empty
        return [p.strip() for p in paragraphs if p.strip()]

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are too small."""
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for chunk in chunks[1:]:
            combined_len = len(current) + len(chunk)

            # If combining keeps us under limit, merge
            if combined_len < self.config.chunk_size:
                current = current + '\n\n' + chunk
            else:
                # Save current if it's big enough
                if len(current) >= self.config.min_chunk_size:
                    merged.append(current)
                current = chunk

        # Don't forget the last chunk
        if len(current) >= self.config.min_chunk_size:
            merged.append(current)

        return merged

    def _chunk_large_section(self, text: str) -> List[str]:
        """Break up sections that are too large using sentence splitting."""
        if len(text) <= self.config.chunk_size:
            return [text]

        # Try paragraph-based splitting first
        if self.config.respect_paragraphs:
            paragraphs = self._split_into_paragraphs(text)

            chunks = []
            current_chunk = []
            current_size = 0

            for para in paragraphs:
                para_size = len(para)

                # If single paragraph is too large, use sentence splitter
                if para_size > self.config.chunk_size:
                    # Save current chunk first
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0

                    # Split the large paragraph
                    nodes = self._sentence_splitter.get_nodes_from_documents(
                        [Document(text=para)]
                    )
                    chunks.extend([n.text for n in nodes])
                    continue

                # Check if adding this paragraph exceeds limit
                if current_size + para_size > self.config.chunk_size and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_size = para_size
                else:
                    current_chunk.append(para)
                    current_size += para_size

            # Don't forget last chunk
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))

            return chunks

        # Fallback to sentence splitter
        nodes = self._sentence_splitter.get_nodes_from_documents(
            [Document(text=text)]
        )
        return [n.text for n in nodes]

    def chunk_document(self, document: Document) -> List[TextNode]:
        """Chunk a single book document into nodes.

        Args:
            document: LlamaIndex Document from book

        Returns:
            List of TextNode objects
        """
        text = document.text
        metadata = document.metadata.copy()

        nodes = []

        # Step 1: Split into sections (chapter-aware)
        if self.config.respect_chapters:
            sections = self._split_into_sections(text)
        else:
            sections = [text]

        # Step 2: Process each section
        chunk_idx = 0
        for section_idx, section in enumerate(sections):
            # Extract section heading if present
            lines = section.split('\n')
            section_heading = None
            if lines and (lines[0].startswith('#') or self._is_chapter_heading(lines[0])):
                section_heading = lines[0].strip().lstrip('#').strip()

            # Step 3: Break up large sections
            chunks = self._chunk_large_section(section)
            chunks = self._merge_small_chunks(chunks)

            # Step 4: Create nodes
            for chunk in chunks:
                if len(chunk.strip()) < self.config.min_chunk_size:
                    continue

                # Build node metadata
                node_metadata = metadata.copy()
                if section_heading:
                    node_metadata['section'] = section_heading
                node_metadata['chunk_index'] = chunk_idx
                node_metadata['section_index'] = section_idx

                # Create node
                node = TextNode(
                    text=chunk,
                    metadata=node_metadata,
                )
                nodes.append(node)
                chunk_idx += 1

        logger.debug(f"Chunked document into {len(nodes)} nodes")
        return nodes

    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        """Chunk multiple book documents.

        Args:
            documents: List of Documents from books

        Returns:
            List of all TextNode objects
        """
        all_nodes = []

        for doc in documents:
            book_title = doc.metadata.get('book_title', 'Unknown')
            nodes = self.chunk_document(doc)
            all_nodes.extend(nodes)
            logger.info(f"Chunked '{book_title}' into {len(nodes)} nodes")

        logger.info(f"Total: {len(all_nodes)} nodes from {len(documents)} documents")
        return all_nodes
