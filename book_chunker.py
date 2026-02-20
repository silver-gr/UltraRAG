"""Book-specific chunking strategies for EPUB and PDF documents."""
import logging
import re
from typing import List, Optional, Callable
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)


@dataclass
class BookChunkConfig:
    """Configuration for book chunking."""
    # NOTE: LlamaIndex SentenceSplitter interprets chunk_size/chunk_overlap as *tokens*.
    chunk_size: int = 1024  # tokens
    chunk_overlap: int = 128  # tokens
    min_chunk_size: int = 100  # characters (skip tiny chunks)
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

        self._tokenizer: Callable[[str], List] = self._get_tokenizer()

        # Fallback sentence splitter for large sections
        self._sentence_splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            tokenizer=self._tokenizer,
        )

    @staticmethod
    def _get_tokenizer() -> Callable[[str], List]:
        """Best-effort tokenizer compatible with LlamaIndex splitters."""
        try:
            from llama_index.core.utils import get_tokenizer
            return get_tokenizer()
        except Exception:
            # Fallback: approximate tokens by splitting on whitespace.
            return lambda s: (s or "").split()

    def _token_count(self, text: str) -> int:
        try:
            return len(self._tokenizer(text or ""))
        except Exception:
            return len((text or "").split())

    def _paragraphs_token_total(self, paragraphs: List[str]) -> int:
        """Return total token count across paragraph list."""
        return sum(self._token_count(p) for p in paragraphs)

    def _tail_overlap_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Select tail paragraphs up to configured overlap budget.

        Keeps paragraph boundaries intact. If the last paragraph alone exceeds
        overlap budget, no overlap is applied for that boundary.
        """
        if not paragraphs or self.config.chunk_overlap <= 0:
            return []

        overlap: list[str] = []
        tokens = 0

        for para in reversed(paragraphs):
            para_tokens = self._token_count(para)
            if para_tokens > self.config.chunk_overlap:
                break
            if tokens + para_tokens > self.config.chunk_overlap:
                break
            overlap.insert(0, para)
            tokens += para_tokens
            if tokens >= self.config.chunk_overlap:
                break

        return overlap

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

    @staticmethod
    def _looks_like_layout_noise(line: str) -> bool:
        """Heuristic filter for common PDF extraction artifacts."""
        s = line.strip()
        if not s:
            return True
        if len(s) <= 2:
            return True
        # Repeated page artifact tokens often seen in OCR/PDF output
        if re.search(r'\b(?:ffirs|ftoc|indd)\b', s, re.IGNORECASE):
            return True
        # Timestamp-like scanner artifacts
        if re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', s) and re.search(r'\b\d{1,2}:\d{2}', s):
            return True
        # Mostly numeric or punctuation lines
        alnum = sum(ch.isalnum() for ch in s)
        if alnum == 0:
            return True
        if alnum <= 3 and len(s) <= 8:
            return True
        return False

    def _preprocess_pdf_text(self, text: str) -> str:
        """Clean noisy PDF text and recover paragraph boundaries."""
        if not text:
            return ""

        # Normalize line endings and remove soft hyphenation
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        raw_lines = [ln.rstrip() for ln in text.split("\n")]
        lines = [ln.strip() for ln in raw_lines if not self._looks_like_layout_noise(ln)]
        if not lines:
            return ""

        paragraphs: list[str] = []
        current = ""
        for line in lines:
            # Keep explicit headings/section markers as paragraph boundaries
            heading_like = self._is_chapter_heading(line) or line.startswith("#")
            if heading_like:
                if current:
                    paragraphs.append(current.strip())
                    current = ""
                paragraphs.append(line.strip())
                continue

            if not current:
                current = line
                continue

            # Start a new paragraph on likely sentence boundary + title-case line
            starts_like_new_para = bool(re.match(r"^[A-Z][A-Za-z0-9\"'(\[]", line))
            prev_ends_sentence = bool(re.search(r"[.!?\"')\]]$", current))
            if prev_ends_sentence and starts_like_new_para:
                paragraphs.append(current.strip())
                current = line
            else:
                current += " " + line

        if current:
            paragraphs.append(current.strip())

        # Remove tiny fragments post-merge
        paragraphs = [p for p in paragraphs if len(p) >= 40]
        return "\n\n".join(paragraphs)

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are too small."""
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for chunk in chunks[1:]:
            combined_tokens = self._token_count(current) + self._token_count(chunk)

            # If combining keeps us under limit, merge
            if combined_tokens < self.config.chunk_size:
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
        if self._token_count(text) <= self.config.chunk_size:
            return [text]

        # Try paragraph-based splitting first
        if self.config.respect_paragraphs:
            paragraphs = self._split_into_paragraphs(text)

            chunks = []
            current_chunk = []
            current_tokens = 0

            for para in paragraphs:
                para_tokens = self._token_count(para)

                # If single paragraph is too large, use sentence splitter
                if para_tokens > self.config.chunk_size:
                    # Save current chunk first
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_tokens = 0

                    # Split the large paragraph
                    nodes = self._sentence_splitter.get_nodes_from_documents(
                        [Document(text=para)]
                    )
                    chunks.extend([n.text for n in nodes])
                    continue

                # Check if adding this paragraph exceeds limit
                if current_tokens + para_tokens > self.config.chunk_size and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    overlap_paras = self._tail_overlap_paragraphs(current_chunk)
                    # Ensure overlap + new paragraph still respects chunk budget.
                    while overlap_paras and (
                        self._paragraphs_token_total(overlap_paras) + para_tokens
                    ) > self.config.chunk_size:
                        overlap_paras.pop(0)

                    current_chunk = overlap_paras + [para]
                    current_tokens = para_tokens
                    if overlap_paras:
                        current_tokens += self._paragraphs_token_total(overlap_paras)
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens

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
        file_type = (metadata.get("file_type") or "").lower()

        # PDF extraction is often line-broken/noisy; normalize before chunking.
        if file_type == "pdf":
            text = self._preprocess_pdf_text(text)

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
