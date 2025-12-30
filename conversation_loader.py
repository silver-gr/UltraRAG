"""Document loader for AI conversation exports (ChatGPT, Claude, Gemini)."""
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
import json

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


@dataclass
class Conversation:
    """Represents a full AI conversation."""
    path: Path
    title: str
    source: str  # CHATGPT, CLAUDE, GEMINI
    date: Optional[str]
    turns: List[ConversationTurn]
    total_messages: int
    total_characters: int
    metadata: Dict


class ConversationLoader:
    """Load and parse AI conversation exports from various platforms."""

    # Pattern to match conversation metadata header
    METADATA_PATTERN = re.compile(
        r'^## Metadata\s*\n'
        r'(?P<metadata>.*?)'
        r'^---',
        re.MULTILINE | re.DOTALL
    )

    # Pattern to match individual turns
    TURN_PATTERN = re.compile(
        r'^### (?P<role>USER|ASSISTANT|Human|Assistant)\s*(?:\((?P<time>[^)]+)\))?\s*\n'
        r'(?P<content>.*?)(?=^### (?:USER|ASSISTANT|Human|Assistant)|^---|\Z)',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )

    def __init__(self, conversations_path: Path):
        """
        Initialize conversation loader.

        Args:
            conversations_path: Path to the AI conversations export directory
        """
        self.conversations_path = conversations_path

    def _parse_metadata(self, content: str) -> Dict:
        """Extract metadata from conversation header."""
        metadata = {}

        # Look for metadata section
        lines = content.split('\n')
        in_metadata = False

        for line in lines:
            if line.strip() == '## Metadata':
                in_metadata = True
                continue
            if line.strip() == '---' and in_metadata:
                break
            if in_metadata and line.strip().startswith('- **'):
                # Parse "- **Key**: Value" format
                match = re.match(r'- \*\*([^*]+)\*\*:\s*(.+)', line)
                if match:
                    key = match.group(1).lower().replace(' ', '_')
                    value = match.group(2).strip()
                    metadata[key] = value

        return metadata

    def _parse_turns(self, content: str) -> List[ConversationTurn]:
        """Extract conversation turns from content."""
        turns = []

        # Find all turns
        matches = self.TURN_PATTERN.finditer(content)

        for match in matches:
            role_raw = match.group('role').upper()
            role = 'user' if role_raw in ('USER', 'HUMAN') else 'assistant'
            timestamp = match.group('time')
            turn_content = match.group('content').strip()

            # Clean up the content - remove trailing separators
            turn_content = re.sub(r'\n---\s*$', '', turn_content).strip()

            if turn_content:  # Only add non-empty turns
                turns.append(ConversationTurn(
                    role=role,
                    content=turn_content,
                    timestamp=timestamp
                ))

        return turns

    def load_conversation(self, conv_path: Path) -> Optional[Conversation]:
        """Load a single conversation file."""
        try:
            # Validate path is within conversations directory
            resolved_path = conv_path.resolve()
            base_resolved = self.conversations_path.resolve()
            if not str(resolved_path).startswith(str(base_resolved)):
                logger.warning(f"Security: Skipping file outside directory: {conv_path}")
                return None

            with open(conv_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Get title from first line (# Title)
            title_match = re.match(r'^# (.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else conv_path.stem

            # Parse metadata
            metadata = self._parse_metadata(content)

            # Determine source
            source = metadata.get('source', 'UNKNOWN').upper()
            if 'chatgpt' in conv_path.parts or 'chatgpt' in str(conv_path).lower():
                source = 'CHATGPT'
            elif 'claude' in conv_path.parts or 'claude' in str(conv_path).lower():
                source = 'CLAUDE'
            elif 'gemini' in conv_path.parts or 'gemini' in str(conv_path).lower():
                source = 'GEMINI'

            # Parse turns
            turns = self._parse_turns(content)

            # Extract date
            date = metadata.get('date', None)
            if not date:
                # Try to get from filename (YYYYMMDD_title.md format)
                date_match = re.match(r'^(\d{8})_', conv_path.name)
                if date_match:
                    date_str = date_match.group(1)
                    try:
                        date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
                    except ValueError:
                        pass

            # Parse numbers that may have commas (e.g., "16,647")
            def parse_int(val, default=0):
                if isinstance(val, int):
                    return val
                if isinstance(val, str):
                    return int(val.replace(',', ''))
                return default

            return Conversation(
                path=conv_path,
                title=title,
                source=source,
                date=date,
                turns=turns,
                total_messages=parse_int(metadata.get('messages'), len(turns)),
                total_characters=parse_int(metadata.get('total_characters'), sum(len(t.content) for t in turns)),
                metadata=metadata
            )

        except FileNotFoundError:
            logger.error(f"File not found: {conv_path}")
            return None
        except PermissionError:
            logger.error(f"Permission denied: {conv_path}")
            return None
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in {conv_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {conv_path}: {e}", exc_info=True)
            return None

    def load_all_conversations(self, recursive: bool = True) -> List[Conversation]:
        """Load all conversations from the directory."""
        logger.info(f"Loading conversations from: {self.conversations_path}")
        conversations = []

        if not self.conversations_path.exists():
            raise FileNotFoundError(f"Conversations path not found: {self.conversations_path}")

        if not self.conversations_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.conversations_path}")

        # Find all markdown files (skip INDEX.md files)
        pattern = '**/*.md' if recursive else '*.md'
        md_files = [
            f for f in self.conversations_path.glob(pattern)
            if not f.name.upper().startswith('INDEX')
            and 'conversations' in str(f).lower()  # Only files in "conversations" subdirs
        ]

        logger.info(f"Found {len(md_files)} conversation files")

        for conv_path in tqdm(md_files, desc="Loading conversations"):
            conv = self.load_conversation(conv_path)
            if conv and conv.turns:  # Only add if has actual turns
                conversations.append(conv)

        logger.info(f"Successfully loaded {len(conversations)} conversations")
        return conversations

    def conversations_to_documents(
        self,
        conversations: List[Conversation],
        include_full_context: bool = True
    ) -> List[Document]:
        """
        Convert conversations to LlamaIndex Documents.

        Args:
            conversations: List of Conversation objects
            include_full_context: If True, include full conversation context in each doc

        Returns:
            List of Document objects
        """
        documents = []

        for conv in conversations:
            # Build metadata (consistent schema for LanceDB)
            metadata = {
                'file_path': str(conv.path),
                'file_name': conv.path.name,
                'title': conv.title,
                'source': conv.source,
                'source_type': 'ai_conversation',  # Distinguish from vault notes
                'date': conv.date or '',
                'total_messages': conv.total_messages,
                'total_characters': conv.total_characters,
                'num_turns': len(conv.turns),
                'tags': '',  # No tags in conversations
                'wikilinks': '',  # No wikilinks in conversations
                'num_wikilinks': 0,
                'created_date': conv.date or '',
                'modified_date': '',
                'extra_metadata': json.dumps(conv.metadata) if conv.metadata else '{}'
            }

            if include_full_context:
                # Create single document with full conversation
                full_text = self._format_conversation(conv)
                doc = Document(
                    text=full_text,
                    metadata=metadata,
                    id_=str(conv.path)
                )
                documents.append(doc)
            else:
                # Create separate documents per turn (for finer-grained retrieval)
                for i, turn in enumerate(conv.turns):
                    turn_metadata = {
                        **metadata,
                        'turn_index': i,
                        'turn_role': turn.role,
                        'turn_timestamp': turn.timestamp or ''
                    }

                    # Add context from surrounding turns
                    context_before = ''
                    context_after = ''

                    if i > 0:
                        prev_turn = conv.turns[i-1]
                        context_before = f"[Previous {prev_turn.role}]: {prev_turn.content[:200]}..."

                    if i < len(conv.turns) - 1:
                        next_turn = conv.turns[i+1]
                        context_after = f"[Next {next_turn.role}]: {next_turn.content[:200]}..."

                    # Format turn with context
                    turn_text = f"# {conv.title}\n\n"
                    turn_text += f"Source: {conv.source} | Date: {conv.date}\n\n"
                    if context_before:
                        turn_text += f"{context_before}\n\n"
                    turn_text += f"**{turn.role.upper()}:**\n{turn.content}"
                    if context_after:
                        turn_text += f"\n\n{context_after}"

                    doc = Document(
                        text=turn_text,
                        metadata=turn_metadata,
                        id_=f"{conv.path}:turn_{i}"
                    )
                    documents.append(doc)

        return documents

    def _format_conversation(self, conv: Conversation) -> str:
        """Format a conversation as readable text."""
        lines = [
            f"# {conv.title}",
            f"",
            f"**Source:** {conv.source}",
            f"**Date:** {conv.date or 'Unknown'}",
            f"**Messages:** {conv.total_messages}",
            f"",
            "---",
            ""
        ]

        for turn in conv.turns:
            role_label = "USER" if turn.role == "user" else "ASSISTANT"
            timestamp = f" ({turn.timestamp})" if turn.timestamp else ""
            lines.append(f"### {role_label}{timestamp}")
            lines.append("")
            lines.append(turn.content)
            lines.append("")

        return '\n'.join(lines)


class ConversationChunker:
    """Chunking strategy optimized for AI conversation format."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 75,
        respect_turn_boundaries: bool = True,
        min_turn_size: int = 50
    ):
        """
        Initialize conversation chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            respect_turn_boundaries: Try to keep turns together
            min_turn_size: Minimum turn size to keep standalone
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_turn_boundaries = respect_turn_boundaries
        self.min_turn_size = min_turn_size

        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def _approximate_tokens(self, text: str) -> int:
        """Approximate token count."""
        return int(len(text.split()) * 1.3)

    def chunk_conversation(self, doc: Document) -> List[TextNode]:
        """
        Chunk a conversation document respecting turn boundaries.

        Args:
            doc: Document containing conversation

        Returns:
            List of TextNode objects
        """
        text = doc.text
        metadata = doc.metadata

        # If not respecting boundaries or small doc, use standard chunking
        if not self.respect_turn_boundaries or self._approximate_tokens(text) <= self.chunk_size:
            nodes = self.sentence_splitter.get_nodes_from_documents([doc])
            return nodes

        # Parse turns from formatted text
        turn_pattern = re.compile(
            r'^### (USER|ASSISTANT).*?\n\n(.*?)(?=^### (?:USER|ASSISTANT)|\Z)',
            re.MULTILINE | re.DOTALL
        )

        matches = list(turn_pattern.finditer(text))

        if not matches:
            # No turn structure found, fall back to standard chunking
            nodes = self.sentence_splitter.get_nodes_from_documents([doc])
            return nodes

        # Extract header (title, metadata)
        first_turn_start = matches[0].start() if matches else 0
        header = text[:first_turn_start].strip()

        nodes = []
        current_chunk = header + "\n\n" if header else ""
        current_tokens = self._approximate_tokens(current_chunk)

        for match in matches:
            role = match.group(1)
            content = match.group(2).strip()
            turn_text = f"### {role}\n\n{content}\n\n"
            turn_tokens = self._approximate_tokens(turn_text)

            # If turn alone exceeds chunk size, split it
            if turn_tokens > self.chunk_size:
                # First, save current chunk if not empty
                if current_chunk.strip() and current_tokens > self.min_turn_size:
                    node = TextNode(
                        text=current_chunk.strip(),
                        metadata={
                            **metadata,
                            'chunk_strategy': 'conversation_aware'
                        }
                    )
                    nodes.append(node)
                    current_chunk = ""
                    current_tokens = 0

                # Split large turn using sentence splitter
                turn_doc = Document(text=turn_text, metadata=metadata)
                turn_nodes = self.sentence_splitter.get_nodes_from_documents([turn_doc])
                for tn in turn_nodes:
                    tn.metadata['chunk_strategy'] = 'conversation_aware'
                nodes.extend(turn_nodes)
                continue

            # If adding this turn exceeds limit, save current and start new
            if current_tokens + turn_tokens > self.chunk_size:
                if current_chunk.strip():
                    node = TextNode(
                        text=current_chunk.strip(),
                        metadata={
                            **metadata,
                            'chunk_strategy': 'conversation_aware'
                        }
                    )
                    nodes.append(node)
                current_chunk = turn_text
                current_tokens = turn_tokens
            else:
                current_chunk += turn_text
                current_tokens += turn_tokens

        # Don't forget the last chunk
        if current_chunk.strip():
            node = TextNode(
                text=current_chunk.strip(),
                metadata={
                    **metadata,
                    'chunk_strategy': 'conversation_aware'
                }
            )
            nodes.append(node)

        return nodes

    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        """Chunk all conversation documents."""
        all_nodes = []

        for doc in documents:
            nodes = self.chunk_conversation(doc)
            all_nodes.extend(nodes)

        logger.info(f"Created {len(all_nodes)} chunks from {len(documents)} conversations")
        return all_nodes
