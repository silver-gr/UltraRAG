"""Document loader for Obsidian vault with wikilink and metadata support."""
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import frontmatter
from llama_index.core import Document
from llama_index.core.schema import TextNode
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ObsidianNote:
    """Represents an Obsidian note with metadata."""
    path: Path
    title: str
    content: str
    metadata: Dict
    wikilinks: List[str]
    tags: List[str]
    created_date: Optional[str] = None
    modified_date: Optional[str] = None


class ObsidianLoader:
    """Load and parse Obsidian vault documents."""
    
    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.wikilink_pattern = re.compile(r'\[\[([^\]]+)\]\]')
        # Support nested tags (#tag/subtag) and hyphens (#my-tag)
        self.tag_pattern = re.compile(r'#([\w/-]+)')
        
    def extract_wikilinks(self, content: str) -> List[str]:
        """Extract wikilink references from content."""
        matches = self.wikilink_pattern.findall(content)
        # Handle [[note|alias]] format
        return [link.split('|')[0].strip() for link in matches]
    
    def extract_tags(self, content: str, frontmatter_tags: List = None) -> List[str]:
        """Extract tags from content and frontmatter."""
        tags = set()
        
        # From frontmatter
        if frontmatter_tags:
            if isinstance(frontmatter_tags, list):
                tags.update(frontmatter_tags)
            elif isinstance(frontmatter_tags, str):
                tags.add(frontmatter_tags)
        
        # From content (inline tags)
        inline_tags = self.tag_pattern.findall(content)
        tags.update(inline_tags)
        
        return list(tags)
    
    def load_note(self, note_path: Path) -> Optional[ObsidianNote]:
        """Load a single Obsidian note."""
        try:
            # Security: Validate path is within vault directory (prevent path traversal)
            try:
                resolved_path = note_path.resolve()
                vault_resolved = self.vault_path.resolve()
                if not str(resolved_path).startswith(str(vault_resolved)):
                    logger.warning(f"Security: Skipping file outside vault: {note_path}")
                    return None
            except Exception as e:
                logger.error(f"Error resolving path {note_path}: {e}")
                return None

            # Load frontmatter (python-frontmatter uses safe YAML loading by default)
            with open(note_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)

            content = post.content
            metadata = dict(post.metadata)

            # Extract wikilinks and tags
            wikilinks = self.extract_wikilinks(content)
            tags = self.extract_tags(content, metadata.get('tags', []))

            # Get title from frontmatter or filename
            title = metadata.get('title', note_path.stem)

            return ObsidianNote(
                path=note_path,
                title=title,
                content=content,
                metadata=metadata,
                wikilinks=wikilinks,
                tags=tags,
                created_date=metadata.get('created', None),
                modified_date=metadata.get('modified', None)
            )
        except FileNotFoundError:
            logger.error(f"File not found: {note_path}")
            return None
        except PermissionError:
            logger.error(f"Permission denied reading file: {note_path}")
            return None
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in {note_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {note_path}: {e}", exc_info=True)
            return None
    
    def load_vault(self, extensions: List[str] = ['.md']) -> List[ObsidianNote]:
        """Load all notes from the vault."""
        logger.info(f"Loading vault from: {self.vault_path}")
        notes = []

        # Validate vault path exists
        if not self.vault_path.exists():
            logger.error(f"Vault path does not exist: {self.vault_path}")
            raise FileNotFoundError(
                f"Vault path not found: {self.vault_path}. "
                "Please check your .env file and ensure VAULT_PATH is correct."
            )

        if not self.vault_path.is_dir():
            logger.error(f"Vault path is not a directory: {self.vault_path}")
            raise NotADirectoryError(
                f"Vault path is not a directory: {self.vault_path}"
            )

        # Find all markdown files
        try:
            markdown_files = []
            for ext in extensions:
                markdown_files.extend(self.vault_path.rglob(f'*{ext}'))

            logger.info(f"Found {len(markdown_files)} notes in vault")

            for note_path in tqdm(markdown_files, desc="Loading notes"):
                note = self.load_note(note_path)
                if note:
                    notes.append(note)

            logger.info(f"Successfully loaded {len(notes)} notes")
            return notes

        except PermissionError as e:
            logger.error(f"Permission denied accessing vault: {e}")
            raise PermissionError(
                f"Cannot access vault directory {self.vault_path}. "
                "Please check file permissions."
            ) from e
        except Exception as e:
            logger.error(f"Error loading vault: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load vault: {e}") from e
    
    def notes_to_documents(self, notes: List[ObsidianNote]) -> List[Document]:
        """Convert ObsidianNote objects to LlamaIndex Documents."""
        documents = []

        for note in notes:
            # Create rich metadata for each document
            # IMPORTANT: LanceDB requires flat metadata (str, int, float, None only)
            # Convert lists to comma-separated strings, dates to ISO strings
            metadata = {
                'file_path': str(note.path),
                'file_name': note.path.name,
                'title': note.title,
                'tags': ', '.join(note.tags) if note.tags else '',  # Flatten list to string
                'wikilinks': ', '.join(note.wikilinks) if note.wikilinks else '',  # Flatten list to string
                'num_wikilinks': len(note.wikilinks),
                'created_date': str(note.created_date) if note.created_date else None,  # Convert date to string
                'modified_date': str(note.modified_date) if note.modified_date else None,  # Convert date to string
            }

            # Add custom frontmatter fields (flatten any non-primitive types)
            for key, value in note.metadata.items():
                if isinstance(value, (list, tuple)):
                    metadata[key] = ', '.join(str(v) for v in value)
                elif isinstance(value, dict):
                    metadata[key] = str(value)  # Convert dict to string
                elif hasattr(value, 'isoformat'):  # datetime/date objects
                    metadata[key] = value.isoformat()
                else:
                    metadata[key] = value

            # Create document with content and metadata
            doc = Document(
                text=note.content,
                metadata=metadata,
                id_=str(note.path)
            )
            documents.append(doc)
        
        return documents
    
    def build_wikilink_graph(self, notes: List[ObsidianNote]) -> Dict[str, List[str]]:
        """Build a graph of wikilink connections."""
        # Create mapping of note titles to paths
        title_to_path = {}
        for note in notes:
            title_to_path[note.title] = str(note.path)
            # Also map filename without extension
            title_to_path[note.path.stem] = str(note.path)
        
        # Build adjacency list
        graph = {}
        for note in notes:
            note_id = str(note.path)
            connections = []
            
            for wikilink in note.wikilinks:
                # Try to resolve wikilink to actual note
                if wikilink in title_to_path:
                    connections.append(title_to_path[wikilink])
            
            graph[note_id] = connections
        
        return graph
