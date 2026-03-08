"""Main RAG system orchestrator."""
import os

# Suppress tokenizers parallelism warning when forking processes
# Must be set before importing transformers/tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import sys
import json
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Set
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from gemini_cli import GeminiCLI
from tracked_llm import wrap_llm_with_tracking
from tqdm import tqdm

from config import load_config, RAGConfig
from loader import ObsidianLoader
from embeddings import get_embedding_model, get_reranker
from chunking import ObsidianChunker
from vector_store import get_vector_store, create_vector_index, load_vector_index, index_exists
from query_engine import RAGQueryEngine, HybridQueryEngine
from query_transform import QueryTransformer
from cache import EmbeddingCache
from token_tracker import get_tracker
from conversation_loader import ConversationLoader, ConversationChunker
from federated_query import FederatedQueryEngine, IndexSource
from temporal_filter import create_temporal_filter, DateFilterPreset
from raptor_index import RaptorIndexManager, RaptorMode

# New unified modules (AI-optimized)
import indexing as indexing_module
import retrieval as retrieval_module
from models import (
    QueryResult as ModelsQueryResult,
    ResearchResult as ModelsResearchResult,
    IndexNotFoundError,
    ConfigurationError,
    BookFilter,
    _normalize_category,
)

# Configure logging (with guard to prevent duplicate handlers in Streamlit)
def _setup_logging():
    """Setup logging once, avoiding duplicate handlers when imported multiple times."""
    root_logger = logging.getLogger()

    # Check if we already have our handlers (prevents Streamlit duplicate logs)
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root_logger.handlers
    )

    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add file handler if not present
    if not has_file_handler:
        file_handler = logging.FileHandler('ultrarag.log')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Add stream handler if not present (avoids duplication with Streamlit's handler)
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

_setup_logging()
logger = logging.getLogger(__name__)


# ============================================
# Research Method Dataclasses
# ============================================

@dataclass
class Source:
    """Represents a source in research results."""
    type: str  # "vault", "conversations", "web"
    path: Optional[str]  # file path for vault, None for web
    url: Optional[str]  # URL for web sources
    relevance: float
    snippet: str
    title: str


@dataclass
class Citation:
    """Formatted citation for inline use."""
    index: int
    text: str  # e.g., "[1] Title - vault" or "[2] Title - web"


@dataclass
class ResearchResult:
    """Result from research() method."""
    summary: str
    sources: List[Source]
    citations: List[Citation]
    vault_sources: int
    web_sources: int
    query: str


# ============================================
# Module-level synthesis helpers
# ============================================

def _build_synthesis_context(nodes, start_index=1):
    """Build numbered context string for synthesis prompt."""
    context_parts = []
    for i, node in enumerate(nodes, start_index):
        md = node.metadata
        source_type = md.get('source_type', 'vault')
        title = md.get('title') or md.get('book_title') or md.get('file_name', 'Unknown')
        file_path = md.get('file_path', '')
        if source_type == 'saved_items':
            display_path = md.get('display_label') or file_path
        elif source_type == 'books':
            author = md.get('book_author', '')
            display_path = f"{title} — {author}" if author else title
        else:
            display_path = file_path
        context_parts.append(
            f"[{i}] Source: {title}\n"
            f"File: {display_path} ({source_type})\n"
            f"Content:\n{node.node.text}\n"
        )
    return "\n---\n".join(context_parts)


def _build_chunk_prompt(nodes, query_str, chunk_idx=0, is_continuation=False):
    """Build synthesis prompt for a chunk. Handles batch notes and word count."""
    from query_engine import RESEARCH_TEMPLATE

    num_sources = len(nodes)
    numbered_context = _build_synthesis_context(nodes, start_index=1)

    research_prompt = RESEARCH_TEMPLATE.replace("{num_sources}", str(num_sources))
    research_prompt = research_prompt.replace("{context_str}", numbered_context)
    research_prompt = research_prompt.replace("{query_str}", query_str)

    # Batch note for continuations
    if is_continuation:
        batch_note = (
            "\n\n--- BATCH NOTE ---\n"
            "This is an additional batch covering more sources. "
            "Skip the executive summary. Focus on NEW insights from these sources only. "
            "Do NOT repeat themes covered in earlier batches. "
            "The user's formatting instructions above still apply.\n"
            "--- END BATCH NOTE ---\n"
        )
        research_prompt += batch_note

    # Word count enforcement for large chunks
    if num_sources >= 300:
        length_note = (
            f"\n\n--- LENGTH REQUIREMENT ---\n"
            f"With {num_sources} sources provided, you MUST generate a MINIMUM of 3,000 words. "
            f"Aim for 5,000-10,000 words to adequately cover all relevant information. "
            f"Do NOT stop early.\n"
            f"--- END LENGTH REQUIREMENT ---\n"
        )
        research_prompt += length_note

    return research_prompt


class UltraRAG:
    """Main RAG system for Obsidian vault."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG system."""
        logger.info("Initializing UltraRAG system...")
        print("Initializing UltraRAG system...")

        try:
            self.config = config or load_config()

            # Initialize caching
            self.embedding_cache = EmbeddingCache()
            logger.info("Embedding cache initialized")

            # Initialize Voyage AI token tracking
            self.token_tracker = get_tracker(
                embedding_limit=self.config.embedding.token_limit,
                rerank_limit=self.config.retrieval.reranker_token_limit
            )
            logger.info("Token usage tracking initialized")

            # Initialize components
            self._setup_llm()
            self._setup_embeddings()
            self._setup_vector_store()
            self._setup_query_transformer()

            self.index = None
            self.query_engine = None
            self.nodes = None  # Store nodes for BM25 retrieval
            self.bm25_retriever = None  # Store BM25 retriever
            self.loader = ObsidianLoader(self.config.vault_path)

            # Conversation index (federated retrieval)
            self.conversations_index = None
            self.conversations_nodes = None
            self.conversations_vector_store = None
            self.federated_engine = None

            # Books index (federated retrieval)
            self.books_index = None
            self.books_nodes = None
            self.books_vector_store = None

            # RAPTOR index (hierarchical summaries)
            self.raptor_manager = None
            if self.config.raptor.enabled:
                self._setup_raptor()

            # Web search retriever
            self.web_retriever = None
            if self.config.web_search.enabled:
                self._setup_web_retriever()

            # Saved items (TheSource) retriever
            self.saved_items_retriever = None

            logger.info("UltraRAG system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize UltraRAG system: {e}", exc_info=True)
            raise RuntimeError(f"System initialization failed: {e}") from e

    def _setup_llm(self):
        """Setup LLM for response generation."""
        backend = self.config.llm.backend
        logger.info(f"Setting up LLM: {self.config.llm.model} (backend: {backend})")
        print(f"Setting up LLM: {self.config.llm.model} (backend: {backend})")

        try:
            if backend == "cli":
                # Use Gemini CLI for separate free tier quota (1000/day)
                # No token tracking needed - CLI quota is free
                self.llm = GeminiCLI(
                    model=self.config.llm.model,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens
                )
                logger.info("Using Gemini CLI backend (free tier: 1000 requests/day, no cost tracking)")
            else:
                # Use Google Gemini API directly
                google_key = self.config.google_api_key.get_secret_value()
                if not google_key:
                    logger.error("Google API key not found")
                    raise ValueError(
                        "GOOGLE_API_KEY not found. Please set it in your .env file.\n"
                        "Get your API key from: https://makersuite.google.com/app/apikey"
                    )

                base_llm = GoogleGenAI(
                    model=self.config.llm.model,
                    api_key=google_key,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens
                )

                # Wrap with token tracking (only for paid API)
                self.llm = wrap_llm_with_tracking(base_llm, model_name=self.config.llm.model)
                logger.info(f"LLM token tracking enabled for {self.config.llm.model}")

            Settings.llm = self.llm
            logger.debug("LLM setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup LLM: {e}", exc_info=True)
            raise RuntimeError(f"LLM initialization failed: {e}") from e
    
    def _setup_embeddings(self):
        """Setup embedding model."""
        logger.info(f"Setting up embeddings: {self.config.embedding.model}")
        print(f"Setting up embeddings: {self.config.embedding.model}")

        try:
            self.embed_model = get_embedding_model(
                self.config.embedding,
                api_key=self.config.voyage_api_key
            )

            Settings.embed_model = self.embed_model
            logger.debug("Embedding model setup completed")

            # Setup reranker
            try:
                self.reranker = get_reranker(
                    model_name=self.config.retrieval.reranker_model,
                    api_key=self.config.voyage_api_key,
                    top_n=self.config.retrieval.rerank_top_n
                )
                logger.info(f"Reranker initialized: {self.config.retrieval.reranker_model}")
                print(f"Reranker initialized: {self.config.retrieval.reranker_model}")
            except Exception as e:
                logger.warning(f"Could not initialize reranker: {e}")
                print(f"Could not initialize reranker: {e}")
                self.reranker = None

        except Exception as e:
            logger.error(f"Failed to setup embeddings: {e}", exc_info=True)
            raise RuntimeError(f"Embedding model initialization failed: {e}") from e
    
    def _setup_vector_store(self, mode: str = None):
        """Setup vector database.

        Args:
            mode: Mode for vector store - "create", "append", or "overwrite".
                  If None, automatically determines based on whether index exists.
        """
        logger.info(f"Setting up vector store: {self.config.vector_db.db_type}")
        print(f"Setting up vector store: {self.config.vector_db.db_type}")

        try:
            # Auto-detect mode if not specified
            if mode is None:
                if index_exists(self.config.vector_db, table_name=self.config.vector_db.vault_table):
                    mode = "append"
                    logger.info("Existing index found, using append mode")
                else:
                    mode = "create"
                    logger.info("No existing index found, using create mode")

            self.vector_store = get_vector_store(
                self.config.vector_db,
                mode=mode,
                table_name=self.config.vector_db.vault_table
            )
            logger.debug("Vector store setup completed")
        except Exception as e:
            logger.error(f"Failed to setup vector store: {e}", exc_info=True)
            raise RuntimeError(f"Vector store initialization failed: {e}") from e

    def _setup_query_transformer(self):
        """Setup query transformer for HyDE and multi-query expansion."""
        logger.info("Setting up query transformer...")
        print("Setting up query transformer...")

        try:
            # Initialize query transformer with LLM
            self.query_transformer = QueryTransformer(
                llm=self.llm,
                embed_model=self.embed_model,
                hyde_temperature=self.config.retrieval.hyde_temperature
            )

            # Log the configuration
            method = self.config.retrieval.query_transform_method
            if method in ["none", "disabled"]:
                logger.info("Query transformation is disabled")
                print("Query transformation: Disabled")
            else:
                logger.info(f"Query transformation enabled: {method}")
                print(f"Query transformation: {method}")
                if method in ["multi_query", "both"]:
                    num_queries = self.config.retrieval.query_transform_num_queries
                    print(f"  - Number of query variations: {num_queries}")

        except Exception as e:
            logger.warning(f"Failed to setup query transformer: {e}")
            print(f"Warning: Could not initialize query transformer: {e}")
            self.query_transformer = None

    def _setup_web_retriever(self):
        """Setup web search retriever using Tavily API."""
        logger.info("Setting up web retriever...")
        print("Setting up web search retriever...")

        try:
            from web_retriever import WebRetriever

            self.web_retriever = WebRetriever(
                max_results=self.config.web_search.max_results
            )

            if self.web_retriever.is_available():
                logger.info("Web search retriever initialized successfully")
                print("Web search: Enabled (Tavily API)")
            else:
                logger.warning("Web search retriever disabled (TAVILY_API_KEY not found)")
                print("Web search: Disabled (TAVILY_API_KEY not found)")
                self.web_retriever = None

        except ImportError as e:
            logger.warning(f"Could not import web_retriever: {e}")
            print(f"Warning: Web retriever not available: {e}")
            self.web_retriever = None
        except Exception as e:
            logger.warning(f"Failed to setup web retriever: {e}")
            print(f"Warning: Could not initialize web retriever: {e}")
            self.web_retriever = None

    def _get_exclusion_patterns(self) -> list[dict]:
        """Get file exclusion patterns from settings.

        Returns:
            List of exclusion pattern dicts, or empty list if none configured
        """
        try:
            from settings_store import get_exclusions
            return get_exclusions(str(self.config.vector_db.lancedb_path))
        except Exception as e:
            logger.warning(f"Could not load exclusion patterns: {e}")
            return []

    def _get_checkpoint_file(self) -> Path:
        """Get path to checkpoint file.

        Returns:
            Path to the checkpoint file
        """
        checkpoint_dir = Path("./data")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / "index_checkpoint.json"

    def _load_checkpoint(self) -> Set[str]:
        """Load checkpoint of processed files.

        Returns:
            Set of file paths that have been processed
        """
        checkpoint_file = self._get_checkpoint_file()
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    processed_files = set(data.get('processed_files', []))
                    logger.info(f"Loaded checkpoint with {len(processed_files)} processed files")
                    return processed_files
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
                return set()
        return set()

    def _save_checkpoint(self, processed_files: Set[str]):
        """Save checkpoint of processed files.

        Args:
            processed_files: Set of file paths that have been processed
        """
        checkpoint_file = self._get_checkpoint_file()
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'processed_files': list(processed_files),
                    'total_files': len(processed_files)
                }, f, indent=2)
            logger.debug(f"Checkpoint saved with {len(processed_files)} files")
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")

    def _clear_checkpoint(self):
        """Clear checkpoint file."""
        checkpoint_file = self._get_checkpoint_file()
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info("Checkpoint cleared")
            except Exception as e:
                logger.warning(f"Could not clear checkpoint: {e}")

    def load_existing_index(self):
        """Load an existing vector index from storage.

        Returns:
            True if index was loaded successfully, False otherwise
        """
        if not index_exists(self.config.vector_db, table_name=self.config.vector_db.vault_table):
            return False

        try:
            print("\n=== Loading Existing Index ===")
            print("Loading vector index from storage...")

            self.index = load_vector_index(
                vector_store=self.vector_store,
                embed_model=self.embed_model,
                config=self.config.vector_db,
                table_name=self.config.vector_db.vault_table
            )

            print("Index loaded successfully!")

            # Prepare nodes for BM25 if hybrid search is enabled
            if self.config.retrieval.enable_hybrid_search:
                try:
                    print("Preparing documents for BM25 retrieval...")
                    # Get documents from index
                    from llama_index.core.schema import TextNode
                    # Retrieve all nodes from the index using docstore
                    docstore = self.index.docstore
                    all_nodes = list(docstore.docs.values())
                    self.nodes = all_nodes
                    print(f"Loaded {len(self.nodes)} nodes for hybrid search")
                except Exception as e:
                    print(f"Could not prepare BM25 nodes: {e}")
                    print("BM25 will not be available for this session")
                    self.nodes = None

            # Load wikilink graph if available (respecting exclusions)
            try:
                exclusion_patterns = self._get_exclusion_patterns()
                notes = self.loader.load_vault(exclusion_patterns=exclusion_patterns)
                self.wikilink_graph = self.loader.build_wikilink_graph(notes)
                print(f"Wikilink graph loaded with {len(self.wikilink_graph)} nodes")
            except Exception as e:
                print(f"Could not load wikilink graph: {e}")
                self.wikilink_graph = {}

            # Setup query engine
            self._setup_query_engine()

            # Load all optional sources WITHOUT rebuilding federated engine each time
            self._auto_load_books(rebuild_federated=False)
            if self.config.conversations.enabled and self.config.conversations.path:
                self._auto_load_conversations(rebuild_federated=False)
            self._auto_load_saved_items(rebuild_federated=False)

            # Build federated engine ONCE with all loaded sources
            has_extra = (
                self.conversations_index is not None
                or self.books_index is not None
                or self.saved_items_retriever is not None
            )
            if has_extra:
                self._setup_federated_engine()

            return True

        except Exception as e:
            print(f"Failed to load existing index: {e}")
            return False

    def _auto_load_conversations(self, rebuild_federated: bool = True):
        """Auto-load or index conversations if enabled in config."""
        conv_path = self.config.conversations.path

        if not conv_path or not conv_path.exists():
            logger.info(f"Conversations path not found: {conv_path}")
            return

        # Check if conversations index exists
        if self.conversations_index_exists():
            print("\n📚 Loading conversations index...")
            if self.load_conversations_index():
                if rebuild_federated:
                    self._setup_federated_engine()
                print("✅ Federated search enabled (vault + conversations)")
        else:
            # Auto-index conversations (non-interactive mode)
            print(f"\n📚 Auto-indexing conversations from: {conv_path}")
            self.index_conversations(conv_path, force_reindex=False, interactive=False)

    def _auto_load_books(self, rebuild_federated: bool = True):
        """Auto-load books index if it exists on disk."""
        if not self.config.books.enabled:
            return
        if self.books_index is not None:
            return  # Already loaded
        if not self.books_index_exists():
            return
        try:
            print("\n📚 Auto-loading books index...")
            if self.load_books_index():
                if rebuild_federated and self.index is not None:
                    self._setup_federated_engine()
                print("✅ Books index loaded (federated search updated)")
            else:
                print("⚠️  Books index exists but failed to load")
        except Exception as e:
            logger.warning(f"Failed to auto-load books: {e}")

    def _auto_load_saved_items(self, rebuild_federated: bool = True):
        """Auto-load saved_items (TheSource) retriever if enabled in config."""
        if not self.config.saved_items.enabled:
            return
        try:
            from saved_items_retriever import SavedItemsRetriever
            retriever = SavedItemsRetriever(
                config=self.config.saved_items,
                top_k=self.config.retrieval.federated_top_k_per_source,
            )
            if retriever.validate():
                self.saved_items_retriever = retriever
                if rebuild_federated and self.index is not None:
                    self._setup_federated_engine()
                print("✅ TheSource (saved items) loaded")
            else:
                print("⚠️  TheSource table not found or dim mismatch — saved items disabled")
        except Exception as e:
            logger.warning(f"Failed to load saved items: {e}")

    def index_vault(self, force_reindex: bool = False, batch_size: Optional[int] = None):
        """Index the entire Obsidian vault with batch processing and checkpointing.

        Args:
            force_reindex: If True, recreate index even if one exists
            batch_size: Number of notes to process in each batch (defaults to config value)
        """
        print("\n=== Indexing Obsidian Vault ===")

        # Use config batch size if not specified
        if batch_size is None:
            batch_size = self.config.embedding.batch_size

        # Check if index already exists
        if not force_reindex and index_exists(self.config.vector_db, table_name=self.config.vector_db.vault_table):
            print("\nAn existing index was found.")
            print("Options:")
            print("  1. Load existing index (fast)")
            print("  2. Recreate index (slow, overwrites existing data)")
            print("  3. Cancel")

            choice = input("\nYour choice (1/2/3): ").strip()

            if choice == "1":
                if self.load_existing_index():
                    print("\nExisting index loaded successfully!")
                    return
                else:
                    print("\nFailed to load existing index. Creating new index...")

            elif choice == "2":
                print("\nRecreating index (existing data will be overwritten)...")
                # Reinitialize vector store in overwrite mode
                self._setup_vector_store(mode="overwrite")
                # Clear checkpoint when recreating
                if self.config.enable_checkpointing:
                    self._clear_checkpoint()

            elif choice == "3":
                print("\nIndexing cancelled.")
                return

            else:
                print("\nInvalid choice. Cancelling indexing.")
                return

        # If force_reindex is True, reinitialize in overwrite mode
        if force_reindex:
            print("\nForce reindex enabled. Recreating index...")
            self._setup_vector_store(mode="overwrite")
            # Clear checkpoint when force reindexing
            if self.config.enable_checkpointing:
                self._clear_checkpoint()

        # Load checkpoint if enabled
        processed_files: Set[str] = set()
        if self.config.enable_checkpointing and not force_reindex:
            processed_files = self._load_checkpoint()
            if processed_files:
                print(f"\nResuming from checkpoint: {len(processed_files)} files already processed")

        # Get all note paths
        print(f"\nScanning notes from: {self.config.vault_path}")
        note_paths = list(self.config.vault_path.rglob("*.md"))
        logger.info(f"Found {len(note_paths)} notes to index")
        print(f"Found {len(note_paths)} notes total")

        # Apply exclusion patterns if configured
        exclusion_patterns = self._get_exclusion_patterns()
        if exclusion_patterns:
            from exclusion_matcher import ExclusionMatcher
            matcher = ExclusionMatcher(exclusion_patterns)
            note_paths, excluded_count = matcher.filter_files(
                note_paths, self.config.vault_path
            )
            if excluded_count > 0:
                print(f"Excluded {excluded_count} files (based on settings)")
                print(f"Remaining: {len(note_paths)} notes")

        # Filter out already processed files
        if processed_files:
            note_paths = [p for p in note_paths if str(p) not in processed_files]
            print(f"Remaining to process: {len(note_paths)} notes")

        if not note_paths:
            print("\nAll files already processed! Loading existing index...")
            if self.load_existing_index():
                return
            else:
                print("Failed to load index. Please try force_reindex=True")
                return

        # Process in batches with INCREMENTAL indexing
        all_nodes = []
        total_batches = (len(note_paths) + batch_size - 1) // batch_size

        print(f"\nProcessing in batches of {batch_size} notes...")
        print(f"Total batches: {total_batches}")

        for batch_idx in range(0, len(note_paths), batch_size):
            batch_num = batch_idx // batch_size + 1
            batch_paths = note_paths[batch_idx:batch_idx + batch_size]

            print(f"\n--- Batch {batch_num}/{total_batches} ---")
            print(f"Processing {len(batch_paths)} notes...")

            try:
                # Load notes for this batch
                notes = [self.loader.load_note(p) for p in batch_paths]
                notes = [n for n in notes if n]  # Filter out None values

                if not notes:
                    logger.warning(f"Batch {batch_num}: No valid notes loaded")
                    continue

                # Convert to documents
                documents = self.loader.notes_to_documents(notes)

                # Chunk documents (contextual retrieval optional - uses LLM tokens!)
                chunker = ObsidianChunker(
                    config=self.config.embedding,
                    embed_model=self.embed_model,
                    strategy=self.config.embedding.chunking_strategy,
                    use_contextual_retrieval=self.config.embedding.use_contextual_retrieval,
                    llm=self.llm if self.config.embedding.use_contextual_retrieval else None
                )
                batch_nodes = chunker.chunk_documents(documents)
                batch_nodes = chunker.add_parent_document_context(batch_nodes)

                print(f"Created {len(batch_nodes)} chunks from this batch")

                # INCREMENTAL INDEXING: Add to index immediately after each batch
                if self.index is None:
                    # First batch: create the index
                    print("Creating vector index...")
                    self.index = create_vector_index(
                        nodes=batch_nodes,
                        vector_store=self.vector_store,
                        embed_model=self.embed_model,
                        show_progress=True
                    )
                    print(f"Index created with {len(batch_nodes)} nodes")
                else:
                    # Subsequent batches: insert into existing index
                    print("Adding to existing index...")
                    self.index.insert_nodes(batch_nodes, show_progress=True)
                    print(f"Added {len(batch_nodes)} nodes to index")

                all_nodes.extend(batch_nodes)

                # Update checkpoint AFTER index is updated
                if self.config.enable_checkpointing:
                    processed_files.update(str(p) for p in batch_paths)
                    self._save_checkpoint(processed_files)
                    print(f"Checkpoint saved: {len(processed_files)} files indexed")

                # Clear memory
                del notes, documents, batch_nodes
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}", exc_info=True)
                print(f"Warning: Error in batch {batch_num}: {e}")
                print("Continuing with next batch...")
                continue

        if not all_nodes:
            logger.error("No nodes created from vault")
            print("\nError: No valid nodes were created. Please check your vault path and file contents.")
            return

        print(f"\nTotal chunks indexed: {len(all_nodes)}")

        # Store nodes for BM25 retrieval
        self.nodes = all_nodes
        print(f"Stored {len(self.nodes)} nodes for hybrid search")

        print("✅ Indexing complete!")

        # Clear checkpoint after successful indexing
        if self.config.enable_checkpointing:
            self._clear_checkpoint()

        # Build wikilink graph for future use
        print("\nBuilding wikilink graph...")
        # Load all notes for graph building (respecting exclusions)
        all_notes = self.loader.load_vault(exclusion_patterns=exclusion_patterns)
        self.wikilink_graph = self.loader.build_wikilink_graph(all_notes)
        print(f"Graph contains {len(self.wikilink_graph)} nodes")

        # Setup query engine
        self._setup_query_engine()

        # Auto-load saved_items if enabled
        self._auto_load_saved_items()

    def _setup_query_engine(self):
        """Setup query engine after indexing."""
        if self.index is None:
            raise ValueError("Index not created. Run index_vault() first.")

        if self.config.retrieval.enable_hybrid_search:
            print("Using hybrid query engine")
            self.query_engine = HybridQueryEngine(
                index=self.index,
                config=self.config,
                reranker=self.reranker,
                bm25_retriever=self.bm25_retriever,  # Pass existing BM25 retriever if available
                nodes=self.nodes,  # Pass nodes to build BM25 retriever if needed
                wikilink_graph=getattr(self, 'wikilink_graph', {}),  # Pass wikilink graph if available
                query_transformer=self.query_transformer  # Pass query transformer
            )
            # Cache BM25 retriever so _setup_federated_engine() can reuse it without rebuilding
            if hasattr(self.query_engine, 'bm25_retriever') and self.query_engine.bm25_retriever is not None:
                self.bm25_retriever = self.query_engine.bm25_retriever
        else:
            print("Using standard query engine")
            self.query_engine = RAGQueryEngine(
                index=self.index,
                config=self.config,
                reranker=self.reranker,
                query_transformer=self.query_transformer  # Pass query transformer
            )
    
    def query(self, query_str: str, return_sources: bool = True, max_sources: int = None, date_filter: DateFilterPreset = "all_time"):
        """Query the RAG system.

        Args:
            query_str: Query string
            return_sources: Whether to return source nodes
            max_sources: Maximum sources to include (None = all retrieved)
            date_filter: Date filter preset to apply ("all_time", "last_7_days", etc.)
        """
        if self.query_engine is None:
            logger.error("Query attempted before system initialization")
            raise ValueError("System not initialized. Run index_vault() or load_existing_index() first.")

        logger.info(f"Processing query: {query_str[:100]}...")
        if date_filter != "all_time":
            logger.info(f"Date filter active: {date_filter}")
        print(f"\n🔍 Query: {query_str}")
        print("Searching knowledge base...\n")

        try:
            response = self.query_engine.query(query_str)

            # Apply temporal filter if specified
            source_nodes = response.source_nodes
            if date_filter != "all_time":
                temporal_filter = create_temporal_filter(preset=date_filter)
                if temporal_filter:
                    source_nodes = temporal_filter._postprocess_nodes(source_nodes)

            total_sources = len(source_nodes)
            logger.info(f"Query successful, {total_sources} sources found")

            if return_sources:
                sources = self._format_sources(source_nodes, max_sources)
                return {
                    'answer': str(response),
                    'sources': sources,
                    'total_sources': total_sources,
                    'raw_response': response
                }

            return str(response)

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            # Re-raise the exception with context
            raise RuntimeError(f"Query execution failed: {e}") from e

    def _format_sources(self, source_nodes, max_sources: int = None):
        """Format source nodes for display.

        Args:
            source_nodes: List of source nodes
            max_sources: Maximum sources to include (None = all)
        """
        sources = []
        nodes_to_format = source_nodes if max_sources is None else source_nodes[:max_sources]

        for idx, node in enumerate(nodes_to_format, 1):
            # Use original text (without [N] prefix) for display excerpts
            display_text = node.metadata.get('_original_text', node.text)
            sources.append({
                'rank': idx,
                'title': node.metadata.get('title', 'Unknown'),
                'file': node.metadata.get('file_name', 'Unknown'),
                'score': node.score,
                'excerpt': display_text[:1500] + "..." if len(display_text) > 1500 else display_text,
                'source_type': node.metadata.get('source_type', 'vault'),
                'retrieval_source': node.metadata.get('retrieval_source', 'vault')
            })
        return sources
    
    def search_notes(self, query_str: str, top_k: int = 10, date_filter: DateFilterPreset = "all_time"):
        """Search for relevant notes without generation.

        Args:
            query_str: Query string
            top_k: Maximum number of results
            date_filter: Date filter preset to apply
        """
        if self.index is None:
            raise ValueError("Index not created. Run index_vault() first.")

        engine = RAGQueryEngine(
            index=self.index,
            config=self.config,
            reranker=self.reranker
        )

        nodes = engine.get_relevant_nodes(query_str, top_k=top_k)

        # Apply temporal filter if specified
        if date_filter != "all_time":
            temporal_filter = create_temporal_filter(preset=date_filter)
            if temporal_filter:
                nodes = temporal_filter._postprocess_nodes(nodes)

        return self._format_sources(nodes)

    def _setup_conversations_vector_store(self, mode: str = "create"):
        """Setup vector store for conversations index."""
        import lancedb

        # Use same LanceDB path but different table
        db_path = self.config.vector_db.lancedb_path
        table_name = self.config.vector_db.conversations_table

        logger.info(f"Setting up conversations vector store: {db_path}/{table_name}")

        db = lancedb.connect(str(db_path))

        # Check if table exists
        tables_resp = db.list_tables()
        existing_tables = tables_resp.tables if hasattr(tables_resp, "tables") else tables_resp
        table_exists = table_name in existing_tables

        if mode == "overwrite" and table_exists:
            db.drop_table(table_name)
            table_exists = False

        from llama_index.vector_stores.lancedb import LanceDBVectorStore

        self.conversations_vector_store = LanceDBVectorStore(
            uri=str(db_path),
            table_name=table_name,
            mode="overwrite" if not table_exists else "append"
        )

        return table_exists

    def conversations_index_exists(self) -> bool:
        """Check if conversations index exists."""
        import lancedb

        try:
            db_path = self.config.vector_db.lancedb_path
            table_name = self.config.vector_db.conversations_table

            if not db_path.exists():
                return False

            db = lancedb.connect(str(db_path))
            tables_resp = db.list_tables()
            existing_tables = tables_resp.tables if hasattr(tables_resp, "tables") else tables_resp
            return table_name in existing_tables
        except Exception:
            return False

    def index_conversations(
        self,
        conversations_path: Optional[Path] = None,
        force_reindex: bool = False,
        batch_size: int = 50,
        interactive: bool = True
    ):
        """Index AI conversation exports for federated retrieval.

        Args:
            conversations_path: Path to conversations directory (defaults to config)
            force_reindex: Force recreation of index
            batch_size: Number of conversations per batch
            interactive: If True, prompt for choices; if False, auto-load existing
        """
        print("\n=== Indexing AI Conversations ===")

        # Determine path
        conv_path = conversations_path or self.config.conversations.path
        if not conv_path:
            print("❌ No conversations path specified.")
            print("Set CONVERSATIONS_PATH in .env or pass conversations_path argument.")
            return

        conv_path = Path(conv_path)
        if not conv_path.exists():
            print(f"❌ Conversations path not found: {conv_path}")
            return

        # Check for existing index
        if not force_reindex and self.conversations_index_exists():
            if interactive:
                print("\nExisting conversations index found.")
                print("Options:")
                print("  1. Load existing (fast)")
                print("  2. Recreate (slow)")
                print("  3. Cancel")

                choice = input("\nChoice (1/2/3): ").strip()
                if choice == "1":
                    if self.load_conversations_index():
                        print("Conversations index loaded!")
                        self._setup_federated_engine()
                        return
                    print("Failed to load. Recreating...")
                elif choice == "3":
                    return
                # choice == "2" continues to recreate
            else:
                # Non-interactive: just load existing
                if self.load_conversations_index():
                    print("Conversations index loaded!")
                    self._setup_federated_engine()
                    return
                print("Failed to load. Will create new index...")

        # Setup vector store
        mode = "overwrite" if force_reindex or self.conversations_index_exists() else "create"
        self._setup_conversations_vector_store(mode=mode)

        # Load conversations
        print(f"\nLoading conversations from: {conv_path}")
        conv_loader = ConversationLoader(conv_path)

        try:
            conversations = conv_loader.load_all_conversations()
        except Exception as e:
            print(f"❌ Error loading conversations: {e}")
            return

        if not conversations:
            print("No conversations found!")
            return

        print(f"Found {len(conversations)} conversations")

        # Convert to documents
        print("Converting to documents...")
        documents = conv_loader.conversations_to_documents(
            conversations,
            include_full_context=True
        )

        # Chunk with conversation-aware strategy
        print("Chunking conversations...")
        chunker = ConversationChunker(
            chunk_size=self.config.embedding.chunk_size,
            chunk_overlap=self.config.embedding.chunk_overlap,
            respect_turn_boundaries=True
        )

        all_nodes = []
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(documents), batch_size):
            batch_num = batch_idx // batch_size + 1
            batch_docs = documents[batch_idx:batch_idx + batch_size]

            print(f"\n--- Batch {batch_num}/{total_batches} ---")

            try:
                batch_nodes = chunker.chunk_documents(batch_docs)
                print(f"Created {len(batch_nodes)} chunks")

                if self.conversations_index is None:
                    print("Creating conversations index...")
                    self.conversations_index = create_vector_index(
                        nodes=batch_nodes,
                        vector_store=self.conversations_vector_store,
                        embed_model=self.embed_model,
                        show_progress=True
                    )
                else:
                    print("Adding to conversations index...")
                    self.conversations_index.insert_nodes(batch_nodes, show_progress=True)

                all_nodes.extend(batch_nodes)

            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}", exc_info=True)
                print(f"Warning: Error in batch {batch_num}: {e}")
                continue

        self.conversations_nodes = all_nodes
        print(f"\n✅ Indexed {len(all_nodes)} conversation chunks!")

        # Setup federated engine if vault index exists
        if self.index is not None:
            self._setup_federated_engine()

    def load_conversations_index(self) -> bool:
        """Load existing conversations index."""
        if not self.conversations_index_exists():
            return False

        try:
            print("Loading conversations index...")

            # Setup vector store in append mode
            self._setup_conversations_vector_store(mode="append")

            # Load index
            from llama_index.core import StorageContext

            storage_context = StorageContext.from_defaults(
                vector_store=self.conversations_vector_store
            )

            from llama_index.core import VectorStoreIndex

            self.conversations_index = VectorStoreIndex.from_vector_store(
                vector_store=self.conversations_vector_store,
                embed_model=self.embed_model,
                storage_context=storage_context
            )

            # Reconstruct nodes from LanceDB (from_vector_store creates empty docstore)
            from vector_store import reconstruct_nodes_from_lancedb
            table_name = self.config.vector_db.conversations_table
            self.conversations_nodes = reconstruct_nodes_from_lancedb(
                self.config.vector_db, table_name=table_name
            )

            # Also populate the docstore for consistency
            for node in self.conversations_nodes:
                self.conversations_index.docstore.add_documents([node])

            print(f"Loaded {len(self.conversations_nodes)} conversation nodes")

            return True

        except Exception as e:
            logger.error(f"Failed to load conversations index: {e}", exc_info=True)
            return False

    # ============================================
    # Books Indexing Methods
    # ============================================

    def _setup_books_vector_store(self, mode: str = "append") -> bool:
        """Setup vector store for books index.

        Args:
            mode: "append" to add to existing, "overwrite" to recreate

        Returns:
            True if table already existed
        """
        import lancedb

        db_path = self.config.vector_db.lancedb_path
        table_name = self.config.books.table_name

        db = lancedb.connect(str(db_path))
        tables_resp = db.list_tables()
        existing_tables = tables_resp.tables if hasattr(tables_resp, "tables") else tables_resp
        table_exists = table_name in existing_tables

        if mode == "overwrite" and table_exists:
            logger.info(f"Dropping existing books table: {table_name}")
            db.drop_table(table_name)
            table_exists = False

        from llama_index.vector_stores.lancedb import LanceDBVectorStore

        self.books_vector_store = LanceDBVectorStore(
            uri=str(db_path),
            table_name=table_name,
            mode="overwrite" if not table_exists else "append",
            # Books metadata includes list fields (e.g. book_categories).
            flat_metadata=False,
        )

        return table_exists

    def books_index_exists(self) -> bool:
        """Check if books index exists."""
        import lancedb

        try:
            db_path = self.config.vector_db.lancedb_path
            table_name = self.config.books.table_name

            if not db_path.exists():
                return False

            db = lancedb.connect(str(db_path))
            tables_resp = db.list_tables()
            existing_tables = tables_resp.tables if hasattr(tables_resp, "tables") else tables_resp
            return table_name in existing_tables
        except Exception:
            return False

    def _get_books_indexed_paths(self) -> set[str]:
        """Return distinct file paths currently present in the books index table."""
        import lancedb

        if not self.books_index_exists():
            return set()

        db = lancedb.connect(str(self.config.vector_db.lancedb_path))
        table = db.open_table(self.config.books.table_name)

        paths: set[str] = set()
        try:
            rows = table.search().limit(2_000_000).to_list()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                metadata = row.get("metadata", {})
                if isinstance(metadata, dict):
                    file_path = metadata.get("file_path")
                    if file_path:
                        paths.add(file_path)
        except Exception as e:
            logger.warning("Could not inspect books index coverage: %s", e)
            return set()

        return paths

    def _books_index_alignment(self, current_books: set[str]) -> dict:
        """Compare current BOOKS_PATH files to paths present in books index."""
        indexed_books = self._get_books_indexed_paths()
        overlap = len(indexed_books & current_books)
        current_count = len(current_books)
        coverage = (overlap / current_count) if current_count else 0.0
        return {
            "current_count": current_count,
            "indexed_count": len(indexed_books),
            "overlap_count": overlap,
            "coverage": coverage,
            "missing_from_index": max(0, current_count - overlap),
            "indexed_not_in_current": max(0, len(indexed_books) - overlap),
        }

    def index_books(
        self,
        books_path: Optional[Path] = None,
        force_reindex: bool = False,
        batch_size: int = 10,
        interactive: bool = True
    ):
        """Index books (EPUB/PDF) for federated retrieval.

        Args:
            books_path: Path to books directory (defaults to config)
            force_reindex: Force recreation of index
            batch_size: Number of books per batch
            interactive: If True, prompt for choices; if False, auto-load existing
        """
        from book_loader import BookLoader

        print("\n=== Indexing Books ===")

        # Determine path
        bks_path = books_path or self.config.books.path
        if not bks_path:
            print("❌ No books path specified.")
            print("Set BOOKS_PATH in .env or pass books_path argument.")
            return

        bks_path = Path(bks_path)
        if not bks_path.exists():
            print(f"❌ Books path not found: {bks_path}")
            return

        # Discover current books and compute stats
        loader = BookLoader(bks_path)
        all_book_paths = loader.discover_books()
        current_book_paths = {str(p) for p in all_book_paths}
        stats = {
            "total_books": len(all_book_paths),
            "by_type": {"epub": 0, "pdf": 0},
            "total_size_mb": 0.0,
        }
        for book_path in all_book_paths:
            file_type = book_path.suffix.lower().lstrip(".")
            stats["by_type"][file_type] = stats["by_type"].get(file_type, 0) + 1
            stats["total_size_mb"] += book_path.stat().st_size / (1024 * 1024)
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        print(f"\nFound {stats['total_books']} books ({stats['total_size_mb']} MB)")
        print(f"  EPUB: {stats['by_type'].get('epub', 0)}")
        print(f"  PDF: {stats['by_type'].get('pdf', 0)}")

        if stats['total_books'] == 0:
            print("No books found!")
            return

        # Track what this run should index (all books by default)
        selected_book_paths: list[Path] = list(all_book_paths)
        incremental_mode = False
        existing_extra_count = 0

        # Check for existing index
        if not force_reindex and self.books_index_exists():
            indexed_book_paths = self._get_books_indexed_paths()
            overlap_count = len(indexed_book_paths & current_book_paths)
            missing_paths = sorted(current_book_paths - indexed_book_paths)
            extra_paths = sorted(indexed_book_paths - current_book_paths)
            alignment = {
                "current_count": len(current_book_paths),
                "indexed_count": len(indexed_book_paths),
                "overlap_count": overlap_count,
                "coverage": (overlap_count / len(current_book_paths)) if current_book_paths else 0.0,
                "missing_from_index": len(missing_paths),
                "indexed_not_in_current": len(extra_paths),
            }
            existing_extra_count = len(extra_paths)
            has_mismatch = alignment["coverage"] < 0.95 or alignment["indexed_not_in_current"] > 0

            if interactive:
                print("\nExisting books index found.")
                print(
                    f"Index coverage vs current BOOKS_PATH: "
                    f"{alignment['overlap_count']}/{alignment['current_count']} "
                    f"({alignment['coverage']:.1%})"
                )
                if has_mismatch:
                    print("⚠️  Index appears stale or built from a different books path.")
                    print(
                        f"    Missing from index: {alignment['missing_from_index']} | "
                        f"Extra in index: {alignment['indexed_not_in_current']}"
                    )
                print("Options:")
                if has_mismatch:
                    print("  1. Load existing (fast, NOT recommended)")
                    if missing_paths:
                        print("  2. Incrementally index missing books (recommended)")
                        print("  3. Recreate (slow)")
                        print("  4. Cancel")
                    else:
                        print("  2. Recreate (recommended)")
                        print("  3. Cancel")
                else:
                    print("  1. Load existing (fast)")
                    print("  2. Recreate (slow)")
                    print("  3. Cancel")

                choice_prompt = "\nChoice (1/2/3): " if (not missing_paths or not has_mismatch) else "\nChoice (1/2/3/4): "
                choice = input(choice_prompt).strip()
                allow_incremental = bool(missing_paths and has_mismatch)
                if choice == "1":
                    if self.load_books_index():
                        print("Books index loaded!")
                        self._setup_federated_engine()
                        return
                    print("Failed to load. Recreating...")
                elif allow_incremental and choice == "2":
                    if self.load_books_index():
                        selected_book_paths = [Path(p) for p in missing_paths]
                        incremental_mode = True
                        print(f"Will incrementally index {len(selected_book_paths)} missing books...")
                    else:
                        print("Failed to load existing index. Recreating...")
                elif allow_incremental and choice == "3":
                    # Continue to recreate
                    pass
                elif allow_incremental and choice == "4":
                    return
                elif not allow_incremental and choice == "2":
                    # Continue to recreate
                    pass
                elif not allow_incremental and choice == "3":
                    return
                else:
                    print("Invalid choice. Cancelling.")
                    return
            else:
                # Non-interactive: load existing only if coverage looks valid
                if not has_mismatch and self.load_books_index():
                    print("Books index loaded!")
                    self._setup_federated_engine()
                    return
                if alignment["missing_from_index"] > 0:
                    if self.load_books_index():
                        selected_book_paths = [Path(p) for p in missing_paths]
                        incremental_mode = True
                        print(
                            f"Existing books index missing {len(selected_book_paths)} current books. "
                            "Incrementally indexing missing books..."
                        )
                    else:
                        print("Failed to load existing books index. Rebuilding...")
                elif has_mismatch:
                    print(
                        "Existing books index appears stale for current BOOKS_PATH. "
                        "Rebuilding..."
                    )
                else:
                    print("Failed to load existing books index. Will create new index...")

        # Setup vector store when creating/recreating index
        if not incremental_mode:
            mode = "overwrite" if force_reindex or self.books_index_exists() else "create"
            self._setup_books_vector_store(mode=mode)

        # Load selected books
        if incremental_mode:
            print(f"\nLoading {len(selected_book_paths)} new books from: {bks_path}")
            documents = []
            for book_path in tqdm(selected_book_paths, desc="Loading books"):
                documents.extend(loader.load_book(book_path))
        else:
            print(f"\nLoading books from: {bks_path}")
            documents = loader.load_all_books(show_progress=True)

        if not documents:
            print("No documents extracted from books!")
            return

        print(f"Extracted {len(documents)} documents from books")

        # Chunk documents with book-specific chunker
        print("Chunking book documents...")
        from book_chunker import BookChunker, BookChunkConfig

        chunk_config = BookChunkConfig(
            chunk_size=self.config.books.book_chunk_size,
            chunk_overlap=self.config.books.book_chunk_overlap,
            min_chunk_size=self.config.books.book_min_chunk_size,
            respect_chapters=self.config.books.book_respect_chapters,
            respect_paragraphs=self.config.books.book_respect_paragraphs,
        )
        chunker = BookChunker(config=chunk_config)

        all_nodes = []
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(documents), batch_size):
            batch_num = batch_idx // batch_size + 1
            batch_docs = documents[batch_idx:batch_idx + batch_size]

            print(f"\n--- Batch {batch_num}/{total_batches} ---")

            try:
                batch_nodes = chunker.chunk_documents(batch_docs)
                print(f"Created {len(batch_nodes)} chunks")

                if self.books_index is None:
                    print("Creating books index...")
                    self.books_index = create_vector_index(
                        nodes=batch_nodes,
                        vector_store=self.books_vector_store,
                        embed_model=self.embed_model,
                        show_progress=True
                    )
                else:
                    print("Adding to books index...")
                    self.books_index.insert_nodes(batch_nodes, show_progress=True)

                all_nodes.extend(batch_nodes)

            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}", exc_info=True)
                print(f"Warning: Error in batch {batch_num}: {e}")
                continue

        if incremental_mode and self.books_nodes:
            self.books_nodes.extend(all_nodes)
        else:
            self.books_nodes = all_nodes

        if incremental_mode:
            print(f"\n✅ Incrementally indexed {len(all_nodes)} book chunks from {len(selected_book_paths)} books!")
            if existing_extra_count > 0:
                print(
                    f"⚠️  Index still contains {existing_extra_count} books not in current directory. "
                    "Use recreate to fully sync removals."
                )
        else:
            print(f"\n✅ Indexed {len(all_nodes)} book chunks from {stats['total_books']} books!")

        # Setup federated engine if vault index exists
        if self.index is not None:
            self._setup_federated_engine()

    def load_books_index(self) -> bool:
        """Load existing books index."""
        if not self.books_index_exists():
            return False

        try:
            print("Loading books index...")

            # Setup vector store in append mode
            self._setup_books_vector_store(mode="append")

            # Load index
            from llama_index.core import StorageContext, VectorStoreIndex

            storage_context = StorageContext.from_defaults(
                vector_store=self.books_vector_store
            )

            self.books_index = VectorStoreIndex.from_vector_store(
                vector_store=self.books_vector_store,
                embed_model=self.embed_model,
                storage_context=storage_context
            )

            # Reconstruct nodes from LanceDB
            from vector_store import reconstruct_nodes_from_lancedb
            table_name = self.config.books.table_name
            self.books_nodes = reconstruct_nodes_from_lancedb(
                self.config.vector_db, table_name=table_name
            )

            # Populate docstore
            for node in self.books_nodes:
                self.books_index.docstore.add_documents([node])

            print(f"Loaded {len(self.books_nodes)} book nodes")

            return True

        except Exception as e:
            logger.error(f"Failed to load books index: {e}", exc_info=True)
            return False

    def _setup_federated_engine(self, book_filter=None):
        """Setup federated query engine for both indexes."""
        if self.index is None:
            logger.warning("Vault index not available for federated engine")
            return

        sources = []

        # Vault source — pass prebuilt BM25 to avoid rebuilding
        sources.append(IndexSource(
            name="vault",
            index=self.index,
            source_type="vault",
            weight=1.0,
            nodes=self.nodes,
            wikilink_graph=getattr(self, 'wikilink_graph', {}),
            prebuilt_bm25=self.bm25_retriever,  # Reuse vault BM25 built in _setup_query_engine
        ))

        # Conversations source
        if self.conversations_index is not None:
            sources.append(IndexSource(
                name="conversations",
                index=self.conversations_index,
                source_type="conversations",
                weight=self.config.conversations.weight,
                nodes=self.conversations_nodes
            ))

        # Books source
        if self.books_index is not None:
            sources.append(IndexSource(
                name="books",
                index=self.books_index,
                source_type="books",
                weight=self.config.books.weight,
                nodes=self.books_nodes
            ))

        # Saved items (TheSource) source — uses custom retriever (flat schema)
        if self.saved_items_retriever is not None:
            sources.append(IndexSource(
                name="saved_items",
                source_type="saved_items",
                weight=self.config.saved_items.weight,
                custom_retriever=self.saved_items_retriever,
            ))

        if len(sources) > 1:
            self.federated_engine = FederatedQueryEngine(
                sources=sources,
                config=self.config,
                reranker=self.reranker,
                query_transformer=self.query_transformer,
                book_filter=book_filter,
            )
            print(f"Federated engine ready with {len(sources)} sources")
        else:
            logger.info("Only one source available, federated engine not needed")

    def get_book_categories(self) -> list[dict]:
        """Get category catalog from in-memory books nodes (no LanceDB scan)."""
        from collections import Counter
        cats = Counter()
        for node in self.books_nodes or []:
            for cat in node.metadata.get("book_categories", []):
                cats[cat] += 1
        return [{"name": k, "count": v} for k, v in cats.most_common()]

    def query_federated(
        self,
        query_str: str,
        source_filter: Optional[List[str]] = None,
        return_sources: bool = True,
        max_sources: int = None,
        date_filter: DateFilterPreset = "all_time",
        book_filter: Optional[BookFilter] = None,
    ):
        """Query both vault and conversations with federated retrieval.

        Uses numbered context for proper [1], [2], [3] citations that match displayed sources.

        Args:
            query_str: Query string
            source_filter: Optional list of sources ("vault", "conversations")
            return_sources: Include source information in response
            max_sources: Maximum sources to include (None = use rerank_top_n)
            date_filter: Date filter preset to apply
            book_filter: Optional BookFilter for category/author filtering on books
        """
        if self.federated_engine is None:
            # Fallback to regular query if no federated engine
            if self.conversations_index is not None and self.index is not None:
                self._setup_federated_engine()

            if self.federated_engine is None:
                logger.warning("Federated engine not available, using standard query")
                return self.query(query_str, return_sources=return_sources, max_sources=max_sources, date_filter=date_filter)

        # Rebuild engine when book_filter changes (including clearing)
        if book_filter != getattr(self, '_active_book_filter', None):
            self._active_book_filter = book_filter
            self._setup_federated_engine(book_filter=book_filter)

        if date_filter != "all_time":
            logger.info(f"Federated query with date filter: {date_filter}")

        filter_desc = ""
        if book_filter:
            parts = []
            if book_filter.categories:
                parts.append(f"categories={book_filter.categories}")
            if book_filter.authors:
                parts.append(f"authors={book_filter.authors}")
            filter_desc = f" [filter: {', '.join(parts)}]"

        print(f"\n🔍 Federated Query: {query_str}{filter_desc}")
        print("Searching vault and conversations...\n")

        try:
            # Step 1: Retrieve nodes (includes query transformation, reranking)
            all_nodes = self.federated_engine.retrieve(
                query_str,
                source_filter=source_filter
            )

            # Step 2: Apply temporal filter if specified
            if date_filter != "all_time":
                temporal_filter = create_temporal_filter(preset=date_filter)
                if temporal_filter:
                    all_nodes = temporal_filter._postprocess_nodes(all_nodes)

            total_retrieved = len(all_nodes)

            # Step 3: Limit nodes for synthesis (default to rerank_top_n if max_sources not specified)
            synthesis_limit = max_sources if max_sources is not None else self.config.retrieval.rerank_top_n
            nodes_for_synthesis = all_nodes[:synthesis_limit] if synthesis_limit > 0 else all_nodes
            num_sources = len(nodes_for_synthesis)

            logger.info(f"Federated synthesis: using {num_sources} of {total_retrieved} nodes")

            # Step 4: Build numbered context for proper [1], [2], [3] citations
            context_parts = []
            for i, node in enumerate(nodes_for_synthesis, 1):
                # Handle NodeWithScore wrapper
                node_obj = node.node if hasattr(node, 'node') else node
                metadata = node_obj.metadata
                source_type = metadata.get('source_type', 'vault')
                # Books store title as book_title; fall back gracefully
                title = (
                    metadata.get('title')
                    or metadata.get('book_title')
                    or metadata.get('file_name', 'Unknown')
                )
                file_path = metadata.get('file_path', metadata.get('file_name', ''))
                # For saved_items use display_label (domain) instead of raw item_id
                # For books, use the book title + author for readability
                if source_type == 'saved_items':
                    display_path = metadata.get('display_label') or file_path
                elif source_type == 'books':
                    author = metadata.get('book_author', '')
                    display_path = f"{title} — {author}" if author else title
                else:
                    display_path = file_path
                context_parts.append(
                    f"[{i}] Source: {title}\n"
                    f"File: {display_path} (source_type: {source_type})\n"
                    f"Content:\n{node_obj.text}\n"
                )
            numbered_context = "\n---\n".join(context_parts)

            # Step 5: Build synthesis prompt with dynamic template
            from federated_query import _get_federated_template
            source_types = list(set(
                (node.node if hasattr(node, 'node') else node).metadata.get('source_type', 'vault')
                for node in nodes_for_synthesis
            ))
            template = _get_federated_template(source_types)
            prompt = template.replace("{context_str}", numbered_context).replace("{query_str}", query_str)

            # Step 6: Call LLM directly for synthesis
            from llama_index.core import Settings
            response = Settings.llm.complete(prompt)
            answer = response.text

            # Step 7: Build source summary
            source_summary = {
                "total_nodes": total_retrieved,
                "by_source": {},
                "by_type": {"vault": 0, "conversations": 0, "books": 0, "saved_items": 0},
            }
            for node in nodes_for_synthesis:
                node_obj = node.node if hasattr(node, 'node') else node
                st = node_obj.metadata.get('source_type', 'vault')
                if st in source_summary["by_type"]:
                    source_summary["by_type"][st] += 1

            if return_sources:
                # Build sources list matching synthesis nodes (citations will match)
                sources = []
                for i, node in enumerate(nodes_for_synthesis, 1):
                    node_obj = node.node if hasattr(node, 'node') else node
                    metadata = node_obj.metadata
                    stype = metadata.get('source_type', 'vault')
                    # Books store title as book_title; fall back gracefully
                    title = (
                        metadata.get('title')
                        or metadata.get('book_title')
                        or metadata.get('file_name', 'Unknown')
                    )
                    # For books, display_label = "Title — Author" for the link line
                    if stype == 'books':
                        author = metadata.get('book_author', '')
                        display_label = f"{title} — {author}" if author else title
                    else:
                        display_label = metadata.get('display_label')
                    sources.append({
                        'rank': i,
                        'title': title,
                        'file': metadata.get('file_path', metadata.get('file_name', 'Unknown')),
                        'score': node.score if hasattr(node, 'score') else 0.0,
                        'excerpt': node_obj.text[:1500] + "..." if len(node_obj.text) > 1500 else node_obj.text,
                        'source_type': stype,
                        'url': metadata.get('url'),
                        'domain': metadata.get('domain'),
                        'display_label': display_label,
                    })

                return {
                    'answer': answer,
                    'sources': sources,
                    'total_sources': total_retrieved,
                    'source_summary': source_summary
                }

            return answer

        except Exception as e:
            logger.error(f"Federated query failed: {e}", exc_info=True)
            raise RuntimeError(f"Federated query failed: {e}") from e

    def query_vault_only(self, query_str: str, return_sources: bool = True, max_sources: int = None, date_filter: DateFilterPreset = "all_time"):
        """Query only the vault index (exclude conversations)."""
        return self.query_federated(
            query_str,
            source_filter=["vault"],
            return_sources=return_sources,
            max_sources=max_sources,
            date_filter=date_filter
        )

    def query_conversations_only(self, query_str: str, return_sources: bool = True, max_sources: int = None, date_filter: DateFilterPreset = "all_time"):
        """Query only the conversations index."""
        return self.query_federated(
            query_str,
            source_filter=["conversations"],
            return_sources=return_sources,
            max_sources=max_sources,
            date_filter=date_filter
        )

    def query_research(
        self,
        query_str: str,
        return_sources: bool = True,
        max_sources: int = None,
        date_filter: DateFilterPreset = "all_time",
        force_exhaustive: bool = False,
        source_filter: Optional[List[str]] = None
    ):
        """Execute multi-step research mode for complex queries.

        Research mode performs iterative retrieval with gap analysis and query refinement.
        This is 3-5x slower but provides 141% accuracy improvement (based on Khoj benchmarks).

        Args:
            query_str: User query (supports @all prefix for exhaustive search)
            return_sources: Whether to return source nodes (default: True)
            max_sources: Maximum sources to include in response (None = all)
            date_filter: Date filter preset to apply
            force_exhaustive: Force all iterations regardless of confidence threshold
            source_filter: List of source names to query (e.g. ["vault", "conversations", "books", "saved_items"]).
                           None = vault only (legacy default).

        Returns:
            Dictionary with answer, sources, and research summary
        """
        if not self.query_engine:
            raise RuntimeError("Query engine not initialized. Please run index_vault() or load_existing_index() first.")

        # Parse @all prefix for exhaustive search
        if query_str.startswith("@all "):
            force_exhaustive = True
            query_str = query_str[5:].strip()
            logger.info("Exhaustive mode enabled via @all prefix")

        logger.info(f"Research mode query: {query_str[:100]}...")
        if date_filter != "all_time":
            logger.info(f"Research mode with date filter: {date_filter}")

        try:
            # Import research module
            from research_mode import ResearchRetriever, llm_complete_with_retry
            from llama_index.core.retrievers import VectorIndexRetriever

            # Determine if we should use federated retrieval (multiple sources)
            use_federated = (
                source_filter is not None
                and len(source_filter) > 1
                and any(s != "vault" for s in source_filter)
            )

            if use_federated:
                # Build a FederatedRetriever covering all checked sources
                from federated_query import FederatedRetriever, IndexSource
                fed_sources = []

                if "vault" in source_filter and self.index is not None:
                    fed_sources.append(IndexSource(
                        name="vault", index=self.index, source_type="vault",
                        weight=1.0, nodes=self.nodes,
                        wikilink_graph=getattr(self, 'wikilink_graph', {}),
                        prebuilt_bm25=self.bm25_retriever,
                    ))
                if "conversations" in source_filter and self.conversations_index is not None:
                    fed_sources.append(IndexSource(
                        name="conversations", index=self.conversations_index,
                        source_type="conversations",
                        weight=self.config.conversations.weight,
                        nodes=self.conversations_nodes,
                    ))
                if "books" in source_filter and self.books_index is not None:
                    fed_sources.append(IndexSource(
                        name="books", index=self.books_index,
                        source_type="books",
                        weight=self.config.books.weight,
                        nodes=self.books_nodes,
                    ))
                if "saved_items" in source_filter and self.saved_items_retriever is not None:
                    fed_sources.append(IndexSource(
                        name="saved_items", source_type="saved_items",
                        weight=self.config.saved_items.weight,
                        custom_retriever=self.saved_items_retriever,
                    ))

                if len(fed_sources) > 1:
                    base_retriever = FederatedRetriever(
                        sources=fed_sources,
                        config=self.config,
                        query_transformer=self.query_transformer,
                        reranker=self.reranker,
                        top_k_per_source=self.config.retrieval.top_k,
                    )
                    logger.info(f"Research mode: federated retriever with {len(fed_sources)} sources: "
                                f"{[s.name for s in fed_sources]}")
                else:
                    # Only one source actually available, fall back to vault-only
                    use_federated = False

            if not use_federated:
                # Vault-only retriever (legacy path)
                # Research mode has its own iterative refinement, so no self-correction wrapper
                base_retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=self.config.retrieval.top_k
                )

                # Add query transformation if enabled (but NOT self-correction)
                if self.query_transformer and self.config.retrieval.query_transform_method not in ["none", "disabled"]:
                    from query_engine import QueryTransformRetriever
                    base_retriever = QueryTransformRetriever(
                        base_retriever=base_retriever,
                        query_transformer=self.query_transformer,
                        transform_method=self.config.retrieval.query_transform_method,
                        num_queries=self.config.retrieval.query_transform_num_queries
                    )
                    logger.info("Research mode: vault-only with query transformation")

            # Create research retriever
            research_retriever = ResearchRetriever(
                base_retriever=base_retriever,
                llm=self.llm,
                max_iterations=self.config.retrieval.research_max_iterations,
                confidence_threshold=self.config.retrieval.research_confidence_threshold,
                max_subqueries=self.config.retrieval.research_max_subqueries,
                enable_research=self.config.retrieval.enable_research_mode
            )

            # Execute research (pass exhaustive flag)
            research_result = research_retriever.research(
                query_str, force_exhaustive=force_exhaustive
            )

            logger.info(
                f"Research completed: {research_result.total_iterations} iterations, "
                f"{research_result.total_nodes_retrieved} nodes, "
                f"confidence={research_result.final_confidence:.2f}"
            )

            # Generate final answer using retrieved context with research template
            from llama_index.core import Settings
            from query_engine import RESEARCH_TEMPLATE

            # Apply temporal filter if specified
            all_retrieved = research_result.final_nodes
            if date_filter != "all_time":
                temporal_filter = create_temporal_filter(preset=date_filter)
                if temporal_filter:
                    all_retrieved = temporal_filter._postprocess_nodes(all_retrieved)

            # Research mode uses ALL sources for synthesis (0 = unlimited, default)
            # UI dropdown only controls display count, not synthesis depth
            total_retrieved = len(all_retrieved)  # Original count for reporting
            synthesis_limit = self.config.retrieval.research_max_synthesis_sources

            # Apply user-configured limit if set, otherwise use all
            nodes_for_retry = all_retrieved
            if synthesis_limit > 0:
                nodes_for_retry = all_retrieved[:synthesis_limit]

            # Synthesis strategy: estimate total tokens, pick single-call or chunked path
            from citations import offset_citations, validate_citations

            TOKEN_QUOTA = 1_000_000
            HARD_FLOOR = 150  # ~525K tokens at 3500/node — safe under 1M
            available_nodes = len(nodes_for_retry)

            # Estimate avg tokens per node (rough: 1 token ≈ 4 chars)
            if available_nodes > 0:
                sample = nodes_for_retry[:min(20, available_nodes)]
                avg_chars = sum(len(n.node.text) + 200 for n in sample) / len(sample)  # +200 for metadata/template
                est_tokens_per_node = avg_chars / 3.5  # conservative estimate
            else:
                est_tokens_per_node = 2000

            total_est_tokens = available_nodes * est_tokens_per_node
            nodes_for_synthesis = None
            combined_answer = None

            if total_est_tokens <= TOKEN_QUOTA * 0.9:
                # === SINGLE-CALL PATH ===
                logger.info(f"Single-call synthesis: {available_nodes} nodes (~{total_est_tokens/1e6:.1f}M tokens)")

                research_prompt = _build_chunk_prompt(nodes_for_retry, query_str)
                try:
                    response = llm_complete_with_retry(Settings.llm, research_prompt, max_retries=2)
                    combined_answer = response.text
                    nodes_for_synthesis = nodes_for_retry
                    logger.info(f"Single-call synthesis succeeded with {available_nodes} nodes")
                except Exception as e:
                    error_str = str(e)
                    is_recoverable = ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                                      or "MAX_TOKENS" in error_str
                                      or "token count exceeds" in error_str.lower())
                    if is_recoverable:
                        # Fall back to progressive retry with reduced nodes
                        logger.warning(f"Single-call failed ({error_str[:80]}), falling back to progressive retry")
                        for pct in [0.66, 0.50, 0.33]:
                            limit = max(HARD_FLOOR, int(available_nodes * pct))
                            subset = nodes_for_retry[:limit]
                            retry_prompt = _build_chunk_prompt(subset, query_str)
                            try:
                                response = llm_complete_with_retry(Settings.llm, retry_prompt, max_retries=2)
                                combined_answer = response.text
                                nodes_for_synthesis = subset
                                logger.info(f"Progressive retry succeeded with {len(subset)} nodes")
                                break
                            except Exception:
                                continue
                        if combined_answer is None:
                            raise
                    else:
                        raise

            elif self.config.llm.backend == "cli":
                # === CLI FALLBACK ===
                # CLI has 120s hard timeout, too short for multi-chunk
                logger.warning("Chunked synthesis not supported with CLI backend, using progressive retry")
                for pct in [0.66, 0.50, 0.33]:
                    limit = max(HARD_FLOOR, int(available_nodes * pct))
                    subset = nodes_for_retry[:limit]
                    research_prompt = _build_chunk_prompt(subset, query_str)
                    try:
                        response = llm_complete_with_retry(Settings.llm, research_prompt, max_retries=2)
                        combined_answer = response.text
                        nodes_for_synthesis = subset
                        logger.info(f"CLI progressive retry succeeded with {len(subset)} nodes")
                        break
                    except Exception:
                        continue
                if combined_answer is None:
                    # Last resort: hard floor
                    subset = nodes_for_retry[:HARD_FLOOR]
                    research_prompt = _build_chunk_prompt(subset, query_str)
                    response = llm_complete_with_retry(Settings.llm, research_prompt, max_retries=2)
                    combined_answer = response.text
                    nodes_for_synthesis = subset

            else:
                # === CHUNKED SYNTHESIS PATH (API backend only) ===
                CHUNK_BUDGET = int(TOKEN_QUOTA * 0.8)  # 800K per chunk
                nodes_per_chunk = max(50, int(CHUNK_BUDGET / est_tokens_per_node))
                chunks = [nodes_for_retry[i:i + nodes_per_chunk]
                          for i in range(0, available_nodes, nodes_per_chunk)]

                logger.info(
                    f"Chunked synthesis: {len(chunks)} chunks of ~{nodes_per_chunk} nodes "
                    f"(~{nodes_per_chunk * est_tokens_per_node / 1e6:.1f}M tokens each)"
                )

                chunk_responses = []
                global_start_index = 0

                for chunk_idx, chunk_nodes in enumerate(chunks):
                    num_in_chunk = len(chunk_nodes)
                    is_continuation = chunk_idx > 0

                    research_prompt = _build_chunk_prompt(
                        chunk_nodes, query_str, chunk_idx, is_continuation
                    )

                    try:
                        response = llm_complete_with_retry(Settings.llm, research_prompt, max_retries=2)
                    except Exception as e:
                        error_str = str(e)
                        is_recoverable = ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                                          or "MAX_TOKENS" in error_str
                                          or "token count exceeds" in error_str.lower())
                        if is_recoverable and num_in_chunk > 60:
                            logger.warning(f"Chunk {chunk_idx + 1} failed, splitting in half")
                            half = num_in_chunk // 2
                            for sub_nodes in [chunk_nodes[:half], chunk_nodes[half:]]:
                                sub_prompt = _build_chunk_prompt(
                                    sub_nodes, query_str, chunk_idx, is_continuation=True
                                )
                                sub_resp = llm_complete_with_retry(Settings.llm, sub_prompt, max_retries=2)
                                chunk_text = offset_citations(sub_resp.text, global_start_index)
                                chunk_responses.append(chunk_text)
                                global_start_index += len(sub_nodes)
                            continue
                        raise

                    logger.info(f"Chunk {chunk_idx + 1}/{len(chunks)} succeeded ({num_in_chunk} nodes)")
                    chunk_text = offset_citations(response.text, global_start_index)
                    chunk_responses.append(chunk_text)
                    global_start_index += num_in_chunk

                # Merge chunk responses
                if len(chunk_responses) > 1:
                    parts = [f"## Part {i}\n\n{text}" for i, text in enumerate(chunk_responses, 1)]
                    combined_answer = "\n\n---\n\n".join(parts)
                else:
                    combined_answer = chunk_responses[0]

                nodes_for_synthesis = nodes_for_retry  # ALL nodes for source list

            # Validate citations and clean whitespace
            combined_answer = validate_citations(combined_answer, len(nodes_for_synthesis))

            # Format result
            result = {
                'answer': combined_answer,
                'research_summary': research_result.get_iteration_summary(),
                'gap_analyses': research_result.get_gap_analyses(),
                'gap_analyses_markdown': research_result.get_gap_analyses_markdown()
            }

            if return_sources:
                sources = []
                # Use same nodes that were used for synthesis (so citations match)
                for i, node in enumerate(nodes_for_synthesis, 1):
                    md = node.metadata
                    stype = md.get('source_type', 'vault')
                    title = md.get('title') or md.get('book_title') or md.get('file_name', 'Unknown')
                    if stype == 'books':
                        author = md.get('book_author', '')
                        display_label = f"{title} — {author}" if author else title
                    else:
                        display_label = md.get('display_label')
                    sources.append({
                        'rank': i,
                        'title': title,
                        'file': md.get('file_path', 'Unknown'),
                        'file_path': md.get('file_path', 'Unknown'),
                        'score': node.score or 0.0,
                        'excerpt': node.node.text[:1500] + "..." if len(node.node.text) > 1500 else node.node.text,
                        'source_type': stype,
                        'url': md.get('url'),
                        'domain': md.get('domain'),
                        'display_label': display_label,
                    })
                result['sources'] = sources
                result['total_sources'] = total_retrieved  # Total retrieved, not just used for synthesis

            return result

        except Exception as e:
            logger.error(f"Research mode query failed: {e}", exc_info=True)
            raise RuntimeError(f"Research mode query failed: {e}") from e

    def research(self, topic: str, depth: str = "standard") -> ResearchResult:
        """Research a topic using vault and optionally web sources.

        This method provides a unified interface for research that combines
        vault knowledge with optional web search. It returns a structured
        ResearchResult with synthesized summary, sources, and citations.

        Args:
            topic: The research topic/question
            depth: Research depth level:
                - "quick": Vault only (fastest)
                - "standard": Vault + web search if enabled
                - "deep": Full research mode with iterative retrieval

        Returns:
            ResearchResult with synthesized summary and sources

        Example:
            >>> result = rag.research("What are the benefits of meditation?", depth="standard")
            >>> print(result.summary)
            >>> for source in result.sources:
            ...     print(f"{source.title} ({source.type})")
        """
        logger.info(f"Research query: {topic[:100]}... (depth={depth})")
        print(f"\n🔍 Research: {topic}")
        print(f"Depth: {depth}")

        all_sources: List[Source] = []
        all_nodes = []
        vault_count = 0
        web_count = 0
        conv_count = 0

        try:
            # Step 1: Gather vault/conversation sources based on depth
            if depth == "quick":
                # Quick: vault only
                print("Searching vault only (quick mode)...")
                result = self.query_vault_only(topic, return_sources=True)
                if result and 'sources' in result:
                    for src in result['sources']:
                        source_type = src.get('source_type', 'vault')
                        all_sources.append(Source(
                            type=source_type,
                            path=src.get('file'),
                            url=None,
                            relevance=src.get('score', 0.0),
                            snippet=src.get('excerpt', ''),
                            title=src.get('title', 'Unknown')
                        ))
                        if source_type == 'vault':
                            vault_count += 1
                        elif source_type == 'conversations':
                            conv_count += 1
                    summary = result.get('answer', '')

            elif depth == "deep":
                # Deep: use existing research mode
                print("Running deep research (iterative retrieval)...")
                result = self.query_research(topic, return_sources=True)
                if result and 'sources' in result:
                    for src in result['sources']:
                        source_type = src.get('source_type', 'vault')
                        all_sources.append(Source(
                            type=source_type,
                            path=src.get('file'),
                            url=None,
                            relevance=src.get('score', 0.0),
                            snippet=src.get('excerpt', ''),
                            title=src.get('title', 'Unknown')
                        ))
                        if source_type == 'vault':
                            vault_count += 1
                        elif source_type == 'conversations':
                            conv_count += 1
                    summary = result.get('answer', '')

            else:
                # Standard: federated query (vault + conversations if available)
                print("Searching vault and conversations...")
                result = self.query_federated(topic, return_sources=True)
                if result and 'sources' in result:
                    for src in result['sources']:
                        source_type = src.get('source_type', 'vault')
                        all_sources.append(Source(
                            type=source_type,
                            path=src.get('file'),
                            url=None,
                            relevance=src.get('score', 0.0),
                            snippet=src.get('excerpt', ''),
                            title=src.get('title', 'Unknown')
                        ))
                        if source_type == 'vault':
                            vault_count += 1
                        elif source_type == 'conversations':
                            conv_count += 1
                    summary = result.get('answer', '')

            # Step 2: Add web search if enabled and depth is not "quick"
            if (
                depth != "quick" and
                self.config.web_search.enabled and
                self.config.web_search.include_in_research and
                self.web_retriever is not None
            ):
                print("Adding web search results...")
                try:
                    web_nodes = self.web_retriever.retrieve(
                        topic,
                        max_results=self.config.web_search.max_results
                    )

                    # Apply weight to web results
                    web_weight = self.config.web_search.weight
                    for node in web_nodes:
                        weighted_score = (node.score or 0.5) * web_weight
                        all_sources.append(Source(
                            type="web",
                            path=None,
                            url=node.node.metadata.get('url', ''),
                            relevance=weighted_score,
                            snippet=node.node.text[:1500] + "..." if len(node.node.text) > 1500 else node.node.text,
                            title=node.node.metadata.get('title', 'Web Result')
                        ))
                        web_count += 1
                        all_nodes.append(node)

                    logger.info(f"Added {web_count} web results to research")

                except Exception as e:
                    logger.warning(f"Web search failed during research: {e}")
                    print(f"Warning: Web search failed: {e}")

            # Step 3: Re-synthesize if we added web results
            if web_count > 0 and all_nodes:
                print("Re-synthesizing with web sources...")

                # Build combined context
                context_parts = []
                for i, source in enumerate(all_sources, 1):
                    type_icon = {"vault": "📓", "conversations": "💬", "web": "🌐"}.get(source.type, "📄")
                    location = source.url if source.type == "web" else (source.path or "unknown")
                    context_parts.append(
                        f"[{i}] {type_icon} {source.title}\n"
                        f"Source: {location} ({source.type})\n"
                        f"Content:\n{source.snippet}\n"
                    )
                combined_context = "\n---\n".join(context_parts)

                # Build synthesis prompt
                from federated_query import _get_federated_template
                source_types = list(set(s.type for s in all_sources))
                template = _get_federated_template(source_types)
                prompt = template.replace("{context_str}", combined_context).replace("{query_str}", topic)

                # Generate combined summary
                response = Settings.llm.complete(prompt)
                summary = response.text

            # Step 4: Build citations
            citations = []
            for i, source in enumerate(all_sources, 1):
                type_label = {"vault": "vault", "conversations": "conv", "web": "web"}.get(source.type, source.type)
                citations.append(Citation(
                    index=i,
                    text=f"[{i}] {source.title} - {type_label}"
                ))

            # Sort sources by relevance
            all_sources.sort(key=lambda s: s.relevance, reverse=True)

            logger.info(f"Research complete: {vault_count} vault, {conv_count} conv, {web_count} web sources")

            return ResearchResult(
                summary=summary,
                sources=all_sources,
                citations=citations,
                vault_sources=vault_count + conv_count,  # Combined local sources
                web_sources=web_count,
                query=topic
            )

        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
            raise RuntimeError(f"Research failed: {e}") from e

    def get_token_usage(self) -> dict:
        """Get current Voyage AI token usage statistics."""
        return self.token_tracker.get_status()

    def print_token_usage(self):
        """Print formatted token usage status."""
        self.token_tracker.print_status()

    # ============================================
    # RAPTOR: Hierarchical Summaries
    # ============================================

    def _setup_raptor(self):
        """Initialize RAPTOR index manager."""
        logger.info("Setting up RAPTOR index manager...")
        print("Setting up RAPTOR hierarchical summaries...")

        try:
            self.raptor_manager = RaptorIndexManager(
                embed_model=self.embed_model,
                llm=self.llm,
                raptor_path=self.config.raptor.raptor_path,
                chunk_size=self.config.raptor.chunk_size,
                chunk_overlap=self.config.raptor.chunk_overlap,
                similarity_top_k=self.config.raptor.similarity_top_k,
                default_mode=self.config.raptor.mode
            )
            logger.info(f"RAPTOR manager initialized: mode={self.config.raptor.mode}")
        except Exception as e:
            logger.warning(f"Could not initialize RAPTOR manager: {e}")
            print(f"Warning: RAPTOR initialization failed: {e}")
            self.raptor_manager = None

    def raptor_index_exists(self) -> bool:
        """Check if RAPTOR index exists.

        Returns:
            True if RAPTOR index exists and has data
        """
        if self.raptor_manager is None:
            return False
        return self.raptor_manager.index_exists()

    def index_raptor(self, force_reindex: bool = False, interactive: bool = True):
        """Build RAPTOR hierarchical index from vault documents.

        RAPTOR creates a tree of summaries through recursive clustering:
        1. Documents are chunked and embedded
        2. Similar chunks are clustered
        3. Each cluster is summarized by the LLM
        4. Process repeats to build hierarchy

        Args:
            force_reindex: Force rebuild even if index exists
            interactive: If True, prompt for user confirmation
        """
        print("\n=== Building RAPTOR Hierarchical Index ===")

        # Initialize RAPTOR manager if not already done
        if self.raptor_manager is None:
            self._setup_raptor()

        if self.raptor_manager is None:
            print("RAPTOR manager not available. Check your configuration.")
            return

        # Check for existing index
        if not force_reindex and self.raptor_index_exists():
            if interactive:
                print("\nExisting RAPTOR index found.")
                print("Options:")
                print("  1. Load existing (fast)")
                print("  2. Rebuild (slow, uses LLM for summarization)")
                print("  3. Cancel")

                choice = input("\nChoice (1/2/3): ").strip()
                if choice == "1":
                    if self.load_raptor_index():
                        print("RAPTOR index loaded!")
                        return
                    print("Failed to load. Rebuilding...")
                elif choice == "3":
                    print("Cancelled.")
                    return
                # choice == "2" continues to rebuild
            else:
                # Non-interactive: just load
                if self.load_raptor_index():
                    print("RAPTOR index loaded!")
                    return
                print("Failed to load. Will create new index...")

        # Load documents from vault (respecting exclusions)
        print(f"\nLoading documents from: {self.config.vault_path}")
        exclusion_patterns = self._get_exclusion_patterns()
        notes = self.loader.load_vault(exclusion_patterns=exclusion_patterns)

        if not notes:
            print("No notes found in vault!")
            return

        print(f"Found {len(notes)} notes")

        # Convert to LlamaIndex documents
        documents = self.loader.notes_to_documents(notes)
        print(f"Converted to {len(documents)} documents")

        # Build RAPTOR index
        print("\nBuilding RAPTOR tree (this may take several minutes)...")
        print("Note: RAPTOR uses LLM to generate cluster summaries")

        try:
            success = self.raptor_manager.build_index(
                documents=documents,
                force_rebuild=force_reindex
            )

            if success:
                print("\nRAPTOR index built successfully!")

                # Print stats
                stats = self.raptor_manager.get_summary_stats()
                print(f"  Nodes in tree: {stats.get('node_count', 'unknown')}")
                print(f"  Default mode: {stats.get('default_mode', 'unknown')}")
            else:
                print("\nFailed to build RAPTOR index.")

        except Exception as e:
            logger.error(f"RAPTOR indexing failed: {e}", exc_info=True)
            print(f"\nError building RAPTOR index: {e}")

    def load_raptor_index(self) -> bool:
        """Load existing RAPTOR index.

        Returns:
            True if index was loaded successfully
        """
        if self.raptor_manager is None:
            self._setup_raptor()

        if self.raptor_manager is None:
            return False

        return self.raptor_manager.load_index()

    def query_raptor(
        self,
        query_str: str,
        mode: Optional[RaptorMode] = None,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        max_sources: Optional[int] = None
    ):
        """Query using RAPTOR hierarchical summaries.

        RAPTOR retrieval modes:
        - "collapsed": Treats entire tree as flat list, simple top-k (faster)
        - "tree_traversal": Traverses hierarchy, top-k at each level (more comprehensive)

        Args:
            query_str: Query string
            mode: Retrieval mode ("collapsed" or "tree_traversal")
            top_k: Number of results to retrieve
            return_sources: Include source information in response
            max_sources: Maximum sources to include (None = all)

        Returns:
            Dictionary with answer and sources
        """
        if self.raptor_manager is None:
            raise RuntimeError(
                "RAPTOR index not available. "
                "Run index_raptor() first or enable RAPTOR in config."
            )

        if not self.raptor_index_exists():
            raise RuntimeError(
                "RAPTOR index not built. Run index_raptor() first."
            )

        mode = mode or self.config.raptor.mode
        top_k = top_k or self.config.raptor.similarity_top_k

        logger.info(f"RAPTOR query: {query_str[:100]}... (mode={mode})")
        print(f"\n🌳 RAPTOR Query (mode={mode}): {query_str}")
        print("Searching hierarchical summaries...\n")

        try:
            # Retrieve nodes from RAPTOR tree
            nodes = self.raptor_manager.retrieve(
                query=query_str,
                mode=mode,
                top_k=top_k
            )

            if not nodes:
                return {
                    'answer': "No relevant information found in the RAPTOR index.",
                    'sources': [],
                    'total_sources': 0
                }

            # Apply max_sources limit
            total_retrieved = len(nodes)
            nodes_for_synthesis = nodes if max_sources is None else nodes[:max_sources]

            # Build context for LLM
            context_parts = []
            for i, node in enumerate(nodes_for_synthesis, 1):
                # Handle both NodeWithScore and TextNode
                if hasattr(node, 'node'):
                    text = node.node.text
                    metadata = node.node.metadata
                    score = node.score or 0.0
                else:
                    text = node.text
                    metadata = node.metadata
                    score = 0.0

                title = metadata.get('title', 'Summary')
                context_parts.append(
                    f"[{i}] {title}\n{text}\n"
                )

            context = "\n---\n".join(context_parts)

            # Generate response using LLM
            prompt = f"""Based on the following hierarchical summaries from your knowledge base,
answer the user's question. Use numbered citations like [1], [2] to reference specific sources.

Context (from RAPTOR hierarchical summaries):
{context}

Question: {query_str}

Answer:"""

            response = Settings.llm.complete(prompt)

            # Format result
            result = {
                'answer': response.text,
                'total_sources': total_retrieved,
                'raptor_mode': mode
            }

            if return_sources:
                sources = []
                for i, node in enumerate(nodes_for_synthesis, 1):
                    if hasattr(node, 'node'):
                        text = node.node.text
                        metadata = node.node.metadata
                        score = node.score or 0.0
                    else:
                        text = node.text
                        metadata = node.metadata
                        score = 0.0

                    sources.append({
                        'rank': i,
                        'title': metadata.get('title', 'Summary'),
                        'file': metadata.get('file_path', 'RAPTOR Summary'),
                        'score': score,
                        'excerpt': text[:1500] + "..." if len(text) > 1500 else text,
                        'source_type': 'raptor'
                    })
                result['sources'] = sources

            return result

        except Exception as e:
            logger.error(f"RAPTOR query failed: {e}", exc_info=True)
            raise RuntimeError(f"RAPTOR query failed: {e}") from e

    def get_raptor_stats(self) -> dict:
        """Get RAPTOR index statistics.

        Returns:
            Dictionary with index stats
        """
        if self.raptor_manager is None:
            return {
                "exists": False,
                "enabled": self.config.raptor.enabled,
                "error": "RAPTOR manager not initialized"
            }

        return self.raptor_manager.get_summary_stats()


def main():
    """Main entry point for CLI usage."""
    logger.info("Starting UltraRAG CLI application")

    # Check if .env file exists
    if not Path(".env").exists():
        logger.error(".env file not found")
        print("⚠️  .env file not found!")
        print("Please copy .env.example to .env and configure your settings.")
        sys.exit(1)

    # Initialize system
    try:
        rag = UltraRAG()
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}", exc_info=True)
        print(f"\n❌ Failed to initialize UltraRAG: {e}")
        print("\nPlease check:")
        print("  1. Your .env file has all required API keys")
        print("  2. Your vault path is correct")
        print("  3. Network connection is available")
        print("\nSee ultrarag.log for detailed error information.")
        sys.exit(1)

    # Check for existing index
    try:
        has_existing_index = index_exists(rag.config.vector_db, table_name=rag.config.vector_db.vault_table)
    except Exception as e:
        logger.error(f"Failed to check for existing index: {e}", exc_info=True)
        print(f"❌ Error checking for existing index: {e}")
        sys.exit(1)

    if has_existing_index:
        print("\nExisting index detected!")
        print("Options:")
        print("  1. Load existing index (recommended, fast)")
        print("  2. Create new index (slow, will prompt for overwrite)")
        print("  3. Skip and exit")

        choice = input("\nYour choice (1/2/3): ").strip()

        if choice == "1":
            try:
                if rag.load_existing_index():
                    print("\nIndex loaded successfully!")
                else:
                    print("\nFailed to load index. Exiting.")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to load index: {e}", exc_info=True)
                print(f"❌ Error loading index: {e}")
                sys.exit(1)

        elif choice == "2":
            try:
                rag.index_vault()
            except Exception as e:
                logger.error(f"Failed to create index: {e}", exc_info=True)
                print(f"\n❌ Error creating index: {e}")
                print("\nPlease check:")
                print("  1. Your vault path exists and is accessible")
                print("  2. You have permission to read the vault files")
                print("  3. Your API keys are valid")
                print("\nSee ultrarag.log for detailed error information.")
                sys.exit(1)

        elif choice == "3":
            print("\nExiting.")
            sys.exit(0)

        else:
            print("\nInvalid choice. Exiting.")
            sys.exit(1)

    else:
        print("\nNo existing index found.")
        print("\nDo you want to create an index now? (y/n): ", end="")
        choice = input().strip().lower()

        if choice == 'y':
            try:
                rag.index_vault()
            except Exception as e:
                logger.error(f"Failed to create index: {e}", exc_info=True)
                print(f"\n❌ Error creating index: {e}")
                print("\nPlease check:")
                print("  1. Your vault path exists and is accessible")
                print("  2. You have permission to read the vault files")
                print("  3. Your API keys are valid")
                print("\nSee ultrarag.log for detailed error information.")
                sys.exit(1)
        else:
            print("Skipping indexing. Configure .env and run again.")
            sys.exit(0)

    # Verify index is ready
    if rag.index is None:
        print("\nNo index available. Exiting.")
        sys.exit(1)

    # Check for conversations index (unless already auto-loaded during vault load)
    if rag.conversations_index is None:
        has_conv_index = rag.conversations_index_exists()
        if has_conv_index:
            print("\n📚 Conversations index detected!")
            if rag.load_conversations_index():
                rag._setup_federated_engine()
                print("Federated search enabled (vault + conversations)")

    # Check for RAPTOR index
    if rag.config.raptor.enabled:
        has_raptor = rag.raptor_index_exists()
        if has_raptor:
            print("\n🌳 RAPTOR index detected!")
            if rag.load_raptor_index():
                print("RAPTOR hierarchical summaries ready")

    # Interactive query loop
    print("\n" + "="*50)
    print("RAG system ready!")
    print("Commands:")
    print("  'quit' - exit")
    print("  'usage' - check token usage")
    print("  'conv' - index AI conversations")
    print("  'enrich' - run book metadata enrichment (Calibre + web)")
    print("  'raptor' - build RAPTOR hierarchical index")
    print("  'cache' - invalidate disk cache (force reload)")
    print("  '@vault <query>' - search vault only")
    print("  '@conv <query>' - search conversations only")
    print("  '@all <query>' - search both (federated)")
    print("  '@books:category <query>' - search books filtered by category")
    print("  '@raptor-books <query>' - 2-stage RAPTOR: find relevant books, then search within them")
    print("  '@research <query>' - multi-step research mode (3-5x slower, higher accuracy)")
    print("  '@raptor <query>' - search using RAPTOR hierarchical summaries")
    print("="*50 + "\n")

    while True:
        query = input("\n💭 Your query: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if query.lower() == 'usage':
            rag.print_token_usage()
            continue

        if query.lower() == 'conv':
            # Index conversations
            conv_path = input("Conversations path (or Enter for default): ").strip()
            if conv_path:
                rag.index_conversations(Path(conv_path))
            else:
                rag.index_conversations()
            continue

        if query.lower() == 'raptor':
            # Build RAPTOR index
            rag.index_raptor()
            continue

        if query.lower() == 'books':
            # Index books
            books_path = input("Books path (or Enter for default): ").strip()
            if books_path:
                rag.index_books(Path(books_path))
            else:
                rag.index_books()
            continue

        if query.lower() == 'enrich':
            # Run book metadata enrichment
            print("\nRunning book metadata enrichment...")
            try:
                stats = indexing_module.enrich_books(rag.config)
                print(f"\nEnrichment complete:")
                print(f"  Total books: {stats['total']}")
                print(f"  Calibre matched: {stats['calibre_matched']} (avg confidence {stats['avg_confidence']:.2f})")
                print(f"  Web enriched: {stats['web_enriched']}")
                print(f"  No match (filename): {stats['no_match']}")
                print(f"  Already cached: {stats['already_cached']}")
            except Exception as e:
                print(f"Enrichment failed: {e}")
            continue

        if query.lower() == 'cache':
            # Invalidate cache
            from vector_store import invalidate_cache
            if invalidate_cache():
                print("✅ Cache invalidated. Restart app to reload.")
            else:
                print("No cache to invalidate.")
            continue

        if not query:
            continue

        try:
            # Parse query modifiers
            if query.startswith('@vault '):
                query_text = query[7:]
                result = rag.query_vault_only(query_text)
                mode = "vault"
            elif query.startswith('@conv '):
                query_text = query[6:]
                result = rag.query_conversations_only(query_text)
                mode = "conversations"
            elif query.startswith('@all '):
                query_text = query[5:]
                result = rag.query_federated(query_text)
                mode = "federated"
            elif query.startswith('@research '):
                query_text = query[10:]
                print(f"\n🔬 Research mode enabled (this may take 30-60 seconds)...")
                result = rag.query_research(query_text)
                mode = "research"
            elif query.startswith('@books:'):
                # @books:category query — filter by category
                rest = query[7:]  # after "@books:"
                if ' ' in rest:
                    category, query_text = rest.split(' ', 1)
                else:
                    print("Usage: @books:category_name your query")
                    continue
                if rag.books_index is None:
                    print("Books index not loaded. Run 'books' command first.")
                    continue
                bf = BookFilter(categories=[_normalize_category(category)])
                result = rag.query_federated(query_text, source_filter=["books"], book_filter=bf)
                mode = "books"
            elif query.startswith('@books '):
                query_text = query[7:]
                if rag.books_index is None:
                    print("Books index not loaded. Run 'books' command first.")
                    continue
                result = rag.query_federated(query_text, source_filter=["books"])
                mode = "books"
            elif query.startswith('@raptor-books '):
                query_text = query[14:]
                if rag.books_index is None:
                    print("Books index not loaded. Run 'books' command first.")
                    continue
                print(f"\nRAPTOR-Books 2-stage query...")
                try:
                    from book_raptor import BookRaptorManager
                    from calibre_metadata import BookMetadataCache
                    cache_path = rag.config.books.metadata_cache_path
                    if not Path(cache_path).exists():
                        print("No metadata cache found. Run 'enrich' first.")
                        continue
                    br = BookRaptorManager(
                        embed_model=Settings.embed_model,
                        llm=Settings.llm,
                    )
                    if not br.index_exists():
                        print("Building book-summary RAPTOR index...")
                        cache = BookMetadataCache(Path(cache_path))
                        br.build_summary_index(cache)
                    else:
                        br.load_index()
                    raptor_result = br.two_stage_query(query_text, rag.books_index, rag.config)
                    result = {
                        'answer': raptor_result.answer,
                        'sources': [{'file_path': s.file_path, 'file_name': s.file_name, 'score': s.score, 'excerpt': s.excerpt, 'source_type': s.source_type} for s in raptor_result.sources],
                    }
                    mode = "raptor-books"
                except Exception as e:
                    print(f"RAPTOR-Books query failed: {e}")
                    continue
            elif query.startswith('@raptor '):
                query_text = query[8:]
                print(f"\n🌳 RAPTOR mode enabled...")
                result = rag.query_raptor(query_text)
                mode = "raptor"
            else:
                # Default: use federated if available, otherwise vault only
                if rag.federated_engine is not None:
                    result = rag.query_federated(query)
                    mode = "federated"
                else:
                    result = rag.query(query)
                    mode = "vault"

            print(f"\n📝 Answer ({mode} search):")
            print("-" * 50)
            print(result['answer'])

            # Show research summary for research mode
            if mode == "research" and 'research_summary' in result:
                print("\n🔬 Research Summary:")
                print("-" * 50)
                print(result['research_summary'])

            # Show source summary for federated queries
            if mode == "federated" and 'source_summary' in result:
                summary = result['source_summary']
                if summary:
                    by_type = summary.get('by_type', {})
                    print(f"\n📊 Sources: {by_type.get('vault', 0)} from vault, {by_type.get('conversations', 0)} from conversations")

            print("\n📚 Sources:")
            print("-" * 50)
            for source in result['sources'][:5]:
                source_type = source.get('source_type', 'vault')
                type_icon = "📓" if source_type == "vault" else "💬"
                print(f"{source['rank']}. {type_icon} {source['title']} (score: {source['score']:.3f})")
                print(f"   {source['excerpt']}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
            print("The error has been logged. You can try another query.")

    logger.info("UltraRAG CLI application shutting down")
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
