"""Main RAG system orchestrator."""
import os

# Suppress tokenizers parallelism warning when forking processes
# Must be set before importing transformers/tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import sys
import json
import gc
from pathlib import Path
from typing import Optional, List, Set
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from gemini_cli import GeminiCLI
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultrarag.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


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
                self.llm = GeminiCLI(
                    model=self.config.llm.model,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens
                )
                logger.info("Using Gemini CLI backend (free tier: 1000 requests/day)")
            else:
                # Use Google Gemini API directly
                google_key = self.config.google_api_key.get_secret_value()
                if not google_key:
                    logger.error("Google API key not found")
                    raise ValueError(
                        "GOOGLE_API_KEY not found. Please set it in your .env file.\n"
                        "Get your API key from: https://makersuite.google.com/app/apikey"
                    )

                self.llm = GoogleGenAI(
                    model=self.config.llm.model,
                    api_key=google_key,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens
                )

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
                if index_exists(self.config.vector_db):
                    mode = "append"
                    logger.info("Existing index found, using append mode")
                else:
                    mode = "create"
                    logger.info("No existing index found, using create mode")

            self.vector_store = get_vector_store(self.config.vector_db, mode=mode)
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
                embed_model=self.embed_model
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
        if not index_exists(self.config.vector_db):
            return False

        try:
            print("\n=== Loading Existing Index ===")
            print("Loading vector index from storage...")

            self.index = load_vector_index(
                vector_store=self.vector_store,
                embed_model=self.embed_model,
                config=self.config.vector_db
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

            # Load wikilink graph if available
            try:
                notes = self.loader.load_vault()
                self.wikilink_graph = self.loader.build_wikilink_graph(notes)
                print(f"Wikilink graph loaded with {len(self.wikilink_graph)} nodes")
            except Exception as e:
                print(f"Could not load wikilink graph: {e}")
                self.wikilink_graph = {}

            # Setup query engine
            self._setup_query_engine()

            # Auto-load conversations index if enabled
            if self.config.conversations.enabled and self.config.conversations.path:
                self._auto_load_conversations()

            return True

        except Exception as e:
            print(f"Failed to load existing index: {e}")
            return False

    def _auto_load_conversations(self):
        """Auto-load or index conversations if enabled in config."""
        conv_path = self.config.conversations.path

        if not conv_path or not conv_path.exists():
            logger.info(f"Conversations path not found: {conv_path}")
            return

        # Check if conversations index exists
        if self.conversations_index_exists():
            print("\n📚 Loading conversations index...")
            if self.load_conversations_index():
                self._setup_federated_engine()
                print("✅ Federated search enabled (vault + conversations)")
        else:
            # Auto-index conversations (non-interactive mode)
            print(f"\n📚 Auto-indexing conversations from: {conv_path}")
            self.index_conversations(conv_path, force_reindex=False, interactive=False)

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
        if not force_reindex and index_exists(self.config.vector_db):
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
        # Load all notes for graph building
        all_notes = self.loader.load_vault()
        self.wikilink_graph = self.loader.build_wikilink_graph(all_notes)
        print(f"Graph contains {len(self.wikilink_graph)} nodes")

        # Setup query engine
        self._setup_query_engine()
    
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
        else:
            print("Using standard query engine")
            self.query_engine = RAGQueryEngine(
                index=self.index,
                config=self.config,
                reranker=self.reranker,
                query_transformer=self.query_transformer  # Pass query transformer
            )
    
    def query(self, query_str: str, return_sources: bool = True):
        """Query the RAG system."""
        if self.query_engine is None:
            logger.error("Query attempted before system initialization")
            raise ValueError("System not initialized. Run index_vault() or load_existing_index() first.")

        logger.info(f"Processing query: {query_str[:100]}...")
        print(f"\n🔍 Query: {query_str}")
        print("Searching knowledge base...\n")

        try:
            response = self.query_engine.query(query_str)
            logger.info(f"Query successful, {len(response.source_nodes)} sources found")

            if return_sources:
                return {
                    'answer': str(response),
                    'sources': self._format_sources(response.source_nodes),
                    'raw_response': response
                }

            return str(response)

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            # Re-raise the exception with context
            raise RuntimeError(f"Query execution failed: {e}") from e
    
    def _format_sources(self, source_nodes):
        """Format source nodes for display."""
        sources = []
        for idx, node in enumerate(source_nodes, 1):
            sources.append({
                'rank': idx,
                'title': node.metadata.get('title', 'Unknown'),
                'file': node.metadata.get('file_name', 'Unknown'),
                'score': node.score,
                'excerpt': node.text[:200] + "...",
                'source_type': node.metadata.get('source_type', 'vault'),
                'retrieval_source': node.metadata.get('retrieval_source', 'vault')
            })
        return sources
    
    def search_notes(self, query_str: str, top_k: int = 10):
        """Search for relevant notes without generation."""
        if self.index is None:
            raise ValueError("Index not created. Run index_vault() first.")

        engine = RAGQueryEngine(
            index=self.index,
            config=self.config,
            reranker=self.reranker
        )

        nodes = engine.get_relevant_nodes(query_str, top_k=top_k)
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
        existing_tables = db.table_names()
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
            return table_name in db.table_names()
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

            # Get nodes for BM25
            docstore = self.conversations_index.docstore
            self.conversations_nodes = list(docstore.docs.values())
            print(f"Loaded {len(self.conversations_nodes)} conversation nodes")

            return True

        except Exception as e:
            logger.error(f"Failed to load conversations index: {e}", exc_info=True)
            return False

    def _setup_federated_engine(self):
        """Setup federated query engine for both indexes."""
        if self.index is None:
            logger.warning("Vault index not available for federated engine")
            return

        sources = []

        # Vault source
        sources.append(IndexSource(
            name="vault",
            index=self.index,
            source_type="vault",
            weight=1.0,
            nodes=self.nodes,
            wikilink_graph=getattr(self, 'wikilink_graph', {})
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

        if len(sources) > 1:
            self.federated_engine = FederatedQueryEngine(
                sources=sources,
                config=self.config,
                reranker=self.reranker,
                query_transformer=self.query_transformer
            )
            print(f"Federated engine ready with {len(sources)} sources")
        else:
            logger.info("Only one source available, federated engine not needed")

    def query_federated(
        self,
        query_str: str,
        source_filter: Optional[List[str]] = None,
        return_sources: bool = True
    ):
        """Query both vault and conversations with federated retrieval.

        Args:
            query_str: Query string
            source_filter: Optional list of sources ("vault", "conversations")
            return_sources: Include source information in response
        """
        if self.federated_engine is None:
            # Fallback to regular query if no federated engine
            if self.conversations_index is not None and self.index is not None:
                self._setup_federated_engine()

            if self.federated_engine is None:
                logger.warning("Federated engine not available, using standard query")
                return self.query(query_str, return_sources=return_sources)

        print(f"\n🔍 Federated Query: {query_str}")
        print("Searching vault and conversations...\n")

        try:
            response = self.federated_engine.query(
                query_str,
                source_filter=source_filter
            )

            if return_sources:
                # Include source summary
                source_summary = response.metadata.get('source_summary', {}) if hasattr(response, 'metadata') else {}

                return {
                    'answer': str(response),
                    'sources': self._format_sources(response.source_nodes),
                    'source_summary': source_summary,
                    'raw_response': response
                }

            return str(response)

        except Exception as e:
            logger.error(f"Federated query failed: {e}", exc_info=True)
            raise RuntimeError(f"Federated query failed: {e}") from e

    def query_vault_only(self, query_str: str, return_sources: bool = True):
        """Query only the vault index (exclude conversations)."""
        return self.query_federated(
            query_str,
            source_filter=["vault"],
            return_sources=return_sources
        )

    def query_conversations_only(self, query_str: str, return_sources: bool = True):
        """Query only the conversations index."""
        return self.query_federated(
            query_str,
            source_filter=["conversations"],
            return_sources=return_sources
        )

    def query_research(self, query_str: str, return_sources: bool = True):
        """Execute multi-step research mode for complex queries.

        Research mode performs iterative retrieval with gap analysis and query refinement.
        This is 3-5x slower but provides 141% accuracy improvement (based on Khoj benchmarks).

        Args:
            query_str: User query
            return_sources: Whether to return source nodes (default: True)

        Returns:
            Dictionary with answer, sources, and research summary
        """
        if not self.query_engine:
            raise RuntimeError("Query engine not initialized. Please run index_vault() or load_existing_index() first.")

        logger.info(f"Research mode query: {query_str[:100]}...")

        try:
            # Import research module
            from research_mode import ResearchRetriever

            # Get the base retriever from query engine
            # For HybridQueryEngine, use the hybrid retriever
            # For RAGQueryEngine, use the vector retriever
            if hasattr(self.query_engine, 'query_engine'):
                base_retriever = self.query_engine.query_engine._retriever
            else:
                # Fallback: create a simple vector retriever
                from llama_index.core.retrievers import VectorIndexRetriever
                base_retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=self.config.retrieval.top_k
                )

            # Create research retriever
            research_retriever = ResearchRetriever(
                base_retriever=base_retriever,
                llm=self.llm,
                max_iterations=self.config.retrieval.research_max_iterations,
                confidence_threshold=self.config.retrieval.research_confidence_threshold,
                max_subqueries=self.config.retrieval.research_max_subqueries,
                enable_research=self.config.retrieval.enable_research_mode
            )

            # Execute research
            research_result = research_retriever.research(query_str)

            logger.info(
                f"Research completed: {research_result.total_iterations} iterations, "
                f"{research_result.total_nodes_retrieved} nodes, "
                f"confidence={research_result.final_confidence:.2f}"
            )

            # Generate final answer using retrieved context
            from llama_index.core.response_synthesizers import get_response_synthesizer
            from llama_index.core.prompts import PromptTemplate
            from query_engine import PTCF_TEMPLATE

            response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                text_qa_template=PromptTemplate(PTCF_TEMPLATE),
                use_async=False
            )

            # Synthesize response from research results
            response = response_synthesizer.synthesize(
                query=query_str,
                nodes=research_result.final_nodes
            )

            # Format result
            result = {
                'answer': response.response,
                'research_summary': research_result.get_iteration_summary()
            }

            if return_sources:
                sources = []
                for i, node in enumerate(research_result.final_nodes[:10], 1):
                    sources.append({
                        'rank': i,
                        'title': node.metadata.get('title', 'Unknown'),
                        'file_path': node.metadata.get('file_path', 'Unknown'),
                        'score': node.score or 0.0,
                        'excerpt': node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text
                    })
                result['sources'] = sources

            return result

        except Exception as e:
            logger.error(f"Research mode query failed: {e}", exc_info=True)
            raise RuntimeError(f"Research mode query failed: {e}") from e

    def get_token_usage(self) -> dict:
        """Get current Voyage AI token usage statistics."""
        return self.token_tracker.get_status()

    def print_token_usage(self):
        """Print formatted token usage status."""
        self.token_tracker.print_status()


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
        has_existing_index = index_exists(rag.config.vector_db)
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

    # Check for conversations index
    has_conv_index = rag.conversations_index_exists()
    if has_conv_index:
        print("\n📚 Conversations index detected!")
        if rag.load_conversations_index():
            rag._setup_federated_engine()
            print("Federated search enabled (vault + conversations)")

    # Interactive query loop
    print("\n" + "="*50)
    print("RAG system ready!")
    print("Commands:")
    print("  'quit' - exit")
    print("  'usage' - check token usage")
    print("  'conv' - index AI conversations")
    print("  '@vault <query>' - search vault only")
    print("  '@conv <query>' - search conversations only")
    print("  '@all <query>' - search both (federated)")
    print("  '@research <query>' - multi-step research mode (3-5x slower, higher accuracy)")
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
