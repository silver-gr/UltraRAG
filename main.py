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

            return True

        except Exception as e:
            print(f"Failed to load existing index: {e}")
            return False

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
                'excerpt': node.text[:200] + "..."
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

    # Interactive query loop
    print("\n" + "="*50)
    print("RAG system ready!")
    print("Commands: 'quit' to exit, 'usage' to check token usage")
    print("="*50 + "\n")

    while True:
        query = input("\n💭 Your query: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if query.lower() == 'usage':
            rag.print_token_usage()
            continue

        if not query:
            continue

        try:
            result = rag.query(query)

            print("\n📝 Answer:")
            print("-" * 50)
            print(result['answer'])

            print("\n📚 Sources:")
            print("-" * 50)
            for source in result['sources'][:5]:
                print(f"{source['rank']}. {source['title']} (score: {source['score']:.3f})")
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
