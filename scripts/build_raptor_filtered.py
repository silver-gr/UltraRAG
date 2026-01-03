#!/usr/bin/env python3
"""Build RAPTOR index from a filtered set of notes.

Usage:
    python scripts/build_raptor_filtered.py data/top_150_notes.json

This script:
1. Reads a JSON file with scored notes (from score_notes.py)
2. Loads only those specific documents
3. Builds the RAPTOR hierarchical index
"""
import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config import load_config
from loader import ObsidianLoader
from embeddings import get_embedding_model
from raptor_index import RaptorIndexManager
from llama_index.llms.google_genai import GoogleGenAI


def patch_raptor_clustering():
    """Patch RAPTOR clustering to use more stable GMM parameters.

    The default llama_index RAPTOR uses GMM without regularization,
    which can fail with high-dimensional embeddings. We add reg_covar
    to stabilize the covariance matrices.
    """
    try:
        from llama_index.packs.raptor import clustering
        from sklearn.mixture import GaussianMixture
        import numpy as np

        RANDOM_SEED = 224

        def get_optimal_clusters_stable(
            embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
        ) -> int:
            """Stable version with regularization."""
            max_clusters = min(max_clusters, len(embeddings))
            n_clusters = np.arange(1, max_clusters)
            bics = []
            for n in n_clusters:
                gm = GaussianMixture(
                    n_components=n,
                    random_state=random_state,
                    reg_covar=1e-4,  # Add regularization for stability
                    max_iter=200,
                    n_init=3
                )
                try:
                    gm.fit(embeddings)
                    bics.append(gm.bic(embeddings))
                except Exception:
                    bics.append(float('inf'))  # Skip failed fits
            return n_clusters[np.argmin(bics)]

        def GMM_cluster_stable(embeddings: np.ndarray, threshold: float, random_state: int = 0):
            """Stable GMM clustering with regularization."""
            n_clusters = get_optimal_clusters_stable(embeddings)
            gm = GaussianMixture(
                n_components=n_clusters,
                random_state=random_state,
                reg_covar=1e-4,  # Add regularization for stability
                max_iter=200,
                n_init=3
            )
            gm.fit(embeddings)
            probs = gm.predict_proba(embeddings)
            labels = [np.where(prob > threshold)[0] for prob in probs]
            return labels, n_clusters

        # Monkey-patch the clustering module
        clustering.get_optimal_clusters = get_optimal_clusters_stable
        clustering.GMM_cluster = GMM_cluster_stable

        print("Applied RAPTOR clustering stability patch (reg_covar=1e-4)")
        return True

    except Exception as e:
        print(f"Warning: Could not patch RAPTOR clustering: {e}")
        return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_filtered_documents(loader: ObsidianLoader, note_paths: list) -> list:
    """Load specific documents from the vault.

    Args:
        loader: ObsidianLoader instance
        note_paths: List of file paths to load

    Returns:
        List of LlamaIndex Documents
    """
    from tqdm import tqdm

    notes = []
    print(f"Loading {len(note_paths)} specific notes...")

    for path_str in tqdm(note_paths, desc="Loading notes"):
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"Note not found: {path}")
            continue

        note = loader.load_note(path)
        if note:
            notes.append(note)

    print(f"Successfully loaded {len(notes)} notes")

    # Convert to documents
    documents = loader.notes_to_documents(notes)
    print(f"Converted to {len(documents)} documents")

    return documents


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build RAPTOR index from filtered notes')
    parser.add_argument('notes_file', type=Path, help='JSON file with scored notes')
    parser.add_argument('--force', action='store_true', help='Force rebuild even if index exists')

    args = parser.parse_args()

    if not args.notes_file.exists():
        print(f"Error: Notes file not found: {args.notes_file}")
        sys.exit(1)

    # Load scored notes
    print(f"Reading notes from: {args.notes_file}")
    with open(args.notes_file) as f:
        scored_notes = json.load(f)

    note_paths = [note['path'] for note in scored_notes]
    print(f"Found {len(note_paths)} notes to index")

    # Print sample
    print("\nTop 10 notes by value score:")
    for i, note in enumerate(scored_notes[:10], 1):
        print(f"  {i}. {note['title'][:50]:50} score={note['score']:.3f}")

    # Initialize config
    config = load_config()
    print(f"\nConfig loaded:")
    print(f"  LLM: {config.llm.model}")
    print(f"  Embedding: {config.embedding.model}")
    print(f"  RAPTOR path: {config.raptor.raptor_path}")

    # Initialize components
    print("\nInitializing embedding model...")
    embed_model = get_embedding_model(config.embedding)

    print("Initializing LLM...")
    import os
    # Handle SecretStr type for API key
    google_key = None
    if config.google_api_key:
        google_key = config.google_api_key.get_secret_value()
    if not google_key:
        google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("Error: GOOGLE_API_KEY not found")
        sys.exit(1)

    llm = GoogleGenAI(
        model=config.llm.model,
        api_key=google_key,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
    )

    # Initialize loader
    loader = ObsidianLoader(config.vault_path)

    # Load filtered documents
    documents = load_filtered_documents(loader, note_paths)

    if not documents:
        print("No documents loaded!")
        sys.exit(1)

    # Apply clustering stability patch before building RAPTOR
    patch_raptor_clustering()

    # Initialize RAPTOR manager
    print("\nInitializing RAPTOR manager...")
    raptor_manager = RaptorIndexManager(
        embed_model=embed_model,
        llm=llm,
        raptor_path=config.raptor.raptor_path,
        chunk_size=config.raptor.chunk_size,
        chunk_overlap=config.raptor.chunk_overlap,
        similarity_top_k=config.raptor.similarity_top_k,
        default_mode=config.raptor.mode
    )

    # Check existing index
    if raptor_manager.index_exists() and not args.force:
        print("\nRAPTOR index already exists!")
        stats = raptor_manager.get_summary_stats()
        print(f"  Nodes in tree: {stats.get('node_count', 'unknown')}")
        print("\nUse --force to rebuild.")
        sys.exit(0)

    # Build RAPTOR index
    print(f"\nBuilding RAPTOR index from {len(documents)} documents...")
    print("This may take several minutes (LLM generates cluster summaries)")
    print("-" * 60)

    try:
        success = raptor_manager.build_index(
            documents=documents,
            force_rebuild=args.force
        )

        if success:
            print("-" * 60)
            print("\nRAPTOR index built successfully!")

            stats = raptor_manager.get_summary_stats()
            print(f"  Nodes in tree: {stats.get('node_count', 'unknown')}")
            print(f"  Default mode: {stats.get('default_mode', 'unknown')}")
            print(f"  Index path: {stats.get('path', 'unknown')}")
        else:
            print("\nFailed to build RAPTOR index.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"RAPTOR indexing failed: {e}", exc_info=True)
        print(f"\nError building RAPTOR index: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
