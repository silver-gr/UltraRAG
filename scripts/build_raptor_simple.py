#!/usr/bin/env python3
"""Build RAPTOR index with pre-patched stable clustering.

This script patches the llama_index RAPTOR clustering BEFORE importing,
to ensure we use stable GMM parameters and suppress the endless warnings.
"""
import sys
import warnings
from pathlib import Path

# Suppress the sklearn warnings BEFORE anything else
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch the clustering module BEFORE importing raptor_index
def patch_clustering_early():
    """Patch clustering before it's imported elsewhere."""
    import numpy as np
    from sklearn.mixture import GaussianMixture

    RANDOM_SEED = 224

    def get_optimal_clusters_stable(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
    ) -> int:
        """Stable version with regularization and lower max clusters."""
        # Limit max clusters more aggressively
        max_clusters = min(20, max_clusters, len(embeddings) // 3 + 1)
        n_clusters = np.arange(1, max(2, max_clusters))
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(
                n_components=n,
                random_state=random_state,
                reg_covar=1e-3,  # Higher regularization
                max_iter=100,
                n_init=1,
                covariance_type='diag'  # Simpler covariance for stability
            )
            try:
                gm.fit(embeddings)
                bics.append(gm.bic(embeddings))
            except Exception:
                bics.append(float('inf'))
        return n_clusters[np.argmin(bics)]

    def GMM_cluster_stable(embeddings: np.ndarray, threshold: float, random_state: int = 0):
        """Stable GMM clustering."""
        n_clusters = get_optimal_clusters_stable(embeddings)
        gm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            reg_covar=1e-3,
            max_iter=100,
            n_init=1,
            covariance_type='diag'
        )
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters

    # Import and patch the module
    try:
        from llama_index.packs.raptor import clustering
        clustering.get_optimal_clusters = get_optimal_clusters_stable
        clustering.GMM_cluster = GMM_cluster_stable
        print("Successfully patched RAPTOR clustering (diag covariance, reg=1e-3)")
        return True
    except Exception as e:
        print(f"Failed to patch clustering: {e}")
        return False


# Apply patch FIRST
if not patch_clustering_early():
    print("Warning: Clustering patch failed, proceeding anyway...")

# NOW import everything else
import json
import logging

from dotenv import load_dotenv
load_dotenv()

from config import load_config
from loader import ObsidianLoader
from embeddings import get_embedding_model
from raptor_index import RaptorIndexManager
from llama_index.llms.google_genai import GoogleGenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_filtered_documents(loader, note_paths):
    """Load specific documents from the vault."""
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
    documents = loader.notes_to_documents(notes)
    print(f"Converted to {len(documents)} documents")
    return documents


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Build RAPTOR index')
    parser.add_argument('notes_file', type=Path, help='JSON file with scored notes')
    parser.add_argument('--force', action='store_true', help='Force rebuild')

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

    # Initialize config
    config = load_config()
    print(f"\nConfig: LLM={config.llm.model}, Embedding={config.embedding.model}")

    # Initialize components
    print("\nInitializing components...")
    embed_model = get_embedding_model(config.embedding)

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

    # Load documents
    loader = ObsidianLoader(config.vault_path)
    documents = load_filtered_documents(loader, note_paths)

    if not documents:
        print("No documents loaded!")
        sys.exit(1)

    # Initialize RAPTOR
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

    # Build index
    print(f"\nBuilding RAPTOR index from {len(documents)} documents...")
    print("This may take several minutes...")
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
            print(f"  Nodes: {stats.get('node_count', 'unknown')}")
            print(f"  Mode: {stats.get('default_mode', 'unknown')}")
        else:
            print("\nFailed to build RAPTOR index.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"RAPTOR indexing failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
