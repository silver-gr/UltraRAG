#!/usr/bin/env python3
"""Test script for late chunking implementation."""

import sys
from pathlib import Path
from llama_index.core import Document
from config import EmbeddingConfig
from chunking import ObsidianChunker

# Mock embedding model for testing
class MockEmbedding:
    """Mock embedding model that returns dummy embeddings."""

    def get_text_embedding(self, text: str):
        """Return a dummy embedding vector."""
        import numpy as np
        # Return a random normalized embedding
        embedding = np.random.randn(1024)
        return (embedding / np.linalg.norm(embedding)).tolist()


def test_late_chunking():
    """Test the late chunking strategy."""
    print("Testing Late Chunking Implementation...")
    print("=" * 60)

    # Create a test document
    test_text = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from and make decisions based on data. Rather than being explicitly programmed to perform a task, these systems use algorithms to identify patterns and make predictions.

## Supervised Learning

Supervised learning is the most common type of machine learning. In this approach, the algorithm learns from labeled training data, helping to predict outcomes for unforeseen data. Common supervised learning algorithms include linear regression, logistic regression, and support vector machines.

## Unsupervised Learning

Unlike supervised learning, unsupervised learning works with unlabeled data. The algorithm tries to identify patterns and relationships in the data without any predefined labels. Clustering and dimensionality reduction are common unsupervised learning techniques.

## Deep Learning

Deep learning is a subset of machine learning that uses neural networks with multiple layers. These deep neural networks can automatically learn hierarchical representations of data, making them particularly effective for tasks like image recognition and natural language processing.
    """.strip()

    # Create document
    doc = Document(
        text=test_text,
        metadata={
            'file_path': '/test/ml_intro.md',
            'file_name': 'ml_intro.md'
        }
    )

    # Create config with late chunking parameters
    config = EmbeddingConfig(
        model="test",
        chunk_size=256,
        chunk_overlap=50,
        late_chunking_alpha=0.7
    )

    # Create chunker with mock embedding model
    mock_embed_model = MockEmbedding()
    chunker = ObsidianChunker(
        config=config,
        embed_model=mock_embed_model,
        strategy="late_chunking"
    )

    # Test late chunking
    print("\nChunking document with late_chunking strategy...")
    nodes = chunker.chunk_documents([doc])

    print(f"\n✓ Successfully created {len(nodes)} chunks")

    # Verify node properties
    print("\nVerifying chunk properties:")
    for i, node in enumerate(nodes):
        print(f"\nChunk {i + 1}:")
        print(f"  - Text length: {len(node.text)} characters")
        print(f"  - Has embedding: {node.embedding is not None}")
        print(f"  - Embedding dimension: {len(node.embedding) if node.embedding else 'N/A'}")
        print(f"  - Chunk strategy: {node.metadata.get('chunk_strategy', 'N/A')}")
        print(f"  - Alpha value: {node.metadata.get('alpha', 'N/A')}")
        print(f"  - Section index: {node.metadata.get('section_idx', 'N/A')}")
        print(f"  - Chunk index: {node.metadata.get('chunk_idx', 'N/A')}")
        print(f"  - Preview: {node.text[:100]}...")

    # Validate embeddings
    print("\n" + "=" * 60)
    print("Validation:")
    all_have_embeddings = all(node.embedding is not None for node in nodes)
    all_correct_strategy = all(
        node.metadata.get('chunk_strategy') in ['late_chunking', 'late_chunking_fallback']
        for node in nodes
    )
    all_correct_alpha = all(
        node.metadata.get('alpha') == config.late_chunking_alpha
        for node in nodes
        if node.metadata.get('chunk_strategy') == 'late_chunking'
    )

    print(f"✓ All chunks have embeddings: {all_have_embeddings}")
    print(f"✓ All chunks have correct strategy: {all_correct_strategy}")
    print(f"✓ All chunks have correct alpha: {all_correct_alpha}")

    if all_have_embeddings and all_correct_strategy and all_correct_alpha:
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        return False


if __name__ == "__main__":
    try:
        success = test_late_chunking()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
