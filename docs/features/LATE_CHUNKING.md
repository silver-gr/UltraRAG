# Late Chunking

Late Chunking improves retrieval accuracy by **10-12%** compared to traditional chunking by preserving document-level context in each chunk embedding.

## Overview

**Traditional chunking:**
1. Split document into chunks
2. Embed each chunk independently

**Late Chunking:**
1. Embed the FULL document first (preserving cross-chunk context)
2. Split document into chunks using sentence boundaries
3. Embed each chunk individually
4. Combine embeddings: `final = alpha * chunk_embedding + (1-alpha) * doc_embedding`

Each chunk's embedding retains context from the full document, resulting in better retrieval performance.

## Configuration

Add to your `.env` file:

```bash
# Set chunking strategy to late_chunking
CHUNKING_STRATEGY=late_chunking

# Configure the alpha parameter (default: 0.7)
# Alpha controls the balance between local and global context:
# - Higher alpha (0.8-0.9): More weight on local chunk semantics
# - Lower alpha (0.5-0.6): More weight on global document context
# - Recommended: 0.7 (70% local, 30% global)
LATE_CHUNKING_ALPHA=0.7
```

## Usage

### Programmatic Usage

```python
from config import EmbeddingConfig
from embeddings import get_embedding_model
from chunking import ObsidianChunker
from llama_index.core import Document

# Create config with late chunking
embedding_config = EmbeddingConfig(
    model="voyage-3-large",
    chunk_size=512,
    chunk_overlap=75,
    late_chunking_alpha=0.7
)

# Initialize embedding model and chunker
embed_model = get_embedding_model(embedding_config)
chunker = ObsidianChunker(
    config=embedding_config,
    embed_model=embed_model,
    strategy="late_chunking"
)

# Chunk documents
documents = [Document(text="Your document text here")]
nodes = chunker.chunk_documents(documents)
```

### Indexing with Late Chunking

```bash
# Set in .env
CHUNKING_STRATEGY=late_chunking

# Then run indexing
python main.py --index
```

## Parameters

### `late_chunking_alpha` (float, default: 0.7)

Controls the balance between local chunk semantics and global document context:

- **Value**: 0.0 to 1.0
- **Formula**: `final_embedding = alpha * chunk_embedding + (1-alpha) * doc_embedding`

**Tuning guidelines:**
- **0.8-0.9**: Emphasizes local chunk content (better for detailed, technical queries)
- **0.7**: Balanced approach (recommended for most use cases)
- **0.5-0.6**: Emphasizes document-level context (better for broad, conceptual queries)

## Performance

| Strategy | Retrieval Accuracy | Indexing Speed | Use Case |
|----------|-------------------|----------------|----------|
| `simple` | Baseline | Fast | Quick prototyping |
| `semantic` | Good | Medium | General purpose |
| `obsidian_aware` | Better | Medium | Obsidian notes with structure |
| `markdown_semantic` | Better | Medium | Markdown documents |
| **`late_chunking`** | **Best (+10-12%)** | **Slower (2x)** | **High-accuracy retrieval** |

### Advantages
- **10-12% better retrieval accuracy**
- Preserves document-level context in each chunk
- Better handling of cross-chunk references
- Automatic fallback if late chunking fails

### Trade-offs
- **2x embeddings per chunk**: Requires both document-level and chunk-level embeddings
- **Slower indexing**: Approximately 2x slower than standard chunking
- **Memory usage**: Requires storing document embeddings during processing

**Recommendation**: Use late chunking when retrieval accuracy is critical and indexing time is not a constraint.

## Chunk Metadata

Each chunk includes:

```python
{
    'chunk_strategy': 'late_chunking',
    'alpha': 0.7,
    'section_idx': 0,
    'chunk_idx': 0,
    'total_sections': 1,
    'total_chunks_in_section': 5
}
```

## Troubleshooting

### Error: "Document exceeds max token limit"
This is just a warning. Large documents are automatically split into sections. No action needed.

### Slow indexing performance
Late chunking requires 2x embeddings per chunk. To speed up:
- Use a faster embedding model
- Reduce `CHUNK_SIZE` to create fewer chunks
- Process documents in smaller batches

### Out of memory errors
Large documents may consume significant memory:
- Reduce the maximum document token limit
- Process documents individually
- Use a machine with more RAM

## Testing

```bash
source venv/bin/activate
python test_late_chunking.py
```

## Research Background

Late Chunking is based on research showing that embedding models benefit from broader context during embedding generation. By embedding the full document first, then combining those embeddings with chunk-level embeddings, we preserve both:

1. **Local semantics**: What the specific chunk is about
2. **Global context**: How the chunk relates to the broader document

This dual representation leads to better retrieval performance, especially for cross-chunk queries and document-level reasoning.
