# Contextual Retrieval Implementation

## Overview

This implementation adds **Anthropic's Contextual Retrieval** technique to UltraRAG, achieving **67% fewer retrieval failures** according to Anthropic's research.

## What is Contextual Retrieval?

Contextual Retrieval is a technique that enhances chunk embeddings by prepending LLM-generated context to each chunk before embedding. This gives the embeddings more semantic context about what the chunk discusses and its relevance within the document.

### Example

**Original Chunk:**
```
The company's revenue grew by 3% over the previous quarter.
```

**Enhanced Chunk (with context):**
```
This chunk discusses the financial performance of the company in Q2 2023.
It specifically focuses on revenue growth metrics compared to Q1 2023.

The company's revenue grew by 3% over the previous quarter.
```

The enhanced chunk now has explicit context that improves retrieval accuracy.

## Implementation Details

### Location
- **File**: `/Users/silver/Projects/UltraRAG/chunking.py`
- **Class**: `ObsidianChunker`

### Key Components

#### 1. New Parameters

```python
ObsidianChunker(
    config: EmbeddingConfig,
    embed_model: BaseEmbedding,
    strategy: str = "obsidian_aware",
    use_contextual_retrieval: bool = True,  # NEW: Enable/disable contextual retrieval
    llm: Optional[LLM] = None                # NEW: LLM for generating context
)
```

#### 2. Core Method: `_add_contextual_retrieval`

This method:
- Takes a list of `TextNode` objects
- Processes them in batches (batch_size=10) for efficiency
- Uses async/await for parallel LLM calls
- Generates 2-3 sentence context for each chunk
- Prepends context to the original chunk text
- Stores original text in metadata for display

```python
def _add_contextual_retrieval(self, nodes: List[TextNode], llm: LLM) -> List[TextNode]:
    """Add contextual retrieval to chunks using LLM-generated context."""
```

#### 3. Context Generation: `_generate_context_for_node`

This async method:
- Extracts document metadata (title, path)
- Creates a prompt for the LLM
- Generates 2-3 sentence context
- Returns enhanced node with context prepended

```python
async def _generate_context_for_node(self, node: TextNode, llm: LLM) -> TextNode:
    """Generate contextual information for a single node."""
```

### Prompt Template

```
Given this document excerpt from "{doc_title}", provide 2-3 sentences of context
explaining what this chunk discusses and its relevance within the document.
Be concise and specific.

Document Path: {doc_path}

Chunk text:
{chunk_text}

Context:
```

## Integration

Contextual retrieval is automatically applied to **all chunking strategies**:

1. **Obsidian-aware chunking** (`_obsidian_aware_chunking`)
2. **Markdown-semantic chunking** (`_markdown_semantic_chunking`)
3. **Semantic chunking** (`_semantic_chunking`)
4. **Markdown chunking** (`_markdown_chunking`)
5. **Simple chunking** (`_simple_chunking`)

Each strategy now includes:
```python
# Apply contextual retrieval if enabled
if self.use_contextual_retrieval and self.llm:
    nodes = self._add_contextual_retrieval(nodes, self.llm)
```

## Main.py Integration

The chunker is instantiated with contextual retrieval enabled:

```python
chunker = ObsidianChunker(
    config=self.config.embedding,
    embed_model=self.embed_model,
    strategy="markdown_semantic",
    use_contextual_retrieval=True,  # Enable contextual retrieval
    llm=self.llm                     # Pass LLM instance
)
```

## Performance Optimizations

### Batch Processing
- Processes chunks in batches of 10
- Reduces API overhead
- Parallel async LLM calls using `asyncio.gather()`

### Error Handling
- Graceful degradation: If context generation fails, uses original chunk
- Per-chunk error handling with logging
- Per-batch error handling with fallback to original chunks

### Memory Efficiency
- Processes in batches to avoid memory spikes
- Stores only essential metadata
- Original text preserved in `metadata['original_text']`

## Metadata Structure

Each enhanced node includes:

```python
{
    'original_text': str,        # Original chunk text (for display)
    'contextual_prefix': str,    # Generated context
    'file_name': str,            # From original metadata
    'file_path': str,            # From original metadata
    ...                          # Other original metadata
}
```

## Usage

### Enable (Default)
```python
chunker = ObsidianChunker(
    config=config,
    embed_model=embed_model,
    use_contextual_retrieval=True,  # Enabled by default
    llm=llm
)
```

### Disable
```python
chunker = ObsidianChunker(
    config=config,
    embed_model=embed_model,
    use_contextual_retrieval=False,  # Disable contextual retrieval
    llm=None  # LLM not needed if disabled
)
```

## Expected Results

According to Anthropic's research, Contextual Retrieval should achieve:

- **67% reduction in retrieval failures**
- Better semantic understanding of chunks
- Improved relevance of retrieved results
- More accurate answers to user queries

## Logging

The implementation includes comprehensive logging:

```
INFO: Adding contextual retrieval to 150 chunks...
INFO: Successfully enhanced 150 chunks with contextual retrieval
```

Error cases are also logged:
```
ERROR: Error generating context for chunk: {error}
ERROR: Error processing batch 5: {error}
```

## Future Enhancements

Potential improvements:

1. **Configurable batch size** via `EmbeddingConfig`
2. **Caching** of generated contexts to avoid re-generation
3. **Different prompt templates** for different document types
4. **Context length optimization** based on chunk size
5. **A/B testing framework** to measure improvement

## References

- [Anthropic's Contextual Retrieval Blog Post](https://www.anthropic.com/news/contextual-retrieval)
- Original research showing 67% reduction in retrieval failures
