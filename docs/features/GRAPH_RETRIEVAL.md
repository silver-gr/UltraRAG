# Wikilink Graph-Enhanced Retrieval Implementation

## Summary

Successfully implemented wikilink graph-enhanced retrieval for UltraRAG, which expands search results by following wikilink connections in Obsidian notes. This improves retrieval quality by including contextually related documents that are linked to the initially retrieved results.

## Implementation Details

### 1. Configuration (`config.py`)

**Added to `RetrievalConfig` class (lines 50-60):**
- `enable_graph_retrieval: bool = Field(default=True)` - Enable/disable graph-enhanced retrieval
- `graph_retrieval_depth: int = Field(default=1)` - Number of link hops to traverse (currently 1-hop implemented)
- `graph_retrieval_max_links: int = Field(default=3)` - Maximum number of links to follow per retrieved node

**Environment variable support (lines 192-194):**
```python
enable_graph_retrieval=os.getenv("ENABLE_GRAPH_RETRIEVAL", "true").lower() == "true",
graph_retrieval_depth=int(os.getenv("GRAPH_RETRIEVAL_DEPTH", "1")),
graph_retrieval_max_links=int(os.getenv("GRAPH_RETRIEVAL_MAX_LINKS", "3"))
```

### 2. GraphEnhancedRetriever Class (`query_engine.py`)

**New class added (lines 136-226):**

```python
class GraphEnhancedRetriever(BaseRetriever):
    """Retriever that expands results using wikilink graph connections."""
```

**Key Features:**
- Wraps any base retriever (vector, BM25, or fusion)
- Receives wikilink graph dictionary mapping file paths to linked file paths
- Expands initial retrieval results by following wikilinks

**Algorithm (lines 164-226):**
1. Get base results from underlying retriever (vector/hybrid)
2. Extract file paths from retrieved nodes
3. For each retrieved node, look up connected notes in wikilink graph
4. Limit to `max_links` (default: 3) connections per node
5. Retrieve chunks from linked documents via docstore
6. Assign lower scores to linked nodes (base_min_score - 0.1)
7. Combine base results + linked results

**Score Assignment:**
- Base retriever results keep original scores
- Linked nodes get score = min(base_scores) - 0.1
- Ensures base results rank higher than graph-expanded results

### 3. HybridQueryEngine Integration (`query_engine.py`)

**Updated constructor (lines 337-360):**
- Added `wikilink_graph` parameter
- Defaults to empty dict if not provided

**Updated `_build_query_engine` method (lines 373-414):**
```python
# Wrap with graph-enhanced retriever if enabled and graph is available
if self.config.retrieval.enable_graph_retrieval and self.wikilink_graph:
    logger.info(f"Graph-enhanced retrieval enabled (depth={...}, max_links={...})")
    retriever = GraphEnhancedRetriever(
        base_retriever=fusion_retriever,
        wikilink_graph=self.wikilink_graph,
        index=self.index,
        depth=self.config.retrieval.graph_retrieval_depth,
        max_links=self.config.retrieval.graph_retrieval_max_links
    )
else:
    retriever = fusion_retriever
```

### 4. Main Integration (`main.py`)

**Required change in `_setup_query_engine` method (around lines 422-430):**

The wikilink_graph needs to be passed to HybridQueryEngine:

```python
# BEFORE (original code):
self.query_engine = HybridQueryEngine(
    index=self.index,
    config=self.config,
    reranker=self.reranker,
    bm25_retriever=self.bm25_retriever,
    nodes=self.nodes
)

# AFTER (with graph support):
wikilink_graph = getattr(self, 'wikilink_graph', None)
self.query_engine = HybridQueryEngine(
    index=self.index,
    config=self.config,
    reranker=self.reranker,
    bm25_retriever=self.bm25_retriever,
    nodes=self.nodes,
    wikilink_graph=wikilink_graph  # NEW: Pass wikilink graph
)
```

**Note:** A patch script has been created at `/Users/silver/Projects/UltraRAG/apply_graph_patch.py` to apply this change automatically if needed.

## File Modifications Summary

### `/Users/silver/Projects/UltraRAG/config.py`
- **Line 58:** Added `enable_graph_retrieval: bool = Field(default=True)`
- **Lines 59-60:** Added `graph_retrieval_depth` and `graph_retrieval_max_links` config fields
- **Lines 192-194:** Added environment variable loading for new config options

### `/Users/silver/Projects/UltraRAG/query_engine.py`
- **Lines 1-11:** Updated imports to include `Dict`, `Set`, `BaseRetriever`, `QueryBundle`
- **Lines 136-226:** Added complete `GraphEnhancedRetriever` class implementation
- **Lines 337-360:** Updated `HybridQueryEngine.__init__` to accept `wikilink_graph` parameter
- **Lines 373-414:** Updated `_build_query_engine` to conditionally wrap retriever with graph enhancement
- **Line 443:** Changed from `fusion_retriever` to `retriever` variable in query engine construction

### `/Users/silver/Projects/UltraRAG/main.py`
- **Needs manual update:** Lines 422-430 in `_setup_query_engine` method
- Alternative: Run `/Users/silver/Projects/UltraRAG/apply_graph_patch.py` to apply automatically

## Configuration Options

Add to `.env` file (all optional, showing defaults):

```bash
# Enable wikilink graph-enhanced retrieval
ENABLE_GRAPH_RETRIEVAL=true

# Number of hops to traverse (1 = direct links only)
GRAPH_RETRIEVAL_DEPTH=1

# Maximum links to follow per retrieved node
GRAPH_RETRIEVAL_MAX_LINKS=3
```

## How It Works - Example

**Query:** "What are the benefits of meditation?"

**Step 1 - Base Retrieval:**
- Vector/Hybrid retriever returns 10 most relevant chunks
- Top result from: `Meditation Practices.md`
- Score: 0.85

**Step 2 - Graph Expansion:**
- Check `Meditation Practices.md` in wikilink graph
- Find linked notes: `[[Mindfulness]]`, `[[Stress Management]]`, `[[Sleep Quality]]`
- Limit to 3 links (GRAPH_RETRIEVAL_MAX_LINKS)

**Step 3 - Retrieve Linked Content:**
- Fetch all chunks from linked documents
- Assign score: 0.74 (min base score - 0.1)

**Step 4 - Combined Results:**
- Original 10 chunks from base retrieval
- Additional chunks from 3 linked documents
- All passed to reranker and LLM for final answer

## Benefits

1. **Contextual Expansion:** Retrieves related concepts even if not directly matching query
2. **Knowledge Graph Traversal:** Leverages existing wikilink structure in Obsidian vault
3. **Configurable:** Can adjust depth and link limits based on needs
4. **Non-Breaking:** Gracefully degrades if no wikilink graph available
5. **Preserves Ranking:** Original retrieval results rank higher than graph expansions

## Performance Considerations

- Graph expansion adds minimal overhead (dictionary lookups + docstore access)
- Controlled by `max_links` parameter to prevent retrieval explosion
- Only active when `enable_graph_retrieval=true` and wikilink graph exists
- Works with any base retriever (vector, BM25, hybrid fusion)

## Testing

To test the implementation:

1. Ensure wikilink graph is built during indexing (already implemented in main.py)
2. Run a query on notes with wikilinks
3. Check logs for: `"Graph expansion: X base nodes -> Y linked documents"`
4. Verify linked documents appear in source references

## Future Enhancements

- Multi-hop traversal (depth > 1) with decay scoring
- Bidirectional link following (backlinks)
- Link weight based on note importance/centrality
- Filtering by link type or note tags
- Caching of graph traversal results
