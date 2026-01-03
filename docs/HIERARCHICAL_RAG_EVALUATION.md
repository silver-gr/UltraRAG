# Hierarchical RAG Approaches: Evaluation Report

**Date:** January 2026
**Context:** UltraRAG system for Obsidian vault (2,371 notes)
**Current Setup:** Hybrid search (dense + sparse) + Voyage reranking

---

## Executive Summary

This report documents our evaluation of hierarchical RAG approaches (RAPTOR and GraphRAG) as potential enhancements to UltraRAG. After extensive testing, we found that:

1. **RAPTOR** is fundamentally incompatible with high-dimensional Voyage embeddings due to sklearn GMM clustering limitations
2. **GraphRAG** is technically viable but extremely LLM-token intensive
3. **Current hybrid RAG** with reranking provides excellent quality without hierarchical indexing overhead

**Recommendation:** Keep hierarchical approaches in reserve. Deploy only if retrieval quality proves insufficient for specific use cases.

---

## Table of Contents

1. [Background: What Problem Do Hierarchical Approaches Solve?](#background)
2. [RAPTOR: Recursive Abstractive Processing](#raptor)
3. [GraphRAG: Knowledge Graph Approach](#graphrag)
4. [Lightweight Alternatives](#lightweight-alternatives)
5. [Token Cost Analysis](#token-cost-analysis)
6. [Decision Framework](#decision-framework)
7. [Implementation Artifacts](#implementation-artifacts)
8. [Lessons Learned](#lessons-learned)

---

## Background: What Problem Do Hierarchical Approaches Solve? {#background}

### The Multi-Hop Reasoning Problem

Standard RAG retrieves chunks based on query similarity. This works well for:
- Direct factual questions ("What is X?")
- Single-document answers

It struggles with:
- **Multi-document synthesis** ("How do concepts A and B relate across my notes?")
- **Thematic queries** ("What are my main ideas about topic X?")
- **Abstractive questions** ("Summarize my thinking on Y")

### How Hierarchical Approaches Help

Both RAPTOR and GraphRAG create **higher-level abstractions** that capture relationships across documents:

```
Standard RAG:
Query → Similar Chunks → Answer

Hierarchical RAG:
Query → [Summary Nodes / Community Summaries] + Chunks → Richer Answer
```

---

## RAPTOR: Recursive Abstractive Processing {#raptor}

### How It Works

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builds a tree of summaries:

```
Level 3:    [Root Summary]
                 ↑
Level 2:  [Summary A]  [Summary B]  [Summary C]
               ↑            ↑            ↑
Level 1:  [S1] [S2]    [S3] [S4]    [S5] [S6]
               ↑            ↑            ↑
Level 0:  [Original Document Chunks...]
```

**Process:**
1. Chunk documents into ~400 token pieces
2. Embed all chunks
3. **Cluster** chunks using UMAP dimensionality reduction + GMM
4. **Summarize** each cluster using LLM
5. Embed summaries as new nodes
6. Recursively repeat until tree collapses to root

**Retrieval Modes:**
- **Collapsed:** Treat entire tree as flat index, retrieve top-k from any level
- **Tree Traversal:** Start at top, descend through relevant branches

### Why RAPTOR Failed for Our Vault

#### Root Cause: GMM Clustering with High-Dimensional Embeddings

The llama_index RAPTOR implementation uses sklearn's `GaussianMixture` for clustering. This algorithm:

1. Assumes data follows Gaussian distributions
2. Estimates covariance matrices for each cluster
3. **Fails when covariance matrices become singular** (ill-defined)

With Voyage embeddings (1024 dimensions):
- Embeddings are dense and semantically similar across an Obsidian vault
- UMAP reduction to 10 dimensions doesn't sufficiently separate clusters
- GMM covariance estimation produces NaN/Inf values
- Results in endless warnings and eventual failure

#### Error Symptoms

```python
sklearn.mixture._gaussian_mixture.py: RuntimeWarning:
  divide by zero encountered in matmul
sklearn.mixture._gaussian_mixture.py: RuntimeWarning:
  invalid value encountered in matmul
ValueError: Fitting the mixture model failed because some components
  have ill-defined empirical covariance
```

#### Attempted Fixes

| Fix Attempted | Result |
|---------------|--------|
| Reduce document count (50 notes) | Still failed - same GMM issues |
| Add regularization (`reg_covar=1e-3`) | Reduced warnings, still unstable |
| Use diagonal covariance | Still hit numerical limits |
| Early patch before imports | Clustering module already imported internally |

### RAPTOR Advantages (When It Works)

- **Retrieval at multiple abstraction levels** - Can find both specific facts and high-level themes
- **Pre-computed summaries** - No runtime LLM cost for summarization
- **Works well with** smaller, more diverse corpora

### RAPTOR Disadvantages

- **Clustering algorithm sensitivity** - GMM fails with certain embedding distributions
- **High embedding token cost** - Every chunk embedded, then summaries embedded at each level
- **Rebuilds required** - Adding new documents requires re-clustering
- **Not incremental** - Can't add single documents efficiently

### Token Cost Estimate for Our Vault

```
Documents: 2,371 notes
Avg chunks/note: ~5 (at 400 tokens/chunk)
Total chunks: ~11,855

Level 0 embeddings: 11,855 × 400 = 4.7M tokens
Level 1 (assume 5:1 reduction): 2,371 summaries × 800 = 1.9M tokens
Level 2: ~474 summaries × 800 = 0.4M tokens
Level 3: ~95 summaries × 800 = 0.08M tokens

Total embedding tokens: ~7-10M tokens
LLM tokens (summarization): ~5M output tokens
```

---

## GraphRAG: Knowledge Graph Approach {#graphrag}

### How It Works

GraphRAG (Microsoft Research, 2024) extracts a knowledge graph from documents:

```
Documents → Entity/Relation Extraction → Knowledge Graph → Community Detection → Community Summaries
```

**Process:**
1. For each chunk, use LLM to extract:
   - **Entities** (people, concepts, tools, etc.)
   - **Relationships** between entities
2. Build a graph with entities as nodes, relationships as edges
3. Run community detection (Louvain/Leiden algorithm)
4. Generate LLM summaries for each community
5. Index both chunks and community summaries

**Query Modes:**
- **Local Search:** Traditional chunk retrieval + entity context
- **Global Search:** Query community summaries for thematic questions

### Why GraphRAG Might Work for Our Vault

#### No GMM Clustering

GraphRAG uses graph-based community detection:
- Louvain/Leiden algorithms are **deterministic and stable**
- No covariance estimation, no numerical instability
- Works regardless of embedding dimensionality

#### Obsidian Has Explicit Relationships

Your vault already contains entity relationships via wikilinks:
- `[[Concept A]]` → explicit entity marker
- Links between notes → explicit relationships
- This is exactly what GraphRAG extracts (but you have it for free!)

### GraphRAG Advantages

- **Stable algorithms** - No numerical clustering issues
- **Rich context** - Entity relationships provide semantically meaningful connections
- **Global queries** - Community summaries enable thematic search
- **Incremental updates** - Can add new nodes to existing graph

### GraphRAG Disadvantages

- **Extremely LLM-intensive** - Entity extraction requires LLM call per chunk
- **Complex infrastructure** - Requires graph database or in-memory graph
- **Entity resolution** - Same entity in different forms needs deduplication
- **Quality depends on extraction** - Bad entity extraction = bad graph

### Token Cost Estimate for Our Vault

```
Chunks: ~11,855

Entity extraction per chunk:
  Input: ~500 tokens (chunk + prompt)
  Output: ~200 tokens (entities/relations)

Relationship extraction per chunk:
  Input: ~500 tokens
  Output: ~200 tokens

Community summaries (assume 500 communities):
  Input: ~1,000 tokens per community
  Output: ~500 tokens per summary

Total LLM Input: ~12M tokens
Total LLM Output: ~5M tokens + ~250K community summaries = ~5.25M tokens

Plus embedding tokens: ~15M tokens
```

**Cost comparison (Gemini pricing):**
- Input: ~12M tokens × $0.075/1M = ~$0.90
- Output: ~5.25M tokens × $0.30/1M = ~$1.58
- **Total LLM cost: ~$2.50** (plus embedding costs)

---

## Lightweight Alternatives {#lightweight-alternatives}

### Alternative 1: Wikilink Graph Traversal

Your vault already has a graph via wikilinks. A lightweight approach:

```python
def graph_enhanced_retrieval(query, initial_results, depth=1):
    """Expand results by following wikilinks."""
    expanded = set(initial_results)

    for doc in initial_results:
        # Get wikilinks from document metadata
        linked_notes = doc.metadata.get('wikilinks', [])
        # Add linked documents (1-hop neighbors)
        expanded.update(retrieve_by_title(linked_notes))

    return list(expanded)
```

**Pros:** Zero additional indexing cost, uses existing metadata
**Cons:** Limited to explicit links, no semantic clustering

### Alternative 2: Research Mode (Already Implemented!)

Your `research_mode.py` provides iterative multi-step retrieval:

1. Initial query → retrieve chunks
2. LLM identifies **knowledge gaps**
3. Generate follow-up queries
4. Retrieve additional context
5. Synthesize comprehensive answer

**Pros:** No pre-indexing cost, adaptive to query needs
**Cons:** Higher runtime cost per query

### Alternative 3: Query-Time Clustering

Instead of pre-clustering, cluster results at query time:

```python
def clustered_results(query, k=50, num_clusters=5):
    """Retrieve k results, cluster into themes, return representatives."""
    results = retriever.retrieve(query, k=k)

    # Cluster at query time
    embeddings = [r.embedding for r in results]
    clusters = kmeans(embeddings, n_clusters=num_clusters)

    # Return top result from each cluster
    representatives = []
    for cluster_id in range(num_clusters):
        cluster_docs = [r for r, c in zip(results, clusters) if c == cluster_id]
        representatives.append(cluster_docs[0])  # Most relevant in cluster

    return representatives
```

**Pros:** No pre-indexing, provides diversity
**Cons:** Adds latency, k-means simpler than GMM

---

## Token Cost Analysis {#token-cost-analysis}

### Embedding Token Consumption During RAPTOR Attempts

| Attempt | Documents | Tokens Used | Result |
|---------|-----------|-------------|--------|
| Initial (full vault) | 2,371 | ~8M | GMM failure after 40 min |
| Second (150 notes) | 150 | ~5M | GMM warnings, stuck |
| Third (50 notes, patched) | 50 | ~5M | Still failing |
| **Total consumed** | - | **~18M** | No index created |

**Monthly budget impact:** 18M / 200M = 9% of free tier consumed with zero output

### Cost Comparison Table

| Approach | Embedding Tokens | LLM Tokens | Stability | Incremental |
|----------|-----------------|------------|-----------|-------------|
| **Current RAG** | 0 (done) | Query-time only | Excellent | Yes |
| **RAPTOR** | ~10M | ~5M | Poor (GMM) | No |
| **GraphRAG** | ~15M | ~15M+ | Good | Partial |
| **Research Mode** | 0 | ~2K/query | Excellent | N/A |
| **Wikilink Graph** | 0 | 0 | Excellent | Yes |

---

## Decision Framework {#decision-framework}

### When to Consider Hierarchical RAG

Deploy hierarchical approaches **only if** you observe:

1. **Thematic queries failing** - "What are my main ideas about X?" returns fragmented results
2. **Multi-document synthesis weak** - Can't connect concepts across notes
3. **Abstractive answers poor** - System can't summarize themes
4. **Research Mode insufficient** - Iterative retrieval doesn't fill gaps

### When to Stick with Current Setup

Your current RAG is likely sufficient if:

1. **Factual queries work well** - "What did I write about X?" finds relevant chunks
2. **Reranking improves results** - Top 10 after reranking are high quality
3. **Research Mode helps complex queries** - Multi-step retrieval fills gaps
4. **Wikilinks provide context** - Graph traversal could enhance if needed

### Implementation Priority

If you decide to enhance:

1. **First:** Try wikilink graph traversal (zero cost, uses existing metadata)
2. **Second:** Tune Research Mode parameters (already implemented)
3. **Third:** Consider query-time clustering (adds latency, no pre-indexing)
4. **Last resort:** GraphRAG (high cost, requires infrastructure)

**Skip RAPTOR entirely** - The GMM clustering issue is fundamental with Voyage embeddings.

---

## Implementation Artifacts {#implementation-artifacts}

### Scripts Created During Evaluation

```
scripts/
├── score_notes.py              # Score vault notes by value metrics
├── build_raptor_filtered.py    # RAPTOR build with filtered documents
└── build_raptor_simple.py      # RAPTOR build with stability patches
```

### Note Scoring Criteria (score_notes.py)

The scoring script evaluates notes by:

| Metric | Weight | Rationale |
|--------|--------|-----------|
| Word count | 0.20 | Longer notes have more content |
| Outgoing wikilinks | 0.25 | Hub notes connect concepts |
| Incoming backlinks | 0.25 | Referenced notes are valuable |
| Tag count | 0.10 | Tagged notes are organized |
| Frontmatter richness | 0.10 | Metadata indicates intentionality |
| Recency | 0.10 | Recently modified = actively used |

### Generated Files

```
data/
├── top_150_notes.json          # Ranked list of 150 most valuable notes
├── top_50_notes.json           # Ranked list of 50 most valuable notes
└── raptor/                     # Empty - no successful index created
```

### RAPTOR Clustering Stability Patch

Attempted patch for GMM stability (did not fully resolve issue):

```python
def patch_raptor_clustering():
    """Patch with regularization and diagonal covariance."""
    from sklearn.mixture import GaussianMixture

    def GMM_cluster_stable(embeddings, threshold, random_state=0):
        gm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            reg_covar=1e-3,        # Add regularization
            max_iter=100,
            n_init=1,
            covariance_type='diag'  # Simpler covariance
        )
        # ... rest of implementation
```

---

## Lessons Learned {#lessons-learned}

### Technical Lessons

1. **Embedding dimensionality matters for clustering** - 1024-dim Voyage embeddings create numerical issues for GMM
2. **UMAP reduction isn't sufficient** - Even 10-dim reduction doesn't make GMM stable
3. **Monkey-patching has limits** - Internal imports can bypass patches
4. **Test with small subsets first** - Would have saved tokens if we started with 10 docs

### Architectural Lessons

1. **Simple solutions often sufficient** - Hybrid search + reranking covers most use cases
2. **Query-time > index-time** - Research Mode's iterative approach avoids pre-indexing costs
3. **Use existing structure** - Wikilinks are a free knowledge graph
4. **Token costs compound** - Hierarchical approaches multiply base embedding costs

### Process Lessons

1. **Measure before optimizing** - Quantify retrieval quality issues before adding complexity
2. **Cost-benefit analysis first** - 18M tokens for RAPTOR attempts was expensive
3. **Incremental testing** - Start with 10 docs, not 150 or 2,371
4. **Keep logs** - This document preserves learnings from failed experiments

---

## References

### Papers

- **RAPTOR:** "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (Sarthi et al., 2024)
- **GraphRAG:** "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research, 2024)

### Implementations

- llama_index RAPTOR pack: `llama-index-packs-raptor`
- Microsoft GraphRAG: `github.com/microsoft/graphrag`

### Related UltraRAG Features

- `research_mode.py` - Iterative multi-step retrieval
- `loader.py` - Wikilink extraction (existing graph structure)
- `query_engine.py` - Hybrid search with reranking

---

## Appendix: Configuration Reference

### Current Working Configuration

```env
# Embedding (working well)
EMBEDDING_MODEL=voyage-3.5-lite
CHUNK_SIZE=512
CHUNK_OVERLAP=75

# Retrieval (working well)
TOP_K=75
RERANK_TOP_N=10
RERANKER_MODEL=rerank-2.5
SIMILARITY_THRESHOLD=0.3

# RAPTOR (disabled)
ENABLE_RAPTOR=false
RAPTOR_MODE=collapsed
RAPTOR_CHUNK_SIZE=400
RAPTOR_TOP_K=10

# Research Mode (enabled, use for complex queries)
RESEARCH_MAX_ITERATIONS=3
RESEARCH_CONFIDENCE_THRESHOLD=0.8
```

### If You Enable RAPTOR Later

For small, diverse document sets (<100 docs with varied topics):

```env
ENABLE_RAPTOR=true
RAPTOR_MODE=collapsed
RAPTOR_CHUNK_SIZE=400
RAPTOR_TOP_K=10
```

Note: May still fail with Voyage embeddings. Consider using a different embedding model (e.g., OpenAI) for RAPTOR specifically if needed.

---

*Report generated from UltraRAG hierarchical RAG evaluation session, January 2026*
