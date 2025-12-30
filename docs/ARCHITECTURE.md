# UltraRAG System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │   CLI Interface  │              │  Web Interface   │         │
│  │    (main.py)     │              │    (app.py)      │         │
│  │                  │              │   Streamlit UI   │         │
│  └────────┬─────────┘              └────────┬─────────┘         │
│           │                                  │                   │
│           └──────────────┬───────────────────┘                   │
│                          │                                       │
└──────────────────────────┼───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    ORCHESTRATION LAYER                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    UltraRAG (main.py)                      │ │
│  │  - System initialization                                   │ │
│  │  - Component coordination                                  │ │
│  │  - Query routing                                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   INGESTION   │  │   RETRIEVAL   │  │  GENERATION   │
│     LAYER     │  │     LAYER     │  │     LAYER     │
└───────────────┘  └───────────────┘  └───────────────┘

═══════════════════════════════════════════════════════════════════
                        DETAILED FLOW
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                      1. INGESTION PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

   Obsidian Vault                 ObsidianLoader
        │                              │
        │  ┌────────────────────────┐  │
        └─►│  Load .md files        │──┘
           │  Parse frontmatter     │
           │  Extract wikilinks     │
           │  Extract tags          │
           │  Build metadata        │
           └────────┬───────────────┘
                    │
                    ▼
           ┌────────────────────────┐
           │  Convert to Documents  │
           │  - Rich metadata       │
           │  - File paths          │
           │  - Relationships       │
           └────────┬───────────────┘
                    │
                    ▼
           ┌────────────────────────┐
           │   ObsidianChunker      │
           │  - Markdown-aware      │
           │  - Semantic splitting  │
           │  - 512 token chunks    │
           │  - 15% overlap         │
           └────────┬───────────────┘
                    │
                    ▼
           ┌────────────────────────┐
           │  Add Parent Context    │
           │  - Document summary    │
           │  - Chunk position      │
           │  - Total chunks        │
           └────────┬───────────────┘
                    │
                    ▼
           ┌────────────────────────┐
           │  Embedding Model       │
           │  Voyage-3-large OR     │
           │  Qwen3-8B OR           │
           │  OpenAI-3-large        │
           └────────┬───────────────┘
                    │
                    ▼
           ┌────────────────────────┐
           │   Vector Database      │
           │  LanceDB (embedded) OR │
           │  Qdrant (scalable)     │
           └────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                     2. RETRIEVAL PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

   User Query
        │
        ▼
   ┌─────────────────────────┐
   │  Query Processing       │
   │  - Clean & normalize    │
   │  - Optional expansion   │
   └───────┬─────────────────┘
           │
           ▼
   ┌─────────────────────────┐
   │  Hybrid Retrieval       │
   │                         │
   │  ┌─────────────────┐    │
   │  │ Vector Search   │    │
   │  │ (similarity)    │    │
   │  └────────┬────────┘    │
   │           │              │
   │  ┌────────▼────────┐    │
   │  │ Query Fusion    │    │
   │  │ (3 variations)  │    │
   │  └────────┬────────┘    │
   │           │              │
   │  ┌────────▼────────┐    │
   │  │ Top-K=75        │    │
   │  │ candidates      │    │
   │  └─────────────────┘    │
   └───────┬─────────────────┘
           │
           ▼
   ┌─────────────────────────┐
   │  Post-Processing        │
   │                         │
   │  ┌─────────────────┐    │
   │  │ Reranking       │    │
   │  │ Voyage Rerank 2 │    │
   │  └────────┬────────┘    │
   │           │              │
   │  ┌────────▼────────┐    │
   │  │ Similarity      │    │
   │  │ Filtering       │    │
   │  └────────┬────────┘    │
   │           │              │
   │  ┌────────▼────────┐    │
   │  │ Top-N=10        │    │
   │  │ final results   │    │
   │  └─────────────────┘    │
   └───────┬─────────────────┘
           │
           ▼
   ┌─────────────────────────┐
   │  Retrieved Context      │
   │  - Source chunks        │
   │  - Metadata             │
   │  - Relevance scores     │
   └─────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                     3. GENERATION PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

   Retrieved Context + Query
            │
            ▼
   ┌─────────────────────────┐
   │  PTCF Prompt Builder    │
   │                         │
   │  P - Premise (context)  │
   │  T - Task (query)       │
   │  C - Constraints        │
   │  F - Format             │
   └───────┬─────────────────┘
           │
           ▼
   ┌─────────────────────────┐
   │  Gemini 2.0 Flash       │
   │  - 1M context window    │
   │  - Thinking mode        │
   │  - Temp: 0.1            │
   │  - Max tokens: 8192     │
   └───────┬─────────────────┘
           │
           ▼
   ┌─────────────────────────┐
   │  Response Synthesis     │
   │  - Compact mode         │
   │  - Refine if needed     │
   └───────┬─────────────────┘
           │
           ▼
   ┌─────────────────────────┐
   │  Final Response         │
   │  - Answer text          │
   │  - Source citations     │
   │  - Relevance scores     │
   └─────────────────────────┘


═══════════════════════════════════════════════════════════════════
                        DATA FLOW DIAGRAM
═══════════════════════════════════════════════════════════════════

┌──────────────┐
│ Obsidian.md  │  Raw notes with wikilinks, tags, frontmatter
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Document    │  Parsed with metadata, relationships
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Chunks     │  512-token semantic chunks with context
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embeddings  │  1024-dim vectors (Voyage/Qwen)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Vector Index │  Searchable index with metadata
└──────┬───────┘
       │
       │  (At Query Time)
       │
       ▼
┌──────────────┐
│Query Vector  │  User question embedded
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Top-K Chunks  │  75 similar chunks retrieved
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Reranked    │  10 most relevant chunks
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Context    │  Assembled context for LLM
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Answer     │  Generated response with sources
└──────────────┘


═══════════════════════════════════════════════════════════════════
                      COMPONENT INTERACTIONS
═══════════════════════════════════════════════════════════════════

config.py ──────► Provides configuration to all components
   │
   ├──► loader.py ──────► Loads vault & extracts metadata
   │
   ├──► embeddings.py ──► Initializes embedding models
   │
   ├──► chunking.py ────► Splits documents intelligently
   │
   ├──► vector_store.py ► Manages vector database
   │
   ├──► query_engine.py ► Orchestrates retrieval
   │
   └──► main.py ────────► Coordinates everything


═══════════════════════════════════════════════════════════════════
                        EXTENSIBILITY POINTS
═══════════════════════════════════════════════════════════════════

Phase 2 Ready:
   │
   ├─► Graph Database (Neo4j)
   │   └─► Add wikilink traversal to retrieval
   │
   ├─► RAPTOR Summaries
   │   └─► Add hierarchical clustering to indexing
   │
   ├─► Temporal Filtering
   │   └─► Add date-based filtering to query engine
   │
   └─► Adaptive Router
       └─► Add query classification before retrieval


═══════════════════════════════════════════════════════════════════
                        DEPLOYMENT OPTIONS
═══════════════════════════════════════════════════════════════════

Option 1: Local Development
   └─► python main.py (CLI)
   └─► streamlit run app.py (Web UI)

Option 2: Self-Hosted Server
   └─► Qdrant for vector storage
   └─► Qwen3-8B for local embeddings
   └─► No API costs

Option 3: Cloud APIs
   └─► Voyage API for embeddings/reranking
   └─► Google Gemini for generation
   └─► LanceDB local storage

Option 4: Hybrid
   └─► Local embeddings (Qwen3-8B)
   └─► Cloud LLM (Gemini)
   └─► Best of both worlds
