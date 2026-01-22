# UltraRAG Documentation

Complete documentation for UltraRAG - a production-grade RAG system for Obsidian vaults.

## Getting Started

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Get running in under 10 minutes |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and data flow |
| [TESTING.md](TESTING.md) | Running and writing tests |

## Feature Documentation

Detailed guides for each advanced feature:

| Feature | Description | Benefit |
|---------|-------------|---------|
| [Late Chunking](features/LATE_CHUNKING.md) | Document-aware chunk embeddings | +10-12% retrieval accuracy |
| [Query Transformation](features/QUERY_TRANSFORMATION.md) | HyDE and multi-query expansion | +20-40% retrieval accuracy |
| [Self-Correction](features/SELF_CORRECTION.md) | Self-RAG and CRAG patterns | Auto-refines poor retrievals |
| [Contextual Retrieval](features/CONTEXTUAL_RETRIEVAL.md) | LLM-enhanced chunk context | -67% retrieval failures |
| [Graph Retrieval](features/GRAPH_RETRIEVAL.md) | Wikilink graph expansion | Finds related notes via links |
| [Research Mode](features/RESEARCH_MODE_ENHANCEMENTS.md) | Iterative multi-step retrieval | +141% accuracy on complex queries |
| [File Exclusions](features/FILE_EXCLUSIONS.md) | Exclude files/folders from indexing | Keep index clean |
| [Query Intent Classification](features/QUERY_INTENT_CLASSIFICATION.md) | Auto-detect query intent | *Future feature* |

## Reference

| Document | Description |
|----------|-------------|
| [RAG_STRATEGY.md](reference/RAG_STRATEGY.md) | Original research-backed strategy document |

## Quick Links

- **Main README**: [../README.md](../README.md)
- **Claude Code Guide**: [../CLAUDE.md](../CLAUDE.md)
- **Tests README**: [../tests/README.md](../tests/README.md)

## Documentation Map

```
docs/
├── README.md                    # This file - documentation index
├── QUICKSTART.md                # Getting started guide
├── ARCHITECTURE.md              # System architecture diagrams
├── TESTING.md                   # Testing guide
├── EVALUATION.md                # RAGAS evaluation setup
├── BILINGUAL_RETRIEVAL_REPORT.md # Cross-lingual retrieval analysis
├── HIERARCHICAL_RAG_EVALUATION.md # RAPTOR/GraphRAG evaluation
├── KHOJ_INSPIRED_FEATURES.md    # Research mode origins
├── LANGSMITH_INTEGRATION.md     # Observability setup
├── features/
│   ├── LATE_CHUNKING.md
│   ├── QUERY_TRANSFORMATION.md
│   ├── SELF_CORRECTION.md
│   ├── CONTEXTUAL_RETRIEVAL.md
│   ├── GRAPH_RETRIEVAL.md
│   ├── RESEARCH_MODE_ENHANCEMENTS.md
│   ├── FILE_EXCLUSIONS.md
│   └── QUERY_INTENT_CLASSIFICATION.md  # Future feature
└── reference/
    └── RAG_STRATEGY.md          # Original strategy document (historical)
```
