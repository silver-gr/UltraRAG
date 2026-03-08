# Prompt: UltraRAG Competitive Analysis

Use this prompt in a new Claude Code session from the UltraRAG project root.

---

<context>
You are analyzing UltraRAG, a production RAG system for Obsidian vaults, against 23 open-source RAG implementations. The goal is to identify where UltraRAG leads, where it lags, and what features are worth adopting.

UltraRAG's architecture is documented in CLAUDE.md. The competitor list is in docs/COMPETITOR_RAG_SYSTEMS.md. Each competitor is a GitHub repo you can fetch READMEs from.

This analysis matters because UltraRAG is a personal tool being refined iteratively. Understanding the competitive landscape reveals blind spots and validates existing design choices. We care about practical capability, not star counts.
</context>

<task>
Conduct a structured competitive analysis of UltraRAG against the RAG systems listed in docs/COMPETITOR_RAG_SYSTEMS.md.

Phase 1 - Understand UltraRAG:
Read CLAUDE.md to extract UltraRAG's full feature set. Build the feature taxonomy from what actually exists in the codebase.

Phase 2 - Research competitors:
For each of the top 12 competitors (by relevance, not stars), fetch their GitHub README to extract:
- Retrieval strategy (vector, hybrid, graph, reranking)
- Chunking approach
- Embedding models supported
- LLM integration (which models, how)
- Query transformation (HyDE, multi-query, decomposition)
- Self-correction / answer validation
- Research / iterative retrieval mode
- Document types supported
- UI (web, CLI, API)
- Deployment model (local, cloud, Docker)
- Unique differentiators

Use parallel agents to fetch READMEs concurrently.

Phase 3 - Compare:
Produce a feature matrix (markdown table) with UltraRAG as the first column and competitors as subsequent columns. Use checkmarks, X marks, or brief notes per cell.

Evaluation dimensions:
1. Retrieval quality (hybrid search, reranking, graph expansion)
2. Query intelligence (transformation, intent classification, bilingual)
3. Answer quality (self-correction, citation, synthesis)
4. Research capability (iterative retrieval, gap analysis, convergence)
5. Document handling (chunking strategies, format support, metadata)
6. Observability (token tracking, cost tracking, usage analytics)
7. Developer experience (CLI, API, configuration, extensibility)
8. Production readiness (caching, checkpointing, error recovery)

Phase 4 - Insights:
Write a concise report covering:
- Where UltraRAG leads (unique strengths)
- Where UltraRAG lags (gaps to close)
- Top 5 features worth adopting (with which competitor to reference)
- Top 5 features where UltraRAG already exceeds the field
- Architectural patterns worth studying from specific competitors
</task>

<output_format>
Write the full analysis to docs/COMPETITIVE_ANALYSIS.md with these sections:
1. Feature Matrix (table)
2. UltraRAG Strengths
3. UltraRAG Gaps
4. Recommendations (prioritized)
5. Architectural Insights

Keep the matrix scannable. Use the insights section for depth.
</output_format>

<constraints>
- Only compare features documented in READMEs. Do not infer capabilities.
- If a README is unavailable, mark that competitor as "README unavailable" and skip.
- Do not modify any UltraRAG source code. This is read-only analysis.
- Limit to top 12 most relevant competitors to keep analysis focused.
</constraints>
