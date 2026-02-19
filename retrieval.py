"""
UltraRAG Retrieval Module

Handles all query operations: simple, hybrid, federated, research.
Agent note: This is the ONLY file you need to read for retrieval/query tasks.

PUBLIC API:
- query(query_str, index, mode) -> QueryResult
- federated_query(query_str, indexes, mode) -> QueryResult
- research(query_str, indexes, exhaustive) -> ResearchResult
"""

import logging
import time
from typing import Literal

from llama_index.core import VectorStoreIndex

from config import RAGConfig, load_config
from models import (
    QueryResult, ResearchResult, SourceNode, IndexSource, SourceType,
    IndexNotFoundError, LLMRateLimitError,
    QueryMode, SourceFilter, BookFilter,
)

logger = logging.getLogger("ultrarag.retrieval")


# === PUBLIC API ===

def query(
    query_str: str,
    index: VectorStoreIndex | None,
    config: RAGConfig | None = None,
    mode: QueryMode = "hybrid",
) -> QueryResult:
    """
    Execute query against an index.

    Args:
        query_str: The query to execute
        index: VectorStoreIndex to query
        config: Optional config (uses default if None)
        mode: Query mode - "simple", "hybrid", or "research"

    Returns:
        QueryResult with answer, sources, and metadata

    Raises:
        IndexNotFoundError: If index is None
        LLMRateLimitError: If Gemini rate limit hit

    Example:
        >>> result = query("What is RAG?", index)
        >>> print(result.answer)
        >>> for source in result.sources:
        ...     print(f"- {source.file_name}: {source.score:.2f}")
    """
    if index is None:
        raise IndexNotFoundError("vault")

    config = config or load_config()
    start_time = time.time()

    try:
        # Build appropriate query engine
        engine = _build_query_engine(index, config, mode)

        # Execute query
        response = engine.query(query_str)

        # Convert to QueryResult
        sources = extract_sources(getattr(response, 'source_nodes', []))

        return QueryResult(
            answer=str(response.response) if hasattr(response, 'response') else str(response),
            sources=sources,
            tokens_used=_get_tokens_used(response),
            exec_time=time.time() - start_time,
            query=query_str,
            mode=mode,
        )

    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            raise LLMRateLimitError()
        raise


def federated_query(
    query_str: str,
    indexes: dict[str, IndexSource],
    config: RAGConfig | None = None,
    mode: QueryMode = "hybrid",
    source_filter: SourceFilter = None,
    book_filter: "BookFilter | None" = None,
) -> QueryResult:
    """
    Query multiple indexes with weighted merging.

    Args:
        query_str: The query to execute
        indexes: Dict of name -> IndexSource
        config: Optional config
        mode: Query mode
        source_filter: Limit to specific source (vault, conversations, etc)

    Returns:
        QueryResult with merged results from all sources

    Example:
        >>> sources = {"vault": vault_source, "conv": conv_source}
        >>> result = federated_query("RAG techniques", sources)
    """
    if not indexes:
        raise IndexNotFoundError("federated")

    config = config or load_config()
    start_time = time.time()

    # Retrieve from all sources
    all_nodes = _federated_retrieve(query_str, indexes, config, source_filter, book_filter)

    # Synthesize answer
    llm = get_llm(config)
    answer = synthesize_answer(query_str, all_nodes, llm)

    sources = extract_sources(all_nodes)

    return QueryResult(
        answer=answer,
        sources=sources,
        tokens_used=0,  # TODO: track tokens
        exec_time=time.time() - start_time,
        query=query_str,
        mode=mode,
    )


def research(
    query_str: str,
    indexes: dict[str, IndexSource] | VectorStoreIndex | None,
    config: RAGConfig | None = None,
    exhaustive: bool = False,
) -> ResearchResult:
    """
    Iterative research with gap analysis.

    Args:
        query_str: The research query
        indexes: Single index or dict of indexes
        config: Optional config
        exhaustive: Force all iterations (for "all/every" queries)

    Returns:
        ResearchResult with iterations, gaps, and confidence

    Example:
        >>> result = research("find all RAG techniques", index, exhaustive=True)
        >>> print(f"Confidence: {result.confidence:.1%}")
        >>> print(f"Iterations: {result.iterations}")
    """
    if indexes is None:
        raise IndexNotFoundError("research")

    config = config or load_config()
    start_time = time.time()

    # Normalize indexes to dict
    if isinstance(indexes, VectorStoreIndex):
        indexes = {
            "vault": IndexSource(
                name="vault",
                index=indexes,
                source_type=SourceType.VAULT,
            )
        }

    # Delegate to internal research function
    result = _do_research(query_str, indexes, config, exhaustive)

    # Ensure timing is set
    result.exec_time = time.time() - start_time

    return result


# === INTERNAL HELPERS ===

def _build_query_engine(index: VectorStoreIndex, config: RAGConfig, mode: str):
    """Build appropriate query engine based on mode."""
    llm = get_llm(config)

    if mode == "simple":
        # Simple vector retrieval
        return index.as_query_engine(
            llm=llm,
            similarity_top_k=config.retrieval.top_k,
        )
    else:
        # Hybrid retrieval (vector + BM25)
        # For now, use simple as fallback - hybrid requires docstore
        return index.as_query_engine(
            llm=llm,
            similarity_top_k=config.retrieval.top_k,
        )


def get_llm(config: RAGConfig):
    """Get configured LLM instance."""
    try:
        from tracked_llm import wrap_llm_with_tracking

        if config.llm.backend == "cli":
            from gemini_cli import GeminiCLI
            return GeminiCLI()
        else:
            from llama_index.llms.google_genai import GoogleGenAI
            llm = GoogleGenAI(model=config.llm.model)
            return wrap_llm_with_tracking(llm)
    except Exception as e:
        logger.warning(f"Failed to get LLM: {e}, using default")
        from llama_index.llms.google_genai import GoogleGenAI
        return GoogleGenAI(model=config.llm.model)


def _federated_retrieve(
    query_str: str,
    indexes: dict[str, IndexSource],
    config: RAGConfig,
    source_filter: SourceFilter = None,
    book_filter: BookFilter | None = None,
) -> list:
    """Retrieve from multiple indexes and merge with guaranteed source diversity.

    Uses reserved-slot strategy: each source gets a minimum number of slots
    proportional to its presence, ensuring non-primary sources aren't squeezed
    out by score dominance from a single source.
    """
    per_source_nodes: dict[str, list] = {}
    active_sources = []

    for name, source in indexes.items():
        if source_filter and source_filter != "all" and source.source_type.value != source_filter:
            continue

        active_sources.append(name)
        try:
            if source.source_type.value == "books" and book_filter:
                from books_retriever import get_books_retriever
                retriever = get_books_retriever(source.index, config.retrieval.top_k, book_filter)
            else:
                retriever = source.index.as_retriever(similarity_top_k=config.retrieval.top_k)
            nodes = retriever.retrieve(query_str)

            # Tag each node with its source name for tracking
            for node in nodes:
                if hasattr(node, 'score') and node.score is not None:
                    node.score *= source.weight
                if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                    node.node.metadata['_federation_source'] = name

            per_source_nodes[name] = sorted(
                nodes, key=lambda n: getattr(n, 'score', 0) or 0, reverse=True
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve from {name}: {e}")

    total_limit = config.retrieval.rerank_top_n
    num_sources = len(per_source_nodes)

    # Single source or no results — no diversity needed
    if num_sources <= 1:
        all_nodes = []
        for nodes in per_source_nodes.values():
            all_nodes.extend(nodes)
        all_nodes.sort(key=lambda n: getattr(n, 'score', 0) or 0, reverse=True)
        return all_nodes[:total_limit]

    # Multi-source: reserve minimum slots per non-primary source
    # Primary source (highest weight) gets remaining slots after reservations
    # Reserve 15% of total per non-primary source (min 10, max 50)
    reserve_per_source = max(10, min(50, int(total_limit * 0.15)))

    # Phase 1: Fill reserved slots for each source (top-N from each)
    reserved: dict[str, list] = {}
    used_ids: set = set()
    for name, nodes in per_source_nodes.items():
        reserved[name] = []
        for node in nodes[:reserve_per_source]:
            node_id = id(node)
            if node_id not in used_ids:
                reserved[name].append(node)
                used_ids.add(node_id)

    # Phase 2: Fill remaining slots by global score ranking
    remaining_slots = total_limit - sum(len(v) for v in reserved.values())
    overflow = []
    for name, nodes in per_source_nodes.items():
        for node in nodes:
            if id(node) not in used_ids:
                overflow.append(node)
    overflow.sort(key=lambda n: getattr(n, 'score', 0) or 0, reverse=True)

    # Combine: reserved slots + best remaining
    all_nodes = []
    for nodes in reserved.values():
        all_nodes.extend(nodes)
    all_nodes.extend(overflow[:max(0, remaining_slots)])

    # Final sort by score
    all_nodes.sort(key=lambda n: getattr(n, 'score', 0) or 0, reverse=True)

    if num_sources > 1:
        source_counts = {}
        for n in all_nodes:
            src = n.node.metadata.get('_federation_source', '?') if hasattr(n, 'node') else '?'
            source_counts[src] = source_counts.get(src, 0) + 1
        logger.info(f"Federated retrieve: {len(all_nodes)} nodes from {source_counts}")

    return all_nodes[:total_limit]


def synthesize_answer(query_str: str, nodes: list, llm) -> str:
    """Synthesize answer from retrieved nodes."""
    if not nodes:
        return "No relevant information found."

    # Build context from nodes
    context_parts = []
    for i, node in enumerate(nodes[:10]):  # Limit to top 10
        text = node.node.text if hasattr(node, 'node') else str(node)
        context_parts.append(f"[{i+1}] {text[:500]}")

    context = "\n\n".join(context_parts)

    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query_str}

Answer:"""

    try:
        response = llm.complete(prompt)
        return str(response.text) if hasattr(response, 'text') else str(response)
    except Exception as e:
        logger.error(f"Failed to synthesize answer: {e}")
        return f"Error synthesizing answer: {e}"


def _do_research(
    query_str: str,
    indexes: dict[str, IndexSource],
    config: RAGConfig,
    exhaustive: bool = False,
) -> ResearchResult:
    """Perform iterative research with gap analysis."""
    try:
        from research_mode import ResearchRetriever
        from federated_query import FederatedRetriever

        # Build federated retriever
        sources_list = list(indexes.values())
        base_retriever = FederatedRetriever(
            sources=sources_list,
            config=config,
        )

        llm = get_llm(config)

        research_retriever = ResearchRetriever(
            base_retriever=base_retriever,
            llm=llm,
            max_iterations=config.retrieval.research_max_iterations,
            confidence_threshold=config.retrieval.research_confidence_threshold,
        )

        result = research_retriever.research(query_str, force_exhaustive=exhaustive)

        # Convert to our ResearchResult type
        return ResearchResult(
            answer=result.answer if hasattr(result, 'answer') else str(result),
            sources=extract_sources(result.source_nodes if hasattr(result, 'source_nodes') else []),
            tokens_used=_get_tokens_used(result),
            exec_time=0,  # Will be set by caller
            query=query_str,
            mode="research",
            iterations=result.iterations if hasattr(result, 'iterations') else 1,
            confidence=result.confidence if hasattr(result, 'confidence') else 0.0,
            is_exhaustive=exhaustive,
        )
    except Exception as e:
        logger.error(f"Research failed: {e}")
        # Return minimal result on failure
        return ResearchResult(
            answer=f"Research failed: {e}",
            sources=[],
            tokens_used=0,
            exec_time=0,
            query=query_str,
            mode="research",
            iterations=0,
            confidence=0.0,
        )


def extract_sources(source_nodes) -> list[SourceNode]:
    """Convert LlamaIndex source nodes to our SourceNode type."""
    sources = []
    for node in source_nodes:
        try:
            inner_node = node.node if hasattr(node, 'node') else node
            metadata = inner_node.metadata if hasattr(inner_node, 'metadata') else {}
            text = inner_node.text if hasattr(inner_node, 'text') else str(inner_node)

            sources.append(SourceNode(
                file_path=metadata.get('file_path', ''),
                file_name=metadata.get('file_name', ''),
                excerpt=text[:500] if text else '',
                score=node.score if hasattr(node, 'score') else 0.0,
                source_type=metadata.get('source_type', 'vault'),
                metadata=metadata,
            ))
        except Exception as e:
            logger.debug(f"Failed to extract source: {e}")

    return sources


def _get_tokens_used(response) -> int:
    """Extract token count from response."""
    if hasattr(response, 'metadata') and isinstance(response.metadata, dict):
        return response.metadata.get('token_count', 0)
    return 0
