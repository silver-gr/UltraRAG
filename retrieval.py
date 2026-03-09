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
    QueryMode, SourceFilter, BookFilter, MetadataFilter,
)

logger = logging.getLogger("ultrarag.retrieval")


# === PUBLIC API ===

def query(
    query_str: str,
    index: VectorStoreIndex | None,
    config: RAGConfig | None = None,
    mode: QueryMode = "hybrid",
    metadata_filter: MetadataFilter | None = None,
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
    timings: dict[str, float] = {}

    try:
        # Build appropriate query engine
        t0 = time.time()
        engine = _build_query_engine(index, config, mode, metadata_filter=metadata_filter)
        timings["engine_build_ms"] = round((time.time() - t0) * 1000, 1)

        # Execute query
        t0 = time.time()
        response = engine.query(query_str)
        timings["query_ms"] = round((time.time() - t0) * 1000, 1)

        # Extract structured confidence from response metadata
        metadata = getattr(response, 'metadata', None) or {}
        validation_result = metadata.get('validation_warning')
        relevance_grade = metadata.get('relevance_grade')

        # Convert to QueryResult
        source_nodes = getattr(response, 'source_nodes', [])
        sources = extract_sources(source_nodes)

        # Compute confidence from source scores (normalized to 0-1)
        confidence = _compute_confidence(source_nodes)

        total_ms = round((time.time() - start_time) * 1000, 1)
        timings["total_ms"] = total_ms

        return QueryResult(
            answer=str(response.response) if hasattr(response, 'response') else str(response),
            sources=sources,
            tokens_used=_get_tokens_used(response),
            exec_time=time.time() - start_time,
            query=query_str,
            mode=mode,
            confidence=confidence,
            validation_result=validation_result,
            relevance_grade=relevance_grade,
            timings=timings,
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
    metadata_filter: MetadataFilter | None = None,
) -> QueryResult:
    """
    Query multiple indexes with reranking, BM25, self-correction, and proper synthesis.

    Delegates to FederatedQueryEngine which handles all retrieval features consistently.

    Args:
        query_str: The query to execute
        indexes: Dict of name -> IndexSource (models.IndexSource)
        config: Optional config
        mode: Query mode
        source_filter: Limit to specific source (vault, conversations, etc)
        book_filter: Optional BookFilter for book category/author filtering

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
    timings: dict[str, float] = {}

    from federated_query import FederatedQueryEngine
    from federated_query import IndexSource as FedIndexSource

    # Convert models.IndexSource -> federated_query.IndexSource
    fed_sources = []
    for name, src in indexes.items():
        if source_filter and source_filter != "all" and src.source_type.value != source_filter:
            continue
        fed_sources.append(FedIndexSource(
            name=src.name,
            source_type=src.source_type.value,
            index=src.index,
            weight=src.weight,
            nodes=src.nodes,
            wikilink_graph=src.wikilink_graph,
        ))

    if not fed_sources:
        raise IndexNotFoundError("federated")

    t0 = time.time()
    engine = FederatedQueryEngine(
        sources=fed_sources, config=config, book_filter=book_filter,
        metadata_filter=metadata_filter,
    )
    timings["engine_build_ms"] = round((time.time() - t0) * 1000, 1)

    t0 = time.time()
    response = engine.query(query_str)
    timings["query_ms"] = round((time.time() - t0) * 1000, 1)

    source_nodes = getattr(response, 'source_nodes', [])
    confidence = _compute_confidence(source_nodes)

    total_ms = round((time.time() - start_time) * 1000, 1)
    timings["total_ms"] = total_ms

    return QueryResult(
        answer=str(response.response) if hasattr(response, 'response') else str(response),
        sources=extract_sources(source_nodes),
        tokens_used=_get_tokens_used(response),
        exec_time=time.time() - start_time,
        query=query_str,
        mode=mode,
        confidence=confidence,
        timings=timings,
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

def _build_query_engine(index: VectorStoreIndex, config: RAGConfig, mode: str, metadata_filter=None):
    """Build appropriate query engine based on mode.

    Uses the same RAGQueryEngine/HybridQueryEngine as the web UI and interactive CLI,
    ensuring consistent reranking, BM25, postprocessors, and query transformation.
    """
    from query_engine import RAGQueryEngine, HybridQueryEngine

    if mode == "simple":
        return RAGQueryEngine(index, config, metadata_filter=metadata_filter)
    else:
        return HybridQueryEngine(index, config, metadata_filter=metadata_filter)


def _compute_confidence(source_nodes) -> float | None:
    """Compute confidence from source node scores, normalized to 0-1.

    Uses average of top-N scores. Reranker scores are typically 0-1 already;
    raw cosine similarity may need clamping.
    """
    if not source_nodes:
        return None

    scores = [n.score for n in source_nodes if hasattr(n, 'score') and n.score is not None]
    if not scores:
        return None

    # Use top 5 scores for confidence (most relevant results)
    top_scores = sorted(scores, reverse=True)[:5]
    avg = sum(top_scores) / len(top_scores)

    # Clamp to 0-1 range
    return round(max(0.0, min(1.0, avg)), 4)


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
            gap_analysis_top_n=config.retrieval.research_gap_analysis_top_n,
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
