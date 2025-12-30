"""Federated query engine for querying multiple indexes."""
import logging
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate

from config import RAGConfig
from query_engine import (
    LRUCache,
    _get_query_hash,
    PTCF_TEMPLATE,
    REFINE_TEMPLATE,
    QueryTransformRetriever,
    GraphEnhancedRetriever
)
from query_transform import QueryTransformer
from self_correction import SelfCorrectingRetriever

logger = logging.getLogger(__name__)

try:
    from llama_index.retrievers.bm25 import BM25Retriever
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


# Enhanced prompt template for federated retrieval
FEDERATED_TEMPLATE = """You are a personal knowledge assistant with access to both personal notes and AI conversation history.

Context from personal knowledge base:
---------------------
{context_str}
---------------------

Some context may come from past AI conversations (marked with source: ai_conversation).
Treat these as reference material that may contain useful information.

Instructions:
- PREMISE: What does the user already know based on their vault and past conversations?
- TASK: What specific question or need does the user have?
- CONSTRAINTS: Only use information from the provided context. If uncertain, acknowledge limitations.
- FORMAT: Provide clear, well-structured answers. Cite sources when relevant (note titles or conversation dates).

Query: {query_str}

Think through the relevant concepts, then provide a comprehensive answer:
"""


@dataclass
class IndexSource:
    """Represents a source index in the federation."""
    name: str
    index: VectorStoreIndex
    source_type: Literal["vault", "conversations"]
    weight: float = 1.0  # Score multiplier for this source
    nodes: Optional[List] = None  # For BM25 retriever
    wikilink_graph: Optional[Dict[str, List[str]]] = None


class FederatedRetriever(BaseRetriever):
    """Retriever that queries multiple indexes and merges results."""

    def __init__(
        self,
        sources: List[IndexSource],
        config: RAGConfig,
        query_transformer: Optional[QueryTransformer] = None,
        reranker=None,
        top_k_per_source: Optional[int] = None,
        final_top_k: Optional[int] = None,
        parallel: bool = True
    ):
        """
        Initialize federated retriever.

        Args:
            sources: List of IndexSource objects to query
            config: RAG configuration
            query_transformer: Optional query transformer
            reranker: Optional reranker model
            top_k_per_source: How many results to get from each source (default: config.top_k)
            final_top_k: Final number of results after merge (default: config.top_k)
            parallel: Whether to query sources in parallel
        """
        self.sources = sources
        self.config = config
        self.query_transformer = query_transformer
        self.reranker = reranker
        self.top_k_per_source = top_k_per_source or config.retrieval.top_k
        self.final_top_k = final_top_k or config.retrieval.top_k
        self.parallel = parallel

        # Build retrievers for each source
        self.retrievers: Dict[str, BaseRetriever] = {}
        self._build_retrievers()

        super().__init__()

    def _build_retrievers(self):
        """Build retriever for each source."""
        for source in self.sources:
            # Base vector retriever
            base_retriever = VectorIndexRetriever(
                index=source.index,
                similarity_top_k=self.top_k_per_source
            )

            retriever = base_retriever

            # Add BM25 fusion for hybrid search if nodes available
            if BM25_AVAILABLE and source.nodes and self.config.retrieval.enable_hybrid_search:
                try:
                    from llama_index.core.retrievers import QueryFusionRetriever

                    bm25_retriever = BM25Retriever.from_defaults(
                        nodes=source.nodes,
                        similarity_top_k=self.top_k_per_source
                    )

                    retriever = QueryFusionRetriever(
                        retrievers=[base_retriever, bm25_retriever],
                        similarity_top_k=self.top_k_per_source,
                        num_queries=1,
                        mode="reciprocal_rerank",
                        use_async=False
                    )
                    logger.info(f"Built hybrid retriever for source: {source.name}")
                except Exception as e:
                    logger.warning(f"Failed to build hybrid retriever for {source.name}: {e}")

            # Add graph enhancement for vault sources
            if source.source_type == "vault" and source.wikilink_graph and self.config.retrieval.enable_graph_retrieval:
                retriever = GraphEnhancedRetriever(
                    base_retriever=retriever,
                    wikilink_graph=source.wikilink_graph,
                    index=source.index,
                    depth=self.config.retrieval.graph_retrieval_depth,
                    max_links=self.config.retrieval.graph_retrieval_max_links
                )
                logger.info(f"Added graph enhancement for source: {source.name}")

            # Add query transformation
            if self.query_transformer and self.config.retrieval.query_transform_method not in ["none", "disabled"]:
                retriever = QueryTransformRetriever(
                    base_retriever=retriever,
                    query_transformer=self.query_transformer,
                    transform_method=self.config.retrieval.query_transform_method,
                    num_queries=self.config.retrieval.query_transform_num_queries
                )

            self.retrievers[source.name] = retriever
            logger.info(f"Built retriever for source: {source.name} (type: {source.source_type})")

    def _retrieve_from_source(
        self,
        source: IndexSource,
        query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """Retrieve from a single source."""
        try:
            retriever = self.retrievers[source.name]
            nodes = retriever.retrieve(query_bundle)

            # Apply source weight and tag
            weighted_nodes = []
            for node in nodes:
                # Tag with source info
                node.node.metadata['retrieval_source'] = source.name
                node.node.metadata['source_type'] = source.source_type

                # Apply weight
                original_score = node.score or 0.0
                weighted_score = original_score * source.weight

                weighted_nodes.append(NodeWithScore(
                    node=node.node,
                    score=weighted_score
                ))

            logger.debug(f"Retrieved {len(weighted_nodes)} nodes from {source.name}")
            return weighted_nodes

        except Exception as e:
            logger.error(f"Error retrieving from {source.name}: {e}")
            return []

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve from all sources and merge results."""
        all_nodes: List[NodeWithScore] = []

        if self.parallel and len(self.sources) > 1:
            # Parallel retrieval
            with ThreadPoolExecutor(max_workers=len(self.sources)) as executor:
                futures = {
                    executor.submit(
                        self._retrieve_from_source, source, query_bundle
                    ): source
                    for source in self.sources
                }

                for future in futures:
                    try:
                        nodes = future.result(timeout=30)
                        all_nodes.extend(nodes)
                    except Exception as e:
                        source = futures[future]
                        logger.error(f"Timeout/error retrieving from {source.name}: {e}")
        else:
            # Sequential retrieval
            for source in self.sources:
                nodes = self._retrieve_from_source(source, query_bundle)
                all_nodes.extend(nodes)

        # Deduplicate by node_id (same chunk might exist in multiple indexes)
        seen_ids = set()
        unique_nodes = []
        for node in all_nodes:
            node_id = node.node.node_id
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_nodes.append(node)

        # Sort by score descending
        unique_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)

        # Limit to final_top_k
        result_nodes = unique_nodes[:self.final_top_k]

        logger.info(
            f"Federated retrieval: {len(all_nodes)} total -> "
            f"{len(unique_nodes)} unique -> {len(result_nodes)} final"
        )

        return result_nodes


class FederatedQueryEngine:
    """Query engine that queries multiple indexes with unified response generation."""

    def __init__(
        self,
        sources: List[IndexSource],
        config: RAGConfig,
        reranker=None,
        query_transformer: Optional[QueryTransformer] = None,
        cache_size: int = 100,
        use_cache: bool = True,
        source_filter: Optional[List[str]] = None
    ):
        """
        Initialize federated query engine.

        Args:
            sources: List of IndexSource objects
            config: RAG configuration
            reranker: Optional reranker model
            query_transformer: Optional query transformer
            cache_size: LRU cache size
            use_cache: Whether to cache queries
            source_filter: Optional list of source names to query (None = all)
        """
        self.sources = sources
        self.config = config
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.use_cache = use_cache
        self.query_cache = LRUCache(max_size=cache_size) if use_cache else None
        self.source_filter = source_filter

        # Filter sources if specified
        if source_filter:
            self.active_sources = [s for s in sources if s.name in source_filter]
        else:
            self.active_sources = sources

        self.query_engine = self._build_query_engine()

        logger.info(
            f"FederatedQueryEngine initialized with {len(self.active_sources)} sources: "
            f"{[s.name for s in self.active_sources]}"
        )

    def _build_query_engine(self) -> RetrieverQueryEngine:
        """Build the query engine with federated retriever."""
        from llama_index.core.postprocessor import SimilarityPostprocessor

        # Create federated retriever
        federated_retriever = FederatedRetriever(
            sources=self.active_sources,
            config=self.config,
            query_transformer=self.query_transformer,
            reranker=self.reranker
        )

        # Wrap with self-correction if enabled
        retriever = federated_retriever
        if self.config.retrieval.use_self_correction:
            from llama_index.core import Settings
            retriever = SelfCorrectingRetriever(
                base_retriever=federated_retriever,
                llm=Settings.llm,
                max_retries=self.config.retrieval.self_correction_max_retries,
                enable_correction=True
            )
            logger.info("Self-correcting retrieval enabled for federated engine")

        # Post-processors
        node_postprocessors = []

        if self.reranker:
            node_postprocessors.append(self.reranker)
            logger.info("Reranker enabled for federated engine")

        # Similarity filter (only if no reranker)
        if self.config.retrieval.similarity_threshold > 0 and not self.reranker:
            node_postprocessors.append(
                SimilarityPostprocessor(
                    similarity_cutoff=self.config.retrieval.similarity_threshold
                )
            )

        # Response synthesizer with federated prompt
        qa_prompt = PromptTemplate(FEDERATED_TEMPLATE)
        refine_prompt = PromptTemplate(REFINE_TEMPLATE)

        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=qa_prompt,
            refine_template=refine_prompt,
            use_async=False
        )

        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors
        )

    def query(
        self,
        query_str: str,
        use_cache: Optional[bool] = None,
        source_filter: Optional[List[str]] = None
    ):
        """
        Execute federated query.

        Args:
            query_str: Query string
            use_cache: Override cache setting for this query
            source_filter: Filter to specific sources for this query

        Returns:
            Query response with source attribution
        """
        logger.info(f"Executing federated query: {query_str[:100]}...")

        # If source filter changed, rebuild engine
        if source_filter and set(source_filter) != set(s.name for s in self.active_sources):
            logger.info(f"Rebuilding engine with source filter: {source_filter}")
            self.active_sources = [s for s in self.sources if s.name in source_filter]
            self.query_engine = self._build_query_engine()

        should_use_cache = use_cache if use_cache is not None else self.use_cache

        # Check cache
        if should_use_cache and self.query_cache:
            cache_key = _get_query_hash(query_str + str(source_filter or []))
            cached = self.query_cache.get(cache_key)
            if cached:
                logger.info("Returning cached federated result")
                return cached

        try:
            response = self.query_engine.query(query_str)

            # Add source summary to response metadata
            if hasattr(response, 'source_nodes'):
                source_summary = self._summarize_sources(response.source_nodes)
                response.metadata = response.metadata or {}
                response.metadata['source_summary'] = source_summary

            # Cache result
            if should_use_cache and self.query_cache:
                cache_key = _get_query_hash(query_str + str(source_filter or []))
                self.query_cache.set(cache_key, response)

            return response

        except Exception as e:
            logger.error(f"Federated query failed: {e}", exc_info=True)
            raise RuntimeError(f"Federated query failed: {e}") from e

    def _summarize_sources(self, nodes: List[NodeWithScore]) -> Dict[str, Any]:
        """Summarize which sources contributed to the response."""
        summary = {
            "total_nodes": len(nodes),
            "by_source": {},
            "by_type": {"vault": 0, "conversations": 0}
        }

        for node in nodes:
            source = node.node.metadata.get('retrieval_source', 'unknown')
            source_type = node.node.metadata.get('source_type', 'unknown')

            if source not in summary["by_source"]:
                summary["by_source"][source] = 0
            summary["by_source"][source] += 1

            if source_type in summary["by_type"]:
                summary["by_type"][source_type] += 1

        return summary

    def query_vault_only(self, query_str: str) -> Any:
        """Query only vault sources."""
        vault_sources = [s.name for s in self.sources if s.source_type == "vault"]
        return self.query(query_str, source_filter=vault_sources)

    def query_conversations_only(self, query_str: str) -> Any:
        """Query only conversation sources."""
        conv_sources = [s.name for s in self.sources if s.source_type == "conversations"]
        return self.query(query_str, source_filter=conv_sources)

    def get_source_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed sources."""
        stats = {}
        for source in self.sources:
            stats[source.name] = {
                "type": source.source_type,
                "weight": source.weight,
                "active": source.name in [s.name for s in self.active_sources]
            }
        return stats
