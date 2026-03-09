"""Federated query engine for querying multiple indexes."""
import logging
import hashlib
import re
from collections import defaultdict
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


def _get_federated_template(source_types: List[str]) -> str:
    """Generate appropriate prompt template based on active sources."""
    has_vault = "vault" in source_types
    has_conv = "conversations" in source_types
    has_web = "web" in source_types
    has_books = "books" in source_types
    has_saved = "saved_items" in source_types

    # Build intro based on available sources
    sources_list = []
    if has_vault:
        sources_list.append("personal notes from an Obsidian vault")
    if has_conv:
        sources_list.append("AI conversation history")
    if has_books:
        sources_list.append("book library (EPUB/PDF)")
    if has_saved:
        sources_list.append("saved links and articles from various platforms")
    if has_web:
        sources_list.append("web search results")

    if len(sources_list) > 1:
        intro = f"You are a personal knowledge assistant with access to {', '.join(sources_list[:-1])} and {sources_list[-1]}."
    elif sources_list:
        intro = f"You are a knowledge assistant analyzing {sources_list[0]}."
    else:
        intro = "You are a personal knowledge assistant."

    # Build notes about source types
    notes = []
    if has_conv:
        notes.append("Some context may come from past AI conversations (marked with source_type: conversations).\nTreat these as reference material that may contain useful information.")
    if has_books:
        notes.append("Some context comes from books in the library (marked with source_type: books).\nBook sources include title and chapter information.")
    if has_saved:
        notes.append("Some context comes from saved links/articles (marked with source_type: saved_items).\nThese are curated saves — content may be summaries or excerpts, not full articles.")
    if has_web:
        notes.append("Some context comes from web search results (marked with source_type: web).\nWeb sources are marked with their URL for reference.")

    note_section = "\n".join(notes)
    if note_section:
        note_section = "\n" + note_section

    return f"""{intro}

Context from knowledge base (numbered for citation):
---------------------
{{context_str}}
---------------------
{note_section}

Instructions:
- TASK: Answer the user's question using ONLY the provided context.
- CONSTRAINTS: If the context doesn't contain relevant information, acknowledge this.
- CITATIONS: Use inline citations [1], [2], etc. to reference the numbered sources above.
  Example: "Habits form through repetition [3]."
- IMPORTANT: Do NOT add a "Sources" or "References" section at the end - sources will be displayed separately by the system.
- **USER OVERRIDE**: If the user specifies a format in their query (e.g., "as a table", "in Greek", "bullet points"), follow that format instead of the default.

Query: {{query_str}}

Provide a comprehensive answer with inline citations:
"""


# Default template (backwards compatibility)
FEDERATED_TEMPLATE = _get_federated_template(["vault", "conversations"])


@dataclass
class IndexSource:
    """Represents a source index in the federation."""
    name: str
    source_type: Literal["vault", "conversations", "web", "saved_items", "books"]
    index: Optional[VectorStoreIndex] = None
    weight: float = 1.0  # Score multiplier for this source
    nodes: Optional[List] = None  # For BM25 retriever
    wikilink_graph: Optional[Dict[str, List[str]]] = None
    custom_retriever: Optional[BaseRetriever] = None  # For flat-schema sources (e.g. saved_items)
    prebuilt_bm25: Optional[BaseRetriever] = None  # Pre-built BM25 retriever to reuse (avoids rebuild)

    def __post_init__(self):
        if self.index is None and self.custom_retriever is None:
            raise ValueError(
                f"IndexSource '{self.name}' must have either 'index' or 'custom_retriever'"
            )


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
        parallel: bool = True,
        book_filter=None,
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
            book_filter: Optional BookFilter for category/author filtering on books
        """
        self.sources = sources
        self.config = config
        self.query_transformer = query_transformer
        self.reranker = reranker
        self.top_k_per_source = top_k_per_source or config.retrieval.federated_top_k_per_source
        self.final_top_k = final_top_k or config.retrieval.federated_final_top_k
        self.max_chunks_per_document = config.retrieval.federated_max_chunks_per_document
        self.parallel = parallel
        self._book_filter = book_filter

        # Build retrievers for each source
        self.retrievers: Dict[str, BaseRetriever] = {}
        self._build_retrievers()

        super().__init__()

    def _build_retrievers(self):
        """Build retriever for each source."""
        for source in self.sources:
            # Custom retriever path (e.g. saved_items with flat schema — no LlamaIndex index)
            if source.custom_retriever is not None:
                retriever = source.custom_retriever
                # Still apply query transformation if configured
                if self.query_transformer and self.config.retrieval.query_transform_method not in ["none", "disabled"]:
                    retriever = QueryTransformRetriever(
                        base_retriever=retriever,
                        query_transformer=self.query_transformer,
                        transform_method=self.config.retrieval.query_transform_method,
                        num_queries=self.config.retrieval.query_transform_num_queries,
                        rrf_k=self.config.retrieval.rrf_k,
                    )
                self.retrievers[source.name] = retriever
                logger.info(f"Using custom retriever for source: {source.name} (type: {source.source_type})")
                continue

            # For books with active filter: use filtered retriever, skip BM25
            if source.source_type == "books" and self._book_filter:
                from books_retriever import get_books_retriever
                base_retriever = get_books_retriever(
                    source.index, self.top_k_per_source, self._book_filter
                )
                retriever = base_retriever
                # CRITICAL: Skip BM25 fusion when filter active.
                # BM25Retriever searches ALL source.nodes (no WHERE support),
                # which leaks out-of-filter results into QueryFusionRetriever.
                # Vector-only retrieval with LanceDB WHERE is correct here.
                logger.info("Books filter active — BM25 fusion disabled for books source")
            else:
                # Normal path: standard vector retriever
                base_retriever = VectorIndexRetriever(
                    index=source.index,
                    similarity_top_k=self.top_k_per_source
                )
                retriever = base_retriever

                # Add BM25 fusion for hybrid search — reuse prebuilt if available
                if BM25_AVAILABLE and self.config.retrieval.enable_hybrid_search and (source.prebuilt_bm25 or source.nodes):
                    try:
                        from llama_index.core.retrievers import QueryFusionRetriever

                        if source.prebuilt_bm25 is not None:
                            bm25_retriever = source.prebuilt_bm25
                            logger.info(f"Reusing prebuilt BM25 for source: {source.name}")
                        else:
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

        # Deduplicate by node_id and content signature (file_path + normalized text hash)
        seen_ids = set()
        seen_content_keys = set()
        unique_nodes = []
        for node in all_nodes:
            node_id = node.node.node_id
            if node_id not in seen_ids:
                content_key = self._content_dedupe_key(node)
                if content_key and content_key in seen_content_keys:
                    continue
                seen_ids.add(node_id)
                if content_key:
                    seen_content_keys.add(content_key)
                unique_nodes.append(node)

        # Sort by score descending
        unique_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)

        # Enforce source diversity by limiting chunks per document.
        diversified_nodes: List[NodeWithScore] = []
        per_doc_counts: dict[str, int] = defaultdict(int)
        for node in unique_nodes:
            doc_key = self._document_key(node)
            if doc_key:
                if per_doc_counts[doc_key] >= self.max_chunks_per_document:
                    continue
                per_doc_counts[doc_key] += 1
            diversified_nodes.append(node)

        # Limit to final_top_k after diversification
        result_nodes = diversified_nodes[:self.final_top_k]

        logger.info(
            f"Federated retrieval: {len(all_nodes)} total -> "
            f"{len(unique_nodes)} unique -> {len(diversified_nodes)} diversified -> {len(result_nodes)} final"
        )

        return result_nodes

    @staticmethod
    def _document_key(node_with_score: NodeWithScore) -> str:
        """Stable document identity used for diversity constraints."""
        node = node_with_score.node
        md = node.metadata or {}
        return (
            md.get("file_path")
            or md.get("doc_id")
            or md.get("document_id")
            or md.get("book_uid")
            or node.node_id
        )

    @staticmethod
    def _content_dedupe_key(node_with_score: NodeWithScore) -> str:
        """Content-level dedupe key: file identity + normalized chunk hash."""
        node = node_with_score.node
        md = node.metadata or {}
        file_path = (
            md.get("file_path")
            or md.get("doc_id")
            or md.get("document_id")
            or md.get("book_uid")
            or "unknown"
        )
        text = (node.text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        if not text:
            return ""
        text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"{file_path}:{text_hash}"


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
        source_filter: Optional[List[str]] = None,
        book_filter=None,
        metadata_filter=None,
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
            book_filter: Optional BookFilter for category/author filtering on books
            metadata_filter: Optional MetadataFilter for query-time filtering
        """
        self.sources = sources
        self.config = config
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.use_cache = use_cache
        self.query_cache = LRUCache(max_size=cache_size) if use_cache else None
        self.source_filter = source_filter
        self._book_filter = book_filter
        self._metadata_filter = metadata_filter

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
            reranker=self.reranker,
            book_filter=self._book_filter,
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

        # Metadata filter FIRST (reduces reranker workload)
        if self._metadata_filter:
            from query_engine import MetadataFilterPostprocessor
            node_postprocessors.append(
                MetadataFilterPostprocessor(
                    filter_tags=self._metadata_filter.tags,
                    filter_date_after=self._metadata_filter.date_after,
                    filter_date_before=self._metadata_filter.date_before,
                    filter_path_prefix=self._metadata_filter.path_prefix,
                )
            )
            logger.info("Metadata filter enabled for federated engine")

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

        # Post-rerank score threshold (separate scale from cosine similarity)
        if self.reranker and self.config.retrieval.rerank_score_threshold > 0:
            node_postprocessors.append(
                SimilarityPostprocessor(
                    similarity_cutoff=self.config.retrieval.rerank_score_threshold
                )
            )
            logger.info(f"Post-rerank threshold: {self.config.retrieval.rerank_score_threshold}")

        # Response synthesizer with dynamic prompt based on active sources
        source_types = [s.source_type for s in self.active_sources]
        qa_prompt = PromptTemplate(_get_federated_template(source_types))
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
            "by_type": {"vault": 0, "conversations": 0, "web": 0, "saved_items": 0, "books": 0}
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

    def retrieve(
        self,
        query_str: str,
        source_filter: Optional[List[str]] = None
    ) -> List[NodeWithScore]:
        """Retrieve nodes without synthesis (for manual context building).

        Args:
            query_str: Query string
            source_filter: Optional list of source names to query

        Returns:
            List of NodeWithScore objects (reranked)
        """
        logger.info(f"Federated retrieval (no synthesis): {query_str[:100]}...")

        # If source filter changed, rebuild engine
        if source_filter and set(source_filter) != set(s.name for s in self.active_sources):
            logger.info(f"Rebuilding engine with source filter: {source_filter}")
            self.active_sources = [s for s in self.sources if s.name in source_filter]
            self.query_engine = self._build_query_engine()

        # Use the retriever from the query engine
        nodes = self.query_engine.retriever.retrieve(query_str)
        logger.info(f"Retrieved {len(nodes)} nodes for manual synthesis")
        return nodes

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
