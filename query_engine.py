"""Query engine with advanced retrieval strategies."""
import logging
from typing import List, Optional, Dict, Set
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle
from config import RAGConfig
from query_transform import QueryTransformer
from self_correction import SelfCorrectingRetriever

logger = logging.getLogger(__name__)

try:
    from llama_index.retrievers.bm25 import BM25Retriever
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("BM25Retriever not available. Install with: pip install llama-index-retrievers-bm25")


# Import additional dependencies for caching
import hashlib
from typing import Any
from collections import OrderedDict


class LRUCache:
    """Simple LRU (Least Recently Used) cache for query results."""

    def __init__(self, max_size: int = 100):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        logger.info(f"Query cache initialized with max size: {max_size}")

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache, moving it to end (most recently used).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            logger.debug(f"Cache hit for query: {key[:50]}...")
            return self.cache[key]
        logger.debug(f"Cache miss for query: {key[:50]}...")
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Add item to cache, removing oldest if at max size.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.max_size:
                # Remove oldest item (first item)
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
                logger.debug(f"Evicted oldest cache entry: {oldest_key[:50]}...")

        self.cache[key] = value
        logger.debug(f"Cached query result: {key[:50]}...")

    def clear(self) -> None:
        """Clear all cached items."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {count} cached query results")

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache count and max size
        """
        return {
            "count": len(self.cache),
            "max_size": self.max_size,
            "utilization_pct": round((len(self.cache) / self.max_size) * 100, 2) if self.max_size > 0 else 0
        }


# PTCF Prompting Template (from the strategy document)
PTCF_TEMPLATE = """You are a personal knowledge assistant for an Obsidian vault. Use the retrieved context to provide accurate, detailed answers.

Context information from the knowledge base:
---------------------
{context_str}
---------------------

Instructions:
- PREMISE: What does the user already know based on their vault?
- TASK: What specific question or need does the user have?
- CONSTRAINTS: Only use information from the provided context. If uncertain, acknowledge limitations.
- FORMAT: Provide clear, well-structured answers with references to specific notes when relevant.

Query: {query_str}

Think through the relevant concepts from the knowledge base, then provide a comprehensive answer:
"""


REFINE_TEMPLATE = """The original query is as follows: {query_str}

We have provided an existing answer: {existing_answer}

We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context_msg}
------------

Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.
"""


class GraphEnhancedRetriever(BaseRetriever):
    """Retriever that expands results using wikilink graph connections."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        wikilink_graph: Dict[str, List[str]],
        index: VectorStoreIndex,
        depth: int = 1,
        max_links: int = 3
    ):
        """
        Initialize graph-enhanced retriever.

        Args:
            base_retriever: Base retriever (vector or hybrid)
            wikilink_graph: Dictionary mapping file_path to list of linked file_paths
            index: Vector store index for retrieving linked documents
            depth: Number of hops to traverse (default: 1)
            max_links: Maximum number of links to follow per node (default: 3)
        """
        self.base_retriever = base_retriever
        self.wikilink_graph = wikilink_graph
        self.index = index
        self.depth = depth
        self.max_links = max_links
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes with graph expansion."""
        # Get base results from underlying retriever
        base_nodes = self.base_retriever.retrieve(query_bundle)

        if not self.wikilink_graph:
            logger.debug("No wikilink graph available, returning base results")
            return base_nodes

        # Track which file paths we've already retrieved
        retrieved_paths: Set[str] = set()
        for node in base_nodes:
            file_path = node.metadata.get('file_path')
            if file_path:
                retrieved_paths.add(file_path)

        # Expand with linked nodes
        linked_paths: Set[str] = set()

        for node in base_nodes:
            file_path = node.metadata.get('file_path')
            if not file_path or file_path not in self.wikilink_graph:
                continue

            # Get linked notes (limited by max_links)
            links = self.wikilink_graph[file_path][:self.max_links]
            for linked_path in links:
                if linked_path not in retrieved_paths:
                    linked_paths.add(linked_path)

        if not linked_paths:
            logger.debug("No linked nodes found, returning base results")
            return base_nodes

        logger.info(f"Graph expansion: {len(base_nodes)} base nodes -> {len(linked_paths)} linked documents")

        # Retrieve nodes from linked documents
        # We'll use the docstore to get nodes by file_path metadata
        linked_nodes = []
        docstore = self.index.docstore

        for doc_id, node in docstore.docs.items():
            node_file_path = node.metadata.get('file_path')
            if node_file_path in linked_paths:
                # Create NodeWithScore with lower score to rank below base results
                # Use minimum base score - 0.1 as the score for linked nodes
                min_base_score = min(n.score for n in base_nodes) if base_nodes else 0.5
                linked_node_score = max(0.0, min_base_score - 0.1)

                linked_nodes.append(
                    NodeWithScore(
                        node=node,
                        score=linked_node_score
                    )
                )

        logger.debug(f"Retrieved {len(linked_nodes)} nodes from {len(linked_paths)} linked documents")

        # Combine base results with linked results
        # Base results come first (higher scores), then linked results
        combined_nodes = base_nodes + linked_nodes

        return combined_nodes


def _get_query_hash(query_str: str) -> str:
    """Generate a hash for a query string to use as cache key."""
    return hashlib.sha256(query_str.encode()).hexdigest()


class QueryTransformRetriever(BaseRetriever):
    """Retriever wrapper that applies query transformation before retrieval."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        query_transformer: Optional[QueryTransformer] = None,
        transform_method: str = "hyde",
        num_queries: int = 3
    ):
        """
        Initialize query transform retriever.

        Args:
            base_retriever: Underlying retriever to use after transformation
            query_transformer: QueryTransformer instance (optional)
            transform_method: Transformation method to use
            num_queries: Number of query variations for multi-query
        """
        self.base_retriever = base_retriever
        self.query_transformer = query_transformer
        self.transform_method = transform_method.lower()
        self.num_queries = num_queries
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes with query transformation."""
        original_query = query_bundle.query_str

        # If no transformer or method is 'none'/'disabled', use base retriever
        if self.query_transformer is None or self.transform_method in ["none", "disabled"]:
            logger.debug("Query transformation disabled, using original query")
            return self.base_retriever.retrieve(query_bundle)

        # Apply query transformation
        logger.info(f"Applying query transformation: {self.transform_method}")

        if self.transform_method == "hyde":
            # Generate hypothetical document and retrieve with it
            transformed_query = self.query_transformer.hyde_transform(original_query)
            logger.debug(f"HyDE transformed query (preview): {transformed_query[:200]}...")

            # Create new query bundle with transformed query
            transformed_bundle = QueryBundle(
                query_str=transformed_query,
                custom_embedding_strs=[transformed_query]
            )

            # Retrieve using hypothetical document
            nodes = self.base_retriever.retrieve(transformed_bundle)
            logger.info(f"Retrieved {len(nodes)} nodes using HyDE transformation")
            return nodes

        elif self.transform_method == "multi_query":
            # Generate multiple query variations and combine results
            query_variations = self.query_transformer.multi_query_expand(
                original_query,
                num_queries=self.num_queries
            )
            logger.info(f"Generated {len(query_variations)} query variations")

            # Retrieve with each variation
            all_nodes: Dict[str, NodeWithScore] = {}  # Use dict to deduplicate by node_id

            for i, query_var in enumerate(query_variations):
                logger.debug(f"Retrieving with variation {i+1}: {query_var[:100]}...")
                var_bundle = QueryBundle(query_str=query_var)
                var_nodes = self.base_retriever.retrieve(var_bundle)

                # Add nodes to dict (later queries can override scores)
                for node in var_nodes:
                    node_id = node.node.node_id
                    if node_id in all_nodes:
                        # Use reciprocal rank fusion scoring
                        # RRF score = sum(1 / (k + rank)) where k=60 is standard
                        existing_score = all_nodes[node_id].score or 0
                        new_score = node.score or 0
                        # Combine scores using max (we want best retrieval)
                        all_nodes[node_id].score = max(existing_score, new_score)
                    else:
                        all_nodes[node_id] = node

            # Convert back to list and sort by score
            combined_nodes = list(all_nodes.values())
            combined_nodes.sort(key=lambda x: x.score or 0, reverse=True)

            logger.info(f"Retrieved {len(combined_nodes)} unique nodes from multi-query expansion")
            return combined_nodes

        elif self.transform_method == "both":
            # Generate hypothetical documents for each query variation
            query_variations = self.query_transformer.multi_query_expand(
                original_query,
                num_queries=self.num_queries
            )
            logger.info(f"Generated {len(query_variations)} query variations for HyDE transformation")

            # Generate hypothetical document for each variation
            all_nodes: Dict[str, NodeWithScore] = {}

            for i, query_var in enumerate(query_variations):
                logger.debug(f"HyDE transform variation {i+1}/{len(query_variations)}")
                hyde_doc = self.query_transformer.hyde_transform(query_var)

                hyde_bundle = QueryBundle(
                    query_str=hyde_doc,
                    custom_embedding_strs=[hyde_doc]
                )

                var_nodes = self.base_retriever.retrieve(hyde_bundle)

                # Combine using same deduplication strategy
                for node in var_nodes:
                    node_id = node.node.node_id
                    if node_id in all_nodes:
                        existing_score = all_nodes[node_id].score or 0
                        new_score = node.score or 0
                        all_nodes[node_id].score = max(existing_score, new_score)
                    else:
                        all_nodes[node_id] = node

            # Convert and sort
            combined_nodes = list(all_nodes.values())
            combined_nodes.sort(key=lambda x: x.score or 0, reverse=True)

            logger.info(f"Retrieved {len(combined_nodes)} unique nodes from combined HyDE + multi-query")
            return combined_nodes

        else:
            logger.warning(f"Unknown transform method: {self.transform_method}, using original query")
            return self.base_retriever.retrieve(query_bundle)


class RAGQueryEngine:
    """Advanced query engine with retrieval and generation."""

    def __init__(
        self,
        index: VectorStoreIndex,
        config: RAGConfig,
        reranker=None,
        query_transformer: Optional[QueryTransformer] = None,
        cache_size: int = 100,
        use_cache: bool = True
    ):
        """
        Initialize RAG query engine.

        Args:
            index: Vector store index
            config: RAG configuration
            reranker: Optional reranker model
            query_transformer: Optional query transformer for HyDE/multi-query
            cache_size: Size of LRU cache for query results (default: 100)
            use_cache: Whether to enable query caching (default: True)
        """
        self.index = index
        self.config = config
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.use_cache = use_cache
        self.query_cache = LRUCache(max_size=cache_size) if use_cache else None
        self.query_engine = self._build_query_engine()
    
    def _build_query_engine(self) -> RetrieverQueryEngine:
        """Build query engine with retrievers and post-processors."""

        # Configure base retriever
        base_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.config.retrieval.top_k,
        )

        # Wrap with query transformation if enabled
        if self.query_transformer and self.config.retrieval.query_transform_method not in ["none", "disabled"]:
            logger.info(f"Query transformation enabled: {self.config.retrieval.query_transform_method}")
            retriever = QueryTransformRetriever(
                base_retriever=base_retriever,
                query_transformer=self.query_transformer,
                transform_method=self.config.retrieval.query_transform_method,
                num_queries=self.config.retrieval.query_transform_num_queries
            )
        else:
            logger.info("Query transformation disabled")
            retriever = base_retriever

        # Wrap with self-correction if enabled
        if self.config.retrieval.use_self_correction:
            from llama_index.core import Settings
            logger.info(f"Self-correcting retrieval enabled (max_retries={self.config.retrieval.self_correction_max_retries})")
            retriever = SelfCorrectingRetriever(
                base_retriever=retriever,
                llm=Settings.llm,
                max_retries=self.config.retrieval.self_correction_max_retries,
                enable_correction=True
            )
        else:
            logger.info("Self-correcting retrieval disabled")

        # Configure post-processors
        node_postprocessors = []
        
        # Add reranker if available
        if self.reranker:
            node_postprocessors.append(self.reranker)
        
        # Add similarity filter
        node_postprocessors.append(
            SimilarityPostprocessor(
                similarity_cutoff=self.config.retrieval.similarity_threshold
            )
        )
        
        # Configure response synthesizer with custom prompts
        qa_prompt = PromptTemplate(PTCF_TEMPLATE)
        refine_prompt = PromptTemplate(REFINE_TEMPLATE)
        
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=qa_prompt,
            refine_template=refine_prompt,
            use_async=False
        )
        
        # Build query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
        )
        
        return query_engine
    
    def query(self, query_str: str, use_cache: Optional[bool] = None, **kwargs):
        """
        Execute query and return response.

        Args:
            query_str: Query string
            use_cache: Override instance cache setting for this query (default: None, uses instance setting)
            **kwargs: Additional arguments passed to query engine

        Returns:
            Query response
        """
        logger.info(f"Executing query: {query_str[:100]}...")

        # Determine if we should use cache for this query
        should_use_cache = use_cache if use_cache is not None else self.use_cache

        # Check cache if enabled
        if should_use_cache and self.query_cache is not None:
            cache_key = _get_query_hash(query_str)
            cached_response = self.query_cache.get(cache_key)
            if cached_response is not None:
                logger.info("Returning cached query result")
                return cached_response

        try:
            response = self.query_engine.query(query_str)
            logger.debug(f"Query completed successfully, {len(response.source_nodes)} sources retrieved")

            # Store in cache if enabled
            if should_use_cache and self.query_cache is not None:
                cache_key = _get_query_hash(query_str)
                self.query_cache.set(cache_key, response)

            return response

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)

            # Provide helpful error messages based on error type
            if "api" in str(e).lower() or "key" in str(e).lower():
                raise RuntimeError(
                    "API error during query execution. Please check your API keys and network connection."
                ) from e
            elif "rate" in str(e).lower() or "quota" in str(e).lower():
                raise RuntimeError(
                    "API rate limit exceeded. Please wait a moment and try again."
                ) from e
            else:
                raise RuntimeError(f"Query execution failed: {e}") from e
    
    def streaming_query(self, query_str: str):
        """Execute streaming query."""
        streaming_response = self.query_engine.query(query_str)
        return streaming_response
    
    def get_relevant_nodes(self, query_str: str, top_k: Optional[int] = None):
        """Get relevant nodes without generation (for debugging)."""
        if top_k is None:
            top_k = self.config.retrieval.top_k
        
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )
        
        nodes = retriever.retrieve(query_str)
        return nodes


class HybridQueryEngine:
    """Hybrid retrieval combining vector + BM25 keyword search."""

    def __init__(
        self,
        index: VectorStoreIndex,
        config: RAGConfig,
        reranker=None,
        bm25_retriever: Optional['BM25Retriever'] = None,
        nodes: Optional[List] = None,
        wikilink_graph: Optional[Dict[str, List[str]]] = None,
        query_transformer: Optional[QueryTransformer] = None,
        cache_size: int = 100,
        use_cache: bool = True
    ):
        """
        Initialize hybrid query engine.

        Args:
            index: Vector store index
            config: RAG configuration
            reranker: Optional reranker model
            bm25_retriever: Pre-built BM25 retriever (optional)
            nodes: Document nodes for building BM25 retriever if not provided
            wikilink_graph: Optional wikilink graph for graph-enhanced retrieval
            query_transformer: Optional query transformer for HyDE/multi-query
            cache_size: Size of LRU cache for query results (default: 100)
            use_cache: Whether to enable query caching (default: True)
        """
        self.index = index
        self.config = config
        self.reranker = reranker
        self.bm25_retriever = bm25_retriever
        self.nodes = nodes
        self.wikilink_graph = wikilink_graph or {}
        self.query_transformer = query_transformer
        self.use_cache = use_cache
        self.query_cache = LRUCache(max_size=cache_size) if use_cache else None

        # Build BM25 retriever if needed
        if self.bm25_retriever is None and BM25_AVAILABLE and self.nodes:
            try:
                logger.info("Building BM25 retriever from nodes...")
                self.bm25_retriever = BM25Retriever.from_defaults(
                    nodes=self.nodes,
                    similarity_top_k=self.config.retrieval.top_k
                )
                logger.info(f"BM25 retriever initialized with {len(self.nodes)} nodes")
            except Exception as e:
                logger.warning(f"Failed to build BM25 retriever: {e}")
                self.bm25_retriever = None

        self.query_engine = self._build_query_engine()

    def _build_query_engine(self) -> RetrieverQueryEngine:
        """Build query engine with hybrid retrieval."""
        from llama_index.core.retrievers import QueryFusionRetriever

        # Vector retriever
        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.config.retrieval.top_k,
        )

        # Prepare retrievers list
        retrievers = [vector_retriever]

        # Add BM25 retriever if available
        if self.bm25_retriever is not None:
            retrievers.append(self.bm25_retriever)
            logger.info("Hybrid search enabled: Vector + BM25 (fusion mode: reciprocal_rerank)")
        else:
            logger.info("BM25 not available - using vector retrieval only with query variations")

        # Create fusion retriever with reciprocal rank fusion
        fusion_retriever = QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=self.config.retrieval.top_k,
            num_queries=3,  # Generate 3 variations of the query
            mode="reciprocal_rerank",  # Use reciprocal rank fusion
            use_async=False
        )

        # Wrap with graph-enhanced retriever if enabled and graph is available
        if self.config.retrieval.enable_graph_retrieval and self.wikilink_graph:
            logger.info(f"Graph-enhanced retrieval enabled (depth={self.config.retrieval.graph_retrieval_depth}, max_links={self.config.retrieval.graph_retrieval_max_links})")
            base_retriever = GraphEnhancedRetriever(
                base_retriever=fusion_retriever,
                wikilink_graph=self.wikilink_graph,
                index=self.index,
                depth=self.config.retrieval.graph_retrieval_depth,
                max_links=self.config.retrieval.graph_retrieval_max_links
            )
        else:
            base_retriever = fusion_retriever
            if self.config.retrieval.enable_graph_retrieval:
                logger.info("Graph retrieval enabled but no wikilink graph available")

        # Wrap with query transformation if enabled
        if self.query_transformer and self.config.retrieval.query_transform_method not in ["none", "disabled"]:
            logger.info(f"Query transformation enabled: {self.config.retrieval.query_transform_method}")
            retriever = QueryTransformRetriever(
                base_retriever=base_retriever,
                query_transformer=self.query_transformer,
                transform_method=self.config.retrieval.query_transform_method,
                num_queries=self.config.retrieval.query_transform_num_queries
            )
        else:
            logger.info("Query transformation disabled")
            retriever = base_retriever

        # Wrap with self-correction if enabled
        if self.config.retrieval.use_self_correction:
            from llama_index.core import Settings
            logger.info(f"Self-correcting retrieval enabled (max_retries={self.config.retrieval.self_correction_max_retries})")
            retriever = SelfCorrectingRetriever(
                base_retriever=retriever,
                llm=Settings.llm,
                max_retries=self.config.retrieval.self_correction_max_retries,
                enable_correction=True
            )
        else:
            logger.info("Self-correcting retrieval disabled")

        # Configure post-processors
        node_postprocessors = []

        # Add reranker if available
        if self.reranker:
            node_postprocessors.append(self.reranker)
            logger.info(f"Reranker enabled: {self.config.retrieval.reranker_model}")
        else:
            logger.info("Reranker disabled (None)")

        # Add similarity filter (disabled by default - reranker handles relevance)
        # LanceDB cosine similarity scores may not align with traditional 0-1 range
        if self.config.retrieval.similarity_threshold > 0 and not self.reranker:
            node_postprocessors.append(
                SimilarityPostprocessor(
                    similarity_cutoff=self.config.retrieval.similarity_threshold
                )
            )
            logger.info(f"Similarity filter enabled: cutoff={self.config.retrieval.similarity_threshold}")
        else:
            logger.info("Similarity filter disabled (reranker handles relevance)")

        # Configure response synthesizer with custom prompts
        qa_prompt = PromptTemplate(PTCF_TEMPLATE)
        refine_prompt = PromptTemplate(REFINE_TEMPLATE)

        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=qa_prompt,
            refine_template=refine_prompt,
            use_async=False
        )

        # Build query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors
        )

        return query_engine

    def query(self, query_str: str, use_cache: Optional[bool] = None):
        """
        Execute hybrid query with fusion.

        Args:
            query_str: Query string
            use_cache: Override instance cache setting for this query (default: None, uses instance setting)

        Returns:
            Query response
        """
        logger.info(f"Executing hybrid query: {query_str[:100]}...")

        # Determine if we should use cache for this query
        should_use_cache = use_cache if use_cache is not None else self.use_cache

        # Check cache if enabled
        if should_use_cache and self.query_cache is not None:
            cache_key = _get_query_hash(query_str)
            cached_response = self.query_cache.get(cache_key)
            if cached_response is not None:
                logger.info("Returning cached hybrid query result")
                return cached_response

        try:
            response = self.query_engine.query(query_str)
            logger.debug(f"Hybrid query completed successfully")

            # Store in cache if enabled
            if should_use_cache and self.query_cache is not None:
                cache_key = _get_query_hash(query_str)
                self.query_cache.set(cache_key, response)

            return response

        except Exception as e:
            logger.error(f"Hybrid query failed: {e}", exc_info=True)

            # Provide helpful error messages based on error type
            if "api" in str(e).lower() or "key" in str(e).lower():
                raise RuntimeError(
                    "API error during query execution. Please check your API keys and network connection."
                ) from e
            elif "rate" in str(e).lower() or "quota" in str(e).lower():
                raise RuntimeError(
                    "API rate limit exceeded. Please wait a moment and try again."
                ) from e
            else:
                raise RuntimeError(f"Hybrid query execution failed: {e}") from e
