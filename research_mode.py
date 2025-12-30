"""Iterative retrieval / research mode for UltraRAG.

Implements multi-step retrieval with query refinement based on initial results.
Based on Khoj's research mode (141% accuracy improvement on benchmarks).
"""

import logging
from typing import List, Optional, Set, Dict, Any
from dataclasses import dataclass
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


@dataclass
class ResearchIteration:
    """Single iteration in the research process."""
    iteration: int
    query: str
    nodes: List[NodeWithScore]
    gaps_identified: Optional[str] = None
    confidence_score: float = 0.0


@dataclass
class ResearchResult:
    """Result of multi-step research process."""
    original_query: str
    final_nodes: List[NodeWithScore]
    iterations: List[ResearchIteration]
    total_iterations: int
    total_nodes_retrieved: int
    final_confidence: float

    def get_all_sources(self) -> List[str]:
        """Get unique list of all source documents across iterations."""
        sources = set()
        for node in self.final_nodes:
            file_path = node.metadata.get('file_path', 'Unknown')
            sources.add(file_path)
        return sorted(list(sources))

    def get_iteration_summary(self) -> str:
        """Get human-readable summary of research iterations."""
        lines = [
            f"Research completed in {self.total_iterations} iterations",
            f"Total unique nodes: {len(self.final_nodes)}",
            f"Final confidence: {self.final_confidence:.2f}",
            "\nIteration breakdown:"
        ]

        for iter_result in self.iterations:
            lines.append(
                f"  Iteration {iter_result.iteration}: "
                f"{len(iter_result.nodes)} nodes retrieved "
                f"(confidence: {iter_result.confidence_score:.2f})"
            )
            if iter_result.gaps_identified:
                lines.append(f"    Gaps: {iter_result.gaps_identified[:100]}...")

        return "\n".join(lines)


class ResearchRetriever:
    """Iterative retrieval with query refinement based on initial results.

    Multi-step research process:
    1. Initial retrieval with base retriever
    2. LLM analyzes gaps in retrieved content
    3. Generates refined sub-queries for missing information
    4. Retrieves again with sub-queries (up to max_iterations)
    5. Aggregates and deduplicates results across iterations
    6. Synthesizes final answer from all retrieved content
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm,
        max_iterations: int = 3,
        confidence_threshold: float = 0.8,
        max_subqueries: int = 3,
        enable_research: bool = True
    ):
        """Initialize research retriever.

        Args:
            base_retriever: Underlying retriever to use for each iteration
            llm: Language model for gap analysis and sub-query generation
            max_iterations: Maximum research iterations (default: 3)
            confidence_threshold: Stop if confidence exceeds this (default: 0.8)
            max_subqueries: Maximum sub-queries per iteration (default: 3)
            enable_research: Whether research mode is enabled (default: True)
        """
        self.base_retriever = base_retriever
        self.llm = llm
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.max_subqueries = max_subqueries
        self.enable_research = enable_research

        logger.info(
            f"ResearchRetriever initialized "
            f"(max_iterations={max_iterations}, "
            f"confidence_threshold={confidence_threshold}, "
            f"enabled={enable_research})"
        )

    def research(self, query: str) -> ResearchResult:
        """Execute multi-step research process.

        Args:
            query: Original user query

        Returns:
            ResearchResult with aggregated nodes and iteration details
        """
        if not self.enable_research:
            logger.info("Research mode disabled, using base retrieval")
            query_bundle = QueryBundle(query_str=query)
            nodes = self.base_retriever.retrieve(query_bundle)

            # Return single-iteration result
            iteration = ResearchIteration(
                iteration=1,
                query=query,
                nodes=nodes,
                confidence_score=1.0
            )

            return ResearchResult(
                original_query=query,
                final_nodes=nodes,
                iterations=[iteration],
                total_iterations=1,
                total_nodes_retrieved=len(nodes),
                final_confidence=1.0
            )

        logger.info(f"Starting research mode for query: {query[:100]}...")

        # Track all iterations
        iterations: List[ResearchIteration] = []

        # Track all retrieved nodes (deduplicate by node_id)
        all_nodes: Dict[str, NodeWithScore] = {}

        # Track which file paths we've already retrieved from
        retrieved_paths: Set[str] = set()

        current_query = query

        for iteration_num in range(1, self.max_iterations + 1):
            logger.info(f"Research iteration {iteration_num}/{self.max_iterations}")

            # Retrieve with current query
            query_bundle = QueryBundle(query_str=current_query)
            nodes = self.base_retriever.retrieve(query_bundle)

            logger.info(f"Iteration {iteration_num}: Retrieved {len(nodes)} nodes")

            # Add to deduplication tracking
            for node in nodes:
                node_id = node.node.node_id
                file_path = node.metadata.get('file_path')

                # Track unique nodes
                if node_id not in all_nodes:
                    all_nodes[node_id] = node
                else:
                    # Keep higher-scoring version
                    existing_score = all_nodes[node_id].score or 0
                    new_score = node.score or 0
                    if new_score > existing_score:
                        all_nodes[node_id] = node

                # Track file paths
                if file_path:
                    retrieved_paths.add(file_path)

            # Analyze gaps and compute confidence
            gaps, confidence = self._analyze_gaps(query, list(all_nodes.values()))

            logger.info(
                f"Iteration {iteration_num}: Confidence={confidence:.2f}, "
                f"Total unique nodes={len(all_nodes)}"
            )

            if gaps:
                logger.debug(f"Identified gaps: {gaps[:200]}...")

            # Store iteration result
            iteration_result = ResearchIteration(
                iteration=iteration_num,
                query=current_query,
                nodes=nodes,
                gaps_identified=gaps,
                confidence_score=confidence
            )
            iterations.append(iteration_result)

            # Check if we should stop
            if confidence >= self.confidence_threshold:
                logger.info(
                    f"Confidence threshold reached ({confidence:.2f} >= {self.confidence_threshold}), "
                    f"stopping research"
                )
                break

            # Check if we have more iterations available
            if iteration_num >= self.max_iterations:
                logger.info(f"Maximum iterations reached ({self.max_iterations})")
                break

            # Generate sub-queries for next iteration
            if gaps:
                subqueries = self._generate_subqueries(query, gaps, retrieved_paths)

                if not subqueries:
                    logger.info("No sub-queries generated, stopping research")
                    break

                # Use first sub-query for next iteration
                # (in a more advanced implementation, could retrieve in parallel)
                current_query = subqueries[0]
                logger.info(f"Next iteration query: {current_query[:100]}...")
            else:
                logger.info("No gaps identified, stopping research")
                break

        # Aggregate final results
        final_nodes = list(all_nodes.values())

        # Sort by score descending
        final_nodes.sort(key=lambda x: x.score or 0, reverse=True)

        final_confidence = iterations[-1].confidence_score if iterations else 0.0

        result = ResearchResult(
            original_query=query,
            final_nodes=final_nodes,
            iterations=iterations,
            total_iterations=len(iterations),
            total_nodes_retrieved=len(final_nodes),
            final_confidence=final_confidence
        )

        logger.info(
            f"Research completed: {result.total_iterations} iterations, "
            f"{result.total_nodes_retrieved} unique nodes, "
            f"confidence={result.final_confidence:.2f}"
        )

        return result

    def _analyze_gaps(
        self,
        query: str,
        nodes: List[NodeWithScore]
    ) -> tuple[Optional[str], float]:
        """Analyze gaps in retrieved content using LLM.

        Args:
            query: Original query
            nodes: Currently retrieved nodes

        Returns:
            Tuple of (gaps description, confidence score 0-1)
        """
        if not nodes:
            logger.debug("No nodes to analyze, returning low confidence")
            return "No relevant information found", 0.0

        # Build context from top nodes (limit to save tokens)
        context_chunks = []
        for idx, node in enumerate(nodes[:5], 1):
            chunk_text = node.node.text[:400]  # Truncate for token efficiency
            title = node.metadata.get('title', 'Unknown')
            context_chunks.append(f"[Source {idx}: {title}]\n{chunk_text}")

        context = "\n\n".join(context_chunks)

        prompt = f"""Analyze whether the retrieved information fully answers the user's query.

Query: {query}

Retrieved Information:
{context}

Evaluate:
1. Does this information fully answer the query?
2. What key aspects or details are missing (if any)?
3. What is your confidence that the query can be fully answered with this information?

Respond in this exact format:
CONFIDENCE: <number between 0.0 and 1.0>
GAPS: <brief description of missing information, or "None" if complete>

Example:
CONFIDENCE: 0.6
GAPS: Missing information about implementation details and code examples"""

        try:
            result = self.llm.complete(prompt).text.strip()
            logger.debug(f"Gap analysis result: {result[:200]}...")

            # Parse confidence score
            confidence = 0.5  # Default
            gaps = None

            for line in result.split('\n'):
                line = line.strip()
                if line.startswith('CONFIDENCE:'):
                    try:
                        confidence_str = line.split(':', 1)[1].strip()
                        confidence = float(confidence_str)
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse confidence: {e}")

                elif line.startswith('GAPS:'):
                    gaps_str = line.split(':', 1)[1].strip()
                    if gaps_str.lower() not in ['none', 'no gaps', 'n/a']:
                        gaps = gaps_str

            return gaps, confidence

        except Exception as e:
            logger.error(f"Error during gap analysis: {e}", exc_info=True)
            return "Analysis failed", 0.5

    def _generate_subqueries(
        self,
        original_query: str,
        gaps: str,
        retrieved_paths: Set[str]
    ) -> List[str]:
        """Generate refined sub-queries to fill information gaps.

        Args:
            original_query: Original user query
            gaps: Description of information gaps
            retrieved_paths: Set of file paths already retrieved (for context)

        Returns:
            List of refined sub-queries (up to max_subqueries)
        """
        # Build context about what we've already searched
        paths_context = ""
        if retrieved_paths:
            paths_list = sorted(list(retrieved_paths))[:10]  # Limit for token efficiency
            paths_context = f"\nAlready retrieved from: {', '.join(paths_list)}"

        prompt = f"""Generate {self.max_subqueries} targeted search queries to find missing information.

Original Query: {original_query}

Information Gaps: {gaps}
{paths_context}

Generate {self.max_subqueries} specific, targeted queries that would help find the missing information.
Each query should:
- Focus on a specific aspect of the gaps
- Use different keywords/phrasings than the original
- Be likely to match relevant documents

Respond with one query per line, numbered:
1. <first query>
2. <second query>
3. <third query>"""

        try:
            result = self.llm.complete(prompt).text.strip()
            logger.debug(f"Sub-query generation result: {result[:200]}...")

            # Parse queries
            subqueries = []
            for line in result.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Remove numbering (e.g., "1. ", "2. ")
                if line[0].isdigit() and '. ' in line:
                    line = line.split('. ', 1)[1]

                subqueries.append(line)

                if len(subqueries) >= self.max_subqueries:
                    break

            logger.info(f"Generated {len(subqueries)} sub-queries")
            return subqueries

        except Exception as e:
            logger.error(f"Error during sub-query generation: {e}", exc_info=True)
            return []

    def _aggregate_results(
        self,
        all_results: List[List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """Combine and deduplicate results from multiple iterations.

        Args:
            all_results: List of node lists from each iteration

        Returns:
            Deduplicated and ranked list of nodes
        """
        # Deduplicate by node_id
        nodes_by_id: Dict[str, NodeWithScore] = {}

        for iteration_nodes in all_results:
            for node in iteration_nodes:
                node_id = node.node.node_id

                if node_id not in nodes_by_id:
                    nodes_by_id[node_id] = node
                else:
                    # Keep higher-scoring version
                    existing_score = nodes_by_id[node_id].score or 0
                    new_score = node.score or 0

                    if new_score > existing_score:
                        nodes_by_id[node_id] = node

        # Convert to list and sort by score
        aggregated = list(nodes_by_id.values())
        aggregated.sort(key=lambda x: x.score or 0, reverse=True)

        logger.debug(
            f"Aggregated {len(aggregated)} unique nodes from "
            f"{sum(len(r) for r in all_results)} total retrievals"
        )

        return aggregated
