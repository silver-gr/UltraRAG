"""Iterative retrieval / research mode for UltraRAG.

Implements multi-step retrieval with query refinement based on initial results.
Based on Khoj's research mode (141% accuracy improvement on benchmarks).
"""

import logging
import re
from typing import List, Optional, Set, Dict, Any
from dataclasses import dataclass
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever

# LangSmith observability (opt-in, no-op if not configured)
from observability import trace_chain, trace_span, is_tracing_enabled

logger = logging.getLogger(__name__)


@dataclass
class ResearchIteration:
    """Single iteration in the research process."""
    iteration: int
    query: str
    nodes: List[NodeWithScore]
    gaps_identified: Optional[str] = None
    confidence_score: float = 0.0
    full_analysis: Optional[str] = None  # Full gap analysis LLM response


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
                lines.append(f"    Gaps: {iter_result.gaps_identified}")

        return "\n".join(lines)

    def get_gap_analyses(self) -> List[Dict[str, Any]]:
        """Get all gap analyses from iterations with full context."""
        analyses = []
        for iter_result in self.iterations:
            if iter_result.full_analysis:
                analyses.append({
                    'iteration': iter_result.iteration,
                    'query': iter_result.query,
                    'confidence': iter_result.confidence_score,
                    'gaps': iter_result.gaps_identified,
                    'full_analysis': iter_result.full_analysis,
                    'nodes_retrieved': len(iter_result.nodes)
                })
        return analyses

    def get_gap_analyses_markdown(self) -> str:
        """Get gap analyses formatted as markdown for display."""
        analyses = self.get_gap_analyses()
        if not analyses:
            return ""

        lines = ["## Gap Analysis Summary\n"]
        for analysis in analyses:
            lines.append(f"### Iteration {analysis['iteration']}")
            lines.append(f"**Query:** {analysis['query']}")
            lines.append(f"**Confidence:** {analysis['confidence']:.2f}")
            lines.append(f"**Nodes Retrieved:** {analysis['nodes_retrieved']}")
            if analysis['full_analysis']:
                # Clean up the analysis text for display
                full_text = analysis['full_analysis']
                # Remove the CONFIDENCE/GAPS prefix lines, keep the insights
                lines_to_keep = []
                for line in full_text.split('\n'):
                    # Skip the structured prefix lines
                    if line.strip().upper().startswith('CONFIDENCE:'):
                        continue
                    if line.strip().upper().startswith('GAPS:'):
                        continue
                    lines_to_keep.append(line)
                cleaned = '\n'.join(lines_to_keep).strip()
                if cleaned:
                    lines.append(f"\n**Analysis:**\n{cleaned}")
            lines.append("\n---\n")

        return '\n'.join(lines)


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

    # Patterns that indicate exhaustive/comprehensive retrieval is needed
    EXHAUSTIVE_PATTERNS = [
        r'\ball\b',                    # "all habits", "list all"
        r'\bevery\b',                  # "every routine"
        r'\bcomplete\s+list\b',        # "complete list of"
        r'\bcomprehensive\b',          # "comprehensive overview"
        r'\bexhaustive\b',             # "exhaustive list"
        r'\bentire\b',                 # "entire collection"
        r'\bfull\s+list\b',            # "full list"
        r'\bόλα\b|\bόλες\b|\bόλους\b', # Greek: all (neuter/feminine/masculine)
        r'\bκάθε\b',                   # Greek: every
        r'\bπλήρης?\b',                # Greek: complete/full
    ]

    # Maximum iterations for exhaustive queries (overrides config)
    EXHAUSTIVE_MAX_ITERATIONS = 5

    # Delay between iterations to avoid rate limiting (seconds)
    # Set to 0 to disable rate limiting delays
    ITERATION_DELAY = 0

    # Model for gap analysis (same as main LLM for consistency)
    GAP_ANALYSIS_MODEL = "gemini-3-flash-preview"

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

        # Create lightweight LLM for gap analysis (reduces rate limiting)
        self.gap_analysis_llm = self._create_gap_analysis_llm()

        logger.info(
            f"ResearchRetriever initialized "
            f"(max_iterations={max_iterations}, "
            f"confidence_threshold={confidence_threshold}, "
            f"enabled={enable_research}, "
            f"gap_analysis_model={self.GAP_ANALYSIS_MODEL})"
        )

    def _create_gap_analysis_llm(self):
        """Create a lightweight LLM specifically for gap analysis.

        Uses a faster model with AFC disabled to reduce rate limiting.
        """
        try:
            import os
            from llama_index.llms.google_genai import GoogleGenAI
            from tracked_llm import wrap_llm_with_tracking

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("No GOOGLE_API_KEY found, falling back to main LLM for gap analysis")
                return self.llm

            # Create lightweight LLM with AFC disabled
            base_llm = GoogleGenAI(
                model=self.GAP_ANALYSIS_MODEL,
                api_key=api_key,
                temperature=0.1,
                max_tokens=4096,  # Increased: model may provide detailed analysis with large context
                is_function_calling_model=False,  # Disable AFC
            )
            # Wrap with token tracking
            gap_llm = wrap_llm_with_tracking(base_llm, model_name=self.GAP_ANALYSIS_MODEL)
            logger.info(f"Created gap analysis LLM: {self.GAP_ANALYSIS_MODEL} (AFC disabled, tracking enabled)")
            return gap_llm
        except Exception as e:
            logger.warning(f"Failed to create gap analysis LLM: {e}, falling back to main LLM")
            return self.llm

    def _is_exhaustive_query(self, query: str) -> bool:
        """Detect if query requests exhaustive/comprehensive results.

        Args:
            query: User query string

        Returns:
            True if query contains patterns indicating exhaustive retrieval needed
        """
        query_lower = query.lower()
        for pattern in self.EXHAUSTIVE_PATTERNS:
            if re.search(pattern, query_lower):
                logger.info(f"Detected exhaustive query pattern: {pattern}")
                return True
        return False

    @trace_chain
    def research(self, query: str, force_exhaustive: bool = False) -> ResearchResult:
        """Execute multi-step research process.

        Args:
            query: Original user query
            force_exhaustive: If True, run all iterations regardless of confidence
                              (also triggered by @all prefix or exhaustive query patterns)

        Returns:
            ResearchResult with aggregated nodes and iteration details
        """
        # Detect if this is an exhaustive query (auto-detect or forced)
        is_exhaustive = force_exhaustive or self._is_exhaustive_query(query)

        # Determine effective max iterations
        effective_max_iterations = self.max_iterations
        if is_exhaustive:
            effective_max_iterations = self.EXHAUSTIVE_MAX_ITERATIONS
            logger.info(
                f"Exhaustive mode enabled: force={force_exhaustive}, "
                f"auto_detect={self._is_exhaustive_query(query)}, "
                f"max_iterations={effective_max_iterations}"
            )

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

        logger.info(f"Starting research mode for query: {query}")

        # Track all iterations
        iterations: List[ResearchIteration] = []

        # Track all retrieved nodes (deduplicate by node_id)
        all_nodes: Dict[str, NodeWithScore] = {}

        # Track which file paths we've already retrieved from
        retrieved_paths: Set[str] = set()

        current_query = query

        for iteration_num in range(1, effective_max_iterations + 1):
            # Add delay between iterations to avoid rate limiting (skip first iteration)
            if iteration_num > 1 and self.ITERATION_DELAY > 0:
                import time
                logger.info(f"Waiting {self.ITERATION_DELAY}s before iteration {iteration_num} to avoid rate limiting...")
                time.sleep(self.ITERATION_DELAY)

            logger.info(f"Research iteration {iteration_num}/{effective_max_iterations}")

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
            gaps, confidence, full_analysis = self._analyze_gaps(
                query, list(all_nodes.values()), is_exhaustive=is_exhaustive
            )

            logger.info(
                f"Iteration {iteration_num}: Confidence={confidence:.2f}, "
                f"Total unique nodes={len(all_nodes)}, exhaustive={is_exhaustive}"
            )

            if gaps:
                logger.info(f"Identified gaps: {gaps}")

            # Store iteration result with full analysis
            iteration_result = ResearchIteration(
                iteration=iteration_num,
                query=current_query,
                nodes=nodes,
                gaps_identified=gaps,
                confidence_score=confidence,
                full_analysis=full_analysis
            )
            iterations.append(iteration_result)

            # Check if we should stop (skip early stopping for exhaustive queries)
            if confidence >= self.confidence_threshold and not is_exhaustive:
                logger.info(
                    f"Confidence threshold reached ({confidence:.2f} >= {self.confidence_threshold}), "
                    f"stopping research"
                )
                break
            elif confidence >= self.confidence_threshold and is_exhaustive:
                logger.info(
                    f"Confidence threshold reached but exhaustive mode - continuing "
                    f"(iteration {iteration_num}/{effective_max_iterations})"
                )

            # Check if we have more iterations available
            if iteration_num >= effective_max_iterations:
                logger.info(f"Maximum iterations reached ({effective_max_iterations})")
                break

            # Generate sub-queries for next iteration
            # For exhaustive queries, always generate sub-queries even if no explicit gaps
            if gaps or is_exhaustive:
                subqueries = self._generate_subqueries(
                    query, gaps or "Find more related content", retrieved_paths
                )

                if not subqueries:
                    logger.info("No sub-queries generated, stopping research")
                    break

                # Use first sub-query for next iteration
                # (in a more advanced implementation, could retrieve in parallel)
                current_query = subqueries[0]
                logger.info(f"Next iteration query: {current_query}")
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
        nodes: List[NodeWithScore],
        is_exhaustive: bool = False
    ) -> tuple[Optional[str], float, Optional[str]]:
        """Analyze gaps in retrieved content using LLM.

        Args:
            query: Original query
            nodes: Currently retrieved nodes
            is_exhaustive: If True, use stricter gap analysis for comprehensive queries

        Returns:
            Tuple of (gaps description, confidence score 0-1, full analysis text)
        """
        if not nodes:
            logger.debug("No nodes to analyze, returning low confidence")
            return "No relevant information found", 0.0, None

        # Build context from top nodes for gap analysis
        # 20 nodes gives good coverage without excessive token usage
        context_chunks = []
        for idx, node in enumerate(nodes[:20], 1):  # Analyze top 20 nodes
            title = node.metadata.get('title', 'Unknown')
            file_path = node.metadata.get('file_path', '')
            context_chunks.append(f"[Source {idx}: {title}]\nPath: {file_path}\n{node.node.text}")

        context = "\n\n".join(context_chunks)

        # Add special instructions for exhaustive queries
        exhaustive_note = ""
        if is_exhaustive:
            exhaustive_note = """
IMPORTANT: This is an EXHAUSTIVE query - the user wants ALL/EVERY matching item.
- Do NOT report high confidence unless you are CERTAIN all relevant items have been found
- For exhaustive queries like "list all X" or "every Y", prefer reporting GAPS over premature confidence
- Consider: Are there likely more items in different folders, with different names, or phrased differently?
- Only report 0.9+ confidence if you see clear evidence this is a complete list

"""

        prompt = f"""Analyze whether the retrieved information fully answers the user's query.
{exhaustive_note}
Query: {query}

Retrieved Information (top 20 chunks from knowledge base):
{context}

Evaluate:
1. Does this information comprehensively answer the query?
2. What specific topics, details, or perspectives are missing?
3. Rate your confidence (0.0-1.0) that we have sufficient information to answer well.

Respond in this exact format:
CONFIDENCE: <number between 0.0 and 1.0>
GAPS: <specific missing topics, or "None" if comprehensive>

Guidelines:
- 0.9+ = Query fully answered with rich detail
- 0.7-0.9 = Core question answered, some details missing
- 0.5-0.7 = Partial answer, significant gaps
- <0.5 = Insufficient or irrelevant information"""

        try:
            import time
            start_time = time.time()
            result = self.gap_analysis_llm.complete(prompt).text.strip()
            elapsed = time.time() - start_time
            logger.info(f"Gap analysis LLM call took {elapsed:.1f}s (model: {self.GAP_ANALYSIS_MODEL})")
            logger.info(f"Gap analysis LLM response:\n{result}")

            # Parse confidence score with robust regex matching
            confidence = 0.5  # Default
            gaps = None

            # Try to find confidence value with flexible patterns
            # Handles: "CONFIDENCE: 0.6", "**CONFIDENCE:** 0.6", "Confidence: 0.6", etc.
            confidence_patterns = [
                r'\*?\*?CONFIDENCE\*?\*?\s*[:：]\s*([\d.]+)',  # **CONFIDENCE:** 0.6
                r'confidence\s*[:：]\s*([\d.]+)',  # confidence: 0.6
                r'Confidence\s*[:：]\s*([\d.]+)',  # Confidence: 0.6
            ]

            for pattern in confidence_patterns:
                match = re.search(pattern, result, re.IGNORECASE)
                if match:
                    try:
                        confidence = float(match.group(1))
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                        logger.info(f"Parsed confidence: {confidence}")
                        break
                    except ValueError:
                        continue

            # Parse gaps with flexible matching
            gaps_patterns = [
                r'\*?\*?GAPS\*?\*?\s*[:：]\s*(.+?)(?:\n|$)',
                r'gaps\s*[:：]\s*(.+?)(?:\n|$)',
                r'Gaps\s*[:：]\s*(.+?)(?:\n|$)',
            ]

            for pattern in gaps_patterns:
                match = re.search(pattern, result, re.IGNORECASE | re.DOTALL)
                if match:
                    gaps_str = match.group(1).strip()
                    if gaps_str.lower() not in ['none', 'no gaps', 'n/a', 'none.', 'no gaps.']:
                        gaps = gaps_str  # Full gaps text
                        logger.info(f"Parsed gaps: {gaps}")
                    break

            if confidence == 0.5:
                logger.warning(f"Could not parse confidence from response, using default 0.5")

            return gaps, confidence, result  # Include full analysis text

        except Exception as e:
            logger.error(f"Error during gap analysis: {e}", exc_info=True)
            return "Analysis failed", 0.5, None

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
            import time
            start_time = time.time()
            result = self.llm.complete(prompt).text.strip()
            elapsed = time.time() - start_time
            logger.info(f"Sub-query generation LLM call took {elapsed:.1f}s (main LLM)")
            logger.info(f"Sub-query generation result:\n{result}")

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

            logger.info(f"Generated {len(subqueries)} sub-queries: {subqueries}")
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
