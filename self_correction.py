"""Self-correcting RAG with relevance grading and query refinement."""

import logging
from enum import Enum
from typing import List, Tuple, Optional
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class RelevanceGrade(Enum):
    """Grading for document relevance to query."""
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class SelfCorrectingRetriever(BaseRetriever):
    """
    Self-correcting retriever implementing Self-RAG and CRAG patterns.

    After retrieval, evaluates:
    - Is the retrieved context relevant to the query?
    - Should we re-retrieve with a modified query?

    CRAG approach:
    - Grades retrieved documents as: CORRECT, AMBIGUOUS, INCORRECT
    - If AMBIGUOUS/INCORRECT: reformulate query and re-retrieve
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm,
        max_retries: int = 2,
        enable_correction: bool = True
    ):
        """
        Initialize self-correcting retriever.

        Args:
            base_retriever: The underlying retriever to wrap
            llm: Language model for grading and query refinement
            max_retries: Maximum number of retry attempts (default: 2)
            enable_correction: Whether to enable self-correction (default: True)
        """
        super().__init__()
        self.base_retriever = base_retriever
        self.llm = llm
        self.max_retries = max_retries
        self.enable_correction = enable_correction

        logger.info(
            f"SelfCorrectingRetriever initialized "
            f"(max_retries={max_retries}, enabled={enable_correction})"
        )

    def grade_relevance(
        self,
        query: str,
        nodes: List[NodeWithScore]
    ) -> RelevanceGrade:
        """
        Grade retrieved documents for relevance using LLM.

        Args:
            query: Original query string
            nodes: Retrieved nodes to grade

        Returns:
            RelevanceGrade indicating quality of retrieval
        """
        if not nodes:
            logger.debug("No nodes to grade, returning INCORRECT")
            return RelevanceGrade.INCORRECT

        # Take top 3 nodes for grading (to save tokens)
        context_chunks = []
        for idx, node in enumerate(nodes[:3], 1):
            # Truncate each chunk to 500 chars
            chunk_text = node.node.text[:500]
            title = node.metadata.get('title', 'Unknown')
            context_chunks.append(f"[Document {idx}: {title}]\n{chunk_text}")

        context = "\n\n".join(context_chunks)

        prompt = f"""Grade the relevance of the retrieved context to the user's query.

Query: {query}

Retrieved Context:
{context}

Analyze whether this context can answer the query:
- CORRECT: Context directly answers the query with relevant information
- AMBIGUOUS: Context is partially relevant but incomplete or tangential
- INCORRECT: Context does not address the query or is irrelevant

Respond with ONLY one word: CORRECT, AMBIGUOUS, or INCORRECT"""

        try:
            result = self.llm.complete(prompt).text.strip().upper()
            logger.debug(f"Relevance grading result: {result}")

            # Parse result - check for exact matches first to avoid substring conflicts
            if "INCORRECT" in result:
                return RelevanceGrade.INCORRECT
            elif "AMBIGUOUS" in result:
                return RelevanceGrade.AMBIGUOUS
            elif "CORRECT" in result:
                return RelevanceGrade.CORRECT
            else:
                # Default to AMBIGUOUS if unclear
                logger.warning(f"Unclear grading result: {result}, defaulting to AMBIGUOUS")
                return RelevanceGrade.AMBIGUOUS

        except Exception as e:
            logger.error(f"Error during relevance grading: {e}")
            # On error, assume AMBIGUOUS to trigger one retry
            return RelevanceGrade.AMBIGUOUS

    def refine_query(
        self,
        original_query: str,
        failed_context: str,
        attempt: int
    ) -> str:
        """
        Refine query based on what didn't work.

        Args:
            original_query: The original user query
            failed_context: Context that was not relevant
            attempt: Which retry attempt this is (1-based)

        Returns:
            Refined query string
        """
        prompt = f"""The search query returned irrelevant or incomplete results.

Original Query: {original_query}

Irrelevant Context Retrieved:
{failed_context[:300]}...

This is retry attempt #{attempt}. Rewrite the query to be more specific and likely to find relevant information.
Focus on different keywords, synonyms, or reformulations that might match better.

Respond with ONLY the new query text, nothing else:"""

        try:
            refined = self.llm.complete(prompt).text.strip()
            logger.info(f"Query refinement (attempt {attempt}): '{original_query}' -> '{refined}'")
            return refined

        except Exception as e:
            logger.error(f"Error during query refinement: {e}")
            # On error, return original query
            return original_query

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve with self-correction loop.

        Args:
            query_bundle: Query bundle containing query string

        Returns:
            Retrieved nodes, potentially after correction
        """
        if not self.enable_correction:
            # Bypass correction if disabled
            return self.base_retriever.retrieve(query_bundle)

        original_query = query_bundle.query_str
        current_query = original_query
        best_nodes = []
        best_grade = RelevanceGrade.INCORRECT

        for attempt in range(self.max_retries + 1):
            logger.info(
                f"Retrieval attempt {attempt + 1}/{self.max_retries + 1} "
                f"with query: '{current_query[:100]}...'"
            )

            # Create query bundle for current query
            current_bundle = QueryBundle(query_str=current_query)

            # Retrieve with base retriever
            nodes = self.base_retriever.retrieve(current_bundle)

            if not nodes:
                logger.warning(f"No nodes retrieved on attempt {attempt + 1}")
                if attempt < self.max_retries:
                    # Try refining query even with no results
                    current_query = self.refine_query(
                        original_query,
                        "No results found",
                        attempt + 1
                    )
                continue

            # Grade relevance (always grade against ORIGINAL query)
            grade = self.grade_relevance(original_query, nodes)
            logger.info(f"Relevance grade: {grade.value}")

            # Track best results
            if grade.value == RelevanceGrade.CORRECT.value:
                # Perfect match, return immediately
                logger.info("CORRECT grade achieved, returning results")
                return nodes

            # Keep best results so far
            if best_nodes == [] or self._grade_rank(grade) > self._grade_rank(best_grade):
                best_nodes = nodes
                best_grade = grade

            # If we still have retries and results aren't perfect, refine and retry
            if attempt < self.max_retries and grade != RelevanceGrade.CORRECT:
                # Build context from failed retrieval
                context_sample = "\n".join([
                    f"{n.metadata.get('title', 'Unknown')}: {n.node.text[:200]}"
                    for n in nodes[:2]
                ])

                current_query = self.refine_query(
                    original_query,
                    context_sample,
                    attempt + 1
                )

        # Return best results after all attempts
        logger.info(
            f"Self-correction complete after {self.max_retries + 1} attempts. "
            f"Best grade: {best_grade.value}"
        )
        return best_nodes

    def _grade_rank(self, grade: RelevanceGrade) -> int:
        """Convert grade to numeric rank for comparison."""
        ranking = {
            RelevanceGrade.CORRECT: 3,
            RelevanceGrade.AMBIGUOUS: 2,
            RelevanceGrade.INCORRECT: 1
        }
        return ranking.get(grade, 0)


class SelfRAGValidator:
    """
    Post-retrieval validator for Self-RAG pattern.

    Validates generated responses against retrieved context:
    - Is the response supported by the context?
    - Does the response hallucinate information?
    """

    def __init__(self, llm):
        """
        Initialize Self-RAG validator.

        Args:
            llm: Language model for validation
        """
        self.llm = llm
        logger.info("SelfRAGValidator initialized")

    def validate_response(
        self,
        query: str,
        response: str,
        context: str
    ) -> Tuple[bool, str]:
        """
        Validate that response is supported by context.

        Args:
            query: Original query
            response: Generated response to validate
            context: Retrieved context

        Returns:
            Tuple of (is_valid, explanation)
        """
        prompt = f"""Validate whether the response is supported by the retrieved context.

Query: {query}

Retrieved Context:
{context[:1000]}

Generated Response:
{response}

Answer these questions:
1. Is the response factually supported by the context?
2. Does the response hallucinate or add information not in the context?
3. Is the response faithful to the retrieved information?

Respond with:
VALID - if response is fully supported by context
INVALID - if response contains unsupported claims or hallucinations

Then explain briefly why.

Format:
VALID|INVALID
Explanation: <your explanation>"""

        try:
            result = self.llm.complete(prompt).text.strip()
            first_line = result.split('\n')[0].upper()

            # Check for INVALID first to avoid substring conflicts with VALID
            if "INVALID" in first_line:
                is_valid = False
            elif "VALID" in first_line:
                is_valid = True
            else:
                # Default to True if unclear
                is_valid = True

            # Extract explanation
            lines = result.split('\n')
            explanation = '\n'.join(lines[1:]).strip()
            if not explanation:
                explanation = "No explanation provided"

            logger.info(f"Response validation: {'VALID' if is_valid else 'INVALID'}")
            return is_valid, explanation

        except Exception as e:
            logger.error(f"Error during response validation: {e}")
            return True, f"Validation error: {e}"
