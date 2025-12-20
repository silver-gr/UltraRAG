"""Query transformation techniques for improved retrieval.

This module implements advanced query transformation techniques that significantly
improve retrieval quality by bridging the query-document vocabulary gap:

1. HyDE (Hypothetical Document Embeddings): Generates a hypothetical answer to the
   query and embeds that instead of the raw query. Since hypothetical answers are
   more similar to actual documents, this improves retrieval accuracy.

2. Multi-Query Expansion: Generates multiple variations of the query from different
   perspectives and combines results using reciprocal rank fusion.

3. Combined Approach: Generates multiple query variations AND creates hypothetical
   documents for each, combining the benefits of both techniques.
"""

import logging
from typing import List, Union, Optional
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


class QueryTransformer:
    """Query transformer implementing HyDE and Multi-Query expansion techniques."""

    def __init__(self, llm: GoogleGenAI, embed_model: Optional[BaseEmbedding] = None):
        """
        Initialize query transformer.

        Args:
            llm: Language model for generating transformations
            embed_model: Optional embedding model (for future use)
        """
        self.llm = llm
        self.embed_model = embed_model
        logger.info("QueryTransformer initialized")

    def hyde_transform(self, query: str) -> str:
        """
        Generate hypothetical document for query using HyDE technique.

        HyDE works by generating a hypothetical answer that resembles the structure
        and vocabulary of actual documents. This bridges the query-document gap since
        documents are more similar to other documents than to queries.

        Args:
            query: Original user query

        Returns:
            Hypothetical document text
        """
        logger.debug(f"Applying HyDE transformation to query: {query[:100]}...")

        # Craft a prompt that generates a document-like response
        # Using thinking mode for better quality hypothetical documents
        prompt = f"""You are helping to search a personal knowledge base (Obsidian vault).

For the following question, write a detailed, informative passage that would appear in someone's notes and would answer this question. Write as if you're composing a note entry, not as if you're answering the question directly.

Question: {query}

Write a comprehensive note passage (2-3 paragraphs) that would contain the answer to this question. Use the style and vocabulary typical of knowledge base notes:"""

        try:
            # Generate hypothetical document
            response = self.llm.complete(prompt)
            hypothetical_doc = response.text.strip()

            logger.debug(f"Generated hypothetical document ({len(hypothetical_doc)} chars)")
            logger.debug(f"Hypothetical doc preview: {hypothetical_doc[:200]}...")

            return hypothetical_doc

        except Exception as e:
            logger.warning(f"HyDE transformation failed: {e}. Falling back to original query.")
            return query

    def multi_query_expand(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple query variations from different perspectives.

        Multi-query expansion generates semantically similar queries that approach
        the information need from different angles. This increases recall by
        capturing documents that might match one variation better than others.

        Args:
            query: Original user query
            num_queries: Number of variations to generate (default: 3)

        Returns:
            List of query variations including the original
        """
        logger.debug(f"Generating {num_queries} query variations for: {query[:100]}...")

        # Craft a prompt that generates diverse query reformulations
        prompt = f"""You are helping to improve search in a personal knowledge base.

Generate {num_queries} different versions of the following search query. Each version should:
- Approach the topic from a slightly different angle
- Use alternative vocabulary and phrasing
- Maintain the same core information need
- Be concise (1-2 sentences each)

Original query: {query}

Generate {num_queries} alternative search queries, one per line, without numbering:"""

        try:
            # Generate query variations
            response = self.llm.complete(prompt)
            variations_text = response.text.strip()

            # Parse the variations (split by newlines and clean)
            variations = [
                line.strip()
                for line in variations_text.split('\n')
                if line.strip() and not line.strip().startswith(('#', '-', '*', str))
            ]

            # Remove any numbering or bullets that might be present
            cleaned_variations = []
            for var in variations:
                # Remove common prefixes like "1.", "1)", "-", "*", etc.
                cleaned = var
                if var and len(var) > 2:
                    # Remove leading numbers and punctuation
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '1)', '2)', '3)', '4)', '5)', '- ', '* ']:
                        if cleaned.startswith(prefix):
                            cleaned = cleaned[len(prefix):].strip()
                cleaned_variations.append(cleaned)

            # Take only the requested number of variations
            variations = cleaned_variations[:num_queries]

            # Always include the original query first
            all_queries = [query] + variations

            logger.info(f"Generated {len(variations)} query variations (+ original)")
            for i, q in enumerate(all_queries):
                logger.debug(f"  Query {i+1}: {q[:100]}...")

            return all_queries

        except Exception as e:
            logger.warning(f"Multi-query expansion failed: {e}. Using original query only.")
            return [query]

    def transform_query(
        self,
        query: str,
        method: str = "hyde",
        num_queries: int = 3
    ) -> Union[str, List[str]]:
        """
        Apply selected query transformation method.

        Args:
            query: Original user query
            method: Transformation method - "hyde", "multi_query", or "both"
            num_queries: Number of query variations for multi_query (default: 3)

        Returns:
            Transformed query (string for hyde, list for multi_query/both)
        """
        logger.info(f"Transforming query with method: {method}")

        if method == "hyde":
            return self.hyde_transform(query)

        elif method == "multi_query":
            return self.multi_query_expand(query, num_queries)

        elif method == "both":
            # Generate query variations first
            variations = self.multi_query_expand(query, num_queries)

            # Generate hypothetical document for each variation
            logger.debug("Applying HyDE to each query variation...")
            hypothetical_docs = []
            for i, variation in enumerate(variations):
                logger.debug(f"HyDE transform {i+1}/{len(variations)}")
                hyde_doc = self.hyde_transform(variation)
                hypothetical_docs.append(hyde_doc)

            logger.info(f"Generated {len(hypothetical_docs)} hypothetical documents from {len(variations)} queries")
            return hypothetical_docs

        elif method == "none" or method == "disabled":
            # No transformation
            logger.debug("Query transformation disabled, returning original query")
            return query

        else:
            logger.warning(f"Unknown transformation method: {method}. Using original query.")
            return query

    def get_embedding_queries(
        self,
        query: str,
        method: str = "hyde",
        num_queries: int = 3
    ) -> List[str]:
        """
        Get list of queries to embed for retrieval.

        This is a convenience method that ensures the output is always a list,
        making it easier to integrate with retrievers.

        Args:
            query: Original user query
            method: Transformation method
            num_queries: Number of variations for multi_query

        Returns:
            List of queries/documents to embed
        """
        result = self.transform_query(query, method, num_queries)

        # Ensure we always return a list
        if isinstance(result, str):
            return [result]
        elif isinstance(result, list):
            return result
        else:
            logger.warning(f"Unexpected transform result type: {type(result)}")
            return [query]
