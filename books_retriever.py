"""Shared books retriever helper.

Neutral module used by both federated_query.py and retrieval.py to avoid
circular imports. Builds a VectorIndexRetriever with optional LanceDB
WHERE filtering for book metadata.
"""

import logging
from typing import TYPE_CHECKING

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

if TYPE_CHECKING:
    from models import BookFilter

logger = logging.getLogger(__name__)


def get_books_retriever(
    index: VectorStoreIndex,
    top_k: int,
    book_filter: "BookFilter | None" = None,
) -> VectorIndexRetriever:
    """Build a books retriever with optional LanceDB WHERE filtering.

    Uses VectorIndexRetriever with vector_store_kwargs={"where": ...} for
    native LanceDB filtering. This is necessary because:
    - Categories are list[str] -> needs array_has_any() (DataFusion)
    - MetadataFilters doesn't support array operations
    - where and MetadataFilters are mutually exclusive in LanceDB adapter

    Args:
        index: The books VectorStoreIndex
        top_k: Number of results to retrieve
        book_filter: Optional BookFilter with category/author/uid constraints

    Returns:
        VectorIndexRetriever configured with optional WHERE clause
    """
    kwargs = {}
    if book_filter:
        where = book_filter.to_lance_where()
        if where:
            kwargs["vector_store_kwargs"] = {"where": where}
            logger.info("Books filter active: %s", where)

    return VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        **kwargs,
    )
