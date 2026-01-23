"""Content Research Engine - Weighted federated search across all knowledge sources.

Implements multi-source vector search with configurable weights, parallel execution,
and cosine-similarity-based deduplication across vault, conversations, and saved items.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger(__name__)

# Source priority for deduplication (higher index = higher priority)
SOURCE_PRIORITY = {"saved_items": 0, "conversations": 1, "vault": 2}

# LanceDB paths per source
ULTRARAG_LANCEDB_PATH = Path(__file__).parent / "data" / "lancedb"
SHARED_LANCEDB_PATH = Path.home() / "Projects" / "shared-data" / "lancedb"

# Table names per source
TABLE_NAMES = {
    "vault": "obsidian_embeddings",
    "conversations": "conversations",
    "saved_items": "saved_items",
}


@dataclass
class ResearchResult:
    """A single result from federated search across knowledge sources.

    Attributes:
        rank: 1-based position in the final sorted results.
        source: Origin source identifier ("vault", "conversations", "saved_items").
        title: Human-readable title for display.
        chunk: The text content of this result chunk.
        raw_score: Cosine similarity score (1 - distance) before weighting.
        weighted_score: Score after applying source weight multiplier.
        metadata: Source-specific metadata dict.
        embedding: Optional vector for deduplication comparison.
    """

    rank: int
    source: str
    title: str
    chunk: str
    raw_score: float
    weighted_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


def _get_lancedb_path(source: str) -> Path:
    """Return the correct LanceDB path for a given source."""
    if source == "saved_items":
        return SHARED_LANCEDB_PATH
    return ULTRARAG_LANCEDB_PATH


def _extract_metadata_field(row: dict, source: str, field_name: str, default=None):
    """Safely extract a field from a row, handling nested metadata structs.

    For vault and conversations tables, the metadata is stored as a nested struct.
    When converted to pandas and then to dict, it may appear as:
      - row['metadata']['field_name'] (dict form)
      - row['metadata.field_name'] (flattened form)
      - row['field_name'] (if pandas auto-flattened)

    For saved_items, all columns are flat.
    """
    if source in ("vault", "conversations"):
        # Try nested dict access first
        meta = row.get("metadata")
        if isinstance(meta, dict):
            return meta.get(field_name, default)
        # Try flattened column name
        flattened_key = f"metadata.{field_name}"
        if flattened_key in row:
            return row[flattened_key]
        # Fallback to direct access
        return row.get(field_name, default)
    else:
        return row.get(field_name, default)


def _build_research_result(
    row: dict, source: str, weight: float
) -> Optional[ResearchResult]:
    """Build a ResearchResult from a LanceDB row dict.

    Args:
        row: Dict representation of a pandas row.
        source: Source identifier.
        weight: Weight multiplier for this source.

    Returns:
        ResearchResult or None if essential data is missing.
    """
    # LanceDB cosine metric returns distance; similarity = 1 - distance
    distance = row.get("_distance", 1.0)
    raw_score = 1.0 - distance

    # Extract text content
    chunk = row.get("text", "")
    if not chunk:
        return None

    # Extract embedding vector
    embedding = row.get("vector")
    if embedding is not None:
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            embedding = list(embedding)

    # Source-specific metadata and title extraction
    if source == "vault":
        title = (
            _extract_metadata_field(row, source, "title")
            or _extract_metadata_field(row, source, "file_name")
            or "Untitled Note"
        )
        metadata = {
            "file_path": _extract_metadata_field(row, source, "file_path"),
            "header_path": _extract_metadata_field(row, source, "header_path"),
            "tags": _extract_metadata_field(row, source, "tags"),
        }
    elif source == "conversations":
        title = _extract_metadata_field(row, source, "title") or "Untitled Conversation"
        metadata = {
            "platform": _extract_metadata_field(row, source, "source"),
            "date": _extract_metadata_field(row, source, "date"),
        }
    elif source == "saved_items":
        title = row.get("title") or "Untitled Item"
        metadata = {
            "url": row.get("url"),
            "source_platform": row.get("source"),
            "domain": row.get("domain"),
            "tags": row.get("tags"),
            "usefulness_score": row.get("usefulness_score"),
            "ai_summary": row.get("ai_summary"),
        }
    else:
        logger.warning(f"Unknown source type: {source}")
        return None

    return ResearchResult(
        rank=0,  # Assigned after sorting
        source=source,
        title=title,
        chunk=chunk,
        raw_score=raw_score,
        weighted_score=raw_score * weight,
        metadata=metadata,
        embedding=embedding,
    )


def _search_source(
    source: str,
    query_vector: List[float],
    weight: float,
    top_k: int,
    threshold: float,
) -> List[ResearchResult]:
    """Search a single LanceDB source table.

    Args:
        source: Source identifier ("vault", "conversations", "saved_items").
        query_vector: Pre-computed query embedding vector (1024 dims).
        weight: Score weight multiplier for this source.
        top_k: Maximum results to retrieve per source.
        threshold: Minimum raw_score to include in results.

    Returns:
        List of ResearchResult objects from this source.
    """
    import lancedb

    db_path = _get_lancedb_path(source)
    table_name = TABLE_NAMES[source]

    if not db_path.exists():
        logger.warning(f"LanceDB path does not exist for {source}: {db_path}")
        return []

    try:
        db = lancedb.connect(str(db_path))

        # Check if table exists
        try:
            available_tables = db.table_names()
        except Exception:
            available_tables = []
        if table_name not in available_tables:
            logger.warning(f"Table '{table_name}' not found in {db_path}")
            return []

        table = db.open_table(table_name)
        logger.debug(f"Searching {source} ({table.count_rows()} rows) with top_k={top_k}")

        # Perform vector search
        search_results = (
            table.search(query_vector)
            .metric("cosine")
            .limit(top_k)
            .to_pandas()
        )

        results: List[ResearchResult] = []
        for _, row in search_results.iterrows():
            row_dict = row.to_dict()
            result = _build_research_result(row_dict, source, weight)
            if result is not None and result.raw_score >= threshold:
                results.append(result)

        logger.info(f"Source '{source}': {len(results)} results above threshold {threshold}")
        return results

    except Exception as e:
        logger.error(f"Error searching source '{source}': {e}", exc_info=True)
        return []


def weighted_federated_search(
    query: str,
    weights: Dict[str, float],
    top_k: int = 30,
    threshold: float = 0.4,
    enabled_sources: Optional[List[str]] = None,
    embed_model: Optional[Any] = None,
) -> List[ResearchResult]:
    """Execute weighted federated vector search across multiple knowledge sources.

    Queries all enabled LanceDB tables in parallel using ThreadPoolExecutor,
    applies source-specific weight multipliers to raw cosine similarity scores,
    filters by threshold, and returns a merged sorted list.

    Args:
        query: The search query text to embed and search for.
        weights: Source weight multipliers, e.g. {"vault": 1.0, "conversations": 0.7, "saved_items": 0.5}.
        top_k: Maximum results to retrieve per source (default 30).
        threshold: Minimum raw cosine similarity to include (default 0.4).
        enabled_sources: List of sources to query. None means all sources with weight > 0.
        embed_model: Pre-initialized embedding model. If None, one is created from config.

    Returns:
        List of ResearchResult sorted by weighted_score descending, with 1-based ranks assigned.

    Raises:
        RuntimeError: If embedding model initialization fails.
    """
    # Determine which sources to query
    if enabled_sources is None:
        enabled_sources = [s for s, w in weights.items() if w > 0]

    if not enabled_sources:
        logger.warning("No sources enabled for federated search")
        return []

    logger.info(
        f"Federated search: query='{query[:80]}...', sources={enabled_sources}, "
        f"top_k={top_k}, threshold={threshold}"
    )

    # Initialize embedding model if not provided
    if embed_model is None:
        try:
            from config import load_config
            from embeddings import get_embedding_model

            config = load_config()
            embed_model = get_embedding_model(config.embedding, config.voyage_api_key)
            logger.debug("Initialized embedding model from config")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {e}") from e

    # Embed the query
    try:
        query_vector = embed_model.get_query_embedding(query)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}", exc_info=True)
        raise RuntimeError(f"Query embedding failed: {e}") from e

    # Search all sources in parallel
    all_results: List[ResearchResult] = []

    with ThreadPoolExecutor(max_workers=len(enabled_sources)) as executor:
        futures = {
            executor.submit(
                _search_source,
                source,
                query_vector,
                weights.get(source, 0.5),
                top_k,
                threshold,
            ): source
            for source in enabled_sources
            if source in TABLE_NAMES
        }

        for future in as_completed(futures):
            source = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Source '{source}' search failed: {e}", exc_info=True)

    # Sort by weighted_score descending
    all_results.sort(key=lambda r: r.weighted_score, reverse=True)

    # Assign 1-based ranks
    for i, result in enumerate(all_results, start=1):
        result.rank = i

    logger.info(f"Federated search complete: {len(all_results)} total results")
    return all_results


def deduplicate_results(
    results: List[ResearchResult],
    similarity_threshold: float = 0.85,
) -> Tuple[List[ResearchResult], List[dict]]:
    """Remove near-duplicate results across sources using cosine similarity.

    Compares embedding vectors between all pairs of results. When two results
    exceed the similarity threshold, the one from the lower-priority source
    is removed. If both are from the same source, the lower weighted_score
    is removed.

    Priority order: vault > conversations > saved_items.

    Args:
        results: List of ResearchResult objects with embeddings populated.
        similarity_threshold: Cosine similarity above which results are
            considered duplicates (default 0.85).

    Returns:
        Tuple of:
            - Deduplicated results list with re-assigned 1-based ranks.
            - Dedup log: list of dicts recording each removal decision.
    """
    if not results:
        return [], []

    # Filter results that have embeddings for comparison
    results_with_embeddings = [r for r in results if r.embedding is not None]
    results_without_embeddings = [r for r in results if r.embedding is None]

    if len(results_with_embeddings) <= 1:
        # Nothing to deduplicate
        all_kept = results_with_embeddings + results_without_embeddings
        for i, r in enumerate(all_kept, start=1):
            r.rank = i
        return all_kept, []

    # Build embedding matrix for vectorized comparison
    embeddings_matrix = np.array(
        [r.embedding for r in results_with_embeddings], dtype=np.float32
    )

    # Normalize vectors for cosine similarity via dot product
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    # Prevent division by zero
    norms = np.where(norms == 0, 1.0, norms)
    normalized = embeddings_matrix / norms

    # Compute pairwise cosine similarity matrix (upper triangle only)
    similarity_matrix = normalized @ normalized.T

    # Track which indices to remove
    removed_indices: set = set()
    dedup_log: List[dict] = []

    n = len(results_with_embeddings)
    for i in range(n):
        if i in removed_indices:
            continue
        for j in range(i + 1, n):
            if j in removed_indices:
                continue

            sim = float(similarity_matrix[i, j])
            if sim >= similarity_threshold:
                r_i = results_with_embeddings[i]
                r_j = results_with_embeddings[j]

                # Decide which to keep based on source priority, then score
                priority_i = SOURCE_PRIORITY.get(r_i.source, -1)
                priority_j = SOURCE_PRIORITY.get(r_j.source, -1)

                if priority_i > priority_j:
                    remove_idx, keep_idx = j, i
                elif priority_j > priority_i:
                    remove_idx, keep_idx = i, j
                else:
                    # Same source priority: keep higher weighted_score
                    if r_i.weighted_score >= r_j.weighted_score:
                        remove_idx, keep_idx = j, i
                    else:
                        remove_idx, keep_idx = i, j

                removed_indices.add(remove_idx)
                removed_result = results_with_embeddings[remove_idx]
                kept_result = results_with_embeddings[keep_idx]

                dedup_log.append(
                    {
                        "removed_title": removed_result.title,
                        "removed_source": removed_result.source,
                        "kept_title": kept_result.title,
                        "kept_source": kept_result.source,
                        "similarity": round(sim, 4),
                    }
                )

                logger.debug(
                    f"Dedup: removed '{removed_result.title}' ({removed_result.source}) "
                    f"kept '{kept_result.title}' ({kept_result.source}) "
                    f"sim={sim:.4f}"
                )

    # Build deduplicated list
    kept_results = [
        r for i, r in enumerate(results_with_embeddings) if i not in removed_indices
    ]
    # Add back results without embeddings (cannot deduplicate these)
    kept_results.extend(results_without_embeddings)

    # Re-sort by weighted_score and re-rank
    kept_results.sort(key=lambda r: r.weighted_score, reverse=True)
    for i, r in enumerate(kept_results, start=1):
        r.rank = i

    logger.info(
        f"Deduplication: {len(results)} -> {len(kept_results)} results "
        f"({len(removed_indices)} removed, threshold={similarity_threshold})"
    )

    return kept_results, dedup_log
