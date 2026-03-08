"""Custom LanceDB retriever for saved_items (TheSource) with flat schema."""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional

from llama_index.core import Settings
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

if TYPE_CHECKING:
    from config import SavedItemsConfig

logger = logging.getLogger(__name__)


def _is_null(value) -> bool:
    """Return True for None/NaN/pandas.NA-like values."""
    if value is None:
        return True
    try:
        import pandas as pd
        return bool(pd.isna(value))
    except Exception:
        try:
            return math.isnan(float(value))
        except Exception:
            return False


def _coalesce_str(*values: object, fallback: str = "") -> str:
    """Return first non-null value as string."""
    for value in values:
        if not _is_null(value):
            text = str(value).strip()
            if text:
                return text
    return fallback


def _safe_int(value: object, default: int = 0) -> int:
    """Parse int from mixed LanceDB/pandas values."""
    if _is_null(value):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


class SavedItemsRetriever(BaseRetriever):
    """Direct LanceDB vector retriever for saved_items (flat schema).

    The saved_items table is NOT LlamaIndex-compatible (no metadata struct,
    no _node_content). This retriever directly queries LanceDB and wraps
    results as TextNode / NodeWithScore for use in the federated engine.
    """

    def __init__(self, config: "SavedItemsConfig", top_k: int = 75):
        self._config = config
        self._top_k = top_k
        self._table = None
        self._validated = False
        super().__init__()

    def validate(self) -> bool:
        """Open table and validate embedding dimension. Idempotent (cached after first call)."""
        if self._validated:
            return self._table is not None
        self._validated = True
        try:
            import lancedb

            db = lancedb.connect(str(self._config.shared_lancedb_path))
            available = db.table_names()
            if self._config.table_name not in available:
                logger.warning(
                    f"Table '{self._config.table_name}' not found in shared LanceDB at "
                    f"{self._config.shared_lancedb_path}"
                )
                return False

            self._table = db.open_table(self._config.table_name)

            # Validate vector field exists and dimension matches current embed model
            schema = self._table.schema
            vector_field = next(
                (f for f in schema if f.name == "vector"), None
            )
            if vector_field is None or not hasattr(vector_field.type, "list_size"):
                logger.warning(
                    "saved_items table missing 'vector' field or unexpected schema"
                )
                self._table = None
                return False

            schema_dim = vector_field.type.list_size
            try:
                actual_dim = len(Settings.embed_model.get_query_embedding("dim check"))
            except Exception as e:
                logger.warning(f"Could not check embedding dim: {e}")
                # Proceed optimistically — dim mismatch will surface at query time
                return True

            if schema_dim != actual_dim:
                logger.warning(
                    f"saved_items dim mismatch: table={schema_dim}, "
                    f"embed_model={actual_dim}. Disabling saved_items."
                )
                self._table = None
                return False

            return True

        except Exception as e:
            logger.warning(f"Failed to validate saved_items: {e}")
            self._table = None
            return False

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if not self.validate():
            return []

        try:
            # Embed query using the shared embedding model
            query_vector = Settings.embed_model.get_query_embedding(
                query_bundle.query_str
            )

            # Search — explicitly exclude heavy vector column, include _distance
            results = (
                self._table.search(query_vector)
                .metric("cosine")
                .limit(self._top_k)
                .select(
                    [
                        "text",
                        "title",
                        "url",
                        "domain",
                        "source",
                        "tags",
                        "item_id",
                        "chunk_index",
                        "usefulness_score",
                        "_distance",
                    ]
                )
                .to_pandas()
            )

        except Exception as e:
            logger.error(f"saved_items retrieval failed: {e}")
            return []

        nodes: List[NodeWithScore] = []
        for _, row in results.iterrows():
            distance = row.get("_distance", None)
            score = 0.0
            if not _is_null(distance):
                try:
                    dist = float(distance)
                    if not math.isnan(dist):
                        score = float(1.0 - dist)
                except Exception:
                    score = 0.0

            item_id = _coalesce_str(row.get("item_id"), row.get("url"), fallback="unknown")
            chunk_index = _safe_int(row.get("chunk_index", 0), default=0)
            domain = _coalesce_str(row.get("domain"), fallback="") or None
            url = _coalesce_str(row.get("url"), fallback="") or None
            title = _coalesce_str(row.get("title"), fallback="Untitled Item")

            node = TextNode(
                id_=f"saved_items:{item_id}:{chunk_index}",
                text=str(row.get("text", "") or ""),
                metadata={
                    "title": title,
                    "url": url,
                    "domain": domain,
                    "source_platform": row.get("source"),
                    "tags": row.get("tags"),
                    "source_type": "saved_items",
                    "document_id": item_id,
                    # file_path = item_id so _document_key() dedupes per-item (not per-domain)
                    "file_path": item_id,
                    # display_label used in UI instead of raw item_id
                    "display_label": domain or "saved item",
                },
            )
            nodes.append(NodeWithScore(node=node, score=score))

        return nodes
