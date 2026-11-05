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

# These models share one embedding space, so a table built with one is valid for the others.
_VOYAGE_4_FAMILY = {"voyage-4", "voyage-4-large", "voyage-4-lite"}


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


def _sql_quote(value: str) -> str:
    """Escape a string literal for a LanceDB filter expression."""
    return "'" + value.replace("'", "''") + "'"


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

    def __init__(
        self,
        config: "SavedItemsConfig",
        top_k: int = 75,
        embedding_model: Optional[str] = None,
    ):
        self._config = config
        self._top_k = top_k
        self._embedding_model = embedding_model
        self._has_context = False
        self._table = None
        self._validated = False
        # Item ids a scoped caller may see (x3lixi's per-viewer queries). None = unscoped.
        self.allowed_item_ids: Optional[set[str]] = None
        super().__init__()

    def _build_where(self) -> Optional[str]:
        """LanceDB predicate: the viewer's scope or the quality filter, plus the operator's."""
        clauses = []
        if self.allowed_item_ids is not None:
            # No quality filter when scoped, same as x3lixi's semantic search: user items may
            # have no enrichment scores.
            ids = ", ".join(_sql_quote(item_id) for item_id in sorted(map(str, self.allowed_item_ids)))
            clauses.append(f"item_id IN ({ids})")
        elif self._config.min_quality_filter:
            clauses.append("(usefulness_score > 0 OR relevance_score > 0)")
        if self._config.where:
            clauses.append(f"({self._config.where})")
        return " AND ".join(clauses) or None

    def _model_check_passes(self) -> bool:
        """Compare the model that embedded the table with the configured one."""
        rows = self._table.search().select(["embedding_model"]).limit(1).to_pandas()
        table_model = _coalesce_str(*rows["embedding_model"])
        configured = self._embedding_model
        if (
            not table_model
            or not configured
            or table_model == configured
            or {table_model, configured} <= _VOYAGE_4_FAMILY
        ):
            return True
        strict = self._config.strict_model_check
        logger.warning(
            f"saved_items embedding model mismatch: table={table_model}, "
            f"configured={configured}.{' Disabling saved_items.' if strict else ''}"
        )
        return not strict

    def validate(self) -> bool:
        """Open table and validate embedding dimension. Idempotent (cached after first call)."""
        if self._validated:
            return self._table is not None
        self._validated = True
        try:
            import lancedb

            db = lancedb.connect(str(self._config.shared_lancedb_path))
            # table_names() returns only the first 10 names; list_tables() returns them all.
            if self._config.table_name not in db.list_tables().tables:
                logger.warning(
                    f"Table '{self._config.table_name}' not found in shared LanceDB at "
                    f"{self._config.shared_lancedb_path}"
                )
                return False

            self._table = db.open_table(self._config.table_name)
            logger.info(f"saved_items predicate: {self._build_where() or 'none'}")

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

            # Selecting a column the table lacks raises, and tables gain these only on rebuild
            columns = {f.name for f in schema}
            self._has_context = "context" in columns

            # Before the dim check, which returns early when the embedder is unreachable
            if "embedding_model" in columns and not self._model_check_passes():
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

        if self.allowed_item_ids is not None:
            logger.info(f"saved_items query scoped to {len(self.allowed_item_ids)} items")
            # An empty scope matches nothing, and IN () is not valid SQL
            if not self.allowed_item_ids:
                return []

        try:
            # Embed query using the shared embedding model
            query_vector = Settings.embed_model.get_query_embedding(
                query_bundle.query_str
            )

            query = self._table.search(query_vector).metric("cosine")
            where = self._build_where()
            if where:
                # Prefilter so filtered-out rows don't use up top_k slots
                query = query.where(where, prefilter=True)

            # Search — explicitly exclude heavy vector column, include _distance
            results = (
                query
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
                        *(["context"] if self._has_context else []),
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
            body = str(row.get("text", "") or "")
            context = _coalesce_str(row.get("context"))

            node = TextNode(
                id_=f"saved_items:{item_id}:{chunk_index}",
                text=f"{context}\n---\n{body}" if context else body,
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
                # chunk_body repeats the text, so in metadata it would send every chunk twice
                excluded_llm_metadata_keys=["chunk_body"],
                excluded_embed_metadata_keys=["chunk_body"],
            )
            if context:
                node.metadata["chunk_body"] = body
            nodes.append(NodeWithScore(node=node, score=score))

        return nodes
