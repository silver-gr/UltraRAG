"""Retrieval evaluation harness for UltraRAG.

Runs the *real* query engines (HybridQueryEngine / FederatedQueryEngine) in
retrieval-only mode and scores the result against a document-level golden set.

Why document-level: chunk ids change on every re-index, file paths don't. A
golden set keyed on ``file_path`` survives chunking, embedding and index-time
changes, which is exactly what you want to ablate.

Public API
----------
- doc_id(node)                                  -> stable document identifier
- load_golden_set(path) / save_golden_set(...)  -> GoldenSet
- score_query(pre_docs, post_docs, relevant, k) -> dict of metrics
- aggregate(rows)                                -> mean metrics (+ per category)
- RetrievalProbe(rag)                            -> .run(query, scope) -> ProbeResult
- apply_env_overrides(overrides)                 -> context manager
- INDEX_LEVEL_KEYS                               -> env keys that need a reload

Heavy imports (llama_index, main.UltraRAG) are deferred so the pure parts can
be unit-tested without the full stack.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger("ultrarag.eval_harness")

GOLDEN_SET_VERSION = 1
DEFAULT_GOLDEN_PATH = Path("tests/golden/golden_set.json")

# Env keys whose change invalidates the loaded index / nodes (needs full reload).
# Query-time keys only need the engine rebuilt.
INDEX_LEVEL_KEYS = {
    "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "CHUNKING_STRATEGY",
    "LATE_CHUNKING_ALPHA", "USE_CONTEXTUAL_RETRIEVAL", "VAULT_TABLE",
    "LANCEDB_PATH", "VECTOR_DB", "CONVERSATIONS_TABLE", "BOOKS_TABLE_NAME",
    "BOOKS_ENABLED", "CONVERSATIONS_ENABLED", "SAVED_ITEMS_ENABLED",
    "SAVED_ITEMS_MIN_QUALITY_FILTER", "SAVED_ITEMS_WHERE",
    "ENABLE_HYBRID_SEARCH",
}


# ---------------------------------------------------------------------------
# Document identity
# ---------------------------------------------------------------------------

# ULTRARAG_EVAL_DOC_KEY=basename collapses copies of the same note living in several
# folders (a vault that has been reorganised) into one document for scoring.
DOC_KEY_MODE = os.environ.get("ULTRARAG_EVAL_DOC_KEY", "path").strip().lower()


def doc_id(node_or_meta: Any) -> str:
    """Stable document id for a LlamaIndex node (or a metadata dict / source dict).

    Order: file_path > url (saved items) > file_name > book_title > node_id.
    With ULTRARAG_EVAL_DOC_KEY=basename, a file_path is reduced to its file name.
    """
    meta: Dict[str, Any]
    if isinstance(node_or_meta, dict):
        meta = node_or_meta
    else:
        inner = getattr(node_or_meta, "node", node_or_meta)
        meta = getattr(inner, "metadata", None) or {}
    for key in ("file_path", "url", "file_name", "book_title"):
        val = meta.get(key)
        if val:
            val = str(val)
            if key == "file_path" and DOC_KEY_MODE == "basename":
                return val.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            return val
    inner = getattr(node_or_meta, "node", node_or_meta)
    return str(getattr(inner, "node_id", "") or getattr(inner, "id_", "") or "unknown")


def node_title(node: Any) -> str:
    inner = getattr(node, "node", node)
    meta = getattr(inner, "metadata", None) or {}
    return str(meta.get("title") or meta.get("book_title") or meta.get("file_name") or doc_id(node))


def node_text(node: Any) -> str:
    inner = getattr(node, "node", node)
    meta = getattr(inner, "metadata", None) or {}
    return str(meta.get("_original_text") or getattr(inner, "text", "") or "")


def dedupe_docs(nodes: Sequence[Any]) -> List[str]:
    """Ordered unique document ids from a ranked node list."""
    seen: set = set()
    out: List[str] = []
    for n in nodes:
        d = doc_id(n)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------

@dataclass
class GoldenQuery:
    id: str
    query: str
    relevant: Dict[str, int]                 # doc_id -> grade (1 relevant, 2 highly)
    language: str = "en"                     # en | el | mixed
    category: str = "factual"                # factual | conceptual | exhaustive | comparison | source
    scope: str = "vault"                     # vault | federated
    status: str = "confirmed"                # candidate | proposed | confirmed | rejected
    notes: str = ""
    source: str = "manual"                   # manual | history | synthetic


@dataclass
class GoldenSet:
    queries: List[GoldenQuery]
    version: int = GOLDEN_SET_VERSION
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""

    def confirmed(self) -> List[GoldenQuery]:
        return [q for q in self.queries if q.status == "confirmed" and q.relevant]


def load_golden_set(path: Path) -> GoldenSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = []
    for item in data.get("queries", []):
        rel = item.get("relevant") or {}
        if isinstance(rel, list):  # tolerate a plain list of doc ids
            rel = {d: 1 for d in rel}
        queries.append(GoldenQuery(
            id=item["id"], query=item["query"], relevant={str(k): int(v) for k, v in rel.items()},
            language=item.get("language", "en"), category=item.get("category", "factual"),
            scope=item.get("scope", "vault"), status=item.get("status", "confirmed"),
            notes=item.get("notes", ""), source=item.get("source", "manual"),
        ))
    return GoldenSet(queries=queries, version=data.get("version", GOLDEN_SET_VERSION),
                     created=data.get("created", ""), description=data.get("description", ""))


def save_golden_set(gs: GoldenSet, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": gs.version, "created": gs.created, "description": gs.description,
               "queries": [asdict(q) for q in gs.queries]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Metrics (document level)
# ---------------------------------------------------------------------------

def _dcg(grades: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    gains = [float(relevant.get(d, 0)) for d in ranked[:k]]
    ideal = sorted((float(g) for g in relevant.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / k


def mrr(ranked: Sequence[str], relevant: Dict[str, int]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def score_query(
    pre_docs: Sequence[str],
    post_docs: Sequence[str],
    relevant: Dict[str, int],
    k_pre: int = 20,
    k_post: int = 10,
) -> Dict[str, float]:
    """Metrics for one query.

    pre_docs:  ranked unique doc ids from the candidate stage (before reranker/postprocessors)
    post_docs: ranked unique doc ids after reranker + postprocessors
    """
    return {
        f"recall@{k_pre}_pre": recall_at_k(pre_docs, relevant, k_pre),
        "recall@50_pre": recall_at_k(pre_docs, relevant, 50),
        f"recall@{k_post}_post": recall_at_k(post_docs, relevant, k_post),
        f"ndcg@{k_post}_post": ndcg_at_k(post_docs, relevant, k_post),
        f"p@{k_post}_post": precision_at_k(post_docs, relevant, k_post),
        "mrr_post": mrr(post_docs, relevant),
        "hit@1_post": 1.0 if post_docs and post_docs[0] in relevant else 0.0,
        "n_pre": float(len(pre_docs)),
        "n_post": float(len(post_docs)),
    }


def aggregate(rows: Iterable[Dict[str, Any]], group_key: Optional[str] = None) -> Dict[str, Any]:
    """Mean of every numeric metric across rows; optional per-group breakdown."""
    rows = list(rows)
    if not rows:
        return {"n": 0}

    def _mean(subset: List[Dict[str, Any]]) -> Dict[str, float]:
        keys = [k for k, v in subset[0]["metrics"].items() if isinstance(v, (int, float))]
        out = {}
        for k in keys:
            vals = [r["metrics"][k] for r in subset if k in r["metrics"]]
            out[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
        out["n"] = len(subset)
        return out

    result: Dict[str, Any] = {"overall": _mean(rows)}
    if group_key:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(str(r.get(group_key, "?")), []).append(r)
        result["by_" + group_key] = {g: _mean(rs) for g, rs in sorted(groups.items())}
    return result


# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def apply_env_overrides(overrides: Dict[str, str]):
    """Temporarily set env vars (config.load_config() reads os.environ at call time)."""
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def needs_index_reload(overrides: Dict[str, str]) -> bool:
    return any(k in INDEX_LEVEL_KEYS for k in overrides)


# ---------------------------------------------------------------------------
# Retrieval probe (real engines, no synthesis)
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    query: str
    scope: str
    pre_nodes: List[Any]
    post_nodes: List[Any]
    timings: Dict[str, float]

    @property
    def pre_docs(self) -> List[str]:
        return dedupe_docs(self.pre_nodes)

    @property
    def post_docs(self) -> List[str]:
        return dedupe_docs(self.post_nodes)


class RetrievalProbe:
    """Retrieval-only access to UltraRAG's configured pipelines.

    Builds engines exactly like ``main.UltraRAG`` does, then calls the
    retriever and the node postprocessors directly, so one retrieval pass
    yields both the candidate set (pre-rerank) and the final set (post-rerank)
    without an LLM synthesis call.
    """

    def __init__(self, rag: Any, config: Any = None):
        self.rag = rag
        self.config = config or rag.config
        self._vault_engine = None
        self._federated_engine = None

    # -- engine construction ------------------------------------------------

    def rebuild(self, config: Any) -> None:
        """Rebuild engines for a new query-time config (reranker, transformer, fusion...)."""
        from embeddings import get_reranker
        from query_transform import QueryTransformer
        from query_engine import HybridQueryEngine, RAGQueryEngine

        self.config = config
        rag = self.rag

        try:
            reranker = get_reranker(
                model_name=config.retrieval.reranker_model,
                api_key=config.voyage_api_key,
                top_n=config.retrieval.rerank_top_n,
            )
        except Exception as e:  # pragma: no cover - network/credential issue
            logger.warning(f"Reranker unavailable for probe: {e}")
            reranker = None

        try:
            transformer = QueryTransformer(
                llm=rag.llm, embed_model=rag.embed_model,
                hyde_temperature=config.retrieval.hyde_temperature,
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"Query transformer unavailable for probe: {e}")
            transformer = None

        # BM25 top_k is baked in at build time; reuse only when candidates unchanged.
        prebuilt_bm25 = None
        existing = getattr(rag, "bm25_retriever", None)
        if existing is not None and getattr(existing, "similarity_top_k", None) == config.retrieval.retrieval_candidates:
            prebuilt_bm25 = existing

        if config.retrieval.enable_hybrid_search:
            self._vault_engine = HybridQueryEngine(
                index=rag.index, config=config, reranker=reranker,
                bm25_retriever=prebuilt_bm25, nodes=getattr(rag, "nodes", None),
                wikilink_graph=getattr(rag, "wikilink_graph", {}) or {},
                query_transformer=transformer, use_cache=False,
            )
        else:
            self._vault_engine = RAGQueryEngine(
                index=rag.index, config=config, reranker=reranker,
                query_transformer=transformer, use_cache=False,
            )

        self._federated_engine = None
        if getattr(rag, "federated_engine", None) is not None:
            from federated_query import FederatedQueryEngine
            fed = rag.federated_engine
            self._federated_engine = FederatedQueryEngine(
                sources=fed.sources, config=config, reranker=reranker,
                query_transformer=transformer, use_cache=False,
            )

    def _engine(self, scope: str):
        if self._vault_engine is None:
            self.rebuild(self.config)
        if scope == "federated":
            if self._federated_engine is None:
                logger.warning("No federated engine available; falling back to vault scope")
                return self._vault_engine, "vault"
            return self._federated_engine, "federated"
        return self._vault_engine, "vault"

    # -- retrieval ----------------------------------------------------------

    def run(self, query: str, scope: str = "vault") -> ProbeResult:
        from llama_index.core.schema import QueryBundle

        engine, scope = self._engine(scope)
        bundle = QueryBundle(query_str=query)
        if scope == "federated":
            # The two steps FederatedQueryEngine.retrieve runs for UltraRAG.query_federated
            retrieve, postprocess = engine.retrieve_candidates, engine.rerank_candidates
        else:
            inner = getattr(engine, "query_engine", engine)  # RetrieverQueryEngine
            retriever = getattr(inner, "retriever", None) or getattr(inner, "_retriever")
            retrieve = retriever.retrieve
            postprocess = lambda nodes, query_bundle: self._postprocess(inner, nodes, query_bundle)

        timings: Dict[str, float] = {}
        t0 = time.time()
        pre_nodes = list(retrieve(bundle))
        timings["retrieve_ms"] = round((time.time() - t0) * 1000, 1)

        t0 = time.time()
        post_nodes = list(postprocess(pre_nodes, bundle))
        timings["postprocess_ms"] = round((time.time() - t0) * 1000, 1)
        timings["total_ms"] = round(timings["retrieve_ms"] + timings["postprocess_ms"], 1)

        return ProbeResult(query=query, scope=scope, pre_nodes=pre_nodes,
                           post_nodes=post_nodes, timings=timings)

    @staticmethod
    def _postprocess(inner: Any, nodes: List[Any], bundle: Any) -> List[Any]:
        apply = getattr(inner, "_apply_node_postprocessors", None)
        if callable(apply):
            return apply(nodes, query_bundle=bundle)
        # Fallback: apply the list manually
        out = list(nodes)
        for pp in getattr(inner, "_node_postprocessors", []) or []:
            out = pp.postprocess_nodes(out, query_bundle=bundle)
        return out

    # -- pooling helpers for labelling --------------------------------------

    def vector_only(self, query: str, top_k: int = 30) -> List[Any]:
        from llama_index.core.retrievers import VectorIndexRetriever
        return list(VectorIndexRetriever(index=self.rag.index, similarity_top_k=top_k).retrieve(query))

    def bm25_only(self, query: str, top_k: int = 30) -> List[Any]:
        bm25 = getattr(self.rag, "bm25_retriever", None)
        if bm25 is None and self._vault_engine is not None:
            bm25 = getattr(self._vault_engine, "bm25_retriever", None)
        if bm25 is None:
            return []
        nodes = list(bm25.retrieve(query))
        return nodes[:top_k]


# ---------------------------------------------------------------------------
# System bootstrap
# ---------------------------------------------------------------------------

def load_rag_system(config: Any = None) -> Any:
    """Instantiate UltraRAG and load the existing index (vault + optional sources)."""
    from main import UltraRAG

    rag = UltraRAG(config=config)
    if not rag.load_existing_index():
        raise RuntimeError("No existing index found. Index the vault first (python main.py -> index).")
    return rag
