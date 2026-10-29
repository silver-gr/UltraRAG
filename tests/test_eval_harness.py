"""Unit tests for eval_harness + golden_set + run_ablation pure helpers (no LlamaIndex needed)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_harness as eh  # noqa: E402
from scripts import golden_set as gsm  # noqa: E402
from scripts import run_ablation as ra  # noqa: E402


def _node(file_path=None, score=0.5, **meta):
    md = dict(meta)
    if file_path:
        md["file_path"] = file_path
    inner = SimpleNamespace(metadata=md, text=meta.get("text", "body"), node_id="nid-" + str(file_path))
    return SimpleNamespace(node=inner, score=score)


# --- doc identity -----------------------------------------------------------

def test_doc_id_prefers_file_path_then_url_then_name():
    assert eh.doc_id(_node("a/b.md")) == "a/b.md"
    assert eh.doc_id(_node(None, url="https://x/y")) == "https://x/y"
    assert eh.doc_id(_node(None, file_name="n.md")) == "n.md"
    assert eh.doc_id({"book_title": "T"}) == "T"
    assert eh.doc_id(_node(None)).startswith("nid-")


def test_dedupe_docs_keeps_rank_order():
    nodes = [_node("a"), _node("b"), _node("a"), _node("c"), _node("b")]
    assert eh.dedupe_docs(nodes) == ["a", "b", "c"]


# --- metrics ----------------------------------------------------------------

def test_metrics_perfect_ranking():
    rel = {"a": 2, "b": 1}
    m = eh.score_query(["a", "b", "c"], ["a", "b", "c"], rel, k_pre=20, k_post=10)
    assert m["ndcg@10_post"] == pytest.approx(1.0)
    assert m["recall@20_pre"] == 1.0
    assert m["recall@10_post"] == 1.0
    assert m["mrr_post"] == 1.0
    assert m["hit@1_post"] == 1.0
    assert m["p@10_post"] == pytest.approx(0.2)


def test_metrics_partial_and_graded():
    rel = {"a": 2, "b": 1}
    # 'b' ranked first, 'a' missing from post -> ndcg < 1, recall 0.5, mrr 1
    m = eh.score_query(["b", "a"], ["b", "x", "y"], rel, k_pre=20, k_post=10)
    assert m["recall@20_pre"] == 1.0
    assert m["recall@10_post"] == 0.5
    assert 0 < m["ndcg@10_post"] < 1
    assert m["mrr_post"] == 1.0
    # reversed grade order scores lower than ideal order
    better = eh.ndcg_at_k(["a", "b"], rel, 10)
    worse = eh.ndcg_at_k(["b", "a"], rel, 10)
    assert better == pytest.approx(1.0) and worse < better


def test_metrics_no_hits():
    m = eh.score_query(["x"], ["x"], {"a": 1})
    assert m["ndcg@10_post"] == 0 and m["mrr_post"] == 0 and m["recall@20_pre"] == 0


def test_aggregate_with_groups():
    rows = [
        {"category": "factual", "metrics": {"ndcg@10_post": 1.0, "t_total_ms": 100}},
        {"category": "factual", "metrics": {"ndcg@10_post": 0.0, "t_total_ms": 300}},
        {"category": "exhaustive", "metrics": {"ndcg@10_post": 0.5, "t_total_ms": 200}},
    ]
    agg = eh.aggregate(rows, group_key="category")
    assert agg["overall"]["ndcg@10_post"] == pytest.approx(0.5)
    assert agg["overall"]["n"] == 3
    assert agg["by_category"]["factual"]["ndcg@10_post"] == pytest.approx(0.5)
    assert agg["by_category"]["exhaustive"]["n"] == 1
    assert eh.aggregate([]) == {"n": 0}


# --- golden set io ----------------------------------------------------------

def test_golden_set_roundtrip_and_list_tolerance(tmp_path):
    p = tmp_path / "g.json"
    gs = eh.GoldenSet(queries=[eh.GoldenQuery(id="q1", query="x", relevant={"a": 2}, language="el")])
    eh.save_golden_set(gs, p)
    back = eh.load_golden_set(p)
    assert back.queries[0].relevant == {"a": 2} and back.queries[0].language == "el"
    assert len(back.confirmed()) == 1

    p.write_text(json.dumps({"queries": [{"id": "q2", "query": "y", "relevant": ["d1", "d2"]}]}), encoding="utf-8")
    back = eh.load_golden_set(p)
    assert back.queries[0].relevant == {"d1": 1, "d2": 1}


# --- env overrides ----------------------------------------------------------

def test_apply_env_overrides_restores():
    os.environ["ULTRARAG_TEST_X"] = "orig"
    os.environ.pop("ULTRARAG_TEST_Y", None)
    with eh.apply_env_overrides({"ULTRARAG_TEST_X": "new", "ULTRARAG_TEST_Y": "1"}):
        assert os.environ["ULTRARAG_TEST_X"] == "new" and os.environ["ULTRARAG_TEST_Y"] == "1"
    assert os.environ["ULTRARAG_TEST_X"] == "orig" and "ULTRARAG_TEST_Y" not in os.environ
    assert eh.needs_index_reload({"VAULT_TABLE": "x"}) and not eh.needs_index_reload({"RRF_K": "20"})


# --- golden_set script helpers ----------------------------------------------

def test_normalize_and_scope_hint():
    assert gsm.normalize_query("@research   what   is   RAG ") == "what is RAG"
    assert gsm.scope_hint("@research x") == "federated"
    assert gsm.scope_hint("@vault x") == "vault"
    assert gsm.scope_hint("plain") == "vault"
    assert gsm.dedupe_key("What is RAG?") == gsm.dedupe_key("what is rag")


def test_language_and_category():
    assert gsm.detect_language("Τι έχω γράψει για συνήθειες;") == "el"
    assert gsm.detect_language("my notes on habits") == "en"
    assert gsm.detect_language("RAG pipeline για το vault μου") == "mixed"
    assert gsm.classify_category("list all my habit techniques") == "exhaustive"
    assert gsm.classify_category("magnesium threonate vs glycinate") == "comparison"
    assert gsm.classify_category("why does late chunking help") == "conceptual"
    assert gsm.classify_category("neo4j bolt port") == "factual"
    assert gsm.classify_category("Όλες οι σημειώσεις για ύπνο") == "exhaustive"


def test_extract_history_queries_both_shapes():
    assert gsm.extract_history_queries({"queries": [{"query": "a", "timestamp": "t"}]}) == [{"query": "a", "timestamp": "t"}]
    assert gsm.extract_history_queries([{"query": "b"}, "c", {"nope": 1}]) == [
        {"query": "b", "timestamp": ""}, {"query": "c", "timestamp": ""}]


def test_build_candidates_dedupes_and_reserves_greek():
    raw = [{"query": f"english query number {i} about topic {i % 7}", "timestamp": ""} for i in range(60)]
    raw += [{"query": "English Query Number 1 about topic 1", "timestamp": ""}]  # dup
    raw += [{"query": f"ελληνική ερώτηση {i} για συνήθειες", "timestamp": ""} for i in range(10)]
    raw += [{"query": "x", "timestamp": ""}]  # too short
    cands = gsm.build_candidates(raw, n=30, min_greek=8, seed=1)
    assert len(cands) == 30
    assert sum(1 for c in cands if c.language == "el") >= 8
    assert len({c.query.lower() for c in cands}) == 30
    assert all(c.status == "candidate" and c.relevant == {} for c in cands)
    assert cands[0].id == "g0000"


def test_parse_grades_robust():
    txt = 'Sure:\n```json\n[{"idx": 0, "grade": 2, "reason": "central"}, {"idx": "1", "grade": 5}, {"bad": 1}]\n```'
    g = gsm.parse_grades(txt)
    assert g[0]["grade"] == 2 and g[1]["grade"] == 2 and 2 not in g
    assert gsm.parse_grades("no json here") == {}


# --- run_ablation helpers ---------------------------------------------------

def test_build_matrix_product_and_sets():
    m = ra.build_matrix(["A=1,2", "B=x,y"], ["C=0"], None)
    assert len(m) == 4
    assert all(c["overrides"]["C"] == "0" for c in m)
    assert {c["name"] for c in m} == {"A=1,B=x", "A=1,B=y", "A=2,B=x", "A=2,B=y"}
    assert ra.build_matrix([], [], None) == [{"name": "current", "overrides": {}}]


def test_build_matrix_file(tmp_path):
    f = tmp_path / "m.json"
    f.write_text(json.dumps([{"name": "n1", "overrides": {"RRF_K": 20}}]), encoding="utf-8")
    m = ra.build_matrix([], ["X=1"], f)
    assert m == [{"name": "n1", "overrides": {"X": "1", "RRF_K": "20"}}]


def test_compare_to_baseline_gate():
    cur = {"summary": {"overall": {"ndcg@10_post": 0.60, "recall@20_pre": 0.90}}}
    base = {"summary": {"overall": {"ndcg@10_post": 0.65, "recall@20_pre": 0.89}}}
    v = ra.compare_to_baseline(cur, base, gate_points=3.0)
    assert len(v) == 1 and v[0].startswith("ndcg@10_post")
    assert ra.compare_to_baseline(cur, base, gate_points=6.0) == []


def test_results_table_renders():
    res = [{"name": "cfg", "summary": {"overall": {"n": 3, "ndcg@10_post": 0.5, "recall@20_pre": 0.8,
                                                     "t_total_ms": 123.4}}}]
    md = ra.results_table(res, 20, 10)
    assert "| `cfg` | 3 |" in md and "0.500" in md and "123" in md


def test_doc_id_basename_mode(monkeypatch):
    monkeypatch.setattr(eh, "DOC_KEY_MODE", "basename")
    assert eh.doc_id(_node("/a/b/Note.md")) == "Note.md"
    assert eh.doc_id(_node("/x/y/Note.md")) == "Note.md"
    assert eh.doc_id(_node(None, url="https://x/y")) == "https://x/y"
    monkeypatch.setattr(eh, "DOC_KEY_MODE", "path")
    assert eh.doc_id(_node("/a/b/Note.md")) == "/a/b/Note.md"
