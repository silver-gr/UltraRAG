"""Ablation runner: score retrieval configs against the golden set.

One row per configuration, document-level IR metrics, no LLM synthesis.
Query-time flags rebuild the engines in-process; index-level flags
(EMBEDDING_MODEL, VAULT_TABLE, USE_CONTEXTUAL_RETRIEVAL, ...) reload the system.

Examples:
  # baseline with the current .env
  uv run python -m scripts.run_ablation --save-baseline

  # one grid, cartesian product (3 x 2 = 6 configs)
  uv run python -m scripts.run_ablation \
      --grid QUERY_TRANSFORM_METHOD=none,hyde,multi_query \
      --grid USE_SELF_CORRECTION=true,false

  # fusion tuning
  uv run python -m scripts.run_ablation --grid RRF_K=20,60,100 --grid FUSION_BM25_WEIGHT=0.5,1.0,1.5

  # contextual retrieval: index the vault into a second table first, then
  uv run python -m scripts.run_ablation --grid VAULT_TABLE=obsidian_embeddings,obsidian_ctx

  # regression gate against a saved baseline (exit 1 if ndcg@10 or recall@20 drop > 3 points)
  uv run python -m scripts.run_ablation --baseline data/evaluation/baseline.json --gate 3

  # explicit matrix file: [{"name": "...", "overrides": {"KEY": "val"}}, ...]
  uv run python -m scripts.run_ablation --matrix docs/plans/ablation_matrix.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness import (  # noqa: E402
    GoldenQuery, RetrievalProbe, aggregate, apply_env_overrides, load_golden_set,
    load_rag_system, score_query, DEFAULT_GOLDEN_PATH, INDEX_LEVEL_KEYS,
)

logger = logging.getLogger("ultrarag.ablation")

ABLATION_DIR = Path("data/evaluation/ablations")
BASELINE_PATH = Path("data/evaluation/baseline.json")
GATE_METRICS = ("ndcg@10_post", "recall@20_pre")


# ---------------------------------------------------------------------------
# Config matrix
# ---------------------------------------------------------------------------

def parse_kv(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


def build_matrix(grids: List[str], sets: List[str], matrix_file: Optional[Path]) -> List[Dict[str, Any]]:
    """Cartesian product of --grid values, with --set applied to every config."""
    base: Dict[str, str] = dict(parse_kv(s) for s in sets or [])
    configs: List[Dict[str, Any]] = []

    if matrix_file:
        for item in json.loads(Path(matrix_file).read_text(encoding="utf-8")):
            ov = {**base, **{k: str(v) for k, v in item.get("overrides", {}).items()}}
            configs.append({"name": item.get("name") or _auto_name(ov), "overrides": ov})

    if grids:
        axes = []
        for g in grids:
            k, vals = parse_kv(g)
            axes.append([(k, v) for v in vals.split(",") if v != ""])
        for combo in itertools.product(*axes):
            ov = {**base, **dict(combo)}
            configs.append({"name": _auto_name(dict(combo)), "overrides": ov})

    if not configs:
        configs.append({"name": "current" if not base else _auto_name(base), "overrides": base})
    return configs


def _auto_name(ov: Dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in ov.items()) or "current"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def select_queries(queries: List[GoldenQuery], args: argparse.Namespace) -> List[GoldenQuery]:
    qs = queries
    if args.category:
        qs = [q for q in qs if q.category in args.category]
    if args.language:
        qs = [q for q in qs if q.language in args.language]
    if args.ids:
        qs = [q for q in qs if q.id in args.ids]
    if args.limit:
        qs = qs[: args.limit]
    return qs


def run_config(
    probe: RetrievalProbe,
    queries: List[GoldenQuery],
    scope_override: Optional[str],
    k_pre: int,
    k_post: int,
    keep_docs: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    t_start = time.time()
    for i, gq in enumerate(queries, 1):
        scope = scope_override or gq.scope
        try:
            res = probe.run(gq.query, scope=scope)
        except Exception as e:  # keep going; record failure
            logger.error(f"[{gq.id}] retrieval failed: {e}")
            failures.append({"id": gq.id, "error": str(e)[:300]})
            continue
        pre, post = res.pre_docs, res.post_docs
        metrics = score_query(pre, post, gq.relevant, k_pre=k_pre, k_post=k_post)
        metrics.update({f"t_{k}": v for k, v in res.timings.items()})
        rows.append({
            "id": gq.id, "category": gq.category, "language": gq.language, "scope": res.scope,
            "metrics": metrics,
            "pre_docs": pre[:keep_docs], "post_docs": post[:keep_docs],
            "missed": [d for d in gq.relevant if d not in post[:k_post]],
        })
        if i % 10 == 0 or i == len(queries):
            agg = aggregate(rows)["overall"]
            print(f"    {i}/{len(queries)}  ndcg@{k_post}={agg.get(f'ndcg@{k_post}_post', 0):.3f} "
                  f"recall@{k_pre}pre={agg.get(f'recall@{k_pre}_pre', 0):.3f}  "
                  f"avg {agg.get('t_total_ms', 0):.0f} ms")
    summary = aggregate(rows, group_key="category")
    summary.update({k: v for k, v in aggregate(rows, group_key="language").items() if k.startswith("by_")})
    summary["wall_s"] = round(time.time() - t_start, 1)
    summary["failures"] = failures
    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def results_table(results: List[Dict[str, Any]], k_pre: int, k_post: int) -> str:
    cols = [f"recall@{k_pre}_pre", "recall@50_pre", f"recall@{k_post}_post", f"ndcg@{k_post}_post",
            "mrr_post", "hit@1_post", f"p@{k_post}_post", "t_total_ms"]
    head = "| config | n | " + " | ".join(c.replace("_post", "").replace("_pre", "(pre)") for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [head, sep]
    for r in results:
        o = r["summary"]["overall"]
        vals = []
        for c in cols:
            v = o.get(c, 0.0)
            vals.append(f"{v:.0f}" if c.startswith("t_") else f"{v:.3f}")
        lines.append(f"| `{r['name']}` | {o.get('n', 0)} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def breakdown_table(result: Dict[str, Any], key: str, k_post: int) -> str:
    groups = result["summary"].get(f"by_{key}", {})
    if not groups:
        return ""
    lines = [f"| {key} | n | ndcg@{k_post} | recall@{k_post} | mrr |", "|---|---|---|---|---|"]
    for g, m in groups.items():
        lines.append(f"| {g} | {m.get('n', 0)} | {m.get(f'ndcg@{k_post}_post', 0):.3f} | "
                     f"{m.get(f'recall@{k_post}_post', 0):.3f} | {m.get('mrr_post', 0):.3f} |")
    return "\n".join(lines)


def compare_to_baseline(current: Dict[str, Any], baseline: Dict[str, Any], gate_points: float) -> List[str]:
    """Return list of violations ('metric: base -> cur (Δ)') exceeding gate_points (×100)."""
    cur = current["summary"]["overall"]
    base = baseline["summary"]["overall"]
    violations = []
    for m in GATE_METRICS:
        if m in cur and m in base:
            delta = (cur[m] - base[m]) * 100
            if delta < -gate_points:
                violations.append(f"{m}: {base[m]:.3f} -> {cur[m]:.3f} ({delta:+.1f} pts)")
    return violations


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("ultrarag").setLevel(logging.INFO)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    p.add_argument("--grid", action="append", default=[], help="KEY=v1,v2,... (repeatable; cartesian product)")
    p.add_argument("--set", action="append", default=[], help="KEY=value applied to every config")
    p.add_argument("--matrix", type=Path, help="JSON list of {name, overrides}")
    p.add_argument("--scope", choices=["vault", "federated"], default=None, help="override per-query scope")
    p.add_argument("--k-pre", type=int, default=20)
    p.add_argument("--k-post", type=int, default=10)
    p.add_argument("--keep-docs", type=int, default=20, help="ranked doc ids stored per query for debugging")
    p.add_argument("--limit", type=int)
    p.add_argument("--category", nargs="*")
    p.add_argument("--language", nargs="*")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--out", type=Path, help="output dir (default data/evaluation/ablations/<timestamp>)")
    p.add_argument("--baseline", type=Path, help="results.json / baseline.json to compare the first config against")
    p.add_argument("--gate", type=float, default=0.0, help="fail (exit 1) if a gate metric drops more than N points")
    p.add_argument("--save-baseline", action="store_true", help="copy the first config's result to data/evaluation/baseline.json")
    p.add_argument("--allow-unconfirmed", action="store_true", help="include queries whose status != confirmed")
    args = p.parse_args(argv)

    gs = load_golden_set(args.golden)
    queries = gs.queries if args.allow_unconfirmed else gs.confirmed()
    queries = select_queries([q for q in queries if q.relevant], args)
    if not queries:
        print("No labelled queries to evaluate. Build the golden set first (scripts/golden_set.py).")
        return 1

    configs = build_matrix(args.grid, args.set, args.matrix)
    out_dir = args.out or ABLATION_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Golden set: {args.golden} ({len(queries)} queries)  configs: {len(configs)}  out: {out_dir}")

    from config import load_config

    rag = None
    probe: Optional[RetrievalProbe] = None
    results: List[Dict[str, Any]] = []
    rag_env: Dict[str, str] = {}

    for ci, cfg in enumerate(configs, 1):
        ov = cfg["overrides"]
        print(f"\n[{ci}/{len(configs)}] {cfg['name']}")
        with apply_env_overrides(ov):
            config = load_config()
            index_keys = {k: v for k, v in ov.items() if k in INDEX_LEVEL_KEYS}
            if rag is None or index_keys != rag_env:
                print("    loading system (index-level change or first run)...")
                rag = load_rag_system(config=config)
                rag_env = index_keys
                probe = RetrievalProbe(rag, config)
            assert probe is not None
            probe.rebuild(config)
            res = run_config(probe, queries, args.scope, args.k_pre, args.k_post, args.keep_docs)
        res["name"], res["overrides"] = cfg["name"], ov
        res["config_snapshot"] = _config_snapshot(config)
        results.append(res)
        (out_dir / f"{ci:02d}_{_slug(cfg['name'])}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    # report
    md = [f"# Ablation {out_dir.name}", "", f"Golden set: `{args.golden}` — {len(queries)} queries", "",
          results_table(results, args.k_pre, args.k_post), ""]
    for r in results:
        md += [f"## {r['name']}", "", breakdown_table(r, "category", args.k_post), "",
               breakdown_table(r, "language", args.k_post), ""]
        if r["summary"].get("failures"):
            md += [f"Failures: {len(r['summary']['failures'])}", ""]
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(
        {"golden": str(args.golden), "n_queries": len(queries), "k_pre": args.k_pre, "k_post": args.k_post,
         "results": [{k: v for k, v in r.items() if k != "rows"} for r in results]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + results_table(results, args.k_pre, args.k_post))
    print(f"\nWritten: {out_dir / 'results.md'}")

    exit_code = 0
    if args.baseline and args.baseline.exists():
        base = json.loads(args.baseline.read_text(encoding="utf-8"))
        base = base["results"][0] if "results" in base else base
        violations = compare_to_baseline(results[0], base, args.gate)
        if violations:
            print("\nREGRESSION vs baseline:\n  " + "\n  ".join(violations))
            if args.gate > 0:
                exit_code = 1
        else:
            print(f"\nNo regression vs baseline (gate {args.gate} pts on {', '.join(GATE_METRICS)}).")

    if args.save_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_dir / f"01_{_slug(configs[0]['name'])}.json", BASELINE_PATH)
        print(f"Baseline saved: {BASELINE_PATH}")
    return exit_code


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)[:60].strip("_") or "config"


def _config_snapshot(config: Any) -> Dict[str, Any]:
    r, e = config.retrieval, config.embedding
    return {
        "embedding_model": e.model, "chunking_strategy": e.chunking_strategy,
        "contextual_retrieval": e.use_contextual_retrieval, "vault_table": config.vector_db.vault_table,
        "reranker": r.reranker_model, "rerank_top_n": r.rerank_top_n,
        "retrieval_candidates": r.retrieval_candidates, "hybrid": r.enable_hybrid_search,
        "query_transform": r.query_transform_method, "self_correction": r.use_self_correction,
        "graph_retrieval": r.enable_graph_retrieval, "mmr": r.enable_mmr,
        "rrf_k": r.rrf_k, "fusion_vector_weight": r.fusion_vector_weight,
        "fusion_bm25_weight": r.fusion_bm25_weight, "rerank_score_threshold": r.rerank_score_threshold,
        "query_decomposition": r.enable_query_decomposition, "bilingual_expansion": r.enable_bilingual_expansion,
    }


if __name__ == "__main__":
    sys.exit(main())
