"""Build a document-level golden set for retrieval evaluation.

Workflow (each step writes a file you can inspect/edit):

  1. mine      query histories -> tests/golden/candidates.json   (no API calls)
  2. propose   retrieve + LLM-grade candidates -> tests/golden/proposed.json (+ review.md)
  3. (you)     open proposed.json, fix `relevant`, set "status": "confirmed" per query
  4. finalize  confirmed queries -> tests/golden/golden_set.json

Usage:
  uv run python -m scripts.golden_set mine --n 100 --min-greek 20
  uv run python -m scripts.golden_set mine --add "Τι έχω γράψει για συνήθειες;" --add "my notes on RAG eval"
  uv run python -m scripts.golden_set propose --scope vault
  uv run python -m scripts.golden_set finalize
  uv run python -m scripts.golden_set stats

Labels are pooled from the full pipeline + vector-only + BM25-only retrieval so
the golden set is not biased toward whichever single configuration produced it.
Pooling bias still exists (docs none of the three find are never labelled);
add such docs by hand in `relevant` when you know them.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness import (  # noqa: E402
    GoldenQuery, GoldenSet, doc_id, load_golden_set, node_text, node_title,
    save_golden_set,
)

logger = logging.getLogger("ultrarag.golden_set")

GOLDEN_DIR = Path("tests/golden")
CANDIDATES_PATH = GOLDEN_DIR / "candidates.json"
PROPOSED_PATH = GOLDEN_DIR / "proposed.json"
REVIEW_PATH = GOLDEN_DIR / "review.md"
GOLDEN_PATH = GOLDEN_DIR / "golden_set.json"

PREFIX_RE = re.compile(r"^@(research|all|vault|conv|conversations|books|raptor|web)\s+", re.I)
GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")
LATIN_RE = re.compile(r"[A-Za-z]")

EXHAUSTIVE_RE = re.compile(
    r"\b(all|every|everything|complete list|list all|exhaustive|όλα|όλες|όλους|όλη|κάθε|λίστα)\b", re.I)
COMPARISON_RE = re.compile(r"\b(vs\.?|versus|compare|comparison|difference|differences|σύγκριση|διαφορ\w+|ή)\b", re.I)
CONCEPTUAL_RE = re.compile(r"\b(why|how|explain|what is|what are|γιατί|πώς|τι είναι|εξήγησ\w+)\b", re.I)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def normalize_query(q: str) -> str:
    q = PREFIX_RE.sub("", q.strip())
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def dedupe_key(q: str) -> str:
    return re.sub(r"[^\w\s]", "", normalize_query(q).lower()).strip()


def detect_language(q: str) -> str:
    greek = len(GREEK_RE.findall(q))
    latin = len(LATIN_RE.findall(q))
    if greek == 0 and latin == 0:
        return "en"
    if greek and latin:
        ratio = greek / (greek + latin)
        if 0.2 <= ratio <= 0.8:
            return "mixed"
        return "el" if ratio > 0.8 else "en"
    return "el" if greek else "en"


def classify_category(q: str) -> str:
    if EXHAUSTIVE_RE.search(q):
        return "exhaustive"
    if COMPARISON_RE.search(q):
        return "comparison"
    if CONCEPTUAL_RE.search(q):
        return "conceptual"
    return "factual"


def scope_hint(raw_query: str) -> str:
    m = PREFIX_RE.match(raw_query.strip())
    if not m:
        return "vault"
    tag = m.group(1).lower()
    return "vault" if tag in ("vault", "raptor") else "federated"


def extract_history_queries(payload: Any) -> List[Dict[str, Any]]:
    """Accept {"queries": [...]} or a bare list; return [{query, timestamp}]."""
    items = payload.get("queries", []) if isinstance(payload, dict) else payload
    out = []
    for it in items or []:
        if isinstance(it, dict) and it.get("query"):
            out.append({"query": str(it["query"]), "timestamp": str(it.get("timestamp", ""))})
        elif isinstance(it, str):
            out.append({"query": it, "timestamp": ""})
    return out


def build_candidates(
    raw_queries: Iterable[Dict[str, Any]],
    n: int = 100,
    min_greek: int = 20,
    min_words: int = 2,
    max_words: int = 40,
    seed: int = 42,
    per_category_cap: Optional[int] = None,
) -> List[GoldenQuery]:
    """Dedupe, classify and stratified-sample history queries into candidates."""
    rng = random.Random(seed)
    seen: set = set()
    pool: List[GoldenQuery] = []
    for i, item in enumerate(raw_queries):
        raw = item["query"]
        q = normalize_query(raw)
        key = dedupe_key(q)
        wc = len(q.split())
        if not key or key in seen or wc < min_words or wc > max_words:
            continue
        seen.add(key)
        pool.append(GoldenQuery(
            id=f"h{len(pool):04d}", query=q, relevant={}, language=detect_language(q),
            category=classify_category(q), scope=scope_hint(raw), status="candidate",
            source="history", notes=f"history ts={item.get('timestamp', '')}",
        ))

    if len(pool) <= n:
        for i, q in enumerate(pool):
            q.id = f"g{i:04d}"
        return pool

    rng.shuffle(pool)
    greek = [q for q in pool if q.language in ("el", "mixed")]
    others = [q for q in pool if q.language == "en"]

    chosen: List[GoldenQuery] = greek[:min_greek]
    remaining = greek[min_greek:] + others
    rng.shuffle(remaining)

    cap = per_category_cap or max(10, n // 3)
    counts = Counter(q.category for q in chosen)
    for q in remaining:
        if len(chosen) >= n:
            break
        if counts[q.category] >= cap:
            continue
        chosen.append(q)
        counts[q.category] += 1
    # fill if caps left room
    for q in remaining:
        if len(chosen) >= n:
            break
        if q not in chosen:
            chosen.append(q)

    for i, q in enumerate(chosen):
        q.id = f"g{i:04d}"
    return chosen


def parse_grades(text: str) -> Dict[int, Dict[str, Any]]:
    """Parse the LLM's JSON list [{"idx": int, "grade": 0|1|2, "reason": str}] robustly."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or "idx" not in it:
            continue
        try:
            idx = int(it["idx"])
            grade = max(0, min(2, int(it.get("grade", 0))))
        except (TypeError, ValueError):
            continue
        out[idx] = {"grade": grade, "reason": str(it.get("reason", ""))[:300]}
    return out


GRADING_PROMPT = """You are labelling a retrieval evaluation set for a personal knowledge base
(Obsidian notes, books, AI conversations; Greek and English). For the QUERY below, grade each
candidate DOCUMENT by how useful it would be to answer the query from this person's own notes.

Grades:
  2 = directly answers or is clearly central to the query
  1 = partially relevant / useful supporting context
  0 = not relevant (same words, different topic counts as 0)

Be strict: most candidates should be 0. Grade 1 only if the person would actually want this
document in the answer; grade 2 only if leaving it out would make the answer wrong or incomplete.

Judge the document, not the excerpt quality. Cross-language matches (Greek query, English doc
or vice versa) are fine when the topic matches.

QUERY: {query}

DOCUMENTS:
{docs}

Return ONLY a JSON list, one object per document, like:
[{{"idx": 0, "grade": 2, "reason": "..."}}, {{"idx": 1, "grade": 0, "reason": "..."}}]
"""


def format_docs_for_prompt(cands: List[Dict[str, Any]], excerpt_chars: int = 500) -> str:
    lines = []
    for i, c in enumerate(cands):
        ex = re.sub(r"\s+", " ", c["excerpt"])[:excerpt_chars]
        lines.append(f"[{i}] TITLE: {c['title']}\n    SOURCE: {c.get('source_type', '?')}\n    EXCERPT: {ex}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_mine(args: argparse.Namespace) -> int:
    raw: List[Dict[str, Any]] = []
    paths = [Path(p) for p in args.history] if args.history else sorted(Path("data/users").glob("*/query_history.json"))
    for p in paths:
        if not p.exists():
            logger.warning(f"History not found: {p}")
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not read {p}: {e}")
            continue
        got = extract_history_queries(payload)
        logger.info(f"{p}: {len(got)} queries")
        raw.extend(got)

    for extra in args.add or []:
        raw.append({"query": extra, "timestamp": "manual"})

    cands = build_candidates(raw, n=args.n, min_greek=args.min_greek, seed=args.seed)
    for q in cands:
        if q.notes.endswith("ts=manual"):
            q.source = "manual"

    gs = GoldenSet(queries=cands, description="Candidate queries mined from query history (unlabelled)")
    save_golden_set(gs, args.out)
    _print_stats(gs, title=f"Candidates written to {args.out}")
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    from eval_harness import RetrievalProbe, load_rag_system

    src = load_golden_set(args.candidates)
    todo = [q for q in src.queries if q.status in ("candidate", "proposed") and (not args.only or q.id in args.only)]
    if not todo:
        print("Nothing to propose (no queries with status candidate/proposed).")
        return 0

    rag = load_rag_system()
    probe = RetrievalProbe(rag)
    probe.rebuild(rag.config)
    llm = rag.llm

    proposed: List[Dict[str, Any]] = []
    # resume: keep already-proposed/confirmed entries from an earlier run
    if PROPOSED_PATH.exists() and not args.fresh:
        prev = json.loads(PROPOSED_PATH.read_text(encoding="utf-8"))
        keep = {q["id"]: q for q in prev.get("queries", []) if q.get("status") in ("proposed", "confirmed")}
        if keep:
            print(f"Resuming: {len(keep)} queries already done in {PROPOSED_PATH} (use --fresh to redo)")
    else:
        keep = {}

    for i, gq in enumerate(todo, 1):
        if gq.id in keep:
            proposed.append(keep[gq.id])
            continue
        scope = args.scope or gq.scope
        print(f"[{i}/{len(todo)}] ({scope}) {gq.query[:80]}")
        try:
            res = probe.run(gq.query, scope=scope)
            pooled: List[Any] = list(res.post_nodes[:args.pool_post]) + list(res.pre_nodes[:args.pool_pre])
            if scope == "vault" or args.pool_vault_extra:
                pooled += probe.vector_only(gq.query, top_k=args.pool_vector)
                pooled += probe.bm25_only(gq.query, top_k=args.pool_bm25)
        except Exception as e:
            logger.error(f"Retrieval failed for {gq.id}: {e}")
            continue

        # one entry per document: first (best-ranked) excerpt wins
        cands: Dict[str, Dict[str, Any]] = {}
        for n in pooled:
            d = doc_id(n)
            if d in cands:
                continue
            inner = getattr(n, "node", n)
            meta = getattr(inner, "metadata", {}) or {}
            cands[d] = {"doc_id": d, "title": node_title(n), "excerpt": node_text(n)[:800],
                        "source_type": meta.get("source_type", "vault"), "grade": None, "reason": ""}
        cand_list = list(cands.values())[:args.max_candidates]

        # grade in batches
        for b in range(0, len(cand_list), args.batch):
            batch = cand_list[b:b + args.batch]
            prompt = GRADING_PROMPT.format(query=gq.query, docs=format_docs_for_prompt(batch))
            try:
                text = str(llm.complete(prompt))
            except Exception as e:
                logger.error(f"LLM grading failed for {gq.id} batch {b}: {e}")
                continue
            grades = parse_grades(text)
            for j, c in enumerate(batch):
                g = grades.get(j)
                if g:
                    c["grade"], c["reason"] = g["grade"], g["reason"]

        relevant = {c["doc_id"]: c["grade"] for c in cand_list if (c["grade"] or 0) >= 1}
        entry = {
            "id": gq.id, "query": gq.query, "language": gq.language, "category": gq.category,
            "scope": scope, "status": "proposed", "source": gq.source, "notes": gq.notes,
            "relevant": relevant,
            "candidates": cand_list,
            "retrieval_timings": res.timings,
        }
        proposed.append(entry)
        if args.checkpoint_every and i % args.checkpoint_every == 0:
            _write_proposed(proposed, src.description)

    _write_proposed(proposed, src.description)
    _write_review_md(proposed)
    print(f"\nWrote {PROPOSED_PATH} and {REVIEW_PATH}.")
    print("Next: review `relevant` per query in proposed.json, set \"status\": \"confirmed\", then run `finalize`.")
    return 0


def _write_proposed(entries: List[Dict[str, Any]], description: str) -> None:
    PROPOSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "created": datetime.now(timezone.utc).isoformat(),
               "description": description + " | LLM-proposed labels, pending human confirmation",
               "queries": entries}
    PROPOSED_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_review_md(entries: List[Dict[str, Any]]) -> None:
    lines = ["# Golden set review", "",
             "Edit `tests/golden/proposed.json`: fix `relevant` (doc_id -> 1|2, remove wrong ones, add missing ones),",
             "set `\"status\": \"confirmed\"` per query, then run `uv run python -m scripts.golden_set finalize`.", ""]
    for e in entries:
        lines.append(f"## {e['id']} · {e['language']} · {e['category']} · {e['scope']} · {e['status']}")
        lines.append(f"**{e['query']}**")
        lines.append("")
        lines.append("| grade | doc | reason |")
        lines.append("|---|---|---|")
        for c in sorted(e.get("candidates", []), key=lambda c: -(c.get("grade") or 0)):
            g = c.get("grade")
            lines.append(f"| {'' if g is None else g} | `{c['title'][:60]}` | {c.get('reason', '')[:120]} |")
        lines.append("")
    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def cmd_finalize(args: argparse.Namespace) -> int:
    if not PROPOSED_PATH.exists():
        print(f"Missing {PROPOSED_PATH}; run `propose` first.")
        return 1
    payload = json.loads(PROPOSED_PATH.read_text(encoding="utf-8"))
    queries: List[GoldenQuery] = []
    skipped = Counter()
    for e in payload.get("queries", []):
        status = e.get("status")
        if args.accept_proposed and status == "proposed":
            status = "confirmed"
        if status != "confirmed":
            skipped["not confirmed"] += 1
            continue
        rel = {str(k): int(v) for k, v in (e.get("relevant") or {}).items() if int(v) >= 1}
        if not rel:
            skipped["no relevant docs"] += 1
            continue
        queries.append(GoldenQuery(
            id=e["id"], query=e["query"], relevant=rel, language=e.get("language", "en"),
            category=e.get("category", "factual"), scope=e.get("scope", "vault"),
            status="confirmed", notes=e.get("notes", ""), source=e.get("source", "history"),
        ))

    # merge with existing golden set (keeps hand-written entries)
    if args.out.exists() and not args.replace:
        existing = load_golden_set(args.out)
        ids = {q.id for q in queries}
        queries = [q for q in existing.queries if q.id not in ids] + queries

    gs = GoldenSet(queries=queries, description="Document-level retrieval golden set (human-confirmed)")
    save_golden_set(gs, args.out)
    _print_stats(gs, title=f"Golden set written to {args.out}")
    if skipped:
        print("Skipped:", dict(skipped))
    if args.accept_proposed:
        print("WARNING: --accept-proposed used; labels are LLM-only. Treat metrics as provisional.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    path = args.path or GOLDEN_PATH
    if not Path(path).exists():
        print(f"Not found: {path}")
        return 1
    _print_stats(load_golden_set(Path(path)), title=str(path))
    return 0


def _print_stats(gs: GoldenSet, title: str = "") -> None:
    qs = gs.queries
    print(f"\n{title}")
    print(f"  queries: {len(qs)}  confirmed-with-labels: {len(gs.confirmed())}")
    for key in ("language", "category", "scope", "status"):
        print(f"  by {key}: {dict(Counter(getattr(q, key) for q in qs))}")
    labelled = [q for q in qs if q.relevant]
    if labelled:
        avg = sum(len(q.relevant) for q in labelled) / len(labelled)
        print(f"  avg relevant docs/query: {avg:.1f}")


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="mine query histories into candidates")
    m.add_argument("--history", nargs="*", help="history JSON files (default: data/users/*/query_history.json)")
    m.add_argument("--add", action="append", help="add a manual query (repeatable)")
    m.add_argument("--n", type=int, default=100)
    m.add_argument("--min-greek", type=int, default=20)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--out", type=Path, default=CANDIDATES_PATH)
    m.set_defaults(func=cmd_mine)

    pr = sub.add_parser("propose", help="retrieve + LLM-grade candidates")
    pr.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    pr.add_argument("--scope", choices=["vault", "federated"], default=None, help="override per-query scope")
    pr.add_argument("--only", nargs="*", help="limit to these query ids")
    pr.add_argument("--fresh", action="store_true", help="redo everything; ignore entries already in proposed.json")
    pr.add_argument("--pool-post", type=int, default=25)
    pr.add_argument("--pool-pre", type=int, default=25)
    pr.add_argument("--pool-vector", type=int, default=20)
    pr.add_argument("--pool-bm25", type=int, default=20)
    pr.add_argument("--pool-vault-extra", action="store_true", help="also pool vault vector/bm25 for federated queries")
    pr.add_argument("--max-candidates", type=int, default=45)
    pr.add_argument("--batch", type=int, default=15, help="docs per LLM grading call")
    pr.add_argument("--checkpoint-every", type=int, default=5)
    pr.set_defaults(func=cmd_propose)

    f = sub.add_parser("finalize", help="write confirmed queries to golden_set.json")
    f.add_argument("--out", type=Path, default=GOLDEN_PATH)
    f.add_argument("--replace", action="store_true", help="replace instead of merge")
    f.add_argument("--accept-proposed", action="store_true", help="accept LLM labels without review (provisional)")
    f.set_defaults(func=cmd_finalize)

    s = sub.add_parser("stats")
    s.add_argument("path", nargs="?", type=Path)
    s.set_defaults(func=cmd_stats)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
