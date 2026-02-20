#!/usr/bin/env python3
"""
Auto-enrich missing book metadata in data/book_metadata_cache.json.

Goal:
- Fill ONLY missing `categories` and/or `description` for entries.
- Be conservative: if match confidence is low, skip and add to a review list.

Sources (no Tavily):
- Google Books public API
- Open Library public APIs

Writes:
- Updates cache in place (incremental writes every N processed targets)
- Low-confidence review list JSON (entries to inspect/remove)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import _normalize_category  # noqa: E402


CACHE_PATH_DEFAULT = Path("data/book_metadata_cache.json")
BACKUP_PATH_DEFAULT = Path("data/book_metadata_cache.codex_backup.json")
LOW_CONF_PATH_DEFAULT = Path("data/book_metadata_low_confidence_review.json")


USER_AGENT = "UltraRAG-Codex/1.0 (metadata enrichment; contact: local)"


def _load_ordered_json(path: Path) -> OrderedDict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def _atomic_write_json(path: Path, data: OrderedDict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _is_nonempty_str(val: Any) -> bool:
    return isinstance(val, str) and bool(val.strip())


def _is_empty_description(val: Any) -> bool:
    return not _is_nonempty_str(val)


def _is_empty_categories(val: Any) -> bool:
    if not isinstance(val, list):
        return True
    return len([c for c in val if str(c).strip()]) == 0


def _truncate_description(text: str, max_len: int = 500) -> str:
    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len].strip()
    # Prefer not to end mid-word if easy
    if " " in cut and not cut.endswith(".") and not cut.endswith("!") and not cut.endswith("?"):
        last_space = cut.rfind(" ")
        if last_space > max_len * 0.8:
            cut = cut[:last_space].strip()
    return cut


def _normalize_categories(raw: list[str], max_items: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for c in raw:
        c = str(c)
        c = c.replace("&", " and ").replace("/", " ").replace("\\", " ").replace("_", " ")
        c = re.sub(r"\s+", " ", c).strip()
        if not c:
            continue
        c = _normalize_category(c)
        if len(c) < 2 or len(c) > 48:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= max_items:
            break
    return out


def _update_metadata_source(existing: str, enriched: bool) -> str:
    if not enriched:
        return existing
    if existing == "calibre":
        return "calibre+web"
    if existing == "filename":
        return "web"
    return existing


_SPACEISH = re.compile(r"[\s/_\-]+")
_NONWORD = re.compile(r"[^a-z0-9 ]+")


def _norm_for_match(s: str) -> str:
    s = str(s or "").lower().strip()
    s = _SPACEISH.sub(" ", s)
    s = _NONWORD.sub("", s)
    return " ".join(s.split())


def _similarity(a: str, b: str) -> float:
    a_n = _norm_for_match(a)
    b_n = _norm_for_match(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def _normalize_isbn(isbn: str) -> str:
    return re.sub(r"[^0-9xX]", "", str(isbn or "")).strip().upper()


def _looks_like_not_a_book(title: str, file_path: str = "") -> bool:
    t = (title or "").strip().lower()
    fp = (file_path or "").strip().lower()
    if not t:
        return True
    # obvious working/project files
    if any(ext in t for ext in (".indd", ".cdr", ".psd", ".ai", ".ppt", ".pptx", ".key", ".numbers", ".pages")):
        return True
    if any(ext in fp for ext in (".indd", ".cdr", ".psd", ".ai")):
        return True
    # chapter/draft-like
    if t.startswith("chapter ") or t.endswith(".rtf") or t.endswith(".doc") or t.endswith(".docx"):
        return True
    # numeric/asset id-ish (very short with digits and separators)
    if len(t) <= 12 and re.fullmatch(r"[0-9\\-_.]+", t):
        return True
    return False


@dataclass(frozen=True)
class Candidate:
    source: str
    title: str
    authors: list[str]
    categories: list[str]
    description: str
    url: str
    score: float


class WebBookLookup:
    def __init__(self, session: requests.Session, delay_s: float = 0.2, timeout_s: float = 12.0):
        self._s = session
        self._delay_s = delay_s
        self._timeout_s = timeout_s
        self._last_req = 0.0

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        now = time.time()
        delta = now - self._last_req
        if delta < self._delay_s:
            time.sleep(self._delay_s - delta)
        self._last_req = time.time()
        try:
            r = self._s.get(url, params=params, timeout=self._timeout_s)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def google_candidates(self, title: str, author: str = "", isbn: str = "") -> list[Candidate]:
        isbn = _normalize_isbn(isbn)
        q_parts: list[str] = []
        if isbn:
            q_parts.append(f"isbn:{isbn}")
        else:
            if title:
                q_parts.append(f'intitle:\"{title}\"')
            if author:
                q_parts.append(f'inauthor:\"{author}\"')
        if not q_parts:
            return []

        data = self._get_json(
            "https://www.googleapis.com/books/v1/volumes",
            params={
                "q": " ".join(q_parts),
                "maxResults": 5,
                "printType": "books",
                "projection": "full",
            },
        ) or {}

        out: list[Candidate] = []
        for item in (data.get("items") or [])[:5]:
            vi = item.get("volumeInfo") or {}
            c_title = str(vi.get("title") or "").strip()
            c_authors = [str(a).strip() for a in (vi.get("authors") or []) if str(a).strip()]
            c_desc = str(vi.get("description") or "").strip()
            c_cats = [str(c).strip() for c in (vi.get("categories") or []) if str(c).strip()]
            info_url = str(vi.get("infoLink") or vi.get("previewLink") or item.get("selfLink") or "").strip()

            score = _similarity(title, c_title)
            if author and c_authors:
                author_best = max((_similarity(author, a) for a in c_authors), default=0.0)
                score = 0.75 * score + 0.25 * author_best

            out.append(
                Candidate(
                    source="google_books",
                    title=c_title,
                    authors=c_authors,
                    categories=c_cats,
                    description=c_desc,
                    url=info_url,
                    score=score,
                )
            )

        out.sort(key=lambda c: c.score, reverse=True)
        return out

    def openlibrary_candidates(self, title: str, author: str = "", isbn: str = "") -> list[Candidate]:
        isbn = _normalize_isbn(isbn)
        out: list[Candidate] = []

        if isbn:
            bib = self._get_json(
                "https://openlibrary.org/api/books",
                params={"bibkeys": f"ISBN:{isbn}", "format": "json", "jscmd": "data"},
            ) or {}
            record = bib.get(f"ISBN:{isbn}") or {}
            c_title = str(record.get("title") or "").strip()
            c_authors = [str(a.get("name", "")).strip() for a in (record.get("authors") or []) if str(a.get("name", "")).strip()]
            subjects = [str(s.get("name", "")).strip() for s in (record.get("subjects") or []) if str(s.get("name", "")).strip()]
            url = str(record.get("url") or "").strip()

            score = _similarity(title, c_title)
            if author and c_authors:
                author_best = max((_similarity(author, a) for a in c_authors), default=0.0)
                score = 0.75 * score + 0.25 * author_best

            out.append(
                Candidate(
                    source="openlibrary",
                    title=c_title,
                    authors=c_authors,
                    categories=subjects,
                    description="",
                    url=url,
                    score=score,
                )
            )

            works = record.get("works") or []
            if works:
                work_key = works[0].get("key")
                if work_key:
                    work = self._get_json(f"https://openlibrary.org{work_key}.json") or {}
                    desc = work.get("description") or ""
                    if isinstance(desc, dict):
                        desc = desc.get("value") or ""
                    subjects2 = [str(s).strip() for s in (work.get("subjects") or []) if str(s).strip()]
                    out.append(
                        Candidate(
                            source="openlibrary",
                            title=str(work.get("title") or c_title).strip(),
                            authors=c_authors,
                            categories=subjects2,
                            description=str(desc).strip(),
                            url=f"https://openlibrary.org{work_key}",
                            score=score,
                        )
                    )

            out.sort(key=lambda c: c.score, reverse=True)
            return out

        if not title:
            return []

        data = self._get_json(
            "https://openlibrary.org/search.json",
            params={"title": title, "author": author, "limit": 5} if author else {"title": title, "limit": 5},
        ) or {}

        for doc in (data.get("docs") or [])[:5]:
            c_title = str(doc.get("title") or "").strip()
            c_authors = [str(a).strip() for a in (doc.get("author_name") or []) if str(a).strip()]
            work_key = str(doc.get("key") or "").strip()
            url = f"https://openlibrary.org{work_key}" if work_key else ""

            score = _similarity(title, c_title)
            if author and c_authors:
                author_best = max((_similarity(author, a) for a in c_authors), default=0.0)
                score = 0.75 * score + 0.25 * author_best

            categories = [str(s).strip() for s in (doc.get("subject") or []) if str(s).strip()]
            desc = ""
            if work_key.startswith("/works/"):
                work = self._get_json(f"https://openlibrary.org{work_key}.json") or {}
                desc_val = work.get("description") or ""
                if isinstance(desc_val, dict):
                    desc_val = desc_val.get("value") or ""
                desc = str(desc_val).strip()
                categories = [str(s).strip() for s in (work.get("subjects") or categories) if str(s).strip()]

            out.append(
                Candidate(
                    source="openlibrary",
                    title=c_title,
                    authors=c_authors,
                    categories=categories,
                    description=desc,
                    url=url,
                    score=score,
                )
            )

        out.sort(key=lambda c: c.score, reverse=True)
        return out


def _passes_threshold(cand: Candidate | None, has_author: bool) -> bool:
    if cand is None:
        return False
    return cand.score >= (0.78 if has_author else 0.88)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE_PATH_DEFAULT)
    ap.add_argument("--backup", type=Path, default=BACKUP_PATH_DEFAULT)
    ap.add_argument("--low-confidence-out", type=Path, default=LOW_CONF_PATH_DEFAULT)
    ap.add_argument("--write-every", type=int, default=25, help="Write cache every N processed targets")
    ap.add_argument("--max-targets", type=int, default=0, help="0 = all targets")
    ap.add_argument("--delay", type=float, default=0.2)
    args = ap.parse_args()

    cache_path: Path = args.cache
    backup_path: Path = args.backup

    if not cache_path.exists():
        print(f"Cache not found: {cache_path}", file=sys.stderr)
        return 2

    data = _load_ordered_json(cache_path)
    if list(data.keys()) != ["_version", "entries", "_path_index"]:
        print("Refusing to run: cache top-level keys differ from expected", file=sys.stderr)
        return 2

    entries: OrderedDict = data["entries"]

    # Backup before any writes
    shutil.copy2(cache_path, backup_path)

    targets: list[str] = []
    for k, v in entries.items():
        if not isinstance(v, dict):
            continue
        if _is_empty_categories(v.get("categories")) or _is_empty_description(v.get("description")):
            targets.append(k)

    total_entries = len(entries)
    targeted_entries = len(targets)

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    lookup = WebBookLookup(s, delay_s=args.delay)

    enriched_count = 0
    unchanged_count = 0
    failure_count = 0
    added_category_counts: Counter[str] = Counter()
    updated_samples: list[dict[str, Any]] = []

    low_conf: list[dict[str, Any]] = []

    processed = 0
    dirty = False

    for key in sorted(targets):
        if args.max_targets and processed >= args.max_targets:
            break
        meta = entries[key]
        processed += 1

        title = str(meta.get("title") or "").strip()
        author = str(meta.get("author") or "").strip()
        isbn = str(meta.get("isbn") or "").strip()
        file_path = str(meta.get("file_path") or "").strip()

        need_cats = _is_empty_categories(meta.get("categories"))
        need_desc = _is_empty_description(meta.get("description"))

        if _looks_like_not_a_book(title, file_path=file_path):
            low_conf.append(
                {
                    "key": key,
                    "title": title,
                    "author": author,
                    "file_path": file_path,
                    "reason": "looks_like_not_a_book_or_working_file",
                }
            )
            unchanged_count += 1
            continue

        if not title:
            low_conf.append({"key": key, "title": title, "author": author, "file_path": file_path, "reason": "missing_title"})
            unchanged_count += 1
            continue

        google_best = (lookup.google_candidates(title=title, author=author, isbn=isbn) or [None])[0]
        ol_best = (lookup.openlibrary_candidates(title=title, author=author, isbn=isbn) or [None])[0]

        has_author = bool(author and author.lower() not in {"unknown", "άγνωστο"})
        trusted_google = _passes_threshold(google_best, has_author=has_author)
        trusted_ol = _passes_threshold(ol_best, has_author=has_author)

        if not trusted_google and not trusted_ol:
            low_conf.append(
                {
                    "key": key,
                    "title": title,
                    "author": author,
                    "file_path": file_path,
                    "reason": "no_confident_web_match",
                    "google_score": getattr(google_best, "score", None),
                    "openlibrary_score": getattr(ol_best, "score", None),
                }
            )
            unchanged_count += 1
            failure_count += 1
            continue

        before_categories = list(meta.get("categories") or [])
        before_desc = str(meta.get("description") or "")
        used_urls: list[str] = []
        added_categories: list[str] = []

        if need_cats:
            raw_cats: list[str] = []
            if trusted_google and google_best and google_best.categories:
                raw_cats.extend(google_best.categories)
                if google_best.url:
                    used_urls.append(google_best.url)
            if trusted_ol and ol_best and ol_best.categories:
                raw_cats.extend(ol_best.categories)
                if ol_best.url:
                    used_urls.append(ol_best.url)
            normed = _normalize_categories(raw_cats)
            if normed:
                meta["categories"] = normed
                added_categories = [c for c in normed if c not in before_categories]

        if need_desc:
            desc = ""
            if trusted_google and google_best and _is_nonempty_str(google_best.description):
                desc = google_best.description
                if google_best.url:
                    used_urls.append(google_best.url)
            elif trusted_ol and ol_best and _is_nonempty_str(ol_best.description):
                desc = ol_best.description
                if ol_best.url:
                    used_urls.append(ol_best.url)
            if _is_nonempty_str(desc):
                meta["description"] = _truncate_description(desc, 500)

        categories_changed = need_cats and not _is_empty_categories(meta.get("categories"))
        desc_changed = need_desc and not _is_empty_description(meta.get("description"))
        enriched = categories_changed or desc_changed

        if enriched:
            meta["metadata_source"] = _update_metadata_source(str(meta.get("metadata_source") or ""), enriched=True)
            enriched_count += 1
            dirty = True

            for c in added_categories:
                added_category_counts[c] += 1

            updated_samples.append(
                {
                    "key": key,
                    "title": title,
                    "author": author,
                    "added_categories": added_categories,
                    "description_len": len(str(meta.get("description") or "")),
                    "source_urls": list(dict.fromkeys([u for u in used_urls if u]))[:3],
                }
            )
        else:
            # Nothing was added; restore originals defensively
            meta["categories"] = before_categories
            meta["description"] = before_desc
            unchanged_count += 1

        if args.write_every > 0 and dirty and (processed % args.write_every == 0):
            _atomic_write_json(cache_path, data)
            dirty = False

    if dirty:
        _atomic_write_json(cache_path, data)

    # Write low-confidence review list
    low_conf_out = OrderedDict()
    low_conf_out["generated_at_unix"] = int(time.time())
    low_conf_out["remaining_targeted_entries_at_start"] = targeted_entries
    low_conf_out["low_confidence"] = low_conf
    low_conf_path: Path = args.low_confidence_out
    _atomic_write_json(low_conf_path, low_conf_out)

    print("=== Auto-enrichment report ===")
    print(f"Cache: {cache_path}")
    print(f"Backup: {backup_path}")
    print(f"Low-confidence list: {low_conf_path}")
    print(f"Total entries: {total_entries}")
    print(f"Targeted entries (start): {targeted_entries}")
    print(f"Processed targets: {processed}")
    print(f"Successfully enriched: {enriched_count}")
    print(f"Unchanged: {unchanged_count}")
    print(f"Failures (no confident match): {failure_count}")

    top_added = added_category_counts.most_common(15)
    if top_added:
        print("Top categories added:")
        for cat, n in top_added:
            print(f"  - {cat}: {n}")

    print("Sample updated books (up to 20):")
    for s in updated_samples[:20]:
        print(f"- {s['title']} — {s['author']} ({s['key']})")
        print(f"  added_categories={s['added_categories']} desc_len={s['description_len']}")
        print(f"  source_urls={s['source_urls']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
