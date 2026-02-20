#!/usr/bin/env python3
"""
Enrich missing book metadata in data/book_metadata_cache.json using web sources.

Fills ONLY missing `categories` and/or `description` fields (never overwrites
existing non-empty values). Updates `metadata_source` only when web enrichment
adds missing fields, per repository rules.

Web sources (no Tavily):
- Google Books public API
- Open Library public API
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

from models import _normalize_category


CACHE_PATH_DEFAULT = Path("data/book_metadata_cache.json")
BACKUP_PATH_DEFAULT = Path("data/book_metadata_cache.codex_backup.json")


USER_AGENT = "UltraRAG-Codex/1.0 (+https://github.com/; metadata enrichment)"


def _is_nonempty_str(val: Any) -> bool:
    return isinstance(val, str) and bool(val.strip())


def _is_empty_description(val: Any) -> bool:
    return not _is_nonempty_str(val)


def _is_empty_categories(val: Any) -> bool:
    if not isinstance(val, list):
        return True
    return len([c for c in val if str(c).strip()]) == 0


_SPACEISH = re.compile(r"[\s/_\-]+")
_NONWORD = re.compile(r"[^a-z0-9 ]+")


def _norm_for_match(s: str) -> str:
    s = s.lower().strip()
    s = _SPACEISH.sub(" ", s)
    s = _NONWORD.sub("", s)
    return " ".join(s.split())


def _similarity(a: str, b: str) -> float:
    a_n = _norm_for_match(a)
    b_n = _norm_for_match(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def _truncate_description(text: str, max_len: int = 500) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    cut = text[: max_len - 1]
    last_period = cut.rfind(". ")
    if last_period >= max_len * 0.6:
        return cut[: last_period + 1].strip()
    return cut.strip()


def _normalize_categories(raw: list[str], max_items: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for c in raw:
        if not c:
            continue
        c = str(c)
        c = re.sub(r"\s+", " ", c.replace("&", " and ").replace("/", " ")).strip()
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


def _normalize_isbn(isbn: str) -> str:
    isbn = re.sub(r"[^0-9xX]", "", isbn or "").strip()
    return isbn.upper()


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
                q_parts.append(f'intitle:"{title}"')
            if author:
                q_parts.append(f'inauthor:"{author}"')

        if not q_parts:
            return []

        q = " ".join(q_parts)
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {
            "q": q,
            "maxResults": 5,
            "printType": "books",
            "projection": "full",
        }
        data = self._get_json(url, params=params) or {}
        items = data.get("items") or []

        out: list[Candidate] = []
        for item in items:
            vi = item.get("volumeInfo") or {}
            c_title = vi.get("title") or ""
            c_authors = vi.get("authors") or []
            c_desc = vi.get("description") or ""
            c_cats = vi.get("categories") or []
            info_url = vi.get("infoLink") or vi.get("previewLink") or item.get("selfLink") or ""

            score = _similarity(title, c_title)
            if author and c_authors:
                author_best = max((_similarity(author, a) for a in c_authors), default=0.0)
                score = 0.75 * score + 0.25 * author_best

            out.append(
                Candidate(
                    source="google_books",
                    title=c_title,
                    authors=[str(a) for a in c_authors if str(a).strip()],
                    categories=[str(c) for c in c_cats if str(c).strip()],
                    description=str(c_desc).strip(),
                    url=str(info_url).strip(),
                    score=score,
                )
            )

        out.sort(key=lambda c: c.score, reverse=True)
        return out

    def openlibrary_candidates(self, title: str, author: str = "", isbn: str = "") -> list[Candidate]:
        isbn = _normalize_isbn(isbn)
        out: list[Candidate] = []

        # Prefer ISBN lookup if available
        if isbn:
            bib_url = "https://openlibrary.org/api/books"
            bib_params = {"bibkeys": f"ISBN:{isbn}", "format": "json", "jscmd": "data"}
            bib = self._get_json(bib_url, params=bib_params) or {}
            record = bib.get(f"ISBN:{isbn}") or {}
            c_title = record.get("title") or ""
            c_authors = [a.get("name", "") for a in (record.get("authors") or [])]
            subjects = [s.get("name", "") for s in (record.get("subjects") or [])]
            url = record.get("url") or (record.get("key") and f"https://openlibrary.org{record['key']}") or ""
            score = _similarity(title, c_title)
            if author and c_authors:
                author_best = max((_similarity(author, a) for a in c_authors), default=0.0)
                score = 0.75 * score + 0.25 * author_best
            out.append(
                Candidate(
                    source="openlibrary",
                    title=c_title,
                    authors=[a for a in c_authors if str(a).strip()],
                    categories=[s for s in subjects if str(s).strip()],
                    description="",
                    url=str(url).strip(),
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
                    subjects2 = work.get("subjects") or []
                    url2 = f"https://openlibrary.org{work_key}"
                    out.append(
                        Candidate(
                            source="openlibrary",
                            title=c_title or work.get("title", "") or "",
                            authors=[a for a in c_authors if str(a).strip()],
                            categories=[str(s) for s in subjects2 if str(s).strip()],
                            description=str(desc).strip(),
                            url=url2,
                            score=score,
                        )
                    )

            out.sort(key=lambda c: c.score, reverse=True)
            return out

        # Search by title/author
        if not title:
            return []

        search_url = "https://openlibrary.org/search.json"
        params = {"title": title, "limit": 5}
        if author:
            params["author"] = author

        data = self._get_json(search_url, params=params) or {}
        docs = data.get("docs") or []
        for doc in docs[:5]:
            c_title = doc.get("title") or ""
            c_authors = doc.get("author_name") or []
            work_key = (doc.get("key") or "").strip()
            url = f"https://openlibrary.org{work_key}" if work_key else ""

            score = _similarity(title, c_title)
            if author and c_authors:
                author_best = max((_similarity(author, a) for a in c_authors), default=0.0)
                score = 0.75 * score + 0.25 * author_best

            categories = doc.get("subject") or []
            desc = ""
            if work_key and work_key.startswith("/works/"):
                work = self._get_json(f"https://openlibrary.org{work_key}.json") or {}
                desc_val = work.get("description") or ""
                if isinstance(desc_val, dict):
                    desc_val = desc_val.get("value") or ""
                desc = str(desc_val).strip()
                categories = work.get("subjects") or categories

            out.append(
                Candidate(
                    source="openlibrary",
                    title=str(c_title).strip(),
                    authors=[str(a) for a in c_authors if str(a).strip()],
                    categories=[str(s) for s in categories if str(s).strip()],
                    description=str(desc).strip(),
                    url=url,
                    score=score,
                )
            )

        out.sort(key=lambda c: c.score, reverse=True)
        return out


def _pick_best(cands: list[Candidate]) -> Candidate | None:
    return cands[0] if cands else None


def _passes_threshold(cand: Candidate | None, has_author: bool) -> bool:
    if cand is None:
        return False
    return cand.score >= (0.78 if has_author else 0.86)


def _update_metadata_source(existing: str, enriched: bool) -> str:
    if not enriched:
        return existing
    if existing == "calibre":
        return "calibre+web"
    if existing == "filename":
        return "web"
    return existing


def _load_ordered_json(path: Path) -> OrderedDict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def _atomic_write_json(path: Path, data: OrderedDict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE_PATH_DEFAULT)
    ap.add_argument("--backup", type=Path, default=BACKUP_PATH_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--max-entries", type=int, default=0, help="0 = no limit (process all targets)")
    ap.add_argument("--delay", type=float, default=0.2, help="Delay between HTTP requests (seconds)")
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
    total_entries = len(entries)

    targets: list[str] = []
    for k, v in entries.items():
        if not isinstance(v, dict):
            continue
        if _is_empty_categories(v.get("categories")) or _is_empty_description(v.get("description")):
            targets.append(k)

    targeted_entries = len(targets)

    shutil.copy2(cache_path, backup_path)

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    lookup = WebBookLookup(s, delay_s=args.delay)

    enriched_count = 0
    unchanged_count = 0
    failure_count = 0

    added_category_counts: Counter[str] = Counter()
    updated_samples: list[dict[str, Any]] = []

    dirty = False
    processed = 0

    for key in sorted(targets):
        if args.max_entries and processed >= args.max_entries:
            break

        meta = entries[key]
        title = str(meta.get("title") or "").strip()
        author = str(meta.get("author") or "").strip()
        isbn = str(meta.get("isbn") or "").strip()

        need_cats = _is_empty_categories(meta.get("categories"))
        need_desc = _is_empty_description(meta.get("description"))

        processed += 1

        if not title:
            unchanged_count += 1
            continue

        google_best = _pick_best(lookup.google_candidates(title=title, author=author, isbn=isbn))
        ol_best = _pick_best(lookup.openlibrary_candidates(title=title, author=author, isbn=isbn))

        has_author = bool(author)
        trusted_google = _passes_threshold(google_best, has_author=has_author)
        trusted_ol = _passes_threshold(ol_best, has_author=has_author)

        if not trusted_google and not trusted_ol:
            unchanged_count += 1
            failure_count += 1
            continue

        before_categories = list(meta.get("categories") or [])
        before_desc = str(meta.get("description") or "")

        added_categories: list[str] = []
        used_urls: list[str] = []

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
                added_categories = normed

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
                if c and c not in before_categories:
                    added_category_counts[c] += 1

            updated_samples.append(
                {
                    "title": title,
                    "author": author,
                    "added_categories": [c for c in added_categories if c],
                    "description_len": len(str(meta.get("description") or "")),
                    "urls": list(dict.fromkeys([u for u in used_urls if u])),
                }
            )
        else:
            # Nothing was added (e.g., only unreliable fields available)
            meta["categories"] = before_categories
            meta["description"] = before_desc
            unchanged_count += 1

        if processed % args.batch_size == 0 and dirty:
            _atomic_write_json(cache_path, data)
            dirty = False

    if dirty:
        _atomic_write_json(cache_path, data)

    top_added = added_category_counts.most_common(20)

    print("=== Book metadata cache enrichment report ===")
    print(f"Cache: {cache_path}")
    print(f"Backup: {backup_path}")
    print(f"Total entries: {total_entries}")
    print(f"Targeted entries: {targeted_entries}")
    print(f"Processed targets: {processed}")
    print(f"Successfully enriched: {enriched_count}")
    print(f"Unchanged: {unchanged_count}")
    print(f"Failures (no confident match): {failure_count}")
    if top_added:
        print("Top categories added:")
        for cat, n in top_added[:15]:
            print(f"  - {cat}: {n}")
    else:
        print("Top categories added: (none)")

    print("Sample updated books (up to 20):")
    for s in updated_samples[:20]:
        t = s.get("title", "")
        a = s.get("author", "")
        cats = s.get("added_categories") or []
        dlen = s.get("description_len", 0)
        urls = s.get("urls") or []
        print(f"- {t} — {a}")
        print(f"  added_categories={cats} desc_len={dlen}")
        print(f"  urls={urls}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

