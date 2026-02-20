#!/usr/bin/env python3
"""
Apply enrichment updates to UltraRAG's book metadata cache safely.

Rules enforced:
- Preserve top-level keys: _version, entries, _path_index (no schema changes)
- Only update entries[*].categories, entries[*].description, entries[*].metadata_source
- Only fill missing categories/description (never overwrite non-empty)
- Normalize categories via models._normalize_category
- Description max length: 500 chars
- metadata_source updates:
  - calibre -> calibre+web only if web adds missing fields
  - filename -> web only if web adds missing fields
  - web / calibre+web unchanged

Input formats supported:
- Root object is mapping: {\"<entry_key>\": {...}, ...}
- Root object is wrapper: {\"updates\": {\"<entry_key>\": {...}, ...}, ...}

Additional rules:
- Respects payload `confidence` (skips if < 0.70)
- Ignores `source_urls` for storage (used only for reporting)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import _normalize_category  # noqa: E402


CONFIDENCE_MIN = 0.70


def _is_nonempty_str(val: Any) -> bool:
    return isinstance(val, str) and bool(val.strip())


def _is_empty_description(val: Any) -> bool:
    return not _is_nonempty_str(val)


def _is_empty_categories(val: Any) -> bool:
    if not isinstance(val, list):
        return True
    return len([c for c in val if str(c).strip()]) == 0


def _truncate_description(text: str, max_len: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].strip()


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


def _load_ordered_json(path: Path) -> OrderedDict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def _atomic_write_json(path: Path, data: OrderedDict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _clean_url(s: Any) -> str:
    s = str(s or "").strip()
    # Common markdown link form: [text](url)
    m = re.search(r"\((https?://[^)\s]+)\)", s)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(https?://\S+)", s)
    if m2:
        return m2.group(1).rstrip(").,;]}>\\\"'")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/book_metadata_cache.json"))
    ap.add_argument("--backup", type=Path, default=Path("data/book_metadata_cache.codex_backup.json"))
    ap.add_argument("--updates", type=Path, required=True, help="JSON mapping entry_key -> update payload")
    ap.add_argument("--batch-size", type=int, default=25, help="Write cache every N applied updates")
    args = ap.parse_args()

    cache_path: Path = args.cache
    backup_path: Path = args.backup
    updates_path: Path = args.updates

    if not cache_path.exists():
        print(f"Cache not found: {cache_path}", file=sys.stderr)
        return 2
    if not updates_path.exists():
        print(f"Updates not found: {updates_path}", file=sys.stderr)
        return 2

    data = _load_ordered_json(cache_path)
    if list(data.keys()) != ["_version", "entries", "_path_index"]:
        print("Refusing to apply: cache top-level keys differ from expected", file=sys.stderr)
        return 2

    entries: OrderedDict = data["entries"]

    updates_root = _load_ordered_json(updates_path)
    updates: dict[str, Any] | None = None

    # Supported:
    # - {"updates": {...}} wrapper
    # - {"key": {...}} mapping
    # - [{"key": "...", ...}, ...] list items
    if isinstance(updates_root, dict):
        updates = updates_root.get("updates") if isinstance(updates_root.get("updates"), dict) else updates_root
    elif isinstance(updates_root, list):
        updates = {}
        for item in updates_root:
            if not isinstance(item, dict):
                continue
            k = str(item.get("key") or "").strip()
            if not k:
                continue
            payload = {
                "categories": item.get("categories"),
                "description": item.get("description"),
                "source_urls": item.get("source_urls") or item.get("sources") or [],
                "confidence": item.get("confidence"),
            }
            updates[k] = payload

    if not isinstance(updates, dict):
        print("Updates must be a mapping/wrapper or a list of {key,...} items", file=sys.stderr)
        return 2

    shutil.copy2(cache_path, backup_path)

    total_entries = len(entries)
    targeted_entries = 0
    for v in entries.values():
        if isinstance(v, dict) and (_is_empty_categories(v.get("categories")) or _is_empty_description(v.get("description"))):
            targeted_entries += 1

    applied = 0
    unchanged = 0
    failures = 0
    top_added: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    dirty = False

    for key, payload in updates.items():
        if key not in entries or not isinstance(entries[key], dict):
            failures += 1
            continue
        if not isinstance(payload, dict):
            failures += 1
            continue

        conf = payload.get("confidence", 1.0)
        try:
            if isinstance(conf, str):
                c = conf.strip().lower()
                if c in {"high", "h"}:
                    conf_f = 0.90
                elif c in {"medium", "med", "m"}:
                    conf_f = 0.75
                elif c in {"low", "l"}:
                    conf_f = 0.50
                else:
                    conf_f = float(c)
            else:
                conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        if conf_f < CONFIDENCE_MIN:
            unchanged += 1
            continue

        meta = entries[key]
        need_cats = _is_empty_categories(meta.get("categories"))
        need_desc = _is_empty_description(meta.get("description"))

        new_cats = payload.get("categories")
        new_desc = payload.get("description")

        before_categories = list(meta.get("categories") or [])
        enriched = False
        added_categories: list[str] = []

        if need_cats and isinstance(new_cats, list):
            normed = _normalize_categories([str(c) for c in new_cats if str(c).strip()])
            if normed:
                meta["categories"] = normed
                enriched = True
                added_categories = [c for c in normed if c not in before_categories]
                for c in added_categories:
                    top_added[c] = top_added.get(c, 0) + 1

        if need_desc and _is_nonempty_str(new_desc):
            meta["description"] = _truncate_description(str(new_desc), 500)
            enriched = True

        if enriched:
            meta["metadata_source"] = _update_metadata_source(str(meta.get("metadata_source") or ""), enriched=True)
            applied += 1
            dirty = True

            urls = payload.get("source_urls") or []
            urls_clean: list[str] = []
            if isinstance(urls, list):
                for u in urls:
                    cu = _clean_url(u)
                    if cu and cu not in urls_clean:
                        urls_clean.append(cu)

            samples.append(
                {
                    "title": str(meta.get("title") or ""),
                    "author": str(meta.get("author") or ""),
                    "added_categories": added_categories,
                    "desc_len": len(str(meta.get("description") or "")),
                    "source_urls": urls_clean[:3],
                }
            )
        else:
            unchanged += 1

        if args.batch_size > 0 and dirty and (applied % args.batch_size == 0):
            _atomic_write_json(cache_path, data)
            dirty = False

    if dirty:
        _atomic_write_json(cache_path, data)

    top_added_sorted = sorted(top_added.items(), key=lambda kv: (-kv[1], kv[0]))[:20]

    print("=== Book metadata cache enrichment report ===")
    print(f"Cache: {cache_path}")
    print(f"Backup: {backup_path}")
    print(f"Total entries: {total_entries}")
    print(f"Targeted entries (missing categories or description): {targeted_entries}")
    print(f"Successfully enriched (this batch): {applied}")
    print(f"Unchanged (this batch): {unchanged}")
    print(f"Failures (this batch): {failures}")
    if top_added_sorted:
        print("Top categories added (this batch):")
        for cat, n in top_added_sorted[:15]:
            print(f"  - {cat}: {n}")
    else:
        print("Top categories added (this batch): (none)")

    print("Sample updated books (up to 20):")
    for s in samples[:20]:
        print(f"- {s['title']} — {s['author']}")
        print(f"  added_categories={s['added_categories']} desc_len={s['desc_len']}")
        print(f"  source_urls={s['source_urls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
