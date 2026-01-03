#!/usr/bin/env python3
"""Score Obsidian notes by value metrics and output top N notes.

Scoring criteria:
- Content length (word count)
- Wikilink count (connectedness as source)
- Backlink count (referenced by others)
- Tag count (intentional organization)
- Frontmatter richness
- Recency (modification time)
"""
import os
import re
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# Patterns
WIKILINK_PATTERN = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
TAG_PATTERN = re.compile(r'(?:^|\s)#([a-zA-Z][a-zA-Z0-9_/-]*)', re.MULTILINE)
FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)


def extract_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter and return (metadata, body)."""
    match = FRONTMATTER_PATTERN.match(content)
    if match:
        try:
            metadata = yaml.safe_load(match.group(1)) or {}
            body = content[match.end():]
            return metadata, body
        except yaml.YAMLError:
            pass
    return {}, content


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def extract_wikilinks(content: str) -> List[str]:
    """Extract wikilink targets from content."""
    return WIKILINK_PATTERN.findall(content)


def extract_tags(content: str) -> List[str]:
    """Extract tags from content."""
    return TAG_PATTERN.findall(content)


def score_note(
    file_path: Path,
    content: str,
    backlink_count: int,
    now: datetime,
    max_words: int,
    max_wikilinks: int,
    max_backlinks: int,
    max_tags: int,
    max_frontmatter_keys: int
) -> Dict[str, Any]:
    """Score a single note and return detailed metrics."""
    metadata, body = extract_frontmatter(content)

    # Extract metrics
    word_count = count_words(body)
    wikilinks = extract_wikilinks(body)
    tags = extract_tags(body)
    frontmatter_keys = len(metadata.keys())

    # Get modification time
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
    days_since_modified = (now - mtime).days

    # Normalized scores (0-1 scale)
    # Content score: log scale to not overly favor very long notes
    import math
    content_score = min(1.0, math.log1p(word_count) / math.log1p(max_words)) if max_words > 0 else 0

    # Connectivity scores
    wikilink_score = min(1.0, len(wikilinks) / max(1, max_wikilinks))
    backlink_score = min(1.0, backlink_count / max(1, max_backlinks))
    tag_score = min(1.0, len(tags) / max(1, max_tags))
    frontmatter_score = min(1.0, frontmatter_keys / max(1, max_frontmatter_keys))

    # Recency score: exponential decay (half-life of 90 days)
    recency_score = 0.5 ** (days_since_modified / 90)

    # Weighted composite score
    # Higher weights for connectivity (valuable for RAPTOR clustering)
    weights = {
        'content': 0.20,      # Length matters but not too much
        'wikilinks': 0.25,    # Being a hub is valuable
        'backlinks': 0.25,    # Being referenced is very valuable
        'tags': 0.10,         # Tags show organization
        'frontmatter': 0.10,  # Rich metadata is good
        'recency': 0.10       # Recent use matters
    }

    total_score = (
        weights['content'] * content_score +
        weights['wikilinks'] * wikilink_score +
        weights['backlinks'] * backlink_score +
        weights['tags'] * tag_score +
        weights['frontmatter'] * frontmatter_score +
        weights['recency'] * recency_score
    )

    return {
        'path': str(file_path),
        'title': file_path.stem,
        'score': total_score,
        'metrics': {
            'word_count': word_count,
            'wikilink_count': len(wikilinks),
            'backlink_count': backlink_count,
            'tag_count': len(tags),
            'frontmatter_keys': frontmatter_keys,
            'days_since_modified': days_since_modified
        },
        'subscores': {
            'content': content_score,
            'wikilinks': wikilink_score,
            'backlinks': backlink_score,
            'tags': tag_score,
            'frontmatter': frontmatter_score,
            'recency': recency_score
        }
    }


def scan_vault(vault_path: Path, top_n: int = 150) -> List[Dict[str, Any]]:
    """Scan vault and return top N notes by value score."""
    print(f"Scanning vault: {vault_path}", file=sys.stderr)

    # First pass: collect all notes and build backlink index
    notes = {}  # path -> content
    backlinks = defaultdict(int)  # note name -> count of notes linking to it

    md_files = list(vault_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files", file=sys.stderr)

    for file_path in md_files:
        # Skip hidden files and common exclusions
        if any(part.startswith('.') for part in file_path.parts):
            continue
        if 'templates' in file_path.parts or 'Templates' in file_path.parts:
            continue

        try:
            content = file_path.read_text(encoding='utf-8')
            notes[file_path] = content

            # Count outgoing links for backlink index
            for target in extract_wikilinks(content):
                # Normalize target (handle paths and aliases)
                target_name = target.split('/')[-1].split('#')[0].strip()
                backlinks[target_name] += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)

    print(f"Loaded {len(notes)} notes, building scores...", file=sys.stderr)

    # Calculate max values for normalization
    max_words = 0
    max_wikilinks = 0
    max_tags = 0
    max_frontmatter = 0

    for content in notes.values():
        metadata, body = extract_frontmatter(content)
        max_words = max(max_words, count_words(body))
        max_wikilinks = max(max_wikilinks, len(extract_wikilinks(body)))
        max_tags = max(max_tags, len(extract_tags(body)))
        max_frontmatter = max(max_frontmatter, len(metadata.keys()))

    max_backlinks = max(backlinks.values()) if backlinks else 1

    print(f"Max metrics - words: {max_words}, wikilinks: {max_wikilinks}, "
          f"backlinks: {max_backlinks}, tags: {max_tags}", file=sys.stderr)

    # Score all notes
    now = datetime.now()
    scored_notes = []

    for file_path, content in notes.items():
        note_name = file_path.stem
        backlink_count = backlinks.get(note_name, 0)

        scored = score_note(
            file_path, content, backlink_count, now,
            max_words, max_wikilinks, max_backlinks, max_tags, max_frontmatter
        )
        scored_notes.append(scored)

    # Sort by score descending
    scored_notes.sort(key=lambda x: x['score'], reverse=True)

    # Return top N
    top_notes = scored_notes[:top_n]

    print(f"\nTop {len(top_notes)} notes by value score:", file=sys.stderr)
    for i, note in enumerate(top_notes[:10], 1):
        print(f"  {i}. {note['title'][:50]:50} score={note['score']:.3f} "
              f"words={note['metrics']['word_count']} "
              f"links={note['metrics']['wikilink_count']} "
              f"backlinks={note['metrics']['backlink_count']}", file=sys.stderr)
    if len(top_notes) > 10:
        print(f"  ... and {len(top_notes) - 10} more", file=sys.stderr)

    return top_notes


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Score Obsidian notes by value')
    parser.add_argument('vault_path', type=Path, help='Path to Obsidian vault')
    parser.add_argument('-n', '--top', type=int, default=150, help='Number of top notes to return')
    parser.add_argument('-o', '--output', type=Path, help='Output JSON file (default: stdout)')
    parser.add_argument('--paths-only', action='store_true', help='Output only file paths, one per line')

    args = parser.parse_args()

    if not args.vault_path.exists():
        print(f"Error: Vault path does not exist: {args.vault_path}", file=sys.stderr)
        sys.exit(1)

    top_notes = scan_vault(args.vault_path, args.top)

    if args.paths_only:
        output = '\n'.join(note['path'] for note in top_notes)
    else:
        output = json.dumps(top_notes, indent=2)

    if args.output:
        args.output.write_text(output)
        print(f"\nWritten to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
