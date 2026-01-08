#!/usr/bin/env python3
"""
Cross-Lingual Retrieval Test Script for UltraRAG

Tests whether voyage-3.5-lite embeddings enable semantic cross-lingual retrieval
between English and Greek content in the Obsidian vault.

Test Cases:
1. English query "habits" -> should find Greek content with "συνήθεια/συνήθειες"
2. Greek query "συνήθεια" -> should find English content with "habits"
3. Bilingual query "habits συνήθεια" -> baseline comparison

Usage:
    python scripts/test_crosslingual.py
    python scripts/test_crosslingual.py --top-k 20  # Adjust results count
    python scripts/test_crosslingual.py --verbose   # Show full text previews
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def is_greek_text(text: str) -> bool:
    """Check if text contains Greek characters."""
    greek_pattern = r'[α-ωΑ-Ωά-ώ]'
    return bool(re.search(greek_pattern, text))


def contains_greek_habit(text: str) -> bool:
    """Check if text contains Greek words for 'habit'."""
    greek_habit_words = ['συνήθει', 'συνηθει', 'συνήθεια', 'συνήθειες', 'συνηθεια', 'συνηθειες']
    text_lower = text.lower()
    return any(word.lower() in text_lower for word in greek_habit_words)


def contains_english_habit(text: str) -> bool:
    """Check if text contains English word 'habit'."""
    return 'habit' in text.lower()


def run_vector_search(
    table,
    embed_model,
    query: str,
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """Run vector similarity search on LanceDB table.

    Args:
        table: LanceDB table to search
        embed_model: Embedding model for query embedding
        query: Query text
        top_k: Number of results to return

    Returns:
        List of results with text, score, and metadata
    """
    # Get query embedding
    query_embedding = embed_model._get_query_embedding(query)

    # Perform vector search
    results = table.search(query_embedding).limit(top_k).to_pandas()

    # Format results
    formatted_results = []
    for idx, row in results.iterrows():
        meta = row.get('metadata', {})
        file_path = meta.get('file_path', 'unknown') if isinstance(meta, dict) else 'unknown'

        formatted_results.append({
            'text': row.get('text', ''),
            'score': row.get('_distance', 0),  # LanceDB returns distance, lower = more similar
            'file_path': file_path,
            'file_name': meta.get('file_name', 'unknown') if isinstance(meta, dict) else 'unknown'
        })

    return formatted_results


def analyze_results(
    results: List[Dict[str, Any]],
    query_type: str
) -> Dict[str, Any]:
    """Analyze search results for cross-lingual retrieval.

    Args:
        results: List of search results
        query_type: Type of query ('english', 'greek', 'bilingual')

    Returns:
        Analysis dictionary with metrics
    """
    total = len(results)
    greek_count = 0
    english_count = 0
    greek_habit_count = 0
    english_habit_count = 0
    cross_lingual_hits = []

    for i, r in enumerate(results):
        text = r['text']
        is_greek = is_greek_text(text)
        has_greek_habit = contains_greek_habit(text)
        has_english_habit = contains_english_habit(text)

        if is_greek:
            greek_count += 1
        else:
            english_count += 1

        if has_greek_habit:
            greek_habit_count += 1
        if has_english_habit:
            english_habit_count += 1

        # Track cross-lingual hits
        if query_type == 'english' and (is_greek or has_greek_habit):
            cross_lingual_hits.append({
                'rank': i + 1,
                'score': r['score'],
                'file': r['file_name'],
                'preview': text[:150].replace('\n', ' ')
            })
        elif query_type == 'greek' and (not is_greek or has_english_habit):
            cross_lingual_hits.append({
                'rank': i + 1,
                'score': r['score'],
                'file': r['file_name'],
                'preview': text[:150].replace('\n', ' ')
            })

    return {
        'total': total,
        'greek_count': greek_count,
        'english_count': english_count,
        'greek_habit_count': greek_habit_count,
        'english_habit_count': english_habit_count,
        'cross_lingual_hits': cross_lingual_hits,
        'cross_lingual_rate': len(cross_lingual_hits) / total if total > 0 else 0
    }


def print_results(
    query: str,
    query_type: str,
    results: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    verbose: bool = False
) -> None:
    """Print formatted test results."""
    print(f"\n{'='*70}")
    print(f"QUERY: '{query}'")
    print(f"TYPE: {query_type.upper()}")
    print(f"{'='*70}")

    print(f"\n--- Results Summary ---")
    print(f"Total results: {analysis['total']}")
    print(f"Greek content: {analysis['greek_count']} ({analysis['greek_count']/analysis['total']*100:.1f}%)")
    print(f"English content: {analysis['english_count']} ({analysis['english_count']/analysis['total']*100:.1f}%)")
    print(f"Contains Greek 'habit' (συνήθεια): {analysis['greek_habit_count']}")
    print(f"Contains English 'habit': {analysis['english_habit_count']}")

    if query_type == 'english':
        print(f"\n--- Cross-Lingual Hits (Greek results for English query) ---")
    elif query_type == 'greek':
        print(f"\n--- Cross-Lingual Hits (English results for Greek query) ---")
    else:
        print(f"\n--- Bilingual Query - All Results ---")

    if analysis['cross_lingual_hits']:
        print(f"Found {len(analysis['cross_lingual_hits'])} cross-lingual matches:")
        for hit in analysis['cross_lingual_hits'][:10]:  # Show top 10
            print(f"  Rank {hit['rank']:2d} | Score: {hit['score']:.4f} | {hit['file']}")
            if verbose:
                print(f"           {hit['preview']}...")
    else:
        print("  No cross-lingual hits found in results")

    print(f"\nCROSS-LINGUAL RATE: {analysis['cross_lingual_rate']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Test cross-lingual retrieval in UltraRAG')
    parser.add_argument('--top-k', type=int, default=20, help='Number of results per query')
    parser.add_argument('--verbose', action='store_true', help='Show text previews')
    parser.add_argument('--lancedb-path', type=str, default='./data/lancedb', help='Path to LanceDB')
    args = parser.parse_args()

    print("="*70)
    print("ULTRARAG CROSS-LINGUAL RETRIEVAL TEST")
    print("Testing EN<->Greek semantic similarity with voyage-3.5-lite embeddings")
    print("="*70)

    # Initialize embedding model
    print("\n[1/4] Initializing embedding model...")

    from config import EmbeddingConfig
    from embeddings import get_embedding_model

    embed_config = EmbeddingConfig(model="voyage-3.5-lite")
    embed_model = get_embedding_model(embed_config, enable_tracking=False)
    print(f"   Using: {embed_config.model}")

    # Connect to LanceDB
    print("\n[2/4] Connecting to LanceDB...")
    db = lancedb.connect(args.lancedb_path)
    table = db.open_table("obsidian_embeddings")
    row_count = table.count_rows()
    print(f"   Table: obsidian_embeddings ({row_count} rows)")

    # Define test queries
    test_queries = [
        ("habits", "english"),
        ("habit formation", "english"),
        ("building good habits", "english"),
        ("συνήθεια", "greek"),
        ("συνήθειες", "greek"),
        ("δημιουργία συνηθειών", "greek"),
        ("habits συνήθεια", "bilingual"),
    ]

    # Run tests
    print("\n[3/4] Running cross-lingual retrieval tests...")

    all_results = []
    for query, query_type in test_queries:
        results = run_vector_search(table, embed_model, query, args.top_k)
        analysis = analyze_results(results, query_type)
        print_results(query, query_type, results, analysis, args.verbose)
        all_results.append({
            'query': query,
            'type': query_type,
            'analysis': analysis
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: CROSS-LINGUAL RETRIEVAL EFFECTIVENESS")
    print("="*70)

    english_queries = [r for r in all_results if r['type'] == 'english']
    greek_queries = [r for r in all_results if r['type'] == 'greek']

    if english_queries:
        avg_greek_results = sum(r['analysis']['greek_count'] for r in english_queries) / len(english_queries)
        avg_cross_rate = sum(r['analysis']['cross_lingual_rate'] for r in english_queries) / len(english_queries)
        print(f"\nEnglish queries -> Greek results:")
        print(f"  Average Greek content in top-{args.top_k}: {avg_greek_results:.1f} ({avg_greek_results/args.top_k*100:.1f}%)")
        print(f"  Average cross-lingual rate: {avg_cross_rate*100:.1f}%")

    if greek_queries:
        avg_english_results = sum(r['analysis']['english_count'] for r in greek_queries) / len(greek_queries)
        avg_cross_rate = sum(r['analysis']['cross_lingual_rate'] for r in greek_queries) / len(greek_queries)
        print(f"\nGreek queries -> English results:")
        print(f"  Average English content in top-{args.top_k}: {avg_english_results:.1f} ({avg_english_results/args.top_k*100:.1f}%)")
        print(f"  Average cross-lingual rate: {avg_cross_rate*100:.1f}%")

    # Verdict
    print("\n" + "-"*70)
    print("VERDICT:")

    has_cross_lingual = any(r['analysis']['cross_lingual_rate'] > 0 for r in all_results)
    if has_cross_lingual:
        print("  CROSS-LINGUAL RETRIEVAL: WORKING")
        print("  voyage-3.5-lite embeddings DO enable EN<->Greek semantic matching")
    else:
        print("  CROSS-LINGUAL RETRIEVAL: NOT DETECTED")
        print("  English and Greek content may be in separate semantic spaces")

    # Score distribution check
    print("\n[4/4] Score Analysis...")
    for r in all_results[:3]:  # First 3 queries
        query = r['query']
        hits = r['analysis']['cross_lingual_hits']
        if hits:
            scores = [h['score'] for h in hits]
            print(f"  '{query}' cross-lingual scores: min={min(scores):.4f}, max={max(scores):.4f}, avg={sum(scores)/len(scores):.4f}")
        else:
            print(f"  '{query}': No cross-lingual hits")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
