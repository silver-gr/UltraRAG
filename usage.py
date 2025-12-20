#!/usr/bin/env python3
"""CLI utility to check Voyage AI token usage."""
import argparse
import sys
from pathlib import Path

from token_tracker import get_tracker


def main():
    parser = argparse.ArgumentParser(description="Check Voyage AI token usage")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all usage counters (use with caution!)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--embedding-limit",
        type=int,
        default=200_000_000,
        help="Embedding token limit (default: 200M)"
    )
    parser.add_argument(
        "--rerank-limit",
        type=int,
        default=200_000_000,
        help="Rerank token limit (default: 200M)"
    )

    args = parser.parse_args()

    tracker = get_tracker(
        storage_path=Path("./data/voyage_usage.json"),
        embedding_limit=args.embedding_limit,
        rerank_limit=args.rerank_limit
    )

    if args.reset:
        confirm = input("⚠️  Are you sure you want to reset all usage counters? (yes/no): ")
        if confirm.lower() == "yes":
            tracker.reset_usage(confirm=True)
            print("✅ Usage counters reset.")
        else:
            print("❌ Reset cancelled.")
        return

    if args.json:
        import json
        print(json.dumps(tracker.get_status(), indent=2))
    else:
        tracker.print_status()

        # Print warnings if approaching limits
        status = tracker.get_status()
        if status['embedding']['percent_used'] > 90:
            print("⚠️  WARNING: Approaching embedding token limit!")
        if status['rerank']['percent_used'] > 90:
            print("⚠️  WARNING: Approaching rerank token limit!")

        if status['embedding']['exhausted'] or status['rerank']['exhausted']:
            sys.exit(1)


if __name__ == "__main__":
    main()
