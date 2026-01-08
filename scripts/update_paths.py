#!/usr/bin/env python3
"""Update file paths in UltraRAG index after vault reorganization.

This script updates file paths in:
1. LanceDB metadata (file_path field)
2. LanceDB _node_content JSON (nested file_path)
3. Checkpoint file (processed_files list)

Usage:
    # From a moves log JSON (array format from Obsidian reorganization):
    python scripts/update_paths.py --moves-json vault-file-moves.json --vault-path "/path/to/vault"

    # From a moves log CSV (old_path,new_path format):
    python scripts/update_paths.py --moves-csv moves_log.csv --vault-path "/path/to/vault"

    # Dry run (show what would be updated without changing anything):
    python scripts/update_paths.py --moves-json moves.json --vault-path "/path/to/vault" --dry-run

Supported JSON formats:
    # Array format (from Obsidian/git move tracking):
    [
        {"old_path": "Resources/YouTube/note.md", "new_path": "+Captures/YouTube/note.md"},
        ...
    ]

    # Dict format (simple key-value):
    {
        "Resources/YouTube/note.md": "+Captures/YouTube/note.md",
        ...
    }

Note: Paths in the moves file can be relative (to vault) or absolute.
      Use --vault-path to specify vault location for relative path resolution.
"""
import sys
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class UpdateStats:
    """Track update statistics."""
    lancedb_rows_updated: int = 0
    lancedb_rows_skipped: int = 0
    checkpoint_paths_updated: int = 0
    checkpoint_paths_not_found: int = 0
    errors: list = field(default_factory=list)


def load_moves_csv(csv_path: Path, vault_path: Optional[Path] = None) -> Dict[str, str]:
    """Load moves mapping from CSV/TSV file.

    Expected format: old_path,new_path or old_path<tab>new_path (no header)
    If vault_path is provided, relative paths are converted to absolute.
    """
    moves = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Auto-detect delimiter (tab or comma)
        first_line = f.readline()
        f.seek(0)
        delimiter = '\t' if '\t' in first_line else ','

        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) >= 2:
                old_path, new_path = row[0].strip(), row[1].strip()
                if old_path and new_path:
                    moves[old_path] = new_path

    # Convert relative paths to absolute if vault_path provided
    if vault_path:
        converted_moves = {}
        for old_path, new_path in moves.items():
            if not old_path.startswith('/'):
                old_path = str(vault_path / old_path)
            if not new_path.startswith('/'):
                new_path = str(vault_path / new_path)
            converted_moves[old_path] = new_path
        moves = converted_moves

    return moves


def load_moves_json(json_path: Path, vault_path: Optional[Path] = None) -> Dict[str, str]:
    """Load moves mapping from JSON file.

    Supports two formats:
    1. Array format: [{"old_path": "...", "new_path": "..."}, ...]
    2. Dict format: {"old_path": "new_path", ...}

    If vault_path is provided, relative paths are converted to absolute.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    moves = {}

    # Detect format and convert to dict
    if isinstance(data, list):
        # Array format: [{"old_path": "...", "new_path": "..."}, ...]
        for item in data:
            if isinstance(item, dict) and 'old_path' in item and 'new_path' in item:
                old_path = item['old_path']
                new_path = item['new_path']
                moves[old_path] = new_path
    elif isinstance(data, dict):
        # Dict format: {"old_path": "new_path", ...}
        moves = data
    else:
        raise ValueError(f"Unsupported JSON format: expected list or dict, got {type(data)}")

    # Convert relative paths to absolute if vault_path provided
    if vault_path:
        vault_str = str(vault_path)
        converted_moves = {}
        for old_path, new_path in moves.items():
            # Check if paths are relative (don't start with /)
            if not old_path.startswith('/'):
                old_path = str(vault_path / old_path)
            if not new_path.startswith('/'):
                new_path = str(vault_path / new_path)
            converted_moves[old_path] = new_path
        moves = converted_moves

    return moves


def update_lancedb_paths(
    db_path: Path,
    table_name: str,
    moves: Dict[str, str],
    dry_run: bool = False
) -> Tuple[int, int]:
    """Update file paths in LanceDB table.

    Updates both:
    - metadata.file_path
    - metadata._node_content.metadata.file_path

    Returns:
        Tuple of (rows_updated, rows_skipped)
    """
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(db_path))

    # Get table names (handle different LanceDB response formats)
    tables_response = db.list_tables()
    if hasattr(tables_response, 'tables'):
        # Newer LanceDB returns response object with .tables attribute
        table_names = tables_response.tables
    else:
        # Older LanceDB returns list directly
        table_names = list(tables_response)

    if table_name not in table_names:
        logger.warning(f"Table {table_name} not found in database (available: {table_names})")
        return 0, 0

    table = db.open_table(table_name)
    df = table.to_pandas()

    rows_updated = 0
    rows_skipped = 0

    # Build reverse lookup for faster matching
    old_paths_set = set(moves.keys())

    for idx, row in df.iterrows():
        metadata = row.get('metadata', {})
        if not metadata:
            rows_skipped += 1
            continue

        file_path = metadata.get('file_path', '')

        # Check if this row needs updating
        if file_path not in old_paths_set:
            rows_skipped += 1
            continue

        new_path = moves[file_path]

        if dry_run:
            logger.info(f"[DRY RUN] Would update: {file_path} -> {new_path}")
            rows_updated += 1
            continue

        # Update file_path in metadata
        metadata['file_path'] = new_path

        # Update file_name if path changed
        new_file_name = Path(new_path).name
        metadata['file_name'] = new_file_name

        # Update _node_content JSON if present
        node_content = metadata.get('_node_content')
        if node_content:
            try:
                node_data = json.loads(node_content)
                if 'metadata' in node_data:
                    node_data['metadata']['file_path'] = new_path
                    node_data['metadata']['file_name'] = new_file_name
                metadata['_node_content'] = json.dumps(node_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse _node_content for row {idx}: {e}")

        # Update the row
        df.at[idx, 'metadata'] = metadata
        rows_updated += 1

    if not dry_run and rows_updated > 0:
        # Overwrite the table with updated data
        # LanceDB doesn't support in-place updates, so we recreate
        logger.info(f"Writing {len(df)} rows back to {table_name}...")

        # Delete and recreate table
        db.drop_table(table_name)
        db.create_table(table_name, df)

        logger.info(f"Table {table_name} updated successfully")

    return rows_updated, rows_skipped


def update_checkpoint_paths(
    checkpoint_path: Path,
    moves: Dict[str, str],
    dry_run: bool = False
) -> Tuple[int, int]:
    """Update file paths in checkpoint JSON.

    Returns:
        Tuple of (paths_updated, paths_not_found)
    """
    if not checkpoint_path.exists():
        logger.info("No checkpoint file found - nothing to update")
        return 0, 0

    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)

    processed_files = checkpoint_data.get('processed_files', [])

    updated_files = []
    paths_updated = 0
    paths_not_found = 0

    for file_path in processed_files:
        if file_path in moves:
            new_path = moves[file_path]
            if dry_run:
                logger.info(f"[DRY RUN] Checkpoint: {file_path} -> {new_path}")
            updated_files.append(new_path)
            paths_updated += 1
        else:
            updated_files.append(file_path)
            # Check if this is an old path that should have been in moves
            # (file exists at old location = not moved)
            if not Path(file_path).exists():
                paths_not_found += 1

    if not dry_run and paths_updated > 0:
        checkpoint_data['processed_files'] = updated_files
        checkpoint_data['total_files'] = len(updated_files)

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint file updated: {paths_updated} paths changed")

    return paths_updated, paths_not_found


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Update file paths in UltraRAG index after vault reorganization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--moves-csv',
        type=Path,
        help='CSV file with old_path,new_path mappings'
    )
    input_group.add_argument(
        '--moves-json',
        type=Path,
        help='JSON file with {old_path: new_path} mappings'
    )

    # Path options
    parser.add_argument(
        '--vault-path',
        type=Path,
        help='Path to Obsidian vault (required for relative paths in moves file)'
    )
    parser.add_argument(
        '--db-path',
        type=Path,
        default=Path('data/lancedb'),
        help='Path to LanceDB directory (default: data/lancedb)'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=Path,
        default=Path('data/index_checkpoint.json'),
        help='Path to checkpoint file (default: data/index_checkpoint.json)'
    )
    parser.add_argument(
        '--table',
        type=str,
        default='obsidian_embeddings',
        help='LanceDB table name (default: obsidian_embeddings)'
    )

    # Operation options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    parser.add_argument(
        '--skip-lancedb',
        action='store_true',
        help='Skip LanceDB update'
    )
    parser.add_argument(
        '--skip-checkpoint',
        action='store_true',
        help='Skip checkpoint file update'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve vault path if provided
    vault_path = None
    if args.vault_path:
        vault_path = args.vault_path.resolve()
        if not vault_path.exists():
            print(f"Warning: Vault path does not exist: {vault_path}")
        else:
            print(f"Using vault path: {vault_path}")

    # Load moves mapping
    if args.moves_csv:
        if not args.moves_csv.exists():
            print(f"Error: CSV file not found: {args.moves_csv}")
            sys.exit(1)
        moves = load_moves_csv(args.moves_csv, vault_path=vault_path)
        logger.info(f"Loaded {len(moves)} path mappings from CSV/TSV")
    else:
        if not args.moves_json.exists():
            print(f"Error: JSON file not found: {args.moves_json}")
            sys.exit(1)
        moves = load_moves_json(args.moves_json, vault_path=vault_path)
        logger.info(f"Loaded {len(moves)} path mappings from JSON")

    if not moves:
        print("No path mappings found in input file")
        sys.exit(1)

    # Show sample mappings
    print(f"\nPath mappings to process: {len(moves)}")
    sample_items = list(moves.items())[:3]
    for old_path, new_path in sample_items:
        print(f"  {Path(old_path).name} -> {Path(new_path).parent.name}/{Path(new_path).name}")
    if len(moves) > 3:
        print(f"  ... and {len(moves) - 3} more")

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")

    stats = UpdateStats()

    # Update LanceDB
    if not args.skip_lancedb:
        print(f"\nUpdating LanceDB table: {args.table}")
        if not args.db_path.exists():
            print(f"Warning: LanceDB path not found: {args.db_path}")
        else:
            try:
                updated, skipped = update_lancedb_paths(
                    args.db_path,
                    args.table,
                    moves,
                    dry_run=args.dry_run
                )
                stats.lancedb_rows_updated = updated
                stats.lancedb_rows_skipped = skipped
                print(f"  Rows updated: {updated}")
                print(f"  Rows skipped (no match): {skipped}")
            except Exception as e:
                logger.error(f"LanceDB update failed: {e}")
                stats.errors.append(f"LanceDB: {e}")

    # Update checkpoint
    if not args.skip_checkpoint:
        print(f"\nUpdating checkpoint: {args.checkpoint_path}")
        try:
            updated, not_found = update_checkpoint_paths(
                args.checkpoint_path,
                moves,
                dry_run=args.dry_run
            )
            stats.checkpoint_paths_updated = updated
            stats.checkpoint_paths_not_found = not_found
            print(f"  Paths updated: {updated}")
            if not_found > 0:
                print(f"  Paths not in moves (may be stale): {not_found}")
        except Exception as e:
            logger.error(f"Checkpoint update failed: {e}")
            stats.errors.append(f"Checkpoint: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"LanceDB rows updated: {stats.lancedb_rows_updated}")
    print(f"Checkpoint paths updated: {stats.checkpoint_paths_updated}")

    if stats.errors:
        print(f"\nErrors encountered: {len(stats.errors)}")
        for error in stats.errors:
            print(f"  - {error}")

    if args.dry_run:
        print("\n[DRY RUN COMPLETE - No changes were made]")
        print("Remove --dry-run flag to apply changes")
    else:
        print("\nPath updates complete!")
        print("Note: Wikilink graph will be automatically rebuilt on next load")


if __name__ == '__main__':
    main()
