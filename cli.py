"""
UltraRAG CLI - Non-interactive command-line interface.

Enables programmatic access for Claude Code agents and automation.
All output goes to stdout as structured YAML/JSON. Logs go to stderr.

Usage:
    python -m cli research --topic "neuroplasticity" --depth standard
    python -m cli query --query "What is RAG?" --source vault
    python -m cli status
"""

import argparse
import json
import logging
import os
import sys
import time

# Suppress tokenizers parallelism warning before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import load_config, RAGConfig
import indexing
import retrieval
from models import (
    QueryResult, ResearchResult, SourceNode, IndexSource, SourceType,
    IndexNotFoundError, BookFilter, MetadataFilter, _normalize_category,
)

logger = logging.getLogger("ultrarag.cli")


# === OUTPUT FORMATTING ===

def _format_finding(source: SourceNode) -> dict:
    """Convert SourceNode to finding dict matching SuperResearch schema."""
    icons = {
        "vault": "\U0001f4d3",        # 📓
        "conversations": "\U0001f4ac", # 💬
        "conversation": "\U0001f4ac",  # 💬
        "web": "\U0001f310",           # 🌐
        "books": "\U0001f4da",         # 📚
    }

    finding = {
        "source_type": source.source_type,
        "icon": icons.get(source.source_type, "\U0001f4c4"),
        "title": source.file_name or source.metadata.get("title", "Untitled"),
        "relevance": round(source.score, 4) if source.score else 0.0,
        "key_insight": source.excerpt[:300] if source.excerpt else "",
    }

    if source.source_type == "vault":
        finding["path"] = source.file_path
    elif source.source_type in ("conversations", "conversation"):
        finding["path"] = source.file_path
    elif source.source_type == "web":
        finding["url"] = source.metadata.get("url", "")
        finding["publisher"] = source.metadata.get("publisher", "Unknown")
        finding["date"] = source.metadata.get("date", "Unknown")

    return finding


def _format_output(data: dict, fmt: str) -> str:
    """Format output as YAML or JSON."""
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    try:
        import yaml
        return yaml.dump(
            data, default_flow_style=False,
            allow_unicode=True, sort_keys=False,
        )
    except ImportError:
        logger.warning("PyYAML not installed, falling back to JSON")
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _error_output(topic: str, depth: str, error: str, fmt: str) -> str:
    """Format error output."""
    return _format_output({
        "agent": "ultrarag",
        "topic": topic,
        "depth": depth,
        "error": str(error),
        "findings": [],
        "total_sources": 0,
        "quality_notes": f"ERROR: {error}",
    }, fmt)


# === INDEX LOADING ===

def _load_indexes(
    config: RAGConfig,
    sources: list[str],
) -> dict[str, IndexSource]:
    """Load requested indexes into IndexSource dict."""
    source_map = {
        "vault": ("vault", SourceType.VAULT, 1.0),
        "conversations": ("conversations", SourceType.CONVERSATIONS, 0.8),
        "books": ("books", SourceType.BOOKS, 0.7),
    }

    indexes = {}
    for source_name in sources:
        if source_name not in source_map:
            continue

        idx_name, source_type, weight = source_map[source_name]

        if not indexing.index_exists(config, idx_name):
            logger.info("Index '%s' not found, skipping", idx_name)
            continue

        index = indexing.load_index(config, idx_name)
        if index is not None:
            indexes[idx_name] = IndexSource(
                name=idx_name,
                index=index,
                source_type=source_type,
                weight=weight,
            )
            logger.info("Loaded index: %s", idx_name)

    return indexes


class _SuppressStdout:
    """Context manager to redirect stdout to devnull.

    Libraries like token_tracker print status during init.
    We keep stdout clean for structured output only.
    """

    def __enter__(self):
        self._old_stdout = sys.stdout
        self._devnull = open(os.devnull, "w")
        sys.stdout = self._devnull
        return self

    def __exit__(self, *exc):
        sys.stdout = self._old_stdout
        self._devnull.close()
        return False


def _init_settings(config: RAGConfig):
    """Initialize LlamaIndex Settings for query operations."""
    from llama_index.core import Settings
    from embeddings import get_embedding_model

    # Suppress stdout during init — Voyage tracker prints usage stats
    with _SuppressStdout():
        Settings.embed_model = get_embedding_model(config.embedding)


# === COMMANDS ===

def cmd_research(args):
    """Execute research query with specified depth."""
    config = load_config()

    # Determine which indexes to load based on depth
    if args.depth == "quick":
        sources_to_load = ["vault"]
    else:  # standard, deep
        sources_to_load = ["vault", "conversations"]

    # Suppress stdout during init — Voyage tracker prints usage stats
    with _SuppressStdout():
        _init_settings(config)
        indexes = _load_indexes(config, sources_to_load)

    if not indexes:
        print(_error_output(
            args.topic, args.depth,
            "No indexes found. Run 'python main.py' then 'index' to create.",
            args.format,
        ))
        sys.exit(1)

    start_time = time.time()

    try:
        if args.depth == "quick":
            # Vault only, single pass
            vault_index = list(indexes.values())[0].index
            result = retrieval.query(
                args.topic, vault_index, config, mode="hybrid",
            )
        elif args.depth == "standard":
            # Federated (vault + conversations), single pass
            result = retrieval.federated_query(
                args.topic, indexes, config, mode="hybrid",
            )
        else:  # deep
            # Full iterative research
            result = retrieval.research(
                args.topic, indexes, config, exhaustive=False,
            )

        # Build output matching SuperResearch schema
        findings = [_format_finding(s) for s in result.sources]
        exec_time = round(time.time() - start_time, 2)

        output = {
            "agent": "ultrarag",
            "topic": args.topic,
            "depth": args.depth,
            "findings": findings,
            "total_sources": len(findings),
            "exec_time_seconds": exec_time,
        }

        # Add research-specific metadata
        if isinstance(result, ResearchResult):
            output["research_metadata"] = {
                "iterations": result.iterations,
                "confidence": round(result.confidence, 4),
                "is_exhaustive": result.is_exhaustive,
                "total_unique_nodes": result.total_unique_nodes,
            }
            output["quality_notes"] = (
                result.get_gap_analyses_markdown()
                or f"Research completed in {exec_time}s"
            )
        else:
            output["quality_notes"] = f"Query completed in {exec_time}s"

        # Include answer for reference
        if result.answer:
            output["answer"] = result.answer

        print(_format_output(output, args.format))

    except Exception as e:
        logger.error("Research failed: %s", e)
        print(_error_output(args.topic, args.depth, e, args.format))
        sys.exit(1)


def cmd_query(args):
    """Execute a simple query."""
    config = load_config()

    if args.source == "all":
        sources_to_load = ["vault", "conversations", "books"]
    elif args.source == "books":
        sources_to_load = ["books"]
    else:
        sources_to_load = [args.source]

    # Build BookFilter from CLI flags
    book_filter = None
    cat = getattr(args, "category", None)
    auth = getattr(args, "author", None)
    if cat or auth:
        book_filter = BookFilter(
            categories=[_normalize_category(c) for c in cat.split(",")] if cat else None,
            authors=[a.strip() for a in auth.split(",")] if auth else None,
        )

    # Suppress stdout during init — Voyage tracker prints usage stats
    with _SuppressStdout():
        _init_settings(config)
        indexes = _load_indexes(config, sources_to_load)

    if not indexes:
        print(_format_output({
            "error": f"No indexes found for source '{args.source}'",
            "answer": "",
            "sources": [],
            "total_sources": 0,
        }, args.format))
        sys.exit(1)

    # Build MetadataFilter from CLI flags
    metadata_filter = None
    tag_val = getattr(args, "tag", None)
    after_val = getattr(args, "after", None)
    before_val = getattr(args, "before", None)
    prefix_val = getattr(args, "path_prefix", None)
    if any([tag_val, after_val, before_val, prefix_val]):
        metadata_filter = MetadataFilter(
            tags=[t.strip() for t in tag_val.split(",")] if tag_val else None,
            date_after=after_val,
            date_before=before_val,
            path_prefix=prefix_val,
        )

    try:
        if len(indexes) == 1 and not book_filter:
            index = list(indexes.values())[0].index
            result = retrieval.query(
                args.query, index, config, mode=args.mode,
                metadata_filter=metadata_filter,
            )
        else:
            result = retrieval.federated_query(
                args.query, indexes, config, mode=args.mode,
                book_filter=book_filter,
                metadata_filter=metadata_filter,
            )

        output = {
            "query": args.query,
            "source": args.source,
            "mode": args.mode,
            "answer": result.answer,
            "sources": [_format_finding(s) for s in result.sources],
            "total_sources": len(result.sources),
            "exec_time_seconds": round(result.exec_time, 2),
            "tokens_used": result.tokens_used,
        }

        # Include structured confidence / validation if present
        if result.relevance_grade:
            output["relevance_grade"] = result.relevance_grade
        if result.confidence is not None:
            output["confidence"] = result.confidence
        if result.validation_result:
            output["validation_result"] = result.validation_result
        if result.timings:
            output["timings"] = result.timings

        print(_format_output(output, args.format))

    except Exception as e:
        logger.error("Query failed: %s", e)
        print(_format_output({
            "error": str(e),
            "answer": "",
            "sources": [],
            "total_sources": 0,
        }, args.format))
        sys.exit(1)


def cmd_enrich(args):
    """Run book metadata enrichment pipeline."""
    config = load_config()

    if getattr(args, "clear_cache", False):
        cache_path = config.books.metadata_cache_path
        if cache_path and cache_path.exists():
            from calibre_metadata import BookMetadataCache
            cache = BookMetadataCache(cache_path)
            cache.clear()
            cache.save()
            logger.info("Cache cleared")

    try:
        stats = indexing.enrich_books(config)
        print(_format_output({
            "command": "enrich",
            "status": "success",
            **stats,
        }, args.format))
    except Exception as e:
        logger.error("Enrichment failed: %s", e)
        print(_format_output({
            "command": "enrich",
            "status": "error",
            "error": str(e),
        }, args.format))
        sys.exit(1)


def cmd_status(args):
    """Show index status."""
    config = load_config()

    status = {}
    for source in ["vault", "conversations", "books"]:
        status[source] = {"exists": indexing.index_exists(config, source)}

    output = {
        "ultrarag_status": "ready" if status["vault"]["exists"] else "no_index",
        "indexes": status,
        "vault_path": str(config.vault_path) if config.vault_path else None,
    }

    print(_format_output(output, args.format))


# === MAIN ===

def main():
    """CLI entry point."""
    # Shared arguments available on every subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--format", "-f",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format (default: yaml)",
    )
    common.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress/log output on stderr",
    )

    parser = argparse.ArgumentParser(
        prog="ultrarag",
        description="UltraRAG CLI - Non-interactive interface for agents and automation",
        parents=[common],
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- research ---
    rp = subparsers.add_parser(
        "research",
        parents=[common],
        help="Federated research query (SuperResearch compatible)",
    )
    rp.add_argument("--topic", "-t", required=True, help="Research topic/query")
    rp.add_argument(
        "--depth", "-d",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="quick=vault, standard=vault+conversations, deep=iterative research",
    )

    # --- query ---
    qp = subparsers.add_parser("query", parents=[common], help="Simple RAG query")
    qp.add_argument("--query", "-q", required=True, help="Query string")
    qp.add_argument(
        "--source", "-s",
        choices=["vault", "conversations", "books", "all"],
        default="vault",
        help="Source to query (default: vault)",
    )
    qp.add_argument(
        "--mode", "-m",
        choices=["simple", "hybrid"],
        default="hybrid",
        help="Query mode (default: hybrid)",
    )
    qp.add_argument(
        "--category", "-c",
        default=None,
        help="Filter books by category (comma-separated for multiple)",
    )
    qp.add_argument(
        "--author", "-a",
        default=None,
        help="Filter books by author (comma-separated for multiple)",
    )
    qp.add_argument(
        "--tag",
        default=None,
        help="Filter results by tag (comma-separated for multiple)",
    )
    qp.add_argument(
        "--after",
        default=None,
        help="Filter results created after date (ISO: YYYY-MM-DD)",
    )
    qp.add_argument(
        "--before",
        default=None,
        help="Filter results created before date (ISO: YYYY-MM-DD)",
    )
    qp.add_argument(
        "--path-prefix",
        default=None,
        help="Filter results by file path prefix",
    )

    # --- enrich ---
    ep = subparsers.add_parser(
        "enrich",
        parents=[common],
        help="Run book metadata enrichment (Calibre + web)",
    )
    ep.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear existing cache before enrichment",
    )

    # --- status ---
    subparsers.add_parser("status", parents=[common], help="Show index availability")

    args = parser.parse_args()

    # Configure logging — all to stderr, stdout reserved for structured output
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    if args.command == "research":
        cmd_research(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "enrich":
        cmd_enrich(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
