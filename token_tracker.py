"""Token usage tracking for Voyage AI with quota limits."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage statistics for a single model."""
    model: str
    tokens_used: int = 0
    token_limit: int = 200_000_000  # 200M default
    requests_count: int = 0
    last_updated: str = ""

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.token_limit - self.tokens_used)

    @property
    def usage_percent(self) -> float:
        if self.token_limit == 0:
            return 100.0
        return (self.tokens_used / self.token_limit) * 100

    @property
    def is_exhausted(self) -> bool:
        return self.tokens_used >= self.token_limit


@dataclass
class VoyageTokenTracker:
    """Tracks Voyage AI token usage across embedding and reranking.

    Persists usage to disk so limits are maintained across sessions.
    Thread-safe for concurrent access.
    """
    storage_path: Path = field(default_factory=lambda: Path("./data/voyage_usage.json"))
    embedding_limit: int = 200_000_000  # 200M tokens
    rerank_limit: int = 200_000_000     # 200M tokens
    warn_threshold: float = 0.9         # Warn at 90% usage

    embedding_usage: TokenUsage = field(default=None)
    rerank_usage: TokenUsage = field(default=None)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        """Initialize usage tracking from disk or defaults."""
        self.storage_path = Path(self.storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing usage or create new
        self._load_usage()

    def _load_usage(self) -> None:
        """Load usage from disk if exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)

                self.embedding_usage = TokenUsage(
                    model=data.get('embedding', {}).get('model', 'voyage-3.5-lite'),
                    tokens_used=data.get('embedding', {}).get('tokens_used', 0),
                    token_limit=self.embedding_limit,
                    requests_count=data.get('embedding', {}).get('requests_count', 0),
                    last_updated=data.get('embedding', {}).get('last_updated', '')
                )

                self.rerank_usage = TokenUsage(
                    model=data.get('rerank', {}).get('model', 'voyage-rerank-2.5'),
                    tokens_used=data.get('rerank', {}).get('tokens_used', 0),
                    token_limit=self.rerank_limit,
                    requests_count=data.get('rerank', {}).get('requests_count', 0),
                    last_updated=data.get('rerank', {}).get('last_updated', '')
                )

                logger.info(
                    f"Loaded Voyage usage: Embeddings {self.embedding_usage.tokens_used:,}/{self.embedding_limit:,} "
                    f"({self.embedding_usage.usage_percent:.1f}%), "
                    f"Rerank {self.rerank_usage.tokens_used:,}/{self.rerank_limit:,} "
                    f"({self.rerank_usage.usage_percent:.1f}%)"
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load usage data, starting fresh: {e}")
                self._init_fresh()
        else:
            self._init_fresh()

    def _init_fresh(self) -> None:
        """Initialize fresh usage tracking."""
        self.embedding_usage = TokenUsage(
            model='voyage-3.5-lite',
            token_limit=self.embedding_limit
        )
        self.rerank_usage = TokenUsage(
            model='voyage-rerank-2.5',
            token_limit=self.rerank_limit
        )
        logger.info("Initialized fresh Voyage token tracking")

    def _save_usage(self) -> None:
        """Persist usage to disk."""
        data = {
            'embedding': asdict(self.embedding_usage),
            'rerank': asdict(self.rerank_usage),
            'saved_at': datetime.now().isoformat()
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def estimate_tokens(self, texts: list[str]) -> int:
        """Estimate token count for a list of texts.

        Uses approximation: 1 token ≈ 4 characters (conservative for English).
        Voyage charges by actual tokens, so this is an estimate.
        """
        total_chars = sum(len(t) for t in texts)
        # Conservative estimate: 1 token per 3.5 chars (accounts for non-ASCII)
        return int(total_chars / 3.5)

    def check_embedding_quota(self, estimated_tokens: int) -> tuple[bool, str]:
        """Check if embedding quota allows the operation.

        Returns:
            (can_proceed, message)
        """
        with self._lock:
            if self.embedding_usage.is_exhausted:
                return False, (
                    f"❌ Voyage embedding quota exhausted! "
                    f"Used {self.embedding_usage.tokens_used:,}/{self.embedding_limit:,} tokens. "
                    f"Please upgrade or wait for quota reset."
                )

            projected = self.embedding_usage.tokens_used + estimated_tokens
            if projected > self.embedding_limit:
                return False, (
                    f"❌ Operation would exceed embedding quota! "
                    f"Current: {self.embedding_usage.tokens_used:,}, "
                    f"Estimated: {estimated_tokens:,}, "
                    f"Limit: {self.embedding_limit:,}"
                )

            # Warning at threshold
            if projected / self.embedding_limit > self.warn_threshold:
                remaining = self.embedding_limit - projected
                logger.warning(
                    f"⚠️ Approaching embedding quota: {remaining:,} tokens remaining "
                    f"({100 - (projected/self.embedding_limit)*100:.1f}% left)"
                )

            return True, "OK"

    def check_rerank_quota(self, estimated_tokens: int) -> tuple[bool, str]:
        """Check if rerank quota allows the operation."""
        with self._lock:
            if self.rerank_usage.is_exhausted:
                return False, (
                    f"❌ Voyage rerank quota exhausted! "
                    f"Used {self.rerank_usage.tokens_used:,}/{self.rerank_limit:,} tokens."
                )

            projected = self.rerank_usage.tokens_used + estimated_tokens
            if projected > self.rerank_limit:
                return False, (
                    f"❌ Operation would exceed rerank quota! "
                    f"Current: {self.rerank_usage.tokens_used:,}, "
                    f"Estimated: {estimated_tokens:,}"
                )

            if projected / self.rerank_limit > self.warn_threshold:
                remaining = self.rerank_limit - projected
                logger.warning(
                    f"⚠️ Approaching rerank quota: {remaining:,} tokens remaining"
                )

            return True, "OK"

    def record_embedding_usage(self, tokens: int, model: str = None) -> None:
        """Record embedding token usage."""
        with self._lock:
            self.embedding_usage.tokens_used += tokens
            self.embedding_usage.requests_count += 1
            self.embedding_usage.last_updated = datetime.now().isoformat()
            if model:
                self.embedding_usage.model = model
            self._save_usage()

            logger.debug(
                f"Recorded {tokens:,} embedding tokens. "
                f"Total: {self.embedding_usage.tokens_used:,}/{self.embedding_limit:,} "
                f"({self.embedding_usage.usage_percent:.2f}%)"
            )

    def record_rerank_usage(self, tokens: int, model: str = None) -> None:
        """Record rerank token usage."""
        with self._lock:
            self.rerank_usage.tokens_used += tokens
            self.rerank_usage.requests_count += 1
            self.rerank_usage.last_updated = datetime.now().isoformat()
            if model:
                self.rerank_usage.model = model
            self._save_usage()

            logger.debug(
                f"Recorded {tokens:,} rerank tokens. "
                f"Total: {self.rerank_usage.tokens_used:,}/{self.rerank_limit:,}"
            )

    def get_status(self) -> dict:
        """Get current usage status."""
        with self._lock:
            return {
                'embedding': {
                    'model': self.embedding_usage.model,
                    'used': self.embedding_usage.tokens_used,
                    'limit': self.embedding_limit,
                    'remaining': self.embedding_usage.tokens_remaining,
                    'percent_used': self.embedding_usage.usage_percent,
                    'requests': self.embedding_usage.requests_count,
                    'exhausted': self.embedding_usage.is_exhausted
                },
                'rerank': {
                    'model': self.rerank_usage.model,
                    'used': self.rerank_usage.tokens_used,
                    'limit': self.rerank_limit,
                    'remaining': self.rerank_usage.tokens_remaining,
                    'percent_used': self.rerank_usage.usage_percent,
                    'requests': self.rerank_usage.requests_count,
                    'exhausted': self.rerank_usage.is_exhausted
                }
            }

    def print_status(self) -> None:
        """Print formatted usage status."""
        status = self.get_status()

        print("\n" + "="*60)
        print("📊 VOYAGE AI TOKEN USAGE")
        print("="*60)

        for service, data in status.items():
            icon = "🔴" if data['exhausted'] else "🟡" if data['percent_used'] > 80 else "🟢"
            print(f"\n{icon} {service.upper()} ({data['model']})")
            print(f"   Used:      {data['used']:>15,} tokens")
            print(f"   Remaining: {data['remaining']:>15,} tokens")
            print(f"   Limit:     {data['limit']:>15,} tokens")
            print(f"   Usage:     {data['percent_used']:>14.2f}%")
            print(f"   Requests:  {data['requests']:>15,}")

        print("\n" + "="*60 + "\n")

    def reset_usage(self, confirm: bool = False) -> None:
        """Reset all usage tracking (use with caution)."""
        if not confirm:
            logger.warning("Reset requires confirm=True")
            return

        with self._lock:
            self._init_fresh()
            self._save_usage()
            logger.info("Token usage tracking reset")


# Global tracker instance (singleton pattern)
_tracker: Optional[VoyageTokenTracker] = None


def get_tracker(
    storage_path: Path = None,
    embedding_limit: int = 200_000_000,
    rerank_limit: int = 200_000_000
) -> VoyageTokenTracker:
    """Get or create the global token tracker."""
    global _tracker

    if _tracker is None:
        _tracker = VoyageTokenTracker(
            storage_path=storage_path or Path("./data/voyage_usage.json"),
            embedding_limit=embedding_limit,
            rerank_limit=rerank_limit
        )

    return _tracker


class QuotaExhaustedError(Exception):
    """Raised when Voyage AI quota is exhausted."""
    pass
