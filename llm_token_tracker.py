"""LLM token usage and cost tracking for Gemini models.

Tracks token usage per day with cost calculations based on current Gemini pricing.
Similar to token_tracker.py for Voyage, but focused on LLM calls.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# Gemini pricing per 1M tokens (as of Jan 2026)
# Source: https://ai.google.dev/gemini-api/docs/pricing
GEMINI_PRICING = {
    "gemini-3-flash-preview": {
        "input": 0.50,   # $0.50 per 1M input tokens
        "output": 3.00,  # $3.00 per 1M output tokens
    },
    "gemini-3-pro-preview": {
        "input": 2.00,   # $2.00 per 1M input tokens (<=200k context)
        "output": 12.00, # $12.00 per 1M output tokens
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 2.50,
    },
    "gemini-2.5-flash-preview-09-2025": {
        "input": 0.30,
        "output": 2.50,
    },
    "gemini-flash-latest": {  # Used for gap analysis
        "input": 0.30,
        "output": 2.50,
    },
    # Default fallback for unknown models
    "default": {
        "input": 0.50,
        "output": 3.00,
    }
}


@dataclass
class DailyUsage:
    """Token usage for a single day."""
    date: str
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


@dataclass
class LLMTokenTracker:
    """Tracks LLM token usage and costs per day.

    Persists usage to disk so tracking is maintained across sessions.
    Thread-safe for concurrent access.
    """
    storage_path: Path = field(default_factory=lambda: Path("./data/llm_usage.json"))
    daily_usage: Dict[str, DailyUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        """Initialize usage tracking from disk or defaults."""
        self.storage_path = Path(self.storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_usage()

    def _load_usage(self) -> None:
        """Load usage from disk if exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)

                for date_str, day_data in data.get('daily', {}).items():
                    self.daily_usage[date_str] = DailyUsage(
                        date=date_str,
                        input_tokens=day_data.get('input_tokens', 0),
                        output_tokens=day_data.get('output_tokens', 0),
                        requests=day_data.get('requests', 0),
                        input_cost=day_data.get('input_cost', 0.0),
                        output_cost=day_data.get('output_cost', 0.0),
                    )

                logger.info(f"Loaded LLM usage data: {len(self.daily_usage)} days tracked")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load LLM usage data, starting fresh: {e}")
                self.daily_usage = {}
        else:
            logger.info("No existing LLM usage data, starting fresh")

    def _save_usage(self) -> None:
        """Persist usage to disk."""
        data = {
            'daily': {
                date_str: asdict(usage)
                for date_str, usage in self.daily_usage.items()
            },
            'saved_at': datetime.now().isoformat()
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for a model, with fallback to default."""
        # Try exact match
        if model in GEMINI_PRICING:
            return GEMINI_PRICING[model]

        # Try partial match (for versioned models)
        for key in GEMINI_PRICING:
            if key in model or model in key:
                return GEMINI_PRICING[key]

        logger.debug(f"Using default pricing for unknown model: {model}")
        return GEMINI_PRICING["default"]

    def _calculate_cost(self, tokens: int, rate_per_million: float) -> float:
        """Calculate cost for token count at given rate."""
        return (tokens / 1_000_000) * rate_per_million

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gemini-3-flash-preview"
    ) -> None:
        """Record token usage for an LLM call.

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            model: Model name for pricing lookup
        """
        with self._lock:
            today = date.today().isoformat()

            # Get or create daily usage
            if today not in self.daily_usage:
                self.daily_usage[today] = DailyUsage(date=today)

            usage = self.daily_usage[today]
            pricing = self._get_pricing(model)

            # Update tokens
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens
            usage.requests += 1

            # Update costs
            usage.input_cost += self._calculate_cost(input_tokens, pricing["input"])
            usage.output_cost += self._calculate_cost(output_tokens, pricing["output"])

            self._save_usage()

            logger.debug(
                f"Recorded LLM usage: {input_tokens:,} in / {output_tokens:,} out "
                f"(${usage.total_cost:.4f} today)"
            )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses approximation: ~4 characters per token for English text.
        This is a fallback when actual token counts aren't available.
        """
        return max(1, len(text) // 4)

    def get_today_usage(self) -> Optional[DailyUsage]:
        """Get today's usage stats."""
        with self._lock:
            today = date.today().isoformat()
            return self.daily_usage.get(today)

    def get_all_daily_usage(self) -> list[DailyUsage]:
        """Get all daily usage stats sorted by date (newest first)."""
        with self._lock:
            return sorted(
                self.daily_usage.values(),
                key=lambda x: x.date,
                reverse=True
            )

    def get_total_stats(self) -> Dict[str, Any]:
        """Get aggregate stats across all days."""
        with self._lock:
            total_input = sum(u.input_tokens for u in self.daily_usage.values())
            total_output = sum(u.output_tokens for u in self.daily_usage.values())
            total_requests = sum(u.requests for u in self.daily_usage.values())
            total_input_cost = sum(u.input_cost for u in self.daily_usage.values())
            total_output_cost = sum(u.output_cost for u in self.daily_usage.values())

            return {
                'total_input_tokens': total_input,
                'total_output_tokens': total_output,
                'total_tokens': total_input + total_output,
                'total_requests': total_requests,
                'total_input_cost': total_input_cost,
                'total_output_cost': total_output_cost,
                'total_cost': total_input_cost + total_output_cost,
                'days_tracked': len(self.daily_usage),
            }

    def get_usage_table(self, limit: int = 30) -> list[Dict[str, Any]]:
        """Get usage data formatted for table display.

        Args:
            limit: Maximum number of days to return

        Returns:
            List of dicts with formatted usage data
        """
        daily = self.get_all_daily_usage()[:limit]
        return [
            {
                'Date': usage.date,
                'Input Tokens': f"{usage.input_tokens:,}",
                'Output Tokens': f"{usage.output_tokens:,}",
                'Total Tokens': f"{usage.total_tokens:,}",
                'Requests': usage.requests,
                'Input Cost': f"${usage.input_cost:.4f}",
                'Output Cost': f"${usage.output_cost:.4f}",
                'Total Cost': f"${usage.total_cost:.4f}",
            }
            for usage in daily
        ]

    def print_status(self) -> None:
        """Print formatted usage status."""
        stats = self.get_total_stats()
        today = self.get_today_usage()

        print("\n" + "=" * 70)
        print("LLM TOKEN USAGE (Gemini)")
        print("=" * 70)

        if today:
            print(f"\nToday ({today.date}):")
            print(f"  Input tokens:  {today.input_tokens:>12,}")
            print(f"  Output tokens: {today.output_tokens:>12,}")
            print(f"  Requests:      {today.requests:>12,}")
            print(f"  Cost:          ${today.total_cost:>11.4f}")

        print(f"\nAll-time totals ({stats['days_tracked']} days):")
        print(f"  Input tokens:  {stats['total_input_tokens']:>12,}")
        print(f"  Output tokens: {stats['total_output_tokens']:>12,}")
        print(f"  Total tokens:  {stats['total_tokens']:>12,}")
        print(f"  Requests:      {stats['total_requests']:>12,}")
        print(f"  Total cost:    ${stats['total_cost']:>11.4f}")

        print("\n" + "=" * 70 + "\n")

    def reset_usage(self, confirm: bool = False) -> None:
        """Reset all usage tracking (use with caution)."""
        if not confirm:
            logger.warning("Reset requires confirm=True")
            return

        with self._lock:
            self.daily_usage = {}
            self._save_usage()
            logger.info("LLM token usage tracking reset")


# Global tracker instance (singleton pattern)
_tracker: Optional[LLMTokenTracker] = None


def get_llm_tracker(storage_path: Path = None) -> LLMTokenTracker:
    """Get or create the global LLM token tracker."""
    global _tracker

    if _tracker is None:
        _tracker = LLMTokenTracker(
            storage_path=storage_path or Path("./data/llm_usage.json")
        )

    return _tracker


def record_llm_usage(
    input_tokens: int,
    output_tokens: int,
    model: str = "gemini-3-flash-preview"
) -> None:
    """Convenience function to record LLM usage."""
    tracker = get_llm_tracker()
    tracker.record_usage(input_tokens, output_tokens, model)
