"""Temporal filtering for RAG retrieval based on document dates."""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Literal, Tuple
from dataclasses import dataclass
from dateutil import parser as date_parser

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor

logger = logging.getLogger(__name__)


# Date filter preset types
DateFilterPreset = Literal[
    "all_time",
    "last_7_days",
    "last_30_days",
    "last_90_days",
    "this_year",
    "custom"
]


@dataclass
class DateRange:
    """Represents a date range for filtering."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    @classmethod
    def from_preset(cls, preset: DateFilterPreset) -> "DateRange":
        """Create DateRange from a preset.

        Args:
            preset: One of the predefined date filter presets

        Returns:
            DateRange with appropriate start/end dates
        """
        now = datetime.now()

        if preset == "all_time":
            return cls(start_date=None, end_date=None)

        elif preset == "last_7_days":
            start = now - timedelta(days=7)
            return cls(start_date=start, end_date=now)

        elif preset == "last_30_days":
            start = now - timedelta(days=30)
            return cls(start_date=start, end_date=now)

        elif preset == "last_90_days":
            start = now - timedelta(days=90)
            return cls(start_date=start, end_date=now)

        elif preset == "this_year":
            start = datetime(now.year, 1, 1)
            return cls(start_date=start, end_date=now)

        else:
            # Default to all time for unknown presets
            logger.warning(f"Unknown preset '{preset}', defaulting to all_time")
            return cls(start_date=None, end_date=None)

    @classmethod
    def from_custom(cls, start_date: datetime, end_date: datetime) -> "DateRange":
        """Create DateRange from custom dates.

        Args:
            start_date: Start of the range
            end_date: End of the range

        Returns:
            DateRange with the specified dates
        """
        return cls(start_date=start_date, end_date=end_date)

    def is_active(self) -> bool:
        """Check if this filter is active (not all_time)."""
        return self.start_date is not None or self.end_date is not None

    def contains(self, date: datetime) -> bool:
        """Check if a date falls within this range.

        Args:
            date: The date to check

        Returns:
            True if date is within range, False otherwise
        """
        if not self.is_active():
            return True

        # Normalize to naive datetime for comparison (strip timezone if present)
        cmp_date = date.replace(tzinfo=None) if hasattr(date, 'tzinfo') and date.tzinfo else date

        if self.start_date and cmp_date < self.start_date:
            return False

        if self.end_date and cmp_date > self.end_date:
            return False

        return True


class TemporalFilter(BaseNodePostprocessor):
    """Post-processor that filters nodes based on document dates.

    Uses created_date or modified_date metadata fields to filter
    retrieved nodes to a specific date range.
    """

    # Pydantic field declarations (BaseNodePostprocessor is a Pydantic model)
    date_range: Optional[DateRange] = None
    date_field: str = "modified_date"
    fallback_field: str = "created_date"
    include_undated: bool = True

    class Config:
        """Pydantic config to allow arbitrary types like DateRange."""
        arbitrary_types_allowed = True

    def __init__(
        self,
        date_range: Optional[DateRange] = None,
        date_field: str = "modified_date",
        fallback_field: str = "created_date",
        include_undated: bool = True,
        **kwargs
    ):
        """Initialize temporal filter.

        Args:
            date_range: DateRange to filter by (None = no filtering)
            date_field: Primary metadata field to use for date (default: modified_date)
            fallback_field: Fallback field if primary is empty (default: created_date)
            include_undated: Whether to include documents without dates (default: True)
        """
        super().__init__(
            date_range=date_range,
            date_field=date_field,
            fallback_field=fallback_field,
            include_undated=include_undated,
            **kwargs
        )

        if date_range and date_range.is_active():
            logger.info(
                f"Temporal filter initialized: {date_field} "
                f"from {date_range.start_date} to {date_range.end_date}"
            )

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse a date string to datetime.

        Handles various date formats commonly used in Obsidian frontmatter.

        Args:
            date_str: Date string to parse

        Returns:
            Parsed datetime or None if parsing fails
        """
        if not date_str or date_str.strip() == '':
            return None

        try:
            # Use dateutil for flexible parsing
            return date_parser.parse(date_str)
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse date '{date_str}': {e}")
            return None

    def _get_node_date(self, node: NodeWithScore) -> Optional[datetime]:
        """Extract date from node metadata.

        Args:
            node: The node to extract date from

        Returns:
            Parsed datetime or None
        """
        metadata = node.node.metadata

        # Try primary field first
        date_str = metadata.get(self.date_field, '')
        date = self._parse_date(date_str)

        if date:
            return date

        # Try fallback field
        fallback_str = metadata.get(self.fallback_field, '')
        return self._parse_date(fallback_str)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """Filter nodes based on date range.

        Args:
            nodes: List of nodes to filter
            query_bundle: Query context (not used for temporal filtering)

        Returns:
            Filtered list of nodes
        """
        # If no date range or not active, return all nodes
        if not self.date_range or not self.date_range.is_active():
            return nodes

        filtered_nodes = []
        filtered_count = 0
        undated_count = 0

        for node in nodes:
            node_date = self._get_node_date(node)

            if node_date is None:
                # Handle undated documents
                undated_count += 1
                if self.include_undated:
                    filtered_nodes.append(node)
                else:
                    filtered_count += 1
            elif self.date_range.contains(node_date):
                filtered_nodes.append(node)
            else:
                filtered_count += 1

        logger.info(
            f"Temporal filter: {len(nodes)} nodes -> {len(filtered_nodes)} nodes "
            f"(filtered: {filtered_count}, undated: {undated_count})"
        )

        return filtered_nodes


def create_temporal_filter(
    preset: DateFilterPreset = "all_time",
    custom_start: Optional[datetime] = None,
    custom_end: Optional[datetime] = None,
    date_field: str = "modified_date",
    include_undated: bool = True
) -> Optional[TemporalFilter]:
    """Factory function to create a temporal filter.

    Args:
        preset: Date filter preset to use
        custom_start: Custom start date (only used if preset is "custom")
        custom_end: Custom end date (only used if preset is "custom")
        date_field: Which metadata field to filter on
        include_undated: Whether to include documents without dates

    Returns:
        TemporalFilter instance or None if preset is all_time
    """
    if preset == "all_time":
        return None

    if preset == "custom":
        if custom_start is None or custom_end is None:
            logger.warning("Custom preset requires both start and end dates")
            return None
        date_range = DateRange.from_custom(custom_start, custom_end)
    else:
        date_range = DateRange.from_preset(preset)

    return TemporalFilter(
        date_range=date_range,
        date_field=date_field,
        include_undated=include_undated
    )


def get_preset_label(preset: DateFilterPreset) -> str:
    """Get human-readable label for a preset.

    Args:
        preset: The preset to get label for

    Returns:
        Human-readable label
    """
    labels = {
        "all_time": "All Time",
        "last_7_days": "Last 7 Days",
        "last_30_days": "Last 30 Days",
        "last_90_days": "Last 90 Days",
        "this_year": "This Year",
        "custom": "Custom Range"
    }
    return labels.get(preset, preset)


def get_all_presets() -> List[Tuple[str, DateFilterPreset]]:
    """Get all available presets with labels.

    Returns:
        List of (label, preset) tuples
    """
    presets: List[DateFilterPreset] = [
        "all_time",
        "last_7_days",
        "last_30_days",
        "last_90_days",
        "this_year"
    ]
    return [(get_preset_label(p), p) for p in presets]
