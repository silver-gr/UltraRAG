"""Obsession Radar -- Cross-Source Topic Convergence Detection.

Scans 5 knowledge sources (Vault, AI Conversations, TheSource, Audio Reflections,
Published Articles) for topic convergence. Topics appearing across multiple sources
are "obsessions" -- the best candidates for content creation because they carry
genuine depth, authentic interest, and multi-angle knowledge.

Usage:
    python -m obsession_radar scan "dopamine" "sleep" --format yaml
    python -m obsession_radar discover --top 10 --format markdown
    python -m obsession_radar scan "meditation" --pipeline --new-only
    python -m obsession_radar mine --top 20 --format markdown
    python -m obsession_radar mine --skip-scan --top 50 --format json
    python -m obsession_radar mine --content-only --since 90d --top 15
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("obsession_radar")

# ---------------------------------------------------------------------------
# LanceDB source paths (mirrors content_research_engine.py)
# ---------------------------------------------------------------------------
ULTRARAG_LANCEDB_PATH = Path(__file__).parent / "data" / "lancedb"
SHARED_LANCEDB_PATH = Path.home() / "Projects" / "shared-data" / "lancedb"

TABLE_NAMES = {
    "vault": "obsidian_embeddings",
    "conversations": "conversations",
    "saved_items": "saved_items",
}

# Default paths to file-based sources
DEFAULT_AUDIO_DIR = Path.home() / "Projects" / "AI~udio-Reflections" / "03-transcripts"
DEFAULT_PIPELINE_DIR = Path.home() / "Projects" / "x3lixi-content-pipeline"

# ---------------------------------------------------------------------------
# Source weights for scoring
# ---------------------------------------------------------------------------
SOURCE_WEIGHTS = {
    "vault": 1.0,
    "conversations": 0.8,
    "saved_items": 0.6,
    "audio": 0.7,
    "published": -0.3,
}

# Relevance thresholds per vector source
SOURCE_THRESHOLDS = {
    "vault": 0.55,
    "conversations": 0.50,
    "saved_items": 0.50,
}

# ---------------------------------------------------------------------------
# 10 canonical x3lixi categories with seed topics
# ---------------------------------------------------------------------------
CATEGORY_SEED_TOPICS: dict[str, list[str]] = {
    "relationships": [
        "communication skills", "conflict resolution", "boundaries",
        "attachment styles", "trust building", "vulnerability",
        "social skills", "friendship maintenance", "family dynamics",
        "romantic relationships", "codependency", "emotional intimacy",
    ],
    "health-fitness": [
        "exercise routines", "nutrition basics", "strength training",
        "cardiovascular health", "flexibility mobility", "body composition",
        "supplements evidence", "injury prevention", "sports performance",
        "physical therapy", "walking benefits", "cold exposure",
    ],
    "finances-wealth": [
        "budgeting basics", "investing fundamentals", "financial independence",
        "debt management", "passive income", "tax planning",
        "cryptocurrency", "real estate investing", "emergency fund",
        "retirement planning", "side hustles", "financial literacy",
    ],
    "emotional-wellbeing": [
        "anxiety management", "depression coping", "emotional regulation",
        "self-compassion", "grief processing", "anger management",
        "therapy approaches", "trauma healing", "resilience building",
        "loneliness", "self-esteem", "perfectionism",
    ],
    "mindfulness-stress": [
        "meditation practice", "stress management", "mindfulness techniques",
        "breathing exercises", "yoga benefits", "journaling",
        "digital detox", "gratitude practice", "present moment awareness",
        "burnout prevention", "relaxation techniques", "nervous system regulation",
    ],
    "rest-recovery": [
        "sleep quality", "sleep hygiene", "napping science",
        "circadian rhythm", "recovery protocols", "rest days",
        "power of rest", "sleep disorders", "melatonin",
        "dream science", "polyphasic sleep", "sleep tracking",
    ],
    "productivity-work": [
        "time management", "deep work", "focus techniques",
        "habit building", "procrastination", "goal setting",
        "productivity systems", "task prioritization", "flow state",
        "remote work", "energy management", "decision fatigue",
    ],
    "creativity-leisure": [
        "creative process", "artistic expression", "music creation",
        "writing practice", "hobbies benefits", "play importance",
        "creative blocks", "inspiration sources", "creative routines",
        "digital art", "storytelling", "improvisation",
    ],
    "learning-growth": [
        "learning techniques", "memory improvement", "speed reading",
        "critical thinking", "cognitive biases", "neuroplasticity",
        "skill acquisition", "deliberate practice", "growth mindset",
        "spaced repetition", "note-taking systems", "mental models",
    ],
    "purpose-spirituality": [
        "finding purpose", "meaning of life", "values clarification",
        "spiritual practices", "existential questions", "stoicism",
        "philosophy daily life", "legacy thinking", "ikigai",
        "self-actualization", "moral philosophy", "death awareness",
    ],
}

# Keyword mapping for category suggestion
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "relationships": [
        "relationship", "communication", "conflict", "boundaries", "attachment",
        "trust", "social", "friendship", "family", "romantic", "codependency",
        "intimacy", "partner", "marriage", "dating", "love",
        # Greek
        "σχέση", "σχέσεις", "επικοινωνία", "οικογένεια", "φιλία",
    ],
    "health-fitness": [
        "exercise", "nutrition", "strength", "cardiovascular", "fitness",
        "body", "supplement", "injury", "sports", "physical", "walking",
        "cold exposure", "muscle", "workout", "diet", "weight", "health",
        "dopamine", "neuroscience", "brain", "biology", "pharmacology",
        "harm reduction",
        # Greek
        "υγεία", "νευροεπιστήμη", "βιολογία", "άσκηση", "διατροφή",
        "φαρμακολογία", "σωματική", "ντοπαμίνη", "εγκέφαλος",
        "μείωση βλάβης", "μείωση-βλάβης",
    ],
    "finances-wealth": [
        "budget", "investing", "financial", "debt", "income", "tax",
        "crypto", "real estate", "retirement", "money", "wealth", "saving",
        # Greek
        "οικονομικά", "χρήματα", "επένδυση", "πλούτος",
    ],
    "emotional-wellbeing": [
        "anxiety", "depression", "emotional", "compassion", "grief",
        "anger", "therapy", "trauma", "resilience", "loneliness",
        "self-esteem", "perfectionism", "mental health", "wellbeing",
        "adhd", "adhd management", "psychology",
        # Greek
        "ψυχολογία", "ψυχική υγεία", "ψυχική-υγεία", "άγχος",
        "κατάθλιψη", "θεραπεία", "adhd διαχείριση", "διαχείριση adhd",
        "ψυχική", "συναισθηματικ",
    ],
    "mindfulness-stress": [
        "meditation", "stress", "mindfulness", "breathing", "yoga",
        "journaling", "digital detox", "gratitude", "present moment",
        "burnout", "relaxation", "nervous system",
        # Greek
        "διαλογισμός", "στρες", "ενσυνειδητ", "αναπνοή", "ευγνωμοσύνη",
    ],
    "rest-recovery": [
        "sleep", "rest", "nap", "circadian", "recovery", "melatonin",
        "dream", "insomnia",
        # Greek
        "ύπνος", "ξεκούραση", "ανάπαυση", "αποκατάσταση",
    ],
    "productivity-work": [
        "time management", "deep work", "focus", "habit", "procrastination",
        "goal setting", "productivity", "task", "flow state", "remote work",
        "energy management", "decision", "programming", "software",
        # Greek
        "παραγωγικότητα", "προγραμματισμός", "ανάπτυξη λογισμικού",
        "ανάπτυξη-λογισμικού", "συγκέντρωση", "εστίαση", "συνήθεια",
    ],
    "creativity-leisure": [
        "creative", "artistic", "music", "writing", "hobby", "play",
        "inspiration", "art", "storytelling", "improvisation",
        # Greek
        "δημιουργικ", "μουσική", "τέχνη", "γραφή",
    ],
    "learning-growth": [
        "learning", "memory", "reading", "critical thinking", "cognitive",
        "neuroplasticity", "skill", "deliberate practice", "growth mindset",
        "spaced repetition", "note-taking", "mental model",
        "personal development", "self-improvement",
        # Greek
        "μάθηση", "ανάπτυξη", "προσωπική ανάπτυξη", "προσωπική-ανάπτυξη",
        "μηχανική μάθηση", "γνωστικ", "νευροπλαστικότητα",
    ],
    "purpose-spirituality": [
        "purpose", "meaning", "values", "spiritual", "existential",
        "stoicism", "philosophy", "legacy", "ikigai", "self-actualization",
        "moral", "death",
        # Greek
        "σκοπός", "νόημα", "αξίες", "πνευματικ", "στωικ", "φιλοσοφία",
    ],
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SourceProbe:
    """Result from probing a single knowledge source for a topic."""
    source: str                    # "vault", "conversations", "saved_items", "audio", "published"
    found: bool
    match_count: int
    avg_relevance: float
    top_excerpts: list[str]        # Top 3 snippets
    top_titles: list[str]


@dataclass
class MinedTopic:
    """A topic extracted from saved_items with mining scores."""
    topic: str                    # Normalized canonical name
    raw_variants: list[str]       # Original forms found
    item_count: int               # Unique items with this topic
    mining_score: float           # Composite score
    frequency_score: float
    recency_weight: float
    quality_factor: float
    source_diversity: int         # Unique source platforms
    content_types: dict[str, int] # Breakdown by content_type
    category_suggestion: str      # Via existing _suggest_category()


@dataclass
class TopicScore:
    """Scored topic with convergence data across all sources."""
    topic: str
    convergence: int               # Number of sources with matches (0-5)
    total_score: float
    probes: dict[str, SourceProbe]
    category_suggestion: str
    already_published: bool
    existing_briefs: int
    mining_data: Optional[MinedTopic] = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

_BILINGUAL_MAP: dict[str, list[str]] = {
    "sleep": ["ύπνος", "ύπνο", "ύπνου", "κοιμ", "αϋπν"],
    "habit": ["συνήθεια", "συνήθειες", "συνηθει"],
    "meditation": ["διαλογισμός", "διαλογισμ"],
    "exercise": ["άσκηση", "γυμναστικ"],
    "anxiety": ["άγχος", "αγχ"],
    "depression": ["κατάθλιψη", "καταθλιψ"],
    "stress": ["στρες", "αγχ"],
    "nutrition": ["διατροφή", "διατροφ"],
    "dopamine": ["ντοπαμίνη", "ντοπαμιν"],
    "morning": ["πρωινό", "πρωιν"],
    "routine": ["ρουτίνα", "ρουτιν"],
    "stoicism": ["στωικισμός", "στωικ"],
    "focus": ["εστίαση", "συγκέντρωση"],
    "productivity": ["παραγωγικότητα", "παραγωγικ"],
    "creativity": ["δημιουργικότητα", "δημιουργικ"],
    "resilience": ["ανθεκτικότητα", "ανθεκτικ"],
    "mindfulness": ["ενσυνειδητότητα"],
    "gratitude": ["ευγνωμοσύνη"],
    "journaling": ["ημερολόγιο"],
    "cold exposure": ["κρύο", "κρύου"],
    "fasting": ["νηστεία", "νηστει"],
    "breathing": ["αναπνοή", "αναπνο"],
    "reading": ["διάβασμα", "διαβάζ"],
    "walking": ["περπάτημα", "περπατ"],
    "rest": ["ξεκούραση", "ανάπαυση"],
    "recovery": ["αποκατάσταση"],
    "learning": ["μάθηση"],
    "growth": ["ανάπτυξη"],
    "purpose": ["σκοπός"],
    "spirituality": ["πνευματικότητα"],
    "relationship": ["σχέση", "σχέσεις"],
    "communication": ["επικοινωνία"],
    "finance": ["οικονομικά", "χρήματα"],
    "wealth": ["πλούτος"],
    "ADHD": ["ΔΕΠΥ"],
}

# Build reverse map: Greek -> English
_REVERSE_BILINGUAL: dict[str, str] = {}
for _eng, _greek_variants in _BILINGUAL_MAP.items():
    for _gv in _greek_variants:
        _REVERSE_BILINGUAL[_gv.lower()] = _eng.lower()


def _topic_matches(topic: str, text: str) -> bool:
    """Check if a topic is mentioned in text.

    Uses case-insensitive matching with three strategies:
    1. Direct phrase match
    2. Multi-word: all significant words appear
    3. Bilingual matching (English <-> Greek term mapping)
    """
    topic_lower = topic.lower()
    text_lower = text.lower()

    # 1. Direct phrase match
    if topic_lower in text_lower:
        return True

    # 2. Multi-word: all significant words appear
    words = topic_lower.split()
    if len(words) > 1:
        if all(w in text_lower for w in words if len(w) > 2):
            return True

    # 3. Bilingual: English topic -> Greek variants in text
    for eng_term, greek_variants in _BILINGUAL_MAP.items():
        if eng_term.lower() in topic_lower or topic_lower in eng_term.lower():
            for gv in greek_variants:
                if gv.lower() in text_lower:
                    return True

    # 4. Bilingual: Greek topic -> English in text
    for greek_fragment, eng_term in _REVERSE_BILINGUAL.items():
        if greek_fragment in topic_lower:
            if eng_term in text_lower:
                return True

    return False


def _parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from markdown content.

    Returns empty dict if no frontmatter found.
    Simple parser that handles common patterns without requiring PyYAML.
    """
    if not content.startswith("---"):
        return {}

    # Find the closing ---
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return {}

    fm_text = content[3:end_idx].strip()
    if not fm_text:
        return {}

    result = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value:
                result[key] = value

    return result


def _suggest_category(topic: str) -> str:
    """Suggest a content pipeline category for a topic.

    Uses keyword matching against category-associated terms.
    Returns the best matching category slug, or empty string if no match.
    """
    topic_lower = topic.lower()
    best_cat = ""
    best_score = 0

    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in topic_lower:
                # Longer keyword matches are more valuable
                score += len(kw)
        if score > best_score:
            best_score = score
            best_cat = cat

    return best_cat


# ---------------------------------------------------------------------------
# Topic normalization and mining utilities
# ---------------------------------------------------------------------------

_TOPIC_VARIANT_MAP: dict[str, str] = {
    "personal dev": "personal development",
    "personal-dev": "personal development",
    "ai infrastructure": "ai/llm infrastructure",
    "ai infra": "ai/llm infrastructure",
    "llm infrastructure": "ai/llm infrastructure",
    "machine learning": "ai/machine learning",
    "ml": "ai/machine learning",
    "deep learning": "ai/machine learning",
    "self improvement": "self-improvement",
    "self help": "self-improvement",
    "selfimprovement": "self-improvement",
    "mental-health": "mental health",
    "mentalhealth": "mental health",
    "web dev": "web development",
    "web-dev": "web development",
    "webdev": "web development",
    "dev tools": "developer tools",
    "devtools": "developer tools",
    "dev-tools": "developer tools",
    "open source": "open-source",
    "opensource": "open-source",
    "oss": "open-source",
    "adhd": "adhd",
    "a.d.h.d": "adhd",
    "attention deficit": "adhd",
}


def _normalize_topic(topic: str) -> str:
    """Normalize a topic string to canonical form.

    Steps: lowercase, hyphens→spaces, collapse whitespace, variant map lookup.
    """
    t = topic.lower().strip()
    t = t.replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    return _TOPIC_VARIANT_MAP.get(t, t)


def _recency_weight(saved_at: datetime, now: datetime, half_life_days: float = 90.0) -> float:
    """Compute exponential decay weight for a timestamp.

    Returns a value between 0 and 1. Items saved today → ~1.0,
    items saved half_life_days ago → ~0.5.
    """
    days_ago = max((now - saved_at).total_seconds() / 86400.0, 0.0)
    return math.exp(-0.693 * days_ago / half_life_days)


def _parse_since(since_str: str) -> datetime:
    """Parse a --since time window string into a datetime cutoff.

    Supports: 30d, 90d, 6m, 1y (days, months, years).
    """
    since_str = since_str.strip().lower()
    now = datetime.now(timezone.utc)

    match = re.match(r"^(\d+)(d|m|y)$", since_str)
    if not match:
        raise ValueError(f"Invalid --since format: '{since_str}'. Use e.g. 30d, 6m, 1y")

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "d":
        return now - timedelta(days=value)
    elif unit == "m":
        return now - timedelta(days=value * 30)
    elif unit == "y":
        return now - timedelta(days=value * 365)
    else:
        raise ValueError(f"Unknown time unit: {unit}")


# Content-relevant categories for --content-only filtering
_CONTENT_CATEGORIES = {
    "relationships", "health-fitness", "finances-wealth",
    "emotional-wellbeing", "mindfulness-stress", "rest-recovery",
    "productivity-work", "creativity-leisure", "learning-growth",
    "purpose-spirituality",
}


def _compute_score(probes: dict[str, SourceProbe]) -> float:
    """Compute the weighted convergence score for a set of probes.

    Formula:
        score = sum(weight * (avg_relevance + 0.3 * min(match_count/10, 1.0)))
                for each source where found=True

    Published sources have negative weight to reduce priority for already-covered topics.
    """
    score = 0.0
    for source_name, probe in probes.items():
        if not probe.found:
            continue
        weight = SOURCE_WEIGHTS.get(source_name, 0.0)
        depth_bonus = 0.3 * min(probe.match_count / 10.0, 1.0)
        score += weight * (probe.avg_relevance + depth_bonus)

    return round(score, 4)


# ---------------------------------------------------------------------------
# LanceDB search helper (thin wrapper around content_research_engine)
# ---------------------------------------------------------------------------

def _search_source(
    source: str,
    query_vector: list[float],
    weight: float,
    top_k: int,
    threshold: float,
) -> list:
    """Search a single LanceDB source table.

    Delegates to content_research_engine._search_source for the actual query.
    Imported here so it can be easily mocked in tests.
    """
    from content_research_engine import _search_source as _cre_search
    return _cre_search(source, query_vector, weight, top_k, threshold)


# ---------------------------------------------------------------------------
# Mining: Load saved_items metadata from LanceDB
# ---------------------------------------------------------------------------

def _load_saved_items_metadata() -> list[dict]:
    """Load saved_items from LanceDB, returning metadata only (skip vectors).

    Deduplicates by item_id, merging topics across chunks of the same item.
    Returns list of dicts with keys: item_id, topics, tags, saved_at,
    usefulness_score, content_type, source.
    """
    import lancedb

    db_path = str(SHARED_LANCEDB_PATH)
    table_name = TABLE_NAMES["saved_items"]

    try:
        db = lancedb.connect(db_path)
        table = db.open_table(table_name)
    except Exception as e:
        logger.error("Failed to open saved_items LanceDB table: %s", e)
        return []

    # Load only metadata columns (skip vector column for performance)
    metadata_cols = [
        "item_id", "topics", "tags", "saved_at",
        "usefulness_score", "content_type", "source",
    ]

    try:
        arrow_table = table.to_arrow()
        available_cols = [c for c in metadata_cols if c in arrow_table.column_names]
        if not available_cols:
            logger.error("No metadata columns found in saved_items table")
            return []
        df = arrow_table.select(available_cols).to_pandas()
    except Exception as e:
        logger.error("Failed to load saved_items metadata: %s", e)
        return []

    # Deduplicate by item_id — merge topics across chunks
    items_by_id: dict[str, dict] = {}
    for _, row in df.iterrows():
        item_id = str(row.get("item_id", ""))
        if not item_id:
            continue

        if item_id not in items_by_id:
            items_by_id[item_id] = {
                "item_id": item_id,
                "topics": set(),
                "tags": set(),
                "saved_at": row.get("saved_at"),
                "usefulness_score": row.get("usefulness_score", 50),
                "content_type": str(row.get("content_type", "")),
                "source": str(row.get("source", "")),
            }

        # Merge topics (handles str, list, numpy.ndarray)
        topics_val = row.get("topics", "")
        if isinstance(topics_val, str) and topics_val:
            for t in topics_val.split(","):
                t = t.strip()
                if t:
                    items_by_id[item_id]["topics"].add(t)
        elif hasattr(topics_val, "__iter__") and not isinstance(topics_val, str):
            for t in topics_val:
                t_str = str(t).strip() if t is not None else ""
                if t_str:
                    items_by_id[item_id]["topics"].add(t_str)

        # Merge tags (handles str, list, numpy.ndarray)
        tags_val = row.get("tags", "")
        if isinstance(tags_val, str) and tags_val:
            for t in tags_val.split(","):
                t = t.strip()
                if t:
                    items_by_id[item_id]["tags"].add(t)
        elif hasattr(tags_val, "__iter__") and not isinstance(tags_val, str):
            for t in tags_val:
                t_str = str(t).strip() if t is not None else ""
                if t_str:
                    items_by_id[item_id]["tags"].add(t_str)

    # Convert sets to lists
    result = []
    for item in items_by_id.values():
        item["topics"] = list(item["topics"])
        item["tags"] = list(item["tags"])
        result.append(item)

    logger.info("Loaded %d unique items from saved_items", len(result))
    return result


def _compute_mining_scores(
    items: list[dict],
    min_count: int = 2,
    since: Optional[datetime] = None,
    content_only: bool = False,
) -> list[MinedTopic]:
    """Core mining: flatten topics → normalize → group → filter → score → sort.

    Args:
        items: Deduplicated saved_items metadata dicts.
        min_count: Minimum item count for a topic to be included.
        since: Optional datetime cutoff — only include items saved after this.
        content_only: If True, only include topics that map to x3lixi categories.

    Returns:
        Sorted list of MinedTopic (descending by mining_score).
    """
    now = datetime.now(timezone.utc)

    # Filter by time window
    if since is not None:
        filtered_items = []
        for item in items:
            saved_at = item.get("saved_at")
            if saved_at is None:
                continue
            if isinstance(saved_at, str):
                try:
                    saved_at = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
            if not hasattr(saved_at, "tzinfo") or saved_at.tzinfo is None:
                saved_at = saved_at.replace(tzinfo=timezone.utc)
            if saved_at >= since:
                filtered_items.append(item)
        items = filtered_items

    # Flatten: topic → list of item metadata
    topic_items: dict[str, list[dict]] = {}
    for item in items:
        for raw_topic in item.get("topics", []):
            normalized = _normalize_topic(raw_topic)
            if not normalized:
                continue
            if normalized not in topic_items:
                topic_items[normalized] = []
            topic_items[normalized].append({
                "raw_topic": raw_topic,
                "item": item,
            })

    # Score each topic
    mined: list[MinedTopic] = []
    for canonical, entries in topic_items.items():
        item_count = len(entries)
        if item_count < min_count:
            continue

        # Category check for --content-only
        category = _suggest_category(canonical)
        if content_only and category not in _CONTENT_CATEGORIES:
            continue

        # Collect raw variants
        raw_variants = list({e["raw_topic"] for e in entries})

        # Frequency: log2(count + 1)
        frequency_score = math.log2(item_count + 1)

        # Recency: mean exponential decay
        recency_weights = []
        for e in entries:
            saved_at = e["item"].get("saved_at")
            if saved_at is None:
                recency_weights.append(0.1)  # old unknown items get low weight
                continue
            if isinstance(saved_at, str):
                try:
                    saved_at = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    recency_weights.append(0.1)
                    continue
            if not hasattr(saved_at, "tzinfo") or saved_at.tzinfo is None:
                saved_at = saved_at.replace(tzinfo=timezone.utc)
            recency_weights.append(_recency_weight(saved_at, now))

        mean_recency = sum(recency_weights) / len(recency_weights) if recency_weights else 0.1

        # Quality: mean usefulness_score / 100
        quality_scores = []
        for e in entries:
            score = e["item"].get("usefulness_score", 50)
            if isinstance(score, (int, float)):
                quality_scores.append(score)
            else:
                quality_scores.append(50)
        mean_quality = (sum(quality_scores) / len(quality_scores) / 100.0) if quality_scores else 0.5

        # Source diversity: unique platforms
        sources = {e["item"].get("source", "") for e in entries}
        sources.discard("")
        source_diversity = max(len(sources), 1)
        diversity_bonus = 1.0 + 0.1 * (source_diversity - 1)

        # Content type breakdown
        content_types: dict[str, int] = {}
        for e in entries:
            ct = e["item"].get("content_type", "unknown")
            content_types[ct] = content_types.get(ct, 0) + 1

        # Composite mining score
        mining_score = frequency_score * mean_recency * mean_quality * diversity_bonus

        mined.append(MinedTopic(
            topic=canonical,
            raw_variants=raw_variants,
            item_count=item_count,
            mining_score=round(mining_score, 4),
            frequency_score=round(frequency_score, 4),
            recency_weight=round(mean_recency, 4),
            quality_factor=round(mean_quality, 4),
            source_diversity=source_diversity,
            content_types=content_types,
            category_suggestion=category,
        ))

    # Sort descending by mining_score
    mined.sort(key=lambda m: m.mining_score, reverse=True)
    return mined


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_output(results: list[TopicScore], fmt: str = "yaml") -> str:
    """Format a list of TopicScore results for output.

    Supported formats: yaml, json, markdown, pipeline
    """
    if fmt == "json":
        return _format_json(results)
    elif fmt == "markdown":
        return _format_markdown(results)
    elif fmt == "pipeline":
        return _format_pipeline(results)
    else:  # yaml
        return _format_yaml(results)


def _results_to_dicts(results: list[TopicScore]) -> list[dict]:
    """Convert TopicScore list to serializable dicts."""
    out = []
    for r in results:
        probes_dict = {}
        for src, probe in r.probes.items():
            probes_dict[src] = {
                "found": probe.found,
                "match_count": probe.match_count,
                "avg_relevance": round(probe.avg_relevance, 4),
                "top_excerpts": probe.top_excerpts[:3],
                "top_titles": probe.top_titles[:3],
            }
        entry = {
            "topic": r.topic,
            "convergence": r.convergence,
            "total_score": round(r.total_score, 2),
            "category_suggestion": r.category_suggestion,
            "already_published": r.already_published,
            "existing_briefs": r.existing_briefs,
            "probes": probes_dict,
        }
        if r.mining_data is not None:
            entry["mining_data"] = {
                "item_count": r.mining_data.item_count,
                "mining_score": round(r.mining_data.mining_score, 4),
                "frequency_score": round(r.mining_data.frequency_score, 4),
                "recency_weight": round(r.mining_data.recency_weight, 4),
                "quality_factor": round(r.mining_data.quality_factor, 4),
                "source_diversity": r.mining_data.source_diversity,
                "content_types": r.mining_data.content_types,
                "raw_variants": r.mining_data.raw_variants,
            }
        out.append(entry)
    return out


def _format_json(results: list[TopicScore]) -> str:
    """Format as JSON."""
    data = {
        "agent": "obsession_radar",
        "results": _results_to_dicts(results),
        "total": len(results),
    }
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _format_yaml(results: list[TopicScore]) -> str:
    """Format as YAML."""
    try:
        import yaml
        data = {
            "agent": "obsession_radar",
            "results": _results_to_dicts(results),
            "total": len(results),
        }
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except ImportError:
        logger.warning("PyYAML not installed, falling back to JSON")
        return _format_json(results)


def _format_markdown(results: list[TopicScore]) -> str:
    """Format as a human-readable markdown report."""
    lines = ["# Obsession Radar Report", ""]
    if not results:
        lines.append("No topics found.")
        return "\n".join(lines)

    source_icons = {
        "vault": "Vault",
        "conversations": "Conversations",
        "saved_items": "TheSource",
        "audio": "Audio",
        "published": "Published",
    }

    for i, r in enumerate(results, 1):
        status = "PUBLISHED" if r.already_published else "NEW"
        lines.append(
            f"## {i}. \"{r.topic}\" -- Score: {r.total_score:.1f} "
            f"({r.convergence}/5 sources) [{status}]"
        )
        lines.append("")
        lines.append("| Source | Found | Items | Avg Score | Top Match |")
        lines.append("|--------|-------|-------|-----------|-----------|")

        for src_key, label in source_icons.items():
            probe = r.probes.get(src_key)
            if probe is None:
                continue
            found_str = "YES" if probe.found else "no"
            top_match = probe.top_titles[0] if probe.top_titles else "--"
            avg_str = f"{probe.avg_relevance:.2f}" if probe.found else "--"
            lines.append(
                f"| {label} | {found_str} | {probe.match_count} "
                f"| {avg_str} | {top_match} |"
            )

        lines.append("")
        if r.category_suggestion:
            lines.append(f"**Category:** {r.category_suggestion}")
        if r.mining_data is not None:
            md = r.mining_data
            lines.append(
                f"**Mined:** {md.item_count} saves | "
                f"score={md.mining_score:.2f} | "
                f"recency={md.recency_weight:.2f} | "
                f"quality={md.quality_factor:.2f} | "
                f"{md.source_diversity} platforms"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _format_pipeline(results: list[TopicScore]) -> str:
    """Format as pipeline-ready entries for ideas.md."""
    if not results:
        return "# No topics found\n"
    lines = []
    for r in results:
        status_tag = "PUBLISHED" if r.already_published else "NOT yet published"
        lines.append(f"### {r.topic}")
        lines.append(
            f"Obsession Radar: {r.total_score:.1f}/100 | "
            f"{r.convergence}/5 sources | {status_tag}"
        )

        # Per-source summary
        source_parts = []
        source_icons = {
            "vault": "vault notes",
            "conversations": "AI conversations",
            "saved_items": "saved links",
            "audio": "audio reflections",
        }
        for src_key, label in source_icons.items():
            probe = r.probes.get(src_key)
            if probe and probe.found:
                source_parts.append(f"{probe.match_count} {label}")
        if source_parts:
            lines.append(" | ".join(source_parts))

        # Top matches
        for src_key in ["vault", "conversations"]:
            probe = r.probes.get(src_key)
            if probe and probe.found and probe.top_titles:
                label = src_key.replace("_", " ").title()
                lines.append(
                    f"Top {label}: \"{probe.top_titles[0]}\" ({probe.avg_relevance:.2f})"
                )

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mined output formatting (for --skip-scan mode returning MinedTopic list)
# ---------------------------------------------------------------------------

def format_mined_output(results: list[MinedTopic], fmt: str = "yaml") -> str:
    """Format a list of MinedTopic results for output.

    Supported formats: yaml, json, markdown, pipeline
    """
    if fmt == "json":
        return _format_mined_json(results)
    elif fmt == "markdown":
        return _format_mined_markdown(results)
    elif fmt == "pipeline":
        return _format_mined_pipeline(results)
    else:  # yaml
        return _format_mined_yaml(results)


def _mined_to_dicts(results: list[MinedTopic]) -> list[dict]:
    """Convert MinedTopic list to serializable dicts."""
    out = []
    for m in results:
        out.append({
            "topic": m.topic,
            "item_count": m.item_count,
            "mining_score": round(m.mining_score, 4),
            "frequency_score": round(m.frequency_score, 4),
            "recency_weight": round(m.recency_weight, 4),
            "quality_factor": round(m.quality_factor, 4),
            "source_diversity": m.source_diversity,
            "content_types": m.content_types,
            "category_suggestion": m.category_suggestion,
            "raw_variants": m.raw_variants,
        })
    return out


def _format_mined_json(results: list[MinedTopic]) -> str:
    """Format MinedTopic list as JSON."""
    data = {
        "agent": "obsession_radar",
        "mode": "mine",
        "results": _mined_to_dicts(results),
        "total": len(results),
    }
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _format_mined_yaml(results: list[MinedTopic]) -> str:
    """Format MinedTopic list as YAML."""
    try:
        import yaml
        data = {
            "agent": "obsession_radar",
            "mode": "mine",
            "results": _mined_to_dicts(results),
            "total": len(results),
        }
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except ImportError:
        logger.warning("PyYAML not installed, falling back to JSON")
        return _format_mined_json(results)


def _format_mined_markdown(results: list[MinedTopic]) -> str:
    """Format MinedTopic list as human-readable markdown."""
    lines = ["# Obsession Radar — Mined Topics", ""]
    if not results:
        lines.append("No topics mined.")
        return "\n".join(lines)

    lines.append(f"**{len(results)} topics** extracted from saved_items\n")
    lines.append("| # | Topic | Saves | Score | Recency | Quality | Platforms | Category |")
    lines.append("|---|-------|-------|-------|---------|---------|-----------|----------|")

    for i, m in enumerate(results, 1):
        cat = m.category_suggestion or "--"
        lines.append(
            f"| {i} | {m.topic} | {m.item_count} | {m.mining_score:.2f} "
            f"| {m.recency_weight:.2f} | {m.quality_factor:.2f} "
            f"| {m.source_diversity} | {cat} |"
        )

    lines.append("")

    # Detailed breakdown for top 10
    for i, m in enumerate(results[:10], 1):
        lines.append(f"### {i}. {m.topic}")
        lines.append(f"- **Saves:** {m.item_count} items")
        lines.append(f"- **Mining score:** {m.mining_score:.4f}")
        lines.append(f"- **Frequency:** {m.frequency_score:.2f} | "
                      f"Recency: {m.recency_weight:.2f} | "
                      f"Quality: {m.quality_factor:.2f}")
        lines.append(f"- **Platforms:** {m.source_diversity}")
        if m.content_types:
            ct_parts = [f"{k}: {v}" for k, v in sorted(m.content_types.items(), key=lambda x: -x[1])]
            lines.append(f"- **Content types:** {', '.join(ct_parts)}")
        if m.raw_variants and len(m.raw_variants) > 1:
            lines.append(f"- **Variants:** {', '.join(m.raw_variants[:5])}")
        if m.category_suggestion:
            lines.append(f"- **Category:** {m.category_suggestion}")
        lines.append("")

    return "\n".join(lines)


def _format_mined_pipeline(results: list[MinedTopic]) -> str:
    """Format MinedTopic list as pipeline-ready ideas.md entries."""
    if not results:
        return "# No topics mined\n"
    lines = []
    for m in results:
        cat = m.category_suggestion or "uncategorized"
        lines.append(f"### {m.topic}")
        lines.append(
            f"Mined from {m.item_count} saves | "
            f"score={m.mining_score:.2f} | "
            f"category: {cat}"
        )
        if m.content_types:
            ct_parts = [f"{v} {k}" for k, v in sorted(m.content_types.items(), key=lambda x: -x[1])]
            lines.append(f"Sources: {', '.join(ct_parts)}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ObsessionRadar class
# ---------------------------------------------------------------------------

class ObsessionRadar:
    """Main radar class that orchestrates topic scanning across all sources."""

    def __init__(self, config=None):
        if config is None:
            from config import load_config
            config = load_config()

        self.config = config
        self._embed_model = None  # Lazy init
        self._audio_dir = DEFAULT_AUDIO_DIR
        self._pipeline_dir = DEFAULT_PIPELINE_DIR

    def _get_embed_model(self):
        """Lazy-initialize the embedding model."""
        if self._embed_model is None:
            from embeddings import get_embedding_model
            self._embed_model = get_embedding_model(
                self.config.embedding, self.config.voyage_api_key
            )
        return self._embed_model

    # ------ File-based scanners ------

    def _scan_audio(self, topic: str, audio_dir: Optional[Path] = None) -> SourceProbe:
        """Scan AI~udio-Reflections analysis files for topic mentions."""
        scan_dir = audio_dir or self._audio_dir

        if not scan_dir.exists():
            logger.warning("Audio directory not found: %s", scan_dir)
            return SourceProbe(
                source="audio", found=False, match_count=0,
                avg_relevance=0.0, top_excerpts=[], top_titles=[],
            )

        hits = []
        for analysis_file in scan_dir.glob("*_analysis.md"):
            try:
                content = analysis_file.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.debug("Could not read %s: %s", analysis_file, e)
                continue

            if _topic_matches(topic, content):
                # Extract the most relevant section
                excerpt = _extract_matching_section(topic, content)
                hits.append({
                    "file": analysis_file.name,
                    "excerpt": excerpt[:200] if excerpt else analysis_file.name,
                    "title": analysis_file.stem,
                })

        if not hits:
            return SourceProbe(
                source="audio", found=False, match_count=0,
                avg_relevance=0.0, top_excerpts=[], top_titles=[],
            )

        # Simple relevance: count of topic mentions / max reasonable
        avg_rel = min(len(hits) / 5.0, 1.0)

        return SourceProbe(
            source="audio",
            found=True,
            match_count=len(hits),
            avg_relevance=round(avg_rel, 4),
            top_excerpts=[h["excerpt"] for h in hits[:3]],
            top_titles=[h["title"] for h in hits[:3]],
        )

    def _scan_published(self, topic: str, pipeline_dir: Optional[Path] = None) -> SourceProbe:
        """Scan published articles and processed briefs for topic coverage."""
        scan_dir = pipeline_dir or self._pipeline_dir

        if not scan_dir.exists():
            logger.warning("Pipeline directory not found: %s", scan_dir)
            return SourceProbe(
                source="published", found=False, match_count=0,
                avg_relevance=0.0, top_excerpts=[], top_titles=[],
            )

        hits = []

        # Check 04-published
        published_dir = scan_dir / "04-published"
        if published_dir.exists():
            for article in published_dir.rglob("*.md"):
                try:
                    content = article.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                fm = _parse_frontmatter(content)
                title = fm.get("title", article.stem)
                # Match against title and first 500 chars
                if _topic_matches(topic, title) or _topic_matches(topic, content[:500]):
                    hits.append({
                        "title": title,
                        "file": article.name,
                        "status": "published",
                    })

        # Check 02-processed (briefs)
        processed_dir = scan_dir / "02-processed"
        if processed_dir.exists():
            for brief in processed_dir.rglob("*.md"):
                try:
                    content = brief.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                fm = _parse_frontmatter(content)
                title = fm.get("title", brief.stem)
                if _topic_matches(topic, title) or _topic_matches(topic, content[:500]):
                    hits.append({
                        "title": title,
                        "file": brief.name,
                        "status": "brief",
                    })

        if not hits:
            return SourceProbe(
                source="published", found=False, match_count=0,
                avg_relevance=0.0, top_excerpts=[], top_titles=[],
            )

        return SourceProbe(
            source="published",
            found=True,
            match_count=len(hits),
            avg_relevance=1.0,  # Binary: covered or not
            top_excerpts=[h.get("status", "") for h in hits[:3]],
            top_titles=[h["title"] for h in hits[:3]],
        )

    # ------ Vector-based scanners ------

    def _scan_vector_source(
        self,
        topic: str,
        source: str,
        threshold: float = 0.50,
        top_k: int = 20,
    ) -> SourceProbe:
        """Scan a LanceDB vector source for topic matches.

        Args:
            topic: The topic to search for.
            source: Source name ("vault", "conversations", "saved_items").
            threshold: Minimum raw_score to consider a match.
            top_k: Maximum results to retrieve.
        """
        try:
            embed_model = self._get_embed_model()
            query_vector = embed_model.get_query_embedding(topic)
        except Exception as e:
            logger.error("Failed to embed query '%s': %s", topic, e)
            return SourceProbe(
                source=source, found=False, match_count=0,
                avg_relevance=0.0, top_excerpts=[], top_titles=[],
            )

        weight = SOURCE_WEIGHTS.get(source, 0.5)

        try:
            results = _search_source(source, query_vector, weight, top_k, threshold)
        except Exception as e:
            logger.error("Search failed for source '%s': %s", source, e)
            return SourceProbe(
                source=source, found=False, match_count=0,
                avg_relevance=0.0, top_excerpts=[], top_titles=[],
            )

        # Filter results that meet threshold
        valid = [r for r in results if r.raw_score >= threshold]

        if not valid:
            return SourceProbe(
                source=source, found=False, match_count=0,
                avg_relevance=0.0, top_excerpts=[], top_titles=[],
            )

        avg_rel = sum(r.raw_score for r in valid) / len(valid)

        return SourceProbe(
            source=source,
            found=True,
            match_count=len(valid),
            avg_relevance=round(avg_rel, 4),
            top_excerpts=[r.chunk[:200] for r in valid[:3]],
            top_titles=[r.title for r in valid[:3]],
        )

    # ------ Main scanning interface ------

    def scan_topic(self, topic: str) -> TopicScore:
        """Scan all 5 sources for a single topic and return scored result."""
        probes: dict[str, SourceProbe] = {}

        # Vector sources
        for source, threshold in SOURCE_THRESHOLDS.items():
            try:
                probes[source] = self._scan_vector_source(topic, source, threshold)
            except Exception as e:
                logger.error("Error scanning %s: %s", source, e)
                probes[source] = SourceProbe(
                    source=source, found=False, match_count=0,
                    avg_relevance=0.0, top_excerpts=[], top_titles=[],
                )

        # File-based sources
        probes["audio"] = self._scan_audio(topic)
        probes["published"] = self._scan_published(topic)

        # Compute scores
        convergence = sum(1 for p in probes.values() if p.found and p.source != "published")
        total_score = _compute_score(probes)
        category = _suggest_category(topic)

        published_probe = probes.get("published")
        already_published = published_probe.found if published_probe else False
        existing_briefs = 0
        if published_probe and published_probe.found:
            existing_briefs = published_probe.match_count

        return TopicScore(
            topic=topic,
            convergence=convergence,
            total_score=total_score,
            probes=probes,
            category_suggestion=category,
            already_published=already_published,
            existing_briefs=existing_briefs,
        )

    def scan_topics(self, topics: list[str]) -> list[TopicScore]:
        """Scan multiple topics and return results sorted by score descending."""
        results = [self.scan_topic(t) for t in topics]
        results.sort(key=lambda r: r.total_score, reverse=True)
        return results

    def auto_discover(self, top_k: int = 20, categories: Optional[list[str]] = None) -> list[TopicScore]:
        """Auto-discover topics by scanning category seed topics.

        Args:
            top_k: Maximum number of results to return.
            categories: List of category slugs to scan. None means all.

        Returns:
            Top-k TopicScore results sorted by score descending.
        """
        candidate_topics = []

        cats_to_scan = categories or list(CATEGORY_SEED_TOPICS.keys())
        for cat in cats_to_scan:
            seeds = CATEGORY_SEED_TOPICS.get(cat, [])
            candidate_topics.extend(seeds)

        # Deduplicate
        seen = set()
        unique_topics = []
        for t in candidate_topics:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                unique_topics.append(t)

        logger.info("Auto-discover: scanning %d candidate topics", len(unique_topics))

        results = self.scan_topics(unique_topics)
        return results[:top_k]

    # ------ Mining interface ------

    def mine_topics(
        self,
        top_k: int = 20,
        content_only: bool = False,
        min_count: int = 2,
        since: Optional[str] = None,
        skip_scan: bool = False,
        new_only: bool = False,
        category: Optional[str] = None,
    ) -> list:
        """Mine topics from saved_items and optionally cross-scan for convergence.

        Args:
            top_k: Maximum number of results to return.
            content_only: Only include topics mapping to x3lixi categories.
            min_count: Minimum item count for a topic to be included.
            since: Time window string (e.g. "30d", "6m", "1y").
            skip_scan: If True, return MinedTopic list without convergence scan.
            new_only: Exclude already-published topics.
            category: Filter to a specific category slug.

        Returns:
            List of MinedTopic (if skip_scan) or TopicScore (with mining_data).
        """
        # 1. Load metadata
        items = _load_saved_items_metadata()
        if not items:
            logger.warning("No saved_items found in LanceDB")
            return []

        logger.info("Mining topics from %d saved items...", len(items))

        # 2. Parse time window
        since_dt = _parse_since(since) if since else None

        # 3. Compute mining scores
        mined = _compute_mining_scores(
            items,
            min_count=min_count,
            since=since_dt,
            content_only=content_only,
        )

        # 4. Category filter
        if category:
            mined = [m for m in mined if m.category_suggestion == category]

        # 5. Take top K
        mined = mined[:top_k]

        logger.info("Mined %d topics from saved_items", len(mined))

        if skip_scan:
            return mined

        # 6. Cross-scan for convergence
        topic_names = [m.topic for m in mined]
        logger.info("Running convergence scan on %d mined topics...", len(topic_names))

        scored = self.scan_topics(topic_names)

        # 7. Attach mining_data to each TopicScore
        mining_lookup = {m.topic: m for m in mined}
        for ts in scored:
            ts.mining_data = mining_lookup.get(ts.topic)

        # 8. Filter new-only
        if new_only:
            scored = [ts for ts in scored if not ts.already_published]

        return scored


# ---------------------------------------------------------------------------
# Helper for audio section extraction
# ---------------------------------------------------------------------------

def _extract_matching_section(topic: str, content: str) -> str:
    """Extract the section of content most relevant to the topic.

    Looks for INSIGHTS and NEXT ACTIONS sections first, then falls back
    to the first paragraph containing the topic.
    """
    # Priority sections
    for section_header in ["## INSIGHTS", "## NEXT ACTION", "## NEXT STEPS"]:
        idx = content.find(section_header)
        if idx != -1:
            section_text = content[idx:idx + 500]
            if _topic_matches(topic, section_text):
                return section_text.strip()

    # Fallback: find paragraph containing topic
    paragraphs = content.split("\n\n")
    for para in paragraphs:
        if _topic_matches(topic, para):
            return para.strip()[:200]

    return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="obsession_radar",
        description="Obsession Radar -- Cross-Source Topic Convergence Detection",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # scan
    scan_p = subparsers.add_parser("scan", help="Score specific topics")
    scan_p.add_argument(
        "topics", nargs="+",
        help="Topics to scan for convergence",
    )
    scan_p.add_argument(
        "--format", "-f",
        choices=["yaml", "json", "markdown", "pipeline"],
        default="yaml",
        help="Output format (default: yaml)",
    )
    scan_p.add_argument(
        "--new-only", action="store_true",
        help="Exclude already-published topics",
    )
    scan_p.add_argument(
        "--category", "-c",
        help="Filter results by category slug",
    )
    scan_p.add_argument(
        "--pipeline", action="store_true",
        help="Generate 01-inbox/ideas.md compatible entries",
    )

    # discover
    disc_p = subparsers.add_parser("discover", help="Auto-discover topics from category seeds")
    disc_p.add_argument(
        "--top", "-n", type=int, default=20,
        help="Number of top results to return (default: 20)",
    )
    disc_p.add_argument(
        "--format", "-f",
        choices=["yaml", "json", "markdown", "pipeline"],
        default="yaml",
        help="Output format (default: yaml)",
    )
    disc_p.add_argument(
        "--new-only", action="store_true",
        help="Exclude already-published topics",
    )
    disc_p.add_argument(
        "--categories",
        help="Comma-separated category slugs to scan (default: all)",
    )

    # mine
    mine_p = subparsers.add_parser(
        "mine",
        help="Mine topics from saved_items with optional convergence scan",
    )
    mine_p.add_argument(
        "--top", "-n", type=int, default=20,
        help="Number of top results to return (default: 20)",
    )
    mine_p.add_argument(
        "--format", "-f",
        choices=["yaml", "json", "markdown", "pipeline"],
        default="yaml",
        help="Output format (default: yaml)",
    )
    mine_p.add_argument(
        "--content-only", action="store_true",
        help="Only include topics that map to x3lixi content categories",
    )
    mine_p.add_argument(
        "--skip-scan", action="store_true",
        help="Skip convergence scan (fast mode, no API calls)",
    )
    mine_p.add_argument(
        "--new-only", action="store_true",
        help="Exclude already-published topics",
    )
    mine_p.add_argument(
        "--since",
        help="Time window for saved items (e.g. 30d, 90d, 6m, 1y)",
    )
    mine_p.add_argument(
        "--min-count", type=int, default=2,
        help="Minimum item count for a topic (default: 2)",
    )
    mine_p.add_argument(
        "--category", "-c",
        help="Filter to a specific category slug",
    )
    mine_p.add_argument(
        "--pipeline", action="store_true",
        help="Generate 01-inbox/ideas.md compatible entries",
    )

    # Shared
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress log output on stderr",
    )

    return parser


def main():
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging -- all to stderr
    log_level = logging.WARNING if getattr(args, "quiet", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    if args.command == "scan":
        radar = ObsessionRadar()
        results = radar.scan_topics(args.topics)

        # Filters
        if args.new_only:
            results = [r for r in results if not r.already_published]

        if args.category:
            results = [r for r in results if r.category_suggestion == args.category]

        # Format
        fmt = "pipeline" if args.pipeline else args.format
        print(format_output(results, fmt=fmt))

    elif args.command == "discover":
        cats = args.categories.split(",") if args.categories else None
        radar = ObsessionRadar()
        results = radar.auto_discover(top_k=args.top, categories=cats)

        if args.new_only:
            results = [r for r in results if not r.already_published]

        print(format_output(results, fmt=args.format))

    elif args.command == "mine":
        radar = ObsessionRadar()
        fmt = "pipeline" if getattr(args, "pipeline", False) else args.format

        results = radar.mine_topics(
            top_k=args.top,
            content_only=args.content_only,
            min_count=args.min_count,
            since=args.since,
            skip_scan=args.skip_scan,
            new_only=args.new_only,
            category=args.category,
        )

        if args.skip_scan:
            # Results are MinedTopic list
            print(format_mined_output(results, fmt=fmt))
        else:
            # Results are TopicScore list (with mining_data attached)
            print(format_output(results, fmt=fmt))

    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
