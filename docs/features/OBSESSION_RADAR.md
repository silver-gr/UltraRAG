# Obsession Radar

Discover your recurring interests by scanning 5 knowledge sources for topic convergence. Topics that appear across multiple sources are "obsessions" -- high-signal content creation candidates that carry genuine depth and multi-angle knowledge.

## Quick Start

```bash
# Scan specific topics
python -m obsession_radar scan "dopamine" "sleep" "stoicism"

# Auto-discover from category seeds
python -m obsession_radar discover --top 10

# Mine topics from your saved items
python -m obsession_radar mine --top 20 --format markdown
```

## How It Works

The Obsession Radar probes 5 knowledge sources for each topic and produces a convergence score:

```
Topic: "sleep optimization"
                │
    ┌───────────┼───────────────────┐
    ▼           ▼                   ▼
┌────────┐ ┌──────────┐ ┌──────────────────┐
│ Vault  │ │ AI Convs │ │ TheSource        │
│ (1.0)  │ │ (0.8)    │ │ (saved items)    │
│        │ │          │ │ (0.6)            │
└────────┘ └──────────┘ └──────────────────┘
    ▼           ▼                   ▼
┌────────┐ ┌──────────┐
│ Audio  │ │Published │
│ Reflec.│ │ Articles │
│ (0.7)  │ │ (-0.3)   │ ← negative! reduces score
└────────┘ └──────────┘
                │
                ▼
    Convergence: 4/5 sources
    Score: 2.85
    → High-signal content candidate
```

### 5 Knowledge Sources

| Source | Type | Weight | Description |
|--------|------|--------|-------------|
| Vault | Vector search | 1.0 | Obsidian vault (consciously curated notes) |
| Conversations | Vector search | 0.8 | AI conversation exports (Claude, Gemini, ChatGPT) |
| TheSource | Vector search | 0.6 | Saved items (Reddit, YouTube, bookmarks, etc.) |
| Audio Reflections | File scan | 0.7 | Audio recording transcripts and analysis |
| Published Articles | File scan | -0.3 | Already published content (reduces score!) |

The **negative weight** on published articles is intentional: topics you've already covered should rank lower to surface fresh content opportunities.

### Scoring Formula

For each topic:

```
score = sum(weight * (avg_relevance + 0.3 * min(match_count/10, 1.0)))
        for each source where found = True
```

**Convergence** = count of non-published sources where the topic was found (0-5).

### Bilingual Matching

The radar includes a built-in bilingual map (30+ English-Greek pairs) so searching "sleep" also matches Greek content about "ύπνος", and "στωικισμός" matches English "stoicism". Three matching strategies are used:

1. **Direct phrase match** -- exact phrase search
2. **Multi-word all-words match** -- all words must appear
3. **Bilingual cross-match** -- translated term search

### Relevance Thresholds

Only matches above these thresholds count:

| Source | Threshold |
|--------|-----------|
| Vault | 0.55 |
| Conversations | 0.50 |
| TheSource | 0.50 |

## Operating Modes

### 1. Scan Mode

Score specific topics you name explicitly.

```bash
# Basic scan
python -m obsession_radar scan "dopamine" "sleep" "stoicism"

# Markdown report format
python -m obsession_radar scan "dopamine" --format markdown

# Exclude already-published topics
python -m obsession_radar scan "sleep" --new-only

# Content pipeline output (ideas.md-compatible)
python -m obsession_radar scan "habits" --pipeline

# Filter by category in output
python -m obsession_radar scan "focus" --category "productivity-work"
```

### 2. Discover Mode

Auto-discover topics from 10 category seed topics. Each category generates 12 seed topics, scanning ~120 candidates total.

```bash
# Top 10 discoveries
python -m obsession_radar discover --top 10

# Markdown report, exclude published
python -m obsession_radar discover --top 20 --format markdown --new-only

# Limit to specific categories
python -m obsession_radar discover --categories "health-fitness,rest-recovery"
```

**10 Category Seeds:**
`relationships`, `health-fitness`, `finances-wealth`, `emotional-wellbeing`, `mindfulness-stress`, `rest-recovery`, `productivity-work`, `creativity-leisure`, `learning-growth`, `purpose-spirituality`

### 3. Mine Mode

Mine topics from your TheSource saved items metadata (tags, topics), then optionally run convergence scan on top results.

```bash
# Mine and scan top 20
python -m obsession_radar mine --top 20 --format markdown

# Fast mining only (no API calls, no convergence scan)
python -m obsession_radar mine --skip-scan --top 50 --format json

# Content-only categories (x3lixi categories)
python -m obsession_radar mine --content-only --since 90d --top 15

# Recent and unpublished
python -m obsession_radar mine --new-only --since 30d

# Filter by category
python -m obsession_radar mine --category "health-fitness"

# Pipeline output with minimum occurrence threshold
python -m obsession_radar mine --min-count 3 --pipeline
```

#### Mining Scoring

Topics mined from saved items are scored with a composite formula:

```
mining_score = frequency_score * mean_recency * mean_quality * diversity_bonus
```

| Factor | Formula | Description |
|--------|---------|-------------|
| `frequency_score` | `log2(count + 1)` | How often the topic appears |
| `mean_recency` | Exponential decay, 90-day half-life | Recent items score higher |
| `mean_quality` | `mean(usefulness_score / 100)` | Higher-quality items score higher |
| `diversity_bonus` | `1.0 + 0.1 * (unique_platforms - 1)` | Multi-platform bonus |

## Output Formats

| Format | Flag | Description |
|--------|------|-------------|
| YAML | `--format yaml` | Default. Structured output with all probe data |
| JSON | `--format json` | Machine-readable JSON |
| Markdown | `--format markdown` | Human-readable report with per-source tables |
| Pipeline | `--pipeline` | `ideas.md`-compatible entries for content creation pipeline |

### Example Markdown Output

```markdown
## dopamine (Convergence: 4/5, Score: 2.85)

| Source | Found | Matches | Avg Relevance | Top Excerpts |
|--------|-------|---------|---------------|--------------|
| Vault | Yes | 12 | 0.72 | "Dopamine detox protocol..." |
| Conversations | Yes | 8 | 0.65 | "We discussed dopamine..." |
| TheSource | Yes | 15 | 0.58 | "Reddit: dopamine fasting..." |
| Audio | Yes | 3 | 0.70 | "Morning routine impact..." |
| Published | No | 0 | 0.00 | — |
```

### Example Pipeline Output

```markdown
- **dopamine** (Score: 2.85, Conv: 4/5, Cat: health-fitness)
  Vault: 12 matches | Conversations: 8 | TheSource: 15 | Audio: 3
```

## Common Flags

| Flag | Description |
|------|-------------|
| `--format yaml\|json\|markdown` | Output format |
| `--top N` | Return top N results |
| `--new-only` | Exclude topics already published |
| `--pipeline` | Output in ideas.md format |
| `--category "cat-name"` | Filter output to specific category |
| `--quiet` | Suppress INFO logs to stderr |
| `--since Nd` | Time filter for mining (e.g., `90d`, `30d`) |
| `--skip-scan` | Mining only, skip convergence scan (fast, no API calls) |
| `--content-only` | Limit to x3lixi content categories |
| `--min-count N` | Minimum occurrence count for mining |

## Use Cases

### Content Creation Pipeline

Find your next article topic:

```bash
# 1. Mine recent saves for trending interests
python -m obsession_radar mine --since 30d --new-only --top 10 --format markdown

# 2. Deep-scan the most promising candidates
python -m obsession_radar scan "topic1" "topic2" "topic3" --pipeline

# 3. Feed into content pipeline
# Copy the --pipeline output into x3lixi-content-pipeline ideas.md
```

### Knowledge Gap Analysis

Find what you know deeply but haven't written about:

```bash
python -m obsession_radar discover --top 20 --new-only --format markdown
```

High convergence + "not published" = untapped expertise.

### Trend Detection

See what topics are gaining traction in your saves:

```bash
python -m obsession_radar mine --since 14d --skip-scan --top 30 --format markdown
```

## Data Types

### TopicScore

The primary output type for each analyzed topic:

| Field | Type | Description |
|-------|------|-------------|
| `topic` | str | Topic name |
| `convergence` | int | Number of non-published sources found (0-5) |
| `total_score` | float | Weighted composite score |
| `probes` | dict | Per-source `SourceProbe` results |
| `category_suggestion` | str | Suggested content category |
| `already_published` | bool | Whether topic was found in published articles |
| `existing_briefs` | list | Any existing content briefs for this topic |
| `mining_data` | MinedTopic | Mining metadata (mine mode only) |

### SourceProbe

Result from scanning one source:

| Field | Type | Description |
|-------|------|-------------|
| `source` | str | Source name |
| `found` | bool | Whether topic was found |
| `match_count` | int | Number of matching documents |
| `avg_relevance` | float | Average relevance score |
| `top_excerpts` | list | Sample matching text snippets |
| `top_titles` | list | Titles of matching documents |

## Key Files

| File | Purpose |
|------|---------|
| `obsession_radar.py` | All logic: `ObsessionRadar`, scoring, mining, bilingual map, CLI |
| `content_research_engine.py` | Delegates `_search_source` for vector searches |
| `tests/test_obsession_radar.py` | 23-phase test suite |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No results found" | Check that LanceDB tables exist (run `python -m cli status`) |
| Low convergence scores | Try broader topic names or check bilingual map coverage |
| Mining returns empty | Ensure TheSource `saved_items` table has `topic`/`tag` metadata |
| Slow discovery mode | Normal -- scans ~120 candidates across 5 sources. Use `--quiet` to reduce noise |
| Audio source not found | Check that `AI~udio-Reflections/03-transcripts/` path exists |
