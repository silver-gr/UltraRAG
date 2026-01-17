# Research Mode Enhancements

This document describes the enhancements made to UltraRAG's research mode for improved performance, reliability, and comprehensive retrieval.

## Overview

Research mode (`@research <query>`) performs iterative retrieval with LLM-powered gap analysis. These enhancements address:

1. **Exhaustive query support** - Force full iterations for "all/every" queries
2. **Dual-model architecture** - Separate models for different tasks to reduce rate limiting
3. **Progressive retry** - Handle MAX_TOKENS errors gracefully
4. **Rate limiting mitigation** - Delays and optimizations to prevent API throttling

## Features

### 1. Exhaustive Query Mode

For queries that request comprehensive results (e.g., "all habits", "every goal"), research mode can be forced to run all iterations regardless of confidence score.

**Activation methods:**

1. **Prefix**: Use `@all <query>` to force exhaustive mode
   ```
   @all all my habits and routines
   ```

2. **Auto-detection**: Patterns like "all", "every", "complete list", "comprehensive" are detected automatically

**Detected patterns (English & Greek):**
- `\ball\b` - "all habits", "list all"
- `\bevery\b` - "every routine"
- `\bcomplete\s+list\b` - "complete list of"
- `\bcomprehensive\b` - "comprehensive overview"
- `\bexhaustive\b` - "exhaustive list"
- `\bentire\b` - "entire collection"
- `\bfull\s+list\b` - "full list"
- `\bόλα\b|\bόλες\b|\bόλους\b` - Greek: all
- `\bκάθε\b` - Greek: every
- `\bπλήρης?\b` - Greek: complete/full

**Configuration:**
```python
# research_mode.py
EXHAUSTIVE_MAX_ITERATIONS = 5  # vs default 3
```

### 2. Dual-Model Architecture

Research mode uses two different Gemini models to optimize for speed and reduce rate limiting:

| Function | Model | AFC | Max Tokens | Purpose |
|----------|-------|-----|------------|---------|
| Gap Analysis | `gemini-flash-latest` (2.5 Flash) | Disabled | 1024 | Fast, lightweight |
| Sub-query Generation | `gemini-3-flash-preview` | Enabled | Default | Quality queries |
| Final Synthesis | `gemini-3-flash-preview` | Enabled | 65536 | Comprehensive output |

**Why separate models?**

1. **Rate limiting**: Gap analysis runs multiple times per research session. Using a lighter model with AFC disabled reduces the chance of hitting rate limits.

2. **Speed**: `gemini-flash-latest` (based on Gemini 2.5 Flash) is faster for simple tasks like confidence scoring.

3. **AFC overhead**: Automatic Function Calling adds latency. Disabled for gap analysis since it doesn't need tool use.

**Implementation:**
```python
# research_mode.py
GAP_ANALYSIS_MODEL = "gemini-flash-latest"

def _create_gap_analysis_llm(self):
    return GoogleGenAI(
        model=self.GAP_ANALYSIS_MODEL,
        api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
        is_function_calling_model=False,  # Disable AFC
    )
```

### 3. Progressive Retry for MAX_TOKENS

When synthesis hits the model's output token limit, research mode progressively retries with fewer nodes:

```
Attempt 1: 100% of nodes (all retrieved)
    ↓ MAX_TOKENS error
Attempt 2: 80% of nodes
    ↓ MAX_TOKENS error
Attempt 3: 66% of nodes
    ↓ MAX_TOKENS error
Attempt 4: 300 nodes (hard floor)
```

**Example console output:**
```
Research synthesis: 574 retrieved, 574 for synthesis, retry limits: [574, 459, 378, 300]
Synthesis attempt 1/4: using 574 nodes
Synthesis attempt 1 hit MAX_TOKENS with 574 nodes. Retrying with fewer nodes...
Synthesis attempt 2/4: using 459 nodes
Synthesis succeeded on attempt 2 with 459 nodes
```

**Why 300 as hard floor?**

| Sources | Approx Context Tokens | Status |
|---------|----------------------|--------|
| 300 | ~150k | Safe |
| 400 | ~200k | Borderline |
| 500+ | ~250k+ | Risk of MAX_TOKENS |

The top 300 sources (sorted by relevance score) typically contain the most important information.

### 4. Rate Limiting Mitigation

**Inter-iteration delay:**
```python
ITERATION_DELAY = 5  # seconds between iterations
```

This spreads API calls to avoid triggering Gemini's rate limiting, which was causing ~5 minute delays on iteration 3-4.

**Timing logs:**
```
Gap analysis LLM call took 2.3s (model: gemini-flash-latest)
Sub-query generation LLM call took 1.8s (main LLM)
Waiting 5s before iteration 3 to avoid rate limiting...
```

## Configuration Reference

### Class Constants (research_mode.py)

```python
class ResearchRetriever:
    EXHAUSTIVE_MAX_ITERATIONS = 5      # Max iterations for exhaustive queries
    ITERATION_DELAY = 5                 # Seconds between iterations
    GAP_ANALYSIS_MODEL = "gemini-flash-latest"  # Lightweight model for gap analysis
```

### Environment Variables

```bash
# .env
RESEARCH_MAX_ITERATIONS=3              # Default iterations (non-exhaustive)
RESEARCH_CONFIDENCE_THRESHOLD=0.8      # Stop if confidence exceeds this
RESEARCH_MAX_SYNTHESIS_SOURCES=0       # 0 = unlimited (subject to progressive retry)
```

## Console Output Examples

### Normal Research Query
```
ResearchRetriever initialized (max_iterations=3, confidence_threshold=0.8, gap_analysis_model=gemini-flash-latest)
Created gap analysis LLM: gemini-flash-latest (AFC disabled)
Starting research mode for query: what are my productivity habits?
Research iteration 1/3
Iteration 1: Retrieved 75 nodes
Gap analysis LLM call took 2.1s (model: gemini-flash-latest)
Iteration 1: Confidence=0.85, Total unique nodes=75
Confidence threshold reached (0.85 >= 0.8), stopping research
Research completed: 1 iterations, 75 unique nodes, confidence=0.85
```

### Exhaustive Query (@all prefix)
```
Exhaustive mode enabled: force=True, auto_detect=False, max_iterations=5
Starting research mode for query: all habits and routines
Research iteration 1/5
...
Waiting 5s before iteration 2 to avoid rate limiting...
Research iteration 2/5
Gap analysis LLM call took 1.8s (model: gemini-flash-latest)
Confidence threshold reached but exhaustive mode - continuing (iteration 2/5)
...
Maximum iterations reached (5)
Research completed: 5 iterations, 574 unique nodes, confidence=0.70
Research synthesis: 574 retrieved, 574 for synthesis, retry limits: [574, 459, 378, 300]
```

## User Output Format Override

All synthesis templates (PTCF, Research, Federated) now respect user-specified output formats:

```
**USER OVERRIDE**: If the user specifies a format in their query
(e.g., "as a table", "in Greek", "bullet points"), follow that
format instead of the default.
```

**Example queries:**
- `@all all habits - output as markdown table`
- `@all all habits - απάντησε στα ελληνικά`
- `what are my goals? - bullet points only`

## Troubleshooting

### Slow iterations (~5 minutes)

**Symptoms:** Gap analysis or sub-query generation takes ~5 minutes instead of seconds.

**Cause:** Gemini API rate limiting after multiple consecutive calls.

**Solution:** The dual-model architecture and inter-iteration delays should prevent this. If still occurring:
1. Increase `ITERATION_DELAY` to 10 seconds
2. Check if other processes are using the same API key

### MAX_TOKENS errors

**Symptoms:** `RuntimeError: Response was terminated early: MAX_TOKENS`

**Cause:** Too many nodes for synthesis, causing output to exceed model limits.

**Solution:** Progressive retry handles this automatically. If still occurring with 300 nodes:
1. Check if nodes have unusually large content
2. Consider reducing chunk size in indexing config

### Gap analysis using wrong model

**Symptoms:** Logs show `gemini-3-flash-preview` instead of `gemini-flash-latest` for gap analysis.

**Cause:** `GOOGLE_API_KEY` not set, falling back to main LLM.

**Solution:** Ensure `GOOGLE_API_KEY` is set in `.env`.

## Files Modified

- `research_mode.py` - Core research retriever with all enhancements
- `main.py` - Progressive retry logic, `@all` prefix parsing
- `query_engine.py` - User output format override in templates
- `federated_query.py` - User output format override in templates
- `CLAUDE.md` - Documentation updates

## Version History

- **2026-01-16**: Initial implementation
  - Exhaustive query mode with `@all` prefix
  - Dual-model architecture (gemini-flash-latest for gap analysis)
  - Progressive retry for MAX_TOKENS (100% → 80% → 66% → 300)
  - Inter-iteration delay (5s) for rate limiting mitigation
  - User output format override in templates
