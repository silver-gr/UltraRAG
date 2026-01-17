# Query Intent Classification (Future Feature)

> **Status:** Planned for future implementation
> **Complexity:** High (~200+ lines)
> **Dependencies:** Additional LLM call per query

## Overview

LLM-based query classification to automatically detect query intent and dynamically adjust research parameters.

## Query Intent Types

| Intent | Description | Parameters |
|--------|-------------|------------|
| **Exhaustive** | "List ALL X", "Every Y", "Complete list" | max_iterations=5, no early stopping |
| **Focused** | Specific question, single topic | max_iterations=2, normal confidence |
| **Exploratory** | Open-ended, discovery | max_iterations=3, lower confidence threshold |
| **Comparative** | "X vs Y", "Differences between" | Parallel retrieval, comparison prompt |

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────>│ QueryClassifier  │────>│ ResearchConfig  │
└─────────────────┘     │   (LLM call)     │     │  (dynamic)      │
                        └──────────────────┘     └─────────────────┘
                                                          │
                               ┌──────────────────────────┘
                               ▼
                        ┌─────────────────┐
                        │ ResearchRetriever│
                        │ (adjusted params)│
                        └─────────────────┘
```

## Proposed Implementation

### New File: `query_classifier.py`

```python
from enum import Enum
from dataclasses import dataclass
from llama_index.core.llms import LLM

class QueryIntent(Enum):
    EXHAUSTIVE = "exhaustive"
    FOCUSED = "focused"
    EXPLORATORY = "exploratory"
    COMPARATIVE = "comparative"

@dataclass
class QueryAnalysis:
    intent: QueryIntent
    confidence: float
    suggested_iterations: int
    suggested_top_k: int
    reasoning: str

class QueryClassifier:
    """LLM-based query intent classification."""

    CLASSIFICATION_PROMPT = '''
    Analyze this query and classify its intent:

    Query: {query}

    Intent types:
    - EXHAUSTIVE: User wants ALL/EVERY matching items (comprehensive list)
    - FOCUSED: Specific question about a single topic
    - EXPLORATORY: Open-ended discovery, learning about a topic
    - COMPARATIVE: Comparing multiple items or concepts

    Respond with:
    INTENT: [type]
    CONFIDENCE: [0.0-1.0]
    ITERATIONS: [1-5 recommended]
    TOP_K: [50-500 recommended]
    REASONING: [brief explanation]
    '''

    def __init__(self, llm: LLM):
        self.llm = llm

    def classify(self, query: str) -> QueryAnalysis:
        """Classify query intent using LLM."""
        prompt = self.CLASSIFICATION_PROMPT.format(query=query)
        response = self.llm.complete(prompt)
        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> QueryAnalysis:
        # Parse LLM response into QueryAnalysis
        ...
```

### Integration with ResearchRetriever

```python
class ResearchRetriever:
    def __init__(self, ..., classifier: QueryClassifier = None):
        self.classifier = classifier

    def research(self, query: str, ...):
        # Auto-classify if classifier available
        if self.classifier and not force_exhaustive:
            analysis = self.classifier.classify(query)
            if analysis.intent == QueryIntent.EXHAUSTIVE:
                force_exhaustive = True
                self.max_iterations = analysis.suggested_iterations
```

## Trade-offs

### Pros
- Intelligent parameter adjustment
- No manual prefixes needed
- Could improve all query types
- Extensible to other optimizations

### Cons
- Additional LLM call (~500ms latency)
- Extra API cost per query
- Potential classification errors
- More complex debugging

## Configuration

```bash
# .env options
ENABLE_QUERY_CLASSIFICATION=true
QUERY_CLASSIFIER_MODEL=gemini-3-flash-preview  # Fast, cheap model
QUERY_CLASSIFIER_CACHE_TTL=3600  # Cache similar queries
```

## Alternatives Considered

1. **Rule-based detection** (current implementation)
   - Pros: No latency, no cost, deterministic
   - Cons: Limited patterns, false negatives

2. **Embedding-based classification**
   - Pros: No LLM call, fast
   - Cons: Requires training data, less accurate

3. **Hybrid approach**
   - Rule-based first, LLM fallback for ambiguous queries
   - Best of both worlds but more complex

## When to Implement

Consider implementing when:
- Users frequently need different research modes
- Rule-based detection shows limitations
- Latency budget allows extra LLM call
- Want to optimize other parameters (top_k, synthesis limits)

## Related

- `research_mode.py` - Current research implementation
- `docs/features/RESEARCH_MODE.md` - Research mode documentation (if exists)
