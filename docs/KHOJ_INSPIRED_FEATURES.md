# Khoj-Inspired Features for UltraRAG

Based on analysis of [Khoj AI](https://github.com/khoj-ai/khoj) - a 32K-star open-source AI assistant.

## Feature 1: Iterative Retrieval Mode (Research Mode) ⭐⭐⭐

### What Khoj Does
- `/research` command triggers multi-step retrieval
- Instead of single-shot retrieval, iteratively refines queries based on initial results
- **Benchmark results**: 141% accuracy improvement (27% → 63.5% on FRAMES benchmark)
- Metaphor: "Take-home exam" vs "open book exam"

### Implementation Plan for UltraRAG

#### New File: `research_mode.py`
```python
class ResearchRetriever:
    """Iterative retrieval with query refinement based on initial results."""

    def __init__(self, base_retriever, llm, max_iterations=3):
        self.base_retriever = base_retriever
        self.llm = llm
        self.max_iterations = max_iterations

    def research(self, query: str) -> ResearchResult:
        """
        Multi-step research process:
        1. Initial retrieval
        2. LLM analyzes gaps in retrieved content
        3. Generate refined sub-queries for missing info
        4. Retrieve again with sub-queries
        5. Repeat until satisfied or max iterations
        6. Synthesize final answer from all retrieved content
        """
        pass
```

#### Key Components
1. **Gap Analysis**: LLM identifies what's missing from initial retrieval
2. **Sub-Query Generation**: Create targeted queries for gaps
3. **Result Aggregation**: Combine results across iterations, deduplicate
4. **Confidence Scoring**: Track when enough information has been gathered
5. **Citation Tracking**: Track which iteration found each piece of info

#### CLI/Web Integration
- CLI: `@research <query>` prefix (like `@vault`, `@conv`)
- Web: Add "Research Mode" toggle or button
- Longer timeout (research takes 30-60s vs 2-5s normal)

#### Trade-offs
- **Pros**: Much higher accuracy for complex questions
- **Cons**: 3-5x slower, 3-5x more LLM API calls, higher cost

---

## Feature 2: Automated Evaluation (RAGAS) ⭐⭐⭐

### What Khoj Does
- Runs FRAMES and SimpleQA benchmarks on every GitHub release
- Automated evaluation harness measures retrieval quality objectively
- Catches regressions before they ship

### Implementation Plan for UltraRAG

#### New File: `evaluation.py`
```python
class RAGEvaluator:
    """Automated evaluation using RAGAS metrics."""

    def evaluate(self, test_cases: List[TestCase]) -> EvaluationReport:
        """
        Metrics to compute:
        - Faithfulness: Is the answer grounded in retrieved context?
        - Answer Relevancy: Does the answer address the question?
        - Context Precision: Are retrieved docs relevant?
        - Context Recall: Did we retrieve all needed docs?
        """
        pass
```

#### RAGAS Metrics
| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Answer grounded in context (no hallucination) |
| **Answer Relevancy** | Answer addresses the question |
| **Context Precision** | Retrieved docs are relevant |
| **Context Recall** | All needed docs were retrieved |

#### Test Dataset
Create `tests/evaluation_dataset.json`:
```json
[
  {
    "question": "What are my notes about machine learning?",
    "ground_truth": "...",
    "expected_sources": ["ML Notes.md", "AI Research.md"]
  }
]
```

#### CLI Command
```bash
python -m ultrarag.evaluate --dataset tests/evaluation_dataset.json
```

#### GitHub Actions Integration
```yaml
# .github/workflows/evaluate.yml
on:
  release:
    types: [published]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run RAGAS evaluation
        run: python -m ultrarag.evaluate
```

#### Dependencies
```
ragas>=0.1.0
datasets>=2.14.0
```

---

## Implementation Priority

| Feature | Effort | Impact | Priority |
|---------|--------|--------|----------|
| Research Mode | Medium | High (141% accuracy boost) | 1 |
| RAGAS Evaluation | Low-Medium | High (quality assurance) | 2 |

## References

- [Khoj Research Mode Blog](https://blog.khoj.dev/posts/evaluate-khoj-quality/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Google FRAMES Benchmark](https://github.com/google/frames-benchmark)
- [OpenAI SimpleQA](https://openai.com/index/introducing-simpleqa/)
