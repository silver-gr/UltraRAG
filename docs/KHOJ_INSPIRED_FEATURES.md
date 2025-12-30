# Khoj-Inspired Features for UltraRAG

Based on analysis of [Khoj AI](https://github.com/khoj-ai/khoj) - a 32K-star open-source AI assistant.

## Feature 1: Iterative Retrieval Mode (Research Mode) ✅ IMPLEMENTED

### What Khoj Does
- `/research` command triggers multi-step retrieval
- Instead of single-shot retrieval, iteratively refines queries based on initial results
- **Benchmark results**: 141% accuracy improvement (27% → 63.5% on FRAMES benchmark)
- Metaphor: "Take-home exam" vs "open book exam"

### UltraRAG Implementation

#### File: `research_mode.py`
```python
class ResearchRetriever:
    """Iterative retrieval with query refinement based on initial results."""

    def __init__(self, base_retriever, llm, max_iterations=3, confidence_threshold=0.8):
        self.base_retriever = base_retriever
        self.llm = llm
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

    def research(self, query: str) -> ResearchResult:
        """
        Multi-step research process:
        1. Initial retrieval
        2. LLM analyzes gaps in retrieved content
        3. Generate refined sub-queries for missing info
        4. Retrieve again with sub-queries
        5. Repeat until satisfied or max iterations
        6. Return aggregated, deduplicated results
        """
```

#### Key Components
1. **Gap Analysis**: LLM identifies what's missing from initial retrieval
2. **Sub-Query Generation**: Create targeted queries for gaps
3. **Result Aggregation**: Combine results across iterations, deduplicate
4. **Confidence Scoring**: Track when enough information has been gathered
5. **Iteration Tracking**: Summary of what each iteration found

#### Usage
- **CLI**: `@research <query>` prefix
- **Web**: 🔬 Research checkbox (warns about 30-60s duration)
- **Config**: `research_max_iterations=3`, `research_confidence_threshold=0.8`

#### Trade-offs
- **Pros**: Much higher accuracy for complex questions
- **Cons**: 3-5x slower, 3-5x more LLM API calls, higher cost

---

## Feature 2: Automated Evaluation (RAGAS) ✅ IMPLEMENTED

### What Khoj Does
- Runs FRAMES and SimpleQA benchmarks on every GitHub release
- Automated evaluation harness measures retrieval quality objectively
- Catches regressions before they ship

### UltraRAG Implementation

#### File: `evaluation.py`
```python
class RAGEvaluator:
    """Automated evaluation using RAGAS metrics."""

    def evaluate(self, test_cases: List[TestCase]) -> EvaluationReport:
        """
        Metrics computed:
        - Faithfulness: Is the answer grounded in retrieved context?
        - Answer Relevancy: Does the answer address the question?
        - Context Precision: Are retrieved docs relevant?
        - Context Recall: Did we retrieve all needed docs?
        """
```

#### RAGAS Metrics
| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Answer grounded in context (no hallucination) |
| **Answer Relevancy** | Answer addresses the question |
| **Context Precision** | Retrieved docs are relevant |
| **Context Recall** | All needed docs were retrieved |

#### Test Dataset
Located at `tests/evaluation_dataset.json` with 10 sample test cases.

#### Usage
```bash
# Run evaluation
python -m evaluation --dataset tests/evaluation_dataset.json

# Output: JSON report + CSV with per-question scores
```

#### Dependencies (added to requirements.txt)
```
ragas>=0.1.0
datasets>=2.14.0
```

---

## Implementation Status

| Feature | Status | Files |
|---------|--------|-------|
| Research Mode | ✅ Done | `research_mode.py`, `main.py`, `app.py`, `config.py` |
| RAGAS Evaluation | ✅ Done | `evaluation.py`, `tests/evaluation_dataset.json`, `docs/EVALUATION.md` |

## References

- [Khoj Research Mode Blog](https://blog.khoj.dev/posts/evaluate-khoj-quality/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Google FRAMES Benchmark](https://github.com/google/frames-benchmark)
- [OpenAI SimpleQA](https://openai.com/index/introducing-simpleqa/)
