# RAGAS Automated Evaluation

UltraRAG includes automated evaluation using the [RAGAS](https://github.com/explodinggradients/ragas) framework to measure RAG system quality objectively.

## Overview

The evaluation system computes four key metrics:

| Metric | What It Measures | Score Range |
|--------|------------------|-------------|
| **Faithfulness** | Answer grounded in context (no hallucination) | 0-1 (higher is better) |
| **Answer Relevancy** | Answer addresses the question | 0-1 (higher is better) |
| **Context Precision** | Retrieved docs are relevant | 0-1 (higher is better) |
| **Context Recall** | All needed docs were retrieved | 0-1 (higher is better) |

## Installation

The evaluation dependencies are included in `requirements.txt`:

```bash
pip install ragas>=0.1.0 datasets>=2.14.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Running Evaluation

```bash
# Run with default test dataset
python -m evaluation

# Run with custom dataset
python -m evaluation --dataset path/to/your/dataset.json

# Specify output directory
python -m evaluation --output ./my_results

# Set minimum score threshold (default: 0.5)
python -m evaluation --threshold 0.7

# Don't save report to file
python -m evaluation --no-save
```

### Programmatic Usage

```python
from evaluation import RAGEvaluator, load_test_cases
from main import UltraRAG

# Initialize RAG system
rag_system = UltraRAG()
rag_system.load_existing_index()

# Load test cases
test_cases = load_test_cases("tests/evaluation_dataset.json")

# Initialize evaluator
evaluator = RAGEvaluator(
    rag_system=rag_system,
    min_score_threshold=0.5
)

# Run evaluation
report = evaluator.evaluate(
    test_cases=test_cases,
    dataset_name="my_evaluation",
    save_report=True,
    output_dir="./data/evaluation"
)

# Print summary
report.print_summary()

# Access detailed results
print(report.detailed_results)
print(f"Overall score: {report.overall_score}")
print(f"Failed cases: {len(report.failed_cases)}")
```

## Test Dataset Format

Test datasets are JSON files with this structure:

```json
[
  {
    "question": "What is late chunking?",
    "ground_truth": "Late chunking computes embeddings at document level...",
    "expected_sources": ["RAG Architecture.md", "Chunking.md"],
    "metadata": {
      "category": "technical",
      "difficulty": "medium"
    }
  },
  {
    "question": "How does hybrid search work?",
    "ground_truth": "Hybrid search combines vector and keyword search...",
    "expected_sources": ["Retrieval Methods.md"],
    "metadata": {
      "category": "technical",
      "difficulty": "easy"
    }
  }
]
```

### Required Fields

- `question` (string): The question to ask the RAG system
- `ground_truth` (string): Reference answer for evaluation

### Optional Fields

- `expected_sources` (list): Expected source document names (for manual inspection)
- `metadata` (dict): Additional metadata for categorization/analysis

## Evaluation Reports

The evaluation system generates two output files:

### 1. JSON Report

`data/evaluation/{dataset_name}_report.json`:

```json
{
  "dataset_name": "evaluation_dataset",
  "total_cases": 10,
  "metrics": {
    "faithfulness": 0.8523,
    "answer_relevancy": 0.9012,
    "context_precision": 0.7845,
    "context_recall": 0.8234,
    "overall": 0.8403
  },
  "failed_cases_count": 2,
  "failed_cases": [
    {
      "question": "...",
      "ground_truth": "...",
      "answer": "...",
      "scores": {...}
    }
  ],
  "evaluation_config": {
    "min_score_threshold": 0.5,
    "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
  }
}
```

### 2. Detailed CSV

`data/evaluation/{dataset_name}_detailed.csv`:

Per-question scores for detailed analysis:

| question | faithfulness | answer_relevancy | context_precision | context_recall |
|----------|--------------|------------------|-------------------|----------------|
| What is...| 0.85 | 0.92 | 0.78 | 0.81 |
| How does...| 0.91 | 0.89 | 0.85 | 0.88 |

## Integration with CI/CD

Add to GitHub Actions workflow:

```yaml
name: RAG Evaluation

on:
  push:
    branches: [main]
  pull_request:
  release:
    types: [published]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Setup test vault
        run: |
          # Create or download test vault
          mkdir -p test_vault
          # Add test notes...

      - name: Index test vault
        run: |
          # Set up config
          export OBSIDIAN_VAULT_PATH=./test_vault
          export VOYAGE_API_KEY=${{ secrets.VOYAGE_API_KEY }}
          export GOOGLE_API_KEY=${{ secrets.GOOGLE_API_KEY }}
          # Create index
          python -c "from main import UltraRAG; r=UltraRAG(); r.index_vault(force_reindex=True)"

      - name: Run RAGAS evaluation
        run: |
          python -m evaluation --threshold 0.7

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-results
          path: data/evaluation/
```

## Best Practices

### Creating Test Datasets

1. **Diverse Questions**: Cover different query types (factual, conceptual, procedural)
2. **Difficulty Levels**: Mix easy, medium, and hard questions
3. **Ground Truth Quality**: Write accurate, comprehensive reference answers
4. **Real-World Queries**: Use questions users actually ask
5. **Coverage**: Test different topics/categories in your vault

### Interpreting Results

- **Faithfulness < 0.6**: System may be hallucinating - check retrieval quality
- **Answer Relevancy < 0.6**: Answers may be off-topic - check prompt engineering
- **Context Precision < 0.5**: Retrieving too much irrelevant content - tune similarity threshold
- **Context Recall < 0.5**: Missing relevant documents - increase top_k, check chunking

### Continuous Improvement

1. **Baseline**: Establish baseline scores with current configuration
2. **Experiment**: Try different chunking, retrieval, or prompt settings
3. **Compare**: Run evaluation after each change
4. **Track**: Monitor trends over time
5. **Regression Test**: Ensure changes don't degrade performance

## Troubleshooting

### "No module named 'ragas'"

Install evaluation dependencies:
```bash
pip install ragas>=0.1.0 datasets>=2.14.0
```

### "No existing index found"

Create an index first:
```bash
python main.py  # Interactive mode
# Or programmatically:
python -c "from main import UltraRAG; r=UltraRAG(); r.index_vault()"
```

### Evaluation is slow

- Reduce test dataset size for faster iterations
- Use a faster LLM model for evaluation (Gemini Flash instead of Pro)
- Each test case requires multiple LLM calls for RAGAS metrics

### Low scores

- **Check retrieval**: Are relevant documents being retrieved?
- **Improve chunking**: Try different chunking strategies
- **Tune parameters**: Adjust top_k, similarity threshold, reranker settings
- **Enhance prompts**: Improve system prompts for answer generation
- **Review test cases**: Ensure ground truth is accurate and realistic

## References

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [Khoj Evaluation Blog Post](https://blog.khoj.dev/posts/evaluate-khoj-quality/)
- [Google FRAMES Benchmark](https://github.com/google/frames-benchmark)
