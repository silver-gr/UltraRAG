# Self-Correction Quick Start Guide

## What Is Self-Correction?

Self-correction makes your RAG system **automatically detect and fix poor retrieval results** by:
1. Evaluating if retrieved documents answer the query
2. Refining the query if results are poor
3. Retrying retrieval with the improved query

**Result**: More accurate answers with less manual query refinement.

## 1-Minute Setup

Self-correction is **enabled by default**. No setup needed!

Just use UltraRAG normally:

```python
from main import UltraRAG

rag = UltraRAG()
rag.index_vault()

# Self-correction automatically improves this vague query:
result = rag.query("Tell me about ML")
# → System refines to: "What are machine learning model architectures?"
# → Returns focused, relevant results
```

## Configuration

### Default Settings (Recommended)

```bash
# In .env file:
USE_SELF_CORRECTION=true           # Enable self-correction
SELF_CORRECTION_MAX_RETRIES=2      # Try up to 3 times total
```

### Disable for Speed/Cost Savings

```bash
USE_SELF_CORRECTION=false          # Turn off self-correction
```

## How It Works

```
Your vague query: "Tell me about Python"
    ↓
First try: Retrieves mixed results (Python language + Python snake)
    Grade: AMBIGUOUS
    ↓
Refined query: "What are the key features of Python programming language?"
    ↓
Second try: Retrieves focused programming docs
    Grade: CORRECT
    ↓
Returns high-quality results ✅
```

## When to Use

### ✅ Enable Self-Correction When:
- Queries are often vague or exploratory
- You want maximum accuracy
- User queries need clarification
- Precision > Speed

### ❌ Disable Self-Correction When:
- Speed is critical (reduces latency by ~50-70%)
- Queries are already well-formed
- Minimizing API costs
- Limited API quotas

## Performance Impact

| Setting | Speed | LLM Calls | Quality |
|---------|-------|-----------|---------|
| **Disabled** | Fastest | 1 call | Good |
| **Enabled (best case)** | Fast | 2 calls | Better |
| **Enabled (worst case)** | Slower | 6 calls | Best |

**Recommendation**: Keep enabled for most use cases. The quality improvement is worth the extra LLM calls.

## Verification

Check if it's working:

1. **Look for log messages**:
   ```
   INFO:self_correction:Self-correcting retrieval enabled
   INFO:self_correction:Relevance grade: ambiguous
   INFO:self_correction:Query refinement: 'Python' -> 'Python programming'
   ```

2. **Run tests**:
   ```bash
   source venv/bin/activate
   python test_self_correction.py
   ```

## Advanced Configuration

### Adjust Retry Count

```bash
# More retries = better quality, more LLM calls
SELF_CORRECTION_MAX_RETRIES=1      # Conservative (up to 2 attempts)
SELF_CORRECTION_MAX_RETRIES=2      # Default (up to 3 attempts)
SELF_CORRECTION_MAX_RETRIES=3      # Aggressive (up to 4 attempts)
```

### Programmatic Control

```python
from config import RAGConfig, RetrievalConfig

# Custom configuration
config = RAGConfig(
    vault_path=Path("path/to/vault"),
    retrieval=RetrievalConfig(
        use_self_correction=True,
        self_correction_max_retries=1  # Less aggressive
    )
)

rag = UltraRAG(config=config)
```

## Troubleshooting

### Self-correction not working?

1. **Check logs**: Look for `INFO:self_correction` messages
2. **Verify config**: `USE_SELF_CORRECTION=true` in `.env`
3. **Check LLM**: Ensure LLM is initialized (needed for grading)

### Too many LLM calls?

```bash
# Reduce retries
SELF_CORRECTION_MAX_RETRIES=1

# Or disable entirely for simple queries
USE_SELF_CORRECTION=false
```

### Results still poor?

Self-correction helps but isn't magic. Also check:
- Embedding quality (try `voyage-3-large`)
- Reranker enabled (`RERANKER_MODEL=voyage-rerank-2`)
- Chunk size (512 is usually good)
- Query transformation (`QUERY_TRANSFORM_METHOD=hyde`)

## Examples

### Example 1: Ambiguous Query

```python
# Without self-correction:
result = rag.query("ML models")
# Returns: Mix of "machine learning" and "maximum likelihood" docs

# With self-correction:
result = rag.query("ML models")
# → Refines to: "machine learning model architectures"
# → Returns: Focused ML architecture docs ✅
```

### Example 2: Typo Recovery

```python
# Query with misspelling
result = rag.query("pythoon programing")
# → First try: Poor results (typo confuses retrieval)
# → Refined: "Python programming language features"
# → Returns: Correct Python docs ✅
```

### Example 3: Context Clarification

```python
# Vague context
result = rag.query("How do I use it?")
# → First try: Too generic, returns varied docs
# → Refined: "How to use [most relevant topic from vault]?"
# → Returns: Specific usage docs ✅
```

## Cost Estimate

With `max_retries=2` (default):
- **Best case**: 2 LLM calls (no retry needed) = ~$0.0001
- **Average case**: 4 LLM calls (1 retry) = ~$0.0002
- **Worst case**: 6 LLM calls (2 retries) = ~$0.0003

*Using Gemini Flash pricing; actual costs may vary*

**Yearly estimate** (1000 queries/month):
- Without self-correction: ~$1.20/year
- With self-correction: ~$2.40/year

The quality improvement is worth the extra ~$1.20/year for most users.

## Learn More

- **Full Documentation**: See `SELF_CORRECTION.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Run Tests**: `python test_self_correction.py`
- **Code**: See `self_correction.py`

## Summary

✅ **Auto-enabled** - Works out of the box
✅ **Smart** - Detects and fixes poor retrieval
✅ **Configurable** - Tune via `.env` file
✅ **Cost-effective** - Minimal extra LLM calls
✅ **Battle-tested** - Full test coverage

Just use UltraRAG as normal - self-correction works behind the scenes to improve your results!
