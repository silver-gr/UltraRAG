# Self-Correcting RAG Implementation

This document describes the Self-RAG and CRAG (Corrective RAG) implementation in UltraRAG.

## Overview

The self-correction system implements two advanced RAG patterns:

### 1. Self-RAG (Self-Reflective RAG)
After retrieval, the LLM evaluates:
- Is the retrieved context relevant to the query?
- Is the generated response supported by the context?
- Should we re-retrieve with a modified query?

### 2. CRAG (Corrective RAG)
Grades retrieved documents as:
- **CORRECT**: Context directly answers the query with relevant information
- **AMBIGUOUS**: Context is partially relevant but incomplete or tangential
- **INCORRECT**: Context does not address the query or is irrelevant

If retrieval is AMBIGUOUS or INCORRECT, the system:
1. Reformulates the query based on what didn't work
2. Re-retrieves with the refined query
3. Repeats up to `max_retries` times

## Architecture

### Components

1. **`SelfCorrectingRetriever`**: Main retriever wrapper that implements the self-correction loop
2. **`SelfRAGValidator`**: Post-generation validator to check if responses are supported by context

### How It Works

```
User Query
    ↓
Base Retrieval (Vector/Hybrid/Graph)
    ↓
Relevance Grading (LLM)
    ├─→ CORRECT → Return Results
    └─→ AMBIGUOUS/INCORRECT → Query Refinement
            ↓
        Refined Query
            ↓
        Retry Retrieval (up to max_retries)
            ↓
        Return Best Results
```

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Enable/disable self-correction (default: true)
USE_SELF_CORRECTION=true

# Maximum retry attempts (default: 2)
SELF_CORRECTION_MAX_RETRIES=2
```

### Code Configuration

The settings are in `config.py` under `RetrievalConfig`:

```python
class RetrievalConfig(BaseModel):
    # Self-correction settings (Self-RAG and CRAG)
    use_self_correction: bool = Field(default=True)
    self_correction_max_retries: int = Field(default=2)
```

## Usage

### Basic Usage

Self-correction is **automatically enabled** by default when you query the system:

```python
from main import UltraRAG

# Initialize RAG system
rag = UltraRAG()
rag.index_vault()

# Query with self-correction enabled (default)
result = rag.query("What are the best practices for Python testing?")
print(result['answer'])
```

### Disabling Self-Correction

To disable self-correction:

1. **Via environment variable**:
   ```bash
   USE_SELF_CORRECTION=false
   ```

2. **Via code**:
   ```python
   from config import RAGConfig, RetrievalConfig

   config = RAGConfig(
       vault_path=Path("path/to/vault"),
       retrieval=RetrievalConfig(
           use_self_correction=False
       )
   )

   rag = UltraRAG(config=config)
   ```

### Adjusting Retry Count

```bash
# Try up to 3 times before returning results
SELF_CORRECTION_MAX_RETRIES=3
```

## How Self-Correction Improves Results

### Example Scenario

**Original Query**: "Tell me about ML models"

**First Retrieval** (AMBIGUOUS):
- Returns documents about "machine learning" and "maximum likelihood"
- Mixed relevance because "ML" is ambiguous

**Query Refinement**:
- LLM refines query to: "What are machine learning model architectures and training approaches?"

**Second Retrieval** (CORRECT):
- Returns focused documents about ML model architectures
- High relevance, query stops

**Benefit**: More precise results by clarifying ambiguous queries

## Integration Details

### Retriever Chain

The system wraps retrievers in layers:

```
Query
  ↓
Query Transformation (HyDE/Multi-Query) [optional]
  ↓
Self-Correction Wrapper [if enabled]
  ↓
Base Retriever (Vector/Hybrid/Graph)
  ↓
Reranker [if available]
  ↓
Similarity Filter
  ↓
Results
```

### In RAGQueryEngine

```python
# Standard query engine with self-correction
retriever = VectorIndexRetriever(...)

# Wrap with query transformation if enabled
if query_transformer:
    retriever = QueryTransformRetriever(...)

# Wrap with self-correction if enabled
if config.retrieval.use_self_correction:
    retriever = SelfCorrectingRetriever(
        base_retriever=retriever,
        llm=Settings.llm,
        max_retries=config.retrieval.self_correction_max_retries
    )
```

### In HybridQueryEngine

The hybrid engine (Vector + BM25 + Graph) also supports self-correction:

```python
# Fusion retriever (Vector + BM25)
fusion_retriever = QueryFusionRetriever(...)

# Graph enhancement
if graph_enabled:
    retriever = GraphEnhancedRetriever(...)

# Query transformation
if query_transformer:
    retriever = QueryTransformRetriever(...)

# Self-correction wrapper
if use_self_correction:
    retriever = SelfCorrectingRetriever(...)
```

## Logging

Enable debug logging to see self-correction in action:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('self_correction')
logger.setLevel(logging.INFO)
```

### Sample Log Output

```
INFO:self_correction:SelfCorrectingRetriever initialized (max_retries=2, enabled=True)
INFO:self_correction:Retrieval attempt 1/3 with query: 'Tell me about Python...'
INFO:self_correction:Relevance grade: ambiguous
INFO:self_correction:Query refinement (attempt 1): 'Tell me about Python' -> 'What are the key features and syntax of Python programming language?'
INFO:self_correction:Retrieval attempt 2/3 with query: 'What are the key features and syntax of Python programming language?...'
INFO:self_correction:Relevance grade: correct
INFO:self_correction:CORRECT grade achieved, returning results
INFO:self_correction:Self-correction complete after 2 attempts. Best grade: correct
```

## Performance Considerations

### LLM API Calls

- **Without self-correction**: 1 LLM call for response generation
- **With self-correction (best case)**: 2 LLM calls (1 grading + 1 generation)
- **With self-correction (worst case)**: 1 + (2 × max_retries) LLM calls
  - Example with max_retries=2: Up to 5 calls (2 gradings + 2 refinements + 1 generation)

### When to Use

**Enable self-correction when**:
- Queries are often ambiguous or vague
- Precision is more important than speed
- You want the system to self-improve retrieval quality
- Users ask exploratory questions

**Disable self-correction when**:
- Speed is critical
- Queries are already well-formed and specific
- You want to minimize LLM API costs
- Running on limited API quotas

### Cost Optimization

To reduce costs while keeping self-correction:

1. Use a smaller/cheaper LLM for grading and refinement:
   ```python
   # Use Gemini Flash for grading, Opus for generation
   from llama_index.llms.google_genai import GoogleGenAI

   grading_llm = GoogleGenAI(model="gemini-2.0-flash-exp")
   ```

2. Reduce max_retries:
   ```bash
   SELF_CORRECTION_MAX_RETRIES=1  # Only retry once
   ```

3. Use selective self-correction:
   - Enable for complex queries
   - Disable for simple lookups

## Validation with SelfRAGValidator

The `SelfRAGValidator` can validate generated responses:

```python
from self_correction import SelfRAGValidator

validator = SelfRAGValidator(llm=your_llm)

is_valid, explanation = validator.validate_response(
    query="What is Python?",
    response="Python is a high-level programming language...",
    context="Retrieved context about Python..."
)

if not is_valid:
    print(f"Response may contain hallucinations: {explanation}")
```

This is useful for:
- Detecting hallucinations
- Ensuring factual accuracy
- Building trust in RAG outputs

## Testing

Run the test suite:

```bash
source venv/bin/activate
python test_self_correction.py
```

Tests cover:
- Relevance grading (CORRECT, AMBIGUOUS, INCORRECT)
- Query refinement
- Response validation
- Disabled correction bypass
- Full retrieval loop

## Implementation Files

- **`self_correction.py`**: Core implementation
  - `SelfCorrectingRetriever`: Main retriever wrapper
  - `RelevanceGrade`: Enum for grading
  - `SelfRAGValidator`: Response validator

- **`config.py`**: Configuration settings
  - `RetrievalConfig.use_self_correction`
  - `RetrievalConfig.self_correction_max_retries`

- **`query_engine.py`**: Integration points
  - `RAGQueryEngine._build_query_engine()`
  - `HybridQueryEngine._build_query_engine()`

## References

- **Self-RAG**: Self-Reflective Retrieval-Augmented Generation
- **CRAG**: Corrective Retrieval Augmented Generation
- Both methods improve RAG quality through relevance assessment and query refinement

## Future Enhancements

Potential improvements:
1. **Web search fallback**: If all retries fail, fall back to web search
2. **Adaptive retry count**: Adjust max_retries based on query complexity
3. **Confidence scoring**: Return confidence scores with results
4. **A/B testing**: Compare self-correction vs. standard retrieval metrics
