# LangSmith Integration Guide for UltraRAG

This guide explains how to enable LangSmith observability in UltraRAG.

## Quick Start

### 1. Install LangSmith

```bash
pip install langsmith
```

### 2. Set Environment Variables

```bash
# Required
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-api-key-here"

# Optional (defaults to "ultrarag")
export LANGSMITH_PROJECT="ultrarag-production"
```

Or add to your `.env` file:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxxx
LANGSMITH_PROJECT=ultrarag
```

### 3. Verify Setup

```python
from observability import get_tracing_status

status = get_tracing_status()
print(status)
# {'langsmith_available': True, 'tracing_enabled': True, 'project': 'ultrarag', 'api_key_set': True}
```

## Integration Examples

### Basic Function Tracing

```python
from observability import trace_retrieval, trace_generation, trace_chain

# Trace retrieval operations
@trace_retrieval
def my_retrieval_function(query: str):
    # Your retrieval logic
    return results

# Trace LLM generation
@trace_generation
def my_generation_function(context: str, query: str):
    # Your generation logic
    return response

# Trace full pipeline
@trace_chain
def my_rag_pipeline(query: str):
    docs = my_retrieval_function(query)
    response = my_generation_function(docs, query)
    return response
```

### Using the Tracer Class

```python
from observability import get_tracer

tracer = get_tracer("my-project")

# Decorate existing functions
@tracer.trace_retrieval
def retrieve_documents(query):
    ...

@tracer.trace_generation
def generate_response(context, query):
    ...

@tracer.trace_research_mode
def research_iteration(query, iteration):
    ...
```

### Context Manager for Custom Spans

```python
from observability import trace_span

def complex_operation():
    with trace_span("embedding-generation", run_type="tool"):
        embeddings = generate_embeddings(texts)

    with trace_span("vector-search", run_type="retriever"):
        results = vector_store.search(embeddings)

    with trace_span("reranking", run_type="tool"):
        reranked = reranker.rerank(results)

    return reranked
```

## Integrating with UltraRAG Components

### query_engine.py Integration

Add tracing to the main query methods:

```python
# At the top of query_engine.py
from observability import trace_chain, trace_retrieval, is_tracing_enabled

# In RAGQueryEngine class
class RAGQueryEngine:
    @trace_chain
    def query(self, query_str: str, use_cache: Optional[bool] = None, **kwargs):
        """Execute query with tracing."""
        # existing implementation
        ...
```

### research_mode.py Integration

```python
from observability import trace_chain, trace_span

class ResearchMode:
    @trace_chain
    def execute(self, query: str, max_iterations: int = 3):
        """Execute research mode with full tracing."""
        for i in range(max_iterations):
            with trace_span(f"research-iteration-{i+1}", metadata={"iteration": i+1}):
                # iteration logic
                ...
```

### self_correction.py Integration

```python
from observability import trace_chain, trace_span

class SelfCorrectingRetriever:
    @trace_chain
    def retrieve(self, query_bundle):
        """Retrieve with self-correction tracing."""
        with trace_span("initial-retrieval", run_type="retriever"):
            nodes = self.base_retriever.retrieve(query_bundle)

        with trace_span("relevance-grading", run_type="tool"):
            grade = self._grade_relevance(nodes, query_bundle)

        if grade != "CORRECT":
            with trace_span("query-refinement", run_type="tool"):
                refined_query = self._refine_query(query_bundle, grade)
            # retry...
```

## Viewing Traces

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Select your project (default: "ultrarag")
3. View the trace hierarchy:
   - **Chains**: Full RAG pipeline, research iterations
   - **Retrievers**: Vector search, hybrid retrieval
   - **LLMs**: Generation calls
   - **Tools**: Embeddings, reranking, query transformation

## Logging Feedback

```python
from observability import get_tracer

tracer = get_tracer()

# After a query, log user feedback
tracer.log_feedback(
    run_id="run-uuid-here",
    key="correctness",
    score=1.0,  # 0-1
    comment="Answer was accurate and helpful"
)
```

## Debugging with Recent Runs

```python
from observability import get_tracer

tracer = get_tracer()

# Get recent runs
runs = tracer.get_recent_runs(limit=10)
for run in runs:
    print(f"{run['name']}: {run['status']} ({run['total_tokens']} tokens)")

# Get only errors
errors = tracer.get_recent_runs(limit=5, error_only=True)
```

## Best Practices

1. **Use Descriptive Names**: Name your spans clearly for easy debugging
2. **Add Metadata**: Include relevant context (query, iteration count, etc.)
3. **Trace Boundaries**: Trace at logical boundaries (retrieval, generation, not every function)
4. **Production Mode**: Set `LANGSMITH_TRACING=false` to disable in production if needed
5. **Cost Awareness**: Tracing adds minimal overhead but logs data to LangSmith servers

## Troubleshooting

### Tracing not appearing in dashboard

1. Check environment variables are set correctly
2. Verify API key is valid
3. Check network connectivity to LangSmith servers
4. Look for errors in logs

### High latency

LangSmith tracing is async by default. If you see latency:

```python
# Force synchronous tracing (for debugging)
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"
```

### Missing traces in serverless

Ensure traces are flushed before the process exits:

```python
from langsmith import Client

client = Client()
# ... your code ...

# At the end
client.flush()
```
