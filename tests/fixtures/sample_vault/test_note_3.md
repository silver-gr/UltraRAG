---
title: RAG Implementation
tags: [code, python, rag]
created: 2024-01-17
---

# RAG Implementation

## Python Example

```python
from llama_index import VectorStoreIndex
index = VectorStoreIndex.from_documents(docs)
engine = index.as_query_engine()
response = engine.query("What is RAG?")
```

Links back to: [[test_note_1]], [[test_note_2]]
