#!/usr/bin/env python3
"""Re-index conversations with current embedding model (force overwrite)."""
import sys
sys.path.insert(0, ".")

from main import UltraRAG

print("Initializing UltraRAG...", flush=True)
rag = UltraRAG()
print("Starting conversations re-index with force_reindex=True...", flush=True)
rag.index_conversations(force_reindex=True, interactive=False)
print("\n✅ Conversations re-index complete!", flush=True)
