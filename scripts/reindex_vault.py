#!/usr/bin/env python3
"""Re-index Obsidian vault with current embedding model (force overwrite)."""
import sys
sys.path.insert(0, ".")

from main import UltraRAG

print("Initializing UltraRAG...", flush=True)
rag = UltraRAG()
print("Starting vault re-index with force_reindex=True...", flush=True)
rag.index_vault(force_reindex=True)
print("\n✅ Vault re-index complete!", flush=True)
