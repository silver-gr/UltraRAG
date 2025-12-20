#!/usr/bin/env python3
"""Quick test script to query the RAG system."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from main import UltraRAG

def main():
    print("🚀 Loading UltraRAG...")
    rag = UltraRAG()

    # Try to load existing index
    if rag.load_existing_index():
        print("✅ Index loaded successfully!\n")

        # Test queries
        test_queries = [
            "What are the main topics in my notes?",
            "Summarize my notes about productivity",  # Adjust based on your vault
        ]

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"🔍 Query: {query}")
            print('='*60)

            try:
                result = rag.query(query)
                print(f"\n📝 Answer:\n{result['answer'][:500]}...")
                print(f"\n📚 Sources ({len(result['sources'])} found):")
                for src in result['sources'][:3]:
                    print(f"  - {src['title']} (score: {src['score']:.3f})")
            except Exception as e:
                print(f"❌ Error: {e}")

            print()
    else:
        print("❌ No index found. Run indexing first.")

if __name__ == "__main__":
    main()
