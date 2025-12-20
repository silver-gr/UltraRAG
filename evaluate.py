"""Example evaluation script using RAGAS framework."""
import pandas as pd
from typing import List, Dict
from main import UltraRAG


def create_test_dataset() -> List[Dict]:
    """Create a test dataset for evaluation.
    
    You should customize this with actual queries from your vault.
    """
    test_queries = [
        {
            "query": "What are the key principles of effective note-taking?",
            "ground_truth": "Expected answer based on your notes..."
        },
        {
            "query": "Explain the concept of [your favorite topic]",
            "ground_truth": "Expected answer..."
        },
        # Add more test cases
    ]
    return test_queries


def evaluate_rag_system(rag: UltraRAG, test_dataset: List[Dict]) -> pd.DataFrame:
    """Evaluate RAG system using test queries."""
    
    results = []
    
    for item in test_dataset:
        query = item["query"]
        ground_truth = item.get("ground_truth", "")
        
        # Get RAG response
        response = rag.query(query, return_sources=True)
        
        # Collect contexts
        contexts = [node['excerpt'] for node in response['sources']]
        
        results.append({
            'question': query,
            'answer': response['answer'],
            'contexts': contexts,
            'ground_truth': ground_truth
        })
    
    return pd.DataFrame(results)


def calculate_ragas_metrics(results_df: pd.DataFrame):
    """Calculate RAGAS metrics.
    
    Requires: pip install ragas
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        )
        from datasets import Dataset
        
        # Convert to RAGAS dataset format
        dataset = Dataset.from_pandas(results_df)
        
        # Evaluate
        scores = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ]
        )
        
        return scores
        
    except ImportError:
        print("RAGAS not installed. Run: pip install ragas")
        return None


def main():
    """Run evaluation."""
    print("=== UltraRAG Evaluation ===\n")
    
    # Initialize RAG
    print("Initializing RAG system...")
    rag = UltraRAG()
    
    # Load or create test dataset
    print("Loading test dataset...")
    test_dataset = create_test_dataset()
    
    if not test_dataset:
        print("⚠️  No test dataset found!")
        print("Please edit evaluate.py and add your test queries.")
        return
    
    print(f"Testing with {len(test_dataset)} queries\n")
    
    # Run evaluation
    print("Running evaluation...")
    results_df = evaluate_rag_system(rag, test_dataset)
    
    # Calculate metrics
    print("\nCalculating RAGAS metrics...")
    scores = calculate_ragas_metrics(results_df)
    
    if scores:
        print("\n📊 Evaluation Results:")
        print("-" * 50)
        for metric, score in scores.items():
            print(f"{metric}: {score:.3f}")
    
    # Save results
    output_file = "evaluation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()
