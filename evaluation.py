"""Evaluation framework for UltraRAG system.

Two evaluation modes:
1. Lightweight per-query metrics (RetrievalEvaluator) -- enable via ENABLE_EVALUATION_LOGGING=true
   Logs precision@k, recall@k, MRR, NDCG, score stats to data/evaluation_log.json
2. Full RAGAS benchmark (RAGEvaluator) -- run via `python -m evaluation --dataset ...`
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ============================================
# Lightweight per-query evaluation (Feature 11)
# ============================================

EVAL_LOG_PATH = Path("data/evaluation_log.json")


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics for a single query."""
    query: str
    timestamp: float
    num_retrieved: int
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    mrr: Optional[float] = None
    ndcg: Optional[float] = None
    avg_score: Optional[float] = None
    score_variance: Optional[float] = None
    unique_sources: int = 0
    # Latency breakdown (ms)
    engine_build_ms: Optional[float] = None
    query_ms: Optional[float] = None
    total_ms: Optional[float] = None


class RetrievalEvaluator:
    """Lightweight retrieval quality evaluator with standard IR metrics.

    Computes metrics per query and optionally logs to disk. No LLM calls
    unless evaluate_response() is called explicitly.
    """

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_nodes: list,
        relevant_docs: Optional[List[str]] = None,
        k: int = 10,
        timings: Optional[Dict[str, float]] = None,
    ) -> RetrievalMetrics:
        """Compute retrieval metrics.

        Args:
            query: User query
            retrieved_nodes: NodeWithScore objects from retriever
            relevant_docs: Optional list of known-relevant doc paths (for precision/recall)
            k: Cutoff for @k metrics
        """
        top_k_nodes = retrieved_nodes[:k]
        scores = [n.score or 0.0 for n in top_k_nodes]

        avg_score = sum(scores) / len(scores) if scores else 0.0
        score_variance = (
            sum((s - avg_score) ** 2 for s in scores) / len(scores)
            if scores else 0.0
        )

        sources = set()
        for n in top_k_nodes:
            fp = n.node.metadata.get("file_path", n.node.node_id)
            sources.add(fp)

        metrics = RetrievalMetrics(
            query=query,
            timestamp=time.time(),
            num_retrieved=len(retrieved_nodes),
            avg_score=round(avg_score, 4),
            score_variance=round(score_variance, 4),
            unique_sources=len(sources),
            engine_build_ms=timings.get("engine_build_ms") if timings else None,
            query_ms=timings.get("query_ms") if timings else None,
            total_ms=timings.get("total_ms") if timings else None,
        )

        if relevant_docs:
            retrieved_ids = [
                n.node.metadata.get("file_path", n.node.node_id)
                for n in top_k_nodes
            ]
            relevant_set = set(relevant_docs)

            hits = sum(1 for rid in retrieved_ids if rid in relevant_set)
            metrics.precision_at_k = round(hits / k, 4) if k > 0 else 0.0
            metrics.recall_at_k = round(hits / len(relevant_set), 4) if relevant_set else 0.0

            # MRR
            for rank, rid in enumerate(retrieved_ids, 1):
                if rid in relevant_set:
                    metrics.mrr = round(1.0 / rank, 4)
                    break
            else:
                metrics.mrr = 0.0

            # NDCG@k
            def dcg(relevances: List[float]) -> float:
                return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

            rels = [1.0 if rid in relevant_set else 0.0 for rid in retrieved_ids]
            ideal_rels = sorted(rels, reverse=True)
            dcg_val = dcg(rels)
            idcg_val = dcg(ideal_rels)
            metrics.ndcg = round(dcg_val / idcg_val, 4) if idcg_val > 0 else 0.0

        return metrics

    @staticmethod
    def log_metrics(metrics: RetrievalMetrics) -> None:
        """Append metrics to evaluation log file."""
        try:
            EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            entries = []
            if EVAL_LOG_PATH.exists():
                with open(EVAL_LOG_PATH) as f:
                    entries = json.load(f)
            entries.append(asdict(metrics))
            # Keep last 1000 entries
            if len(entries) > 1000:
                entries = entries[-1000:]
            with open(EVAL_LOG_PATH, "w") as f:
                json.dump(entries, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to log evaluation metrics: {e}")

    @staticmethod
    def get_summary(last_n: int = 100) -> Dict[str, Any]:
        """Get summary statistics from evaluation log."""
        if not EVAL_LOG_PATH.exists():
            return {"entries": 0}
        with open(EVAL_LOG_PATH) as f:
            entries = json.load(f)
        recent = entries[-last_n:]
        if not recent:
            return {"entries": 0}

        def avg(key):
            vals = [e[key] for e in recent if e.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        return {
            "entries": len(recent),
            "avg_score": avg("avg_score"),
            "avg_unique_sources": avg("unique_sources"),
            "avg_precision_at_k": avg("precision_at_k"),
            "avg_mrr": avg("mrr"),
            "avg_ndcg": avg("ndcg"),
        }


# ============================================
# Full RAGAS benchmark evaluation
# ============================================

try:
    import pandas as pd
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness as ragas_faithfulness,
        answer_relevancy as ragas_answer_relevancy,
        context_precision as ragas_context_precision,
        context_recall as ragas_context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.debug("RAGAS not available -- full benchmark evaluation disabled")


@dataclass
class TestCase:
    """Single test case for RAG evaluation.

    Attributes:
        question: The question to ask the RAG system
        ground_truth: Expected/reference answer
        expected_sources: Optional list of expected source document titles
        metadata: Optional additional metadata for the test case
    """
    question: str
    ground_truth: str
    expected_sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report with RAGAS metrics.

    Attributes:
        dataset_name: Name of the evaluation dataset
        total_cases: Total number of test cases evaluated
        faithfulness_score: Average faithfulness score (answer grounded in context)
        answer_relevancy_score: Average answer relevancy score
        context_precision_score: Average context precision score
        context_recall_score: Average context recall score
        overall_score: Weighted average of all metrics
        detailed_results: DataFrame with per-question results
        failed_cases: List of test cases that failed evaluation
        evaluation_config: Configuration used for evaluation
    """
    dataset_name: str
    total_cases: int
    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: float
    context_recall_score: float
    overall_score: float
    detailed_results: pd.DataFrame
    failed_cases: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "dataset_name": self.dataset_name,
            "total_cases": self.total_cases,
            "metrics": {
                "faithfulness": self.faithfulness_score,
                "answer_relevancy": self.answer_relevancy_score,
                "context_precision": self.context_precision_score,
                "context_recall": self.context_recall_score,
                "overall": self.overall_score,
            },
            "failed_cases_count": len(self.failed_cases),
            "failed_cases": self.failed_cases,
            "evaluation_config": self.evaluation_config,
        }

    def print_summary(self):
        """Print human-readable summary of evaluation results."""
        print("\n" + "="*70)
        print(f"RAGAS EVALUATION REPORT: {self.dataset_name}")
        print("="*70)
        print(f"\nTotal Test Cases: {self.total_cases}")
        print("\nMetric Scores (0-1, higher is better):")
        print(f"  Faithfulness:      {self.faithfulness_score:.4f}  (Answer grounded in context)")
        print(f"  Answer Relevancy:  {self.answer_relevancy_score:.4f}  (Answer addresses question)")
        print(f"  Context Precision: {self.context_precision_score:.4f}  (Retrieved docs are relevant)")
        print(f"  Context Recall:    {self.context_recall_score:.4f}  (All needed docs retrieved)")
        print(f"\n  Overall Score:     {self.overall_score:.4f}")

        if self.failed_cases:
            print(f"\n⚠️  Failed Cases: {len(self.failed_cases)}")
            for i, case in enumerate(self.failed_cases[:3], 1):
                print(f"  {i}. {case['question'][:60]}...")
        else:
            print("\n✅ All test cases passed!")

        print("="*70 + "\n")

    def save_to_file(self, output_path: Path):
        """Save evaluation report to JSON file."""
        report_dict = self.to_dict()
        # Convert DataFrame to dict for JSON serialization
        report_dict["detailed_results"] = self.detailed_results.to_dict(orient="records")

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Evaluation report saved to: {output_path}")


class RAGEvaluator:
    """Automated evaluation using RAGAS metrics.

    Requires: pip install ragas datasets pandas

    - Faithfulness: Is the answer grounded in retrieved context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are retrieved documents relevant?
    - Context Recall: Did we retrieve all needed documents?
    """

    def __init__(
        self,
        rag_system: Any,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
        min_score_threshold: float = 0.5,
    ):
        """Initialize RAG evaluator.

        Args:
            rag_system: The RAG system to evaluate (must have a query() method)
            llm: Optional LLM to use for evaluation (defaults to rag_system's LLM)
            embeddings: Optional embedding model for evaluation
            min_score_threshold: Minimum acceptable score for metrics (0-1)
        """
        self.rag_system = rag_system
        self.llm = llm or getattr(rag_system, 'llm', None)
        self.embeddings = embeddings or getattr(rag_system, 'embed_model', None)
        self.min_score_threshold = min_score_threshold

        logger.info("RAGEvaluator initialized")
        logger.info(f"Minimum score threshold: {min_score_threshold}")

    def _generate_rag_responses(self, test_cases: List[TestCase]) -> Dict[str, List]:
        """Generate RAG responses for all test cases.

        Args:
            test_cases: List of test cases to evaluate

        Returns:
            Dictionary with questions, answers, contexts, and ground truths
        """
        logger.info(f"Generating RAG responses for {len(test_cases)} test cases...")

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for i, test_case in enumerate(test_cases, 1):
            try:
                print(f"  [{i}/{len(test_cases)}] Processing: {test_case.question[:60]}...")

                # Query the RAG system
                result = self.rag_system.query(test_case.question, return_sources=True)

                # Extract answer and contexts
                answer = result['answer']
                source_contexts = [
                    source['excerpt'] for source in result['sources']
                ]

                questions.append(test_case.question)
                answers.append(answer)
                contexts.append(source_contexts)
                ground_truths.append(test_case.ground_truth)

            except Exception as e:
                logger.error(f"Failed to generate response for question: {test_case.question}")
                logger.error(f"Error: {e}", exc_info=True)
                # Add placeholder to maintain alignment
                questions.append(test_case.question)
                answers.append("ERROR: Failed to generate answer")
                contexts.append(["ERROR: No context retrieved"])
                ground_truths.append(test_case.ground_truth)

        return {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

    def evaluate(
        self,
        test_cases: List[TestCase],
        dataset_name: str = "evaluation",
        save_report: bool = True,
        output_dir: Path = Path("./data/evaluation"),
    ) -> EvaluationReport:
        """Evaluate RAG system on test cases using RAGAS metrics.

        Args:
            test_cases: List of test cases to evaluate
            dataset_name: Name for this evaluation dataset
            save_report: Whether to save the report to file
            output_dir: Directory to save evaluation reports

        Returns:
            EvaluationReport with comprehensive metrics and results
        """
        if not test_cases:
            raise ValueError("No test cases provided for evaluation")

        logger.info(f"Starting RAGAS evaluation: {dataset_name}")
        print(f"\n{'='*70}")
        print(f"RAGAS EVALUATION: {dataset_name}")
        print(f"{'='*70}")
        print(f"Test Cases: {len(test_cases)}")
        print(f"Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")
        print(f"{'='*70}\n")

        # Generate RAG responses
        dataset_dict = self._generate_rag_responses(test_cases)

        # Create RAGAS dataset
        print("\nCreating evaluation dataset...")
        ragas_dataset = Dataset.from_dict(dataset_dict)

        # Run RAGAS evaluation
        print("\nRunning RAGAS evaluation (this may take a few minutes)...")
        if not RAGAS_AVAILABLE:
            raise RuntimeError("RAGAS not installed. Install with: pip install ragas datasets pandas")
        try:
            evaluation_result = ragas_evaluate(
                dataset=ragas_dataset,
                metrics=[
                    ragas_faithfulness,
                    ragas_answer_relevancy,
                    ragas_context_precision,
                    ragas_context_recall,
                ],
            )
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            raise RuntimeError(f"RAGAS evaluation failed: {e}") from e

        # Extract scores
        scores_df = pd.DataFrame(evaluation_result.scores)

        # Calculate average scores
        faithfulness_avg = scores_df['faithfulness'].mean()
        answer_relevancy_avg = scores_df['answer_relevancy'].mean()
        context_precision_avg = scores_df['context_precision'].mean()
        context_recall_avg = scores_df['context_recall'].mean()

        # Calculate overall score (weighted average)
        overall_score = (
            faithfulness_avg * 0.3 +
            answer_relevancy_avg * 0.3 +
            context_precision_avg * 0.2 +
            context_recall_avg * 0.2
        )

        # Identify failed cases (below threshold)
        failed_cases = []
        for idx, row in scores_df.iterrows():
            min_metric_score = min([
                row['faithfulness'],
                row['answer_relevancy'],
                row['context_precision'],
                row['context_recall'],
            ])

            if min_metric_score < self.min_score_threshold:
                failed_cases.append({
                    "question": dataset_dict["question"][idx],
                    "ground_truth": dataset_dict["ground_truth"][idx],
                    "answer": dataset_dict["answer"][idx],
                    "scores": {
                        "faithfulness": float(row['faithfulness']),
                        "answer_relevancy": float(row['answer_relevancy']),
                        "context_precision": float(row['context_precision']),
                        "context_recall": float(row['context_recall']),
                    }
                })

        # Create evaluation report
        report = EvaluationReport(
            dataset_name=dataset_name,
            total_cases=len(test_cases),
            faithfulness_score=float(faithfulness_avg),
            answer_relevancy_score=float(answer_relevancy_avg),
            context_precision_score=float(context_precision_avg),
            context_recall_score=float(context_recall_avg),
            overall_score=float(overall_score),
            detailed_results=scores_df,
            failed_cases=failed_cases,
            evaluation_config={
                "min_score_threshold": self.min_score_threshold,
                "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
            }
        )

        # Save report to file
        if save_report:
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / f"{dataset_name}_report.json"
            report.save_to_file(report_path)

            # Also save detailed CSV
            csv_path = output_dir / f"{dataset_name}_detailed.csv"
            scores_df.to_csv(csv_path, index=False)
            print(f"\n📊 Detailed results saved to: {csv_path}")

        # Print summary
        report.print_summary()

        logger.info(f"Evaluation completed. Overall score: {overall_score:.4f}")
        return report


def load_test_cases(dataset_path: Path) -> List[TestCase]:
    """Load test cases from JSON file.

    Expected JSON format:
    [
        {
            "question": "What is...",
            "ground_truth": "The answer is...",
            "expected_sources": ["Note1.md", "Note2.md"],
            "metadata": {"category": "technical"}
        },
        ...
    ]

    Args:
        dataset_path: Path to JSON file containing test cases

    Returns:
        List of TestCase objects
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of test cases")

    test_cases = []
    for item in data:
        test_cases.append(TestCase(
            question=item['question'],
            ground_truth=item['ground_truth'],
            expected_sources=item.get('expected_sources'),
            metadata=item.get('metadata'),
        ))

    logger.info(f"Loaded {len(test_cases)} test cases from {dataset_path}")
    return test_cases


def main():
    """CLI entry point for running RAGAS evaluation."""
    import argparse
    from main import UltraRAG

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on UltraRAG system"
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('tests/evaluation_dataset.json'),
        help='Path to evaluation dataset JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./data/evaluation'),
        help='Output directory for evaluation reports'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Minimum acceptable score threshold (0-1)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save evaluation report to file'
    )

    args = parser.parse_args()

    # Load test cases
    print(f"\n📋 Loading test cases from: {args.dataset}")
    try:
        test_cases = load_test_cases(args.dataset)
        print(f"✅ Loaded {len(test_cases)} test cases")
    except Exception as e:
        print(f"❌ Failed to load test cases: {e}")
        sys.exit(1)

    # Initialize RAG system
    print("\n🚀 Initializing UltraRAG system...")
    try:
        rag_system = UltraRAG()

        # Load existing index
        if not rag_system.load_existing_index():
            print("❌ No existing index found. Please run 'python main.py' to create an index first.")
            sys.exit(1)

        print("✅ RAG system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        sys.exit(1)

    # Initialize evaluator
    print("\n🔬 Initializing RAGAS evaluator...")
    evaluator = RAGEvaluator(
        rag_system=rag_system,
        min_score_threshold=args.threshold,
    )

    # Run evaluation
    try:
        report = evaluator.evaluate(
            test_cases=test_cases,
            dataset_name=args.dataset.stem,
            save_report=not args.no_save,
            output_dir=args.output,
        )

        # Exit with error code if evaluation failed
        if report.overall_score < args.threshold:
            print(f"\n⚠️  Evaluation failed: Overall score {report.overall_score:.4f} below threshold {args.threshold}")
            sys.exit(1)
        else:
            print(f"\n✅ Evaluation passed: Overall score {report.overall_score:.4f} meets threshold {args.threshold}")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        logger.error("Evaluation failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
