"""Test script for self-correction functionality."""

import logging
from unittest.mock import Mock, MagicMock
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from self_correction import SelfCorrectingRetriever, RelevanceGrade, SelfRAGValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_self_correcting_retriever():
    """Test SelfCorrectingRetriever with mock components."""
    print("\n=== Testing SelfCorrectingRetriever ===\n")

    # Create mock LLM
    mock_llm = Mock()

    # Create mock base retriever
    mock_base_retriever = Mock()

    # Create test nodes
    test_node1 = TextNode(text="This is about Python programming", metadata={"title": "Python Guide"})
    test_node2 = TextNode(text="This is about machine learning", metadata={"title": "ML Guide"})

    nodes_with_score = [
        NodeWithScore(node=test_node1, score=0.9),
        NodeWithScore(node=test_node2, score=0.8)
    ]

    # Mock retriever to return test nodes
    mock_base_retriever.retrieve = Mock(return_value=nodes_with_score)

    # Mock LLM responses
    # First attempt: AMBIGUOUS (triggers retry)
    # Second attempt: CORRECT (stops)
    mock_llm.complete = MagicMock()
    mock_llm.complete.side_effect = [
        Mock(text="AMBIGUOUS"),  # First grading
        Mock(text="What are the key features of Python?"),  # Query refinement
        Mock(text="CORRECT"),  # Second grading
    ]

    # Create self-correcting retriever
    retriever = SelfCorrectingRetriever(
        base_retriever=mock_base_retriever,
        llm=mock_llm,
        max_retries=2,
        enable_correction=True
    )

    # Test retrieval
    query_bundle = QueryBundle(query_str="Tell me about Python")
    results = retriever.retrieve(query_bundle)

    print(f"Retrieved {len(results)} nodes")
    print(f"Base retriever was called {mock_base_retriever.retrieve.call_count} times")
    print(f"LLM was called {mock_llm.complete.call_count} times")

    # Verify results
    assert len(results) > 0, "Should return results"
    assert mock_base_retriever.retrieve.call_count >= 1, "Base retriever should be called"
    print("\n✅ SelfCorrectingRetriever test passed!\n")


def test_relevance_grading():
    """Test relevance grading logic."""
    print("\n=== Testing Relevance Grading ===\n")

    # Create mock LLM
    mock_llm = Mock()
    mock_base_retriever = Mock()

    retriever = SelfCorrectingRetriever(
        base_retriever=mock_base_retriever,
        llm=mock_llm,
        max_retries=1,
        enable_correction=True
    )

    # Test CORRECT grade
    mock_llm.complete = Mock(return_value=Mock(text="CORRECT"))
    test_node = NodeWithScore(
        node=TextNode(text="Python is a programming language"),
        score=0.9
    )
    grade = retriever.grade_relevance("What is Python?", [test_node])
    assert grade == RelevanceGrade.CORRECT, f"Expected CORRECT, got {grade}"
    print("✅ CORRECT grading works")

    # Test AMBIGUOUS grade
    mock_llm.complete = Mock(return_value=Mock(text="AMBIGUOUS"))
    grade = retriever.grade_relevance("What is Python?", [test_node])
    assert grade == RelevanceGrade.AMBIGUOUS, f"Expected AMBIGUOUS, got {grade}"
    print("✅ AMBIGUOUS grading works")

    # Test INCORRECT grade
    mock_llm.complete = Mock(return_value=Mock(text="INCORRECT"))
    grade = retriever.grade_relevance("What is Python?", [test_node])
    assert grade == RelevanceGrade.INCORRECT, f"Expected INCORRECT, got {grade}"
    print("✅ INCORRECT grading works")

    print("\n✅ All relevance grading tests passed!\n")


def test_query_refinement():
    """Test query refinement logic."""
    print("\n=== Testing Query Refinement ===\n")

    mock_llm = Mock()
    mock_base_retriever = Mock()

    retriever = SelfCorrectingRetriever(
        base_retriever=mock_base_retriever,
        llm=mock_llm,
        max_retries=2,
        enable_correction=True
    )

    # Mock refined query
    refined = "What are the key features and syntax of Python programming language?"
    mock_llm.complete = Mock(return_value=Mock(text=refined))

    result = retriever.refine_query(
        original_query="Tell me about Python",
        failed_context="This is about machine learning",
        attempt=1
    )

    assert result == refined, "Should return refined query"
    print(f"Original: 'Tell me about Python'")
    print(f"Refined:  '{result}'")
    print("\n✅ Query refinement test passed!\n")


def test_self_rag_validator():
    """Test SelfRAGValidator."""
    print("\n=== Testing SelfRAGValidator ===\n")

    mock_llm = Mock()
    validator = SelfRAGValidator(llm=mock_llm)

    # Test valid response
    mock_llm.complete = Mock(return_value=Mock(text="VALID\nExplanation: The response is supported by the context."))
    is_valid, explanation = validator.validate_response(
        query="What is Python?",
        response="Python is a programming language",
        context="Python is a high-level programming language"
    )
    assert is_valid == True, "Should be valid"
    print("✅ Valid response detection works")

    # Test invalid response
    mock_llm.complete = Mock(return_value=Mock(text="INVALID\nExplanation: Response contains hallucinated information."))
    is_valid, explanation = validator.validate_response(
        query="What is Python?",
        response="Python was created in 1492",
        context="Python is a programming language"
    )
    assert is_valid == False, "Should be invalid"
    print("✅ Invalid response detection works")

    print("\n✅ SelfRAGValidator tests passed!\n")


def test_disabled_correction():
    """Test that retrieval works when correction is disabled."""
    print("\n=== Testing Disabled Correction ===\n")

    mock_llm = Mock()
    mock_base_retriever = Mock()

    test_nodes = [
        NodeWithScore(node=TextNode(text="Test content"), score=0.9)
    ]
    mock_base_retriever.retrieve = Mock(return_value=test_nodes)

    # Create retriever with correction disabled
    retriever = SelfCorrectingRetriever(
        base_retriever=mock_base_retriever,
        llm=mock_llm,
        max_retries=2,
        enable_correction=False  # Disabled
    )

    query_bundle = QueryBundle(query_str="Test query")
    results = retriever.retrieve(query_bundle)

    # Should call base retriever exactly once and not call LLM
    assert mock_base_retriever.retrieve.call_count == 1, "Should call base retriever once"
    assert mock_llm.complete.call_count == 0, "Should not call LLM when disabled"
    print("✅ Disabled correction bypasses self-correction logic")

    print("\n✅ Disabled correction test passed!\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Self-Correction Tests")
    print("="*60)

    try:
        test_relevance_grading()
        test_query_refinement()
        test_self_rag_validator()
        test_disabled_correction()
        test_self_correcting_retriever()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
