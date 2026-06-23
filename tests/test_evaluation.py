from config.settings import Settings
from evaluation.ragas_eval import RagasEvaluator

def test_ragas_evaluator_metrics_stub() -> None:
    """
    Verifies that the RagasEvaluator returns properly formatted mock metrics payloads in Phase 1.
    """
    settings = Settings(GEMINI_API_KEY="test_key_valid_ok")
    evaluator = RagasEvaluator(settings)
    
    result = evaluator.evaluate_response(
        query_str="How works chunking?",
        answer_str="Chunking splits files into nodes.",
        contexts=["Context segment number one."],
        ground_truth="Chunking splits files into nodes."
    )
    
    # Assert return structure
    assert result["evaluation_status"] == "success"
    assert "metrics" in result
    assert result["metrics"]["faithfulness"] == 0.94
    assert result["metrics"]["answer_relevance"] == 0.91
    assert result["metrics"]["context_recall"] == 0.85
    assert result["evaluator_model"] == settings.LLM_MODEL
