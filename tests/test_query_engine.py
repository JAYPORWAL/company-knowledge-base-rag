from typing import Any
from llama_index.core.base.response.schema import Response, StreamingResponse
from llama_index.core.schema import NodeWithScore, TextNode

from rag.query_engine import RAGQueryEngine, ConversationMemoryPlaceholder

def test_conversation_memory_caching_and_formatting() -> None:
    """
    Verifies ConversationMemoryPlaceholder logs, retrieves, formats, and resets dialogue states.
    """
    memory = ConversationMemoryPlaceholder("session-abc")
    assert memory.get_compiled_history() == "No previous conversation history."

    # Log exchanges
    memory.add_message("user", "Hello assistant")
    memory.add_message("assistant", "Hello! How can I help you today?")
    
    compiled = memory.get_compiled_history()
    assert "Customer: Hello assistant" in compiled
    assert "Assistant: Hello! How can I help you today?" in compiled
    assert len(memory.get_history()) == 2
    
    # Reset history
    memory.clear_history()
    assert len(memory.get_history()) == 0
    assert memory.get_compiled_history() == "No previous conversation history."


def test_query_engine_synchronous_response(mocker: Any) -> None:
    """
    Verifies that the RAGQueryEngine parses answers, formats citations, and tracks token usage.
    """
    mock_index = mocker.MagicMock()
    mock_engine = mocker.MagicMock()
    
    # Mock the LlamaIndex RetrieverQueryEngine initialization
    mocker.patch("rag.query_engine.RetrieverQueryEngine.from_args", return_value=mock_engine)
    
    # Construct mock Response
    node = NodeWithScore(
        node=TextNode(text="referenced section text", metadata={"filename": "book.pdf", "file_type": "pdf"}),
        score=0.90
    )
    mock_response = Response(
        response="Calculated query answer.",
        source_nodes=[node],
        metadata={
            "gemini_usage": {
                "prompt_tokens": 150,
                "completion_tokens": 50,
                "total_tokens": 200
            }
        }
    )
    mock_engine.query.return_value = mock_response

    # Init engine and run query
    engine = RAGQueryEngine(mock_index)
    result = engine.query("What is the handbook policy?")
    
    # Assert return payload structure
    assert result["answer"] == "Calculated query answer."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["filename"] == "book.pdf"
    assert result["token_usage"]["total_tokens"] == 200
    assert result["history_count"] == 2


def test_query_engine_streaming_response(mocker: Any) -> None:
    """
    Verifies that RAGQueryEngine yields streaming wrappers and updates metrics post-stream.
    """
    mock_index = mocker.MagicMock()
    mock_engine = mocker.MagicMock()
    
    mocker.patch("rag.query_engine.RetrieverQueryEngine.from_args", return_value=mock_engine)
    
    # Construct mock nodes
    node = NodeWithScore(
        node=TextNode(text="referenced text", metadata={"filename": "rules.md"}),
        score=0.88
    )
    
    # Stream generator mock
    def mock_response_gen() -> Any:
        yield "Streaming "
        yield "RAG "
        yield "Tokens"
        
    mock_stream = mocker.MagicMock(spec=StreamingResponse)
    mock_stream.response_gen = mock_response_gen()
    mock_stream.source_nodes = [node]
    mock_stream.metadata = {
        "gemini_api_usage": {
            "prompt_tokens": 100,
            "candidates_token_count": 30
        }
    }
    
    mock_engine.query.return_value = mock_stream
    
    # Init and run streaming query
    engine = RAGQueryEngine(mock_index)
    wrapper = engine.query_stream("Stream query")
    
    # Assert citations are available immediately
    assert len(wrapper.citations) == 1
    assert wrapper.citations[0]["filename"] == "rules.md"
    
    # Consume generator stream
    chunks = list(wrapper.response_generator())
    assert chunks == ["Streaming ", "RAG ", "Tokens"]
    
    # Assert conversation memory updated after generator completed consumption
    assert len(engine.memory.get_history()) == 2
    assert engine.memory.get_history()[1]["content"] == "Streaming RAG Tokens"
    
    # Assert token metrics calculated post-stream
    assert wrapper.token_usage["prompt_tokens"] == 100
    assert wrapper.token_usage["completion_tokens"] == 30
    assert wrapper.token_usage["total_tokens"] == 130
