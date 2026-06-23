from typing import Any
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from rag.retriever import KBRetriever

def test_kb_retriever_vector_only_pruning(mocker: Any) -> None:
    """
    Verifies that KBRetriever runs semantic vector search and filters out scores below threshold.
    """
    # 1. Mock index and query retriever
    mock_index = mocker.MagicMock()
    mock_vector_retriever = mocker.MagicMock()
    
    # Construct mock nodes
    node1 = NodeWithScore(
        node=TextNode(text="policy chunk 1", node_id="node-1", metadata={"filename": "work.txt"}),
        score=0.85
    )
    node2 = NodeWithScore(
        node=TextNode(text="policy chunk 2", node_id="node-2", metadata={"filename": "finance.txt"}),
        score=0.20
    )
    
    mock_vector_retriever.retrieve.return_value = [node1, node2]
    mock_index.as_retriever.return_value = mock_vector_retriever

    # Init retriever (hybrid=False, threshold=0.30)
    retriever = KBRetriever(
        mock_index,
        similarity_top_k=2,
        score_threshold=0.30,
        hybrid=False
    )
    
    results = retriever.retrieve("What is the policy?")
    
    # Assert that only node1 is returned (0.85 >= 0.30) and node2 is filtered
    assert len(results) == 1
    assert results[0].node.get_content() == "policy chunk 1"
    
    # Assert as_retriever was called with DEFAULT query mode
    mock_index.as_retriever.assert_called_with(
        similarity_top_k=2,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
        filters=None
    )


def test_kb_retriever_hybrid_reciprocal_rank_fusion(mocker: Any) -> None:
    """
    Verifies that KBRetriever runs hybrid search (vector + text search) and fuses results via RRF.
    """
    mock_index = mocker.MagicMock()
    mock_vector_retriever = mocker.MagicMock()
    mock_text_retriever = mocker.MagicMock()

    # Construct overlapping nodes
    node_a = NodeWithScore(
        node=TextNode(text="semantic context match", id_="id-a", metadata={"filename": "a.txt"}),
        score=0.90
    )
    node_b = NodeWithScore(
        node=TextNode(text="exact keyword match", id_="id-b", metadata={"filename": "b.txt"}),
        score=0.60
    )

    # Vector search returns both (semantic match rank 1, keyword match rank 2)
    mock_vector_retriever.retrieve.return_value = [node_a, node_b]
    
    # Text keyword search returns only node_b (exact keyword match rank 1)
    mock_text_retriever.retrieve.return_value = [node_b]

    # Map query mode side effects
    def as_retriever_mock_router(**kwargs: Any) -> Any:
        if kwargs.get("vector_store_query_mode") == VectorStoreQueryMode.DEFAULT:
            return mock_vector_retriever
        else:
            return mock_text_retriever

    mock_index.as_retriever.side_effect = as_retriever_mock_router

    # Init retriever
    retriever = KBRetriever(
        mock_index,
        similarity_top_k=2,
        score_threshold=0.10,
        hybrid=True
    )
    
    results = retriever.retrieve("target query terms")
    
    # Verify RRF fusion ensembled both results
    assert len(results) == 2
    
    # Node_b should have a higher RRF rank score since it was returned by both retrievers
    # score_b = (1 / (1 + 60)) + (1 / (0 + 60)) ~ 0.033
    # score_a = (1 / (0 + 60)) ~ 0.016
    assert results[0].node.node_id == "id-b"
    assert results[1].node.node_id == "id-a"
