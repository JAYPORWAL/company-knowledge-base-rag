import pytest
from typing import Any
from config.settings import Settings
from embeddings.embedding_model import (
    EmbeddingServiceManager,
    GeminiEmbeddingService,
    OpenAIEmbeddingService,
    EmbeddingFactory
)

def test_embedding_service_manager_is_singleton() -> None:
    """
    Verifies that the EmbeddingServiceManager implements thread-safe Singleton pattern.
    """
    settings = Settings(GEMINI_API_KEY="test_key_valid")
    
    # Reset Singleton class variables to clear previous test caches
    EmbeddingServiceManager._instance = None
    
    manager1 = EmbeddingServiceManager(settings)
    manager2 = EmbeddingServiceManager(settings)
    
    # Assert they point to the identical instance location
    assert manager1 is manager2
    
    service = manager1.get_service()
    assert isinstance(service, GeminiEmbeddingService)


def test_gemini_embedding_service_operations(mocker: Any) -> None:
    """
    Verifies Gemini embedding calculations and error wrappers using mocks.
    """
    # 1. Mock LlamaIndex GoogleGenAIEmbedding class
    mock_class = mocker.patch("embeddings.embedding_model.GoogleGenAIEmbedding")
    mock_instance = mock_class.return_value
    
    # Mock return values for standard methods
    mock_instance.get_text_embedding.return_value = [0.12, 0.34, 0.56]
    mock_instance.get_text_embedding_batch.return_value = [[0.12, 0.34, 0.56]]

    settings = Settings(GEMINI_API_KEY="test_key_valid")
    service = GeminiEmbeddingService(settings)
    
    # Assert model retrieval resolves correctly
    assert service.get_underlying_model() == mock_instance
    
    # Verify single embedding generation calls the mock
    emb = service.get_text_embedding("Corporate policy statement")
    assert emb == [0.12, 0.34, 0.56]
    mock_instance.get_text_embedding.assert_called_once_with("Corporate policy statement")
    
    # Verify batch embedding generation calls the mock
    batch_emb = service.get_text_embeddings_batch(["Corporate policy statement"])
    assert batch_emb == [[0.12, 0.34, 0.56]]
    mock_instance.get_text_embedding_batch.assert_called_once_with(["Corporate policy statement"])


def test_openai_embedding_service_stub_migration() -> None:
    """
    Verifies that the OpenAI stub initializes correctly but throws stubs failures.
    """
    settings = Settings(
        GEMINI_API_KEY="test_key_valid",
        EMBEDDING_PROVIDER="openai"
    )
    # Reset Singleton state
    EmbeddingServiceManager._instance = None
    manager = EmbeddingServiceManager(settings)
    service = manager.get_service()
    
    assert isinstance(service, OpenAIEmbeddingService)
    
    with pytest.raises(NotImplementedError):
        service.get_underlying_model()

    with pytest.raises(NotImplementedError):
        service.get_text_embedding("test content")


def test_embedding_factory_backward_compatibility() -> None:
    """
    Verifies backward compatibility wrappers (EmbeddingFactory) mapping to the Singleton instance.
    """
    settings = Settings(GEMINI_API_KEY="test_key_valid")
    # Reset Singleton
    EmbeddingServiceManager._instance = None
    
    provider = EmbeddingFactory.create_provider(settings)
    model = provider.get_embedding_model()
    
    assert model is not None
