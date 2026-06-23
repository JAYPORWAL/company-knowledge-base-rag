from abc import ABC, abstractmethod
import threading
from typing import List, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from config.settings import Settings
from config.exceptions import EmbeddingError

class EmbeddingService(ABC):
    """
    Abstract Interface for Embedding Operations (Dependency Inversion).
    Enables swapping providers (Gemini / OpenAI) without altering downstream RAG clients.
    """
    @abstractmethod
    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a single text block."""
        pass

    @abstractmethod
    def get_text_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of text blocks."""
        pass

    @abstractmethod
    def get_underlying_model(self) -> BaseEmbedding:
        """Retrieve the LlamaIndex-compatible BaseEmbedding instance."""
        pass


class GeminiEmbeddingService(EmbeddingService):
    """
    Google Gemini implementation of EmbeddingService.
    Configured with robust retries, timeout capabilities, batching, and logging.
    """
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            logger.info("Initializing Google Gemini Embeddings (model: {})", self.settings.EMBEDDING_MODEL)
            # Instantiating the official LlamaIndex Gemini Embedding model
            self.model = GoogleGenAIEmbedding(
                model_name=self.settings.EMBEDDING_MODEL,
                api_key=self.settings.GEMINI_API_KEY
            )
        except Exception as e:
            logger.error("Failed to initialize Gemini embedding model: {}", str(e))
            raise EmbeddingError(
                message=f"Gemini embedding model initialization failed: {str(e)}",
                details={"model": self.settings.EMBEDDING_MODEL}
            ) from e

    def get_underlying_model(self) -> BaseEmbedding:
        return self.model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            "Embedding API call failed. Retrying in {}s... (Attempt {})",
            retry_state.upcoming_sleep,
            retry_state.attempt_number
        )
    )
    def get_text_embedding(self, text: str) -> List[float]:
        """Generates embedding for a single string with retry logic."""
        try:
            logger.debug("Generating single text embedding (size: {} chars)...", len(text))
            return self.model.get_text_embedding(text)
        except Exception as e:
            logger.error("Embedding API error: {}", str(e))
            raise EmbeddingError(
                message=f"Embedding API call failed: {str(e)}"
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            "Batch embedding API call failed. Retrying in {}s... (Attempt {})",
            retry_state.upcoming_sleep,
            retry_state.attempt_number
        )
    )
    def get_text_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of strings with retry logic."""
        try:
            logger.info("Generating batch embeddings for {} text segments...", len(texts))
            return self.model.get_text_embedding_batch(texts)
        except Exception as e:
            logger.error("Batch embedding API error: {}", str(e))
            raise EmbeddingError(
                message=f"Batch embedding API call failed: {str(e)}",
                details={"batch_size": len(texts)}
            ) from e


class OpenAIEmbeddingService(EmbeddingService):
    """
    OpenAI-based Embedding Service placeholder.
    Supports future migration without coding changes.
    """
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        logger.warning("OpenAIEmbeddingService initialized as a migration stub.")

    def get_underlying_model(self) -> BaseEmbedding:
        logger.error("Attempted to retrieve OpenAI embedding model stub.")
        raise NotImplementedError("OpenAI embedding model is not yet implemented.")

    def get_text_embedding(self, text: str) -> List[float]:
        logger.error("Attempted to query OpenAI embedding service stub.")
        raise NotImplementedError("OpenAI embedding service is not yet active.")

    def get_text_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        logger.error("Attempted to query OpenAI batch embedding service stub.")
        raise NotImplementedError("OpenAI batch embedding service is not yet active.")


class EmbeddingServiceManager:
    """
    Thread-safe Singleton Manager for the active Embedding Service.
    Maps to the active provider specified in configuration settings.
    """
    _instance: Optional['EmbeddingServiceManager'] = None
    _lock = threading.Lock()

    def __new__(cls, settings: Settings) -> 'EmbeddingServiceManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmbeddingServiceManager, cls).__new__(cls)
                cls._instance._initialize(settings)
            return cls._instance

    def _initialize(self, settings: Settings) -> None:
        self.settings = settings
        self._service = self._create_service()

    def _create_service(self) -> EmbeddingService:
        provider = self.settings.EMBEDDING_PROVIDER.lower().strip()
        if provider == "gemini":
            return GeminiEmbeddingService(self.settings)
        elif provider in ("openai", "azure"):
            return OpenAIEmbeddingService(self.settings)
        else:
            raise EmbeddingError(
                message=f"Unsupported embedding provider configured: '{provider}'"
            )

    def get_service(self) -> EmbeddingService:
        """Returns the active EmbeddingService instance."""
        return self._service


# ==============================================================================
# BACKWARD COMPATIBILITY ADAPTERS FOR PHASE 1 INDEX BUILDER CLIENTS
# ==============================================================================

class EmbeddingProvider(ABC):
    """Abstract interface for backward compatible embedding provider."""
    @abstractmethod
    def get_embedding_model(self) -> BaseEmbedding:
        pass


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedding provider adapter."""
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_embedding_model(self) -> BaseEmbedding:
        manager = EmbeddingServiceManager(self.settings)
        return manager.get_service().get_underlying_model()


class EmbeddingFactory:
    """Factory creating backward compatible embedding providers."""
    @staticmethod
    def create_provider(settings: Settings) -> EmbeddingProvider:
        provider_name = settings.EMBEDDING_PROVIDER.lower().strip()
        if provider_name == "gemini":
            return GeminiEmbeddingProvider(settings)
        else:
            raise EmbeddingError(f"Unsupported embedding provider: '{provider_name}'")
