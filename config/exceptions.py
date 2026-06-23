from typing import Any, Dict, Optional

class RAGAppError(Exception):
    """Base exception class for the Company Knowledge Base Q&A application."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConfigurationError(RAGAppError):
    """Raised when application configuration or validation fails."""
    pass


class IngestionError(RAGAppError):
    """Raised during document loading, metadata extraction, parsing, or chunking."""
    pass


class EmbeddingError(RAGAppError):
    """Raised when generating embeddings fails."""
    pass


class RetrievalError(RAGAppError):
    """Raised during query retrieval, index querying, or database lookup."""
    pass
