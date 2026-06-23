from abc import ABC, abstractmethod
from typing import List
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.schema import BaseNode

class VectorStoreProvider(ABC):
    """
    Abstract Interface for Vector Database storage providers (Dependency Inversion Principle).
    """
    @abstractmethod
    def get_vector_store(self) -> BasePydanticVectorStore:
        """
        Return the LlamaIndex-compatible vector store instance.
        """
        pass

    @abstractmethod
    def check_connection(self) -> bool:
        """
        Validate database connectivity.
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """
        Delete all chunks and vectors associated with a specific document UUID.
        """
        pass

    @abstractmethod
    def update_document(self, document_id: str, nodes: List[BaseNode]) -> None:
        """
        Replaces/updates chunks and vectors for a specific document UUID.
        """
        pass

    @abstractmethod
    def clear_all(self) -> None:
        """
        Drops all vector entries/collections and resets internal store state.
        """
        pass
