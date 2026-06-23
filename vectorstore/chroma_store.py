from typing import Any, Optional, List
import chromadb
from loguru import logger
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import BaseNode

from config.settings import Settings
from config.exceptions import RetrievalError
from vectorstore.base_store import VectorStoreProvider

class ChromaStoreProvider(VectorStoreProvider):
    """
    ChromaDB adapter implementation of VectorStoreProvider.
    Handles lifecycle operations (persistence, duplicate checks, deletion, updates).
    """
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Optional[Any] = None
        self._vector_store: Optional[ChromaVectorStore] = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            logger.info(
                "Initializing ChromaDB connection at path: '{}' with collection: '{}'",
                self.settings.CHROMA_DB_PATH,
                self.settings.CHROMA_COLLECTION_NAME
            )
            # Establish connection using local file persistence
            self._client = chromadb.PersistentClient(path=self.settings.CHROMA_DB_PATH)
            
            # Retrieve or construct target collection
            chroma_collection = self._client.get_or_create_collection(
                name=self.settings.CHROMA_COLLECTION_NAME
            )
            
            # Wrap standard ChromaDB collection inside LlamaIndex's VectorStore
            self._vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            logger.debug("ChromaDB collection wrapper initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize ChromaDB store: {}", str(e))
            raise RetrievalError(
                message=f"Failed to initialize ChromaDB vector store: {str(e)}",
                details={
                    "chroma_db_path": self.settings.CHROMA_DB_PATH,
                    "collection": self.settings.CHROMA_COLLECTION_NAME
                }
            ) from e

    def get_vector_store(self) -> BasePydanticVectorStore:
        if self._vector_store is None:
            self._initialize()
        assert self._vector_store is not None
        return self._vector_store

    def check_connection(self) -> bool:
        try:
            if self._client is None:
                self._initialize()
            assert self._client is not None
            self._client.heartbeat()
            return True
        except Exception as e:
            logger.error("ChromaDB connection check failed: {}", str(e))
            return False

    def delete_document(self, document_id: str) -> None:
        """
        Deletes all vector store nodes associated with a specific document_id.
        """
        try:
            logger.info("Requesting deletion of document ID '{}' from ChromaDB...", document_id)
            if self._client is None:
                self._initialize()
            assert self._client is not None
            
            # Retrieve underlying collection client
            collection = self._client.get_collection(name=self.settings.CHROMA_COLLECTION_NAME)
            
            # Remove all rows where metadata field 'document_id' matches
            collection.delete(where={"document_id": document_id})
            logger.info("Successfully deleted all vectors matching document ID '{}'.", document_id)
        except Exception as e:
            logger.error("ChromaDB delete failed for document ID '{}': {}", document_id, str(e))
            raise RetrievalError(
                message=f"Failed to delete document vectors: {str(e)}",
                details={"document_id": document_id}
            ) from e

    def update_document(self, document_id: str, nodes: List[BaseNode]) -> None:
        """
        Replaces existing nodes for document_id with new nodes.
        Performs atomic deletion followed by batch insert.
        """
        try:
            logger.info("Requesting update of document ID '{}' with {} nodes in ChromaDB...", document_id, len(nodes))
            # 1. Purge existing elements
            self.delete_document(document_id)
            
            # 2. Add the new nodes to the VectorStore (performs batch insertion)
            vector_store = self.get_vector_store()
            vector_store.add(nodes)
            logger.info("Successfully updated document ID '{}' in ChromaDB.", document_id)
        except Exception as e:
            logger.error("ChromaDB update failed for document ID '{}': {}", document_id, str(e))
            raise RetrievalError(
                message=f"Failed to update document vectors: {str(e)}",
                details={"document_id": document_id}
            ) from e

    def clear_all(self) -> None:
        """
        Drops the ChromaDB collection and clears internal cache state.
        """
        try:
            logger.warning("Clearing all data in Chroma collection: {}", self.settings.CHROMA_COLLECTION_NAME)
            if self._client is None:
                self._initialize()
            assert self._client is not None
            
            try:
                self._client.delete_collection(name=self.settings.CHROMA_COLLECTION_NAME)
            except Exception as drop_err:
                logger.warning("Could not drop collection (might not exist yet): {}", str(drop_err))
                
            # Clear internal cache states and re-initialize
            self._vector_store = None
            self._initialize()
            logger.info("ChromaDB storage cleared successfully.")
        except Exception as e:
            logger.error("Failed to clear ChromaDB storage: {}", str(e))
            raise RetrievalError(
                message=f"Failed to clear ChromaDB storage: {str(e)}"
            ) from e
