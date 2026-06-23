from config.settings import Settings
from vectorstore.base_store import VectorStoreProvider
from vectorstore.chroma_store import ChromaStoreProvider

class VectorStoreFactory:
    """
    Factory to instantiate configured vector databases (Open-Closed Principle).
    """
    @staticmethod
    def create_provider(settings: Settings) -> VectorStoreProvider:
        # For Phase 1 we default to ChromaDB. We can easily route to Pinecone once it is implemented.
        return ChromaStoreProvider(settings)
