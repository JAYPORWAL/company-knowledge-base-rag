from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings as LlamaIndexSettings
)
from llama_index.llms.google_genai import GoogleGenAI

from config.settings import Settings
from config.exceptions import IngestionError, RetrievalError
from embeddings.embedding_model import EmbeddingFactory
from vectorstore.factory import VectorStoreFactory
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.metadata_registry import MetadataRegistry

class IndexBuilder:
    """
    Orchestrates RAG indexing pipeline lifecycle.
    Manages incremental indexing, index rebuilding, and vector store bindings.
    """
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        
        # 1. Initialize Metadata Registry database
        self.registry = MetadataRegistry(settings)
        
        # 2. Initialize ingestion loaders and chunkers
        self.loader = DocumentLoader(settings, self.registry)
        self.chunker = DocumentChunker(settings)
        
        # 3. Configure Embeddings Provider
        self.embed_provider = EmbeddingFactory.create_provider(settings)
        self.embed_model = self.embed_provider.get_embedding_model()
        
        # 4. Configure Google Gemini LLM Provider
        self.llm = GoogleGenAI(
            model=settings.LLM_MODEL,
            api_key=settings.GEMINI_API_KEY
        )

        # 5. Bind globally to LlamaIndex settings
        LlamaIndexSettings.embed_model = self.embed_model
        LlamaIndexSettings.llm = self.llm
        
        # 6. Initialize Vector Store Adapter
        self.vector_store_provider = VectorStoreFactory.create_provider(settings)
        self.vector_store = self.vector_store_provider.get_vector_store()
        
    def get_index(self) -> VectorStoreIndex:
        """
        Loads the VectorStoreIndex using the configured database connection.
        """
        try:
            logger.debug("Loading VectorStoreIndex from ChromaDB...")
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=storage_context
            )
            return index
        except Exception as e:
            logger.error("Failed to load vector store index: {}", str(e))
            raise RetrievalError(
                message=f"Failed to load vector store index: {str(e)}"
            ) from e

    def ingest_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Executes ingestion sequence for a file.
        Implements incremental updates:
        - If filename exists with same SHA256: skip ingestion.
        - If filename exists with different SHA256: delete old vectors, load new content.
        - Else: parse and insert new document.
        """
        try:
            logger.info("Scanning for incremental ingestion on: '{}'", file_path.name)
            self.loader.validate_file(file_path)

            # 1. Look for existing document record by filename in registry
            existing_record = None
            for record in self.registry.get_all_registered_documents().values():
                if record["filename"] == file_path.name:
                    existing_record = record
                    break

            new_sha256 = self.registry.compute_sha256(file_path)

            if existing_record:
                if existing_record["sha256_hash"] == new_sha256:
                    logger.info("File '{}' matches registered SHA256. Ingestion skipped (incremental).", file_path.name)
                    return None
                else:
                    logger.info(
                        "Modified content detected for file '{}'. Purging old chunks before indexing...",
                        file_path.name
                    )
                    # Delete old vectors associated with the parent document ID
                    self.vector_store_provider.delete_document(existing_record["document_id"])
                    
                    # Evict old record key from registry dictionary
                    del self.registry._registry[existing_record["sha256_hash"]]
                    self.registry.save_registry()

            # 2. Parse file content
            documents = self.loader.load_file(file_path)
            if not documents:
                logger.info("Ingestion skipped for '{}' (empty content or duplicate).", file_path.name)
                return None

            # 3. Chunk documents
            nodes = self.chunker.chunk_documents(documents)
            if not nodes:
                logger.warning("Ingestion skipped for '{}' (no chunk nodes generated).", file_path.name)
                return None

            # 4. Insert nodes into VectorStore
            logger.info("Writing {} chunk nodes to ChromaDB...", len(nodes))
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            try:
                # Try inserting into existing index
                index = self.get_index()
                index.insert_nodes(nodes)
            except Exception:
                # Initialize new index structure if collection is empty
                logger.info("Initializing new VectorStoreIndex instance.")
                index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_context
                )

            # 5. Save metadata registry record
            sample_doc = documents[0]
            record = self.registry.register_document(
                document_id=sample_doc.metadata["document_id"],
                filename=sample_doc.metadata["filename"],
                file_type=sample_doc.metadata["file_type"],
                sha256_hash=new_sha256,
                source=str(file_path),
                chunk_count=len(nodes)
            )
            
            logger.info("File '{}' indexed successfully.", file_path.name)
            return record

        except Exception as e:
            logger.error("Ingestion failed for '{}': {}", file_path.name, str(e))
            raise IngestionError(
                message=f"Ingestion failed for file: {str(e)}",
                details={"file_name": file_path.name}
            ) from e

    def rebuild_index(self) -> None:
        """
        Performs full index rebuild:
        1. Drops vector store collection.
        2. Clears processed metadata registry.
        3. Re-scans and indexes raw directory files.
        """
        try:
            logger.warning("Initiating full index rebuild. Purging all database records and registries...")
            
            # 1. Clear Vector Store
            self.vector_store_provider.clear_all()
            self.vector_store = self.vector_store_provider.get_vector_store()

            # 2. Reset Metadata Registry
            self.registry._registry = {}
            self.registry.save_registry()
            logger.debug("Cleared metadata registry database.")

            # 3. Scan Raw Directory and Re-ingest
            raw_dir = Path(self.settings.DATA_RAW_DIR)
            if raw_dir.is_dir():
                logger.info("Scanning directory '{}' for files to re-index...", raw_dir)
                for file_path in raw_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower().lstrip(".") in self.loader.allowed_extensions:
                        logger.info("Re-indexing file: '{}'", file_path.name)
                        try:
                            self.ingest_file(file_path)
                        except Exception as ingest_ex:
                            logger.error("Failed to re-index file '{}': {}", file_path.name, str(ingest_ex))
            else:
                logger.debug("Raw data directory '{}' not found. Creating directory.", raw_dir)
                raw_dir.mkdir(parents=True, exist_ok=True)
                
            logger.info("Index rebuild completed successfully.")

        except Exception as e:
            logger.error("Failed to rebuild index: {}", str(e))
            raise IngestionError(
                message=f"Failed to rebuild index: {str(e)}"
            ) from e
