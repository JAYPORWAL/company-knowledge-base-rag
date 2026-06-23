from typing import List, Optional, Any
from loguru import logger
from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser

from config.settings import Settings
from config.exceptions import IngestionError
from embeddings.embedding_model import EmbeddingFactory

class DocumentChunker:
    """
    Enterprise-grade Document Chunker.
    Supports:
    1. Sentence-aware splitting using LlamaIndex SentenceSplitter (Token-based).
    2. Semantic chunking using LlamaIndex SemanticSplitterNodeParser (Similarity-based).
    Provides parent metadata preservation and generates traceable, structured chunk IDs.
    """
    def __init__(self, settings: Settings, embed_model: Optional[Any] = None) -> None:
        self.settings = settings
        self.embed_model = embed_model
        
        # Configure LlamaIndex SentenceSplitter
        self._sentence_splitter = SentenceSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )
        self._semantic_splitter: Optional[SemanticSplitterNodeParser] = None
        
        logger.debug(
            "DocumentChunker initialized with default SentenceSplitter: chunk_size={}, chunk_overlap={}",
            self.settings.CHUNK_SIZE,
            self.settings.CHUNK_OVERLAP
        )

    def _get_semantic_splitter(self) -> SemanticSplitterNodeParser:
        """Lazily instantiates the SemanticSplitterNodeParser."""
        if self._semantic_splitter is None:
            if self.embed_model is None:
                logger.info("Resolving embedding model for SemanticSplitter...")
                provider = EmbeddingFactory.create_provider(self.settings)
                self.embed_model = provider.get_embedding_model()
            
            logger.info("Initializing SemanticSplitterNodeParser...")
            self._semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.embed_model
            )
        return self._semantic_splitter

    def chunk_documents(self, documents: List[Document], strategy: str = "sentence") -> List[BaseNode]:
        """
        Splits a list of Documents into text node chunks using the selected strategy.
        Enforces:
        - Sentence aware splitting or Semantic similarity splitting
        - Metadata preservation
        - Unique chunk ID generation
        """
        if not documents:
            logger.debug("No documents passed to chunker.")
            return []

        strategy_clean = strategy.lower().strip()
        logger.info(
            "Splitting {} documents using '{}' strategy...",
            len(documents),
            strategy_clean
        )

        try:
            splitter: Any
            if strategy_clean == "semantic":
                splitter = self._get_semantic_splitter()
            else:
                splitter = self._sentence_splitter

            # 1. Run splitting (LlamaIndex automatically preserves parent metadata inside nodes)
            nodes = splitter.get_nodes_from_documents(documents)
            logger.info("Generated {} chunk nodes.", len(nodes))

            # 2. Enforce Chunk IDs and structured metadata preservation details
            for index, node in enumerate(nodes):
                # Retrieve parent metadata details
                doc_id = node.metadata.get("document_id", "unknown_doc")
                filename = node.metadata.get("filename", "unknown_file")

                # Generate a unique, traceable ID for this chunk
                chunk_id = f"{doc_id}_chunk_{index:05d}"
                node.id_ = chunk_id

                # Save ID and indexing metadata fields
                node.metadata["chunk_id"] = chunk_id
                node.metadata["chunk_index"] = index
                node.metadata["chunk_strategy"] = strategy_clean
                
                logger.debug(
                    "Chunk created: {} (File: '{}', Size: {} chars)",
                    chunk_id,
                    filename,
                    len(node.get_content())
                )

            logger.info("Successfully finished chunking. Generated {} nodes.", len(nodes))
            return nodes

        except Exception as e:
            logger.error("Failed to split documents: {}", str(e))
            raise IngestionError(
                message=f"Document chunking failure: {str(e)}"
            ) from e
