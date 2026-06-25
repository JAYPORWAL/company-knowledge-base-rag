import uuid
import datetime
from pathlib import Path
from typing import List, Set, Optional
from loguru import logger
from llama_index.core import Document

from config.settings import Settings
from config.exceptions import IngestionError
from ingestion.document_schema import DocumentMetadata
from ingestion.metadata_registry import MetadataRegistry
from ingestion.parser import DocumentParserRegistry

class DocumentLoader:
    """
    Handles file ingestion, size checking, file extension validation,
    SHA256 duplicate checks, and document parsing using registry readers.
    """
    def __init__(
        self,
        settings: Settings,
        registry: MetadataRegistry,
        parser_registry: Optional[DocumentParserRegistry] = None,
        max_file_size_bytes: int = 15 * 1024 * 1024  # Default 15 MB
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.parser_registry = parser_registry or DocumentParserRegistry()
        self.max_file_size_bytes = max_file_size_bytes
        self.allowed_extensions: Set[str] = {"pdf", "docx", "txt", "pptx", "md"}

    def validate_file(self, file_path: Path) -> None:
        """
        Validates file existence, file extension constraints, and size limits.
        Raises IngestionError if validations fail.
        """
        if not file_path.is_file():
            raise IngestionError(
                message=f"Target path '{file_path}' is not a valid file.",
                details={"file_path": str(file_path)}
            )

        # 1. Extension check
        ext = file_path.suffix.lower().lstrip(".")
        if ext not in self.allowed_extensions:
            raise IngestionError(
                message=f"Unsupported file type: '.{ext}'. Supported formats: {list(self.allowed_extensions)}",
                details={"filename": file_path.name, "file_type": ext}
            )

        # 2. File size validation
        size_bytes = file_path.stat().st_size
        if size_bytes > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            actual_mb = size_bytes / (1024 * 1024)
            raise IngestionError(
                message=f"File exceeds maximum allowed size of {max_mb:.1f} MB (Actual size: {actual_mb:.1f} MB).",
                details={"filename": file_path.name, "size_bytes": size_bytes}
            )

    def load_file(self, file_path: Path) -> List[Document]:
        """
        Loads and parses a document, enforcing constraints and validating schemas.
        Returns a list of LlamaIndex Document instances.
        """
        try:
            logger.info("Executing validation checks for: '{}'", file_path.name)
            
            # Step 1: Validate file constraints
            self.validate_file(file_path)

            # Step 2: Compute SHA256 hash and query metadata registry
            sha256_hash = self.registry.compute_sha256(file_path)
            if self.registry.is_duplicate(sha256_hash):
                logger.info(
                    "Ingestion skipped: duplicate SHA256 checksum detected for '{}'.",
                    file_path.name
                )
                return []

            # Step 3: Fetch the appropriate parser reader
            ext = file_path.suffix.lower().lstrip(".")
            reader = self.parser_registry.get_reader_for_extension(ext)
            if not reader:
                raise IngestionError(
                    message=f"No compatible parser reader was registered for extension: '.{ext}'",
                    details={"filename": file_path.name}
                )

            # Step 4: Populate DocumentMetadata model schema
            doc_id = str(uuid.uuid4())
            file_size = file_path.stat().st_size
            
            metadata = DocumentMetadata(
                document_id=doc_id,
                filename=file_path.name,
                upload_date=datetime.datetime.now(datetime.timezone.utc),
                file_type=ext,
                file_size=file_size,
                sha256_hash=sha256_hash,
                source=str(file_path)
            )

            # Convert metadata payload to dictionary format for LlamaIndex extra_info
            extra_info = metadata.model_dump()
            # Serialize datetime objects into strings for dictionary compatibility
            extra_info["upload_date"] = metadata.upload_date.isoformat()

            # Step 5: Read document content using the mapped reader
            logger.info("Parsing content for '{}' using '.{}' reader...", file_path.name, ext)
            
            try:
                documents = reader.load_data(file_path, extra_info=extra_info)
            except Exception as parse_err:
                logger.error("Parser failed to read content from file '{}': {}", file_path.name, str(parse_err))
                raise IngestionError(
                    message=f"Parser execution error: {str(parse_err)}",
                    details={"filename": file_path.name, "error": str(parse_err)}
                ) from parse_err

            if not documents:
                logger.warning("Parser returned empty contents for file: '{}'", file_path.name)
                return []

            # Step 6: Map metadata schema variables across all LlamaIndex document objects
            for doc in documents:
                doc.metadata.update(extra_info)
                # Remove nested dict to prevent ChromaDB metadata flat schema errors
                if "extra_info" in doc.metadata:
                    del doc.metadata["extra_info"]
                doc.doc_id = f"{doc_id}_{doc.doc_id}"

            logger.info(
                "Ingestion successful for '{}'. Generated {} raw pages/documents.",
                file_path.name,
                len(documents)
            )
            return documents

        except IngestionError as ie:
            # Re-raise already logged ingestion errors
            logger.error("Ingestion failed: {}", ie.message)
            raise
        except Exception as e:
            logger.error("Unhandled error ingesting file '{}': {}", file_path.name, str(e))
            raise IngestionError(
                message=f"Unhandled ingestion error: {str(e)}",
                details={"filename": file_path.name}
            ) from e
