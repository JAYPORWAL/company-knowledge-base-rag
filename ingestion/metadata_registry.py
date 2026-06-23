import json
import datetime
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from config.settings import Settings

class MetadataRegistry:
    """
    Manages document metadata tracking, schema enforcement, and duplicate detection.
    Persists data in a processed_documents.json registry.
    """
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry_path = Path(self.settings.DATA_PROCESSED_DIR) / "processed_documents.json"
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Loads metadata database from disk."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            if self.registry_path.exists():
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self._registry = json.load(f)
                logger.debug("Successfully loaded metadata registry from {}", self.registry_path)
            else:
                self._registry = {}
                logger.debug("No existing metadata registry found. Initializing a new one.")
        except Exception as e:
            logger.error("Failed to load metadata registry: {}", str(e))
            self._registry = {}

    def save_registry(self) -> None:
        """Saves current metadata database state to disk."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=4)
            logger.debug("Saved metadata registry updates to {}", self.registry_path)
        except Exception as e:
            logger.error("Failed to persist metadata registry to disk: {}", str(e))

    def compute_sha256(self, file_path: Path) -> str:
        """Calculate the SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error("Failed to calculate SHA256 checksum for {}: {}", file_path, str(e))
            raise

    def is_duplicate(self, sha256_hash: str) -> bool:
        """Checks if a file with the given SHA256 checksum is already registered."""
        exists = sha256_hash in self._registry
        if exists:
            logger.info("Duplicate file detected with hash: {}", sha256_hash)
        return exists

    def register_document(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        sha256_hash: str,
        source: str,
        chunk_count: int
    ) -> Dict[str, Any]:
        """Registers a document using the defined schema, then persists it."""
        record = {
            "document_id": document_id,
            "filename": filename,
            "file_type": file_type,
            "upload_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sha256_hash": sha256_hash,
            "source": source,
            "chunk_count": chunk_count
        }
        self._registry[sha256_hash] = record
        self.save_registry()
        logger.info("Registered new document: {} with {} chunks", filename, chunk_count)
        return record

    def get_document_by_hash(self, sha256_hash: str) -> Optional[Dict[str, Any]]:
        """Fetch document metadata record using SHA256 checksum."""
        return self._registry.get(sha256_hash)

    def get_all_registered_documents(self) -> Dict[str, Dict[str, Any]]:
        """Return the dictionary of all currently registered documents."""
        return self._registry
