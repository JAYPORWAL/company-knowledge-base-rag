import os
from pathlib import Path
from typing import Dict, Any, Tuple
from loguru import logger
import chromadb
from config.settings import Settings

class HealthChecker:
    """
    Utility class to perform system health checks:
    1. Storage paths validation (exist and writable).
    2. ChromaDB connectivity check.
    3. Gemini API connection check.
    """
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check_storage_paths(self) -> Tuple[bool, str]:
        """Validates that all configured storage directories exist and are writable."""
        paths_to_check = {
            "chroma_db_path": Path(self.settings.CHROMA_DB_PATH),
            "data_raw_dir": Path(self.settings.DATA_RAW_DIR),
            "data_processed_dir": Path(self.settings.DATA_PROCESSED_DIR),
            "log_file_path": Path(self.settings.LOG_FILE_PATH).parent
        }
        
        for name, path in paths_to_check.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                # Test write permissions by creating a temporary file
                test_file = path / f".health_check_write_test_{os.getpid()}"
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                logger.error("Storage check failed for {}: {}. Error: {}", name, path, str(e))
                return False, f"Directory '{path}' ({name}) is not writable: {str(e)}"
        
        logger.debug("All storage path health checks passed.")
        return True, "All directories exist and are writable."

    def check_chromadb(self) -> Tuple[bool, str]:
        """Verifies ChromaDB database connection and database heartbeat."""
        try:
            client = chromadb.PersistentClient(path=self.settings.CHROMA_DB_PATH)
            # Try to list collections or invoke heartbeat check
            client.heartbeat()
            logger.debug("ChromaDB heartbeat health check passed.")
            return True, "ChromaDB connection healthy."
        except Exception as e:
            logger.error("ChromaDB connection check failed. Error: {}", str(e))
            return False, f"ChromaDB connection check failed: {str(e)}"

    def check_gemini(self) -> Tuple[bool, str]:
        """Validates connectivity to Gemini API by sending a simple model check request."""
        # If API key is the default mock key, report unhealthy but explain why
        if self.settings.GEMINI_API_KEY == "mock_gemini_api_key_for_testing":
            return False, "Gemini key is set to default testing mock value. Please configure GEMINI_API_KEY in .env."
        
        try:
            # We import here to fail gracefully if llama-index dependencies are missing
            from llama_index.llms.google_genai import GoogleGenAI
            llm = GoogleGenAI(
                model=self.settings.LLM_MODEL,
                api_key=self.settings.GEMINI_API_KEY
            )
            # Run a lightweight completion call
            response = llm.complete("health_check_ping", max_tokens=5)
            if response.text:
                logger.debug("Gemini connection health check passed.")
                return True, "Gemini API connectivity healthy."
            else:
                return False, "Gemini returned empty response text."
        except Exception as e:
            logger.error("Gemini API connection check failed. Error: {}", str(e))
            return False, f"Gemini API check failed: {str(e)}"

    def check_all(self) -> Dict[str, Any]:
        """Aggregates health checks into a structured payload."""
        storage_ok, storage_msg = self.check_storage_paths()
        chromadb_ok, chromadb_msg = self.check_chromadb()
        gemini_ok, gemini_msg = self.check_gemini()
        
        is_healthy = storage_ok and chromadb_ok and gemini_ok
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "details": {
                "storage_paths": {"healthy": storage_ok, "message": storage_msg},
                "chromadb": {"healthy": chromadb_ok, "message": chromadb_msg},
                "gemini_api": {"healthy": gemini_ok, "message": gemini_msg}
            }
        }
