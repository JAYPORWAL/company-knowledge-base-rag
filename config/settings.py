from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from config.exceptions import ConfigurationError

class Settings(BaseSettings):
    # LLM & Embedding Settings
    GEMINI_API_KEY: str = Field(
        default="mock_gemini_api_key_for_testing",
        description="Google API key for Gemini models"
    )
    MODEL_PROVIDER: Literal["gemini", "openai", "azure"] = Field(
        default="gemini",
        description="LLM Provider to use"
    )
    EMBEDDING_PROVIDER: Literal["gemini", "openai", "azure"] = Field(
        default="gemini",
        description="Embedding Provider to use"
    )
    LLM_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="LLM Model version"
    )
    EMBEDDING_MODEL: str = Field(
        default="models/text-embedding-004",
        description="Embedding Model version"
    )

    # Vector Store Settings
    CHROMA_DB_PATH: str = Field(
        default="./data/chromadb",
        description="Path to store ChromaDB databases locally"
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="company_knowledge_base",
        description="ChromaDB collection name"
    )

    # Ingestion Settings
    DATA_RAW_DIR: str = Field(
        default="./data/raw",
        description="Folder path where users place raw documents"
    )
    DATA_PROCESSED_DIR: str = Field(
        default="./data/processed",
        description="Folder path for storing ingestion metadata"
    )
    CHUNK_SIZE: int = Field(
        default=512,
        description="LlamaIndex chunk size in tokens"
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        description="LlamaIndex chunk overlap in tokens"
    )

    # Logging Settings
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level"
    )
    LOG_FILE_PATH: str = Field(
        default="./logs/app.log",
        description="Path to write application log files"
    )
    LOG_ROTATION: str = Field(
        default="10 MB",
        description="Size or duration rotation threshold for log files"
    )

    # Server Settings
    PORT: int = Field(
        default=8501,
        description="Port for running the Streamlit app"
    )
    HOST: str = Field(
        default="0.0.0.0",
        description="Host for running the Streamlit app"
    )

    # Validate settings fields
    @field_validator("GEMINI_API_KEY")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped or v_stripped == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY cannot be empty or the default placeholder.")
        return v_stripped

    @field_validator("CHUNK_SIZE", "CHUNK_OVERLAP", "PORT")
    @classmethod
    def validate_positive_integers(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Must be a positive integer greater than zero.")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


def get_settings() -> Settings:
    """
    Factory function to retrieve Settings.
    Raises ConfigurationError if the validation fails.
    """
    try:
        # Load from environment and .env file
        return Settings()
    except Exception as e:
        raise ConfigurationError(
            message=f"Application configuration validation failed: {str(e)}",
            details={"error_class": e.__class__.__name__}
        ) from e
