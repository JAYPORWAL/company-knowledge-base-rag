import pytest
from pydantic import ValidationError
from config.settings import Settings, get_settings
from config.exceptions import ConfigurationError

def test_settings_initialization_with_valid_key() -> None:
    """Verifies settings load successfully with valid parameters."""
    settings = Settings(
        GEMINI_API_KEY="valid_test_api_key",
        PORT=8000,
        CHUNK_SIZE=256
    )
    assert settings.GEMINI_API_KEY == "valid_test_api_key"
    assert settings.PORT == 8000
    assert settings.CHUNK_SIZE == 256
    assert settings.MODEL_PROVIDER == "gemini"

def test_settings_invalid_api_key_placeholder() -> None:
    """Verifies that the default placeholder API key is rejected."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(GEMINI_API_KEY="your_gemini_api_key_here")
    assert "GEMINI_API_KEY cannot be empty or the default placeholder." in str(excinfo.value)

def test_settings_empty_api_key() -> None:
    """Verifies that an empty API key is rejected."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(GEMINI_API_KEY="   ")
    assert "GEMINI_API_KEY cannot be empty or the default placeholder." in str(excinfo.value)

def test_settings_non_positive_integers() -> None:
    """Verifies that non-positive port or chunk sizes throw validation errors."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(GEMINI_API_KEY="valid_key", PORT=-100)
    assert "Must be a positive integer greater than zero." in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        Settings(GEMINI_API_KEY="valid_key", CHUNK_SIZE=0)
    assert "Must be a positive integer greater than zero." in str(excinfo.value)

def test_get_settings_exception_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that get_settings raises custom ConfigurationError if variables are invalid."""
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    with pytest.raises(ConfigurationError) as excinfo:
        get_settings()
    assert "Application configuration validation failed" in str(excinfo.value)
