import json
from pathlib import Path
from loguru import logger
from config.settings import Settings
from config.logging_config import configure_logging

def test_logging_configuration_writes_json_to_file(tmp_path: Path) -> None:
    """
    Verifies that calling configure_logging correctly initializes logs
    and serializes output in JSON structure to the target log file.
    """
    log_file = tmp_path / "app_test.log"
    
    settings = Settings(
        GEMINI_API_KEY="test_api_key_valid",
        LOG_FILE_PATH=str(log_file),
        LOG_LEVEL="DEBUG"
    )
    
    # Configure logging using our logging config module
    configure_logging(settings)
    
    test_message = "Test Log Message for Verification Check"
    logger.debug(test_message)
    
    # Confirm the log file was created
    assert log_file.exists()
    
    # Open and verify the JSON serialization format
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    assert len(lines) >= 2
    
    # Load second line as JSON (first line is log configuration message)
    log_record = json.loads(lines[1])
    
    # Assert JSON structure keys from loguru
    assert "text" in log_record
    assert "record" in log_record
    assert test_message in log_record["record"]["message"]
    assert log_record["record"]["level"]["name"] == "DEBUG"
