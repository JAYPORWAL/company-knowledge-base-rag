import sys
from pathlib import Path
from loguru import logger
from config.settings import Settings

def configure_logging(settings: Settings) -> None:
    """
    Configure Loguru with dual-targets:
    1. Colorized human-readable stderr logger.
    2. Structured JSON file logger with rotation and compression.
    """
    # Remove loguru's default handler
    logger.remove()

    # Construct and resolve logging directory
    log_file_path = Path(settings.LOG_FILE_PATH)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Console Handler (Text)
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=console_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # 2. File Handler (JSON)
    # Using loguru's serialize=True converts the record to JSON.
    logger.add(
        str(log_file_path),
        level=settings.LOG_LEVEL,
        serialize=True,
        rotation=settings.LOG_ROTATION,
        compression="zip",
        backtrace=True,
        diagnose=True,
        encoding="utf-8"
    )

    logger.info("Logging configured successfully. Level: {}, Destination: {}", settings.LOG_LEVEL, settings.LOG_FILE_PATH)
