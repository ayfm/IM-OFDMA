import os
from typing import Dict, Optional
import coloredlogs
import logging
import logging.handlers

# Constants
LOG_LEVEL: str = "INFO"
LOG_FORMAT_DEFAULT: str = "[%(asctime)s][%(levelname)-.1s]: %(message)s"
LOG_FORMAT_FILE: str = "[%(asctime)s.%(msecs)03d][%(name)-.15s]: %(message)s"
DATE_FORMAT_DEFAULT: str = "%H:%M:%S"
DATE_FORMAT_FILE: str = "%d%b%y %H:%M:%S"
FILE_MAX_BYTES: int = 100_000_000  # 100 MB
FILE_BACKUP_COUNT: int = 100

# Global variables
loggers: Dict[str, logging.Logger] = {}
handlers: Dict[str, logging.Handler] = {}


def get_logger(path: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger with optional file output.

    Args:
        path (Optional[str]): Path to the log file. If None, only console logging is used.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger("root")

    if "stream" not in handlers:
        _add_stream_handler(logger)

    if path is not None and path not in handlers:
        _add_file_handler(logger, path)

    logger.setLevel(LOG_LEVEL)
    loggers["root"] = logger

    return logger


def _add_stream_handler(logger: logging.Logger) -> None:
    """Add a stream handler to the logger."""
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    formatter = coloredlogs.ColoredFormatter(
        LOG_FORMAT_DEFAULT, datefmt=DATE_FORMAT_DEFAULT
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    handlers["stream"] = stream_handler


def _add_file_handler(logger: logging.Logger, path: str) -> None:
    """Add a file handler to the logger."""
    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=FILE_MAX_BYTES, backupCount=FILE_BACKUP_COUNT
    )
    formatter = logging.Formatter(LOG_FORMAT_FILE, datefmt=DATE_FORMAT_FILE)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # File handler always logs at DEBUG level
    logger.addHandler(file_handler)
    handlers[path] = file_handler
