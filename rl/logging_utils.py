import logging
import os
import sys
from typing import Optional

# Default log destination can be overridden via environment variable.
DEFAULT_LOG_FILE = "escape_game.log"
DEFAULT_LOG_LEVEL = "INFO"


def _resolve_level(level: Optional[int] = None) -> int:
    if isinstance(level, int):
        return level
    try:
        return getattr(logging, str(DEFAULT_LOG_LEVEL).upper())
    except AttributeError:
        return logging.INFO


def _has_file_handler(logger: logging.Logger, path: str) -> bool:
    abs_path = os.path.abspath(path)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(getattr(handler, "baseFilename", "")) == abs_path:
                return True
    return False


def setup_logging(
    log_file: str = "escape_game.log",
    level: Optional[int] = None,
    *,
    include_console: bool = True,
) -> logging.Logger:
    """
    Configure application logging.
    - Always logs to a file (default: escape_game.log or ESCAPE_GAME_LOG_FILE env var).
    - Optionally mirrors logs to stderr so interactive runs still see output.
    Subsequent calls are idempotent for the same file path.
    """
    log_path = os.path.abspath(log_file or DEFAULT_LOG_FILE)
    logger = logging.getLogger()
    logger.setLevel(_resolve_level(level))

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if not _has_file_handler(logger, log_path):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if include_console:
        has_console = any(
            isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
            for h in logger.handlers
        )
        if not has_console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

    return logger


__all__ = ["setup_logging", "DEFAULT_LOG_FILE"]
