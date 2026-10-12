"""
Structured logging module for FlashVSR.
Provides console logging and optional JSONL file output with run_id tracking.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

run_id: ContextVar[str] = ContextVar("run_id", default="-")


class RunIdFilter(logging.Filter):
    """Injects current run_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id") or record.run_id is None:
            record.run_id = run_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """Emits JSONL formatted records: {time, level, logger, message, [exception]}."""

    def format(self, record: logging.LogRecord) -> str:
        created_dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload = {
            "time": created_dt.isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = getattr(record, "run_id", None) or run_id.get()
        if rid:
            payload["run_id"] = rid
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """
    Configures logging for FlashVSR:
    - Clears old handlers marked with `_flashvsr=True`
    - Adds stderr text handler
    - Optionally adds JSONL file handler
    - Attaches RunIdFilter to both handlers
    """
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level

    root_logger = logging.getLogger()
    flash_logger = logging.getLogger("flashvsr")

    # Remove existing _flashvsr handlers from both root and flashvsr logger
    for lgr in (root_logger, flash_logger):
        for handler in list(lgr.handlers):
            if getattr(handler, "_flashvsr", False):
                lgr.removeHandler(handler)

    root_logger.setLevel(numeric_level)
    flash_logger.setLevel(numeric_level)

    # Console (stderr) handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler._flashvsr = True
    stderr_handler.setLevel(numeric_level)
    stderr_handler.addFilter(RunIdFilter())
    stderr_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(run_id)s] %(name)s: %(message)s"
    )
    stderr_handler.setFormatter(stderr_formatter)
    root_logger.addHandler(stderr_handler)

    # Optional JSONL file handler
    if log_file:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler._flashvsr = True
        file_handler.setLevel(numeric_level)
        file_handler.addFilter(RunIdFilter())
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)

    return flash_logger


def get_logger(name: Optional[str] = "flashvsr") -> logging.Logger:
    """Returns a logger instance with the given name (defaults to 'flashvsr')."""
    return logging.getLogger(name or "flashvsr")
