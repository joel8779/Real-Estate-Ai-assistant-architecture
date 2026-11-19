"""Structured Logging Configuration.

Configures standard logging with structured formatting and secret masking.
"""

import logging
import sys
import json
from datetime import datetime, timezone
from typing import Any, Dict


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
        }

        # Include exception info if present
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        # Include custom extra fields if provided
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_payload.update(record.extra_fields)

        return json.dumps(log_payload, default=str)


def setup_logging(log_level: str = "INFO", json_format: bool = False) -> logging.Logger:
    """Initialize root logger configuration."""
    logger = logging.getLogger("darglobal_assistant")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid duplicate handlers on re-init
    if logger.handlers:
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        standard_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(module)s:%(lineno)d - %(message)s"
        handler.setFormatter(logging.Formatter(standard_format))

    logger.addHandler(handler)
    logger.propagate = False
    return logger


# Default logger instance
logger = setup_logging()
