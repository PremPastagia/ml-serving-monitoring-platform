"""Structured JSON logging.

One JSON object per line, with the request id on every record. Plain-text logs force
whoever is debugging a latency spike to write a regex; JSON lines can be filtered with
`jq` immediately and shipped to a log backend without a parsing rule. The request id
is what joins a slow request in the logs to its row in the prediction store.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

#: Set per request by the middleware, so any log call anywhere in the handler picks up
#: the correct id without it being threaded through every function signature.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, stream=None) -> logging.Logger:
    """Install the JSON formatter on the mlserve logger. Idempotent."""
    logger = logging.getLogger("mlserve")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    # Without this the root logger's default handler prints a second, unstructured copy.
    logger.propagate = False
    return logger


def get_logger(name: str = "mlserve") -> logging.Logger:
    return logging.getLogger(name)
