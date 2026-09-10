from __future__ import annotations

import logging
import sys

import structlog

_DEFAULT_LEVEL = "INFO"
_configured = False


def setup_logging(level: str = _DEFAULT_LEVEL) -> None:
    global _configured

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if not _configured:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.handlers.clear()
        root.addHandler(handler)

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer(colors=False),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
            cache_logger_on_first_use=True,
        )
        _configured = True
    else:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    if not _configured:
        setup_logging()
    return structlog.get_logger(name)
