"""
Structured logging setup, shared across every module. Uses a contextvar
for request_id so every log line emitted during a single API request --
across pipeline.py, sql_executor.py, guardrails.py, etc -- can be
correlated together, without threading a request_id parameter through
every function signature.
"""

import logging
import sys
from contextvars import ContextVar

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)


def get_request_id() -> str:
    return _request_id_var.get()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def setup_logging(level: int = logging.INFO, log_to_console: bool = False, log_file: str = "app.log") -> None:
    """By default, logs go ONLY to app.log, not the terminal -- keeps the
    console clean when running the API server or the eval harness. Pass
    log_to_console=True for interactive debugging sessions where you want
    to see log lines live."""
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | req=%(request_id)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RequestIdFilter())

    handlers = [file_handler]

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(RequestIdFilter())
        handlers.append(console_handler)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = handlers

    # quiet down noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)