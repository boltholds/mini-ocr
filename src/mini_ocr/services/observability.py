from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from mini_ocr.core.config import settings

_LOGGER_CONFIGURED = False


def _json_default(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def configure_logging() -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    level_name = str(getattr(settings, "agent_log_level", "INFO") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root_logger = logging.getLogger("mini_ocr")
    root_logger.setLevel(level)
    root_logger.propagate = False

    if not root_logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root_logger.addHandler(console)

        log_file = getattr(settings, "agent_log_file", None)
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    _LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"mini_ocr.{name}")


class AgentTimer:
    """Small timing/logging wrapper for workflow nodes and agents.

    Usage:
        with AgentTimer("extractor", document_id=doc.id, page_from=1, page_to=2) as trace:
            ...
            trace.set(result_items=12)

    It logs a start event, an end event with duration_ms, and an error event if
    the wrapped code raises. Logs are intentionally JSON-like to be grep-friendly.
    """

    def __init__(self, stage: str, **context: Any) -> None:
        self.stage = stage
        self.context = {k: v for k, v in context.items() if v is not None}
        self.started_at = 0.0
        self.extra: dict[str, Any] = {}
        self.logger = get_logger("agents")

    def __enter__(self) -> "AgentTimer":
        self.started_at = time.perf_counter()
        self._log("start")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration_ms = (time.perf_counter() - self.started_at) * 1000
        if exc is not None:
            self._log(
                "error",
                duration_ms=round(duration_ms, 2),
                error_type=getattr(exc_type, "__name__", str(exc_type)),
                error=str(exc),
                **self.extra,
            )
            return False
        self._log("end", duration_ms=round(duration_ms, 2), **self.extra)
        return False

    def set(self, **extra: Any) -> None:
        self.extra.update({k: v for k, v in extra.items() if v is not None})

    def _log(self, event: str, **extra: Any) -> None:
        if not getattr(settings, "enable_agent_tracing", True):
            return
        payload = {
            "event": event,
            "stage": self.stage,
            **self.context,
            **extra,
        }
        self.logger.info(json.dumps(payload, ensure_ascii=False, default=_json_default))


@contextmanager
def timed_stage(stage: str, **context: Any) -> Iterator[AgentTimer]:
    with AgentTimer(stage, **context) as timer:
        yield timer
