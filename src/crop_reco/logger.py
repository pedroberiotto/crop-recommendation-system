from __future__ import annotations
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from .config import LOG_FILE
_RESERVED_LOG_ATTRS = {'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename', 'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated', 'thread', 'threadName', 'processName', 'process', 'taskName'}

class _JSONFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {'timestamp': datetime.now(timezone.utc).isoformat(), 'level': record.levelname, 'logger': record.name, 'message': record.getMessage()}
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and (not key.startswith('_')):
                payload[key] = value
        if record.exc_info:
            payload['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)

def setup_logging(level: int=logging.INFO) -> None:
    formatter = _JSONFormatter()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if LOG_FILE:
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    for handler in handlers:
        handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = handlers

def _emit(logger_name: str, level: int, message: str, **extra) -> None:
    logger = logging.getLogger(logger_name)
    record = logger.makeRecord(logger_name, level, fn='', lno=0, msg=message, args=(), exc_info=None)
    for key, value in extra.items():
        setattr(record, key, value)
    logger.handle(record)

def log_request(*, endpoint: str, latency_ms: float, batch_size: int, model_artifact: str | None, input_modified: bool) -> None:
    _emit('api.request', logging.INFO, 'predict', endpoint=endpoint, latency_ms=round(latency_ms, 2), batch_size=batch_size, model_artifact=model_artifact, input_modified=input_modified)

def log_event(name: str, **extra) -> None:
    _emit('api.event', logging.INFO, name, event=name, **extra)
