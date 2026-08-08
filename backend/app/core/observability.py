"""Small timing helper that logs operational metadata without request content or secrets."""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def log_timing(logger: logging.Logger, operation: str, **fields: str | int) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("operation=%s duration_ms=%s %s", operation, duration_ms, fields)
