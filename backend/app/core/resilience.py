"""Bounded retry support for transient external service failures."""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_operation(operation: str, call: Callable[[], T], retryable: tuple[type[Exception], ...], max_retries: int, initial_backoff_seconds: float, logger: logging.Logger, sleep: Callable[[float], None] = time.sleep) -> T:
    """Retry known transient failures a finite number of times with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return call()
        except retryable:
            if attempt == max_retries:
                raise
            delay = initial_backoff_seconds * (2**attempt)
            logger.warning("operation=%s retry=%d delay_seconds=%.2f", operation, attempt + 1, delay)
            sleep(delay)
    raise AssertionError("unreachable")
