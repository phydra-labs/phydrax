#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_DERIVATIVE_RUNTIME_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "_DERIVATIVE_RUNTIME_CONTEXT", default=None
)


@contextmanager
def derivative_runtime_context() -> Iterator[None]:
    """Create a per-call runtime context for differential-operator memoization."""
    current = _DERIVATIVE_RUNTIME_CONTEXT.get()
    if current is not None:
        yield
        return

    token = _DERIVATIVE_RUNTIME_CONTEXT.set({"partial_eval_cache": {}})
    try:
        yield
    finally:
        _DERIVATIVE_RUNTIME_CONTEXT.reset(token)


def get_partial_eval_cache() -> dict[Any, Any] | None:
    """Return the active partial-derivative evaluation cache, if any."""
    context = _DERIVATIVE_RUNTIME_CONTEXT.get()
    if context is None:
        return None
    cache = context.get("partial_eval_cache")
    if cache is None:
        cache = {}
        context["partial_eval_cache"] = cache
    return cache
