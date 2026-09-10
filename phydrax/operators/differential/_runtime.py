#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_DERIVATIVE_RUNTIME_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "_DERIVATIVE_RUNTIME_CONTEXT", default=None
)

_DERIVATIVE_EXECUTION_CONTEXT: ContextVar[Mapping[tuple[int, str], str] | None] = (
    ContextVar("_DERIVATIVE_EXECUTION_CONTEXT", default=None)
)


@contextmanager
def derivative_execution_context(
    strategies: Mapping[tuple[int, str], str],
    /,
) -> Iterator[None]:
    """Bind traced derivative strategies while constructing one residual graph."""
    token = _DERIVATIVE_EXECUTION_CONTEXT.set(dict(strategies))
    try:
        yield
    finally:
        _DERIVATIVE_EXECUTION_CONTEXT.reset(token)


def get_derivative_execution_strategy(function: Any, variable: str, /) -> str | None:
    """Return the traced strategy for one source callable and variable."""
    strategies = _DERIVATIVE_EXECUTION_CONTEXT.get()
    if strategies is None:
        return None
    return strategies.get((id(function), str(variable)))


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
