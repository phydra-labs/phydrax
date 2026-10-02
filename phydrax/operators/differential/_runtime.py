#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from ._requests import DerivativeExecutionPlan, DerivativeStep


_DERIVATIVE_RUNTIME_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "_DERIVATIVE_RUNTIME_CONTEXT", default=None
)

_DERIVATIVE_EXECUTION_CONTEXT: ContextVar[
    Mapping[tuple[int, tuple[DerivativeStep, ...]], DerivativeExecutionPlan] | None
] = ContextVar("_DERIVATIVE_EXECUTION_CONTEXT", default=None)


@contextmanager
def derivative_execution_context(
    plans: Mapping[tuple[int, tuple[DerivativeStep, ...]], DerivativeExecutionPlan],
    /,
) -> Iterator[None]:
    """Bind ordered source-path plans while constructing one residual graph."""
    token = _DERIVATIVE_EXECUTION_CONTEXT.set(dict(plans))
    try:
        yield
    finally:
        _DERIVATIVE_EXECUTION_CONTEXT.reset(token)


def get_derivative_execution_plan(
    function: Any, steps: tuple[DerivativeStep, ...], /
) -> DerivativeExecutionPlan | None:
    """Return the execution plan for one source callable and complete path."""
    plans = _DERIVATIVE_EXECUTION_CONTEXT.get()
    if plans is None:
        return None
    return plans.get((id(function), steps))


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
