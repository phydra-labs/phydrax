#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any, TypeAlias

import equinox as eqx
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._strict import StrictModule
from ._api import _requires_random_key, IntegrationRealization, materialize


class PerStepIntegration(StrictModule):
    """Materialize one integration realization for each term evaluation."""

    target: Any
    plan: Any

    def __init__(self, target: Any, plan: Any = None, /):
        self.target = target
        self.plan = plan


class FixedIntegration(StrictModule):
    """Reuse one explicitly materialized integration realization."""

    realization: IntegrationRealization

    def __init__(self, realization: IntegrationRealization, /):
        if not isinstance(realization, IntegrationRealization):
            raise TypeError("FixedIntegration requires an IntegrationRealization.")
        self.realization = realization


class CallerIntegration(StrictModule):
    """Require the caller to supply a compatible integration realization."""

    target: Any

    def __init__(self, target: Any, /):
        self.target = target


class AdaptiveIntegration(StrictModule):
    """Solver-managed adaptive realization configuration."""

    target: Any
    initial_plan: Any
    policy: Any

    def __init__(self, target: Any, initial_plan: Any, policy: Any, /):
        self.target = target
        self.initial_plan = initial_plan
        self.policy = policy


IntegrationSource: TypeAlias = (
    PerStepIntegration | FixedIntegration | CallerIntegration | AdaptiveIntegration
)


def per_step(target: Any, plan: Any = None, /) -> PerStepIntegration:
    """Construct a per-evaluation integration source."""
    return PerStepIntegration(target, plan)


def fixed(realization: IntegrationRealization, /) -> FixedIntegration:
    """Construct a fixed integration source from an explicit realization."""
    return FixedIntegration(realization)


def caller(target: Any, /) -> CallerIntegration:
    """Construct a caller-managed integration source."""
    return CallerIntegration(target)


def adaptive(target: Any, initial_plan: Any, policy: Any, /) -> AdaptiveIntegration:
    """Construct a solver-managed adaptive integration source."""
    return AdaptiveIntegration(target, initial_plan, policy)


def _validate_caller_target(
    source_target: Any, realization: IntegrationRealization
) -> None:
    if type(source_target) is not type(realization.target):
        raise TypeError("Caller integration realization has an incompatible target type.")
    if not bool(eqx.tree_equal(source_target, realization.target)):
        raise ValueError("Caller integration realization has an incompatible target.")


def resolve_integration(
    source: IntegrationSource,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    realization: IntegrationRealization | None = None,
) -> IntegrationRealization:
    """Resolve one source without conflating sampling and caller-managed state."""
    if isinstance(source, PerStepIntegration):
        if realization is not None:
            raise ValueError("PerStepIntegration does not accept a caller realization.")
        if _requires_random_key(source.plan):
            return materialize(source.target, source.plan, key=key)
        return materialize(source.target, source.plan)
    if isinstance(source, FixedIntegration):
        if realization is not None:
            raise ValueError("FixedIntegration does not accept a caller realization.")
        return source.realization
    if isinstance(source, CallerIntegration):
        if realization is None:
            raise ValueError("CallerIntegration requires realization=.")
        if not isinstance(realization, IntegrationRealization):
            raise TypeError("realization must be an IntegrationRealization.")
        _validate_caller_target(source.target, realization)
        return realization
    if isinstance(source, AdaptiveIntegration):
        if realization is None:
            raise ValueError("AdaptiveIntegration requires a solver-managed realization.")
        if not isinstance(realization, IntegrationRealization):
            raise TypeError("realization must be an IntegrationRealization.")
        _validate_caller_target(source.target, realization)
        return realization
    raise TypeError(f"Unsupported integration source {type(source).__name__}.")


__all__ = [
    "AdaptiveIntegration",
    "CallerIntegration",
    "FixedIntegration",
    "IntegrationSource",
    "PerStepIntegration",
    "adaptive",
    "caller",
    "fixed",
    "per_step",
    "resolve_integration",
]
