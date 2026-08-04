#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

from ...integration import (
    ComponentTarget,
    DensityTarget,
    integrate as execute_integration,
    IntegrationRealization,
    mean_over,
    reduce as reduce_integration,
)


def integral(
    integrand: Any,
    target_or_realization: Any,
    plan: Any = None,
    /,
    **kwargs: Any,
):
    """Return the value of an unnormalized typed integration execution."""
    if isinstance(target_or_realization, IntegrationRealization):
        if plan is not None:
            raise TypeError("A materialized realization does not take another plan.")
        estimate = reduce_integration(integrand, target_or_realization, **kwargs)
    else:
        estimate = execute_integration(integrand, target_or_realization, plan, **kwargs)
    return estimate.value


def mean(
    integrand: Any,
    target_or_realization: Any,
    plan: Any = None,
    /,
    **kwargs: Any,
):
    """Return a normalized component/density integration value."""
    if isinstance(target_or_realization, IntegrationRealization):
        if plan is not None:
            raise TypeError("A materialized realization does not take another plan.")
        target = target_or_realization.target
        if isinstance(target, ComponentTarget):
            normalized_target = mean_over(target.component, axes=target.axes)
        elif isinstance(target, DensityTarget):
            normalized_target = DensityTarget(
                target.base, target.log_density, normalized=True
            )
        else:
            normalized_target = target
        realization = IntegrationRealization(
            normalized_target,
            target_or_realization.plan,
            target_or_realization.batch,
            target_or_realization.key,
        )
        estimate = reduce_integration(integrand, realization, **kwargs)
    else:
        target = target_or_realization
        if isinstance(target, ComponentTarget):
            target = mean_over(target.component, axes=target.axes)
        elif isinstance(target, DensityTarget):
            target = DensityTarget(target.base, target.log_density, normalized=True)
        estimate = execute_integration(integrand, target, plan, **kwargs)
    return estimate.value


def integrate_interior(
    integrand: Any,
    target_or_realization: Any,
    plan: Any = None,
    /,
    **kwargs: Any,
):
    """Semantic alias; interior selection belongs to the supplied target."""
    return integral(integrand, target_or_realization, plan, **kwargs)


def integrate_boundary(
    integrand: Any,
    target_or_realization: Any,
    plan: Any = None,
    /,
    **kwargs: Any,
):
    """Semantic alias; boundary selection belongs to the supplied target."""
    return integral(integrand, target_or_realization, plan, **kwargs)


__all__ = ["integral", "integrate_boundary", "integrate_interior", "mean"]
