#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

from jaxtyping import ArrayLike

from phydrax.conditions import ConditionSupport, Residual
from phydrax.domain import DomainFunction

from ..integration import IntegrationSource
from ..metrix import ComplexCoordinateConvention, RiemannianMetric
from ..operators.differential import domain_monge_ampere_residual
from ._residual import ResidualPenalty


def ricci_flat_kahler_term(
    potential_field: str,
    on: ConditionSupport,
    source: IntegrationSource,
    reference_metric: RiemannianMetric,
    convention: ComplexCoordinateConvention,
    target_log_volume: Callable,
    /,
    *,
    normalization: ArrayLike = 0.0,
    state_var: str = "x",
    scale: ArrayLike = 1.0,
    label: str = "ricci-flat-kahler",
) -> ResidualPenalty:
    """Build a globalizable Monge–Ampère residual penalty."""
    if not isinstance(potential_field, str) or not potential_field:
        raise ValueError("potential_field must be a non-empty field name.")
    if not callable(target_log_volume):
        raise TypeError("target_log_volume must be callable.")

    def residual(potential: DomainFunction) -> DomainFunction:
        return domain_monge_ampere_residual(
            potential,
            reference_metric,
            convention,
            target_log_volume,
            normalization=normalization,
            var=state_var,
        )

    condition = Residual(
        potential_field,
        on,
        residual,
        label=label,
    )
    return ResidualPenalty(
        condition,
        source,
        scale=scale,
        label=label,
    )


__all__ = ["ricci_flat_kahler_term"]
