#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jaxtyping import ArrayLike

from ..integration import (
    DensityTarget,
    DiscreteMeasureTarget,
    IntegrationRealization,
    WeightedSampleTarget,
)
from ..transport import (
    AbstractGroundCost,
    unbalanced_problem,
    unbalanced_sinkhorn_divergence,
    UnbalancedSinkhorn,
    UnbalancedSinkhornDivergenceResult,
)


SpatialMeasure = (
    DiscreteMeasureTarget | WeightedSampleTarget | DensityTarget | IntegrationRealization
)


def spatial_unbalanced_sinkhorn_divergence(
    source: SpatialMeasure,
    target: SpatialMeasure,
    /,
    *,
    cost: AbstractGroundCost,
    solver: UnbalancedSinkhorn,
    source_marginal_penalty: ArrayLike,
    target_marginal_penalty: ArrayLike,
    source_encoder: Callable[[Any], Any] | None = None,
    target_encoder: Callable[[Any], Any] | None = None,
) -> UnbalancedSinkhornDivergenceResult:
    """Compare physical spatial or intensity measures without normalizing their mass."""
    if not isinstance(cost, AbstractGroundCost):
        raise TypeError("cost must be an AbstractGroundCost.")
    if not isinstance(solver, UnbalancedSinkhorn):
        raise TypeError("solver must be an UnbalancedSinkhorn solver.")
    problem = unbalanced_problem(
        source,
        target,
        cost=cost,
        source_marginal_penalty=source_marginal_penalty,
        target_marginal_penalty=target_marginal_penalty,
        source_encoder=source_encoder,
        target_encoder=target_encoder,
    )
    return unbalanced_sinkhorn_divergence(problem, solver)


__all__ = ["spatial_unbalanced_sinkhorn_divergence"]
