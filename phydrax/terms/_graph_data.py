#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from jaxtyping import ArrayLike

from phydrax.conditions import Observation
from phydrax.domain import DomainComponent, DomainFunction, PointSampling
from phydrax.domain.graph import (
    graph_component_kind,
    GraphComponentKind,
    GraphDatasetDomain,
    GraphTrajectoryDatasetDomain,
)

from ..domain.graph._observation import GraphTarget, GraphTrajectorySignal
from ..integration import mean_over, over, per_step
from ._observation import ObservationPenalty
from ._residual import ResidualPenalty


GraphTargetInterpolation = Literal["nearest", "linear"]


def _component_kind_for_constraint(
    component: DomainComponent,
    graph_label: str,
    /,
) -> GraphComponentKind:
    return graph_component_kind(component.spec.selection_for(graph_label))


def GraphSupervisedTerm(
    field: str,
    component: DomainComponent,
    values: ArrayLike | Sequence[ArrayLike],
    /,
    *,
    sampling: PointSampling,
    weight: DomainFunction | ArrayLike = 1.0,
    reduction: Literal["mean", "integral"] = "mean",
    label: str | None = None,
    data_accuracy_eps: float = 1e-12,
) -> ResidualPenalty:
    """Build a supervised observation penalty for graph-family targets.

    The target data is aligned with the entity kind selected by `component`
    (`Nodes()`, `Edges()`, `Globals()`, or an explicit subset). The returned
    term samples graph cases, evaluates `field`, and penalizes the difference from
    `GraphTarget(...)`.
    """
    if not isinstance(component.domain, GraphDatasetDomain):
        raise TypeError("GraphSupervisedTerm requires a GraphDatasetDomain component.")
    if reduction not in ("mean", "integral"):
        raise ValueError("reduction must be 'mean' or 'integral'.")
    domain = component.domain
    kind = _component_kind_for_constraint(component, domain.label)
    target = GraphTarget(domain, values, component_kind=kind)
    condition = Observation(str(field), component, target, label=label)
    integration_target = mean_over(component) if reduction == "mean" else over(component)
    source = per_step(integration_target, sampling)
    if isinstance(weight, DomainFunction):
        return ObservationPenalty(
            condition,
            source,
            density=weight,
            data_accuracy_eps=data_accuracy_eps,
        )
    return ObservationPenalty(
        condition,
        source,
        scale=weight,
        data_accuracy_eps=data_accuracy_eps,
    )


def GraphTrajectorySupervisedTerm(
    field: str,
    component: DomainComponent,
    values: ArrayLike | Sequence[ArrayLike],
    /,
    *,
    sampling: PointSampling,
    interpolation: GraphTargetInterpolation = "nearest",
    weight: DomainFunction | ArrayLike = 1.0,
    reduction: Literal["mean", "integral"] = "mean",
    label: str | None = None,
    data_accuracy_eps: float = 1e-12,
) -> ResidualPenalty:
    """Build a supervised observation penalty for graph trajectory targets.

    The target data is aligned by graph case, time index, and the entity kind
    selected by `component`. Sampling draws paired graph-time batches and compares
    `field` with `GraphTrajectorySignal(...)`.
    """
    if not isinstance(component.domain, GraphTrajectoryDatasetDomain):
        raise TypeError(
            "GraphTrajectorySupervisedTerm requires a "
            "GraphTrajectoryDatasetDomain component."
        )
    if reduction not in ("mean", "integral"):
        raise ValueError("reduction must be 'mean' or 'integral'.")
    domain = component.domain
    kind = _component_kind_for_constraint(component, domain.graph_label)
    target = GraphTrajectorySignal(
        domain,
        values,
        component_kind=kind,
        interpolation=interpolation,
    )

    condition = Observation(str(field), component, target, label=label)
    integration_target = mean_over(component) if reduction == "mean" else over(component)
    source = per_step(integration_target, sampling)
    if isinstance(weight, DomainFunction):
        return ObservationPenalty(
            condition,
            source,
            density=weight,
            data_accuracy_eps=data_accuracy_eps,
        )
    return ObservationPenalty(
        condition,
        source,
        scale=weight,
        data_accuracy_eps=data_accuracy_eps,
    )


__all__ = [
    "GraphSupervisedTerm",
    "GraphTarget",
    "GraphTargetInterpolation",
    "GraphTrajectorySignal",
    "GraphTrajectorySupervisedTerm",
]
